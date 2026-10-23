//! Prefix cache of `agens serve` (1.3.2): automatic reuse of a prompt's processed prefix across requests,
//! with or without a session id.
//!
//! A recurrent (KDA) state cannot be matched block by block like KV rows, so the cache keeps *snapshots*: at every
//! `every`-token boundary (and only there) of a canonical prefill, the session's KDA conv + recurrent state, its GPU
//! window rings (tiered KV) and its drafter context ring are copied into a snapshot buffer, and the KV pages of the
//! prefix are kept by a parked snapshot session that shares them with the source (refcount, as a fork does: no copy).
//! A new request restores the longest cached boundary of its prompt into a fresh session (pages shared again, state
//! copied into its slot) and prefills only the rest.
//!
//! Exactness: a canonical prefill is one whose chunks all end at multiples of `every` (it starts at position 0 or at a
//! restored boundary, and every chunk is `every` tokens but the last). The cold run of a prompt chunks it the same way,
//! so a restored run computes the same chunks on the same state: greedy output with a hit is token-identical to the
//! cold run (and batch-invariant like it). Conversation turns that continue an unaligned held prefix are not canonical
//! and are not snapshotted.
//!
//! Memory: snapshot buffers come from one pool allocated at startup (GPU memory, or pinned host RAM with
//! --prefix-cache-mem host), sized by --prefix-cache-gib per rank; when it is used up the least recently used snapshot
//! goes. KV pages held only by snapshots are freed by the engine's row eviction (serve.rs make_rows), which takes such
//! snapshots before idle conversations (serve.rs KvPolicy).

use anyhow::Result;
use std::collections::HashMap;
use std::ffi::c_void;
use std::time::Instant;

pub struct Snap {
    pub sess: i32,
    pub key: Vec<u64>,
    buf: *mut c_void,
    pub last: Instant,
    pub hits: u64,
    /// prompt-end snapshot (a whole prompt of a request without a session id, any length): the first token's greedy id
    /// and logits of the cold run, so an identical prompt starts decoding at once
    pub end: Option<(i32, std::sync::Arc<Vec<u16>>)>,
}

pub struct PrefixCache {
    pub every: usize,
    pub dev: bool,
    bytes: usize,
    snaps: HashMap<u64, Snap>,
    index: HashMap<(usize, u64), u64>,
    ends: HashMap<(usize, u64), u64>,
    free: Vec<*mut c_void>,
    next: u64,
    pub stats: (u64, u64, u64, u64), // lookups that hit, missed, snapshots taken, evicted
}

fn mix(h: u64, k: u64) -> u64 {
    let x = (h ^ k).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^ (x >> 29) ^ (x >> 47)
}
fn chain(h: u64, keys: &[u64]) -> u64 {
    keys.iter().fold(h, |a, &k| mix(a, k))
}

impl PrefixCache {
    /// `gib` per rank of snapshot buffers (0: no cache), boundaries every `every` tokens. Allocates the pool now (on the
    /// followers too: forwarded).
    pub fn new(gib: f64, every: usize, dev: bool) -> Result<Option<PrefixCache>> {
        let bytes = agens_cuda::m1::snap_bytes();
        let mut n = ((gib * (1u64 << 30) as f64) as usize) / bytes.max(1);
        if dev && n > 0 {
            // GPU pool: what is left after the load (weights, KV pool, W4 prefill cache, drafter) less 8 GiB of headroom
            // for prefill temporaries; the ranks are symmetric, so the leader's size holds for every rank
            agens_cuda::synchronize()?;
            let (free, total) = agens_cuda::mem_info()?;
            let mut fit = free.saturating_sub(8 << 30) / bytes.max(1);
            // (1.3.5) an auto-sized KV pool's --gpu-gib budget (m1.rs auto_pool): the snapshots stay inside it too
            let budget = crate::m1::AUTO_BUDGET.load(std::sync::atomic::Ordering::Relaxed);
            if budget > 0 {
                fit = fit.min(budget.saturating_sub(total - free + (8 << 30)) / bytes.max(1));
            }
            if fit < n {
                eprintln!("serve: prefix cache: {n} snapshots asked, {fit} fit in the {:.1} GB free{} (8 GiB headroom)",
                          free as f64 / 1e9, if budget > 0 { " / the --gpu-gib budget" } else { "" });
                n = fit;
            }
        }
        if gib <= 0.0 || n == 0 {
            return Ok(None);
        }
        agens_cuda::m1::snap_pool(n, dev);
        let mut free = Vec::with_capacity(n);
        for _ in 0..n {
            free.push(if dev {
                agens_cuda::leak(agens_cuda::device_alloc(bytes)?)
            } else {
                agens_cuda::host_alloc(bytes)?
            });
        }
        eprintln!("serve: prefix cache: {n} snapshots of {:.1} MB ({}), boundaries every {every} tokens",
                  bytes as f64 / 1e6, if dev { "GPU memory" } else { "pinned host RAM" });
        Ok(Some(PrefixCache { every, dev, bytes, snaps: HashMap::new(), index: HashMap::new(), ends: HashMap::new(), free, next: 1, stats: (0, 0, 0, 0) }))
    }

    /// The longest cached boundary b <= max_len of `key` -> (tag, b).
    pub fn lookup(&mut self, key: &[u64], max_len: usize) -> Option<(u64, usize)> {
        let (mut h, mut best) = (0u64, None);
        let mut b = self.every;
        while b <= max_len.min(key.len()) {
            h = chain(h, &key[b - self.every..b]);
            if let Some(&t) = self.index.get(&(b, h)) {
                if let Some(s) = self.snaps.get_mut(&t).filter(|s| s.key[..] == key[..b]) {
                    s.last = Instant::now(); // every matched boundary stays warm (a popular prompt's shorter prefixes too)
                    best = Some((t, b));
                }
            }
            b += self.every;
        }
        if best.is_some() { self.stats.0 += 1 } else { self.stats.1 += 1 }
        best
    }

    /// A prompt-end snapshot of exactly `key` -> (tag, first greedy token, its logits).
    pub fn lookup_end(&mut self, key: &[u64]) -> Option<(u64, i32, std::sync::Arc<Vec<u16>>)> {
        let t = *self.ends.get(&(key.len(), chain(0, key)))?;
        let s = self.snaps.get(&t).filter(|s| s.key[..] == *key)?;
        let (g, l) = s.end.clone()?;
        self.stats.0 += 1;
        Some((t, g, l))
    }
    /// Prompt-end snapshot of session `src`, which holds exactly `key` (a whole prompt), with its first token.
    pub fn insert_end(&mut self, src: i32, key: &[u64], greedy: i32, logits: Vec<u16>) -> Result<bool> {
        if self.ends.contains_key(&(key.len(), chain(0, key))) {
            return Ok(true);
        }
        let Some(tag) = self.take(src, key)? else { return Ok(false) };
        self.snaps.get_mut(&tag).unwrap().end = Some((greedy, std::sync::Arc::new(logits)));
        self.ends.insert((key.len(), chain(0, key)), tag);
        Ok(true)
    }
    /// A new session (a free KDA slot) restored from snapshot `tag`.
    pub fn restore(&mut self, tag: u64) -> Result<i32> {
        let s = self.snaps.get_mut(&tag).expect("snapshot");
        s.last = Instant::now();
        s.hits += 1;
        agens_cuda::m1::session_from_snap(s.sess, s.buf, tag)
    }

    /// Whether `key` (a boundary prefix) is cached; refreshes its LRU time when it is.
    pub fn has(&mut self, key: &[u64]) -> bool {
        let h = key.chunks(self.every).fold(0u64, |a, c| chain(a, c));
        match self.index.get(&(key.len(), h)).copied() {
            Some(t) if self.snaps.get(&t).map_or(false, |s| s.key[..] == *key) => {
                self.snaps.get_mut(&t).unwrap().last = Instant::now();
                true
            }
            _ => false,
        }
    }

    /// Snapshot session `src`, which holds exactly the prefix `key` (a boundary). The least recently used snapshot
    /// makes room when the pool is used up. false: the engine refused (e.g. INT8 dense KV).
    pub fn insert(&mut self, src: i32, key: &[u64]) -> Result<bool> {
        if key.len() % self.every != 0 || self.has(key) {
            return Ok(true);
        }
        let Some(tag) = self.take(src, key)? else { return Ok(false) };
        let h = key.chunks(self.every).fold(0u64, |a, c| chain(a, c));
        self.index.insert((key.len(), h), tag);
        Ok(true)
    }
    /// A snapshot of `src` (holding `key`) into a pool buffer, the LRU snapshot evicted when the pool is used up.
    fn take(&mut self, src: i32, key: &[u64]) -> Result<Option<u64>> {
        if self.free.is_empty() {
            match self.evict_lru()? {
                Some((n, idle)) => eprintln!("serve: prefix cache: evicted the {n}-token snapshot (idle {idle:.0} s) for a new one"),
                None => return Ok(None),
            }
        }
        let buf = self.free.pop().unwrap();
        let tag = self.next;
        self.next += 1;
        match agens_cuda::m1::session_snap(src, buf, tag) {
            Ok(sess) => {
                self.snaps.insert(tag, Snap { sess, key: key.to_vec(), buf, last: Instant::now(), hits: 0, end: None });
                self.stats.2 += 1;
                Ok(Some(tag))
            }
            Err(e) => {
                self.free.push(buf);
                Err(e)
            }
        }
    }

    /// Drop the least recently used snapshot -> (its length, idle seconds); its KV pages go back to the pool unless a
    /// session still shares them.
    pub fn evict_lru(&mut self) -> Result<Option<(usize, f64)>> {
        let Some((&tag, _)) = self.snaps.iter().min_by_key(|(_, s)| s.last) else { return Ok(None) };
        self.evict(tag)
    }
    /// The snapshots as (last use, tag, session, length), least recently used first (serve.rs row eviction).
    pub fn entries(&self) -> Vec<(Instant, u64, i32, usize)> {
        let mut v: Vec<_> = self.snaps.iter().map(|(&t, s)| (s.last, t, s.sess, s.key.len())).collect();
        v.sort_unstable();
        v
    }
    /// The snapshots' sessions (they hold KV pages).
    pub fn sessions(&self) -> impl Iterator<Item = i32> + '_ {
        self.snaps.values().map(|s| s.sess)
    }
    /// Drop snapshot `tag` -> (its length, idle seconds).
    pub fn evict(&mut self, tag: u64) -> Result<Option<(usize, f64)>> {
        let Some(s) = self.snaps.remove(&tag) else { return Ok(None) };
        if s.end.is_some() {
            self.ends.remove(&(s.key.len(), chain(0, &s.key)));
        } else {
            let h = s.key.chunks(self.every).fold(0u64, |a, c| chain(a, c));
            self.index.remove(&(s.key.len(), h));
        }
        agens_cuda::m1::session_free(s.sess)?;
        agens_cuda::m1::snap_drop(tag);
        self.free.push(s.buf);
        self.stats.3 += 1;
        Ok(Some((s.key.len(), s.last.elapsed().as_secs_f64())))
    }
    pub fn len(&self) -> usize {
        self.snaps.len()
    }
    pub fn cap(&self) -> usize {
        self.snaps.len() + self.free.len()
    }
    #[allow(dead_code)]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}
