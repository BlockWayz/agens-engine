//! `agens serve`: the OpenAI-compatible server (crates/agens-server) on the real engine, one GPU.
//!   personal mode: one request decodes at a time (bs = 1, the parity numerics of `agens m1`).
//!   server mode:   continuous batching of up to --max-batch rows (AGENS_M1_BATCH_INVARIANT=1: every row is
//!                  bit-identical to its bs = 1 run; `agens m1-batch` is the gate).
//!
//! One engine thread owns the GPU. Each step it takes new requests, admits what fits, runs one prefill chunk (<= 4096
//! tokens) of the oldest request still prefilling, then one decode step for every decoding request together, and
//! streams each row's token to its request.
//!
//! Sessions: a request with a session hint (`session_id` / `X-Session-Id`) is a conversation. Its main engine session
//! holds the prompt minus its last token; generation runs on a fork (KV rows shared, KDA state copied). The next turn
//! reuses the main session when its tokens are a prefix of the new prompt and prefills only the new part. Idle
//! conversations are parked (KDA state + GPU window rings to pinned host RAM) when slots run short, and freed (LRU)
//! when KV pool rows run short or past --max-convs.
//!
//! Speculative decoding (personal mode, --spec-drafter DIR): a DFlash2 drafter proposes 7 tokens per step and the
//! target verifies the 8-token block (`agens m1-spec`); the engine then runs batch-invariant, so greedy output is
//! token-identical with and without the drafter. Requests whose sampling is greedy or plain temperature use it (the
//! accept rule is sglang's target-only rejection sampling); top-k/top-p/min-p, penalties and biases decode normally.
//! The drafter's context (KV ring of the last 4096 positions) is engine-wide: interleaved conversations cost
//! acceptance, never exactness.
//!
//! --no-fork (personal mode): a conversation generates in its main session instead of a fork, so it holds one KDA
//! state slot (and window ring) instead of two (0.3 GB on a 16 GB card: ~25K tokens of tiered KV). The next turn
//! reuses the session when the new prompt starts with everything the session has consumed (prompt + generated
//! tokens); otherwise it starts over (a recurrent state cannot be rolled back).
//!
//! Memory: --budget-gib plans the KV pool from what is left after weights + scratch (plan.rs) and prints the plan.
//!
//! Images: a request with images carries their decoded pixels; at admission the images whose spans
//! fall in the part to prefill are encoded (vision tower, one at a time) into pinned host memory, and every prefill
//! chunk of the request passes its rope positions and image-embedding rows. Conversation prefixes are compared on
//! keys that include each image's hash.

use crate::sampler::{is_greedy, needs_logits, Budget, BudgetAct, Sampler};
use agens_server::backend::{Backend, BackendError, SessionHint, TokenStream};
use agens_server::sampling::SamplingParams;
use agens_server::vision::MmInput;
use anyhow::{bail, Context, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc as smpsc, Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;

const CHUNK: usize = 4096;
const SPEC_T: usize = 8; // verify block (bonus + 7 drafts)
const VOCAB_VALID: usize = 248077;

/// The sampling a speculative verify reproduces on the device (kernels/m1 samp_pos): temperature (0 = greedy), top-k
/// (0 = whole vocabulary), top-p, min-p -- the host sampler's filters, in its order.
#[derive(Clone, Copy, PartialEq, Debug)]
struct SpecSampling {
    t: f32,
    k: i32,
    p: f32,
    min_p: f32,
}

/// Speculative decoding for this request: Some(sampling) when it is greedy, or temperature sampling with any top-k /
/// top-p / min-p (1.0.1: the on-device accept samples the same filtered distribution as the host sampler; 1.0.0 took
/// plain temperature only, so the default generation_config -- top_k 20, top_p 0.95 -- never used the drafter).
/// Penalties, logit bias, suppressed ids and the reasoning budget need the host sampler: no speculation.
fn spec_sampling(p: &SamplingParams) -> Option<SpecSampling> {
    if p.reasoning_budget.is_some() {
        return None;
    }
    if !needs_logits(p, VOCAB_VALID, 0) {
        return Some(SpecSampling { t: 0.0, k: 0, p: 1.0, min_p: 0.0 });
    }
    let ok = !is_greedy(p)
        && p.logit_bias.is_empty()
        && p.repetition_penalty == 1.0
        && p.frequency_penalty == 0.0
        && p.presence_penalty == 0.0
        && (p.min_new_tokens == 0 || p.stop_token_ids.is_empty())
        && !p.suppress_tokens.iter().any(|&t| (t as usize) < VOCAB_VALID);
    ok.then_some(SpecSampling {
        t: p.temperature as f32,
        k: p.top_k.map(|k| k.min(i32::MAX as u32) as i32).unwrap_or(0),
        p: p.top_p as f32,
        min_p: p.min_p as f32,
    })
}

struct Job {
    id: u64,
    prompt: Vec<u32>,
    /// prefix-comparison keys of the prompt (token ids; image tokens carry their image's hash)
    key: Vec<u64>,
    mm: Option<Arc<MmInput>>,
    sp: SamplingParams,
    conv: Option<String>,
    tx: mpsc::Sender<Result<Vec<u32>, BackendError>>,
}

pub struct EngineBackend {
    tx: Mutex<smpsc::Sender<Job>>,
    ctx: usize,
    next_id: AtomicU64,
}

impl EngineBackend {
    fn submit(&self, prompt: Vec<u32>, mm: Option<Arc<MmInput>>, sampling: SamplingParams, session: Option<SessionHint>)
              -> TokenStream {
        let (tx, rx) = mpsc::channel(256);
        let key = match &mm {
            Some(m) => agens_server::vision::prefix_keys(&prompt, &m.spans, &m.images),
            None => prompt.iter().map(|&t| t as u64).collect(),
        };
        let job = Job {
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            prompt,
            key,
            mm,
            sp: sampling,
            conv: session.map(|s| s.session_id),
            tx: tx.clone(),
        };
        if self.tx.lock().unwrap().send(job).is_err() {
            let _ = tx.try_send(Err(BackendError("engine thread is gone".into())));
        }
        rx
    }
}

impl Backend for EngineBackend {
    fn generate(&self, prompt_ids: Vec<u32>, sampling: SamplingParams, session: Option<SessionHint>) -> TokenStream {
        self.submit(prompt_ids, None, sampling, session)
    }
    fn supports_images(&self) -> bool {
        crate::m1::VISION.load(Ordering::Relaxed)
    }
    fn generate_mm(&self, prompt_ids: Vec<u32>, mm: Arc<MmInput>, sampling: SamplingParams,
                   session: Option<SessionHint>) -> TokenStream {
        self.submit(prompt_ids, Some(mm), sampling, session)
    }
    fn context_length(&self) -> usize {
        self.ctx
    }
}

struct HostBuf(*mut std::ffi::c_void);
unsafe impl Send for HostBuf {}
impl Drop for HostBuf {
    fn drop(&mut self) {
        let _ = agens_cuda::host_free(self.0);
    }
}

struct Conv {
    main: i32,
    held: Vec<u64>, // prefix keys of what the main session holds
    parked: Option<HostBuf>,
    busy: bool,
    last: Instant,
}

/// Image inputs of a prefill feed: rope positions [3][feed len], embedding rows per feed token (-1 = text).
struct MmFeed {
    rpos: Vec<i64>,
    rows: Vec<i32>,
    emb: crate::mm::HostEmb,
}

enum Phase {
    /// prefilling `feed` into `sess`; then fork (conversation) or decode
    Prefill { sess: i32, feed: Vec<i32>, done: usize, mm: Option<MmFeed> },
    /// prefill the prompt's last token on the conversation's fork
    Last { fork: i32 },
    Decode { sess: i32, last: i32 },
}

struct Act {
    job: Job,
    phase: Phase,
    produced: usize,
    sampler: Sampler,
    spec: Option<SpecSampling>, // speculative decoding with this sampling
    fed: Vec<u32>,     // tokens fed to the session after the prompt (--no-fork: what the session holds)
    t0: Instant,
    lk: Option<crate::lookup::Lookup>, // copy-from-context proposals (hybrid drafting), built at the first verify
    bset: bool,                         // its session's batched-verify sampling is set (server mode)
    vsteps: u64,                        // verify steps of this request (bs = 1 or batched) (AGENS_SERVE_REQ_STATS=1 log)
}

/// The request's lookup index (prompt + the bonus token), built on first use.
fn lookup_of<'a>(lk: &'a mut Option<crate::lookup::Lookup>, prompt: &[u32], last: i32, copy: bool)
                 -> &'a mut crate::lookup::Lookup {
    lk.get_or_insert_with(|| {
        let mut l = crate::lookup::Lookup::new();
        if copy {
            l = l.copy_only();
        }
        let ids: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        l.reset(&ids);
        l.extend(&[last]);
        l
    })
}

struct Engine {
    think: (u32, u32), // <|think|>, <|/think|> ids (reasoning budget)
    max_batch: usize,
    max_convs: usize,
    wctx: usize,
    anchor: i32, // a parked session that is always "current" when sessions are freed
    convs: HashMap<String, Conv>,
    stats: (u64, u64, f64), // decode steps, row-steps, decode seconds
    bstats: (u64, u64, f64, u64), // batched speculation since the last report: steps, tokens, seconds, sequences
    pstats: (u64, f64, Option<Instant>), // prefill chunks + seconds since the last report, and when it began
    spec: bool,             // a drafter is loaded (personal mode), or copy-only drafting
    copy: bool,             // copy-only drafting (--spec-copy 1 without a drafter); sampled requests decode on the host without a proposal
    fork: bool,             // conversations generate on a fork of their main session (false: --no-fork)
}

enum Admit {
    Ok(Act),
    Wait,
    Fail(String),
}

impl Engine {
    fn free_session(&self, id: i32) -> Result<()> {
        agens_cuda::m1::session_use(self.anchor)?;
        agens_cuda::m1::session_free(id)
    }

    /// Park idle conversations (LRU) until `need` KDA slots are free. false if impossible.
    fn make_slots(&mut self, need: usize, keep: Option<&str>) -> Result<bool> {
        use agens_cuda::m1;
        while m1::free_slots() < need {
            let victim = self
                .convs
                .iter()
                .filter(|(k, c)| !c.busy && c.parked.is_none() && Some(k.as_str()) != keep)
                .min_by_key(|(_, c)| c.last)
                .map(|(k, _)| k.clone());
            let Some(k) = victim else { return Ok(false) };
            let c = self.convs.get_mut(&k).unwrap();
            let buf = HostBuf(agens_cuda::host_alloc(m1::park_bytes())?);
            m1::session_park(c.main, buf.0)?;
            c.parked = Some(buf);
        }
        Ok(true)
    }

    /// Free idle conversations (LRU) until `rows` KV pool rows are free. false if impossible.
    fn make_rows(&mut self, rows: usize, keep: Option<&str>) -> Result<bool> {
        while agens_cuda::m1::pool_free_rows() < rows {
            let victim = self
                .convs
                .iter()
                .filter(|(k, c)| !c.busy && Some(k.as_str()) != keep)
                .min_by_key(|(_, c)| c.last)
                .map(|(k, _)| k.clone());
            let Some(k) = victim else { return Ok(false) };
            let c = self.convs.remove(&k).unwrap();
            self.free_session(c.main)?;
        }
        Ok(true)
    }

    /// `forks_due`: admitted conversations still prefilling, each of which takes one more slot (its fork) later
    fn admit(&mut self, job: &Job, others_active: bool, forks_due: usize) -> Result<Admit> {
        use agens_cuda::m1;
        let n = job.prompt.len();
        if n == 0 {
            return Ok(Admit::Fail("empty prompt".into()));
        }
        if n + job.sp.max_new_tokens > self.wctx {
            return Ok(Admit::Fail(format!("prompt {n} + max_new_tokens {} exceeds the context {}", job.sp.max_new_tokens, self.wctx)));
        }
        let ids: Vec<i32> = job.prompt.iter().map(|&t| t as i32).collect();
        // a conversation that is not busy: reuse its main session if its tokens prefix the prompt
        let conv = job.conv.clone().filter(|k| self.convs.get(k).map(|c| !c.busy).unwrap_or(true));
        let reuse = conv.as_ref().and_then(|k| self.convs.get(k)).map(|c| c.held.len() < n && job.key.starts_with(&c.held));
        let held = match (&conv, reuse) {
            (Some(k), Some(true)) => self.convs[k].held.len(),
            _ => 0,
        };
        let need_rows = n - held + job.sp.max_new_tokens + if self.spec { SPEC_T } else { 0 };
        let need_slots = match (&conv, reuse) {
            (None, _) => 1,
            (Some(k), Some(true)) => self.fork as usize + self.convs[k].parked.is_some() as usize,
            // a conversation starting over frees its old main session (holding a slot unless parked) before taking one
            (Some(k), _) => {
                let frees = self.convs.get(k).map(|c| c.parked.is_none()).unwrap_or(false) as usize;
                1 + self.fork as usize - frees
            }
        };
        let keep = conv.as_deref();
        if !self.make_rows(need_rows, keep)? || !self.make_slots(need_slots + forks_due, keep)? {
            return Ok(if others_active { Admit::Wait } else { Admit::Fail("not enough KV pool rows / state slots".into()) });
        }
        let mut sampler = Sampler::new(&job.sp, job.id);
        if let Some(n) = job.sp.reasoning_budget {
            sampler.budget = Some(Budget::new(n, job.sp.reasoning_budget_soft_bias, &job.prompt, self.think.0, self.think.1));
        }
        let spec = if self.spec { spec_sampling(&job.sp) } else { None };
        let t0 = Instant::now();
        let mk = |phase| Act {
            job: Job { id: job.id, prompt: job.prompt.clone(), key: job.key.clone(), mm: job.mm.clone(), sp: job.sp.clone(),
                       conv: conv.clone(), tx: job.tx.clone() },
            phase,
            produced: 0,
            sampler,
            spec,
            fed: Vec::new(),
            t0,
            lk: None,
            bset: false,
            vsteps: 0,
        };
        let Some(k) = conv.clone() else {
            let mm = match Self::mm_feed(job, 0, n) {
                Ok(m) => m,
                Err(e) => return Ok(Admit::Fail(format!("{e:#}"))),
            };
            let s = m1::session_new()?;
            return Ok(Admit::Ok(mk(Phase::Prefill { sess: s, feed: ids, done: 0, mm })));
        };
        if reuse != Some(true) {
            if let Some(old) = self.convs.remove(&k) {
                self.free_session(old.main)?; // prefix mismatch: start the conversation over
            }
            let main = m1::session_new()?;
            self.convs.insert(k.clone(), Conv { main, held: Vec::new(), parked: None, busy: false, last: Instant::now() });
        }
        // fork: the main session takes the prompt but its last token (that one goes to the fork); no fork: all of it
        let end = if self.fork { n - 1 } else { n };
        let from = self.convs[&k].held.len();
        let mm = match Self::mm_feed(job, from, end) {
            Ok(m) => m,
            Err(e) => {
                if let Some(old) = self.convs.remove(&k) {
                    self.free_session(old.main)?;
                }
                return Ok(Admit::Fail(format!("{e:#}")));
            }
        };
        let c = self.convs.get_mut(&k).unwrap();
        if let Some(buf) = c.parked.take() {
            m1::session_unpark(c.main, buf.0)?;
        }
        c.busy = true;
        let feed = ids[from..end].to_vec();
        c.held = job.key[..end].to_vec();
        Ok(Admit::Ok(mk(Phase::Prefill { sess: c.main, feed, done: 0, mm })))
    }

    /// The image inputs of prompt[from..to): the images whose spans overlap it are encoded now.
    fn mm_feed(job: &Job, from: usize, to: usize) -> Result<Option<MmFeed>> {
        let Some(mm) = &job.mm else { return Ok(None) };
        let which: Vec<bool> = (0..mm.images.len())
            .map(|i| mm.spans.iter().any(|sp| sp.image == i && sp.start < to && sp.start + sp.len > from))
            .collect();
        let t = Instant::now();
        let emb = crate::mm::encode(&mm.images, &which)?;
        if which.iter().any(|&w| w) {
            eprintln!("serve: request {} encoded {} image(s), {} image tokens, in {:.2} s", job.id,
                      which.iter().filter(|&&w| w).count(), emb.rows, t.elapsed().as_secs_f64());
        }
        let n = job.prompt.len();
        let rp = agens_server::vision::rope_positions(n, &mm.spans, &mm.images);
        Ok(Some(MmFeed { rpos: crate::mm::rope_slice(&rp, n, from, to), rows: crate::mm::emb_rows(from, to, &mm.spans, &emb), emb }))
    }

    /// The token of a row: the reasoning budget may force one; otherwise the GPU argmax for plain greedy rows, else the
    /// CPU sampler on the row's logits (row `row` of the last step).
    fn choose(act: &mut Act, greedy: i32, row: usize) -> Result<i32> {
        let extra = match act.sampler.budget.as_mut().map(|b| b.act(act.produced)) {
            Some(BudgetAct::Force(t)) => return Ok(t as i32),
            Some(BudgetAct::Bias(v)) => v,
            _ => Vec::new(),
        };
        Ok(if !extra.is_empty() || needs_logits(&act.job.sp, VOCAB_VALID, act.produced) {
            act.sampler.sample(&act.job.sp, &agens_cuda::m1::logits_row(row)?, act.produced, &extra) as i32
        } else {
            greedy
        })
    }

    /// Emit a token; true when the request is finished.
    fn emit(act: &mut Act, tok: i32) -> bool {
        act.sampler.accept(tok as u32);
        act.produced += 1;
        let cancelled = act.job.tx.blocking_send(Ok(vec![tok as u32])).is_err();
        let stop = !act.job.sp.ignore_eos && act.job.sp.stop_token_ids.contains(&(tok as u32));
        cancelled || stop || act.produced >= act.job.sp.max_new_tokens
    }

    fn finish(&mut self, act: Act) -> Result<()> {
        let sess = match act.phase {
            Phase::Decode { sess, .. } => Some(sess),
            Phase::Last { fork } => Some(fork),
            Phase::Prefill { sess, .. } => Some(sess),
        };
        if let (Phase::Prefill { .. }, Some(k)) = (&act.phase, &act.job.conv) {
            // cancelled while the conversation's main session was being extended: its tokens no longer match `held`
            if let Some(c) = self.convs.remove(k) {
                self.free_session(c.main)?;
            }
            return Ok(());
        }
        let main = act.job.conv.as_ref().and_then(|k| self.convs.get(k)).map(|c| c.main);
        if let Some(s) = sess {
            if Some(s) != main {
                self.free_session(s)?;
            }
        }
        if let Some(k) = &act.job.conv {
            if let Some(c) = self.convs.get_mut(k) {
                c.busy = false;
                c.last = Instant::now();
                if !self.fork {
                    c.held.extend(act.fed.iter().map(|&t| t as u64)); // the session generated in place
                }
            }
            while self.convs.len() > self.max_convs {
                let Some(old) = self.convs.iter().filter(|(_, c)| !c.busy).min_by_key(|(_, c)| c.last).map(|(k, _)| k.clone())
                else { break };
                let c = self.convs.remove(&old).unwrap();
                self.free_session(c.main)?;
            }
        }
        Ok(())
    }

    fn run(&mut self, rx: smpsc::Receiver<Job>) -> Result<()> {
        use agens_cuda::m1;
        let mut waiting: VecDeque<Job> = VecDeque::new();
        let mut acts: Vec<Act> = Vec::new();
        loop {
            if acts.is_empty() && waiting.is_empty() {
                match rx.recv() {
                    Ok(j) => waiting.push_back(j),
                    Err(_) => return Ok(()),
                }
            }
            while let Ok(j) = rx.try_recv() {
                waiting.push_back(j);
            }
            // admission (requests whose client already went away are dropped here)
            while acts.len() < self.max_batch {
                let Some(j) = waiting.pop_front() else { break };
                if j.tx.is_closed() {
                    continue;
                }
                let forks_due = if self.fork {
                    acts.iter().filter(|a| a.job.conv.is_some() && matches!(a.phase, Phase::Prefill { .. })).count()
                } else {
                    0
                };
                match self.admit(&j, !acts.is_empty(), forks_due)? {
                    Admit::Ok(a) => acts.push(a),
                    Admit::Wait => {
                        waiting.push_front(j);
                        break;
                    }
                    Admit::Fail(e) => {
                        let _ = j.tx.blocking_send(Err(BackendError(e)));
                    }
                }
            }
            // one prefill chunk (oldest first)
            let mut done_ids = Vec::new();
            if let Some(i) = acts.iter().position(|a| !matches!(a.phase, Phase::Decode { .. })) {
                let a = &mut acts[i];
                match &mut a.phase {
                    Phase::Prefill { sess, feed, done, mm } => {
                        let s = *sess;
                        m1::session_use(s)?;
                        let end = (*done + CHUNK).min(feed.len());
                        let mut greedy = 0;
                        if end > *done {
                            let tp = Instant::now();
                            greedy = match mm {
                                Some(f) => {
                                    let rp = crate::mm::rope_slice(&f.rpos, feed.len(), *done, end);
                                    m1::prefill_mm(&feed[*done..end], Some(&rp), Some((f.emb.as_ptr(), &f.rows[*done..end])))?
                                }
                                None => m1::prefill(&feed[*done..end])?,
                            };
                            self.pstats.0 += 1;
                            self.pstats.1 += tp.elapsed().as_secs_f64();
                        }
                        *done = end;
                        let nfeed = feed.len();
                        if end == nfeed {
                            if a.job.conv.is_some() && self.fork {
                                let fork = m1::session_fork(s)?;
                                a.phase = Phase::Last { fork };
                            } else {
                                let t = Self::choose(a, greedy, 0)?;
                                a.phase = Phase::Decode { sess: s, last: t };
                                if Self::emit(a, t) {
                                    done_ids.push(a.job.id);
                                }
                                if a.job.conv.is_some() {
                                    eprintln!("serve: request {} prefilled {} tokens in {:.2} s", a.job.id, nfeed,
                                              a.t0.elapsed().as_secs_f64());
                                }
                            }
                        }
                    }
                    Phase::Last { fork } => {
                        let f = *fork;
                        m1::session_use(f)?;
                        let last = *a.job.prompt.last().unwrap() as i32;
                        let greedy = m1::prefill(&[last])?;
                        let t = Self::choose(a, greedy, 0)?;
                        a.phase = Phase::Decode { sess: f, last: t };
                        if Self::emit(a, t) {
                            done_ids.push(a.job.id);
                        }
                        eprintln!("serve: request {} prefilled {} tokens in {:.2} s", a.job.id, a.job.prompt.len(), a.t0.elapsed().as_secs_f64());
                    }
                    Phase::Decode { .. } => unreachable!(),
                }
            }
            // one decode step for every decoding row
            let rows: Vec<usize> =
                (0..acts.len()).filter(|&i| matches!(acts[i].phase, Phase::Decode { .. }) && !done_ids.contains(&acts[i].job.id)).collect();
            // server mode with the drafter (batched speculation): greedy rows verify together, one step for all
            let bspec = agens_cuda::m1::spec_batch_max();
            // (1.0.1) sampled rows too, each on its own filtered distribution and Philox stream; a sampled row verifies
            // batched even alone (its sample stream lives in its session), a lone greedy row takes the bs = 1 step
            let spec_rows: Vec<usize> = if bspec > 0 {
                rows.iter().copied().filter(|&i| acts[i].spec.is_some()).take(bspec).collect()
            } else {
                Vec::new()
            };
            let batched = spec_rows.len() > 1 || spec_rows.iter().any(|&i| acts[i].spec.map_or(false, |s| s.t > 0.0));
            for &i in &spec_rows {
                let a = &mut acts[i];
                if batched && !a.bset {
                    let (Phase::Decode { sess, .. }, Some(s)) = (&a.phase, a.spec) else { unreachable!() };
                    m1::session_sampling(*sess, s.t, s.k, s.p, s.min_p, a.job.sp.seed.unwrap_or(a.job.id) as i32)?;
                    a.bset = true;
                }
            }
            let rows: Vec<usize> = if batched {
                let (sess, last): (Vec<i32>, Vec<i32>) = spec_rows
                    .iter()
                    .map(|&i| match acts[i].phase {
                        Phase::Decode { sess, last } => (sess, last),
                        _ => unreachable!(),
                    })
                    .unzip();
                let hints: Vec<Vec<i32>> = spec_rows
                    .iter()
                    .zip(&last)
                    .map(|(&i, &l)| {
                        let a = &mut acts[i];
                        lookup_of(&mut a.lk, &a.job.prompt, l, false).propose()
                    })
                    .collect();
                let td = Instant::now();
                let em = if hints.iter().all(|h| h.is_empty()) {
                    m1::spec_batch(&sess, &last)?
                } else {
                    m1::spec_batch_hint(&sess, &last, &hints)?
                };
                for (j, &i) in spec_rows.iter().enumerate() {
                    if let Some(l) = acts[i].lk.as_mut() {
                        l.record(&em[j]);
                    }
                }
                let ntok = em.iter().map(|e| e.len() as u64).sum::<u64>();
                self.stats.0 += 1;
                self.stats.1 += ntok;
                self.stats.2 += td.elapsed().as_secs_f64();
                self.bstats.0 += 1;
                self.bstats.1 += ntok;
                self.bstats.2 += td.elapsed().as_secs_f64();
                self.bstats.3 += sess.len() as u64;
                for (j, &i) in spec_rows.iter().enumerate() {
                    let a = &mut acts[i];
                    a.vsteps += 1;
                    a.fed.push(last[j] as u32);
                    a.fed.extend(em[j][..em[j].len() - 1].iter().map(|&t| t as u32));
                    for &t in &em[j] {
                        if let Phase::Decode { last, .. } = &mut a.phase {
                            *last = t;
                        }
                        if Self::emit(a, t) {
                            done_ids.push(a.job.id);
                            break;
                        }
                    }
                }
                if self.bstats.0 == 128 {
                    let b = self.bstats;
                    let wall = self.pstats.2.map(|t| t.elapsed().as_secs_f64()).unwrap_or(b.2);
                    eprintln!(
                        "serve: last {} batched verify steps: {:.1} sequences, {:.1} tokens/step, {:.1} ms/step, \
                         {:.1} tok/s while verifying; {} prefill chunks {:.2} s; wall {:.2} s ({:.1} tok/s)",
                        b.0, b.3 as f64 / b.0 as f64, b.1 as f64 / b.0 as f64, 1e3 * b.2 / b.0 as f64, b.1 as f64 / b.2,
                        self.pstats.0, self.pstats.1, wall, b.1 as f64 / wall
                    );
                    self.bstats = (0, 0, 0.0, 0);
                    self.pstats = (0, 0.0, Some(Instant::now()));
                }
                rows.into_iter().filter(|i| !spec_rows.contains(i)).collect()
            } else {
                rows
            };
            if rows.len() == 1 && acts[rows[0]].spec.is_some() {
                // personal mode, speculative: one verify step emits 1..8 tokens
                let a = &mut acts[rows[0]];
                let Phase::Decode { sess, last } = a.phase else { unreachable!() };
                m1::session_use(sess)?;
                if a.produced == 1 {
                    // first verify step of the request: accept rule, seed, and the step counter of its sample stream
                    let s = a.spec.unwrap();
                    m1::spec_sampling_filtered(s.t, s.k, s.p, s.min_p, a.job.sp.seed.unwrap_or(a.job.id) as i32)?;
                }
                let copy = self.copy;
                let l = lookup_of(&mut a.lk, &a.job.prompt, last, copy);
                let hint = l.propose();
                let td = Instant::now();
                let em = if copy && hint.is_empty() && a.spec.map_or(false, |s| s.t > 0.0) {
                    // copy-only drafting, no proposal, sampled request: a plain decode step sampled on the host
                    let g = m1::decode(last)?;
                    vec![Self::choose(a, g, 0)?]
                } else if copy {
                    crate::lookup::copy_step(last, &hint)?
                } else if hint.is_empty() {
                    m1::spec_step(last)?.0
                } else {
                    m1::spec_step_hint(last, &hint)?.0
                };
                if let Some(l) = a.lk.as_mut() {
                    l.record(&em);
                }
                a.fed.push(last as u32); // the session committed the bonus and the accepted drafts
                a.fed.extend(em[..em.len() - 1].iter().map(|&t| t as u32));
                self.stats.0 += 1;
                self.stats.1 += em.len() as u64;
                self.stats.2 += td.elapsed().as_secs_f64();
                a.vsteps += 1;
                for &t in &em {
                    if let Phase::Decode { last, .. } = &mut a.phase {
                        *last = t;
                    }
                    if Self::emit(a, t) {
                        done_ids.push(a.job.id);
                        break;
                    }
                }
                if self.stats.0 % 256 == 0 {
                    eprintln!(
                        "serve: {} verify steps, {:.2} tokens/step, {:.1} ms/step, {:.1} tok/s",
                        self.stats.0,
                        self.stats.1 as f64 / self.stats.0 as f64,
                        1e3 * self.stats.2 / self.stats.0 as f64,
                        self.stats.1 as f64 / self.stats.2
                    );
                }
            } else if !rows.is_empty() {
                let (sess, toks): (Vec<i32>, Vec<i32>) = rows
                    .iter()
                    .map(|&i| match acts[i].phase {
                        Phase::Decode { sess, last } => (sess, last),
                        _ => unreachable!(),
                    })
                    .unzip();
                let td = Instant::now();
                // one row: the bs = 1 step (a CUDA graph replay); several: the batched step, whose rows are
                // bit-identical to it (agens m1-batch)
                let next = if sess.len() == 1 {
                    m1::session_use(sess[0])?;
                    vec![m1::decode(toks[0])?]
                } else {
                    m1::decode_batch(&sess, &toks)?
                };
                self.stats.0 += 1;
                self.stats.1 += rows.len() as u64;
                self.stats.2 += td.elapsed().as_secs_f64();
                for (j, &i) in rows.iter().enumerate() {
                    let a = &mut acts[i];
                    a.fed.push(toks[j] as u32);
                    let t = Self::choose(a, next[j], j)?;
                    if let Phase::Decode { last, .. } = &mut a.phase {
                        *last = t;
                    }
                    if Self::emit(a, t) {
                        done_ids.push(a.job.id);
                    }
                }
                if self.stats.0 % 256 == 0 {
                    eprintln!(
                        "serve: {} decode steps, {:.2} rows/step, {:.1} ms/step, {:.1} tok/s aggregate",
                        self.stats.0,
                        self.stats.1 as f64 / self.stats.0 as f64,
                        1e3 * self.stats.2 / self.stats.0 as f64,
                        self.stats.1 as f64 / self.stats.2
                    );
                }
            }
            // retire finished and cancelled requests
            for i in (0..acts.len()).rev() {
                if done_ids.contains(&acts[i].job.id) || acts[i].job.tx.is_closed() {
                    let a = acts.swap_remove(i);
                    if a.vsteps > 0 && std::env::var("AGENS_SERVE_REQ_STATS").map_or(false, |v| v == "1") {
                        // per-request acceptance (benchmarks): tokens generated, verify steps
                        eprintln!("serve: request {} done: prompt {} tokens, {} new tokens, {} verify steps",
                                  a.job.id, a.job.prompt.len(), a.produced, a.vsteps);
                    }
                    self.finish(a)?;
                }
            }
        }
    }
}

pub fn run(args: &[String]) -> Result<()> {
    let usage = "agens serve <ckpt_dir> [--model-dir DIR] [--host 0.0.0.0] [--port 30000] [--served-model-name NAME] \
                 [--mode personal|server] [--max-batch 8] [--max-convs 64] [--budget-gib N] [--spec-drafter DIR] [--spec-copy 1] [--no-fork 1] \
                 [m1 flags: --wctx --pool --sessions --gpu-gib --device]";
    let ckpt = args.first().context(usage)?.clone();
    let (mut model_dir, mut host, mut port, mut name) = (ckpt.clone(), "0.0.0.0".to_string(), 30000u16, None);
    let (mut mode, mut max_batch, mut max_convs, mut drafter) = ("server".to_string(), 8usize, 64usize, None);
    let mut fork = true;
    let mut rest = Vec::new();
    let mut i = 1;
    while i < args.len() {
        let v = args.get(i + 1).context(usage)?.clone();
        match args[i].as_str() {
            "--model-dir" => model_dir = v,
            "--host" => host = v,
            "--port" => port = v.parse()?,
            "--served-model-name" => name = Some(v),
            "--mode" => mode = v,
            "--max-batch" => max_batch = v.parse()?,
            "--max-convs" => max_convs = v.parse()?,
            "--spec-drafter" => drafter = Some(v),
            "--no-fork" => fork = v == "0",
            _ => {
                rest.push(args[i].clone());
                rest.push(v);
            }
        }
        i += 2;
    }
    match mode.as_str() {
        "personal" => max_batch = 1,
        "server" => {
            if !(1..=16).contains(&max_batch) {
                bail!("--max-batch must be 1..16");
            }
            std::env::set_var("AGENS_M1_BATCH_INVARIANT", "1");
            // one set of numerics for server mode, with or without --spec-drafter (1.0.1): the verify's batch-invariant
            // kernels (AGENS_M1_BI) for every decode row, and prefill chunks of up to 128 tokens on the batched W4
            // kernels (short prompts: no dequantized-weight GEMM). 1.0.0 set the second only with a drafter and decoded
            // without one on the AGENS_M1_BATCH_INVARIANT GEMV, so greedy replies differed with and without the drafter
            // (3 of the 13 gate prompts; tools/serve_gate.py). Now spec == non-spec in server mode, at every concurrency.
            for (k, v) in [("AGENS_M1_BI", "1"), ("AGENS_M1_W4_BATCH_MAX", "128")] {
                if std::env::var(k).is_err() {
                    std::env::set_var(k, v);
                }
            }
        }
        m => bail!("--mode {m}: personal or server"),
    }
    if drafter.is_some() && mode != "personal" && std::env::var("AGENS_SPEC_BATCH_MAX").is_err() {
        // server mode: batched speculative decoding of up to --max-batch greedy requests per step
        std::env::set_var("AGENS_SPEC_BATCH_MAX", max_batch.min(16).to_string());
    }
    if !fork && mode != "personal" {
        bail!("--no-fork 1: personal mode only");
    }
    if !rest.iter().any(|a| a == "--sessions") {
        // every decoding row holds a slot (plain session or conversation fork) and an active conversation's main
        // session one more; idle conversations get parked when slots run short
        rest.push("--sessions".into());
        rest.push(if fork { (2 * max_batch).max(2) } else { 1 }.to_string());
    }
    let mut o = crate::m1::parse_opts(&rest, usage)?;
    o.drafter = drafter.clone();
    let copy = o.copy;
    if copy && drafter.is_none() && mode != "personal" {
        bail!("--spec-copy 1 without a drafter: personal mode only");
    }
    let wctx = o.wctx;
    let think = {  // reasoning-budget markers, as the image (VOLUNDR_THINK_START_ID / _END_ID override)
        let tok = agens_server::tokenizer::Tokenizer::from_model_dir(&model_dir)?;
        let id = |env: &str, s: &str, dflt: u32| {
            std::env::var(env).ok().and_then(|v| v.parse().ok()).or_else(|| tok.token_to_id(s)).unwrap_or(dflt)
        };
        (id("VOLUNDR_THINK_START_ID", "<|think|>", 248068), id("VOLUNDR_THINK_END_ID", "<|/think|>", 248069))
    };
    // the engine thread owns the CUDA context: load there, report readiness, then serve jobs
    let (jtx, jrx) = smpsc::channel::<Job>();
    let (rtx, rrx) = smpsc::channel::<Result<()>>();
    std::thread::Builder::new().name("agens-engine".into()).spawn(move || {
        let r = (|| -> Result<Engine> {
            crate::m1::load(&ckpt, &o)?;
            let anchor = agens_cuda::m1::session_current();
            let buf = agens_cuda::host_alloc(agens_cuda::m1::park_bytes())?; // the anchor's state, never read back
            agens_cuda::m1::session_park(anchor, buf)?;
            crate::tp::start()?; // tensor parallel: followers execute the leader's calls from here (no HTTP)
            Ok(Engine { think, max_batch, max_convs, wctx: wctx - CHUNK, anchor, convs: HashMap::new(), stats: (0, 0, 0.0),
                        spec: drafter.is_some() || copy, copy: copy && drafter.is_none(), fork, bstats: (0, 0, 0.0, 0), pstats: (0, 0.0, None) })
        })();
        match r {
            Ok(mut e) => {
                let _ = rtx.send(Ok(()));
                if let Err(err) = e.run(jrx) {
                    eprintln!("serve: engine thread failed: {err:#}");
                    std::process::exit(1);
                }
            }
            Err(err) => {
                let _ = rtx.send(Err(err));
            }
        }
    })?;
    rrx.recv().context("engine thread died during load")??;
    let backend: Arc<dyn Backend> =
        Arc::new(EngineBackend { tx: Mutex::new(jtx), ctx: wctx - CHUNK, next_id: AtomicU64::new(1) });
    let state = agens_server::server::AppState::from_model_dir(&model_dir, backend, name)?;
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let addr = format!("{host}:{port}");
        let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("bind {addr}"))?;
        eprintln!("serve: {} on http://{addr}, mode {mode} (max batch {max_batch}), context {}", state.model_name, wctx - CHUNK);
        axum::serve(listener, agens_server::server::router(Arc::new(state))).await?;
        Ok(())
    })
}
