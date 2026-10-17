//! Copy-from-context drafting (prompt lookup), the host half of hybrid speculation.
//!
//! The session's token ids (prompt + everything generated, the bonus token last) are indexed by their bigrams. Each
//! verify step looks up the longest earlier occurrence of the current suffix and proposes the tokens that followed
//! it (up to 7). The engine then verifies either that chain or the DFlash2 drafter's block (`spec_step_hint`), never
//! both, so the output is the target's own greedy / sampled sequence either way: lossless by construction.
//!
//! Choice per step: whichever proposal has the longer expected accepted run. The expectation is a running mean of the
//! drafts accepted per step, kept for the drafter and for the lookup per match-length bucket (the longer the matched
//! suffix, the likelier the copy continues), with priors that favour the drafter until a bucket has shown it wins.
//! AGENS_SPEC_LOOKUP=0 turns it off; AGENS_SPEC_LOOKUP_LOG=1 prints the per-bucket means at the end of a prompt.

use std::collections::HashMap;

const MAX_DRAFT: usize = 7;
const CANDIDATES: usize = 24; // most recent occurrences of the bigram examined per lookup
const BUCKETS: [usize; 6] = [2, 3, 4, 6, 10, 16]; // match length >= BUCKETS[b] (and < BUCKETS[b + 1])
const PRIOR_W: f64 = 4.0;
const ALPHA: f64 = 0.03; // running-mean weight of a new step once the prior is outweighed

#[derive(Clone, Copy)]
struct Mean {
    v: f64,
    n: f64,
}
impl Mean {
    fn add(&mut self, x: f64) {
        self.n += 1.0;
        let w = (1.0 / self.n).max(ALPHA);
        self.v += w * (x - self.v);
    }
}

pub struct Lookup {
    pub on: bool,
    hist: Vec<i32>,
    idx: HashMap<(i32, i32), Vec<u32>>, // bigram ending at position p -> p (positions < hist.len() - 1 only)
    indexed: usize,                      // positions [0, indexed) are in idx
    lk: [Mean; BUCKETS.len()],
    dr: Mean,
    pub used: usize,
    pub steps: usize,
    pub acc_lookup: usize, // drafts accepted on lookup steps
    pub acc_draft: usize,  // drafts accepted on drafter steps
    pending: Option<(bool, usize)>, // the last proposal: (lookup?, bucket)
}

fn bucket(m: usize) -> usize {
    BUCKETS.iter().rposition(|&b| m >= b).unwrap_or(0)
}

impl Lookup {
    pub fn new() -> Self {
        let on = std::env::var("AGENS_SPEC_LOOKUP").map(|v| v != "0").unwrap_or(true);
        // priors: accepted drafts per step; the drafter ~1.5-2.5 on these prompts, a copy after a short match rarely
        // continues, after a long one usually does
        let pri = [0.6, 1.0, 1.8, 2.6, 3.6, 4.6];
        let mut lk = [Mean { v: 0.0, n: 0.0 }; BUCKETS.len()];
        for (m, p) in lk.iter_mut().zip(pri) {
            *m = Mean { v: p, n: PRIOR_W };
        }
        Lookup { on, hist: Vec::new(), idx: HashMap::new(), indexed: 0, lk, dr: Mean { v: 2.0, n: PRIOR_W }, used: 0,
                 steps: 0, acc_lookup: 0, acc_draft: 0, pending: None }
    }

    /// A new sequence (prompt): the history starts over; the running means are kept (they describe the drafter and the
    /// workload, not one prompt).
    pub fn reset(&mut self, ids: &[i32]) {
        self.hist.clear();
        self.idx.clear();
        self.indexed = 0;
        self.pending = None;
        self.extend(ids);
    }

    /// Tokens appended to the session (prompt continuation, emitted tokens; the bonus last).
    pub fn extend(&mut self, toks: &[i32]) {
        self.hist.extend_from_slice(toks);
    }

    fn index_upto(&mut self, end: usize) {
        while self.indexed < end {
            let p = self.indexed;
            if p >= 1 {
                self.idx.entry((self.hist[p - 1], self.hist[p])).or_default().push(p as u32);
            }
            self.indexed += 1;
        }
    }

    /// The proposal for the step after the current history: (tokens, matched length), or None.
    fn find(&mut self) -> Option<(Vec<i32>, usize)> {
        let n = self.hist.len();
        if n < 3 {
            return None;
        }
        self.index_upto(n - 1); // never the current position itself
        let key = (self.hist[n - 2], self.hist[n - 1]);
        let cands = self.idx.get(&key)?;
        let mut best: Option<(usize, usize)> = None; // (match length, p)
        for &p in cands.iter().rev().take(CANDIDATES) {
            let p = p as usize;
            if p + 1 >= n {
                continue;
            }
            let mut m = 2;
            while m < 64 && m <= p && self.hist[p - m] == self.hist[n - 1 - m] {
                m += 1;
            }
            if best.map(|b| m > b.0).unwrap_or(true) {
                best = Some((m, p));
            }
        }
        let (m, p) = best?;
        let end = (p + 1 + MAX_DRAFT).min(n);
        let cont = self.hist[p + 1..end].to_vec();
        if cont.is_empty() {
            return None;
        }
        Some((cont, m))
    }

    /// The host proposal for this step if it is expected to beat the drafter (empty: use the drafter's block).
    pub fn propose(&mut self) -> Vec<i32> {
        self.steps += 1;
        self.pending = Some((false, 0));
        if !self.on {
            return Vec::new();
        }
        let Some((cont, m)) = self.find() else { return Vec::new() };
        let b = bucket(m);
        // a short continuation caps what the copy can win
        let exp = self.lk[b].v.min(cont.len() as f64);
        if exp > self.dr.v {
            self.pending = Some((true, b));
            self.used += 1;
            return cont;
        }
        Vec::new()
    }

    /// The step's outcome: `emitted` tokens (accepted drafts + the target's own next token).
    pub fn record(&mut self, emitted: &[i32]) {
        let acc = emitted.len().saturating_sub(1);
        match self.pending.take() {
            Some((true, b)) => {
                self.lk[b].add(acc as f64);
                self.acc_lookup += acc;
            }
            Some((false, _)) => {
                self.dr.add(acc as f64);
                self.acc_draft += acc;
            }
            None => {}
        }
        self.extend(emitted);
    }

    pub fn summary(&self) -> String {
        let lk: Vec<String> = BUCKETS.iter().zip(&self.lk).map(|(b, m)| format!(">={b}:{:.2}", m.v)).collect();
        format!(
            "lookup {}/{} steps (accepted drafts per step: lookup {:.2}, drafter {:.2}); means drafter {:.2}, lookup {}",
            self.used,
            self.steps,
            self.acc_lookup as f64 / self.used.max(1) as f64,
            self.acc_draft as f64 / (self.steps - self.used).max(1) as f64,
            self.dr.v,
            lk.join(" ")
        )
    }
}
