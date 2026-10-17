//! Per-row token sampling on the CPU from one logits row (bf16 bits), for `agens serve`. Greedy rows without biases or
//! penalties take the GPU argmax instead (`needs_logits` = false). Order follows sglang's sampler: logit bias,
//! suppressed ids, penalties (repetition / frequency / presence over the generated tokens), min-new-tokens (stop ids
//! masked), temperature, top-k, top-p, min-p, then a draw from the renormalised distribution.

use agens_server::sampling::{SamplingParams, SAMPLING_EPS};

pub struct Sampler {
    rng: u64,
    counts: std::collections::HashMap<u32, u32>,
    produced: usize,
    pub budget: Option<Budget>,
}

/// The reference image's reasoning budget (sglang `volundr_budget.VolundrReasoningBudgetProcessor`), per request:
/// count tokens generated since `<|think|>` opened (in the prompt's last 16 tokens, or generated); after N of them
/// with no `<|/think|>`, force `<|/think|>` (every other logit -inf) and mask `<|think|>` for the rest of the request;
/// with a soft bias b, add b to the `<|/think|>` logit from 80% of the budget on.
pub struct Budget {
    start: u32,
    end: u32,
    n: usize,
    soft: f64,
    open: Option<usize>,
    closed: bool,
    forced: bool,
}

pub enum BudgetAct {
    None,
    Force(u32),
    /// additive logit changes (id, delta); -inf masks
    Bias(Vec<(u32, f32)>),
}

impl Budget {
    pub fn new(n: u32, soft: f64, prompt: &[u32], start: u32, end: u32) -> Self {
        let tail = &prompt[prompt.len().saturating_sub(16)..];
        let open = tail.iter().rposition(|&t| t == start).filter(|&i| !tail[i + 1..].contains(&end)).map(|_| 0);
        Budget { start, end, n: n as usize, soft, open, closed: false, forced: false }
    }
    /// token `t` was generated at output index `j`
    fn accept(&mut self, j: usize, t: u32) {
        if t == self.start {
            if self.open.is_none() && !self.closed {
                self.open = Some(j + 1);
            }
        } else if t == self.end && self.open.is_some() {
            self.open = None;
            self.closed = true;
        }
    }
    /// before sampling output token `produced`
    pub fn act(&mut self, produced: usize) -> BudgetAct {
        if self.closed {
            return if self.forced { BudgetAct::Bias(vec![(self.start, f32::NEG_INFINITY)]) } else { BudgetAct::None };
        }
        let Some(o) = self.open else { return BudgetAct::None };
        let k = produced - o;
        if k >= self.n {
            self.forced = true;
            self.closed = true;
            self.open = None;
            return BudgetAct::Force(self.end);
        }
        if self.soft != 0.0 && k as f64 >= 0.8 * self.n as f64 {
            return BudgetAct::Bias(vec![(self.end, self.soft as f32)]);
        }
        BudgetAct::None
    }
}

fn bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

pub fn is_greedy(p: &SamplingParams) -> bool {
    p.top_k == Some(1) || p.temperature < SAMPLING_EPS
}

/// Whether the row needs its logits on the host (anything beyond a plain argmax over the valid vocabulary).
pub fn needs_logits(p: &SamplingParams, vocab_valid: usize, produced: usize) -> bool {
    !is_greedy(p)
        || !p.logit_bias.is_empty()
        || p.repetition_penalty != 1.0
        || p.frequency_penalty != 0.0
        || p.presence_penalty != 0.0
        || (produced < p.min_new_tokens && !p.stop_token_ids.is_empty())
        || p.suppress_tokens.iter().any(|&t| (t as usize) < vocab_valid)
}

impl Sampler {
    pub fn new(p: &SamplingParams, request_id: u64) -> Self {
        let seed = p.seed.unwrap_or_else(|| {
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
            t ^ request_id.wrapping_mul(0x9e3779b97f4a7c15)
        });
        Sampler { rng: seed | 1, counts: Default::default(), produced: 0, budget: None }
    }

    fn uniform(&mut self) -> f64 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        (self.rng.wrapping_mul(0x2545f4914f6cdd1d) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Record an emitted token (penalties count generated tokens).
    pub fn accept(&mut self, tok: u32) {
        *self.counts.entry(tok).or_insert(0) += 1;
        if let Some(b) = self.budget.as_mut() {
            b.accept(self.produced, tok);
        }
        self.produced += 1;
    }

    /// `extra`: additive logit changes applied first (the reasoning budget's processor runs before sampling).
    pub fn sample(&mut self, p: &SamplingParams, logits: &[u16], produced: usize, extra: &[(u32, f32)]) -> u32 {
        let mut x: Vec<f32> = logits.iter().map(|&b| bf16(b)).collect();
        for &(t, d) in extra {
            if let Some(v) = x.get_mut(t as usize) {
                *v += d;
            }
        }
        for &(t, b) in &p.logit_bias {
            if let Some(v) = x.get_mut(t as usize) {
                *v += b as f32;
            }
        }
        for &t in &p.suppress_tokens {
            if let Some(v) = x.get_mut(t as usize) {
                *v = f32::NEG_INFINITY;
            }
        }
        for (&t, &c) in &self.counts {
            if let Some(v) = x.get_mut(t as usize) {
                if p.repetition_penalty != 1.0 {
                    let r = p.repetition_penalty as f32;
                    *v = if *v > 0.0 { *v / r } else { *v * r };
                }
                *v -= (p.frequency_penalty as f32) * c as f32 + if c > 0 { p.presence_penalty as f32 } else { 0.0 };
            }
        }
        if produced < p.min_new_tokens {
            for &t in &p.stop_token_ids {
                if let Some(v) = x.get_mut(t as usize) {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
        if is_greedy(p) {
            let mut best = 0usize;
            for i in 1..x.len() {
                if x[i] > x[best] {
                    best = i;
                }
            }
            return best as u32;
        }
        let t = p.temperature as f32;
        // candidates: top-k (or all), sorted by logit descending
        let mut idx: Vec<u32> = (0..x.len() as u32).filter(|&i| x[i as usize].is_finite()).collect();
        let k = p.top_k.map(|k| k as usize).unwrap_or(idx.len()).min(idx.len()).max(1);
        if k < idx.len() {
            idx.select_nth_unstable_by(k - 1, |&a, &b| x[b as usize].total_cmp(&x[a as usize]));
            idx.truncate(k);
        }
        idx.sort_unstable_by(|&a, &b| x[b as usize].total_cmp(&x[a as usize]).then(a.cmp(&b)));
        let m = x[idx[0] as usize] / t;
        let mut pr: Vec<f64> = idx.iter().map(|&i| ((x[i as usize] / t - m) as f64).exp()).collect();
        let z: f64 = pr.iter().sum();
        pr.iter_mut().for_each(|v| *v /= z);
        // top-p (keep the smallest prefix with mass >= top_p), min-p (relative to the top probability)
        let mut keep = pr.len();
        if p.top_p < 1.0 {
            let mut c = 0.0;
            for (j, v) in pr.iter().enumerate() {
                c += v;
                if c >= p.top_p {
                    keep = j + 1;
                    break;
                }
            }
        }
        if p.min_p > 0.0 {
            let thr = pr[0] * p.min_p;
            keep = keep.min(pr.iter().take_while(|&&v| v >= thr).count().max(1));
        }
        let z: f64 = pr[..keep].iter().sum();
        let mut u = self.uniform() * z;
        for j in 0..keep {
            u -= pr[j];
            if u <= 0.0 {
                return idx[j];
            }
        }
        idx[keep - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const S: u32 = 248068;
    const E: u32 = 248069;

    #[test]
    fn budget_forces_close_after_n_thinking_tokens() {
        // prompt ends with <|think|>: thinking is open from the first generated token
        let mut b = Budget::new(3, 0.0, &[1, 2, S], S, E);
        for k in 0..3 {
            assert!(matches!(b.act(k), BudgetAct::None));
            b.accept(k, 100 + k as u32);
        }
        assert!(matches!(b.act(3), BudgetAct::Force(t) if t == E));
        b.accept(3, E);
        // reopening is masked for the rest of the request
        assert!(matches!(b.act(4), BudgetAct::Bias(ref v) if v == &vec![(S, f32::NEG_INFINITY)]));
    }

    #[test]
    fn budget_counts_from_a_generated_open_and_applies_the_soft_bias() {
        let mut b = Budget::new(10, 2.5, &[1, 2, 3], S, E); // not open in the prompt
        assert!(matches!(b.act(0), BudgetAct::None));
        b.accept(0, S); // open from output index 1
        for k in 1..11 {
            let a = b.act(k);
            if k - 1 >= 8 {
                assert!(matches!(a, BudgetAct::Bias(ref v) if v == &vec![(E, 2.5)]), "step {k}");
            } else {
                assert!(matches!(a, BudgetAct::None), "step {k}");
            }
            b.accept(k, 7);
        }
        assert!(matches!(b.act(11), BudgetAct::Force(_)));
    }

    #[test]
    fn budget_inactive_after_a_natural_close() {
        let mut b = Budget::new(5, 0.0, &[S, 9, E, 4], S, E); // closed in the prompt: not open
        assert!(matches!(b.act(0), BudgetAct::None));
        let mut b = Budget::new(5, 0.0, &[S], S, E);
        b.accept(0, 1);
        b.accept(1, E); // the model closed it itself
        assert!(matches!(b.act(2), BudgetAct::None));
        b.accept(2, S); // a later <|think|> does not reopen (as the image: closed stays closed)
        assert!(matches!(b.act(3), BudgetAct::None));
    }
}
