//! Request lifecycle between the HTTP layer and a [`Backend`]: finish conditions (max tokens, stop strings, stop
//! ids / EOS), stop-string hold-back while streaming, incremental detokenization and usage counts.
//!
//! This mirrors what the reference splits over its scheduler (`Req.update_finish_state`, the output streamer)
//! and detokenizer, so the text deltas a client sees are the same, step for step.

use crate::backend::{Backend, SessionHint};
use crate::sampling::SamplingParams;
use crate::tokenizer::{char_len, chars_from, chars_to, FinishReason, IncrementalDecoder, Tokenizer};
use std::sync::Arc;
use tokio::sync::mpsc;

/// The reference's `SGLANG_FORCE_STREAM_INTERVAL`: non-streaming requests still detokenize every 50 tokens.
pub const FORCE_STREAM_INTERVAL: usize = 50;

#[derive(Debug, Clone)]
pub struct EngineRequest {
    pub prompt_ids: Vec<u32>,
    pub sampling: SamplingParams,
    pub stop_strs: Vec<String>,
    /// Request `stop_token_ids` (EOS ids come from `eos_token_ids`).
    pub stop_token_ids: Vec<u32>,
    pub eos_token_ids: Vec<u32>,
    pub ignore_eos: bool,
    pub skip_special_tokens: bool,
    pub no_stop_trim: bool,
    pub stream: bool,
    /// Count tokens up to and including `think_end_ids` as reasoning tokens.
    pub require_reasoning: bool,
    pub think_end_ids: Vec<u32>,
    pub vocab_size: usize,
    pub session: Option<SessionHint>,
    /// images of the prompt (expanded placeholders in `prompt_ids`)
    pub mm: Option<Arc<crate::vision::MmInput>>,
}

/// One detokenizer output (what the reference's tokenizer manager hands the OpenAI layer per step).
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    pub delta: String,
    pub finish: Option<FinishReason>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub reasoning_tokens: usize,
}

struct ReqState<'a> {
    tok: &'a Tokenizer,
    r: &'a EngineRequest,
    output_ids: Vec<u32>,
    finished: Option<FinishReason>,
    finished_len: Option<usize>,
    stop_str_max_len: usize,
    reasoning_tokens: usize,
    reasoning_over: bool,
    think_match: usize,
}

impl<'a> ReqState<'a> {
    fn new(tok: &'a Tokenizer, r: &'a EngineRequest) -> Self {
        let stop_str_max_len = r.stop_strs.iter().map(|s| tok.encode(s).len()).max().unwrap_or(0);
        ReqState {
            tok,
            r,
            output_ids: Vec::new(),
            finished: None,
            finished_len: None,
            stop_str_max_len,
            reasoning_tokens: 0,
            reasoning_over: false,
            think_match: 0,
        }
    }

    fn update_reasoning_tokens(&mut self, step: &[u32]) {
        if !self.r.require_reasoning || self.r.think_end_ids.is_empty() || self.reasoning_over {
            return;
        }
        let pat = &self.r.think_end_ids;
        for (pos, &t) in step.iter().enumerate() {
            self.think_match = if t == pat[self.think_match] {
                self.think_match + 1
            } else if t == pat[0] {
                1
            } else {
                0
            };
            if self.think_match == pat.len() {
                self.reasoning_tokens += pos + 1;
                self.reasoning_over = true;
                return;
            }
        }
        self.reasoning_tokens += step.len();
    }

    fn tail_len(&self, new_len: usize) -> usize {
        (self.stop_str_max_len + 1 + new_len.saturating_sub(1)).min(self.output_ids.len())
    }

    fn tail_str(&self, new_len: usize) -> String {
        if self.r.stop_strs.is_empty() {
            return String::new();
        }
        let n = self.tail_len(new_len);
        self.tok.decode(&self.output_ids[self.output_ids.len() - n..], false)
    }

    /// `check_match_stop_str_prefix`: hold streaming output while the tail could still become a stop string.
    fn holds_stop_prefix(&self) -> bool {
        if self.r.stop_strs.is_empty() {
            return false;
        }
        let tail = self.tail_str(1);
        if tail.is_empty() {
            return false;
        }
        let tl = char_len(&tail);
        for stop in &self.r.stop_strs {
            if stop.is_empty() {
                continue;
            }
            if tail.contains(stop.as_str()) {
                return true;
            }
            for i in 1..=tl.min(char_len(stop)) {
                if chars_from(&tail, tl - i) == chars_to(stop, i) {
                    return true;
                }
            }
        }
        false
    }

    fn locate_stop(&self, new_len: usize, stop: &str) -> usize {
        let tail_len = self.tail_len(new_len);
        let start = self.output_ids.len() - tail_len;
        let window = &self.output_ids[start..];
        let from = (window.len() + 1).saturating_sub(new_len).max(1);
        for count in from..window.len() {
            if self.tok.decode(&window[..count], false).contains(stop) {
                return start + count;
            }
        }
        self.output_ids.len()
    }

    /// `Req.update_finish_state`.
    fn update_finish(&mut self, new_len: usize) {
        if self.finished.is_some() {
            return;
        }
        let max_new = self.r.sampling.max_new_tokens;
        if self.output_ids.len() >= max_new {
            self.finished = Some(FinishReason::Length(max_new));
            self.finished_len = Some(max_new);
            return;
        }
        let base = self.output_ids.len() - new_len;
        for i in 0..new_len {
            let t = self.output_ids[base + i];
            if t as usize >= self.r.vocab_size {
                if let Some(&s) = self.r.stop_token_ids.first() {
                    self.output_ids[base + i] = s;
                }
                if let Some(&e) = self.r.eos_token_ids.first() {
                    self.output_ids[base + i] = e;
                }
                self.finished = Some(FinishReason::MatchedStr("NaN happened".into()));
                self.finished_len = Some(base + i + 1);
                return;
            }
        }
        if !self.r.stop_strs.is_empty() {
            let tail = self.tail_str(new_len);
            for stop in &self.r.stop_strs {
                if tail.contains(stop.as_str()) {
                    self.finished = Some(FinishReason::MatchedStr(stop.clone()));
                    self.finished_len = Some(self.locate_stop(new_len, stop));
                    return;
                }
            }
        }
        if !self.r.ignore_eos {
            for i in 0..new_len {
                let t = self.output_ids[base + i];
                if self.r.stop_token_ids.contains(&t) || self.r.eos_token_ids.contains(&t) {
                    self.finished = Some(FinishReason::MatchedToken(t));
                    self.finished_len = Some(base + i + 1);
                    return;
                }
            }
        }
    }

    fn ids_through_stop(&self) -> &[u32] {
        match self.finished_len {
            Some(n) => &self.output_ids[..n.min(self.output_ids.len())],
            None => &self.output_ids,
        }
    }
}

/// Run one request: drive the backend, return per-step outputs (deltas) on a channel. The last item has
/// `finish` set. Dropping the receiver cancels the request.
pub fn run(
    tok: Arc<Tokenizer>,
    backend: Arc<dyn Backend>,
    req: EngineRequest,
) -> mpsc::Receiver<Result<Output, String>> {
    let (tx, rx) = mpsc::channel(64);
    let mut sampling = req.sampling.clone();
    sampling.stop_token_ids = if req.ignore_eos {
        Vec::new()
    } else {
        let mut v = req.eos_token_ids.clone();
        v.extend(&req.stop_token_ids);
        v.sort_unstable();
        v.dedup();
        v
    };
    sampling.ignore_eos = req.ignore_eos;
    let mut steps = match &req.mm {
        Some(mm) => backend.generate_mm(req.prompt_ids.clone(), mm.clone(), sampling, req.session.clone()),
        None => backend.generate(req.prompt_ids.clone(), sampling, req.session.clone()),
    };
    tokio::spawn(async move {
        let tok = tok.as_ref();
        let mut st = ReqState::new(tok, &req);
        let mut dec = IncrementalDecoder::new(&req.prompt_ids, req.skip_special_tokens, req.no_stop_trim);
        let mut sent_ids = 0usize;
        if req.sampling.max_new_tokens == 0 {
            st.finished = Some(FinishReason::Length(0));
            st.finished_len = Some(0);
            let out = Output {
                delta: dec.step(tok, &[], st.finished.as_ref()),
                finish: st.finished.clone(),
                prompt_tokens: req.prompt_ids.len(),
                completion_tokens: 0,
                reasoning_tokens: 0,
            };
            let _ = tx.send(Ok(out)).await;
            return;
        }
        loop {
            // 1.1.1: a request whose client went away (its receiver dropped: a closed connection, streaming or not)
            // ends now, not at its next output; dropping `steps` cancels the backend request at its next step
            let recv = tokio::select! {
                biased;
                _ = tx.closed() => return,
                r = steps.recv() => r,
            };
            let step = match recv {
                Some(Ok(s)) => s,
                Some(Err(e)) => {
                    let _ = tx.send(Err(e.0)).await;
                    return;
                }
                None => {
                    // Backend ended without a finish condition: report it as an abort.
                    st.finished = Some(FinishReason::Abort("backend stream ended early".into()));
                    Vec::new()
                }
            };
            if !step.is_empty() {
                st.output_ids.extend_from_slice(&step);
                st.update_reasoning_tokens(&step);
                st.update_finish(step.len());
            }
            let finished = st.finished.is_some();
            if finished && st.finished_len.is_none() {
                st.finished_len = Some(st.output_ids.len());
            }
            let should_output = finished
                || if req.stream { !st.holds_stop_prefix() } else { st.output_ids.len().is_multiple_of(FORCE_STREAM_INTERVAL) };
            if !should_output {
                continue;
            }
            let ids = st.ids_through_stop();
            let new_ids = ids[sent_ids.min(ids.len())..].to_vec();
            sent_ids = ids.len();
            let out = Output {
                delta: dec.step(tok, &new_ids, st.finished.as_ref()),
                finish: st.finished.clone(),
                prompt_tokens: req.prompt_ids.len(),
                completion_tokens: ids.len(),
                reasoning_tokens: st.reasoning_tokens,
            };
            if tx.send(Ok(out)).await.is_err() || finished {
                return; // dropping `steps` cancels the backend request
            }
        }
    });
    rx
}
