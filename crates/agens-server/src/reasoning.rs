// Ported from SGLang (Apache-2.0). See THIRD_PARTY_NOTICES.md.
//! Port of sglang's `agens` reasoning parser (`AgensReasoningDetector` on `BaseReasoningFormatDetector`).
//!
//! Assistant turns look like `<|think|>reasoning<|/think|>content[<|call|>...]`. The generation prompt already
//! opens `<|think|>` (or writes an empty think block when thinking is off), so with reasoning forced the output
//! starts inside the think block. A `<|call|>` before `<|/think|>` closes the think block implicitly.

pub const THINK_START: &str = "<|think|>";
pub const THINK_END: &str = "<|/think|>";
pub const TOOL_START: &str = "<|call|>";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParseResult {
    pub normal_text: String,
    pub reasoning_text: String,
}

#[derive(Debug, Clone)]
pub struct ReasoningParser {
    in_reasoning: bool,
    stream_reasoning: bool,
    buffer: String,
    stripped_think_start: bool,
    previous_content: String,
    force_nonempty_content: bool,
    accumulated_reasoning: String,
}

impl ReasoningParser {
    /// `force_reasoning`: the output starts inside the think block (thinking enabled for the request).
    /// `previous_content`: the final assistant message when `continue_final_message` is set.
    pub fn new(stream_reasoning: bool, force_reasoning: bool, previous_content: Option<&str>) -> Self {
        let previous_content = previous_content.unwrap_or("").to_string();
        let mut in_reasoning = force_reasoning;
        if previous_content.contains(THINK_START) {
            in_reasoning = true;
        }
        if previous_content.contains(THINK_END) {
            in_reasoning = false;
        }
        ReasoningParser {
            in_reasoning,
            stream_reasoning,
            buffer: String::new(),
            stripped_think_start: false,
            previous_content,
            force_nonempty_content: false,
            accumulated_reasoning: String::new(),
        }
    }

    /// `chat_template_kwargs.force_nonempty_content`: if nothing but reasoning came out, return it as content.
    pub fn with_force_nonempty_content(mut self, on: bool) -> Self {
        self.force_nonempty_content = on;
        self
    }

    fn force_nonempty(&self, mut r: ParseResult) -> ParseResult {
        if self.force_nonempty_content && r.normal_text.is_empty() {
            std::mem::swap(&mut r.normal_text, &mut r.reasoning_text);
        }
        r
    }

    /// One-shot parse of a complete output (`detect_and_parse`).
    pub fn parse_non_stream(&self, text: &str) -> ParseResult {
        let r = self.detect_and_parse(text);
        self.force_nonempty(r)
    }

    fn detect_and_parse(&self, text: &str) -> ParseResult {
        let in_reasoning = self.in_reasoning || text.contains(THINK_START);
        if !in_reasoning {
            return ParseResult { normal_text: text.to_string(), reasoning_text: String::new() };
        }
        let mut processed = text;
        while let Some(rest) = processed.strip_prefix(THINK_START) {
            processed = rest;
        }
        if !processed.contains(THINK_END) && !self.previous_content.contains(THINK_END) {
            if let Some(i) = processed.find(TOOL_START) {
                return ParseResult {
                    reasoning_text: processed[..i].to_string(),
                    normal_text: processed[i..].to_string(),
                };
            }
            return ParseResult { reasoning_text: processed.to_string(), normal_text: String::new() };
        }
        if let Some(i) = processed.find(THINK_END) {
            ParseResult {
                reasoning_text: processed[..i].to_string(),
                normal_text: processed[i + THINK_END.len()..].to_string(),
            }
        } else {
            ParseResult { normal_text: processed.to_string(), reasoning_text: String::new() }
        }
    }

    /// Incremental parse (`parse_streaming_increment`).
    pub fn parse_stream_chunk(&mut self, new_text: &str) -> ParseResult {
        let r = self.stream_impl(new_text);
        if self.force_nonempty_content {
            if self.in_reasoning {
                self.accumulated_reasoning.push_str(&r.reasoning_text);
            } else {
                self.accumulated_reasoning.clear();
            }
        }
        r
    }

    fn stream_impl(&mut self, new_text: &str) -> ParseResult {
        self.buffer.push_str(new_text);
        let current = self.buffer.clone();
        // A strict prefix of a sentinel: keep buffering.
        if [THINK_START, THINK_END, TOOL_START].iter().any(|t| t.starts_with(current.as_str()) && *t != current) {
            return ParseResult::default();
        }
        let mut current = current;
        if !self.stripped_think_start && current.contains(THINK_START) {
            current = current.replacen(THINK_START, "", 1);
            self.stripped_think_start = true;
            self.in_reasoning = true;
        }
        if self.in_reasoning {
            if let Some(i) = current.find(THINK_END) {
                self.buffer.clear();
                self.in_reasoning = false;
                return ParseResult {
                    reasoning_text: current[..i].to_string(),
                    normal_text: current[i + THINK_END.len()..].to_string(),
                };
            }
            if let Some(i) = current.find(TOOL_START) {
                self.buffer.clear();
                self.in_reasoning = false;
                return ParseResult { reasoning_text: current[..i].to_string(), normal_text: current[i..].to_string() };
            }
            if self.stream_reasoning {
                self.buffer.clear();
                return ParseResult { reasoning_text: current, normal_text: String::new() };
            }
            return ParseResult::default();
        }
        self.buffer.clear();
        ParseResult { normal_text: current, reasoning_text: String::new() }
    }

    /// End of stream (`finish`): only does something with `force_nonempty_content`.
    pub fn finish(&mut self) -> ParseResult {
        if self.force_nonempty_content && self.in_reasoning {
            let mut buffer = std::mem::take(&mut self.buffer);
            if let Some(rest) = buffer.strip_prefix(THINK_START) {
                buffer = rest.to_string();
            }
            let normal = std::mem::take(&mut self.accumulated_reasoning) + &buffer;
            if !normal.is_empty() {
                return ParseResult { normal_text: normal, reasoning_text: String::new() };
            }
        }
        ParseResult::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_stream() {
        let p = ReasoningParser::new(false, true, None);
        let r = p.parse_non_stream("think\n<|/think|>\n\nanswer");
        assert_eq!(r.reasoning_text, "think\n");
        assert_eq!(r.normal_text, "\n\nanswer");
        let r = p.parse_non_stream("think <|call|>x");
        assert_eq!((r.reasoning_text.as_str(), r.normal_text.as_str()), ("think ", "<|call|>x"));
        let p = ReasoningParser::new(false, false, None);
        assert_eq!(p.parse_non_stream("plain").normal_text, "plain");
    }

    #[test]
    fn stream_buffers_partial_sentinels() {
        let mut p = ReasoningParser::new(true, true, None);
        assert_eq!(p.parse_stream_chunk("ab").reasoning_text, "ab");
        assert_eq!(p.parse_stream_chunk("<|/th"), ParseResult::default());
        let r = p.parse_stream_chunk("ink|>x");
        assert_eq!((r.reasoning_text.as_str(), r.normal_text.as_str()), ("", "x"));
        // a chunk that only starts like a sentinel but continues otherwise is streamed as-is
        let mut p = ReasoningParser::new(true, true, None);
        assert_eq!(p.parse_stream_chunk("<|/thx").reasoning_text, "<|/thx");
    }
}
