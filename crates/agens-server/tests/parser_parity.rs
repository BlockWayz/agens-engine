//! Reasoning + tool-call parser parity with sglang's `agens` parsers (Python classes run on the same inputs).
//!
//! Corpus: real engine generations on agent sessions (incl. truncated / mid-call outputs), agent transcripts
//! re-rendered to raw model output, and synthetic edge cases (malformed / partial / multiple calls, text after a
//! call, typed parameters, thinking on and off). Each case is checked non-streaming and streaming with three
//! chunkings (the detokenizer's token deltas, one char per chunk, random 1..12-char chunks), in both reasoning
//! modes.

mod common;
use agens_server::pyjson::PyValue;
use agens_server::reasoning::ReasoningParser;
use agens_server::toolcall::{self, ToolCallStreamParser, ToolSet};
use common::*;

fn calls_json(calls: &[toolcall::ToolCallItem]) -> Vec<PyValue> {
    calls
        .iter()
        .map(|c| {
            PyValue::Dict(vec![
                ("tool_index".into(), PyValue::int(c.tool_index)),
                ("name".into(), c.name.clone().map(PyValue::Str).unwrap_or(PyValue::Null)),
                ("parameters".into(), PyValue::Str(c.parameters.clone())),
            ])
        })
        .collect()
}

fn push(ev: &mut Vec<PyValue>, kind: &str, text: &str) {
    if text.is_empty() {
        return;
    }
    if let Some(PyValue::List(last)) = ev.last_mut() {
        if last[0].as_str() == Some(kind) {
            if let PyValue::Str(s) = &mut last[1] {
                s.push_str(text);
                return;
            }
        }
    }
    ev.push(PyValue::List(vec![PyValue::str(kind), PyValue::str(text)]));
}

#[test]
fn parsers_match_sglang() {
    let f = fixture("parser_corpus.json");
    let tool_sets = f.get("tool_sets").unwrap();
    let cases = f.get("cases").unwrap().as_list().unwrap();
    let (mut checks, mut ok) = (0usize, 0usize);
    let mut fails = Vec::new();
    let mut n_modes = 0;
    for c in cases {
        let name = s(c.get("name").unwrap());
        let text = s(c.get("text").unwrap());
        let tools = opt(c, "tools").map(|t| ToolSet::from_openai(tool_sets.get(s(t)).unwrap().as_list().unwrap()));
        let stream_reasoning = c.get("stream_reasoning").unwrap().truthy();
        for m in c.get("modes").unwrap().as_list().unwrap() {
            n_modes += 1;
            let force = m.get("force_reasoning").unwrap().truthy();
            // non-stream
            let want = m.get("nonstream").unwrap();
            let r = ReasoningParser::new(false, force, None).parse_non_stream(text);
            let mut got = vec![
                ("reasoning".to_string(), PyValue::Str(r.reasoning_text.clone())),
                ("normal".to_string(), PyValue::Str(r.normal_text.clone())),
                ("calls".to_string(), PyValue::Null),
                ("tool_normal".to_string(), PyValue::Null),
            ];
            if let Some(ts) = tools.as_ref().filter(|t| !t.is_empty()) {
                if toolcall::has_tool_call(&r.normal_text) {
                    let t = toolcall::detect_and_parse(&r.normal_text, ts);
                    got[3].1 = PyValue::Str(t.normal_text);
                    got[2].1 = PyValue::List(calls_json(&t.calls));
                }
            }
            checks += 1;
            if &PyValue::Dict(got.clone()) == want {
                ok += 1;
            } else {
                fails.push(format!(
                    "{name} force={force} non-stream:\n  got:      {}\n  expected: {}",
                    PyValue::Dict(got).dumps(false),
                    want.dumps(false)
                ));
            }
            // stream, per chunking
            for (cname, chunks) in c.get("chunks").unwrap().as_dict().unwrap() {
                let chunks: Vec<&str> = chunks.as_list().unwrap().iter().map(s).collect();
                let mut rp = ReasoningParser::new(stream_reasoning, force, None);
                let mut tp = ToolCallStreamParser::new();
                let mut ev = Vec::new();
                for (i, ch) in chunks.iter().enumerate() {
                    let mut r = rp.parse_stream_chunk(ch);
                    if i + 1 == chunks.len() {
                        let e = rp.finish();
                        r.reasoning_text.push_str(&e.reasoning_text);
                        r.normal_text.push_str(&e.normal_text);
                    }
                    push(&mut ev, "r", &r.reasoning_text);
                    push(&mut ev, "n", &r.normal_text);
                    if let Some(ts) = tools.as_ref().filter(|t| !t.is_empty()) {
                        let t = tp.parse(&r.normal_text, ts).expect("stream parse");
                        push(&mut ev, "t", &t.normal_text);
                        for cj in calls_json(&t.calls) {
                            ev.push(PyValue::List(vec![PyValue::str("c"), cj]));
                        }
                    }
                }
                let want = m.get("stream").unwrap().get(cname).unwrap();
                checks += 1;
                if &PyValue::List(ev.clone()) == want {
                    ok += 1;
                } else {
                    fails.push(format!(
                        "{name} force={force} stream/{cname}:\n  got:      {}\n  expected: {}",
                        PyValue::List(ev).dumps(false),
                        want.dumps(false)
                    ));
                }
            }
        }
    }
    eprintln!("parser parity: {ok}/{checks} checks over {} cases x modes ({} corpus entries)", n_modes, cases.len());
    assert!(fails.is_empty(), "{} mismatches:\n{}", fails.len(), fails.join("\n"));
}
