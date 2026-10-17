//! OpenAI request -> prompt parity with the reference's chat path (`_validate_request` +
//! `_process_messages` + `to_sampling_params`): tool dumps, argument parsing, `reasoning_effort` /
//! `enable_thinking` resolution, `continue_final_message`, content parts, error messages.

mod common;
use agens_server::openai::{self, ChatRequest};
use agens_server::pyjson::PyValue;
use agens_server::sampling::GenerationDefaults;
use common::*;

#[test]
fn chat_requests_match_reference() {
    let Some(tok) = tokenizer() else { return };
    let tmpl = template();
    let gen = GenerationDefaults::from_model_dir(fixtures().join("model"), tok.len()).unwrap();
    let f = fixture("request_cases.json");
    let cases = f.get("cases").unwrap().as_list().unwrap();
    let mut fails = Vec::new();
    for c in cases {
        let name = s(c.get("name").unwrap());
        let body = c.get("body").unwrap().dumps_compact();
        let res = ChatRequest::parse(body.as_bytes()).and_then(|mut r| {
            openai::validate_chat(&r, 262_144)?;
            let p = openai::plan_chat(&mut r, &tmpl, &tok, &gen)?;
            Ok((r, p))
        });
        match (res, opt(c, "error")) {
            (Err(e), Some(want)) => {
                // "validate": `_validate_request` -> BadRequestError; "value": ValueError -> BadRequest;
                // "validation": pydantic (message text not matched, see crate docs)
                let kind = s(c.get("error_kind").unwrap());
                let want_type = match kind {
                    "validate" => Some("BadRequestError"),
                    "value" => Some("BadRequest"),
                    _ => None,
                };
                if let Some(t) = want_type {
                    if e.message != s(want) || e.err_type != t {
                        fails.push(format!("{name}: error {:?}/{} != {:?}/{t}", e.message, e.err_type, s(want)));
                    }
                }
            }
            (Ok(_), Some(want)) => fails.push(format!("{name}: expected error {want:?}")),
            (Err(e), None) => fails.push(format!("{name}: unexpected error {e:?}")),
            (Ok((r, p)), None) => {
                if p.prompt_ids != ids(c.get("prompt_ids").unwrap()) {
                    fails.push(format!(
                        "{name}: prompt ids differ\n{}",
                        diff(&tok.decode(&p.prompt_ids, false), s(c.get("prompt").unwrap()))
                    ));
                }
                let sv = c.get("sampling").unwrap();
                let num = |k: &str| sv.get(k).and_then(|v| v.as_f64());
                let mut bad = Vec::new();
                if Some(p.sampling.temperature) != num("temperature") {
                    bad.push("temperature");
                }
                if Some(p.sampling.top_p) != num("top_p") {
                    bad.push("top_p");
                }
                if Some(p.sampling.top_k as f64) != num("top_k") {
                    bad.push("top_k");
                }
                if Some(p.sampling.min_p) != num("min_p") {
                    bad.push("min_p");
                }
                if p.sampling.max_new_tokens.map(|m| m as f64) != num("max_new_tokens") {
                    bad.push("max_new_tokens");
                }
                if p.sampling.seed.map(|m| m as f64) != num("sampling_seed") {
                    bad.push("seed");
                }
                if Some(p.sampling.repetition_penalty) != num("repetition_penalty") {
                    bad.push("repetition_penalty");
                }
                let want_stop: Vec<String> = match sv.get("stop") {
                    Some(PyValue::Str(x)) => vec![x.clone()],
                    Some(PyValue::List(v)) => v.iter().map(|x| s(x).to_string()).collect(),
                    _ => vec![],
                };
                if r.stop != want_stop {
                    bad.push("stop");
                }
                let want_ids: Vec<u32> = opt(sv, "stop_token_ids").map(ids).unwrap_or_default();
                if r.stop_token_ids != want_ids {
                    bad.push("stop_token_ids");
                }
                if PyValue::Bool(p.skip_special_tokens) != *c.get("skip_special_tokens").unwrap() {
                    bad.push("skip_special_tokens");
                }
                if PyValue::Bool(p.require_reasoning) != *c.get("require_reasoning").unwrap() {
                    bad.push("require_reasoning");
                }
                let budget =
                    opt(sv, "custom_params").and_then(|cp| cp.get("reasoning_budget")).and_then(|b| b.as_i64());
                if r.reasoning_budget != budget {
                    bad.push("reasoning_budget");
                }
                if !bad.is_empty() {
                    fails.push(format!("{name}: sampling fields differ: {bad:?} (reference {})", sv.dumps(false)));
                }
            }
        }
    }
    eprintln!("request parity: {}/{} cases", cases.len() - fails.len(), cases.len());
    assert!(fails.is_empty(), "{} mismatches:\n{}", fails.len(), fails.join("\n"));
}
