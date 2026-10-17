//! Chat template parity: `tokenizer.apply_chat_template(messages, tools=..., tokenize=False,
//! add_generation_prompt=True, **kwargs)` from transformers vs our minijinja rendering.

mod common;
use common::*;

#[test]
fn template_matches_transformers() {
    let t = template();
    let tok = tokenizer();
    let cases = fixture("template_cases.json");
    let cases = cases.get("cases").unwrap().as_list().unwrap();
    assert!(cases.len() >= 40, "need at least 40 template cases, have {}", cases.len());
    let mut fails = Vec::new();
    let mut id_checks = 0;
    for c in cases {
        let name = s(c.get("name").unwrap());
        let kwargs: Vec<(String, agens_server::pyjson::PyValue)> =
            c.get("kwargs").and_then(|k| k.as_dict()).cloned().unwrap_or_default();
        let got = t.render(c.get("messages").unwrap(), opt(c, "tools"), true, &kwargs);
        match (got, opt(c, "expected"), opt(c, "error")) {
            (Ok(g), Some(e), _) => {
                if g != s(e) {
                    fails.push(format!("{name}: {}", diff(&g, s(e))));
                } else if let Some(tok) = &tok {
                    id_checks += 1;
                    if tok.encode(&g) != ids(c.get("ids").unwrap()) {
                        fails.push(format!("{name}: token ids differ"));
                    }
                }
            }
            (Err(g), None, Some(e)) => {
                if g.to_string() != s(e) {
                    fails.push(format!("{name}: error {:?} != {:?}", g.to_string(), s(e)));
                }
            }
            (g, e, err) => fails.push(format!("{name}: got {g:?}, expected {e:?} / error {err:?}")),
        }
    }
    eprintln!(
        "template parity: {}/{} cases ({} prompt encodings checked)",
        cases.len() - fails.len(),
        cases.len(),
        id_checks
    );
    assert!(fails.is_empty(), "{} template mismatches:\n{}", fails.len(), fails.join("\n"));
}
