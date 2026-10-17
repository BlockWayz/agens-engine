//! `agens-server` — OpenAI-compatible HTTP front end of Agens Engine.
//!
//! The CUDA backend is not wired in yet; `--backend mock` (the default for now) answers every request with a
//! fixed reply so the HTTP surface, templates and parsers can be exercised end to end.

use agens_server::backend::{Backend, MockBackend};
use agens_server::server::{router, AppState};
use anyhow::{bail, Context, Result};
use std::sync::Arc;

const USAGE: &str = "usage:
  agens-server --model-dir <dir> [--host 0.0.0.0] [--port 30000] [--served-model-name NAME]
               [--context-length N] [--backend mock]

  <dir> needs tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json, config.json";

const MOCK_REPLY: &str =
    "The mock backend is answering.\n<|/think|>\n\nThis is the Agens Engine mock backend: no model is loaded.";

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_dir = None;
    let mut host = "0.0.0.0".to_string();
    let mut port = 30000u16;
    let mut name = None;
    let mut context_length = None;
    let mut backend = "mock".to_string();
    let mut i = 0;
    while i < args.len() {
        let val = || args.get(i + 1).with_context(|| format!("{} needs a value\n{USAGE}", args[i]));
        match args[i].as_str() {
            "--model-dir" | "--model-path" => model_dir = Some(val()?.clone()),
            "--host" => host = val()?.clone(),
            "--port" => port = val()?.parse()?,
            "--served-model-name" => name = Some(val()?.clone()),
            "--context-length" => context_length = Some(val()?.parse::<usize>()?),
            "--backend" => backend = val()?.clone(),
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            o => bail!("unknown flag {o}\n{USAGE}"),
        }
        i += 2;
    }
    let model_dir = model_dir.context(USAGE)?;
    let backend: Arc<dyn Backend> = match backend.as_str() {
        "mock" => {
            let tok = agens_server::tokenizer::Tokenizer::from_model_dir(&model_dir)?;
            let mut reply = tok.encode(MOCK_REPLY);
            reply.extend(tok.token_to_id("<|agens_end|>"));
            eprintln!("agens-server: WARNING backend=mock, replies are canned");
            Arc::new(MockBackend::fixed(reply))
        }
        other => bail!("unknown backend {other:?} (available: mock)"),
    };
    let mut state = AppState::from_model_dir(&model_dir, backend, name)?;
    if let Some(n) = context_length {
        state.context_length = n;
    }
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("bind {addr}"))?;
    eprintln!("agens-server: serving {} on http://{addr}", state.model_name);
    axum::serve(listener, router(Arc::new(state))).await?;
    Ok(())
}
