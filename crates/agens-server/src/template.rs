//! Chat template rendering with minijinja, configured to behave like transformers' `apply_chat_template`
//! (`ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)` + `tojson` / `raise_exception`).
//!
//! Differences from stock minijinja that matter for byte parity:
//! * `tojson` is Python `json.dumps(x, ensure_ascii=False)`: ", " / ": " separators, no HTML escaping,
//!   insertion key order, Python float repr.
//! * `trim` strips Python's whitespace set (`str.strip()`), which includes U+001C..U+001F.
//! * str methods (`startswith`, `endswith`, ...) and dict `.items()` come from minijinja-contrib's pycompat.

use crate::pyjson::{self, PyValue};
use minijinja::value::{Kwargs, Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind};
use std::cell::RefCell;

thread_local! {
    static RAISED: RefCell<Option<String>> = const { RefCell::new(None) };
}

#[derive(Debug, Clone, PartialEq)]
pub enum TemplateError {
    /// `raise_exception(msg)` from the template (the reference turns this into a 400 with `msg`).
    Raised(String),
    /// Any other render failure.
    Render(String),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemplateError::Raised(m) | TemplateError::Render(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for TemplateError {}

pub struct ChatTemplate {
    env: Environment<'static>,
}

const NAME: &str = "chat_template";

impl ChatTemplate {
    pub fn new(source: impl Into<String>) -> Result<Self, TemplateError> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_keep_trailing_newline(false);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_filter("trim", py_trim);
        env.add_function("raise_exception", raise_exception);
        env.add_template_owned(NAME, source.into()).map_err(|e| TemplateError::Render(e.to_string()))?;
        Ok(ChatTemplate { env })
    }

    /// Render like `tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, **kwargs)`.
    pub fn render(
        &self,
        messages: &PyValue,
        tools: Option<&PyValue>,
        add_generation_prompt: bool,
        kwargs: &[(String, PyValue)],
    ) -> Result<String, TemplateError> {
        let tmpl = self.env.get_template(NAME).map_err(|e| TemplateError::Render(e.to_string()))?;
        let mut ctx: Vec<(String, Value)> = Vec::new();
        for (k, v) in kwargs {
            ctx.push((k.clone(), to_value(v)));
        }
        ctx.push(("messages".into(), to_value(messages)));
        ctx.push(("tools".into(), tools.map(to_value).unwrap_or(Value::from(()))));
        ctx.push(("documents".into(), Value::from(())));
        ctx.push(("add_generation_prompt".into(), Value::from(add_generation_prompt)));
        let ctx = Value::from_iter(ctx);
        RAISED.with(|r| r.borrow_mut().take());
        match tmpl.render(ctx) {
            Ok(s) => Ok(s),
            Err(e) => match RAISED.with(|r| r.borrow_mut().take()) {
                Some(m) => Err(TemplateError::Raised(m)),
                None => Err(TemplateError::Render(e.to_string())),
            },
        }
    }
}

fn raise_exception(msg: String) -> Result<Value, Error> {
    RAISED.with(|r| *r.borrow_mut() = Some(msg.clone()));
    Err(Error::new(ErrorKind::InvalidOperation, msg))
}

fn py_trim(v: Value, chars: Option<String>) -> Value {
    let s = v.to_string();
    let out = match chars {
        Some(c) => s.trim_matches(|x| c.contains(x)).to_string(),
        None => pyjson::py_strip(&s).to_string(),
    };
    Value::from(out)
}

fn tojson(v: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let indent: Option<Value> = kwargs.get("indent")?;
    let _separators: Option<Value> = kwargs.get("separators")?;
    kwargs.assert_all_used()?;
    if indent.as_ref().is_some_and(|i| !i.is_none()) {
        return Err(Error::new(ErrorKind::InvalidOperation, "tojson(indent=...) is not supported"));
    }
    let py = from_value(&v, sort_keys.unwrap_or(false))?;
    Ok(Value::from_safe_string(py.dumps(ensure_ascii.unwrap_or(false))))
}

/// PyValue -> minijinja value (maps keep insertion order: minijinja's `preserve_order` feature).
pub fn to_value(v: &PyValue) -> Value {
    match v {
        PyValue::Null => Value::from(()),
        PyValue::Bool(b) => Value::from(*b),
        PyValue::Int(s) => {
            if let Ok(i) = s.parse::<i64>() {
                Value::from(i)
            } else if let Ok(u) = s.parse::<u64>() {
                Value::from(u)
            } else if let Ok(i) = s.parse::<i128>() {
                Value::from(i)
            } else {
                Value::from(s.clone())
            }
        }
        PyValue::Float(f) => Value::from(*f),
        PyValue::Str(s) => Value::from(s.clone()),
        PyValue::List(items) => Value::from(items.iter().map(to_value).collect::<Vec<_>>()),
        PyValue::Dict(kv) => Value::from_iter(kv.iter().map(|(k, v)| (k.clone(), to_value(v)))),
    }
}

/// minijinja value -> PyValue for `tojson` (Python `json.dumps` semantics; Undefined is an error there too).
pub fn from_value(v: &Value, sort_keys: bool) -> Result<PyValue, Error> {
    Ok(match v.kind() {
        ValueKind::Undefined => {
            return Err(Error::new(ErrorKind::InvalidOperation, "Object of type Undefined is not JSON serializable"))
        }
        ValueKind::None => PyValue::Null,
        ValueKind::Bool => PyValue::Bool(v.is_true()),
        ValueKind::Number => {
            if !v.is_integer() {
                PyValue::Float(f64::try_from(v.clone()).unwrap_or(f64::NAN))
            } else if let Some(i) = v.as_i64() {
                PyValue::Int(i.to_string())
            } else if let Ok(i) = i128::try_from(v.clone()) {
                PyValue::Int(i.to_string())
            } else if let Ok(u) = u128::try_from(v.clone()) {
                PyValue::Int(u.to_string())
            } else {
                PyValue::Float(f64::try_from(v.clone()).unwrap_or(f64::NAN))
            }
        }
        ValueKind::String => PyValue::Str(v.as_str().unwrap_or_default().to_string()),
        ValueKind::Bytes => PyValue::Str(String::from_utf8_lossy(v.as_bytes().unwrap_or_default()).into_owned()),
        ValueKind::Seq | ValueKind::Iterable => {
            let mut items = Vec::new();
            for it in v.try_iter()? {
                items.push(from_value(&it, sort_keys)?);
            }
            PyValue::List(items)
        }
        ValueKind::Map => {
            let mut kv = Vec::new();
            for k in v.try_iter()? {
                let item = v.get_item(&k)?;
                let key = match k.kind() {
                    ValueKind::String => k.as_str().unwrap_or_default().to_string(),
                    ValueKind::None => "null".into(),
                    ValueKind::Bool => if k.is_true() { "true" } else { "false" }.into(),
                    _ => k.to_string(),
                };
                pyjson::dict_set(&mut kv, key, from_value(&item, sort_keys)?);
            }
            if sort_keys {
                kv.sort_by(|a, b| a.0.cmp(&b.0));
            }
            PyValue::Dict(kv)
        }
        _ => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("Object of type {} is not JSON serializable", v.kind()),
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyjson::{loads, Dialect};

    #[test]
    fn hf_semantics() {
        let t = ChatTemplate::new(
            "{%- for m in messages %}{{ m | tojson }}|{{ m.content|trim }}|{{ m.content.startswith('a') }}\n{% endfor %}\
             {%- if x is undefined %}U{% endif %}{%- set ns = namespace(n=1) %}{%- set ns.n = ns.n + 1 %}{{ ns.n }}\
             {%- for k, v in messages[0]|items %}{{ k }}{% endfor %}{{ (messages + messages)|length }}\
             {%- for m in messages[::-1] %}{{ loop.previtem is defined }}{% endfor %}\
             {%- if flag is true %}T{% endif %}{%- if flag2 is false %}F{% endif %}",
        )
        .unwrap();
        let msgs =
            loads(r#"[{"role":"user","content":" a\u001f ","n":1.0,"x":[1e20,null]}]"#, Dialect::Python).unwrap();
        let s = t
            .render(&msgs, None, true, &[("flag".into(), PyValue::Bool(true)), ("flag2".into(), PyValue::Bool(false))])
            .unwrap();
        assert_eq!(
            s,
            "{\"role\": \"user\", \"content\": \" a\\u001f \", \"n\": 1.0, \"x\": [1e+20, null]}|a|False\nU2rolecontentnx2FalseTF"
        );
    }

    #[test]
    fn raise_exception_message() {
        let t = ChatTemplate::new("{{ raise_exception('bad ' ~ x) }}").unwrap();
        let e = t.render(&PyValue::List(vec![]), None, true, &[("x".into(), PyValue::str("high"))]).unwrap_err();
        assert_eq!(e, TemplateError::Raised("bad high".into()));
    }
}
