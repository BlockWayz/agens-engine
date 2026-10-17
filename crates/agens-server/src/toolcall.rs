// Ported from SGLang (Apache-2.0). See THIRD_PARTY_NOTICES.md.
//! Port of sglang's `agens` tool-call parser (`AgensDetector`: XML-style tool calls with Agens sentinels).
//!
//! ```text
//! <|call|>
//! <function=NAME>
//! <parameter=ARG>
//! VALUE
//! </parameter>
//! </function>
//! <|/call|>
//! ```
//! Parameter values are converted with the tool's JSON schema (int / float / bool / object / array, else the raw
//! string) and re-serialised the way Python does (`json.dumps`, ", " / ": " separators), so argument strings
//! match the reference byte for byte.

use crate::pyjson::{self, Dialect, PyValue};

pub const CALL_START: &str = "<|call|>";
pub const CALL_END: &str = "<|/call|>";
const FUNC_PREFIX: &str = "<function=";
const FUNC_END: &str = "</function>";
const PARAM_PREFIX: &str = "<parameter=";
const PARAM_END: &str = "</parameter>";

/// One parsed (or streamed) tool call item, like sglang's `ToolCallItem`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallItem {
    pub tool_index: i64,
    pub name: Option<String>,
    /// JSON text (complete in non-stream mode, a fragment in stream mode).
    pub parameters: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolParseResult {
    pub normal_text: String,
    pub calls: Vec<ToolCallItem>,
}

/// The tools of a request as the parser sees them: `(type, name, parameters)` per tool (after the reference's
/// schema type normalisation).
#[derive(Debug, Clone, Default)]
pub struct ToolSet {
    tools: Vec<(String, String, Option<PyValue>)>,
}

impl ToolSet {
    /// From OpenAI `tools` entries (`{"type": "function", "function": {"name", "parameters", ...}}`).
    pub fn from_openai(tools: &[PyValue]) -> Self {
        let mut out = Vec::new();
        for t in tools {
            let ty = t.get("type").and_then(|v| v.as_str()).unwrap_or("function").to_string();
            let Some(f) = t.get("function") else { continue };
            let name = f.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let params = f.get("parameters").filter(|p| !p.is_null()).map(|p| {
                let mut p = p.clone();
                normalize_json_schema_types(&mut p);
                p
            });
            out.push((ty, name, params));
        }
        ToolSet { tools: out }
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// `_get_arguments_config`: the `properties` dict (or the parameters dict itself) of the named function.
    fn arguments_config(&self, func_name: &str) -> Option<&PyValue> {
        for (ty, name, params) in &self.tools {
            if ty == "function" && name == func_name {
                return match params {
                    Some(p @ PyValue::Dict(_)) => Some(p.get("properties").unwrap_or(p)),
                    _ => None,
                };
            }
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(pub String);

/// Python `x in config` for the parameter config (dict keys, list items, or substring for str).
fn config_contains(config: Option<&PyValue>, name: &str) -> Result<bool, ParseError> {
    match config {
        None => Ok(false),
        Some(PyValue::Dict(kv)) => Ok(kv.iter().any(|(k, _)| k == name)),
        Some(PyValue::List(items)) => Ok(items.iter().any(|v| v.as_str() == Some(name))),
        Some(PyValue::Str(s)) => Ok(s.contains(name)),
        Some(other) => Err(ParseError(format!("argument of type {other:?} is not iterable"))),
    }
}

/// `_convert_param_value`.
pub fn convert_param_value(value: &str, name: &str, config: Option<&PyValue>) -> Result<PyValue, ParseError> {
    if value.to_lowercase() == "null" {
        return Ok(PyValue::Null);
    }
    if !config_contains(config, name)? {
        return Ok(PyValue::str(value));
    }
    let schema = match config {
        Some(PyValue::Dict(kv)) => kv.iter().find(|(k, _)| k == name).map(|(_, v)| v),
        _ => None, // `config[name]` on a list / str raises in the reference
    };
    let Some(schema) = schema else {
        return Err(ParseError("unsubscriptable parameter config".into()));
    };
    let ty = infer_type_from_json_schema(schema).unwrap_or_else(|| "string".into());
    let ty = pyjson::py_strip(&ty).to_lowercase();
    if ["string", "str", "text", "varchar", "char", "enum"].contains(&ty.as_str()) {
        return Ok(PyValue::str(value));
    }
    if ["int", "uint", "long", "short", "unsigned"].iter().any(|p| ty.starts_with(p)) {
        return Ok(match pyjson::py_int(value) {
            Some(i) => PyValue::Int(i),
            None => PyValue::str(value),
        });
    }
    if ty.starts_with("num") || ty.starts_with("float") {
        let maybe_convert = !(value.contains('.') || value.to_lowercase().contains('e'));
        return Ok(match pyjson::py_float(value) {
            Some(f) if maybe_convert && f.is_finite() && f.fract() == 0.0 => {
                PyValue::Int(pyjson::float_to_int_string(f))
            }
            Some(f) => PyValue::Float(f),
            None => PyValue::str(value),
        });
    }
    if ["boolean", "bool", "binary"].contains(&ty.as_str()) {
        return Ok(PyValue::Bool(value.to_lowercase() == "true"));
    }
    if ["object", "array", "arr"].contains(&ty.as_str()) || ty.starts_with("dict") || ty.starts_with("list") {
        if let Ok(v) = pyjson::loads(value, Dialect::Python) {
            return Ok(v);
        }
    }
    Ok(pyjson::literal_eval(value).unwrap_or_else(|| PyValue::str(value)))
}

/// `infer_type_from_json_schema` (sglang `function_call/utils.py`).
pub fn infer_type_from_json_schema(schema: &PyValue) -> Option<String> {
    let PyValue::Dict(_) = schema else { return None };
    if let Some(t) = schema.get("type") {
        match t {
            PyValue::Str(s) => return Some(s.clone()),
            PyValue::List(items) if !items.is_empty() => {
                let non_null: Vec<&PyValue> = items.iter().filter(|t| t.as_str() != Some("null")).collect();
                return Some(match non_null.first() {
                    Some(PyValue::Str(s)) => s.clone(),
                    Some(other) => py_str(other),
                    None => "string".into(),
                });
            }
            _ => {}
        }
    }
    if schema.get("anyOf").is_some() || schema.get("oneOf").is_some() {
        let any = schema.get("anyOf").filter(|v| v.truthy());
        let schemas = any.or_else(|| schema.get("oneOf"));
        if let Some(PyValue::List(subs)) = schemas {
            let types: Vec<String> =
                subs.iter().filter_map(infer_type_from_json_schema).filter(|t| !t.is_empty()).collect();
            if !types.is_empty() {
                let mut uniq: Vec<&String> = Vec::new();
                for t in &types {
                    if !uniq.contains(&t) {
                        uniq.push(t);
                    }
                }
                if uniq.len() == 1 {
                    return Some(types[0].clone());
                }
                if uniq.len() == 2 && types.iter().any(|t| t == "null") {
                    return types.iter().find(|t| *t != "null").cloned();
                }
                if types.iter().any(|t| t == "string") {
                    return Some("string".into());
                }
                return Some(types[0].clone());
            }
        }
    }
    if let Some(PyValue::List(vals)) = schema.get("enum") {
        if vals.is_empty() {
            return Some("string".into());
        }
        let mut kinds: Vec<&str> = Vec::new();
        for v in vals {
            let k = match v {
                PyValue::Null => "null",
                PyValue::Bool(_) => "boolean",
                PyValue::Int(_) => "integer",
                PyValue::Float(_) => "number",
                PyValue::Str(_) => "string",
                PyValue::List(_) => "array",
                PyValue::Dict(_) => "object",
            };
            if !kinds.contains(&k) {
                kinds.push(k);
            }
        }
        return Some(if kinds.len() == 1 { kinds[0].into() } else { "string".into() });
    }
    if let Some(PyValue::List(subs)) = schema.get("allOf") {
        for s in subs {
            if let Some(t) = infer_type_from_json_schema(s) {
                if !t.is_empty() && t != "string" {
                    return Some(t);
                }
            }
        }
        return Some("string".into());
    }
    if schema.get("properties").is_some() {
        return Some("object".into());
    }
    if schema.get("items").is_some() {
        return Some("array".into());
    }
    None
}

/// Python `str(x)` for the odd non-string `type` entries.
fn py_str(v: &PyValue) -> String {
    match v {
        PyValue::Null => "None".into(),
        PyValue::Bool(true) => "True".into(),
        PyValue::Bool(false) => "False".into(),
        PyValue::Int(s) => s.clone(),
        PyValue::Float(f) => pyjson::float_repr(*f),
        PyValue::Str(s) => s.clone(),
        other => other.dumps(false),
    }
}

// ------------------------------------------------------------------------------------------------ schema types

const STANDARD_TYPES: [&str; 7] = ["null", "boolean", "object", "array", "number", "string", "integer"];
const TYPE_ALIASES: [(&str, &str); 25] = [
    ("str", "string"),
    ("text", "string"),
    ("varchar", "string"),
    ("char", "string"),
    ("enum", "string"),
    ("uuid", "string"),
    ("date", "string"),
    ("datetime", "string"),
    ("time", "string"),
    ("timestamp", "string"),
    ("binary", "string"),
    ("blob", "string"),
    ("bytea", "string"),
    ("bytes", "string"),
    ("varbinary", "string"),
    ("bool", "boolean"),
    ("bigint", "integer"),
    ("smallint", "integer"),
    ("tinyint", "integer"),
    ("double", "number"),
    ("decimal", "number"),
    ("real", "number"),
    ("numeric", "number"),
    ("arr", "array"),
    ("tuple", "array"),
];
const MORE_ALIASES: [(&str, &str); 2] = [("set", "array"), ("map", "object")];
const PREFIX_RULES: [(&[&str], &str); 4] = [
    (&["int", "uint", "long", "short", "unsigned"], "integer"),
    (&["num", "float"], "number"),
    (&["list"], "array"),
    (&["dict"], "object"),
];

fn normalize_single_type(raw: &PyValue) -> PyValue {
    let PyValue::Str(s) = raw else { return raw.clone() };
    if STANDARD_TYPES.contains(&s.as_str()) {
        return raw.clone();
    }
    let base = pyjson::py_strip(s.split('(').next().unwrap_or("")).to_lowercase();
    if STANDARD_TYPES.contains(&base.as_str()) {
        return PyValue::Str(base);
    }
    if let Some((_, t)) = TYPE_ALIASES.iter().chain(MORE_ALIASES.iter()).find(|(a, _)| *a == base) {
        return PyValue::str(*t);
    }
    for (prefixes, target) in PREFIX_RULES {
        for p in prefixes.iter() {
            if base == *p {
                return PyValue::str(target);
            }
            if base.len() > p.len() && base.starts_with(p) {
                let next = base[p.len()..].chars().next().unwrap();
                if "0123456789[<( \t".contains(next) {
                    return PyValue::str(target);
                }
            }
        }
    }
    raw.clone()
}

/// `normalize_json_schema_types`: rewrite DB/ORM-style `type` names (varchar, int, ...) in place. The rewritten
/// schema is also what the chat template renders.
pub fn normalize_json_schema_types(schema: &mut PyValue) {
    match schema {
        PyValue::List(items) => {
            for it in items {
                normalize_json_schema_types(it);
            }
        }
        PyValue::Dict(_) => {
            if let Some(t) = schema.get_mut("type") {
                match t {
                    PyValue::Str(_) => *t = normalize_single_type(t),
                    PyValue::List(items) => {
                        let mut out: Vec<PyValue> = Vec::new();
                        for it in items.iter() {
                            let n = normalize_single_type(it);
                            if !out.contains(&n) {
                                out.push(n);
                            }
                        }
                        *items = out;
                    }
                    _ => {}
                }
            }
            for key in ["properties", "patternProperties", "$defs", "definitions", "dependentSchemas"] {
                if let Some(PyValue::Dict(kv)) = schema.get_mut(key) {
                    for (_, v) in kv.iter_mut() {
                        normalize_json_schema_types(v);
                    }
                }
            }
            for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
                if let Some(PyValue::List(items)) = schema.get_mut(key) {
                    for v in items.iter_mut() {
                        normalize_json_schema_types(v);
                    }
                }
            }
            for key in [
                "items",
                "additionalProperties",
                "not",
                "if",
                "then",
                "else",
                "contains",
                "propertyNames",
                "unevaluatedItems",
                "unevaluatedProperties",
            ] {
                if let Some(v) = schema.get_mut(key) {
                    normalize_json_schema_types(v);
                }
            }
        }
        _ => {}
    }
}

// ------------------------------------------------------------------------------------------------ non-stream

/// `has_tool_call`.
pub fn has_tool_call(text: &str) -> bool {
    text.contains(CALL_START)
}

/// `<function=(.*?)</function>|<function=(.*)$` findall.
fn find_functions(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(i) = s[pos..].find(FUNC_PREFIX) {
        let start = pos + i + FUNC_PREFIX.len();
        match s[start..].find(FUNC_END) {
            Some(j) => {
                out.push(&s[start..start + j]);
                pos = start + j + FUNC_END.len();
            }
            None => {
                out.push(&s[start..]);
                break;
            }
        }
    }
    out
}

/// `<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)` findall (DOTALL; `$` also matches
/// before a final newline).
fn find_parameters(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut pos = 0;
    let dollar = if s.ends_with('\n') { s.len() - 1 } else { s.len() };
    while let Some(i) = s[pos..].find(PARAM_PREFIX) {
        let start = pos + i + PARAM_PREFIX.len();
        let rest = &s[start..];
        let mut best: Option<(usize, usize)> = None; // (end of group, end of match)
        let mut consider = |k: usize, consumed: usize| {
            if best.is_none_or(|(b, _)| k < b) {
                best = Some((k, k + consumed));
            }
        };
        if let Some(k) = rest.find(PARAM_END) {
            consider(start + k, PARAM_END.len());
        }
        if let Some(k) = rest.find(PARAM_PREFIX) {
            consider(start + k, 0);
        }
        if let Some(k) = rest.find(FUNC_END) {
            consider(start + k, 0);
        }
        if dollar >= start {
            consider(dollar, 0);
        } else {
            consider(s.len(), 0);
        }
        let (end, next) = best.unwrap();
        out.push(&s[start..end]);
        pos = next;
    }
    out
}

fn strip_one_newline(v: &str) -> &str {
    let v = v.strip_prefix('\n').unwrap_or(v);
    v.strip_suffix('\n').unwrap_or(v)
}

/// One-shot parse (`detect_and_parse`). On an internal error the reference returns the whole text as normal
/// text with no calls; so do we.
pub fn detect_and_parse(text: &str, tools: &ToolSet) -> ToolParseResult {
    if !text.contains(CALL_START) {
        return ToolParseResult { normal_text: text.to_string(), calls: vec![] };
    }
    match detect_and_parse_inner(text, tools) {
        Ok(r) => r,
        Err(_) => ToolParseResult { normal_text: text.to_string(), calls: vec![] },
    }
}

fn detect_and_parse_inner(text: &str, tools: &ToolSet) -> Result<ToolParseResult, ParseError> {
    let mut raw_calls: Vec<&str> = Vec::new();
    let mut pos = 0;
    while let Some(i) = text[pos..].find(CALL_START) {
        let start = pos + i + CALL_START.len();
        match text[start..].find(CALL_END) {
            Some(j) => {
                raw_calls.push(&text[start..start + j]);
                pos = start + j + CALL_END.len();
            }
            None => break,
        }
    }
    if raw_calls.is_empty() && text.contains(FUNC_PREFIX) {
        raw_calls.push(text);
    }
    let mut calls = Vec::new();
    let mut idx = 0;
    for content in raw_calls {
        for body in find_functions(content) {
            let Some(name_end) = body.find('>') else { continue };
            let name = &body[..name_end];
            let params_str = &body[name_end + 1..];
            let config = tools.arguments_config(name);
            let mut params: Vec<(String, PyValue)> = Vec::new();
            for p in find_parameters(params_str) {
                let Some(pi) = p.find('>') else { continue };
                let pname = &p[..pi];
                let pval = strip_one_newline(&p[pi + 1..]);
                let v = convert_param_value(pval, pname, config)?;
                pyjson::dict_set(&mut params, pname.to_string(), v);
            }
            calls.push(ToolCallItem {
                tool_index: idx,
                name: Some(name.to_string()),
                parameters: PyValue::Dict(params).dumps(false),
            });
            idx += 1;
        }
    }
    let start = text.find(CALL_START).or_else(|| text.find(FUNC_PREFIX));
    let normal = match start {
        Some(i) if i > 0 => text[..i].to_string(),
        _ => String::new(),
    };
    Ok(ToolParseResult { normal_text: normal, calls })
}

// ------------------------------------------------------------------------------------------------ stream

/// Cursor-based streaming parser (`parse_streaming_increment`).
#[derive(Debug, Clone)]
pub struct ToolCallStreamParser {
    buffer: String,
    current_tool_id: i64,
    current_tool_param_count: usize,
    json_started: bool,
    inside_call: bool,
    current_func_name: Option<String>,
}

impl Default for ToolCallStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallStreamParser {
    pub fn new() -> Self {
        ToolCallStreamParser {
            buffer: String::new(),
            current_tool_id: -1,
            current_tool_param_count: 0,
            json_started: false,
            inside_call: false,
            current_func_name: None,
        }
    }

    pub fn parse(&mut self, new_text: &str, tools: &ToolSet) -> Result<ToolParseResult, ParseError> {
        self.buffer.push_str(new_text);
        if self.buffer.is_empty() {
            return Ok(ToolParseResult::default());
        }
        let mut calls = Vec::new();
        let mut normal = String::new();
        let mut parsed = 0usize;
        loop {
            let cur = &self.buffer[parsed..];
            if cur.is_empty() {
                break;
            }
            if cur.starts_with(CALL_START) {
                parsed += CALL_START.len();
                self.inside_call = true;
                continue;
            }
            if cur.starts_with(FUNC_PREFIX) {
                match cur.find('>') {
                    Some(end) => {
                        let name = cur[FUNC_PREFIX.len()..end].to_string();
                        self.current_tool_id += 1;
                        self.current_tool_param_count = 0;
                        self.json_started = false;
                        self.current_func_name = Some(name.clone());
                        calls.push(ToolCallItem {
                            tool_index: self.current_tool_id,
                            name: Some(name),
                            parameters: String::new(),
                        });
                        parsed += end + 1;
                        continue;
                    }
                    None => break,
                }
            }
            if cur.starts_with(PARAM_PREFIX) {
                if let Some(name_end) = cur.find('>') {
                    let rest = &cur[name_end + 1..];
                    let mut best: Option<(usize, usize)> = None;
                    for (k, len) in [
                        (rest.find(PARAM_END), PARAM_END.len()),
                        (rest.find(PARAM_PREFIX), 0),
                        (rest.find(FUNC_END), 0),
                    ] {
                        if let Some(k) = k {
                            if best.is_none_or(|(b, _)| k < b) {
                                best = Some((k, len));
                            }
                        }
                    }
                    if let Some((end_pos, end_len)) = best {
                        let pname = cur[PARAM_PREFIX.len()..name_end].to_string();
                        let raw = strip_one_newline(&rest[..end_pos]).to_string();
                        if !self.json_started {
                            calls.push(ToolCallItem {
                                tool_index: self.current_tool_id,
                                name: None,
                                parameters: "{".into(),
                            });
                            self.json_started = true;
                        }
                        let fname = self.current_func_name.clone().unwrap_or_default();
                        let config =
                            if self.current_func_name.is_some() { tools.arguments_config(&fname) } else { None };
                        let v = convert_param_value(&raw, &pname, config)?;
                        let kv = format!("{}: {}", pyjson::quote(&pname, true), v.dumps(false));
                        let frag = if self.current_tool_param_count > 0 { format!(", {kv}") } else { kv };
                        calls.push(ToolCallItem { tool_index: self.current_tool_id, name: None, parameters: frag });
                        self.current_tool_param_count += 1;
                        parsed += name_end + 1 + end_pos + end_len;
                        continue;
                    }
                }
                break;
            }
            if cur.starts_with(FUNC_END) {
                if !self.json_started {
                    calls.push(ToolCallItem { tool_index: self.current_tool_id, name: None, parameters: "{".into() });
                    self.json_started = true;
                }
                calls.push(ToolCallItem { tool_index: self.current_tool_id, name: None, parameters: "}".into() });
                parsed += FUNC_END.len();
                self.current_func_name = None;
                continue;
            }
            if cur.starts_with(CALL_END) {
                parsed += CALL_END.len();
                self.inside_call = false;
                continue;
            }
            match cur.find('<') {
                None => {
                    if !self.inside_call {
                        normal.push_str(cur);
                    }
                    parsed += cur.len();
                }
                Some(0) => {
                    let potential = [CALL_START, CALL_END, FUNC_PREFIX, FUNC_END, PARAM_PREFIX, PARAM_END]
                        .iter()
                        .any(|t| t.starts_with(cur));
                    if potential {
                        break;
                    }
                    if !self.inside_call {
                        normal.push('<');
                    }
                    parsed += 1;
                }
                Some(i) => {
                    if !self.inside_call {
                        normal.push_str(&cur[..i]);
                    }
                    parsed += i;
                }
            }
        }
        if parsed > 0 {
            self.buffer.drain(..parsed);
        }
        Ok(ToolParseResult { normal_text: normal, calls })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyjson::loads;

    fn tools() -> ToolSet {
        let t = loads(
            r#"[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{
                "i":{"type":"integer"},"n":{"type":"number"},"b":{"type":"boolean"},"o":{"type":"object"}}}}}]"#,
            Dialect::Python,
        )
        .unwrap();
        ToolSet::from_openai(t.as_list().unwrap())
    }

    #[test]
    fn non_stream_basic() {
        let text = "pre\n<|call|>\n<function=f>\n<parameter=i>\n5\n</parameter>\n<parameter=n>\n2\n</parameter>\n\
                    <parameter=b>\nTrue\n</parameter>\n<parameter=o>\n{\"a\": [1, 2]}\n</parameter>\n<parameter=s>\nx\n</parameter>\n</function>\n<|/call|>";
        let r = detect_and_parse(text, &tools());
        assert_eq!(r.normal_text, "pre\n");
        assert_eq!(r.calls[0].parameters, r#"{"i": 5, "n": 2, "b": true, "o": {"a": [1, 2]}, "s": "x"}"#);
    }

    #[test]
    fn stream_fragments() {
        let mut p = ToolCallStreamParser::new();
        let t = tools();
        let mut frags = Vec::new();
        for ch in "<|call|>\n<function=f>\n<parameter=i>\n7\n</parameter>\n<parameter=é>\nx\n</parameter>\n</function>\n<|/call|>".chars() {
            let r = p.parse(&ch.to_string(), &t).unwrap();
            frags.extend(r.calls.into_iter().map(|c| c.parameters));
        }
        assert_eq!(frags, ["", "{", "\"i\": 7", ", \"\\u00e9\": \"x\"", "}"]);
    }
}
