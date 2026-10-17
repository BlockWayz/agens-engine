//! Python-compatible JSON values: `json.loads` / `orjson.loads` parsing, `json.dumps` formatting (key order,
//! separators, `ensure_ascii`, float repr), and the small slice of `ast.literal_eval` the tool parser needs.
//!
//! The reference server is Python: request bodies are parsed with orjson, chat templates serialise with
//! `json.dumps(ensure_ascii=False)`, and the tool-call parser re-serialises converted arguments with
//! `json.dumps`. Byte parity needs the same int/float distinction, insertion order, duplicate-key rule
//! (last value wins, first position kept) and float formatting, which `serde_json` does not give us.

use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq)]
pub enum PyValue {
    Null,
    Bool(bool),
    /// Canonical decimal digits (Python ints are unbounded): optional '-', no leading zeros.
    Int(String),
    Float(f64),
    Str(String),
    List(Vec<PyValue>),
    /// Insertion-ordered dict.
    Dict(Vec<(String, PyValue)>),
}

impl PyValue {
    pub fn int(v: i64) -> Self {
        PyValue::Int(v.to_string())
    }
    pub fn str(s: impl Into<String>) -> Self {
        PyValue::Str(s.into())
    }
    pub fn get(&self, key: &str) -> Option<&PyValue> {
        match self {
            PyValue::Dict(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn get_mut(&mut self, key: &str) -> Option<&mut PyValue> {
        match self {
            PyValue::Dict(kv) => kv.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    /// Python `d[key] = value`: replace in place if present, else append.
    pub fn set(&mut self, key: &str, value: PyValue) {
        if let PyValue::Dict(kv) = self {
            dict_set(kv, key.to_string(), value);
        }
    }
    pub fn remove(&mut self, key: &str) -> Option<PyValue> {
        if let PyValue::Dict(kv) = self {
            if let Some(i) = kv.iter().position(|(k, _)| k == key) {
                return Some(kv.remove(i).1);
            }
        }
        None
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PyValue::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_list(&self) -> Option<&Vec<PyValue>> {
        match self {
            PyValue::List(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_dict(&self) -> Option<&Vec<(String, PyValue)>> {
        match self {
            PyValue::Dict(v) => Some(v),
            _ => None,
        }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, PyValue::Null)
    }
    /// Python truthiness.
    pub fn truthy(&self) -> bool {
        match self {
            PyValue::Null => false,
            PyValue::Bool(b) => *b,
            PyValue::Int(s) => s != "0",
            PyValue::Float(f) => *f != 0.0,
            PyValue::Str(s) => !s.is_empty(),
            PyValue::List(v) => !v.is_empty(),
            PyValue::Dict(v) => !v.is_empty(),
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            PyValue::Int(s) => s.parse().ok(),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            PyValue::Int(s) => s.parse().ok(),
            PyValue::Float(f) => Some(*f),
            _ => None,
        }
    }
    /// Python `json.dumps(v, ensure_ascii=..)` with the default separators (", ", ": ").
    pub fn dumps(&self, ensure_ascii: bool) -> String {
        let mut s = String::new();
        dump_into(&mut s, self, ensure_ascii, ", ", ": ");
        s
    }
    /// Compact form (separators "," and ":"), as orjson / starlette write responses.
    pub fn dumps_compact(&self) -> String {
        let mut s = String::new();
        dump_into(&mut s, self, false, ",", ":");
        s
    }
}

pub fn dict_set(kv: &mut Vec<(String, PyValue)>, key: String, value: PyValue) {
    if let Some(slot) = kv.iter_mut().find(|(k, _)| *k == key) {
        slot.1 = value;
    } else {
        kv.push((key, value));
    }
}

// ------------------------------------------------------------------------------------------------ dumps

pub fn dump_into(out: &mut String, v: &PyValue, ensure_ascii: bool, item_sep: &str, key_sep: &str) {
    match v {
        PyValue::Null => out.push_str("null"),
        PyValue::Bool(true) => out.push_str("true"),
        PyValue::Bool(false) => out.push_str("false"),
        PyValue::Int(s) => out.push_str(s),
        PyValue::Float(f) => out.push_str(&float_json(*f)),
        PyValue::Str(s) => quote_into(out, s, ensure_ascii),
        PyValue::List(items) => {
            out.push('[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                dump_into(out, it, ensure_ascii, item_sep, key_sep);
            }
            out.push(']');
        }
        PyValue::Dict(kv) => {
            out.push('{');
            for (i, (k, it)) in kv.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                quote_into(out, k, ensure_ascii);
                out.push_str(key_sep);
                dump_into(out, it, ensure_ascii, item_sep, key_sep);
            }
            out.push('}');
        }
    }
}

/// JSON string literal as Python's encoder writes it (same escape set as serde_json / orjson / msgspec).
pub fn quote_into(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{:04x}", u);
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn quote(s: &str, ensure_ascii: bool) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    quote_into(&mut o, s, ensure_ascii);
    o
}

/// `float.__repr__` as `json.dumps` writes it (NaN / Infinity spelled the JSON-extension way).
pub fn float_json(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    float_repr(f)
}

/// Python `repr(float)` for finite values: shortest round-trip digits, fixed notation for decimal
/// exponents in [-4, 16), otherwise `d.ddde+XX`.
pub fn float_repr(f: f64) -> String {
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let e = format!("{:e}", f); // shortest round-trip digits, e.g. "-1.25e-7"
    let (mant, exp) = e.split_once('e').expect("LowerExp has an exponent");
    let exp: i32 = exp.parse().expect("exponent");
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                out.push_str(&digits);
                out.extend(std::iter::repeat_n('0', int_len - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        } else {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-exp - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    out
}

/// Exact decimal digits of an integral f64 (Python `int(x)` for `x.is_integer()`).
pub fn float_to_int_string(f: f64) -> String {
    if f.abs() < 1.7e38 {
        return (f as i128).to_string();
    }
    // mantissa * 2^exp with big decimal doubling.
    let bits = f.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32 - 1075;
    let mant = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
    let mut digits: Vec<u8> = mant.to_string().bytes().map(|b| b - b'0').rev().collect();
    for _ in 0..exp {
        let mut carry = 0u8;
        for d in digits.iter_mut() {
            let v = *d * 2 + carry;
            *d = v % 10;
            carry = v / 10;
        }
        if carry > 0 {
            digits.push(carry);
        }
    }
    let mut s: String = digits.iter().rev().map(|d| (b'0' + d) as char).collect();
    if f < 0.0 {
        s.insert(0, '-');
    }
    s
}

// ------------------------------------------------------------------------------------------------ loads

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// Python `json.loads`: NaN / Infinity accepted, unbounded ints.
    Python,
    /// `orjson.loads` (request bodies): no NaN / Infinity, ints limited to the 64-bit range.
    Orjson,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError(pub String);

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for JsonError {}

pub fn loads(s: &str, dialect: Dialect) -> Result<PyValue, JsonError> {
    let mut p = Parser { b: s.as_bytes(), s, i: 0, dialect, depth: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(JsonError(format!("Extra data at {}", p.i)));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    s: &'a str,
    i: usize,
    dialect: Dialect,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn err<T>(&self, m: &str) -> Result<T, JsonError> {
        Err(JsonError(format!("{m} at {}", self.i)))
    }
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn lit(&mut self, w: &str) -> bool {
        if self.s[self.i..].starts_with(w) {
            self.i += w.len();
            true
        } else {
            false
        }
    }
    fn value(&mut self) -> Result<PyValue, JsonError> {
        if self.i >= self.b.len() {
            return self.err("Expecting value");
        }
        self.depth += 1;
        if self.depth > 1000 {
            return self.err("nesting too deep");
        }
        let py = self.dialect == Dialect::Python;
        let c = self.b[self.i];
        let r = if c == b'{' {
            self.object()
        } else if c == b'[' {
            self.array()
        } else if c == b'"' {
            self.string().map(PyValue::Str)
        } else if self.lit("null") {
            Ok(PyValue::Null)
        } else if self.lit("true") {
            Ok(PyValue::Bool(true))
        } else if self.lit("false") {
            Ok(PyValue::Bool(false))
        } else if py && self.lit("NaN") {
            Ok(PyValue::Float(f64::NAN))
        } else if py && self.lit("Infinity") {
            Ok(PyValue::Float(f64::INFINITY))
        } else if py && self.lit("-Infinity") {
            Ok(PyValue::Float(f64::NEG_INFINITY))
        } else if c == b'-' || c.is_ascii_digit() {
            self.number()
        } else {
            self.err("Expecting value")
        };
        self.depth -= 1;
        r
    }
    fn object(&mut self) -> Result<PyValue, JsonError> {
        self.i += 1;
        let mut kv: Vec<(String, PyValue)> = Vec::new();
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b'}' {
            self.i += 1;
            return Ok(PyValue::Dict(kv));
        }
        loop {
            self.ws();
            if self.i >= self.b.len() || self.b[self.i] != b'"' {
                return self.err("Expecting property name enclosed in double quotes");
            }
            let k = self.string()?;
            self.ws();
            if self.i >= self.b.len() || self.b[self.i] != b':' {
                return self.err("Expecting ':' delimiter");
            }
            self.i += 1;
            self.ws();
            let v = self.value()?;
            dict_set(&mut kv, k, v);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(PyValue::Dict(kv));
                }
                _ => return self.err("Expecting ',' delimiter"),
            }
        }
    }
    fn array(&mut self) -> Result<PyValue, JsonError> {
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b']' {
            self.i += 1;
            return Ok(PyValue::List(items));
        }
        loop {
            self.ws();
            items.push(self.value()?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(PyValue::List(items));
                }
                _ => return self.err("Expecting ',' delimiter"),
            }
        }
    }
    fn hex4(&mut self) -> Result<u32, JsonError> {
        if self.i + 4 > self.b.len() {
            return self.err("Invalid \\uXXXX escape");
        }
        let h = &self.s[self.i..self.i + 4];
        let v = u32::from_str_radix(h, 16).or_else(|_| self.err("Invalid \\uXXXX escape"))?;
        if !h.bytes().all(|c| c.is_ascii_hexdigit()) {
            return self.err("Invalid \\uXXXX escape");
        }
        self.i += 4;
        Ok(v)
    }
    fn string(&mut self) -> Result<String, JsonError> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' && self.b[self.i] >= 0x20 {
                self.i += 1;
            }
            out.push_str(&self.s[start..self.i]);
            match self.b.get(self.i) {
                None => return self.err("Unterminated string"),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let c = match self.b.get(self.i) {
                        Some(c) => *c,
                        None => return self.err("Unterminated string"),
                    };
                    self.i += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let u = self.hex4()?;
                            if (0xd800..0xdc00).contains(&u) && self.s[self.i..].starts_with("\\u") {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xdc00..0xe000).contains(&lo) {
                                    let cp = 0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00);
                                    out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                                    continue;
                                }
                                self.i = save;
                            }
                            // Lone surrogates have no UTF-8 form; Python keeps them, we substitute U+FFFD.
                            out.push(char::from_u32(u).unwrap_or('\u{fffd}'));
                        }
                        _ => return self.err("Invalid \\escape"),
                    }
                }
                Some(_) => return self.err("Invalid control character"),
            }
        }
    }
    fn number(&mut self) -> Result<PyValue, JsonError> {
        let start = self.i;
        if self.b[self.i] == b'-' {
            self.i += 1;
        }
        let digits_start = self.i;
        match self.b.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
                    self.i += 1;
                }
            }
            _ => return self.err("Expecting value"),
        }
        let mut is_float = false;
        if self.b.get(self.i) == Some(&b'.') && self.b.get(self.i + 1).is_some_and(|c| c.is_ascii_digit()) {
            is_float = true;
            self.i += 1;
            while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
                self.i += 1;
            }
        }
        if matches!(self.b.get(self.i), Some(b'e' | b'E')) {
            let save = self.i;
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if self.b.get(self.i).is_some_and(|c| c.is_ascii_digit()) {
                is_float = true;
                while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
                    self.i += 1;
                }
            } else {
                self.i = save;
            }
        }
        let text = &self.s[start..self.i];
        if is_float {
            let f: f64 = text.parse().or_else(|_| self.err("bad number"))?;
            if self.dialect == Dialect::Orjson && f.is_infinite() {
                return self.err("number out of range");
            }
            return Ok(PyValue::Float(f));
        }
        let _ = digits_start;
        let canon = canonical_int(text);
        if self.dialect == Dialect::Orjson && canon.parse::<i64>().is_err() && canon.parse::<u64>().is_err() {
            return self.err("Integer exceeds 64-bit range");
        }
        Ok(PyValue::Int(canon))
    }
}

/// Canonical Python int text from an optionally signed run of ASCII digits.
pub fn canonical_int(text: &str) -> String {
    let (neg, digits) = match text.strip_prefix('-') {
        Some(d) => (true, d),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let d = digits.trim_start_matches('0');
    if d.is_empty() {
        return "0".into();
    }
    if neg {
        format!("-{d}")
    } else {
        d.to_string()
    }
}

// ------------------------------------------------------------------------------------------------ str helpers

/// Python `str.isspace()` for one char.
pub fn py_isspace(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{0b}' | '\u{0c}' | '\r' | '\u{1c}'..='\u{1f}' | ' ' | '\u{85}' | '\u{a0}' | '\u{1680}'
            | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}'
    )
}

/// Python `str.strip()`.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

/// Python `int(s)` for a str: whitespace, sign, PEP 515 underscores. Returns canonical digits.
pub fn py_int(s: &str) -> Option<String> {
    let t = py_strip(s);
    let (sign, body) = match t.as_bytes().first() {
        Some(b'-') => ("-", &t[1..]),
        Some(b'+') => ("", &t[1..]),
        _ => ("", t),
    };
    let digits = py_digits(body)?;
    Some(canonical_int(&format!("{sign}{digits}")))
}

/// Decimal digits with single underscores between digits (Unicode decimal digits mapped to ASCII).
fn py_digits(body: &str) -> Option<String> {
    let mut out = String::new();
    let mut prev_us = true; // leading underscore not allowed
    for c in body.chars() {
        if c == '_' {
            if prev_us {
                return None;
            }
            prev_us = true;
            continue;
        }
        let d = unicode_decimal(c)?;
        out.push(d);
        prev_us = false;
    }
    if out.is_empty() || prev_us {
        return None;
    }
    Some(out)
}

fn unicode_decimal(c: char) -> Option<char> {
    if c.is_ascii_digit() {
        return Some(c);
    }
    // Common Unicode Nd blocks (Python int()/float() accept any Nd digit).
    for base in [0x660u32, 0x6f0, 0x966, 0x9e6, 0xe50, 0xff10] {
        let v = c as u32;
        if (base..base + 10).contains(&v) {
            return Some((b'0' + (v - base) as u8) as char);
        }
    }
    None
}

/// Python `float(s)` for a str.
pub fn py_float(s: &str) -> Option<f64> {
    let t = py_strip(s);
    let (neg, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let lower = body.to_ascii_lowercase();
    let v = match lower.as_str() {
        "inf" | "infinity" => Some(f64::INFINITY),
        "nan" => Some(f64::NAN),
        _ => {
            // mantissa [. frac] [e[+-]exp], digit groups may contain single underscores
            let (mant, exp) = match lower.find('e') {
                Some(i) => (&lower[..i], Some(&lower[i + 1..])),
                None => (lower.as_str(), None),
            };
            let (ip, fp) = match mant.split_once('.') {
                Some((a, b)) => (a, Some(b)),
                None => (mant, None),
            };
            let ip_d = if ip.is_empty() { Some(String::new()) } else { py_digits(ip) };
            let fp_d = match fp {
                Some("") => Some(String::new()),
                Some(f) => py_digits(f),
                None => Some(String::new()),
            };
            let (ip_d, fp_d) = (ip_d?, fp_d?);
            if ip_d.is_empty() && fp_d.is_empty() {
                return None;
            }
            let mut norm = format!("{ip_d}.{fp_d}");
            if norm.starts_with('.') {
                norm.insert(0, '0');
            }
            if norm.ends_with('.') {
                norm.push('0');
            }
            if let Some(e) = exp {
                let (es, ed) = match e.as_bytes().first() {
                    Some(b'-') => ("-", &e[1..]),
                    Some(b'+') => ("", &e[1..]),
                    _ => ("", e),
                };
                let ed = py_digits(ed)?;
                norm = format!("{norm}e{es}{ed}");
            }
            norm.parse::<f64>().ok()
        }
    };
    v.map(|x| if neg { -x } else { x })
}

// ------------------------------------------------------------------------------------------------ literal_eval

/// The subset of `ast.literal_eval` that yields JSON-serialisable values: numbers, strings, True/False/None,
/// lists, tuples (as lists) and dicts with string keys. Anything else (sets, bytes, complex, non-string keys,
/// which `json.dumps` would reject anyway) is an error.
pub fn literal_eval(s: &str) -> Option<PyValue> {
    let src = s.trim_start_matches([' ', '\t']);
    let mut p = Lit { c: src.chars().collect(), i: 0, depth: 0 };
    p.ws(true);
    let v = p.expr()?;
    p.ws(true);
    if p.i != p.c.len() {
        return None;
    }
    Some(v)
}

struct Lit {
    c: Vec<char>,
    i: usize,
    depth: usize,
}

impl Lit {
    fn peek(&self) -> Option<char> {
        self.c.get(self.i).copied()
    }
    /// Skip spaces; newlines and comments only inside brackets (or at the very end, `nl`).
    fn ws(&mut self, nl: bool) {
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' || c == '\u{0c}' || (nl && (c == '\n' || c == '\r')) {
                self.i += 1;
            } else if c == '\\' && matches!(self.c.get(self.i + 1), Some('\n')) {
                self.i += 2;
            } else if nl && c == '#' {
                while self.peek().is_some_and(|c| c != '\n') {
                    self.i += 1;
                }
            } else {
                break;
            }
        }
    }
    fn expr(&mut self) -> Option<PyValue> {
        self.depth += 1;
        if self.depth > 200 {
            return None;
        }
        let nl = self.depth > 1;
        self.ws(nl);
        let r = match self.peek()? {
            '[' => self.seq(']'),
            '(' => self.seq(')'),
            '{' => self.dict(),
            '-' | '+' => {
                let neg = self.peek() == Some('-');
                self.i += 1;
                self.ws(nl);
                match self.expr()? {
                    PyValue::Int(s) => Some(PyValue::Int(if neg { canonical_int(&format!("-{s}")) } else { s })),
                    PyValue::Float(f) => Some(PyValue::Float(if neg { -f } else { f })),
                    _ => None,
                }
            }
            c if c.is_ascii_digit() || c == '.' => self.number(),
            c if c == '\'' || c == '"' || c.is_ascii_alphabetic() || c == '_' => self.name_or_string(),
            _ => None,
        };
        self.depth -= 1;
        r
    }
    fn seq(&mut self, close: char) -> Option<PyValue> {
        self.i += 1;
        let mut items = Vec::new();
        let mut saw_comma = false;
        loop {
            self.ws(true);
            if self.peek()? == close {
                self.i += 1;
                break;
            }
            items.push(self.expr()?);
            self.ws(true);
            match self.peek()? {
                ',' => {
                    self.i += 1;
                    saw_comma = true;
                }
                c if c == close => {
                    self.i += 1;
                    break;
                }
                _ => return None,
            }
        }
        if close == ')' && items.len() == 1 && !saw_comma {
            return items.pop(); // parenthesised expression
        }
        Some(PyValue::List(items))
    }
    fn dict(&mut self) -> Option<PyValue> {
        self.i += 1;
        let mut kv = Vec::new();
        loop {
            self.ws(true);
            if self.peek()? == '}' {
                self.i += 1;
                break;
            }
            let k = match self.expr()? {
                PyValue::Str(s) => s,
                _ => return None, // non-string keys: json.dumps would emit them differently; unsupported
            };
            self.ws(true);
            if self.peek()? != ':' {
                return None; // a set literal
            }
            self.i += 1;
            let v = self.expr()?;
            dict_set(&mut kv, k, v);
            self.ws(true);
            match self.peek()? {
                ',' => self.i += 1,
                '}' => {
                    self.i += 1;
                    break;
                }
                _ => return None,
            }
        }
        Some(PyValue::Dict(kv))
    }
    fn number(&mut self) -> Option<PyValue> {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.') {
            // exponent sign
            self.i += 1;
            if matches!(self.c.get(self.i - 1), Some('e' | 'E'))
                && matches!(self.peek(), Some('+' | '-'))
                && !self.c[start..self.i].iter().any(|c| matches!(c, 'x' | 'X'))
            {
                self.i += 1;
            }
        }
        let t: String = self.c[start..self.i].iter().collect();
        let lower = t.to_ascii_lowercase();
        if lower.ends_with('j') {
            return None; // complex
        }
        for (prefix, radix) in [("0x", 16), ("0o", 8), ("0b", 2)] {
            if let Some(d) = lower.strip_prefix(prefix) {
                let d = d.strip_prefix('_').unwrap_or(d);
                if d.is_empty() || d.starts_with('_') || d.ends_with('_') || d.contains("__") {
                    return None;
                }
                let d: String = d.chars().filter(|c| *c != '_').collect();
                let v = u128::from_str_radix(&d, radix).ok()?;
                return Some(PyValue::Int(v.to_string()));
            }
        }
        if lower.contains(['.', 'e']) {
            if lower.contains("inf") || lower.contains("nan") {
                return None;
            }
            return py_float(&t).map(PyValue::Float);
        }
        // decimal int: no leading zeros except all-zero
        let digits: String = t.chars().filter(|c| *c != '_').collect();
        if t.starts_with('_') || t.ends_with('_') || t.contains("__") || !digits.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        if digits.len() > 1 && digits.starts_with('0') && digits.chars().any(|c| c != '0') {
            return None;
        }
        Some(PyValue::Int(canonical_int(&digits)))
    }
    fn name_or_string(&mut self) -> Option<PyValue> {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_alphanumeric() || c == '_') {
            self.i += 1;
        }
        let name: String = self.c[start..self.i].iter().collect();
        if let Some(q) = self.peek().filter(|q| *q == '\'' || *q == '"') {
            let lp = name.to_ascii_lowercase();
            if !matches!(lp.as_str(), "" | "r" | "u") {
                return None; // bytes / f-strings
            }
            let raw = lp == "r";
            let mut s = self.string_lit(q, raw)?;
            // implicit concatenation of adjacent literals
            loop {
                let save = self.i;
                self.ws(self.depth > 1);
                let pstart = self.i;
                while self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                    self.i += 1;
                }
                let pre: String = self.c[pstart..self.i].iter().collect::<String>().to_ascii_lowercase();
                match self.peek() {
                    Some(q2 @ ('\'' | '"')) if matches!(pre.as_str(), "" | "r" | "u") => {
                        s.push_str(&self.string_lit(q2, pre == "r")?);
                    }
                    _ => {
                        self.i = save;
                        break;
                    }
                }
            }
            return Some(PyValue::Str(s));
        }
        match name.as_str() {
            "True" => Some(PyValue::Bool(true)),
            "False" => Some(PyValue::Bool(false)),
            "None" => Some(PyValue::Null),
            _ => None,
        }
    }
    fn string_lit(&mut self, q: char, raw: bool) -> Option<String> {
        let triple = self.c.get(self.i + 1) == Some(&q) && self.c.get(self.i + 2) == Some(&q);
        self.i += if triple { 3 } else { 1 };
        let mut out = String::new();
        loop {
            let c = self.peek()?;
            if c == q {
                if !triple {
                    self.i += 1;
                    return Some(out);
                }
                if self.c.get(self.i + 1) == Some(&q) && self.c.get(self.i + 2) == Some(&q) {
                    self.i += 3;
                    return Some(out);
                }
            }
            if c == '\n' && !triple {
                return None;
            }
            if c == '\\' {
                let n = *self.c.get(self.i + 1)?;
                if raw {
                    out.push('\\');
                    out.push(n);
                    self.i += 2;
                    continue;
                }
                self.i += 2;
                match n {
                    '\n' => {}
                    '\\' => out.push('\\'),
                    '\'' => out.push('\''),
                    '"' => out.push('"'),
                    'a' => out.push('\u{07}'),
                    'b' => out.push('\u{08}'),
                    'f' => out.push('\u{0c}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'v' => out.push('\u{0b}'),
                    'x' | 'u' | 'U' => {
                        let len = match n {
                            'x' => 2,
                            'u' => 4,
                            _ => 8,
                        };
                        let h: String = self.c.get(self.i..self.i + len)?.iter().collect();
                        let v = u32::from_str_radix(&h, 16).ok()?;
                        self.i += len;
                        out.push(char::from_u32(v).unwrap_or('\u{fffd}'));
                    }
                    '0'..='7' => {
                        let mut v = n.to_digit(8)?;
                        for _ in 0..2 {
                            match self.peek().and_then(|c| c.to_digit(8)) {
                                Some(d) => {
                                    v = v * 8 + d;
                                    self.i += 1;
                                }
                                None => break,
                            }
                        }
                        out.push(char::from_u32(v)?);
                    }
                    other => {
                        out.push('\\');
                        out.push(other);
                    }
                }
                continue;
            }
            out.push(c);
            self.i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        for (f, s) in [
            (0.1, "0.1"),
            (1.0, "1.0"),
            (1e16, "1e+16"),
            (1e-5, "1e-05"),
            (1e-4, "0.0001"),
            (123456789.125, "123456789.125"),
            (1e300, "1e+300"),
            (2.5e-7, "2.5e-07"),
            (100.0, "100.0"),
            (-1.5, "-1.5"),
            (1e15, "1000000000000000.0"),
            (1.2345678901234567e19, "1.2345678901234567e+19"),
            (5e-324, "5e-324"),
        ] {
            assert_eq!(float_repr(f), s, "{f}");
        }
    }

    #[test]
    fn loads_dumps_round_trip() {
        let v = loads(r#"{"b": 1, "a": [1.0, 2.5e-7, null, true, "éé😀"], "b": 2}"#, Dialect::Python).unwrap();
        assert_eq!(v.dumps(false), "{\"b\": 2, \"a\": [1.0, 2.5e-07, null, true, \"éé😀\"]}");
        assert_eq!(v.dumps(true), "{\"b\": 2, \"a\": [1.0, 2.5e-07, null, true, \"\\u00e9\\u00e9\\ud83d\\ude00\"]}");
        assert!(loads("NaN", Dialect::Orjson).is_err());
        assert!(loads("123456789012345678901", Dialect::Orjson).is_err());
        assert_eq!(loads("123456789012345678901", Dialect::Python).unwrap().dumps(false), "123456789012345678901");
    }

    #[test]
    fn python_number_parsing() {
        assert_eq!(py_int(" 042 ").as_deref(), Some("42"));
        assert_eq!(py_int("1_000").as_deref(), Some("1000"));
        assert_eq!(py_int("+17").as_deref(), Some("17"));
        assert_eq!(py_int("1__0"), None);
        assert_eq!(py_int("abc"), None);
        assert_eq!(py_float("1e3"), Some(1000.0));
        assert_eq!(py_float(".5"), Some(0.5));
        assert_eq!(py_float("5."), Some(5.0));
        assert!(py_float("inf").unwrap().is_infinite());
        assert_eq!(py_float("1_0.5"), Some(10.5));
        assert_eq!(py_float("0x10"), None);
        assert_eq!(float_to_int_string(1.2345678901234567e19), "12345678901234567168");
        assert_eq!(float_to_int_string(1e40), "10000000000000000303786028427003666890752");
    }

    #[test]
    fn literal_eval_subset() {
        let v = literal_eval("{'py': True, 'n': None, 't': (1, 2), \"s\": 'a' 'b', 'x': -1.5}").unwrap();
        assert_eq!(v.dumps(false), "{\"py\": true, \"n\": null, \"t\": [1, 2], \"s\": \"ab\", \"x\": -1.5}");
        assert!(literal_eval("{1, 2}").is_none());
        assert!(literal_eval("not a list").is_none());
        assert!(literal_eval("042").is_none());
        assert_eq!(literal_eval("0x1F").unwrap(), PyValue::Int("31".into()));
    }
}
