//! Minimal safetensors reader: header parse + mmap. No copies; tensors are byte slices into the mapping.

use anyhow::{bail, ensure, Context, Result};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    BF16,
    F16,
    F32,
    I32,
    I64,
    U8,
    I8,
    F8E4M3,
}

impl Dtype {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "BF16" => Dtype::BF16,
            "F16" => Dtype::F16,
            "F32" => Dtype::F32,
            "I32" => Dtype::I32,
            "I64" => Dtype::I64,
            "U8" => Dtype::U8,
            "I8" => Dtype::I8,
            "F8_E4M3" => Dtype::F8E4M3,
            other => bail!("unsupported safetensors dtype {other}"),
        })
    }
    pub fn size(self) -> usize {
        match self {
            Dtype::BF16 | Dtype::F16 => 2,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::I64 => 8,
            Dtype::U8 | Dtype::I8 | Dtype::F8E4M3 => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Absolute byte range in the file (header offsets + 8 + header length).
    pub start: usize,
    pub end: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn nbytes(&self) -> usize {
        self.end - self.start
    }
}

pub struct SafeTensorsFile {
    pub path: PathBuf,
    pub tensors: BTreeMap<String, TensorInfo>,
    mmap: Mmap,
}

impl SafeTensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: read-only mapping of a file we do not modify; weight files are immutable releases.
        let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
        let tensors = parse_header(&mmap).with_context(|| format!("header of {}", path.display()))?;
        Ok(SafeTensorsFile { path, tensors, mmap })
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.tensors.get(name).with_context(|| format!("tensor {name} not in {}", self.path.display()))?;
        Ok(&self.mmap[t.start..t.end])
    }

    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Ask the kernel to prefetch the whole file (sequential upload path).
    pub fn advise_sequential(&self) {
        #[cfg(unix)]
        {
            let _ = self.mmap.advise(memmap2::Advice::Sequential);
        }
    }
}

pub fn parse_header(buf: &[u8]) -> Result<BTreeMap<String, TensorInfo>> {
    ensure!(buf.len() >= 8, "file shorter than the 8-byte header length");
    let n = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
    ensure!(n <= buf.len() - 8, "header length {n} exceeds file size {}", buf.len());
    let header: Value = serde_json::from_slice(&buf[8..8 + n]).context("header JSON")?;
    let obj = header.as_object().context("header is not an object")?;
    let base = 8 + n;
    let mut out = BTreeMap::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = Dtype::parse(v["dtype"].as_str().context("dtype")?)?;
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .context("shape")?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize).context("shape dim"))
            .collect::<Result<_>>()?;
        let offs = v["data_offsets"].as_array().context("data_offsets")?;
        ensure!(offs.len() == 2, "data_offsets must have 2 entries");
        let (s, e) = (offs[0].as_u64().context("offset")? as usize, offs[1].as_u64().context("offset")? as usize);
        let numel: usize = shape.iter().product();
        ensure!(e >= s && e - s == numel * dtype.size(), "{name}: byte range {s}..{e} != {numel} x {}", dtype.size());
        ensure!(base + e <= buf.len(), "{name}: data past end of file");
        out.insert(name.clone(), TensorInfo { dtype, shape, start: base + s, end: base + e });
    }
    Ok(out)
}

/// Serialise tensors into safetensors bytes (used by tests and tooling).
pub fn serialize(tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
    let dt = |d: Dtype| match d {
        Dtype::BF16 => "BF16",
        Dtype::F16 => "F16",
        Dtype::F32 => "F32",
        Dtype::I32 => "I32",
        Dtype::I64 => "I64",
        Dtype::U8 => "U8",
        Dtype::I8 => "I8",
        Dtype::F8E4M3 => "F8_E4M3",
    };
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (name, d, shape, data) in tensors {
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": dt(*d), "shape": shape, "data_offsets": [off, off + data.len()]}),
        );
        off += data.len();
    }
    let mut h = serde_json::to_vec(&Value::Object(header)).unwrap();
    while (8 + h.len()) % 8 != 0 {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    for (_, _, _, data) in tensors {
        out.extend_from_slice(data);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let a: Vec<u8> = (0..24u8).collect();
        let bytes = serialize(&[("a", Dtype::BF16, vec![3, 4], a.clone()), ("b", Dtype::F32, vec![2], vec![1; 8])]);
        let t = parse_header(&bytes).unwrap();
        assert_eq!(t["a"].shape, vec![3, 4]);
        assert_eq!(&bytes[t["a"].start..t["a"].end], &a[..]);
        assert_eq!(t["b"].dtype, Dtype::F32);
    }

    #[test]
    fn rejects_bad_size() {
        let mut bytes = serialize(&[("a", Dtype::BF16, vec![3, 4], vec![0; 24])]);
        bytes.truncate(bytes.len() - 2);
        assert!(parse_header(&bytes).is_err());
    }
}
