//! Image inputs on the engine: encode the images of a prompt with the vision tower and prefill the
//! prompt with their embeddings and MRoPE positions. Shared by `agens m1` (image cases) and `agens serve`.

use agens_server::vision::{self, MmImage, Span};
use anyhow::{bail, Context, Result};
use std::sync::Arc;

const HID: usize = 5120;

/// Image embeddings of a prompt in pinned host memory: image i's rows start at `first[i]`.
pub struct HostEmb {
    ptr: *mut std::ffi::c_void,
    pub rows: usize,
    pub first: Vec<usize>,
}
unsafe impl Send for HostEmb {}
impl Drop for HostEmb {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = agens_cuda::host_free(self.ptr);
        }
    }
}
impl HostEmb {
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.ptr
    }
    /// image i's embeddings as bf16 bits
    pub fn image(&self, i: usize, n: usize) -> &[u16] {
        unsafe { std::slice::from_raw_parts((self.ptr as *const u16).add(self.first[i] * HID), n * HID) }
    }
}

/// Encode `which` images (others get no rows: their spans must not be prefilled from this buffer).
pub fn encode(images: &[Arc<MmImage>], which: &[bool]) -> Result<HostEmb> {
    if !crate::m1::VISION.load(std::sync::atomic::Ordering::Relaxed) {
        bail!("the vision tower is not loaded (--vision 0, or the checkpoint has none)");
    }
    let mut first = Vec::with_capacity(images.len());
    let mut rows = 0;
    for (im, &w) in images.iter().zip(which) {
        first.push(rows);
        if w {
            rows += im.tokens();
        }
    }
    let ptr = agens_cuda::host_alloc((rows.max(1)) * HID * 2)?;
    let he = HostEmb { ptr, rows, first };
    for (i, im) in images.iter().enumerate() {
        if !which[i] {
            continue;
        }
        let out = unsafe { (ptr as *mut u16).add(he.first[i] * HID) } as *mut std::ffi::c_void;
        agens_cuda::m1::vis_encode(&im.rgb, im.jpeg.as_deref().map(|v| v.as_slice()), im.h, im.w, im.rh, im.rw, out)
            .with_context(|| format!("vision encode of image {i} ({}x{} -> {}x{})", im.w, im.h, im.rw, im.rh))?;
    }
    Ok(he)
}

/// Per token of `ids[from..to]`: its row of `emb` (-1 = text), for the spans of encoded images.
pub fn emb_rows(from: usize, to: usize, spans: &[Span], emb: &HostEmb) -> Vec<i32> {
    let mut r = vec![-1i32; to - from];
    for sp in spans {
        let (a, b) = (sp.start.max(from), (sp.start + sp.len).min(to));
        for t in a..b {
            r[t - from] = (emb.first[sp.image] + (t - sp.start)) as i32;
        }
    }
    r
}

/// Rope positions of ids[from..to] out of the full prompt's [3][n] rows.
pub fn rope_slice(rp: &[i64], n: usize, from: usize, to: usize) -> Vec<i64> {
    let mut o = Vec::with_capacity(3 * (to - from));
    for a in 0..3 {
        o.extend_from_slice(&rp[a * n + from..a * n + to]);
    }
    o
}

/// Prefill `ids` (with images) onto the active session: encode, then prefill with rope positions that continue the
/// session's (its next token's rope position = position + rope delta). -> the first generated token.
pub fn prefill_prompt(ids: &[u32], images: &[Arc<MmImage>], spans: &[Span]) -> Result<i32> {
    prefill_prompt_with(ids, images, spans, None)
}

/// As prefill_prompt; `emb_files` (gate experiments): per image a [tokens][5120] bf16 file used instead of the tower.
pub fn prefill_prompt_with(ids: &[u32], images: &[Arc<MmImage>], spans: &[Span], emb_files: Option<&[String]>)
                           -> Result<i32> {
    let n = ids.len();
    let emb = match emb_files {
        None => encode(images, &vec![true; images.len()])?,
        Some(files) => {
            let mut first = Vec::new();
            let mut rows = 0;
            for im in images {
                first.push(rows);
                rows += im.tokens();
            }
            let ptr = agens_cuda::host_alloc(rows.max(1) * HID * 2)?;
            let he = HostEmb { ptr, rows, first };
            for (i, (im, f)) in images.iter().zip(files).enumerate() {
                let b = std::fs::read(f).with_context(|| f.clone())?;
                if b.len() != im.tokens() * HID * 2 {
                    bail!("{f}: {} bytes for {} image tokens", b.len(), im.tokens());
                }
                unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), (ptr as *mut u8).add(he.first[i] * HID * 2), b.len()) };
            }
            he
        }
    };
    let off = agens_cuda::m1::seq() as i64 + agens_cuda::m1::rope_delta() as i64;
    let rp: Vec<i64> = vision::rope_positions(n, spans, images).into_iter().map(|x| x + off).collect();
    let rows = emb_rows(0, n, spans, &emb);
    let ids32: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
    agens_cuda::m1::prefill_mm(&ids32, Some(&rp), Some((emb.as_ptr(), &rows)))
}

/// Image case of `agens m1`: `ids` (one `<|image_pad|>` per image, as the template renders) + image files -> the
/// expanded ids and the decoded images.
pub fn case_images(ids: &[i32], files: &[String], base: &std::path::Path) -> Result<(Vec<u32>, Vec<Arc<MmImage>>, Vec<Span>)> {
    let cfg = vision::ImageCfg::new(65_536, 16_777_216);
    let mut images = Vec::new();
    for f in files {
        let p = if std::path::Path::new(f).is_absolute() { std::path::PathBuf::from(f) } else { base.join(f) };
        let b = std::fs::read(&p).with_context(|| format!("{}", p.display()))?;
        images.push(Arc::new(vision::decode_bytes(&b, &cfg).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?));
    }
    let raw: Vec<u32> = ids.iter().map(|&t| t as u32).collect();
    let pads = raw.iter().filter(|&&t| t == vision::IMAGE_PAD).count();
    if pads == images.len() {
        let (x, spans) = vision::expand(&raw, &images).map_err(|e| anyhow::anyhow!(e))?;
        return Ok((x, images, spans));
    }
    // already expanded: one run of image tokens per image, in order
    let mut spans = Vec::new();
    let mut t = 0;
    while t < raw.len() {
        if raw[t] == vision::IMAGE_PAD {
            let s = t;
            while t < raw.len() && raw[t] == vision::IMAGE_PAD {
                t += 1;
            }
            let k = spans.len();
            if k >= images.len() || images[k].tokens() != t - s {
                bail!("image case: placeholder runs do not match the images' token counts");
            }
            spans.push(Span { start: s, len: t - s, image: k });
        } else {
            t += 1;
        }
    }
    if spans.len() != images.len() {
        bail!("image case: {} placeholder runs for {} images", spans.len(), images.len());
    }
    Ok((raw, images, spans))
}

/// `agens m1` / `m1-spec` case: prefill its ids onto the active session, with its "images" if any. -> (first token,
/// the ids as fed: image placeholders expanded).
pub fn case_prefill(p: &serde_json::Value, ids: &[i32], base: &std::path::Path) -> Result<(i32, Vec<i32>)> {
    match p.get("images").and_then(|x| x.as_array()) {
        Some(files) => {
            let files: Vec<String> = files.iter().filter_map(|f| f.as_str().map(str::to_string)).collect();
            let (x, images, spans) = case_images(ids, &files, base)?;
            let t = prefill_prompt(&x, &images, &spans)?;
            Ok((t, x.iter().map(|&t| t as i32).collect()))
        }
        None => Ok((agens_cuda::m1::prefill(ids)?, ids.to_vec())),
    }
}
