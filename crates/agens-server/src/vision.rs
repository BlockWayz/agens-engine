//! Image inputs: `image_url` data URLs -> decoded RGB8 + the resize the reference processor would pick,
//! placeholder expansion of `<|image_pad|>`, MRoPE rope positions and the prefix keys of a prompt with images.
//!
//! The pixel work (resize, normalise, patchify) and the tower run in the backend (on the GPU, the reference's ATen ops);
//! here: decoding, sizing and the token-level bookkeeping, all on the CPU and engine-independent.

use base64::Engine as _;
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// `<|image_pad|>` / `<|video_pad|>` / `<|vision_start|>` / `<|vision_end|>` (config.json of the v1 checkpoints).
pub const IMAGE_PAD: u32 = 248056;
pub const VIDEO_PAD: u32 = 248057;
pub const VISION_START: u32 = 248053;
pub const VISION_END: u32 = 248054;
/// patch 16 x spatial merge 2: the resize factor; one image token per 32 x 32 pixels
pub const FACTOR: usize = 32;

/// Image sizing (preprocessor_config.json `size`: shortest_edge = min_pixels, longest_edge = max_pixels), with the
/// engine's own ceiling on max_pixels (`AGENS_IMAGE_MAX_PIXELS`, default 4,194,304 = 4,096 image tokens).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageCfg {
    pub min_pixels: usize,
    pub max_pixels: usize,
    /// images per request (`AGENS_IMAGE_MAX_COUNT`, default 16)
    pub max_images: usize,
}

pub const DEFAULT_MAX_PIXELS: usize = 4_194_304;

impl ImageCfg {
    pub fn from_model_dir(dir: &std::path::Path) -> Option<Self> {
        let t = std::fs::read_to_string(dir.join("preprocessor_config.json")).ok()?;
        let j: serde_json::Value = serde_json::from_str(&t).ok()?;
        let size = j.get("size")?;
        let min = size.get("shortest_edge").and_then(|v| v.as_u64()).unwrap_or(65_536) as usize;
        let max = size.get("longest_edge").and_then(|v| v.as_u64()).unwrap_or(16_777_216) as usize;
        Some(Self::new(min, max))
    }
    pub fn new(min_pixels: usize, reference_max: usize) -> Self {
        let env = |n: &str, d: usize| std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let cap = env("AGENS_IMAGE_MAX_PIXELS", DEFAULT_MAX_PIXELS);
        ImageCfg { min_pixels, max_pixels: reference_max.min(cap).max(min_pixels), max_images: env("AGENS_IMAGE_MAX_COUNT", 16) }
    }
}

/// A decoded image and the size the processor resizes it to.
#[derive(Debug, Clone)]
pub struct MmImage {
    /// RGB8, row-major, `h * w * 3` bytes
    pub rgb: Arc<Vec<u8>>,
    pub h: usize,
    pub w: usize,
    /// resized height / width (multiples of 32)
    pub rh: usize,
    pub rw: usize,
    /// SHA-256 of (h, w, rh, rw, rgb): the image's identity for prefix reuse
    pub hash: [u8; 32],
    /// the encoded bytes of a JPEG: the backend decodes those on the GPU (nvJPEG) like the reference, `rgb` is the
    /// fallback
    pub jpeg: Option<Arc<Vec<u8>>>,
}

impl MmImage {
    /// merged grid rows / columns = image tokens per column / row
    pub fn grid(&self) -> (usize, usize) {
        (self.rh / FACTOR, self.rw / FACTOR)
    }
    pub fn tokens(&self) -> usize {
        let (a, b) = self.grid();
        a * b
    }
    pub fn from_rgb(rgb: Vec<u8>, h: usize, w: usize, cfg: &ImageCfg) -> Result<Self, String> {
        if rgb.len() != h * w * 3 || h == 0 || w == 0 {
            return Err("image: empty or inconsistent RGB buffer".into());
        }
        let (rh, rw) = smart_resize(h, w, FACTOR, cfg.min_pixels, cfg.max_pixels)?;
        let mut sh = Sha256::new();
        for v in [h, w, rh, rw] {
            sh.update((v as u64).to_le_bytes());
        }
        sh.update(&rgb);
        Ok(MmImage { rgb: Arc::new(rgb), h, w, rh, rw, hash: sh.finalize().into(), jpeg: None })
    }
}

/// transformers' `qwen2_vl.smart_resize` (the torchvision-backend processor of the v1 checkpoints), float64 as Python.
pub fn smart_resize(height: usize, width: usize, factor: usize, min_pixels: usize, max_pixels: usize)
                    -> Result<(usize, usize), String> {
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err(format!("absolute aspect ratio must be smaller than 200, got {}", h.max(w) / h.min(w)));
    }
    // Python's round() is half-to-even
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    if h_bar * w_bar > max_pixels as f64 {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// Decode an `image_url` url. v1 takes base64 `data:` URLs only.
pub fn load_url(url: &str, cfg: &ImageCfg) -> Result<MmImage, String> {
    let rest = url.strip_prefix("data:").ok_or_else(|| {
        "image_url: only base64 data URLs (data:image/...;base64,...) are supported".to_string()
    })?;
    let (meta, data) = rest.split_once(',').ok_or("image_url: malformed data URL")?;
    if !meta.ends_with(";base64") {
        return Err("image_url: data URL must be base64-encoded".into());
    }
    let clean: String = data.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(clean.as_bytes())
        .map_err(|e| format!("image_url: invalid base64 ({e})"))?;
    decode_bytes(&bytes, cfg)
}

/// Encoded image bytes (PNG, JPEG, WebP, GIF first frame, BMP) -> RGB8 (alpha dropped, as PIL's convert("RGB")).
pub fn decode_bytes(bytes: &[u8], cfg: &ImageCfg) -> Result<MmImage, String> {
    let img = image::load_from_memory(bytes).map_err(|e| format!("image: cannot decode ({e})"))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut m = MmImage::from_rgb(rgb.into_raw(), h, w, cfg)?;
    if bytes.starts_with(&[0xFF, 0xD8]) {
        m.jpeg = Some(Arc::new(bytes.to_vec()));
    }
    Ok(m)
}

/// One image's placeholder run in an expanded prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub len: usize,
    pub image: usize,
}

/// Images of a request in prompt order, their placeholder runs, and per prompt position the rope positions.
#[derive(Debug, Clone)]
pub struct MmInput {
    pub images: Vec<Arc<MmImage>>,
    pub spans: Vec<Span>,
}

/// Replace the i-th `<|image_pad|>` with images[i].tokens() copies. Errors when the counts differ or a video pad is
/// present (video input is not supported).
pub fn expand(ids: &[u32], images: &[Arc<MmImage>]) -> Result<(Vec<u32>, Vec<Span>), String> {
    let n_pad = ids.iter().filter(|&&t| t == IMAGE_PAD).count();
    if ids.contains(&VIDEO_PAD) {
        return Err("video input is not supported".into());
    }
    if n_pad != images.len() {
        return Err(format!("the prompt has {n_pad} image placeholders for {} images", images.len()));
    }
    let mut out = Vec::with_capacity(ids.len() + images.iter().map(|i| i.tokens()).sum::<usize>());
    let mut spans = Vec::new();
    let mut k = 0;
    for &t in ids {
        if t == IMAGE_PAD {
            let n = images[k].tokens();
            spans.push(Span { start: out.len(), len: n, image: k });
            out.extend(std::iter::repeat(IMAGE_PAD).take(n));
            k += 1;
        } else {
            out.push(t);
        }
    }
    Ok((out, spans))
}

/// `get_rope_index` (transformers / sglang `MRotaryEmbedding`): rope positions [3][n] (rows T, H, W) of an
/// expanded prompt. Text: t = h = w = running position; an image span at running position s with merged grid H x W:
/// token (r, c) -> (s, s + r, s + c), then the running position advances by max(H, W).
pub fn rope_positions(n: usize, spans: &[Span], images: &[Arc<MmImage>]) -> Vec<i64> {
    let mut rp = vec![0i64; 3 * n];
    let (mut cur, mut i) = (0i64, 0usize);
    let mut si = spans.iter().peekable();
    while i < n {
        if let Some(sp) = si.peek().filter(|s| s.start == i) {
            let (gh, gw) = images[sp.image].grid();
            for r in 0..gh {
                for c in 0..gw {
                    let t = i + r * gw + c;
                    rp[t] = cur;
                    rp[n + t] = cur + r as i64;
                    rp[2 * n + t] = cur + c as i64;
                }
            }
            cur += gh.max(gw) as i64;
            i += sp.len;
            si.next();
        } else {
            for a in 0..3 {
                rp[a * n + i] = cur;
            }
            cur += 1;
            i += 1;
        }
    }
    rp
}

/// Prefix-comparison keys: the token id for text, a value derived from the image hash and the offset inside the span
/// for image tokens (every image token has the same id, so ids alone cannot tell two images apart).
pub fn prefix_keys(ids: &[u32], spans: &[Span], images: &[Arc<MmImage>]) -> Vec<u64> {
    let mut k: Vec<u64> = ids.iter().map(|&t| t as u64).collect();
    for sp in spans {
        let h = u64::from_le_bytes(images[sp.image].hash[..8].try_into().unwrap());
        for j in 0..sp.len {
            k[sp.start + j] = (1u64 << 63) | (h.wrapping_add(j as u64) >> 1);
        }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(h: usize, w: usize, cfg: &ImageCfg) -> Arc<MmImage> {
        Arc::new(MmImage::from_rgb(vec![7u8; h * w * 3], h, w, cfg).unwrap())
    }

    #[test]
    fn smart_resize_matches_python() {
        // values from transformers' qwen2_vl.smart_resize(h, w, 32, 65536, 16777216)
        assert_eq!(smart_resize(480, 640, 32, 65536, 16777216).unwrap(), (480, 640));
        assert_eq!(smart_resize(100, 100, 32, 65536, 16777216).unwrap(), (256, 256));
        assert_eq!(smart_resize(1080, 1920, 32, 65536, 16777216).unwrap(), (1088, 1920));
        assert_eq!(smart_resize(48, 80, 32, 65536, 16777216).unwrap(), (224, 352));
        // 16 / 32 = 0.5 -> round half to even -> 0, then the min_pixels branch
        assert_eq!(smart_resize(16, 2000, 32, 65536, 16777216).unwrap(), (32, 2880));
        // max_pixels branch
        assert_eq!(smart_resize(4000, 3000, 32, 65536, 4194304).unwrap(), (2336, 1760));
        assert!(smart_resize(10, 3000, 32, 65536, 16777216).is_err());
    }

    #[test]
    fn expand_and_positions() {
        let cfg = ImageCfg { min_pixels: 65536, max_pixels: 16777216, max_images: 16 };
        let a = img(256, 384, &cfg); // grid 8 x 12 = 96 tokens
        let ids = vec![1, VISION_START, IMAGE_PAD, VISION_END, 5, 6];
        let (x, spans) = expand(&ids, &[a.clone()]).unwrap();
        assert_eq!(x.len(), 5 + 96);
        assert_eq!(spans, vec![Span { start: 2, len: 96, image: 0 }]);
        let rp = rope_positions(x.len(), &spans, &[a.clone()]);
        let n = x.len();
        assert_eq!((rp[0], rp[1]), (0, 1));
        // first image token (0, 0), last (7, 11): t = 2, h = 2 + r, w = 2 + c
        assert_eq!((rp[2], rp[n + 2], rp[2 * n + 2]), (2, 2, 2));
        assert_eq!((rp[97], rp[n + 97], rp[2 * n + 97]), (2, 9, 13));
        // vision_end after the span: 2 + max(8, 12) = 14
        assert_eq!((rp[98], rp[n + 98], rp[2 * n + 98]), (14, 14, 14));
        assert_eq!(rp[n - 1], 16);
        assert!(expand(&ids, &[]).is_err());
        let k1 = prefix_keys(&x, &spans, &[a.clone()]);
        let b = Arc::new(MmImage::from_rgb(vec![8u8; 256 * 384 * 3], 256, 384, &cfg).unwrap());
        let k2 = prefix_keys(&x, &spans, &[b]);
        assert_eq!(k1[..2], k2[..2]);
        assert_ne!(k1[2..98], k2[2..98]);
    }

    #[test]
    fn data_url_png() {
        let mut png = Vec::new();
        let im = image::RgbaImage::from_fn(300, 200, |x, y| image::Rgba([x as u8, y as u8, 9, 128]));
        image::DynamicImage::ImageRgba8(im)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let url = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(&png));
        let cfg = ImageCfg { min_pixels: 65536, max_pixels: 16777216, max_images: 16 };
        let m = load_url(&url, &cfg).unwrap();
        assert_eq!((m.h, m.w, m.rh, m.rw), (200, 300, 224, 320));
        assert_eq!(&m.rgb[..3], &[0, 0, 9]); // alpha dropped
        assert!(load_url("https://example.com/a.png", &cfg).is_err());
    }
}
