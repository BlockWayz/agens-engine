//! CPU reference implementations. Slow and simple; they define the semantics GPU kernels are tested against
//! (bit layouts, hashing). Numerics-exact references for the bf16 path come from the Python oracle, not here.

use half::bf16;

pub fn bf16_from_bytes(b: &[u8]) -> Vec<bf16> {
    b.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect()
}

pub fn bf16_to_bytes(v: &[bf16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

/// compressed-tensors `pack-quantized` W4, symmetric, group 128: value q in [-8, 7] stored as q + 8 in a nibble,
/// 8 nibbles per i32 along the input dim, lowest nibble = lowest input index.
pub mod w4 {
    use super::bf16;
    pub const GROUP: usize = 128;

    pub fn pack_row(q: &[i8]) -> Vec<i32> {
        assert!(q.len() % 8 == 0);
        q.chunks_exact(8)
            .map(|c| {
                let mut w = 0u32;
                for (j, &v) in c.iter().enumerate() {
                    assert!((-8..=7).contains(&v), "w4 value {v} out of range");
                    w |= ((v + 8) as u32 & 0xF) << (4 * j);
                }
                w as i32
            })
            .collect()
    }

    pub fn unpack_row(p: &[i32]) -> Vec<i8> {
        p.iter().flat_map(|&w| (0..8).map(move |j| (((w as u32) >> (4 * j)) & 0xF) as i8 - 8)).collect()
    }

    /// Symmetric per-group quantisation of one row (round-to-nearest; GPTQ uses its own rounding, the format is the same).
    pub fn quantize_row(w: &[f32]) -> (Vec<i8>, Vec<bf16>) {
        assert!(w.len() % GROUP == 0);
        let mut q = Vec::with_capacity(w.len());
        let mut s = Vec::with_capacity(w.len() / GROUP);
        for g in w.chunks_exact(GROUP) {
            let amax = g.iter().fold(0f32, |a, &x| a.max(x.abs()));
            let scale = bf16::from_f32(if amax == 0.0 { 1.0 } else { amax / 7.0 });
            let sf = scale.to_f32();
            q.extend(g.iter().map(|&x| (x / sf).round().clamp(-8.0, 7.0) as i8));
            s.push(scale);
        }
        (q, s)
    }

    /// y[o] = sum_i x[i] * q[o,i] * scale[o, i/128], fp32 accumulation per group then across groups.
    pub fn gemv(x: &[bf16], packed: &[i32], scales: &[bf16], out: usize, inp: usize) -> Vec<f32> {
        assert_eq!(packed.len(), out * inp / 8);
        assert_eq!(scales.len(), out * inp / GROUP);
        (0..out)
            .map(|o| {
                let q = unpack_row(&packed[o * inp / 8..(o + 1) * inp / 8]);
                let mut acc = 0f32;
                for g in 0..inp / GROUP {
                    let mut ga = 0f32;
                    for i in g * GROUP..(g + 1) * GROUP {
                        ga += x[i].to_f32() * q[i] as f32;
                    }
                    acc += ga * scales[o * inp / GROUP + g].to_f32();
                }
                acc
            })
            .collect()
    }
}

/// Dense bf16 [out, inp] from W4 with any group size: bf16(q * scale) rounded to nearest even, the value
/// ak_w4_dequant_kt produces (and Marlin's bf16 dequant). Used at load time for W4 linears with small k.
pub fn w4_dequant_bf16(packed: &[i32], scales: &[bf16], out: usize, inp: usize, group: usize) -> Vec<bf16> {
    assert_eq!(packed.len(), out * inp / 8);
    assert_eq!(scales.len(), out * inp / group);
    (0..out)
        .flat_map(|o| {
            let q = w4::unpack_row(&packed[o * inp / 8..(o + 1) * inp / 8]);
            let s = &scales[o * inp / group..(o + 1) * inp / group];
            (0..inp).map(move |i| bf16::from_f32(q[i] as f32 * s[i / group].to_f32()))
        })
        .collect()
}

/// Plain bf16 GEMV with fp32 accumulation, y = W x, W [out, inp] row-major.
pub fn gemv_bf16(x: &[bf16], w: &[bf16], out: usize, inp: usize) -> Vec<f32> {
    (0..out).map(|o| (0..inp).map(|i| x[i].to_f32() * w[o * inp + i].to_f32()).sum()).collect()
}

/// Engram n-gram hashing, bit-identical to `volundr_extras.Engram.hash_ids` (torch int64 arithmetic).
pub mod engram {
    pub const PRIMES: [u64; 5] = [1000003, 1000033, 1000037, 1000039, 1000081];
    const MASK61: u64 = 0x1FFF_FFFF_FFFF_FFFF;

    /// Row index of the n-gram ending at the last element of `ids` (ids shorter than n are left-padded with -1).
    pub fn hash_last(ids: &[i64], n: usize, rows: u64) -> u64 {
        let mut code: u64 = 0;
        for j in 0..n {
            // position of the j-th token of the n-gram, counting from the oldest
            let pos = ids.len() as isize - n as isize + j as isize;
            let tok = if pos < 0 { -1 } else { ids[pos as usize] };
            let ext = (tok + 1) as u64; // shift so pad = 0
            // torch: (code * p + ext) & mask in wrapping int64; low 61 bits are identical in u64.
            code = code.wrapping_mul(PRIMES[j % PRIMES.len()]).wrapping_add(ext) & MASK61;
        }
        // code < 2^61 is non-negative in int64, so torch's floor-mod equals u64 %.
        code % rows
    }

    /// Row indices for every position of a sequence (prefill), as `hash_ids(ids, n)` returns.
    pub fn hash_seq(ids: &[i64], n: usize, rows: u64) -> Vec<u64> {
        (1..=ids.len()).map(|t| hash_last(&ids[..t], n, rows)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w4_pack_roundtrip_and_layout() {
        let q: Vec<i8> = (0..16).map(|i| (i % 16) as i8 - 8).collect();
        let p = w4::pack_row(&q);
        assert_eq!(w4::unpack_row(&p), q);
        // first word holds q[0..8] = -8..-1 -> nibbles 0..7, lowest first
        assert_eq!(p[0] as u32, 0x7654_3210);
    }

    #[test]
    fn w4_gemv_matches_dequant() {
        let (out, inp) = (3, 256);
        let w: Vec<f32> = (0..out * inp).map(|i| ((i * 37 % 101) as f32 - 50.0) / 100.0).collect();
        let x: Vec<bf16> = (0..inp).map(|i| bf16::from_f32(((i * 13 % 17) as f32 - 8.0) / 8.0)).collect();
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        let mut deq = Vec::new();
        for o in 0..out {
            let (q, s) = w4::quantize_row(&w[o * inp..(o + 1) * inp]);
            deq.extend(q.iter().enumerate().map(|(i, &v)| v as f32 * s[i / w4::GROUP].to_f32()));
            packed.extend(w4::pack_row(&q));
            scales.extend(s);
        }
        let y = w4::gemv(&x, &packed, &scales, out, inp);
        for o in 0..out {
            let r: f32 = (0..inp).map(|i| x[i].to_f32() * deq[o * inp + i]).sum();
            assert!((y[o] - r).abs() < 1e-3 * (1.0 + r.abs()), "{} vs {}", y[o], r);
        }
        // dense dequant: bf16 of the same products, group 128 as stored; group 64 reads one scale per 64 inputs
        let d = w4_dequant_bf16(&packed, &scales, out, inp, w4::GROUP);
        assert!(d.iter().zip(&deq).all(|(a, b)| *a == bf16::from_f32(*b)));
        let s64: Vec<bf16> = scales.iter().flat_map(|&s| [s, s]).collect();
        assert_eq!(w4_dequant_bf16(&packed, &s64, out, inp, 64), d);
    }

    #[test]
    fn engram_hash_matches_torch_semantics() {
        // Values computed by hand from the torch definition (see tools/oracle for the cross-check on real ids).
        let rows = 1u64 << 20;
        // n=2 at t=0: ext = [0 (pad), id0+1]; code = (0*p0 + 0)*p1 + (id0+1)
        assert_eq!(engram::hash_last(&[5], 2, rows), 6);
        // n=2 at t=1: code = (6)*1000033 + 8
        assert_eq!(engram::hash_last(&[5, 7], 2, rows), (6 * 1000033 + 8) % rows);
        // n=3 large ids wrap in 61 bits exactly like int64 & mask
        let ids = [248045i64, 248046, 248069];
        let mut c: u64 = 0;
        for (j, &t) in ids.iter().enumerate() {
            c = c.wrapping_mul(engram::PRIMES[j]).wrapping_add((t + 1) as u64) & 0x1FFF_FFFF_FFFF_FFFF;
        }
        assert_eq!(engram::hash_last(&ids, 3, rows), c % rows);
        assert_eq!(engram::hash_seq(&ids, 3, rows).len(), 3);
    }
}
