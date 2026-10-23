//! FFI to `kernels/` (libagens_kernels, CUDA C++). Without the `cuda` feature every entry point returns an error,
//! so host code and its tests build on machines without CUDA.

use anyhow::{bail, Result};

#[cfg(feature = "cuda")]
mod ffi {
    use std::os::raw::{c_char, c_int, c_void};
    extern "C" {
        pub fn ak_device_count(n: *mut c_int) -> c_int;
        pub fn ak_set_device(dev: c_int) -> c_int;
        pub fn ak_device_name(dev: c_int, buf: *mut c_char, len: c_int) -> c_int;
        pub fn ak_malloc(ptr: *mut *mut c_void, bytes: usize) -> c_int;
        pub fn ak_free(ptr: *mut c_void) -> c_int;
        pub fn ak_memcpy_h2d(dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
        pub fn ak_memcpy_d2h(dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
        pub fn ak_synchronize() -> c_int;
        pub fn ak_error_string(code: c_int) -> *const c_char;
        pub fn ak_mem_info(free: *mut usize, total: *mut usize) -> c_int;
        pub fn ak_host_alloc(ptr: *mut *mut c_void, bytes: usize) -> c_int;
        pub fn ak_host_free(ptr: *mut c_void) -> c_int;
        pub fn ak_rope_cache(out: *mut f32, max_pos: c_int, rot: c_int, base: f32, stream: *mut c_void) -> c_int;
        pub fn ak_w4_kt_weight_bytes(n: c_int, k: c_int) -> usize;
        pub fn ak_w3_kt_weight_bytes(n: c_int, k: c_int) -> usize;
        pub fn ak_w3_repack_kt(packed: *const i32, scales: *const c_void, wt: *mut c_void, st: *mut c_void, n: c_int,
                               k: c_int, group: c_int, stream: *mut c_void) -> c_int;
        pub fn ak_w4_kt_scale_bytes(n: c_int, k: c_int, group: c_int) -> usize;
        pub fn ak_wq_kt_weight_bytes(bits: c_int, n: c_int, k: c_int) -> usize;
        pub fn ak_wq_repack_kt(bits: c_int, packed: *const i32, scales: *const c_void, wt: *mut c_void, st: *mut c_void,
                               n: c_int, k: c_int, group: c_int, stream: *mut c_void) -> c_int;
        pub fn ak_w4_repack_kt(packed: *const i32, scales: *const c_void, wt: *mut c_void, st: *mut c_void, n: c_int,
                               k: c_int, group: c_int, stream: *mut c_void) -> c_int;
    }
}

#[cfg(feature = "cuda")]
fn check(code: std::os::raw::c_int, what: &str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let msg = unsafe { std::ffi::CStr::from_ptr(ffi::ak_error_string(code)) }.to_string_lossy().into_owned();
    bail!("{what}: CUDA error {code} ({msg})")
}

pub fn available() -> bool {
    cfg!(feature = "cuda")
}

/// Owned device allocation.
pub struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    bytes: usize,
}

unsafe impl Send for DeviceBuffer {}

impl DeviceBuffer {
    pub fn len(&self) -> usize {
        self.bytes
    }
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }
}

#[cfg(feature = "cuda")]
mod imp {
    use super::*;

    pub fn device_count() -> Result<usize> {
        let mut n = 0;
        check(unsafe { ffi::ak_device_count(&mut n) }, "device_count")?;
        Ok(n as usize)
    }
    pub fn set_device(dev: usize) -> Result<()> {
        check(unsafe { ffi::ak_set_device(dev as i32) }, "set_device")
    }
    pub fn device_name(dev: usize) -> Result<String> {
        let mut buf = vec![0i8; 256];
        check(unsafe { ffi::ak_device_name(dev as i32, buf.as_mut_ptr() as _, 256) }, "device_name")?;
        Ok(unsafe { std::ffi::CStr::from_ptr(buf.as_ptr() as _) }.to_string_lossy().into_owned())
    }
    pub fn mem_info() -> Result<(usize, usize)> {
        let (mut f, mut t) = (0, 0);
        check(unsafe { ffi::ak_mem_info(&mut f, &mut t) }, "mem_info")?;
        Ok((f, t))
    }
    pub fn upload(data: &[u8]) -> Result<DeviceBuffer> {
        let mut ptr = std::ptr::null_mut();
        check(unsafe { ffi::ak_malloc(&mut ptr, data.len().max(1)) }, "malloc")?;
        let buf = DeviceBuffer { ptr, bytes: data.len() };
        check(unsafe { ffi::ak_memcpy_h2d(ptr, data.as_ptr() as _, data.len()) }, "memcpy_h2d")?;
        Ok(buf)
    }
    pub fn download(buf: &DeviceBuffer) -> Result<Vec<u8>> {
        let mut out = vec![0u8; buf.bytes];
        check(unsafe { ffi::ak_memcpy_d2h(out.as_mut_ptr() as _, buf.ptr, buf.bytes) }, "memcpy_d2h")?;
        Ok(out)
    }
    pub fn synchronize() -> Result<()> {
        check(unsafe { ffi::ak_synchronize() }, "synchronize")
    }
    /// Pinned host memory mapped into the device address space; filled from `data`, never freed (weights).
    pub fn upload_host_mapped(data: &[u8]) -> Result<*mut std::ffi::c_void> {
        let mut ptr = std::ptr::null_mut();
        check(unsafe { ffi::ak_host_alloc(&mut ptr, data.len().max(1)) }, "host_alloc")?;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len()) };
        Ok(ptr)
    }
    pub fn rope_cache(max_pos: usize, rot: usize, base: f32) -> Result<DeviceBuffer> {
        let buf = upload(&vec![0u8; max_pos * rot * 4])?;
        check(unsafe { ffi::ak_rope_cache(buf.ptr as *mut f32, max_pos as i32, rot as i32, base, std::ptr::null_mut()) }, "rope_cache")?;
        synchronize()?;
        Ok(buf)
    }
    /// W4 linear [n, k] in the on-disk compressed-tensors layout (packed I32 [n, k/8], scales BF16 [n, k/group]) ->
    /// the k-tiled device layout the W4 kernels read: (weights, scales). The on-disk copies are only staged on the device.
    pub fn w4_upload_kt(packed: &[u8], scales: &[u8], n: usize, k: usize, group: usize) -> Result<(DeviceBuffer, DeviceBuffer)> {
        if packed.len() != n * k / 2 || scales.len() != n * k / group * 2 {
            bail!("w4_upload_kt: {} / {} bytes for [{n}, {k}] group {group}", packed.len(), scales.len());
        }
        let (p, sc) = (upload(packed)?, upload(scales)?);
        let wt = device_alloc(unsafe { ffi::ak_w4_kt_weight_bytes(n as i32, k as i32) })?;
        let st = device_alloc(unsafe { ffi::ak_w4_kt_scale_bytes(n as i32, k as i32, group as i32) })?;
        check(unsafe { ffi::ak_w4_repack_kt(p.ptr as *const i32, sc.ptr, wt.ptr, st.ptr, n as i32, k as i32, group as i32,
                                            std::ptr::null_mut()) }, "w4_repack_kt")?;
        synchronize()?;
        Ok((wt, st))
    }
    /// W4-container linear whose values all fit 3 bits ([-4, 3]) -> the W3 k-tiled layout (3/8 byte per weight), or
    /// None when some value needs 4 bits.
    pub fn w3_upload_kt(packed: &[u8], scales: &[u8], n: usize, k: usize, group: usize)
                        -> Result<Option<(DeviceBuffer, DeviceBuffer)>> {
        if packed.len() != n * k / 2 || scales.len() != n * k / group * 2 {
            bail!("w3_upload_kt: {} / {} bytes for [{n}, {k}] group {group}", packed.len(), scales.len());
        }
        let (p, sc) = (upload(packed)?, upload(scales)?);
        let wt = device_alloc(unsafe { ffi::ak_w3_kt_weight_bytes(n as i32, k as i32) })?;
        let st = device_alloc(unsafe { ffi::ak_w4_kt_scale_bytes(n as i32, k as i32, group as i32) })?;
        let e = unsafe { ffi::ak_w3_repack_kt(p.ptr as *const i32, sc.ptr, wt.ptr, st.ptr, n as i32, k as i32, group as i32,
                                              std::ptr::null_mut()) };
        if e == 1 {
            return Ok(None); // cudaErrorInvalidValue: a 4-bit value
        }
        check(e, "w3_repack_kt")?;
        Ok(Some((wt, st)))
    }
    /// Mixed precision: int4-container linear (values all within `bits` bits) -> the W{bits} k-tiled layout (bits/8 byte
    /// per weight; W4 = w4_upload_kt). `scales` are the group-`group` scales (a g128 linear's container g64 scales
    /// collapsed by the caller). Errors when a value needs more bits than declared.
    pub fn wq_upload_kt(bits: usize, packed: &[u8], scales: &[u8], n: usize, k: usize, group: usize)
                        -> Result<(DeviceBuffer, DeviceBuffer)> {
        if bits == 4 {
            return w4_upload_kt(packed, scales, n, k, group);
        }
        if packed.len() != n * k / 2 || scales.len() != n * k / group * 2 {
            bail!("wq_upload_kt: {} / {} bytes for [{n}, {k}] group {group}", packed.len(), scales.len());
        }
        let (p, sc) = (upload(packed)?, upload(scales)?);
        let wt = device_alloc(unsafe { ffi::ak_wq_kt_weight_bytes(bits as i32, n as i32, k as i32) })?;
        let st = device_alloc(unsafe { ffi::ak_w4_kt_scale_bytes(n as i32, k as i32, group as i32) })?;
        let e = unsafe { ffi::ak_wq_repack_kt(bits as i32, p.ptr as *const i32, sc.ptr, wt.ptr, st.ptr, n as i32, k as i32,
                                              group as i32, std::ptr::null_mut()) };
        if e == 1 {
            bail!("wq_upload_kt: [{n}, {k}] has values outside {bits} bits");
        }
        check(e, "wq_repack_kt")?;
        Ok((wt, st))
    }
    /// Pinned host buffer (not initialised); free with host_free.
    pub fn host_alloc(bytes: usize) -> Result<*mut std::ffi::c_void> {
        let mut ptr = std::ptr::null_mut();
        check(unsafe { ffi::ak_host_alloc(&mut ptr, bytes.max(1)) }, "host_alloc")?;
        Ok(ptr)
    }
    pub fn host_free(p: *mut std::ffi::c_void) -> Result<()> {
        check(unsafe { ffi::ak_host_free(p) }, "host_free")
    }
    pub fn device_alloc(bytes: usize) -> Result<DeviceBuffer> {
        let mut ptr = std::ptr::null_mut();
        check(unsafe { ffi::ak_malloc(&mut ptr, bytes.max(1)) }, "malloc")?;
        Ok(DeviceBuffer { ptr, bytes })
    }
    /// Leak a device buffer (weights live for the process) and return its pointer.
    pub fn leak(buf: DeviceBuffer) -> *mut std::ffi::c_void {
        let p = buf.ptr;
        std::mem::forget(buf);
        p
    }
    impl Drop for DeviceBuffer {
        fn drop(&mut self) {
            unsafe { ffi::ak_free(self.ptr) };
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod imp {
    use super::*;
    const NO: &str = "built without the `cuda` feature";
    pub fn device_count() -> Result<usize> {
        Ok(0)
    }
    pub fn set_device(_: usize) -> Result<()> {
        bail!(NO)
    }
    pub fn device_name(_: usize) -> Result<String> {
        bail!(NO)
    }
    pub fn mem_info() -> Result<(usize, usize)> {
        bail!(NO)
    }
    pub fn upload(_: &[u8]) -> Result<DeviceBuffer> {
        bail!(NO)
    }
    pub fn download(_: &DeviceBuffer) -> Result<Vec<u8>> {
        bail!(NO)
    }
    pub fn synchronize() -> Result<()> {
        bail!(NO)
    }
}

pub use imp::*;

#[cfg(test)]
mod tests {
    #[test]
    fn stub_reports_no_devices_without_cuda() {
        if !super::available() {
            assert_eq!(super::device_count().unwrap(), 0);
            assert!(super::upload(&[1, 2, 3]).is_err());
        }
    }
}

/// M1 standalone forward (kernels/m1/m1_model.cu).
#[cfg(feature = "m1")]

/// Tensor parallel (one process per rank): the leader (rank 0) forwards every engine call that changes state to the
/// followers as a text line on their stdin, before making it itself, so all ranks run the same calls in the same order
/// (their all-reduces pair up on the GPUs). Followers execute the lines (agens-cli tp::follow).
pub mod tp {
    use std::io::Write;
    use std::sync::Mutex;
    static FOLLOWERS: Mutex<Vec<std::process::ChildStdin>> = Mutex::new(Vec::new());
    static ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    pub fn set_followers(v: Vec<std::process::ChildStdin>) {
        *FOLLOWERS.lock().unwrap() = v;
    }
    /// From here on the leader's calls are forwarded (the followers ran the same setup themselves and now follow).
    pub fn activate() {
        ACTIVE.store(true, std::sync::atomic::Ordering::SeqCst);
    }
    /// Tensor parallel calls are being forwarded (leader of a --tp run).
    pub fn active() -> bool {
        ACTIVE.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn fwd(line: &str) {
        if !ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let mut f = FOLLOWERS.lock().unwrap();
        for s in f.iter_mut() {
            // a follower that died takes its rank's all-reduces with it: the leader's GPU would wait forever
            if s.write_all(line.as_bytes()).and_then(|_| s.write_all(b"\n")).and_then(|_| s.flush()).is_err() {
                eprintln!("tp: a follower is gone (write failed); exiting");
                std::process::exit(3);
            }
        }
    }
    pub fn join(v: &[i32]) -> String {
        if v.is_empty() {
            "-".into()
        } else {
            v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
        }
    }
    pub fn split(s: &str) -> Vec<i32> {
        if s == "-" || s.is_empty() { Vec::new() } else { s.split(',').map(|x| x.parse().unwrap_or(0)).collect() }
    }
    /// Payload too large for a line (image embeddings): a file in /dev/shm per call, named in the forwarded line; the
    /// followers each read it and the last one removes it (`take_blob`). Returns the path.
    pub fn put_blob(bytes: &[u8]) -> std::io::Result<String> {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = std::env::var("AGENS_TP_ID").unwrap_or_else(|_| "0".into());
        let k = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = format!("/dev/shm/agens_tp_{id}_blob{k}");
        std::fs::write(&path, bytes)?;
        let n = FOLLOWERS.lock().unwrap().len();
        std::fs::write(format!("{path}.n"), n.to_string())?;
        Ok(path)
    }
    /// A follower reads a blob; the last reader (by the countdown file) removes both files.
    pub fn take_blob(path: &str) -> std::io::Result<Vec<u8>> {
        let b = std::fs::read(path)?;
        let cnt = format!("{path}.n");
        // countdown under an exclusive lock file (followers run concurrently)
        let lock = format!("{path}.lock");
        loop {
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&lock) {
                Ok(_) => break,
                Err(_) => std::thread::sleep(std::time::Duration::from_micros(200)),
            }
        }
        let left = std::fs::read_to_string(&cnt).ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(1);
        if left <= 1 {
            let _ = std::fs::remove_file(path);
            let _ = std::fs::remove_file(&cnt);
        } else {
            let _ = std::fs::write(&cnt, (left - 1).to_string());
        }
        let _ = std::fs::remove_file(&lock);
        Ok(b)
    }
}

pub mod m1 {
    use super::*;
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int, c_void};
    extern "C" {
        fn ak_m1_register(name: *const c_char, ptr: *mut c_void) -> c_int;
        fn ak_m1_register_w4(name: *const c_char, wt: *mut c_void, st: *mut c_void, n: c_int, k: c_int, group: c_int) -> c_int;
        fn ak_m1_register_w3(name: *const c_char, wt: *mut c_void, st: *mut c_void, n: c_int, k: c_int, group: c_int) -> c_int;
        fn ak_m1_register_wq(name: *const c_char, wt: *mut c_void, st: *mut c_void, n_total: c_int, k: c_int, r0: c_int,
                             n: c_int, group: c_int, bits: c_int) -> c_int;
        fn ak_m1_step_stats(out: *mut f64) -> c_int;
        fn ak_m1_profiler(on: c_int) -> c_int;
        fn ak_m1_spec_init(bi: c_int, cap_layers: *const c_int, cap_rows: c_int) -> c_int;
        fn ak_m1_verify(tokens: *const c_int, t: c_int, emitted: *mut c_int, n: *mut c_int, target: *mut c_int) -> c_int;
        fn ak_m1_draft_register(name: *const c_char, ptr: *mut c_void) -> c_int;
        fn ak_m1_draft_register_w4(name: *const c_char, wt: *mut c_void, st: *mut c_void, n: c_int, k: c_int, group: c_int)
                                   -> c_int;
        fn ak_m1_draft_register_wq(name: *const c_char, wt: *mut c_void, st: *mut c_void, n: c_int, k: c_int, group: c_int,
                                   bits: c_int) -> c_int;
        fn ak_m1_draft_setup(mask_token_id: c_int) -> i64;
        fn ak_m1_mtp_register(name: *const c_char, ptr: *mut c_void) -> c_int;
        fn ak_m1_mtp_register_wq(name: *const c_char, wt: *mut c_void, st: *mut c_void, n: c_int, k: c_int, group: c_int,
                                 bits: c_int) -> c_int;
        fn ak_m1_mtp_setup(max_rows: c_int, window: c_int) -> i64;
        fn ak_m1_mtp_shortlist(ids: *const c_int, n: c_int) -> c_int;
        fn ak_m1_draft_vocab(n: c_int, drafter: c_int) -> c_int;
        fn ak_m1_spec_step(bonus: c_int, emitted: *mut c_int, n: *mut c_int, drafts: *mut c_int) -> c_int;
        fn ak_m1_spec_step_hint(bonus: c_int, hint: *const c_int, nh: c_int, emitted: *mut c_int, n: *mut c_int,
                                drafts: *mut c_int) -> c_int;
        fn ak_m1_spec_batch_init(max_b: c_int) -> c_int;
        fn ak_m1_spec_batch_max() -> c_int;
        fn ak_m1_session_sampling(sess: c_int, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, seed: c_int) -> c_int;
        fn ak_m1_session_penalty(sess: c_int, rep: f32, freq: f32, pres: f32, bias_ids: *const c_int, bias_vals: *const f32,
                                 nb: c_int) -> c_int;
        fn ak_m1_session_penalty_add(sess: c_int, ids: *const c_int, n: c_int) -> c_int;
        fn ak_m1_sample_decode(sess: *const c_int, use_row: *const c_int, b: c_int, out: *mut c_int) -> c_int;
        fn ak_m1_spec_batch_hint(sess: *const c_int, b: c_int, bonus: *const c_int, hint: *const c_int, nh: *const c_int,
                                 emitted: *mut c_int, n: *mut c_int) -> c_int;
        fn ak_m1_spec_batch(sess: *const c_int, b: c_int, bonus: *const c_int, emitted: *mut c_int, n: *mut c_int) -> c_int;
        fn ak_m1_spec_sampling(temperature: f32, top_k: c_int, top_p: f32, min_p: f32, seed: c_int) -> c_int;
        fn ak_m1_tp_config(rank: c_int, size: c_int) -> c_int;
        fn ak_tp_region_alloc(bytes: usize, dev_ptr: *mut *mut c_void, ipc_handle: *mut c_void) -> c_int;
        fn ak_tp_region_open(ipc_handle: *const c_void, dev_ptr: *mut *mut c_void) -> c_int;
        fn ak_tp_region_shm(name: *const std::os::raw::c_char, bytes: usize, create: c_int, dev_ptr: *mut *mut c_void)
                            -> c_int;
        fn ak_tp_can_peer(dev: c_int, peer: c_int) -> c_int;
        fn ak_tp_setup(rank: c_int, size: c_int, regions: *const *mut c_void, slot_elems: i64) -> c_int;
        fn ak_tp_region_bytes(slot_elems: i64) -> usize;
        fn ak_tp_selftest(iters: c_int, n: i64) -> i64;
        fn ak_tp_bench(iters: c_int, n: i64) -> f64;
        fn ak_tp_push_selftest(iters: c_int) -> i64;
        fn ak_tp_bench2(iters: c_int, rows: i64, which: c_int, graph: c_int) -> f64;
        fn ak_m1_init(wctx: c_int, mixers: *const c_char, engram_a: c_int, engram_b: c_int, e2a: *mut c_void,
                      e2b: *mut c_void, e22a: *mut c_void, e22b: *mut c_void, ndev: c_int, devs: *const c_int,
                      split: c_int, pool_rows: c_int, nslots: c_int) -> c_int;
        fn ak_m1_prefill(ids: *const c_int, n: c_int, next: *mut c_int) -> c_int;
        fn ak_m1_prefill_mm(ids: *const c_int, n: c_int, rpos3: *const i64, emb: *const c_void, emb_row: *const c_int,
                            next: *mut c_int) -> c_int;
        fn ak_m1_rope_delta() -> c_int;
        fn ak_m1_set_rope_delta(d: c_int) -> c_int;
        fn ak_m1_vis_register(name: *const c_char, host: *mut c_void, shape: *const i64, ndim: c_int) -> c_int;
        fn ak_m1_vis_ready() -> c_int;
        fn ak_m1_vis_encode(rgb: *const u8, jpeg: *const u8, jlen: i64, h: c_int, w: c_int, rh: c_int, rw: c_int,
                            out: *mut c_void) -> c_int;
        fn ak_m1_vis_pixels(rgb: *const u8, jpeg: *const u8, jlen: i64, h: c_int, w: c_int, rh: c_int, rw: c_int,
                            out: *mut f32) -> c_int;
        fn ak_m1_vis_encode_pixels(pv: *const f32, gh: c_int, gw: c_int, out: *mut c_void) -> c_int;
        fn ak_m1_vis_peak_bytes() -> i64;
        fn ak_m1_decode(token: c_int, next: *mut c_int) -> c_int;
        fn ak_m1_reset() -> c_int;
        fn ak_m1_state_bytes() -> usize;
        fn ak_m1_snapshot(dst: *mut c_void) -> c_int;
        fn ak_m1_restore(src: *const c_void) -> c_int;
        fn ak_m1_seq() -> c_int;
        fn ak_m1_session_new() -> c_int;
        fn ak_m1_session_use(id: c_int) -> c_int;
        fn ak_m1_session_current() -> c_int;
        fn ak_m1_session_fork(src: c_int) -> c_int;
        fn ak_m1_session_free(id: c_int) -> c_int;
        fn ak_m1_pool_free_rows() -> c_int;
        fn ak_m1_rows_freed(ids: *const c_int, n: c_int) -> c_int;
        fn ak_m1_live_sessions(rows_used: *mut c_int) -> c_int;
        fn ak_m1_kv_groups(ids: *const c_int, grp: *const c_int, n: c_int, ng: c_int, out: *mut c_int) -> c_int;
        fn ak_m1_logits(dst: *mut c_void) -> c_int;
        fn ak_m1_logits_row(r: c_int, dst: *mut c_void) -> c_int;
        fn ak_m1_logits_i8_row(r: c_int, dst: *mut c_void) -> c_int;
        fn ak_m1_torch_reserved(release: c_int) -> i64;
        fn ak_m1_kda_pool_bytes(used: c_int) -> i64;
        fn ak_m1_prefill_reserve(kda: i64, torch: i64) -> c_int;
        fn ak_m1_decode_batch(sess: *const c_int, tok: *const c_int, b: c_int, next: *mut c_int) -> c_int;
        fn ak_m1_park_bytes() -> usize;
        fn ak_m1_session_park(id: c_int, dst: *mut c_void) -> c_int;
        fn ak_m1_session_unpark(id: c_int, src: *const c_void) -> c_int;
        fn ak_m1_session_parked(id: c_int) -> c_int;
        fn ak_m1_snap_bytes() -> usize;
        fn ak_m1_w4_cache_bytes() -> usize;
        fn ak_w4h_dual_bytes() -> usize;
        fn ak_m1_w4_cache_build() -> c_int;
        fn ak_m1_session_snap(src: c_int, dst: *mut c_void) -> c_int;
        fn ak_m1_session_from_snap(snap: c_int, src: *const c_void) -> c_int;
        fn ak_m1_free_slots() -> c_int;
        fn ak_m1_recover() -> c_int;
    }
    /// After a failed engine call (e.g. out of device memory under the --budget-gib cap): end an open stream capture,
    /// free every cached CUDA graph, clear the error state. Ok: the context is usable and the engine can go on; Err: a
    /// sticky CUDA error (the process must exit). One device, no tensor parallelism.
    pub fn recover() -> Result<()> {
        check(unsafe { ak_m1_recover() }, "m1_recover")
    }
    /// One batched decode step: row r feeds `tokens[r]` to session `sessions[r]` (distinct, not parked); returns the
    /// greedy token of each row. Rows are bit-identical to bs = 1 decodes of the same sessions.
    pub fn decode_batch(sessions: &[i32], tokens: &[i32]) -> Result<Vec<i32>> {
        assert_eq!(sessions.len(), tokens.len());
        crate::tp::fwd(&format!("decode_batch {} {}", crate::tp::join(sessions), crate::tp::join(tokens)));
        let mut next = vec![0i32; sessions.len()];
        check(unsafe { ak_m1_decode_batch(sessions.as_ptr(), tokens.as_ptr(), sessions.len() as c_int, next.as_mut_ptr()) },
              "m1_decode_batch")?;
        Ok(next)
    }
    /// Logits (bf16 bits) of row r of the last step (row 0 = the last token of a prefill / bs = 1 step).
    /// Bytes torch's caching allocator holds on the device (ATen far branch), after releasing unused ones if asked.
    /// Bytes the default stream-ordered pool (KDA prefill temporaries) holds (reserved, or in use).
    pub fn kda_pool_bytes(used: bool) -> i64 {
        unsafe { ak_m1_kda_pool_bytes(used as c_int) }
    }
    /// Take the planned prefill temporaries into their pools now (budget mode, before any CUDA graph). Every rank
    /// runs it during its own load (no forwarding).
    pub fn prefill_reserve(kda: usize, torch: usize) -> Result<()> {
        check(unsafe { ak_m1_prefill_reserve(kda as i64, torch as i64) }, "m1_prefill_reserve")
    }
    pub fn torch_reserved(release: bool) -> i64 {
        unsafe { ak_m1_torch_reserved(release as c_int) }
    }
    /// The INT8 lm_head's logits for row r of the last head input (both heads registered: AGENS_M1_LMHEAD_INT8=cmp).
    pub fn logits_i8_row(r: usize) -> Result<Vec<u16>> {
        let mut v = vec![0u16; 262144];
        let n = unsafe { ak_m1_logits_i8_row(r as c_int, v.as_mut_ptr() as *mut c_void) };
        if n < 0 { anyhow::bail!("m1_logits_i8_row failed ({})", -n) }
        v.truncate(n as usize);
        Ok(v)
    }
    pub fn logits_row(r: usize) -> Result<Vec<u16>> {
        let mut v = vec![0u16; 262144];
        let n = unsafe { ak_m1_logits_row(r as c_int, v.as_mut_ptr() as *mut c_void) };
        if n < 0 { anyhow::bail!("m1_logits_row failed ({})", -n) }
        v.truncate(n as usize);
        Ok(v)
    }
    /// Bytes of a parked session's state (KDA conv + recurrent state, plus its GPU window rings when tiering).
    pub fn park_bytes() -> usize {
        unsafe { ak_m1_park_bytes() }
    }
    /// Move a session's KDA state (+ rings) to `dst` (pinned host, park_bytes()) and give its slot back.
    pub fn session_park(id: i32, dst: *mut c_void) -> Result<()> {
        crate::tp::fwd(&format!("session_park {id}"));
        check(unsafe { ak_m1_session_park(id, dst) }, "m1_session_park")
    }
    pub fn session_unpark(id: i32, src: *const c_void) -> Result<()> {
        crate::tp::fwd(&format!("session_unpark {id}"));
        check(unsafe { ak_m1_session_unpark(id, src) }, "m1_session_unpark")
    }
    /// Bytes of the W4 prefill weight cache (every W4 linear dequantized to bf16) on this device.
    pub fn w4_cache_bytes() -> usize {
        unsafe { ak_m1_w4_cache_bytes() }
    }
    /// Bytes of the W4H copies of the dual layout (AGENS_W4H_MMAX; kernels/src/w4h.cu).
    pub fn w4h_dual_bytes() -> usize {
        unsafe { ak_w4h_dual_bytes() }
    }
    /// Build the W4 prefill weight cache (all or nothing). Every rank builds its own at load (no forwarding).
    pub fn w4_cache_build() -> Result<()> {
        check(unsafe { ak_m1_w4_cache_build() }, "m1_w4_cache_build")
    }
    /// Bytes of a prefix snapshot's buffer (KDA state + rings as when parked, plus the drafter's context ring).
    pub fn snap_bytes() -> usize {
        unsafe { ak_m1_snap_bytes() }
    }
    /// Prefix cache: a snapshot of session `src` (holding a slot, length a multiple of 4) as a new parked session that
    /// shares its KV pages; the state goes to `dst` (pinned host, snap_bytes()). `key` names the followers' buffers
    /// (tensor parallel: each rank snapshots its own share). -> the snapshot's session id.
    pub fn session_snap(src: i32, dst: *mut c_void, key: u64) -> Result<i32> {
        crate::tp::fwd(&format!("session_snap {src} {key}"));
        let id = unsafe { ak_m1_session_snap(src, dst) };
        if id < 0 { anyhow::bail!("m1_session_snap({src}) failed") }
        Ok(id)
    }
    /// A new session (a free KDA slot, not activated) restored from snapshot session `snap` and its buffer.
    pub fn session_from_snap(snap: i32, src: *const c_void, key: u64) -> Result<i32> {
        crate::tp::fwd(&format!("session_from_snap {snap} {key}"));
        let id = unsafe { ak_m1_session_from_snap(snap, src) };
        if id < 0 { anyhow::bail!("m1_session_from_snap({snap}) failed") }
        Ok(id)
    }
    /// The followers return their buffer of snapshot `key` to their pool (the leader keeps its own pool).
    pub fn snap_drop(key: u64) {
        crate::tp::fwd(&format!("snap_drop {key}"));
    }
    /// The followers allocate their pool of `n` snapshot buffers (device memory or pinned host), as the leader does.
    pub fn snap_pool(n: usize, dev: bool) {
        crate::tp::fwd(&format!("snap_pool {n} {}", if dev { "d" } else { "h" }));
    }
    pub fn session_parked(id: i32) -> bool {
        unsafe { ak_m1_session_parked(id) != 0 }
    }
    pub fn free_slots() -> usize {
        unsafe { ak_m1_free_slots() as usize }
    }
    /// The last forward's logits (bf16 bits, vocab_valid entries) of its last token.
    pub fn logits() -> Result<Vec<u16>> {
        let mut v = vec![0u16; 262144];
        let n = unsafe { ak_m1_logits(v.as_mut_ptr() as *mut c_void) };
        if n < 0 { anyhow::bail!("m1_logits failed ({})", -n) }
        v.truncate(n as usize);
        Ok(v)
    }
    /// A new empty session (not activated). Fails when every KDA state slot is taken.
    pub fn session_new() -> Result<i32> {
        crate::tp::fwd("session_new");
        let id = unsafe { ak_m1_session_new() };
        if id < 0 { anyhow::bail!("m1_session_new: no free KDA slot") }
        Ok(id)
    }
    /// Make `id` the session that prefill/decode/snapshot/restore/reset act on (pointer swap, no copy).
    pub fn session_use(id: i32) -> Result<()> {
        crate::tp::fwd(&format!("session_use {id}"));
        check(unsafe { ak_m1_session_use(id) }, "m1_session_use")
    }
    pub fn session_current() -> i32 {
        unsafe { ak_m1_session_current() }
    }
    /// A new session with `src`'s exact state: KV rows shared, KDA state + Engram pair copied. Not activated.
    pub fn session_fork(src: i32) -> Result<i32> {
        crate::tp::fwd(&format!("session_fork {src}"));
        let id = unsafe { ak_m1_session_fork(src) };
        if id < 0 { anyhow::bail!("m1_session_fork({src}) failed") }
        Ok(id)
    }
    pub fn session_free(id: i32) -> Result<()> {
        crate::tp::fwd(&format!("session_free {id}"));
        check(unsafe { ak_m1_session_free(id) }, "m1_session_free")
    }
    pub fn pool_free_rows() -> usize {
        unsafe { ak_m1_pool_free_rows() as usize }
    }
    /// (live sessions, KV pool pages in use x 4 incl. the padding page) -- leak checks
    pub fn live_sessions() -> (usize, usize) {
        let mut r: c_int = 0;
        let n = unsafe { ak_m1_live_sessions(&mut r) };
        (n.max(0) as usize, r.max(0) as usize)
    }
    /// KV accounting (kernels/m1 ak_m1_kv_groups): sessions `ids` in groups `grp` (0 = idle, 1.. = running requests)
    /// -> [rows evicting group 0 frees, rows each request gives back or leaves evictable when it ends]. Leader-side query.
    pub fn kv_groups(ids: &[i32], grp: &[i32], ng: usize) -> Vec<usize> {
        let mut out = vec![0 as c_int; ng.max(1)];
        unsafe { ak_m1_kv_groups(ids.as_ptr(), grp.as_ptr(), ids.len() as c_int, ng.max(1) as c_int, out.as_mut_ptr()) };
        out.into_iter().map(|x| x.max(0) as usize).collect()
    }
    /// KV pool rows that freeing all of `ids` (distinct sessions) would give back: pages no other session holds.
    /// A leader-side query (the ranks' pools are identical), not forwarded.
    pub fn rows_freed(ids: &[i32]) -> usize {
        if ids.is_empty() {
            return 0;
        }
        unsafe { ak_m1_rows_freed(ids.as_ptr(), ids.len() as c_int).max(0) as usize }
    }
    pub fn state_bytes() -> usize {
        unsafe { ak_m1_state_bytes() }
    }
    /// Snapshot the session into `dst` (device or pinned host pointer of at least state_bytes()).
    /// Snapshot of the active session (Session State v1) into `dst` (host or device, state_bytes()). Tensor parallel:
    /// every rank snapshots its own share (followers into a host buffer of their own; restore reads it back).
    pub fn snapshot(dst: *mut c_void) -> Result<()> {
        crate::tp::fwd("snapshot");
        check(unsafe { ak_m1_snapshot(dst) }, "m1_snapshot")
    }
    pub fn restore(src: *const c_void) -> Result<()> {
        crate::tp::fwd("restore");
        check(unsafe { ak_m1_restore(src) }, "m1_restore")
    }
    pub fn seq() -> usize {
        unsafe { ak_m1_seq() as usize }
    }
    /// Rows of the draft shortlist head (registered as "lm_head.weight.draftsl" + "lm_head.draftsl.ids"); 0 = off. The
    /// MTP chain uses it; the DFlash2 drafter's candidates too when `drafter`.
    pub fn draft_vocab(n: usize, drafter: bool) -> Result<()> {
        check(unsafe { ak_m1_draft_vocab(n as c_int, drafter as c_int) }, "m1_draft_vocab")
    }
    pub fn register(name: &str, ptr: *mut c_void) -> Result<()> {
        let c = CString::new(name)?;
        check(unsafe { ak_m1_register(c.as_ptr(), ptr) }, "m1_register")
    }
    /// A W4 linear in the k-tiled layout (w4_upload_kt), registered under the name of its bf16 weight.
    pub fn register_w3(name: &str, wt: *mut c_void, st: *mut c_void, n: usize, k: usize, group: usize) -> Result<()> {
        let c = CString::new(name)?;
        check(unsafe { ak_m1_register_w3(c.as_ptr(), wt, st, n as i32, k as i32, group as i32) }, "m1_register_w3")
    }
    /// Rows [r0, r0 + n) of the n_total-row linear `name` in the W{bits} KT layout (wq_upload_kt); r0 = 0 starts it.
    pub fn register_wq(name: &str, wt: *mut c_void, st: *mut c_void, n_total: usize, k: usize, r0: usize, n: usize,
                       group: usize, bits: usize) -> Result<()> {
        let c = CString::new(name)?;
        check(unsafe { ak_m1_register_wq(c.as_ptr(), wt, st, n_total as i32, k as i32, r0 as i32, n as i32, group as i32,
                                         bits as i32) }, "m1_register_wq")
    }
    pub fn register_w4(name: &str, wt: *mut c_void, st: *mut c_void, n: usize, k: usize, group: usize) -> Result<()> {
        let c = CString::new(name)?;
        check(unsafe { ak_m1_register_w4(c.as_ptr(), wt, st, n as i32, k as i32, group as i32) }, "m1_register_w4")
    }
    /// Host-side timing of the last forward step (AGENS_M1_STEP_STATS=1): [enqueue ms, wait ms, linears, launches].
    pub fn step_stats() -> [f64; 4] {
        let mut v = [0f64; 4];
        unsafe { ak_m1_step_stats(v.as_mut_ptr()) };
        v
    }
    /// Speculative decoding (one device): batch-invariant decode on/off, and the target layers whose entry residual
    /// stream 0 is captured for the drafter (sglang's DFlash ids + 1), with room for `cap_rows` tokens per forward.
    pub fn spec_init(bi: bool, cap_layers: Option<[i32; 5]>, cap_rows: usize) -> Result<()> {
        let p = cap_layers.as_ref().map(|c| c.as_ptr()).unwrap_or(std::ptr::null());
        check(unsafe { ak_m1_spec_init(bi as c_int, p, cap_rows as c_int) }, "m1_spec_init")
    }
    /// Verify `tokens` = [bonus, drafts..] (<= 8) at the current position: commits n tokens, returns the n emitted
    /// tokens (accepted drafts + the target's next token) and the target's greedy token per verified position.
    pub fn verify(tokens: &[i32]) -> Result<(Vec<i32>, Vec<i32>)> {
        crate::tp::fwd(&format!("verify {}", crate::tp::join(tokens)));
        let (mut em, mut tg, mut n) = (vec![0i32; 8], vec![0i32; 8], 0i32);
        check(unsafe { ak_m1_verify(tokens.as_ptr(), tokens.len() as c_int, em.as_mut_ptr(), &mut n, tg.as_mut_ptr()) },
              "m1_verify")?;
        em.truncate(n as usize);
        tg.truncate(tokens.len());
        Ok((em, tg))
    }
    pub fn draft_register(name: &str, ptr: *mut c_void) -> Result<()> {
        let c = CString::new(name)?;
        check(unsafe { ak_m1_draft_register(c.as_ptr(), ptr) }, "m1_draft_register")
    }
    /// A drafter linear as W4 (k-tiled, w4_upload_kt), under the name of its bf16 weight.
    pub fn draft_register_wq(name: &str, wt: *mut c_void, st: *mut c_void, n: usize, k: usize, group: usize, bits: usize)
                             -> Result<()> {
        let c = CString::new(name)?;
        check(unsafe { ak_m1_draft_register_wq(c.as_ptr(), wt, st, n as i32, k as i32, group as i32, bits as i32) },
              "m1_draft_register_wq")
    }
    pub fn draft_register_w4(name: &str, wt: *mut c_void, st: *mut c_void, n: usize, k: usize, group: usize) -> Result<()> {
        let c = CString::new(name)?;
        check(unsafe { ak_m1_draft_register_w4(c.as_ptr(), wt, st, n as i32, k as i32, group as i32) }, "m1_draft_register_w4")
    }
    /// After the drafter weights and spec_init (capture layers): -> device bytes the drafter allocated (KV ring etc.).
    pub fn draft_setup(mask_token_id: i32) -> Result<usize> {
        let b = unsafe { ak_m1_draft_setup(mask_token_id) };
        if b < 0 { anyhow::bail!("m1_draft_setup failed") }
        Ok(b as usize)
    }
    /// MTP head weights (docs/mtp.md): bf16 device pointer under its name without the "mtp." prefix.
    pub fn mtp_register(name: &str, ptr: *mut c_void) -> Result<()> {
        let c = CString::new(name)?;
        check(unsafe { ak_m1_mtp_register(c.as_ptr(), ptr) }, "m1_mtp_register")
    }
    pub fn mtp_register_wq(name: &str, wt: *mut c_void, st: *mut c_void, n: usize, k: usize, group: usize, bits: usize)
                           -> Result<()> {
        let c = CString::new(name)?;
        check(unsafe { ak_m1_mtp_register_wq(c.as_ptr(), wt, st, n as i32, k as i32, group as i32, bits as i32) },
              "m1_mtp_register_wq")
    }
    /// Before mtp_setup: the lm_head rows the MTP chain scores (with the INT8 head; empty: the whole head).
    pub fn mtp_shortlist(ids: &[i32]) -> Result<()> {
        check(unsafe { ak_m1_mtp_shortlist(ids.as_ptr(), ids.len() as c_int) }, "m1_mtp_shortlist")
    }
    /// After the MTP weights and spec_init: context buffers for `max_rows` rows per call, attention window. -> bytes.
    pub fn mtp_setup(max_rows: usize, window: usize) -> Result<usize> {
        let b = unsafe { ak_m1_mtp_setup(max_rows as c_int, window as c_int) };
        if b < 0 { anyhow::bail!("m1_mtp_setup failed") }
        Ok(b as usize)
    }
    /// Sampled speculative accept (target-only rule over the filtered distribution: temperature, top-k (0 = all),
    /// top-p, min-p as the host sampler applies them): temperature 0 = greedy; Philox seed.
    pub fn spec_sampling(temperature: f32, seed: i32) -> Result<()> {
        spec_sampling_filtered(temperature, 0, 1.0, 0.0, seed)
    }
    pub fn spec_sampling_filtered(temperature: f32, top_k: i32, top_p: f32, min_p: f32, seed: i32) -> Result<()> {
        crate::tp::fwd(&format!("spec_sampling {temperature} {seed} {top_k} {top_p} {min_p}"));
        check(unsafe { ak_m1_spec_sampling(temperature, top_k, top_p, min_p, seed) }, "m1_spec_sampling")
    }
    /// Tensor parallel: this process is rank `rank` of `size` (before init).
    pub fn tp_config(rank: usize, size: usize) -> Result<()> {
        check(unsafe { ak_m1_tp_config(rank as c_int, size as c_int) }, "m1_tp_config")
    }
    /// All-reduce exchange region: device memory with a 64-byte CUDA IPC handle.
    pub fn tp_region_alloc(bytes: usize) -> Result<(*mut c_void, [u8; 64])> {
        let (mut p, mut h) = (std::ptr::null_mut(), [0u8; 64]);
        check(unsafe { ak_tp_region_alloc(bytes, &mut p, h.as_mut_ptr() as *mut c_void) }, "tp_region_alloc")?;
        Ok((p, h))
    }
    pub fn tp_region_open(h: &[u8; 64]) -> Result<*mut c_void> {
        let mut p = std::ptr::null_mut();
        check(unsafe { ak_tp_region_open(h.as_ptr() as *const c_void, &mut p) }, "tp_region_open")?;
        Ok(p)
    }
    /// All-reduce exchange region in POSIX shared memory (GPUs without peer access).
    pub fn tp_region_shm(name: &str, bytes: usize, create: bool) -> Result<*mut c_void> {
        let c = std::ffi::CString::new(name)?;
        let mut p = std::ptr::null_mut();
        check(unsafe { ak_tp_region_shm(c.as_ptr(), bytes, create as c_int, &mut p) }, "tp_region_shm")?;
        Ok(p)
    }
    pub fn tp_can_peer(dev: usize, peer: usize) -> bool {
        unsafe { ak_tp_can_peer(dev as c_int, peer as c_int) != 0 }
    }
    pub fn tp_setup(rank: usize, regions: &[*mut c_void], slot_elems: usize) -> Result<()> {
        check(unsafe { ak_tp_setup(rank as c_int, regions.len() as c_int, regions.as_ptr(), slot_elems as i64) }, "tp_setup")
    }
    /// Mean microseconds per all-reduce of n elements (every rank must call it with the same arguments).
    pub fn tp_bench(iters: usize, n: usize) -> f64 {
        unsafe { ak_tp_bench(iters as c_int, n as i64) }
    }
    /// Push all-reduce vs the pull kernel and fused AR + add + RMSNorm vs the two calls: differing elements (-1: no push path).
    pub fn tp_push_selftest(iters: usize) -> i64 {
        unsafe { ak_tp_push_selftest(iters as c_int) }
    }
    /// Mean us per call of `which` (0 pull AR, 1 push AR, 2 pull AR + add_rmsnorm, 3 fused) on `rows` rows, graphs or not.
    pub fn tp_bench2(iters: usize, rows: usize, which: i32, graph: bool) -> f64 {
        unsafe { ak_tp_bench2(iters as c_int, rows as i64, which as c_int, graph as c_int) }
    }
    pub fn tp_selftest(iters: usize, n: usize) -> i64 {
        unsafe { ak_tp_selftest(iters as c_int, n as i64) }
    }
    pub fn tp_region_bytes(slot_elems: usize) -> usize {
        unsafe { ak_tp_region_bytes(slot_elems as i64) }
    }
    /// One speculative step with the drafter: -> (emitted tokens, the verified chain [bonus, 7 drafts]).
    pub fn spec_step(bonus: i32) -> Result<(Vec<i32>, Vec<i32>)> {
        crate::tp::fwd(&format!("spec_step {bonus}"));
        let (mut em, mut dr, mut n) = (vec![0i32; 8], vec![0i32; 8], 0i32);
        check(unsafe { ak_m1_spec_step(bonus, em.as_mut_ptr(), &mut n, dr.as_mut_ptr()) }, "m1_spec_step")?;
        em.truncate(n as usize);
        Ok((em, dr))
    }
    /// The same with a host proposal (hybrid drafting): `hint` (<= 7 tokens) replaces the drafter's first drafts.
    pub fn spec_step_hint(bonus: i32, hint: &[i32]) -> Result<(Vec<i32>, Vec<i32>)> {
        crate::tp::fwd(&format!("spec_step_hint {bonus} {}", crate::tp::join(hint)));
        let (mut em, mut dr, mut n) = (vec![0i32; 8], vec![0i32; 8], 0i32);
        let nh = hint.len().min(7) as i32;
        check(unsafe { ak_m1_spec_step_hint(bonus, hint.as_ptr(), nh, em.as_mut_ptr(), &mut n, dr.as_mut_ptr()) },
              "m1_spec_step_hint")?;
        em.truncate(n as usize);
        Ok((em, dr))
    }
    /// Batched speculative decoding (server mode): buffers for up to max_b sequences per step (after spec_init,
    /// before draft_setup).
    pub fn spec_batch_init(max_b: usize) -> Result<()> {
        check(unsafe { ak_m1_spec_batch_init(max_b as c_int) }, "m1_spec_batch_init")
    }
    pub fn spec_batch_max() -> usize {
        unsafe { ak_m1_spec_batch_max() }.max(0) as usize
    }
    /// The sampling of a session's batched verify steps (spec_batch): temperature 0 = greedy, else the filtered
    /// distribution (top-k 0 = all, top-p, min-p); its own Philox stream (seed, steps from 0).
    pub fn session_sampling(sess: i32, temperature: f32, top_k: i32, top_p: f32, min_p: f32, seed: i32) -> Result<()> {
        crate::tp::fwd(&format!("session_sampling {sess} {temperature} {top_k} {top_p} {min_p} {seed}"));
        check(unsafe { ak_m1_session_sampling(sess, temperature, top_k, top_p, min_p, seed) }, "m1_session_sampling")
    }
    /// Penalties of a session's speculative accept (single-sequence and batched verify), applied on the device exactly
    /// as the host sampler does: repetition / frequency / presence over the tokens generated so far (the counts start
    /// at 0 here; `session_penalty_add` adds every emitted token) and logit bias (distinct ids). All neutral = off.
    /// One device per process (Err otherwise); under --tp every rank keeps the same state (forwarded; the logits are
    /// replicated, so each rank's accept draws the same tokens).
    pub fn session_penalty(sess: i32, rep: f32, freq: f32, pres: f32, bias: &[(u32, f32)]) -> Result<()> {
        let ids: Vec<c_int> = bias.iter().map(|b| b.0 as c_int).collect();
        let vals: Vec<f32> = bias.iter().map(|b| b.1).collect();
        // f32 Display is the shortest round-trip form: the followers parse the same bits
        crate::tp::fwd(&format!("session_penalty {sess} {rep} {freq} {pres} {} {}", crate::tp::join(&ids),
                                if vals.is_empty() { "-".into() } else { vals.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",") }));
        check(unsafe { ak_m1_session_penalty(sess, rep, freq, pres, ids.as_ptr(), vals.as_ptr(), ids.len() as c_int) },
              "m1_session_penalty")
    }
    /// Device sampling of the last decode step's rows (row b = sessions[b]; a bs = 1 decode: one row): the rows with
    /// use_row[b] draw from their session's filtered, penalised distribution (session_sampling / session_penalty) with the
    /// speculative accept's kernels; -> the token per row (-1 where not used). Forwarded under --tp (every rank advances
    /// the same sample streams).
    pub fn sample_decode(sessions: &[i32], use_row: &[bool]) -> Result<Vec<i32>> {
        let u: Vec<c_int> = use_row.iter().map(|&x| x as c_int).collect();
        crate::tp::fwd(&format!("sample_decode {} {}", crate::tp::join(sessions), crate::tp::join(&u)));
        let mut out = vec![-1; sessions.len()];
        check(unsafe { ak_m1_sample_decode(sessions.as_ptr(), u.as_ptr(), sessions.len() as c_int, out.as_mut_ptr()) },
              "m1_sample_decode")?;
        Ok(out)
    }
    /// The penalty counts of a session += these emitted tokens.
    pub fn session_penalty_add(sess: i32, ids: &[u32]) -> Result<()> {
        if crate::tp::active() {
            let v: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
            crate::tp::fwd(&format!("session_penalty_add {sess} {}", crate::tp::join(&v)));
        }
        check(unsafe { ak_m1_session_penalty_add(sess, ids.as_ptr() as *const c_int, ids.len() as c_int) },
              "m1_session_penalty_add")
    }
    /// One batched step: sequence b = session `sessions[b]` drafts after `bonus[b]`, verifies, commits; -> the emitted
    /// tokens of each (each exactly its own spec_step).
    /// AGENS_TP_SPEC_LOG=1: every batched verify's emitted tokens per session, numbered, on stderr (each tensor-parallel
    /// rank computes them; the ranks' logs must agree line for line)
    fn spec_log(kind: &str, sessions: &[i32], em: &[i32], n: &[i32]) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static K: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if !*ON.get_or_init(|| std::env::var("AGENS_TP_SPEC_LOG").map_or(false, |v| v == "1")) {
            return;
        }
        let k = K.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let rows: Vec<String> = sessions.iter().enumerate()
            .map(|(i, s)| format!("{s}:{:?}", &em[8 * i..8 * i + n[i] as usize])).collect();
        eprintln!("tp-spec {k} {kind} {}", rows.join(" "));
    }
    pub fn spec_batch(sessions: &[i32], bonus: &[i32]) -> Result<Vec<Vec<i32>>> {
        crate::tp::fwd(&format!("spec_batch {} {}", crate::tp::join(sessions), crate::tp::join(bonus)));
        assert_eq!(sessions.len(), bonus.len());
        let b = sessions.len();
        let (mut em, mut n) = (vec![0i32; 8 * b], vec![0i32; b]);
        check(unsafe { ak_m1_spec_batch(sessions.as_ptr(), b as c_int, bonus.as_ptr(), em.as_mut_ptr(), n.as_mut_ptr()) },
              "m1_spec_batch")?;
        spec_log("b", sessions, &em, &n);
        Ok((0..b).map(|i| em[8 * i..8 * i + n[i] as usize].to_vec()).collect())
    }
    /// The same with host proposals (hybrid drafting): hints[i] (<= 7 tokens, may be empty) replaces sequence i's
    /// first drafts.
    pub fn spec_batch_hint(sessions: &[i32], bonus: &[i32], hints: &[Vec<i32>]) -> Result<Vec<Vec<i32>>> {
        crate::tp::fwd(&format!("spec_batch_hint {} {} {}", crate::tp::join(sessions), crate::tp::join(bonus),
                                hints.iter().map(|h| crate::tp::join(h)).collect::<Vec<_>>().join("/")));
        assert_eq!(sessions.len(), bonus.len());
        assert_eq!(sessions.len(), hints.len());
        let b = sessions.len();
        let (mut em, mut n) = (vec![0i32; 8 * b], vec![0i32; b]);
        let mut hv = vec![0i32; 7 * b];
        let nh: Vec<i32> = hints.iter().map(|h| h.len().min(7) as i32).collect();
        for (i, h) in hints.iter().enumerate() {
            for (t, &x) in h.iter().take(7).enumerate() {
                hv[7 * i + t] = x;
            }
        }
        check(unsafe { ak_m1_spec_batch_hint(sessions.as_ptr(), b as c_int, bonus.as_ptr(), hv.as_ptr(), nh.as_ptr(),
                                             em.as_mut_ptr(), n.as_mut_ptr()) }, "m1_spec_batch_hint")?;
        spec_log("h", sessions, &em, &n);
        Ok((0..b).map(|i| em[8 * i..8 * i + n[i] as usize].to_vec()).collect())
    }
    /// Start/stop the CUDA profiler capture range (nsys --capture-range=cudaProfilerApi).
    pub fn profiler(on: bool) -> Result<()> {
        crate::tp::fwd(&format!("profiler {}", on as i32));
        check(unsafe { ak_m1_profiler(on as c_int) }, "m1_profiler")
    }
    /// mixers: one char per layer ('k' KDA, 'b' BCSA, 'd' dense); engram_layers: the two Engram layers (-1 = none).
    /// devs: CUDA devices (1 or 2); with 2, layers [0, split) run on devs[0] and [split, n) on devs[1].
    /// pool_rows: KV rows shared by all sessions (0 = wctx); nslots: max live sessions (0 = 1). Session 0 is active.
    pub fn init(wctx: usize, mixers: &str, engram_layers: [i32; 2], engram: [*mut c_void; 4], devs: &[i32],
                split: usize, pool_rows: usize, nslots: usize) -> Result<()> {
        let m = CString::new(mixers)?;
        check(unsafe { ak_m1_init(wctx as i32, m.as_ptr(), engram_layers[0], engram_layers[1], engram[0], engram[1],
                                  engram[2], engram[3], devs.len() as i32, devs.as_ptr(), split as i32, pool_rows as i32,
                                  nslots as i32) }, "m1_init")
    }
    pub fn reset() -> Result<()> {
        crate::tp::fwd("reset");
        check(unsafe { ak_m1_reset() }, "m1_reset")
    }
    pub fn prefill(ids: &[i32]) -> Result<i32> {
        crate::tp::fwd(&format!("prefill {}", crate::tp::join(ids)));
        let mut next = 0;
        check(unsafe { ak_m1_prefill(ids.as_ptr(), ids.len() as i32, &mut next) }, "m1_prefill")?;
        Ok(next)
    }
    /// Prefill with image inputs: `rpos3` = [3][n] rope positions (MRoPE rows T, H, W) or None
    /// (position + the session's rope delta); `emb` = image embedding rows ([rows][5120] bf16, host), `emb_row[t]` =
    /// the row replacing token t's embedding (-1 = its own). One device, no tensor parallelism.
    pub fn prefill_mm(ids: &[i32], rpos3: Option<&[i64]>, emb: Option<(*const c_void, &[i32])>) -> Result<i32> {
        if crate::tp::active() {
            // tensor parallel: the vision tower runs on the leader only; the followers get the same chunk (ids, rope
            // positions, the embedding rows it uses) through a /dev/shm blob (embeddings are replicated on every rank)
            crate::tp::fwd(&format!("prefill_mm {}", crate::tp::put_blob(&mm_blob(ids, rpos3, emb))?));
        }
        if let Some(r) = rpos3 {
            assert_eq!(r.len(), 3 * ids.len());
        }
        if let Some((_, rows)) = emb {
            assert_eq!(rows.len(), ids.len());
        }
        let mut next = 0;
        let (ep, er) = match emb {
            Some((p, rows)) => (p, rows.as_ptr()),
            None => (std::ptr::null(), std::ptr::null()),
        };
        check(unsafe { ak_m1_prefill_mm(ids.as_ptr(), ids.len() as i32, rpos3.map(|r| r.as_ptr()).unwrap_or(std::ptr::null()),
                                        ep, er, &mut next) }, "m1_prefill_mm")?;
        Ok(next)
    }
    /// One prefill_mm call as bytes: n, has_rpos, n_emb_rows (i32 each), ids, rpos3 (i64, 3n), rows (i32, n; remapped to
    /// 0..n_emb_rows), the used embedding rows (bf16, 5120 each).
    pub fn mm_blob(ids: &[i32], rpos3: Option<&[i64]>, emb: Option<(*const c_void, &[i32])>) -> Vec<u8> {
        const HID: usize = 5120;
        let n = ids.len();
        let mut used: Vec<i32> = Vec::new();
        let mut rows: Vec<i32> = Vec::new();
        if let Some((_, r)) = emb {
            let mut map = std::collections::HashMap::new();
            for &x in r {
                rows.push(if x < 0 { -1 } else { *map.entry(x).or_insert_with(|| { used.push(x); used.len() as i32 - 1 }) });
            }
        }
        let mut b = Vec::with_capacity(12 + 4 * n + 24 * n + 4 * n + used.len() * HID * 2);
        for v in [n as i32, rpos3.is_some() as i32, if emb.is_some() { used.len() as i32 } else { -1 }] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        ids.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
        if let Some(r) = rpos3 {
            r.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
        }
        if let Some((p, _)) = emb {
            rows.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
            for &u in &used {
                let src = unsafe { std::slice::from_raw_parts((p as *const u8).add(u as usize * HID * 2), HID * 2) };
                b.extend_from_slice(src);
            }
        }
        b
    }
    /// A follower's prefill_mm from a blob written by mm_blob.
    pub fn prefill_mm_blob(b: &[u8]) -> Result<i32> {
        let i32_at = |o: usize| i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let (n, has_r, ne) = (i32_at(0) as usize, i32_at(4) != 0, i32_at(8));
        let mut o = 12;
        let ids: Vec<i32> = (0..n).map(|i| i32_at(o + 4 * i)).collect();
        o += 4 * n;
        let rpos: Option<Vec<i64>> = has_r.then(|| {
            let v = (0..3 * n).map(|i| i64::from_le_bytes(b[o + 8 * i..o + 8 * i + 8].try_into().unwrap())).collect();
            o += 24 * n;
            v
        });
        let (rows, emb): (Vec<i32>, Vec<u8>) = if ne >= 0 {
            let r = (0..n).map(|i| i32_at(o + 4 * i)).collect();
            o += 4 * n;
            (r, b[o..].to_vec())
        } else {
            (Vec::new(), Vec::new())
        };
        prefill_mm(&ids, rpos.as_deref(), if ne >= 0 { Some((emb.as_ptr() as *const c_void, &rows[..])) } else { None })
    }
    /// The active session's rope delta (rope position - token position of its next text token).
    pub fn rope_delta() -> i32 {
        unsafe { ak_m1_rope_delta() }
    }
    pub fn set_rope_delta(d: i32) -> Result<()> {
        crate::tp::fwd(&format!("set_rope_delta {d}"));
        check(unsafe { ak_m1_set_rope_delta(d) }, "m1_set_rope_delta")
    }
    /// A vision tower tensor (bf16, row-major) in pinned host memory that stays alive for the process.
    pub fn vis_register(name: &str, host: *mut c_void, shape: &[usize]) -> Result<()> {
        let n = CString::new(name)?;
        let sh: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        check(unsafe { ak_m1_vis_register(n.as_ptr(), host, sh.as_ptr(), sh.len() as c_int) }, "m1_vis_register")
    }
    pub fn vis_ready() -> bool {
        unsafe { ak_m1_vis_ready() != 0 }
    }
    /// One image: RGB8 h x w -> resized rh x rw (multiples of 32) -> (rh / 32) * (rw / 32) embeddings of 5120 bf16 into
    /// `out` (host, pinned or not).
    /// `jpeg`: the encoded JPEG, decoded on the GPU with nvJPEG when it can (the reference's path), else `rgb` is used.
    pub fn vis_encode(rgb: &[u8], jpeg: Option<&[u8]>, h: usize, w: usize, rh: usize, rw: usize, out: *mut c_void)
                      -> Result<()> {
        assert_eq!(rgb.len(), h * w * 3);
        let (jp, jl) = jpeg.map(|j| (j.as_ptr(), j.len() as i64)).unwrap_or((std::ptr::null(), 0));
        check(unsafe { ak_m1_vis_encode(rgb.as_ptr(), jp, jl, h as c_int, w as c_int, rh as c_int, rw as c_int, out) },
              "m1_vis_encode")
    }
    /// The preprocessed pixel values of one image ([(rh / 16) * (rw / 16)][1536] fp32).
    pub fn vis_pixels(rgb: &[u8], jpeg: Option<&[u8]>, h: usize, w: usize, rh: usize, rw: usize) -> Result<Vec<f32>> {
        assert_eq!(rgb.len(), h * w * 3);
        let mut out = vec![0f32; (rh / 16) * (rw / 16) * 1536];
        let (jp, jl) = jpeg.map(|j| (j.as_ptr(), j.len() as i64)).unwrap_or((std::ptr::null(), 0));
        check(unsafe { ak_m1_vis_pixels(rgb.as_ptr(), jp, jl, h as c_int, w as c_int, rh as c_int, rw as c_int,
                                        out.as_mut_ptr()) }, "m1_vis_pixels")?;
        Ok(out)
    }
    /// The tower on given pixel values ([gh * gw][1536] fp32) -> [gh * gw / 4][5120] bf16.
    pub fn vis_encode_pixels(pv: &[f32], gh: usize, gw: usize) -> Result<Vec<u16>> {
        assert_eq!(pv.len(), gh * gw * 1536);
        let mut out = vec![0u16; gh * gw / 4 * 5120];
        check(unsafe { ak_m1_vis_encode_pixels(pv.as_ptr(), gh as c_int, gw as c_int, out.as_mut_ptr() as *mut c_void) },
              "m1_vis_encode_pixels")?;
        Ok(out)
    }
    /// Peak bytes torch's allocator reserved since the last call (resets the peak).
    pub fn vis_peak_bytes() -> i64 {
        unsafe { ak_m1_vis_peak_bytes() }
    }
    pub fn decode(tok: i32) -> Result<i32> {
        crate::tp::fwd(&format!("decode {tok}"));
        let mut next = 0;
        check(unsafe { ak_m1_decode(tok, &mut next) }, "m1_decode")?;
        Ok(next)
    }
}
