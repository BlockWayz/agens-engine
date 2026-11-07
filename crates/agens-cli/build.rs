// With `--features m1` the binary links the CMake-built libagens_m1.so / libagens_kernels_py.so. On Linux, record
// their directory as the binary's rpath so it runs without LD_LIBRARY_PATH (a dependency's build script cannot set
// link args for us). On Windows there is no rpath: deploy the runtime DLLs side-by-side with the binary (the MSVC
// loader's first search directory) — the kernels/bridge DLLs plus the libtorch bin (cudart/cublas/cudnn ship in it).
// MSVC CRT and the CUDA toolkit are machine-wide installs resolved from PATH, the counterpart of ldconfig.
fn main() {
    println!("cargo:rerun-if-env-changed=AGENS_KERNELS_BUILD");
    if std::env::var_os("CARGO_FEATURE_M1").is_none() {
        return;
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/build");
    let dir = std::env::var("AGENS_KERNELS_BUILD").unwrap_or_else(|_| root.to_string_lossy().into_owned());
    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.contains("-windows-") {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{dir}");
        return;
    }
    println!("cargo:rerun-if-changed={dir}");
    // OUT_DIR = <exe_dir>/build/<pkg-hash>/out: strip it to the directory cargo places the binaries in.
    let exe_dir = std::path::Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR"))
        .ancestors().nth(3).expect("OUT_DIR layout").to_path_buf();
    let torch = std::env::var("AGENS_TORCH_PREFIX")
        .unwrap_or_else(|_| root.join("libtorch-win-cu132").to_string_lossy().into_owned());
    for src_dir in [dir.clone(), format!("{dir}/bridge"), format!("{torch}/bin")] {
        let rd = match std::fs::read_dir(&src_dir) {
            Ok(rd) => rd,
            Err(_) => { println!("cargo:warning=runtime DLL deploy: {src_dir} missing (run the CMake build first)"); continue }
        };
        for ent in rd.flatten() {
            let p = ent.path();
            if p.extension().map(|e| e == "dll").unwrap_or(false) {
                let dst = exe_dir.join(p.file_name().unwrap());
                let fresh = std::fs::metadata(&dst).ok().zip(std::fs::metadata(&p).ok())
                    .and_then(|(d, s)| Some(d.modified().ok()?.ge(&s.modified().ok()?))).unwrap_or(false);
                if !fresh && std::fs::copy(&p, &dst).is_err() {
                    println!("cargo:warning=runtime DLL deploy: cannot copy {}", p.display());
                }
            }
        }
    }
}
