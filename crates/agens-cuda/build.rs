// With `--features cuda`, builds kernels/ through CMake and links libagens_kernels + cudart.
// AGENS_CUDA_ARCHS overrides the target list (default "89;90;120": sm89, sm90, sm120).
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-env-changed=AGENS_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=AGENS_KERNELS_BUILD");
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    if std::env::var_os("CARGO_FEATURE_M1").is_some() {
        // M1: link the ONE CMake build of the kernels (the same objects libagens_m1.so contains). Building a second
        // copy here caused symbol interposition: the executable's stale ak_* overrode the .so's at load time.
        let dir = std::env::var("AGENS_KERNELS_BUILD").unwrap_or_else(|_| root.join("build").to_string_lossy().into_owned());
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=agens_m1");
        println!("cargo:rustc-link-lib=dylib=agens_kernels_py");
        println!("cargo:rustc-link-search=native={cuda}/lib64");
        println!("cargo:rustc-link-lib=dylib=cudart");
        return;
    }
    let archs = std::env::var("AGENS_CUDA_ARCHS").unwrap_or_else(|_| "89;90;120".into());
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("kernels");
    std::fs::create_dir_all(&out).unwrap();
    let run = |args: &[&str]| {
        let st = std::process::Command::new("cmake").args(args).status().expect("cmake not found");
        assert!(st.success(), "cmake {args:?} failed");
    };
    run(&[
        "-S",
        root.to_str().unwrap(),
        "-B",
        out.to_str().unwrap(),
        "-DCMAKE_BUILD_TYPE=Release",
        &format!("-DCMAKE_CUDA_ARCHITECTURES={archs}"),
        "-DAGENS_BUILD_BENCH=OFF",
        "-DAGENS_BUILD_ORACLE=OFF",
    ]);
    run(&["--build", out.to_str().unwrap(), "--target", "agens_kernels", "-j"]);
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=agens_kernels");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
