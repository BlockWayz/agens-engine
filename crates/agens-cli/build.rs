// With `--features m1` the binary links the CMake-built libagens_m1.so / libagens_kernels_py.so; record their directory
// as the binary's rpath so it runs without LD_LIBRARY_PATH (a dependency's build script cannot set link args for us).
fn main() {
    println!("cargo:rerun-if-env-changed=AGENS_KERNELS_BUILD");
    if std::env::var_os("CARGO_FEATURE_M1").is_none() {
        return;
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/build");
    let dir = std::env::var("AGENS_KERNELS_BUILD").unwrap_or_else(|_| root.to_string_lossy().into_owned());
    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{dir}");
}
