// With `--features m1` the binary links the CMake-built libagens_m1.so / libagens_kernels_py.so; record their directory
// as the binary's rpath so it runs without LD_LIBRARY_PATH (a dependency's build script cannot set link args for us).
fn main() {
    println!("cargo:rerun-if-env-changed=AGENS_KERNELS_BUILD");
    if std::env::var_os("CARGO_FEATURE_M1").is_none() {
        return;
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/build");
    let dir = std::env::var("AGENS_KERNELS_BUILD").unwrap_or_else(|_| root.to_string_lossy().into_owned());
    // MSVC's linker has no rpath concept (DLLs resolve from PATH/side-by-side); `-Wl` link args do not apply to it, so only
    // pass the GNU-style rpath when linking on a non-Windows target.
    if !std::env::var("TARGET").unwrap_or_default().contains("-windows-") {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{dir}");
    }
}
