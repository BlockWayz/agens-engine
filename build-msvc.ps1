# Activate MSVC environment
$vsPath = "C:\Program Files\Microsoft Visual Studio\18\Community"
Import-Module (Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -Arch x64

# libtorch standalone zip (no Python/venv needed): download once; find_package(Torch) works from its share/cmake config.
$TORCH_ROOT = "$PSScriptRoot\third_party\libtorch"
if (-not (Test-Path $TORCH_ROOT)) {
  $zip = Join-Path (Split-Path $TORCH_ROOT -Parent) "libtorch-cu132.zip"
  Invoke-WebRequest -Uri "https://download.pytorch.org/libtorch/cu132/libtorch-win-shared-with-deps-2.14.1%2Bcu132.zip" -OutFile $zip
  Expand-Archive -LiteralPath $zip -DestinationPath (Split-Path $TORCH_ROOT -Parent)
  Move-Item (Get-ChildItem (Split-Path $TORCH_ROOT -Parent) | Where-Object { $_.PSIsContainer -and $_.Name -like "libtorch*" }).FullName $TORCH_ROOT
  Remove-Item -Path $zip
}

# CUDA_HOME for the Rust build scripts (they link cudart from ${CUDA_HOME}/bin)
$nvcc = Get-Command nvcc -ErrorAction Stop | Select-Object -First 1
$env:CUDA_HOME = Split-Path (Split-Path $nvcc.Source -Parent) -Parent
Write-Host "CUDA_HOME: $env:CUDA_HOME"

 # Configure
cmake -G "Ninja" -S kernels -B kernels/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86;89;90;120" -DAGENS_BUILD_ORACLE=ON `
    -DAGENS_TORCH_PREFIX="$TORCH_ROOT"
if ($LASTEXITCODE -ne 0) { exit 1 }

# Build
cmake --build kernels/build --parallel
if ($LASTEXITCODE -ne 0) { exit 1 }
# Build server and CLI (m1 feature exists only on agens-cli). AGENS_TORCH_PREFIX lets agens-cli's build script
# deploy the libtorch runtime DLLs next to agens.exe on Windows.
$env:AGENS_TORCH_PREFIX = "$TORCH_ROOT"
cargo build --release -p agens-server
if ($LASTEXITCODE -ne 0) { exit 1 }
cargo build --release -p agens-cli --features m1
if ($LASTEXITCODE -ne 0) { exit 1 }
