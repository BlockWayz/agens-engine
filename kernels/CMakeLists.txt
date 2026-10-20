cmake_minimum_required(VERSION 3.24)
# Targets: sm86 (built, not validated on hardware), sm89, sm90, sm120. All four build by default.
# Must be set before project(): enabling CUDA defines a (sm_52) default otherwise.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 89 90 120)
endif()
project(agens_kernels LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(AGENS_BUILD_BENCH "build microbenchmarks (needs cuBLAS)" ON)

find_package(CUDAToolkit 12.4 REQUIRED)

# --fmad=false is NOT global: bit-exact kernels opt out of contraction per expression (__fmul_rn/__fadd_rn),
# quantised-path kernels keep FMA.
set(AGENS_CUDA_FLAGS --expt-relaxed-constexpr -lineinfo -Xptxas=-v)

set(AGENS_SOURCES
  src/runtime.cu
  src/w4_gemv.cu
  src/w4h.cu
  src/bf16_linear.cu
  src/kda_decode.cu
  src/mhc.cu
  src/engram.cu
  src/rope.cu
  src/torch_ops.cu
  src/rmsnorm.cu
  src/fastmath/silu_mul.cu
  src/kda_prefill.cu
  src/far_decode.cu
  src/tp_ar.cu
  src/bi_gemv.cu
  src/kv_tier.cu
  src/bf16_gemv.cu
  src/i8_gemv.cu
)
# Files under src/fastmath/ mirror sglang kernels that are built with --use_fast_math on sm < 100.
set_source_files_properties(src/fastmath/silu_mul.cu PROPERTIES COMPILE_OPTIONS "--use_fast_math")
add_library(agens_kernels STATIC ${AGENS_SOURCES})
target_include_directories(agens_kernels PUBLIC include)
target_compile_options(agens_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${AGENS_CUDA_FLAGS}>)
target_link_libraries(agens_kernels PUBLIC CUDA::cudart CUDA::cublas CUDA::cuda_driver)

# Shared build of the same kernels for the Python tests (ctypes): tools/test_*.py load libagens_kernels_py.so.
add_library(agens_kernels_py SHARED ${AGENS_SOURCES})
target_include_directories(agens_kernels_py PUBLIC include)
target_compile_options(agens_kernels_py PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${AGENS_CUDA_FLAGS}>)
target_link_libraries(agens_kernels_py PUBLIC CUDA::cudart CUDA::cublas CUDA::cuda_driver)

if(AGENS_BUILD_BENCH)
  add_executable(bench_gemv bench/bench_gemv.cu)
  target_compile_options(bench_gemv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  target_link_libraries(bench_gemv PRIVATE agens_kernels)
  add_executable(test_wq bench/test_wq.cu)
  target_link_libraries(test_wq PRIVATE agens_kernels)
  add_executable(test_kf bench/test_kf.cu)
  target_link_libraries(test_kf PRIVATE agens_kernels)
  add_executable(test_w4 bench/test_w4.cu)
  target_link_libraries(test_w4 PRIVATE agens_kernels)
  add_executable(tune_w4 bench/tune_w4.cu)
  target_link_libraries(tune_w4 PRIVATE agens_kernels)
  add_executable(test_w4h bench/test_w4h.cu)
  target_link_libraries(test_w4h PRIVATE agens_kernels)
  add_executable(bench_w4h bench/bench_w4h.cu)
  target_compile_definitions(bench_w4h PRIVATE AGENS_HAVE_W4H)
  target_link_libraries(bench_w4h PRIVATE agens_kernels)
endif()

if(AGENS_BUILD_BENCH)
  add_executable(bench_read bench/bench_read.cu)
endif()

# ---- ORACLE TREE (tests only): vendored FlashInfer templates, as sglv23's sgl_kernel/flashinfer build them.
option(AGENS_BUILD_ORACLE "build the oracle libraries for parity tests" ON)
if(AGENS_BUILD_ORACLE)
  foreach(variant IN ITEMS plain fastmath)
    add_library(agens_oracle_${variant} SHARED oracle/norm_oracle.cu)
    target_include_directories(agens_oracle_${variant} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/flashinfer/include)
    target_compile_options(agens_oracle_${variant} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr
      -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_F16 $<$<STREQUAL:${variant},fastmath>:--use_fast_math>>)
    target_link_libraries(agens_oracle_${variant} PUBLIC CUDA::cudart)
  endforeach()
  # BCSA local decode: flashinfer's AOT batch_prefill module flags (-use_fast_math -O3 -DNDEBUG, c++17)
  add_library(agens_oracle_attn SHARED oracle/bcsa_local_oracle.cu)
  set(FI ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/flashinfer)
  # flashinfer's vendored CCCL must win over the toolkit's (flashinfer passes it with -I, before the CTK)
  target_include_directories(agens_oracle_attn BEFORE PRIVATE ${FI}/cccl/libcudacxx/include ${FI}/cccl/cub
    ${FI}/cccl/thrust ${FI}/include ${CMAKE_CURRENT_SOURCE_DIR}/oracle)
  target_compile_options(agens_oracle_attn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -use_fast_math
    -DNDEBUG -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2>)
  target_link_libraries(agens_oracle_attn PUBLIC CUDA::cudart)
  # separate .so: the windowed-prefill module defines the same generated types with a different config
  add_library(agens_oracle_prefill SHARED oracle/bcsa_prefill_oracle.cu)
  target_include_directories(agens_oracle_prefill BEFORE PRIVATE ${FI}/cccl/libcudacxx/include ${FI}/cccl/cub
    ${FI}/cccl/thrust ${FI}/include ${CMAKE_CURRENT_SOURCE_DIR}/oracle)
  target_compile_options(agens_oracle_prefill PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -use_fast_math
    -DNDEBUG -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2>)
  target_link_libraries(agens_oracle_prefill PUBLIC CUDA::cudart)
endif()

# ---- M1 PARITY BRIDGE (oracle tree): BCSA far branch through libtorch's own ops, built as a separate CMake project
# (find_package(Torch) would rewrite this project's CUDA flags). Pass -DAGENS_TORCH_PREFIX=<torch cmake prefix>.
if(AGENS_BUILD_ORACLE AND AGENS_TORCH_PREFIX)
  set(BRIDGE_DIR ${CMAKE_BINARY_DIR}/bridge)
  add_custom_command(OUTPUT ${BRIDGE_DIR}/libagens_oracle_bridge.so
    COMMAND ${CMAKE_COMMAND} -S ${CMAKE_CURRENT_SOURCE_DIR}/oracle/bridge -B ${BRIDGE_DIR}
            -DCMAKE_PREFIX_PATH=${AGENS_TORCH_PREFIX} -DCMAKE_BUILD_TYPE=Release
    COMMAND ${CMAKE_COMMAND} --build ${BRIDGE_DIR} -j
    DEPENDS oracle/bridge/far_bridge.cpp oracle/bridge/vision.cpp oracle/bridge/CMakeLists.txt)
  add_custom_target(agens_oracle_bridge_build DEPENDS ${BRIDGE_DIR}/libagens_oracle_bridge.so)
  add_library(agens_oracle_bridge SHARED IMPORTED)
  set_target_properties(agens_oracle_bridge PROPERTIES IMPORTED_LOCATION ${BRIDGE_DIR}/libagens_oracle_bridge.so)
  add_dependencies(agens_oracle_bridge agens_oracle_bridge_build)
  # ---- M1 model (bf16 tp1 standalone forward over the parity bridges)
  add_library(agens_m1 SHARED m1/m1_model.cu m1/m1_kv.cu m1/m1_draft.cu m1/m1_mtp.cu m1/m1_dense8.cu)
  target_link_libraries(agens_m1 PUBLIC agens_kernels agens_oracle_attn agens_oracle_prefill agens_oracle_bridge)
endif()
