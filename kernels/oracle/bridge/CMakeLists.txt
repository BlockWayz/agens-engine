# M1 parity bridge built against libtorch in its OWN project: find_package(Torch) rewrites global CUDA arch flags,
# so it must not share a scope with the engine kernels. Built by kernels/CMakeLists.txt via ExternalProject-free
# add_subdirectory-less custom command (see AGENS_TORCH_PREFIX there).
cmake_minimum_required(VERSION 3.24)
project(agens_oracle_bridge LANGUAGES CXX)
find_package(Torch REQUIRED)
add_library(agens_oracle_bridge SHARED far_bridge.cpp vision.cpp)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(agens_oracle_bridge PUBLIC "${TORCH_LIBRARIES}" CUDA::nvjpeg)
# libtorch 2.14+ (cu13x) headers require C++20 (std::strong_ordering, designated init in c10); older trees
# built against 17 — pin the standard to the tree in use (Torch_VERSION from TorchConfigVersion.cmake).
if(Torch_VERSION VERSION_GREATER_EQUAL 2.14)
  set_property(TARGET agens_oracle_bridge PROPERTY CXX_STANDARD 20)
else()
  set_property(TARGET agens_oracle_bridge PROPERTY CXX_STANDARD 17)
endif()
# MSVC: the DLL's import library lists only decorated (dllexport) symbols; the m1 C ABI entry points are plain
# extern "C" definitions, so give them an explicit export file (consumed by the /DEF link option below).
if(MSVC)
  target_link_options(agens_oracle_bridge PRIVATE "LINKER:/DEF:${CMAKE_CURRENT_SOURCE_DIR}/oracle_bridge_exports.def")
endif()
