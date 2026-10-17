# M1 parity bridge built against libtorch in its OWN project: find_package(Torch) rewrites global CUDA arch flags,
# so it must not share a scope with the engine kernels. Built by kernels/CMakeLists.txt via ExternalProject-free
# add_subdirectory-less custom command (see AGENS_TORCH_PREFIX there).
cmake_minimum_required(VERSION 3.24)
project(agens_oracle_bridge LANGUAGES CXX)
find_package(Torch REQUIRED)
add_library(agens_oracle_bridge SHARED far_bridge.cpp)
target_link_libraries(agens_oracle_bridge PUBLIC "${TORCH_LIBRARIES}")
set_property(TARGET agens_oracle_bridge PROPERTY CXX_STANDARD 17)
