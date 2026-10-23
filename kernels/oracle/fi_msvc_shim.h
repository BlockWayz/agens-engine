// CUDA 13.x's MSVC host path parses device function bodies as plain C++, where a GCC function-scope
// __attribute__((...)) in FlashInfer headers is rejected. Force-included into every MSVC TU of this tree;
// on MSVC; harmless no-op elsewhere. GCC/Clang builtin `ushort` is an unknown type spelling to cl (C3689);
#if defined(_MSC_VER)
#define __attribute__(...)
using ushort = unsigned short;
#endif
