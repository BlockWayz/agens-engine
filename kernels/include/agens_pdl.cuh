// Programmatic dependent launch (PDL, compute capability 9.0+) for the engine's graphs.
//
// ak_graph_pdl (runtime.cu) turns a captured graph's kernel -> kernel edges into programmatic edges when the downstream
// kernel is on its list (kPdlKernels): that kernel may then start while its predecessor still runs. Every kernel on the
// list therefore, in every block and on every path:
//   * runs ak_pdl_wait() before it reads memory an earlier kernel of the graph writes, and before it writes memory an
//     earlier kernel reads or writes (griddepcontrol.wait returns once the predecessor grid has completed and its
//     memory is visible; that predecessor waited for its own, so all earlier work is done)
//   * may read before it constant data (weights, scales, norm weights) -- the point of the early start -- and data the
//     immediately preceding kernel does not write: ak_pdl_launch comes after the wait, so when this kernel starts
//     every earlier kernel but that one is complete (e.g. a KDA layer's conv / recurrent state, written by the commits)
//   * calls ak_pdl_launch() only after its wait, so at most one kernel runs ahead of a kernel that is past its wait
// Without the programmatic edge (eager launches, other archs, AGENS_PDL=0) both are no-ops: the same instructions,
// the same results. PDL changes when a kernel starts, never what it computes.
#pragma once

__device__ __forceinline__ void ak_pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void ak_pdl_launch() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
// the common form: wait at the kernel's start, then let the next kernel launch
__device__ __forceinline__ void ak_pdl_entry() {
  ak_pdl_wait();
  ak_pdl_launch();
}
