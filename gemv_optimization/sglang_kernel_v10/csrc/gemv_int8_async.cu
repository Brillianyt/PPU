// gemv_int8_async.cu — Phase 3.2 baseline (v10 + per-row scale)
//
// INT8 GEMV for sglang decode path on PPU001 (T-Head SAIL ISA).
//   K=2048, M=1, NUM_UNROLL=4, NPerThread=4, BlockSize=256
//
// Per-row scale handling:
//   out[n] = (sum_k A[k] * W[n,k]) * As[0] * Bs[n]
//   (MVP: only As[0] * Bs[n] applied by lane 0; As passed as fp32 scalar)
//
// Layout (each lane handles NUM_UNROLL × 16 bytes contiguous):
//   - lane k loads A[k*64 .. k*64+63]
//   - col-stride is 1 within NPerThread (not NLoopStep=8)
//
// What is intentionally NOT here:
//   - No <torch/extension.h> include (would conflict with g++ 13 cmath
//     constexpr vs HGGC __device__ isinf/isnan on PPU nvcc 13).
//   - No HGGC mma path (deep_gemm itself skips mma for m=1; see Phase 3.1
//     failure notes in docs/02_ARCHITECTURE.md).
//   - No cp.async double-buffer (planned for Phase 3.2.1; current file is
//     the v10 baseline at 6.7μs / 4.5× vs deep_gemm).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// NOTE: do NOT include <hgrt/hggc_extend_device_functions.h> here.
// It conflicts with g++ 13 cmath constexpr (isgreaterequal, etc.) when
// inlined under nvcc 13. The v10 kernel uses only:
//
//   - CUDA builtins (__shfl_sync, __float2bfloat16, int4)
//   - SAIL cp.async PTX (inline asm, not from HGGC headers)
//
// so we don't need any HGGC declarations. The HGGC mma path was tried in
// Phase 3.1 and FAILED for m=1 GEMV (see docs/02_ARCHITECTURE.md).

#include "binding.h"

// =============================================================================
// TIX cp.async helpers (copied from deep_gemm; ready for Phase 3.2.1 use)
// =============================================================================

__device__ __forceinline__
uint32_t smem_u32_addr(const void *smemptr) {
    uint32_t u32addr;
    asm volatile(
        "{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(u32addr) : "l"(smemptr)
    );
    return u32addr;
}

__device__ __forceinline__
void ldg_sts_128(const void* smem_ptr, const void* gmem_ptr, bool guard = true) {
    uint32_t smem_addr = smem_u32_addr(smem_ptr);
    asm volatile(
        "{.reg.pred p;\n"
        " ppu.cmpp.ne.b32 p, %2, 0;\n"
        " @p ppu.cp.async.cg.shared.global [%0], [%1], 16;}\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr), "r"((int)guard)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("ppu.cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("ppu.cp.async.wait_group %0;\n" : : "n"(N));
}

// =============================================================================
// Main kernel — body in shared header (used by both v10 .cu and persistent .cu)
// =============================================================================

#include "gemv_int8_async_impl.h"

// =============================================================================
// __global__ entry — explicit instantiation for BlockSize=256, NPerThread=4, NUM_UNROLL=4
// =============================================================================

__global__ void gemv_int8_async_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ W,
    __nv_bfloat16* __restrict__ out,
    int K, int N,
    const float* __restrict__ a_scale,      // As — for MVP M=1 we read [0]
    const float* __restrict__ b_scale       // Bs[N] — per-row weight scale
) {
    // Read As[0] on device (avoids a host<->device sync per call)
    float a_s = __ldg(a_scale);
    gemv_int8_async_impl<256, 4, 4>(A, W, out, K, N, a_s, b_scale);
}

// =============================================================================
// C-style launcher (called from C++ wrapper)
// =============================================================================

extern "C" void launch_gemv_int8_async(
    const int8_t* A,
    const int8_t* W,
    __nv_bfloat16* out,
    int M, int K, int N,
    const float* As,
    const float* Bs,
    cudaStream_t stream
) {
    constexpr int BlockSize = 256;
    constexpr int NPerThread = 4;
    constexpr int NLoopStep = BlockSize / 32;  // 8 warps
    constexpr int NPerBlock = NPerThread * NLoopStep;
    int n_blocks = (N + NPerBlock - 1) / NPerBlock;

    (void)M;  // unused in MVP (M=1)

    // Pass As pointer; kernel reads As[0] on device via __ldg.
    // (Reading it on host would segfault — As.data_ptr<float>() returns a
    // GPU pointer, not a host pointer.)
    gemv_int8_async_kernel<<<n_blocks, BlockSize, 0, stream>>>(
        A, W, out, K, N, As, Bs
    );
}