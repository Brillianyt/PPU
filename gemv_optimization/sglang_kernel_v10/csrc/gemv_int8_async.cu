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
// Main kernel
// =============================================================================

template <int BlockSize, int NPerThread, int NUM_UNROLL>
__device__ __forceinline__ void gemv_int8_async_impl(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ W,
    __nv_bfloat16* __restrict__ out,
    int K, int N,
    float a_scale,                    // As[0] — scalar (MVP supports M=1)
    const float* __restrict__ b_scale // Bs[N] — per-row weight scale
) {
    constexpr int ThreadPerN = 32;
    constexpr int NLoopStep = BlockSize / ThreadPerN;        // 8 warps
    constexpr int alignmentB = sizeof(int4) / sizeof(int8_t); // 16
    constexpr int kstep_lane = NUM_UNROLL * alignmentB;        // 64

    int tid = threadIdx.x;
    int lane_k = tid % ThreadPerN;                          // 0..31
    int warp_id = tid / ThreadPerN;                         // 0..7
    int tid_n = __shfl_sync(0xffffffff, warp_id, 0);        // broadcast
    int id_n = blockIdx.x * (NPerThread * NLoopStep) + tid_n * NPerThread;
    if (id_n >= N) return;

    int lane_k_off = lane_k * kstep_lane;

    // Pre-load A vector (each lane holds 64 bytes of A, contiguous)
    int4 vreg_a[NUM_UNROLL];
    #pragma unroll
    for (int u = 0; u < NUM_UNROLL; u++) {
        int kk = lane_k_off + u * alignmentB;
        vreg_a[u] = (kk + alignmentB <= K) ?
            *reinterpret_cast<const int4*>(&A[kk]) : make_int4(0, 0, 0, 0);
    }

    // Main loop over NPerThread cols (each warp processes 4 contiguous cols)
    for (int nloop = 0; nloop < NPerThread; nloop++) {
        int cur_col = id_n + nloop;
        if (cur_col >= N) continue;

        int4 vreg_b[NUM_UNROLL];
        const int8_t* b_row = W + cur_col * K;
        #pragma unroll
        for (int u = 0; u < NUM_UNROLL; u++) {
            int kk = lane_k_off + u * alignmentB;
            vreg_b[u] = (kk + alignmentB <= K) ?
                *(reinterpret_cast<const int4*>(&b_row[kk])) : make_int4(0, 0, 0, 0);
        }

        int32_t local_sum = 0;
        #pragma unroll
        for (int u = 0; u < NUM_UNROLL; u++) {
            const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&vreg_a[u]);
            const int8_t* b_bytes = reinterpret_cast<const int8_t*>(&vreg_b[u]);
            int32_t acc = 0;
            #pragma unroll
            for (int j = 0; j < 16; j++) acc += (int32_t)a_bytes[j] * (int32_t)b_bytes[j];
            local_sum += acc;
        }

        // Warp reduce (32 lanes → 1)
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            local_sum += __shfl_down_sync(0xffffffff, local_sum, off);

        // Lane 0 writes output with per-row scale applied
        if (lane_k == 0) {
            float scaled = (float)local_sum * a_scale * b_scale[cur_col];
            out[cur_col] = __float2bfloat16(scaled);
        }
    }
}

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