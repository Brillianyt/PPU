// gemv_int8_async_v2.cu — Phase 3.2.1: cp.async double-buffered GEMV
//
// Same algorithm as v10 (HBM-direct SIMT dot product), BUT:
//   - W data for each col is staged through shared memory via cp.async
//   - 2 stages ping-pong: while computing col N, cp.async loads col N+1
//   - Hides ~400 cycle HBM latency behind compute
//
// Per-warp shmem layout (avoids cross-warp contention):
//   warp_smem[2][NLoopStep][K]
//   Each warp owns its 2 stages × K-byte slice. Total shmem = 32 KB.
//
// Access pattern per lane:
//   For col=N, lane k reads from shmem[stage][warp_id][k*16..k*16+63]
//   (NUM_UNROLL=4 chunks of 16 bytes = 64 bytes per lane)
//
// Key invariants (same as v10):
//   - NUM_UNROLL=4 × ThreadPerN=32 × 16 bytes = 2048 = K ✓
//   - NPerThread=4 cols per warp (contiguous, not strided)
//   - As[0] applied at lane_k==0 write path
//   - .cu file does NOT include <torch/extension.h> (g++ 13 + HGGC constexpr trap)
//
// Differences from v10:
//   - W reads go through shared memory (via cp.async) instead of direct HBM
//   - shmem 32 KB per block (within 256 KB budget)
//
// Build: same nvcc flags as v10 (see setup.py).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "binding.h"

// =============================================================================
// TIX cp.async helpers
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
// Main kernel — cp.async double buffer
// =============================================================================
//
// Per-warp perspective: each warp processes NPerThread cols (contiguous).
// shmem layout: shared[stage][warp_id][k_byte] = int8 byte k_byte of stage
// for warp_id. Stage size = K bytes. 2 stages × 8 warps × K bytes.
//
// Pipeline:
//   Prologue:  load col 0 (each warp's first col) into stage 0
//   Loop nloop in [0, NPerThread):
//     Issue cp.async to load NEXT col into stage (nloop+1) % 2
//     Wait for current stage (wait_group<1> keeps the load we just issued
//       in flight; older loads complete)
//     Compute: read vreg_b from shmem, dot with vreg_a, warp reduce, write
//   Epilogue: wait_group<0>

template <int BlockSize, int NPerThread, int NUM_UNROLL>
__device__ __forceinline__ void gemv_int8_async_v2_impl(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ W,
    __nv_bfloat16* __restrict__ out,
    int K, int N,
    float a_scale,
    const float* __restrict__ b_scale
) {
    constexpr int ThreadPerN = 32;
    constexpr int NLoopStep = BlockSize / ThreadPerN;        // 8 warps
    constexpr int alignmentB = sizeof(int4) / sizeof(int8_t); // 16
    constexpr int kstep_lane = NUM_UNROLL * alignmentB;        // 64

    // Per-warp shmem: [stage][warp_id][K bytes]
    __shared__ int8_t warp_smem[2][NLoopStep][/*K*/2048];

    int tid = threadIdx.x;
    int lane_k = tid % ThreadPerN;
    int warp_id = tid / ThreadPerN;

    // id_n is the first col this warp processes
    int id_n = blockIdx.x * (NPerThread * NLoopStep) + warp_id * NPerThread;
    if (id_n >= N) return;

    int lane_k_off = lane_k * kstep_lane;

    // ---- Pre-load A vector into registers (same as v10) ----
    int4 vreg_a[NUM_UNROLL];
    #pragma unroll
    for (int u = 0; u < NUM_UNROLL; u++) {
        int kk = lane_k_off + u * alignmentB;
        vreg_a[u] = (kk + alignmentB <= K) ?
            *reinterpret_cast<const int4*>(&A[kk]) : make_int4(0, 0, 0, 0);
    }

    // Helper: issue cp.async load of one col into a stage for this warp
    auto issue_load = [&](int col, int stage) {
        const int8_t* src = W + col * K;
        int8_t* dst = warp_smem[stage][warp_id];
        #pragma unroll
        for (int u = 0; u < NUM_UNROLL; u++) {
            int kk = lane_k_off + u * alignmentB;
            bool guard = (col < N) && (kk + alignmentB <= K);
            ldg_sts_128(&dst[kk], &src[kk], guard);
        }
        cp_async_commit();
    };

    // ---- Prologue: load col 0 into stage 0 ----
    if (id_n < N) {
        issue_load(id_n, 0);
        cp_async_wait<0>();  // make sure col 0 is ready before main loop
    }

    // ---- Main loop ----
    for (int nloop = 0; nloop < NPerThread; nloop++) {
        int cur_col = id_n + nloop;
        int cur_stage = nloop & 1;
        int next_col = cur_col + 1;
        bool has_next = (nloop + 1 < NPerThread) && (next_col < N);

        // Issue cp.async for NEXT col into stage (nloop+1)%2 (parallel w/ compute)
        if (has_next) {
            issue_load(next_col, cur_stage ^ 1);
        }

        // Wait for current stage to be ready.
        //   If we just issued: wait until ≤1 group remains (older = current col) is done.
        //   If last iteration (no next): wait until all groups done.
        if (has_next) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }

        if (cur_col >= N) continue;

        // ---- Compute: read vreg_b from shmem (current stage) ----
        int4 vreg_b[NUM_UNROLL];
        const int8_t* src = warp_smem[cur_stage][warp_id];
        #pragma unroll
        for (int u = 0; u < NUM_UNROLL; u++) {
            int kk = lane_k_off + u * alignmentB;
            vreg_b[u] = (kk + alignmentB <= K) ?
                *(reinterpret_cast<const int4*>(&src[kk])) : make_int4(0, 0, 0, 0);
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

        // Warp reduce
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            local_sum += __shfl_down_sync(0xffffffff, local_sum, off);

        // Lane 0 writes
        if (lane_k == 0) {
            float scaled = (float)local_sum * a_scale * b_scale[cur_col];
            out[cur_col] = __float2bfloat16(scaled);
        }
    }
}

// =============================================================================
// __global__ entry
// =============================================================================

__global__ void gemv_int8_async_v2_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ W,
    __nv_bfloat16* __restrict__ out,
    int K, int N,
    const float* __restrict__ a_scale,
    const float* __restrict__ b_scale
) {
    float a_s = __ldg(a_scale);
    gemv_int8_async_v2_impl<256, 4, 4>(A, W, out, K, N, a_s, b_scale);
}

// =============================================================================
// C-style launcher
// =============================================================================

extern "C" void launch_gemv_int8_async_v2(
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
    constexpr int NLoopStep = BlockSize / 32;
    constexpr int NPerBlock = NPerThread * NLoopStep;
    int n_blocks = (N + NPerBlock - 1) / NPerBlock;

    (void)M;

    gemv_int8_async_v2_kernel<<<n_blocks, BlockSize, 0, stream>>>(
        A, W, out, K, N, As, Bs
    );
}