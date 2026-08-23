// gemv_fused_qkv.cu — Phase 3.4: fused Q+K+V INT8 GEMV (1 launch, 3 outputs)
//
// One kernel launch computes 3 GEMV projections sharing the same A input.
//   A:  (1, K)          shared activation vector
//   Wq: (N, K)          Q projection weight
//   Wk: (N, K)          K projection weight
//   Wv: (N, K)          V projection weight
//   out_q/k/v: (1, N)   3 output buffers (separate)
//
// Layout (same NUM_UNROLL=4, NPerThread=4, BlockSize=256 as v10):
//   - Each warp processes NPerThread=4 cols for EACH of Q, K, V
//   - A vector pre-loaded once into registers (shared across Q/K/V)
//   - Per iteration: read 3 rows (Wq, Wk, Wv), dot with same vreg_a, write 3 outputs
//   - n_blocks = ceil(N / (32 cols/block))  -- same as v10
//
// Why fused:
//   - A loaded once (vs 3x in separate calls)
//   - 3 outputs written from the same warp (better register reuse)
//   - 1 launch vs 3 (in standalone benchmark only — sglang CUDA graph
//     already captures the launches)
//   - Wq, Wk, Wv read together — potentially better L2 prefetching
//
// MVP constraint: N_q == N_k == N_v (assumed equal). For Qwen self-attn this
// is true (all N=hidden_size). For linear-attn in_proj_qkv, the QKV is
// already fused in the weight matrix (N=6144, single matmul) so no fusion
// needed.
//
// Build: same nvcc flags as v10 (see setup.py).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "binding.h"

template <int BlockSize, int NPerThread, int NUM_UNROLL>
__device__ __forceinline__ void fused_qkv_impl(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ Wq,
    const int8_t* __restrict__ Wk,
    const int8_t* __restrict__ Wv,
    __nv_bfloat16* __restrict__ out_q,
    __nv_bfloat16* __restrict__ out_k,
    __nv_bfloat16* __restrict__ out_v,
    int K, int N,
    float a_scale,
    const float* __restrict__ bs_q,
    const float* __restrict__ bs_k,
    const float* __restrict__ bs_v
) {
    constexpr int ThreadPerN = 32;
    constexpr int NLoopStep = BlockSize / ThreadPerN;        // 8 warps
    constexpr int alignmentB = sizeof(int4) / sizeof(int8_t); // 16
    constexpr int kstep_lane = NUM_UNROLL * alignmentB;        // 64

    int tid = threadIdx.x;
    int lane_k = tid % ThreadPerN;
    int warp_id = tid / ThreadPerN;
    int tid_n = __shfl_sync(0xffffffff, warp_id, 0);
    int id_n = blockIdx.x * (NPerThread * NLoopStep) + tid_n * NPerThread;
    if (id_n >= N) return;

    int lane_k_off = lane_k * kstep_lane;

    // Pre-load A vector — shared across all 3 projections.
    int4 vreg_a[NUM_UNROLL];
    #pragma unroll
    for (int u = 0; u < NUM_UNROLL; u++) {
        int kk = lane_k_off + u * alignmentB;
        vreg_a[u] = (kk + alignmentB <= K) ?
            *reinterpret_cast<const int4*>(&A[kk]) : make_int4(0, 0, 0, 0);
    }

    // Main loop: 4 cols each for Q, K, V
    for (int nloop = 0; nloop < NPerThread; nloop++) {
        int cur_col = id_n + nloop;
        if (cur_col >= N) continue;

        // Load 3 weight rows (Wq, Wk, Wv) for this col
        int4 vreg_q[NUM_UNROLL], vreg_k[NUM_UNROLL], vreg_v[NUM_UNROLL];
        const int8_t* row_q = Wq + cur_col * K;
        const int8_t* row_k = Wk + cur_col * K;
        const int8_t* row_v = Wv + cur_col * K;
        #pragma unroll
        for (int u = 0; u < NUM_UNROLL; u++) {
            int kk = lane_k_off + u * alignmentB;
            bool in_range = (kk + alignmentB <= K);
            vreg_q[u] = in_range ? *(reinterpret_cast<const int4*>(&row_q[kk])) : make_int4(0, 0, 0, 0);
            vreg_k[u] = in_range ? *(reinterpret_cast<const int4*>(&row_k[kk])) : make_int4(0, 0, 0, 0);
            vreg_v[u] = in_range ? *(reinterpret_cast<const int4*>(&row_v[kk])) : make_int4(0, 0, 0, 0);
        }

        // Dot product: A · Wq, A · Wk, A · Wv
        auto dot_a_w = [&](const int4 vreg_w[NUM_UNROLL]) -> int32_t {
            int32_t s = 0;
            #pragma unroll
            for (int u = 0; u < NUM_UNROLL; u++) {
                const int8_t* a_bytes = reinterpret_cast<const int8_t*>(&vreg_a[u]);
                const int8_t* b_bytes = reinterpret_cast<const int8_t*>(&vreg_w[u]);
                int32_t acc = 0;
                #pragma unroll
                for (int j = 0; j < 16; j++) acc += (int32_t)a_bytes[j] * (int32_t)b_bytes[j];
                s += acc;
            }
            return s;
        };

        int32_t sum_q = dot_a_w(vreg_q);
        int32_t sum_k = dot_a_w(vreg_k);
        int32_t sum_v = dot_a_w(vreg_v);

        // 3 separate warp reduces (3 outputs from same warp)
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_q += __shfl_down_sync(0xffffffff, sum_q, off);
            sum_k += __shfl_down_sync(0xffffffff, sum_k, off);
            sum_v += __shfl_down_sync(0xffffffff, sum_v, off);
        }

        // Lane 0 writes 3 outputs
        if (lane_k == 0) {
            out_q[cur_col] = __float2bfloat16((float)sum_q * a_scale * bs_q[cur_col]);
            out_k[cur_col] = __float2bfloat16((float)sum_k * a_scale * bs_k[cur_col]);
            out_v[cur_col] = __float2bfloat16((float)sum_v * a_scale * bs_v[cur_col]);
        }
    }
}

// =============================================================================
// __global__ entry
// =============================================================================

__global__ void fused_qkv_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ Wq,
    const int8_t* __restrict__ Wk,
    const int8_t* __restrict__ Wv,
    __nv_bfloat16* __restrict__ out_q,
    __nv_bfloat16* __restrict__ out_k,
    __nv_bfloat16* __restrict__ out_v,
    int K, int N,
    const float* __restrict__ a_scale,
    const float* __restrict__ bs_q,
    const float* __restrict__ bs_k,
    const float* __restrict__ bs_v
) {
    float a_s = __ldg(a_scale);
    fused_qkv_impl<256, 4, 4>(
        A, Wq, Wk, Wv, out_q, out_k, out_v, K, N, a_s, bs_q, bs_k, bs_v
    );
}

// =============================================================================
// C-style launcher
// =============================================================================

extern "C" void launch_fused_qkv(
    const int8_t* A,
    const int8_t* Wq, const int8_t* Wk, const int8_t* Wv,
    __nv_bfloat16* out_q, __nv_bfloat16* out_k, __nv_bfloat16* out_v,
    int M, int K, int N,
    const float* As,
    const float* Bs_q, const float* Bs_k, const float* Bs_v,
    cudaStream_t stream
) {
    constexpr int BlockSize = 256;
    constexpr int NPerThread = 4;
    constexpr int NLoopStep = BlockSize / 32;
    constexpr int NPerBlock = NPerThread * NLoopStep;
    int n_blocks = (N + NPerBlock - 1) / NPerBlock;

    (void)M;

    fused_qkv_kernel<<<n_blocks, BlockSize, 0, stream>>>(
        A, Wq, Wk, Wv, out_q, out_k, out_v, K, N, As, Bs_q, Bs_k, Bs_v
    );
}