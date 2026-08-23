// gemv_int8_async_impl.h — shared implementation of v10's GEMV kernel body.
//
// This is the .h counterpart to gemv_int8_async.cu's `gemv_int8_async_impl`
// template. We keep it as a header (rather than duplicate) so the
// persistent decoder can dispatch to the exact same implementation.
//
// Only the kernel-body template is exposed here. The launcher and the
// global `__global__` entry stay in gemv_int8_async.cu.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

template <int BlockSize, int NPerThread, int NUM_UNROLL>
__device__ __forceinline__ void gemv_int8_async_impl(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ W,
    __nv_bfloat16* __restrict__ out,
    int K, int N,
    float a_scale,
    const float* __restrict__ b_scale
) {
    constexpr int ThreadPerN = 32;
    constexpr int NLoopStep = BlockSize / ThreadPerN;
    constexpr int alignmentB = sizeof(int4) / sizeof(int8_t); // 16
    constexpr int kstep_lane = NUM_UNROLL * alignmentB;        // 64

    int tid = threadIdx.x;
    int lane_k = tid % ThreadPerN;
    int warp_id = tid / ThreadPerN;
    int tid_n = __shfl_sync(0xffffffff, warp_id, 0);
    int id_n = blockIdx.x * (NPerThread * NLoopStep) + tid_n * NPerThread;
    if (id_n >= N) return;

    int lane_k_off = lane_k * kstep_lane;

    int4 vreg_a[NUM_UNROLL];
    #pragma unroll
    for (int u = 0; u < NUM_UNROLL; u++) {
        int kk = lane_k_off + u * alignmentB;
        vreg_a[u] = (kk + alignmentB <= K) ?
            *reinterpret_cast<const int4*>(&A[kk]) : make_int4(0, 0, 0, 0);
    }

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

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            local_sum += __shfl_down_sync(0xffffffff, local_sum, off);

        if (lane_k == 0) {
            float scaled = (float)local_sum * a_scale * b_scale[cur_col];
            out[cur_col] = __float2bfloat16(scaled);
        }
    }
}