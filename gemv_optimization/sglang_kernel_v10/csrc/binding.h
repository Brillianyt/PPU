// binding.h — C-linkage declarations shared between .cu kernel and .cpp wrapper.
//
// Why a separate header:
//   - .cu file is compiled by nvcc; .cpp wrapper by g++
//   - They share no C++ types beyond this minimal C-linkage surface
//   - This avoids pulling <torch/extension.h> into the .cu file (which would
//     re-trigger the nvcc 13 + g++ 13 cmath constexpr vs HGGC isinf/isnan
//     conflict that took Phase 3.1 down)
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" {

// Kernel launcher — implemented in gemv_int8_async.cu.
//   A:   int8,   (M, K)       activation (row-major)
//   W:   int8,   (N, K)       weight (row-major, no transpose needed)
//   out: bf16,   (M, N)       output
//   M, K, N: standard dims
//   As:  fp32,   (M,)         per-row activation scale (unused in MVP)
//   Bs:  fp32,   (N,)         per-row weight scale (unused in MVP)
void launch_gemv_int8_async(
    const int8_t* A,
    const int8_t* W,
    __nv_bfloat16* out,
    int M, int K, int N,
    const float* As,
    const float* Bs,
    cudaStream_t stream
);

// Phase 3.2.1: cp.async double-buffered variant (shared-memory staged).
// Same signature, internally uses 2-stage cp.async ping-pong over shared mem.
void launch_gemv_int8_async_v2(
    const int8_t* A,
    const int8_t* W,
    __nv_bfloat16* out,
    int M, int K, int N,
    const float* As,
    const float* Bs,
    cudaStream_t stream
);

// Phase 3.3: persistent decoder — one launch processes a queue of GEMVs.
// work_items points to device memory; total_work is the queue length.
void launch_persistent_gemv_decoder(
    int total_work,
    const void* work_items,   // actually GemvWorkItem* (opaque to keep header C-clean)
    cudaStream_t stream
);

// Phase 3.4: fused Q+K+V INT8 GEMV — 1 launch produces 3 outputs sharing A.
// MVP: N_q == N_k == N_v.
void launch_fused_qkv(
    const int8_t* A,
    const int8_t* Wq, const int8_t* Wk, const int8_t* Wv,
    __nv_bfloat16* out_q, __nv_bfloat16* out_k, __nv_bfloat16* out_v,
    int M, int K, int N,
    const float* As,
    const float* Bs_q, const float* Bs_k, const float* Bs_v,
    cudaStream_t stream
);

}  // extern "C"