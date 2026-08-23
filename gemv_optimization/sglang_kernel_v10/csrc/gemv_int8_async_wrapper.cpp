// gemv_int8_async_wrapper.cpp — PyTorch binding for v10 + v2 kernels.
//
// Both kernels live in one .so, registered as separate pybind11 symbols.
// The .cu files are compiled separately and linked into this wrapper.
//
// v10: Phase 3.2 baseline — HBM-direct SIMT dot product (no shmem).
// v2:  Phase 3.2.1 — cp.async double-buffer (shared-memory staged).

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

#include "binding.h"

// =============================================================================
// v10 — HBM-direct INT8 GEMV
// =============================================================================

torch::Tensor gemv_int8_async(
    torch::Tensor A,
    torch::Tensor W,
    torch::Tensor As,
    torch::Tensor Bs,
    torch::Tensor out
) {
    TORCH_CHECK(A.is_cuda() && W.is_cuda() && As.is_cuda() &&
                Bs.is_cuda() && out.is_cuda(), "all tensors must be on CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kInt8,                "A must be int8");
    TORCH_CHECK(W.scalar_type() == torch::kInt8,                "W must be int8");
    TORCH_CHECK(As.scalar_type() == torch::kFloat32,            "As must be fp32");
    TORCH_CHECK(Bs.scalar_type() == torch::kFloat32,            "Bs must be fp32");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16,          "out must be bf16");
    TORCH_CHECK(out.is_contiguous(),                            "out must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = W.size(0);
    TORCH_CHECK(W.size(1) == K,                                "K mismatch (W.size(1) vs A.size(1))");
    TORCH_CHECK(M == 1,                                         "Phase 3.2 MVP only supports M=1");
    TORCH_CHECK(K == 2048,                                      "Phase 3.2 MVP only supports K=2048");

    launch_gemv_int8_async(
        A.data_ptr<int8_t>(),
        W.data_ptr<int8_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        M, K, N,
        As.data_ptr<float>(),
        Bs.data_ptr<float>(),
        c10::cuda::getCurrentCUDAStream()
    );
    return out;
}

// =============================================================================
// v2 — cp.async double-buffered INT8 GEMV (Phase 3.2.1)
// =============================================================================

torch::Tensor gemv_int8_async_v2(
    torch::Tensor A,
    torch::Tensor W,
    torch::Tensor As,
    torch::Tensor Bs,
    torch::Tensor out
) {
    TORCH_CHECK(A.is_cuda() && W.is_cuda() && As.is_cuda() &&
                Bs.is_cuda() && out.is_cuda(), "all tensors must be on CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kInt8,                "A must be int8");
    TORCH_CHECK(W.scalar_type() == torch::kInt8,                "W must be int8");
    TORCH_CHECK(As.scalar_type() == torch::kFloat32,            "As must be fp32");
    TORCH_CHECK(Bs.scalar_type() == torch::kFloat32,            "Bs must be fp32");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16,          "out must be bf16");
    TORCH_CHECK(out.is_contiguous(),                            "out must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = W.size(0);
    TORCH_CHECK(W.size(1) == K, "K mismatch");
    TORCH_CHECK(M == 1, "Phase 3.2.1 MVP only supports M=1");
    TORCH_CHECK(K == 2048, "Phase 3.2.1 MVP only supports K=2048");

    launch_gemv_int8_async_v2(
        A.data_ptr<int8_t>(),
        W.data_ptr<int8_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        M, K, N,
        As.data_ptr<float>(),
        Bs.data_ptr<float>(),
        c10::cuda::getCurrentCUDAStream()
    );
    return out;
}

// =============================================================================
// pybind11 module
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_int8_async",   &gemv_int8_async,    "INT8 GEMV HBM-direct (PPU001, ISA Phase 3.2)");
    m.def("gemv_int8_async_v2", &gemv_int8_async_v2, "INT8 GEMV cp.async double-buffer (PPU001, ISA Phase 3.2.1)");
}