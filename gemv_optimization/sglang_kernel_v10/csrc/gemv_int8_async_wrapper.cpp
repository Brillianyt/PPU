// gemv_int8_async_wrapper.cpp — PyTorch binding for v10 + v2 + fused-QKV kernels.
//
// All kernels (including the persistent decoder and queue-builder from
// gemv_persistent_decoder_wrapper.cpp) are registered in this single
// pybind11 module. The persistent wrapper's functions are forward-declared
// here and the linker pulls them in from its own .o.
//
// v10: Phase 3.2 baseline — HBM-direct SIMT dot product (no shmem).
// v2:  Phase 3.2.1 — cp.async double-buffer (shared-memory staged).
// fused_qkv: Phase 3.4 — 1 launch produces Q+K+V outputs sharing A.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "binding.h"

namespace py = pybind11;

// Forward-declared from gemv_persistent_decoder_wrapper.cpp — defined in
// its own .o so we can compile a "kernels-only" wheel that omits the
// persistent decoder. We pull them in unconditionally here; the linker
// will only succeed if the persistent wrapper's .o is part of the build.
std::tuple<torch::Tensor, int64_t> build_gemv_queue(py::list items_list);
void persistent_gemv_decoder_launch(torch::Tensor queue, int64_t total_work);

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
// fused Q+K+V (Phase 3.4)
// =============================================================================

torch::Tensor fused_qkv(
    torch::Tensor A,
    torch::Tensor Wq, torch::Tensor Wk, torch::Tensor Wv,
    torch::Tensor As,
    torch::Tensor Bs_q, torch::Tensor Bs_k, torch::Tensor Bs_v,
    torch::Tensor out_q, torch::Tensor out_k, torch::Tensor out_v
) {
    for (auto* t : {&A, &Wq, &Wk, &Wv, &As, &Bs_q, &Bs_k, &Bs_v, &out_q, &out_k, &out_v}) {
        TORCH_CHECK(t->is_cuda(), "all tensors must be on CUDA");
    }
    TORCH_CHECK(A.scalar_type()     == torch::kInt8,    "A must be int8");
    TORCH_CHECK(Wq.scalar_type()    == torch::kInt8,    "Wq must be int8");
    TORCH_CHECK(Wk.scalar_type()    == torch::kInt8,    "Wk must be int8");
    TORCH_CHECK(Wv.scalar_type()    == torch::kInt8,    "Wv must be int8");
    TORCH_CHECK(As.scalar_type()    == torch::kFloat32,  "As must be fp32");
    TORCH_CHECK(Bs_q.scalar_type()  == torch::kFloat32,  "Bs_q must be fp32");
    TORCH_CHECK(Bs_k.scalar_type()  == torch::kFloat32,  "Bs_k must be fp32");
    TORCH_CHECK(Bs_v.scalar_type()  == torch::kFloat32,  "Bs_v must be fp32");
    TORCH_CHECK(out_q.scalar_type() == torch::kBFloat16, "out_q must be bf16");
    TORCH_CHECK(out_k.scalar_type() == torch::kBFloat16, "out_k must be bf16");
    TORCH_CHECK(out_v.scalar_type() == torch::kBFloat16, "out_v must be bf16");

    const int M = A.size(0);
    const int K = A.size(1);
    TORCH_CHECK(M == 1,                "Phase 3.4 MVP only supports M=1");
    TORCH_CHECK(K == 2048,             "Phase 3.4 MVP only supports K=2048");
    TORCH_CHECK(Wq.size(1) == K,       "Wq K mismatch");
    TORCH_CHECK(Wk.size(1) == K,       "Wk K mismatch");
    TORCH_CHECK(Wv.size(1) == K,       "Wv K mismatch");
    const int N = Wq.size(0);
    TORCH_CHECK(Wk.size(0) == N,       "Wk N != Wq N (MVP assumes N_q == N_k == N_v)");
    TORCH_CHECK(Wv.size(0) == N,       "Wv N != Wq N (MVP assumes N_q == N_k == N_v)");

    launch_fused_qkv(
        A.data_ptr<int8_t>(),
        Wq.data_ptr<int8_t>(), Wk.data_ptr<int8_t>(), Wv.data_ptr<int8_t>(),
        reinterpret_cast<__nv_bfloat16*>(out_q.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out_k.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out_v.data_ptr()),
        M, K, N,
        As.data_ptr<float>(),
        Bs_q.data_ptr<float>(), Bs_k.data_ptr<float>(), Bs_v.data_ptr<float>(),
        c10::cuda::getCurrentCUDAStream()
    );
    return out_q;
}

// =============================================================================
// pybind11 module — v10 + v2 + fused-QKV
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_int8_async",   &gemv_int8_async,    "INT8 GEMV HBM-direct (PPU001, ISA Phase 3.2)");
    m.def("gemv_int8_async_v2", &gemv_int8_async_v2, "INT8 GEMV cp.async double-buffer (PPU001, ISA Phase 3.2.1)");
    m.def("fused_qkv",          &fused_qkv,          "Fused Q+K+V INT8 GEMV (PPU001, ISA Phase 3.4)");
    m.def("build_gemv_queue",   &build_gemv_queue,        "Pack Python tensor list → GPU work-item queue");
    m.def("persistent_gemv_decoder_launch", &persistent_gemv_decoder_launch, "Launch persistent kernel (1 launch → N GEMVs)");
}