// gemv_persistent_decoder_wrapper.cpp — PyTorch binding for Phase 3.3
// persistent INT8 GEMV decoder.
//
// Python API (after extension is built into the same .so as v10):
//   queue, total = build_gemv_queue([A, W, K, N, As, Bs, out, ...])
//   persistent_gemv_decoder_launch(queue, total)
//
// Implementation note: pybind11 doesn't auto-convert std::vector<Tensor>
// in all pytorch builds. We accept a Python list (py::list) and convert
// explicitly.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "binding.h"

namespace py = pybind11;

struct alignas(16) GemvWorkItem {
    const int8_t* A;
    const int8_t* W;
    __nv_bfloat16* out;
    int32_t K;
    int32_t N;
    const float* As;
    const float* Bs;
    int32_t _pad;
};

// Build a queue from a Python list of 7-tuples:
//   [A, W, K_int, N_int, As, Bs, out, A, W, K_int, N_int, As, Bs, out, ...]
std::tuple<torch::Tensor, int64_t> build_gemv_queue(py::list items_list) {
    int64_t total = items_list.size() / 7;
    TORCH_CHECK(items_list.size() % 7 == 0, "items_list size must be multiple of 7");
    TORCH_CHECK(total > 0, "queue empty");

    std::vector<GemvWorkItem> h_items(total);
    for (int64_t i = 0; i < total; i++) {
        auto A_  = items_list[i*7 + 0].cast<torch::Tensor>();
        auto W_  = items_list[i*7 + 1].cast<torch::Tensor>();
        auto K_  = items_list[i*7 + 2].cast<int32_t>();
        auto N_  = items_list[i*7 + 3].cast<int32_t>();
        auto As_ = items_list[i*7 + 4].cast<torch::Tensor>();
        auto Bs_ = items_list[i*7 + 5].cast<torch::Tensor>();
        auto Ou_ = items_list[i*7 + 6].cast<torch::Tensor>();

        TORCH_CHECK(A_.is_cuda()  && A_.scalar_type()  == torch::kInt8);
        TORCH_CHECK(W_.is_cuda()  && W_.scalar_type()  == torch::kInt8);
        TORCH_CHECK(As_.is_cuda() && As_.scalar_type() == torch::kFloat32);
        TORCH_CHECK(Bs_.is_cuda() && Bs_.scalar_type() == torch::kFloat32);
        TORCH_CHECK(Ou_.is_cuda() && Ou_.scalar_type() == torch::kBFloat16);
        TORCH_CHECK(K_ == 2048, "Phase 3.3 MVP only supports K=2048");

        auto& it = h_items[i];
        it.A    = A_.data_ptr<int8_t>();
        it.W    = W_.data_ptr<int8_t>();
        it.K    = K_;
        it.N    = N_;
        it.As   = As_.data_ptr<float>();
        it.Bs   = Bs_.data_ptr<float>();
        it.out  = reinterpret_cast<__nv_bfloat16*>(Ou_.data_ptr());
        it._pad = 0;
    }

    int64_t bytes = total * sizeof(GemvWorkItem);
    auto queue_bytes = torch::empty(
        {(int64_t)(bytes / sizeof(int8_t))},
        torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA)
    );
    cudaMemcpyAsync(queue_bytes.data_ptr(), h_items.data(), bytes,
                    cudaMemcpyHostToDevice, c10::cuda::getCurrentCUDAStream());
    return {queue_bytes, total};
}

void persistent_gemv_decoder_launch(torch::Tensor queue, int64_t total_work) {
    launch_persistent_gemv_decoder(
        (int)total_work,
        reinterpret_cast<const GemvWorkItem*>(queue.data_ptr()),
        c10::cuda::getCurrentCUDAStream()
    );
}