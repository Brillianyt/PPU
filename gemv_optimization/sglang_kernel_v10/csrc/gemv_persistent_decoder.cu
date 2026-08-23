// gemv_persistent_decoder.cu — Phase 3.3: persistent INT8 GEMV dispatcher
//
// ONE kernel launch processes a whole queue of GEMV work items.
//
// Architecture:
//   - 2D grid: gridDim.x = max(1, (N + 31) / 32)  (col blocks per work item)
//              gridDim.y = total_work             (work items in queue)
//   - blockDim = 256 threads = 8 warps
//   - blockIdx.y = which work item
//   - blockIdx.x = which col block of that work item
//   - For each (work_item, col_block): dispatch to v10's impl
//
// Why 2D grid:
//   v10's gemv_int8_async_impl uses blockIdx.x to compute the col offset.
//   In a persistent kernel, blockIdx.x already means "which SM", not
//   "which col block of which GEMV". A 2D grid lets us put the work-item
//   index on .y and keep col-block index on .x, so v10's logic works
//   unchanged.
//
// Constraints (same as v10 MVP):
//   - K = 2048 only
//   - M = 1 only (A is shape [1, K])
//   - N can be any value; n_blocks = ceil(N / 32) col blocks per work item
//
// Build: same nvcc flags as v10/v2 (see setup.py).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "binding.h"
#include "gemv_int8_async_impl.h"

// =============================================================================
// Work item — packed 64-byte struct
// =============================================================================

struct alignas(16) GemvWorkItem {
    const int8_t* A;      // [1, K]
    const int8_t* W;      // [N, K]
    __nv_bfloat16* out;   // [1, N]
    int K;
    int N;
    const float* As;      // [1] — we use [0] only (M=1)
    const float* Bs;      // [N]
    int _pad;
};

// =============================================================================
// Persistent kernel
// =============================================================================

__global__ void persistent_gemv_decoder_kernel(
    int total_work,
    const GemvWorkItem* __restrict__ work_items
) {
    int work_idx  = blockIdx.y;
    int col_block = blockIdx.x;   // 0 .. ceil(N/32)-1 for this work item

    if (work_idx >= total_work) return;

    const GemvWorkItem item = work_items[work_idx];
    if (item.N <= 0 || item.K != 2048) return;

    float a_s = __ldg(item.As);

    // Dispatch to v10's gemv_int8_async_impl. Inside the impl, blockIdx.x is
    // this kernel's blockIdx.x (col_block), which is what v10 expects.
    gemv_int8_async_impl<256, 4, 4>(
        item.A, item.W, item.out, item.K, item.N, a_s, item.Bs
    );
}

// =============================================================================
// Launcher
// =============================================================================

extern "C" void launch_persistent_gemv_decoder(
    int total_work,
    const void* work_items,
    cudaStream_t stream
) {
    // Need to know max N across all work items to set gridDim.x. For MVP
    // we assume all work items share the same N (sglang decode path: each
    // layer has multiple GEMVs but they share a common hidden size, and the
    // larger GEMVs like lm_head aren't in decode anyway).
    //
    // To handle mixed N: caller should pre-pad the queue so all items have
    // the same N, OR we run multiple kernel launches (one per N bucket).
    //
    // For Qwen3.5-2B decode path, all GEMVs have K=2048 and N in
    // {2048, 6144, 8192, 12288}. The max N is 12288 → max col blocks = 384.
    constexpr int MAX_COL_BLOCKS = 384;
    dim3 grid(MAX_COL_BLOCKS, total_work);
    dim3 block(256);
    persistent_gemv_decoder_kernel<<<grid, block, 0, stream>>>(
        total_work, reinterpret_cast<const GemvWorkItem*>(work_items)
    );
}