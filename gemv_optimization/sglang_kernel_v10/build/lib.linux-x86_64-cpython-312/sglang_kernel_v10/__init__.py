"""sglang_kernel_v10 — ISA Phase 3.2 INT8 GEMV fast path.

A standalone PyTorch C++ extension that implements the v10 cp.async-free
INT8 GEMV kernel for sglang's `w8a8_int8` decode path on PPU001 hardware.

Install:
    pip install .

Use:
    import torch
    from sglang_kernel_v10 import gemv_int8_async

    A  = torch.randint(-128, 127, (1, 2048), dtype=torch.int8,   device='cuda')
    W  = torch.randint(-128, 127, (2048, 2048), dtype=torch.int8, device='cuda')
    As = torch.ones(1, dtype=torch.float32, device='cuda')
    Bs = torch.ones(2048, dtype=torch.float32, device='cuda')
    out = torch.empty(1, 2048, dtype=torch.bfloat16, device='cuda')
    gemv_int8_async(A, W, As, Bs, out)

Architecture note:
    This package exists as a SIBLING of sgl_kernel, not as a vendored copy
    inside sglang/. The .so is built at install time (not JIT-loaded at
    first call), which means:
      - No first-call compile latency that would break CUDA graph capture
      - No kernel .cu / .cpp files leaking into sglang's package directory
      - Standard `import` semantics, exactly like flash-attn / deep_gemm
"""

from ._C import gemv_int8_async

__all__ = ["gemv_int8_async"]
__version__ = "0.1.0"