"""Setup script for sglang_kernel_v10.

Builds the v10 INT8 GEMV kernel as a proper PyTorch C++ extension,
installed as `_C` inside the `sglang_kernel_v10` Python package.

The .so is built at install time (`pip install .`), NOT at first call.
This is critical because:
  1. JIT-load latency on first request would tank TTFT
  2. JIT-load during CUDA graph capture is undefined behavior (one reason
     the prior integration regressed TPS by ~5% in the benchmark)

PPU001 specifics:
  - arch flag is `compute_80a` (not `sm_80`). The trailing `a` matters.
  - The kernel uses SAIL cp.async PTX (`ppu.cp.async.*`), not NVIDIA
    `cp.async.*`. This is intentional — see ISA notes docs/02_ARCHITECTURE.md
    for the Phase 3.1 HGGC-mma failure story.
  - HGGC SDK headers are required at /usr/local/PPU_SDK/...
"""
import os
import sys

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# =============================================================================
# PPU001-specific build configuration
# =============================================================================

PPU_ARCH_FLAG = "-gencode=arch=compute_80a,code=sm_80a"
DIAG_SUPPRESS = "--diag-suppress=39,174,177,940"

# The v10 kernel does NOT include any HGGC headers (Phase 3.1 lesson —
# g++ 13 cmath constexpr vs HGGC __device__ isinf/isnan crash). So no
# PPU_SDK_INCLUDE is needed for the build. If a future kernel phase
# (e.g. 3.3 with awmma) needs HGGC, add PPU_SDK_INCLUDE here AND
# configure `-D__HIGHC__` etc. carefully.
PPU_SDK_ROOT = os.environ.get("PPU_SDK_ROOT", "/usr/local/PPU_SDK")

# =============================================================================
# Sources & includes
# =============================================================================

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
CSRC_DIR = os.path.join(PKG_DIR, "csrc")

EXT_NAME = "sglang_kernel_v10._C"   # → installs as sglang_kernel_v10/_C*.so

CUDA_SOURCES = [
    os.path.join(CSRC_DIR, "gemv_int8_async.cu"),
    os.path.join(CSRC_DIR, "gemv_int8_async_v2.cu"),
    os.path.join(CSRC_DIR, "gemv_persistent_decoder.cu"),
    os.path.join(CSRC_DIR, "gemv_fused_qkv.cu"),
]
# Single C++ wrapper that registers all pybind11 symbols.
CPP_SOURCES = [
    os.path.join(CSRC_DIR, "gemv_int8_async_wrapper.cpp"),
    os.path.join(CSRC_DIR, "gemv_persistent_decoder_wrapper.cpp"),
]
ALL_SOURCES = CUDA_SOURCES + CPP_SOURCES

INCLUDE_DIRS = [
    CSRC_DIR,                        # for binding.h
]

CUDA_CFLAGS = [
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    PPU_ARCH_FLAG,
    DIAG_SUPPRESS,
    "--compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi",
]

CFLAGS = ["-O3", "-fPIC", "-Wno-deprecated-declarations", "-Wno-abi"]

# =============================================================================
# Setup
# =============================================================================

setup(
    name="sglang_kernel_v10",
    version="0.1.4",
    description="ISA Phase 3.2/3.2.1 — INT8 GEMV fast path (4.5× vs deep_gemm at K=2048, M=1)",
    long_description=open(os.path.join(PKG_DIR, "README.md")).read()
                   if os.path.exists(os.path.join(PKG_DIR, "README.md")) else "",
    long_description_content_type="text/markdown",
    packages=["sglang_kernel_v10"],
    package_dir={"sglang_kernel_v10": "sglang_kernel_v10"},
    package_data={"sglang_kernel_v10": ["*.py"]},
    include_package_data=False,
    ext_modules=[
        # Single .so with both v10 (HBM-direct) and v2 (cp.async double-buffer)
        # registered as separate pybind11 symbols.
        CUDAExtension(
            name="sglang_kernel_v10._C",
            sources=ALL_SOURCES,
            include_dirs=INCLUDE_DIRS,
            extra_compile_args={
                "cxx": CFLAGS,
                "nvcc": CUDA_CFLAGS,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch"],
)