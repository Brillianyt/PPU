#!/usr/bin/env python3
"""SGLang launcher for Qwen3.5-2B INT8 on PPU (方案 B: S2 + ACEXT + GEMV 分派).

默认路径 (§37 推荐): ACEXT 接管 dense linear (PPU sgl_kernel.int8_gemm) + GEMV 作 fallback。
对照路径: 设 SGLANG_SAIL_USE_ACEXT_CUDA=0 关 ACEXT,走自研 GEMV kernel。

dispatch 链路 (与 GEMM 完全同构):
    w8a8_int8.apply()
        ├─ SGLANG_SAIL_USE_ACEXT_CUDA=1 → sgl_kernel.int8_gemm (PPU ACEXT 库)
        └─ M ≤ SGLANG_SAIL_GEMV_M_MAX (默认 8)
            └─ gemv_int8_i8i8bf16 (custom op,支持 CUDA graph)
                └─ deep_gemm_wrapper.gemv_nt_i8i8bf16 (wrapper)
                    └─ deep_gemm.gemv_int8_int8_bf16_nt (复用 Gemvt 内核)

env vars:
    SGLANG_SAIL_DEEPGEMM_DENSE=1   # 进入 DeepGEMM 分支(必须)
    SGLANG_SAIL_USE_GEMV=1         # 启用 GEMV(默认开,作 ACEXT 的 fallback)
    SGLANG_SAIL_GEMV_M_MAX=8       # GEMV 上界(M 超过则回退 GEMM)
    SGLANG_SAIL_PREFILL_BF16=1     # prefill (M>8) 反量化走 BF16 GEMM
    SGLANG_SAIL_FLA_CUDA=1         # PPU 原生 GDN CUDA kernel
    SGLANG_SAIL_USE_ACEXT_CUDA=1   # ★ 启用 PPU ACEXT INT8 GEMM (方案 B 默认)

Usage:
    # 推荐 (ACEXT + GEMV 分派)
    python sglang_launcher.py --model-path /path/to/Qwen3.5-2B-S2-aggressive ...

    # 对照实验 (仅 GEMV, 关 ACEXT)
    SGLANG_SAIL_USE_ACEXT_CUDA=0 python sglang_launcher.py ...
"""

import os
import sys

# 必须在 sglang/quantization 模块 import 之前设置,因为 environ.py
# 在 import 时把这些 env 注册成 EnvBool/EnvInt,后续由 .get() 读取。
os.environ.setdefault("PPU_SDK", "/usr/local/PPU_SDK")
os.environ["SGLANG_SAIL_DEEPGEMM_DENSE"] = "1"
os.environ.setdefault("SGLANG_SAIL_USE_GEMV", "1")
os.environ.setdefault("SGLANG_SAIL_GEMV_M_MAX", "8")
os.environ["SGLANG_SAIL_PREFILL_BF16"] = "1"
os.environ["SGLANG_SAIL_FLA_CUDA"] = "1"
# ★ 方案 B 默认:ACEXT 接管 dense linear
os.environ.setdefault("SGLANG_SAIL_USE_ACEXT_CUDA", "1")

use_acext = os.environ.get("SGLANG_SAIL_USE_ACEXT_CUDA") == "1"
print(
    "[sglang_launcher] 方案 B 启动: "
    f"DEEPGEMM_DENSE=1 USE_GEMV=1 GEMV_M_MAX=8 PREFILL_BF16=1 FLA_CUDA=1 "
    f"USE_ACEXT_CUDA={'1 (PPU ACEXT)' if use_acext else '0 (仅 GEMV)'}",
    file=sys.stderr,
    flush=True,
)

# Hand off to sglang.launch_server as if invoked with -m
sys.argv[0] = "sglang.launch_server"
import runpy
runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)