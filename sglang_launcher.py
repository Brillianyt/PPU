#!/usr/bin/env python3
"""SGLang launcher with INT8 GEMV fast path enabled (PPU).

§32 调研方案:在 DeepGEMM dense 分支内启用 M≤8 GEMV fast path,
复用 Gemvt 内核(已在 sgl_kernel 中被超级优化),通过 deep_gemm
包装层接入。GEMV 与 GEMM bit-level 数值一致(max_diff=0),
目标机微基准 13.0us vs Triton 27.9us = 2.15×。

dispatch 链路(与 GEMM 完全同构):
    w8a8_int8.apply()
        └─ M ≤ SGLANG_SAIL_GEMV_M_MAX (默认 8)
            └─ gemv_int8_i8i8bf16 (custom op,支持 CUDA graph)
                └─ deep_gemm_wrapper.gemv_nt_i8i8bf16 (wrapper)
                    └─ deep_gemm.gemv_int8_int8_bf16_nt (复用 Gemvt 内核)

env vars (设置后 DeepGEMM dense 路径下,M≤8 的 decode 会自动走 GEMV):
    SGLANG_SAIL_DEEPGEMM_DENSE=1  # 进入 DeepGEMM 分支(必须)
    SGLANG_SAIL_USE_GEMV=1        # 启用 GEMV(默认关,风险隔离)
    SGLANG_SAIL_GEMV_M_MAX=8      # GEMV 上界(M 超过则回退 GEMM)

Usage:
    python sglang_launcher.py --model-path /path/to/Qwen3.5-2B ...
"""

import os
import sys

# §33 — S3-full INT8 GEMV fast path (186 个 int8 层, 4× 覆盖 v3)。
# 必须在 sglang/quantization 模块 import 之前设置,因为 environ.py
# 在 import 时把这些 env 注册成 EnvBool/EnvInt,后续由 .get() 读取。
os.environ.setdefault("PPU_SDK", "/usr/local/PPU_SDK")
os.environ["SGLANG_SAIL_DEEPGEMM_DENSE"] = "1"
# GEMV 默认开,可由外部 SGLANG_SAIL_USE_GEMV=0 关掉(用于 §34 对照实验)
os.environ.setdefault("SGLANG_SAIL_USE_GEMV", "1")
os.environ.setdefault("SGLANG_SAIL_GEMV_M_MAX", "8")
# 方向 B: prefill (M > 8) 反量化 int8 权重走 BF16 GEMM,降低 TTFT;
# decode (M ≤ 8) 仍走 GEMV(不变)。
os.environ["SGLANG_SAIL_PREFILL_BF16"] = "1"
os.environ["SGLANG_SAIL_FLA_CUDA"] = "1"

print(
    "[sglang_launcher] §36 S2 + GEMV (decode) + PREFILL_BF16: "
    "DEEPGEMM_DENSE=1 USE_GEMV=1 GEMV_M_MAX=8 PREFILL_BF16=1 FLA_CUDA=1",
    file=sys.stderr,
    flush=True,
)

# Hand off to sglang.launch_server as if invoked with -m
sys.argv[0] = "sglang.launch_server"
import runpy
runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)