# 05. 量化策略 + GEMV Decode 算子

## 5.1 量化策略对比

| 策略 | 量化层数 | 保护层 | 准确率 (20) | 准确率 (大样本) | 备注 |
|---|---|---|---|---|---|
| **v3 (低覆盖)** | 42 | 全部敏感层 | 0.75 | — | GEMV 端到端无收益 |
| **⭐ S2-aggressive (推荐)** | 126 | `in_proj_a/b`, `out_proj`, 19/23 qkv | 0.85 | 0.76-0.82 | 推荐用于 ACEXT 路径 |
| **S3-full** | 186 | 无 | 0.65 | — | 否决 (acc 太低) |

## 5.2 量化 Recipe

`recipes/qwen3.5-2b_amq_s2_aggressive.yaml` (核心片段):

```yaml
quant_stage:
  quant_modifiers:
    generalDay0Modifier:
      scheme: W8A8
      ignore:
        # ---- Standard ignores ----
        - "re:.*lm_head"
        - "re:.*embed_tokens"
        - "re:.*mlp\\.gate$"
        # 1D / 3D tensors
        - "re:.*norm\\.weight$"
        - "re:.*norm1\\.weight$"
        - "re:.*norm2\\.weight$"
        - "re:.*q_norm\\.weight$"
        - "re:.*k_norm\\.weight$"
        - "re:.*linear_attn\\.A_log$"
        - "re:.*linear_attn\\.dt_bias$"
        - "re:.*linear_attn\\.conv1d.*"
        # MTP / Vision
        - "re:.*mtp\\..*"
        - "re:.*visual\\..*"
        # ---- AMQ 高误差层保持 BF16 ----
        # in_proj_a/b: 误差最高 (mean 0.0154), recurrent 状态投影
        - "re:.*linear_attn\\.in_proj_a$"
        - "re:.*linear_attn\\.in_proj_b$"
        # out_proj: GDN 环路输出投影
        - "re:.*linear_attn\\.out_proj$"
        # 19/23 full_attention qkv: 误差 > 0.015
        - "re:.*layers\\.19\\.self_attn\\.(q_proj|k_proj|v_proj)$"
        - "re:.*layers\\.23\\.self_attn\\.(q_proj|k_proj|v_proj)$"
        # ---- S2 激进: gate/up 和 in_proj_qkv/z 放回 INT8 ----
        # (它们权重重建误差最低: gate_up mean 0.0092, in_proj_qkvz mean 0.0096)
```

工具: QLean (主办方工具)。命令: 用 recipe 对 BF16 模型执行量化。

## 5.3 三个关键优化点

### 5.3.1 敏感层保留 BF16

- AMQ 量化策略:对 175 个线性层做 INT8 重建误差分析
- 量化误差最低的 MLP/注意力投影
- 保护误差最高的 GDN 状态投影 (`in_proj_a/b`)
- 结果:30 样本与 BF16 无损,比 S3 (186 层) 更稳

### 5.3.2 GEMV Decode 算子 (deep_gemm)

- 内核: 复用 DeepGEMM `gemvt.cuh` 的 `Gemvt<int8_t>` (PPU 原生 AIU 异步搬运 + multistage)
- 入口: `deep_gemm.gemv_int8_int8_bf16_nt`
- microbench: M=1 时 **13.0 μs vs Triton fast path 27.9 μs, 2.15× 加速**
- 端到端价值: 高 INT8 覆盖率下绕开 `gemm_nt_i8i8bf16` 的 Python dispatch 开销
- 数值与 GEMM **bit 级一致** (max_diff = 0)

### 5.3.3 Prefill BF16 反量化 (方向 B)

- 问题: int8 GEMM 在 prefill (M 大) 比 BF16 GEMM 慢 (+46%),量化层越多 prefill 越慢
- 方案: prefill (M>8) 时把 int8 权重临时反量化成 BF16,走 `F.linear` (BF16 GEMM);decode (M≤8) 仍走 int8 GEMV
- 权重始终只存 int8 一份,反量化结果懒缓存
- 效果: TTFT 从 106.7 ms 回落到 72.1 ms

## 5.4 分派逻辑 (核心代码)

文件: `/usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/w8a8_int8.py`

```python
# w8a8_int8.py apply()
if SGLANG_SAIL_PREFILL_BF16 and _is_ppu and bias is None and x.shape[0] > M_MAX:
    # prefill: 反量化 + BF16 GEMM (快)
    w_bf16 = layer._w_bf16_cache or dequant(layer.weight)
    return F.linear(x, w_bf16)

x_q, x_scale = per_token_quant_int8(x)

if SGLANG_SAIL_DEEPGEMM_DENSE and _is_ppu and bias is None:
    if SGLANG_SAIL_USE_GEMV and x_q.shape[0] <= M_MAX:
        # decode: int8 GEMV (Gemvt, 快)
        gemv_int8_i8i8bf16(...)
    else:
        gemm_nt_i8i8bf16(...)
```

## 5.5 ACEXT (PPU 原生 INT8 GEMM)

- `sgl_kernel.int8_scaled_mm` → 底层 `acext.int8_gemm` (CUTLASS, BLOCK_M ≥ 16)
- 与 GEMV **互斥**(开 ACEXT 后不再走 GEMV)
- 比 GEMV 又快 **2.4%** (423 vs 418 tok/s)
- 开启: `SGLANG_SAIL_USE_ACEXT_CUDA=1` (默认关)

## 5.6 环境变量总表

| 变量 | 默认 | 说明 |
|---|---|---|
| `SGLANG_SAIL_FLA_CUDA` | 1 (wrapper 强制) | PPU GDN CUDA kernel 启用 |
| `SGLANG_SAIL_DEEPGEMM_DENSE` | 1 | DeepGEMM 路径启用 (INT8 only) |
| `SGLANG_SAIL_USE_GEMV` | 1 (wrapper) | decode 小 M 走自研 GEMV |
| `SGLANG_SAIL_GEMV_M_MAX` | 8 | GEMV/GEMM 分派阈值 |
| `SGLANG_SAIL_PREFILL_BF16` | 1 (wrapper) | prefill 反量化走 BF16 GEMM |
| `SGLANG_SAIL_USE_ACEXT_CUDA` | 0 | ACEXT INT8 GEMM (与 GEMV 互斥) |
| `SGLANG_SAIL_GEMV_WARMUP_ON_LOAD` | 1 (wrapper) | 每层加载后自动 warmup |

## 5.7 修改文件清单

按原仓库路径保留在 `gemv_optimization/`:

```
gemv_optimization/
├── DeepGEMM-for-sail/
│   └── deep_gemm/
│       ├── __init__.py                          # 导出 gemv_int8_int8_bf16_nt
│       └── jit_kernels/
│           ├── __init__.py                      # 导出
│           └── gemv_int8.py                    # GEMV 主文件
└── sglang-for-sail/
    └── python/sglang/srt/
        ├── environ.py                           # 3 个新环境变量
        └── layers/
            ├── quantization/
            │   └── w8a8_int8.py                 # GEMV custom op + apply 分派 + prefill 反量化
            └── deep_gemm_wrapper/
                ├── compile_utils.py             # GEMV kernel 枚举 + warmup executor
                └── entrypoint.py                # gemv_nt_i8i8bf16 wrapper
```

## 5.8 量化决策树

```
需要量化吗?
├── 不需要 → G 组 bf16 (§17) — 简单、稳、acc 0.81+
│
└── 需要 → 接受 ~5pp accuracy 损失换 +15% 吞吐量?
    ├── 不接受 → 退回 G 组
    └── 接受 → S2-aggressive 量化 + ACEXT + Piecewise CUDA Graph
        ├── 启动命令: 见 README_REPRODUCE.md §2.2
        └── 注意: TTFT 稳态 ~78ms (不是均值),acc 0.76-0.80
```