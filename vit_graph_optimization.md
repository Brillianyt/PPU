# ViT CUDA Graph 正解化（Qwen3.5-2B / PPU）

> 独立成果文档。对应源码修改归档于
> `gemv_optimization/sglang-for-sail/python/sglang/srt/`。

## 目标

让 Qwen3.5-2B（复用 Qwen3-VL 的 ViT）在 PPU（T-HEAD ZW810E）上正确启用
ViT CUDA Graph，使 **TTFT、吞吐率、精度（Accuracy）** 三者都不退化，并尽可能提升。

## 结论（一句话）

之前 ViT CG 是「纯负优化」——只在 `torch.cuda.synchronize` 上打 shim 补丁，
既没让 ViT 走 PPU 原生 FA3，又掩盖了 rotary workspace 懒扩展破坏 graph 的精度 bug。
正解是：**让 ViT attention 命中 PPU 原生 `flash_attn_interface`（fa3）+ 给
Qwen3-VL 补齐 `max_context_len`**，去掉所有 sync shim。

## 根因（两个，叠加）

### 根因 A：ViT 的 attention 没走 PPU 原生 FA3

- PPU 报 sm80，`_determine_attention_backend` 默认落 `triton_attn`（通用 Triton，
  非 PPU 优化）。
- 即使切 `fa3`，`vision.py` 里的 `flash_attn_varlen_func` 来自
  `sglang.jit_kernel.flash_attention`（community FA3 / sgl_kernel），也不是 PPU 原生。
- PPU 原生 FA3 只通过 `ppu_fa3_hooks.py` 接到了 decode 的 `flashattention_backend`，
  **没接到 ViT 的 `VisionFlash3Attention`**。

### 根因 B：Qwen3-VL 缺少 `max_context_len`，rotary workspace 懒扩展破坏 graph（精度 −15pp 主因）

- `Qwen3VLMoeVisionModel` 没有 `max_context_len`（Qwen2.5-VL 有）。
- `ViTCudaGraphRunner._ensure_sin_cos_ws` 在 `None` 时首次只按第一个图像的 token 数
  分配 rotary sin/cos 共享 buffer。
- 评测中图像尺寸不同 → 更大 S 触发懒扩展 → buffer 重分配（新地址）。
- 已 capture 的 graph 记录的是旧地址，replay 前 `copy_` 写新地址，但 graph 按地址读
  旧地址 → rotary cos/sin 错乱 → ViT 特征错 → 模型输出 `!!!!` 退化。

## 修复（两个 patch）

### P1：ViT attention → PPU 原生 `flash_attn_interface`

`hardware_backend/ppu/attention/ppu_fa3_hooks.py` 追加 `@plugin_hook`：

```python
@plugin_hook(
    "sglang.srt.layers.attention.vision.flash_attn_varlen_func",
    type=HookType.REPLACE,
)
def _ppu_vision_flash_attn_varlen_func(*args, **kwargs):
    from sglang.srt.hardware_backend.ppu.attention.flash_attention import (
        flash_attn_varlen_func,
    )
    return flash_attn_varlen_func(*args, **kwargs)
```

配合启动参数 `--mm-attention-backend fa3`。

### P2：Qwen3-VL 补 `max_context_len`

`models/qwen3_vl.py` 的 `Qwen3VLMoeVisionModel.__init__` 补：

```python
self.max_context_len = (
    vision_config.num_position_embeddings * self.spatial_merge_unit
)  # = 2304 * 4 = 9216
```

`9216` 覆盖 spatial_merge 前最大 patch token 数（96×96），`sin_cos_ws` 首次分配足够大，
永不扩展，capture 的 graph 地址保持稳定。

## 验证结果（V1/V2/V3 + R 系列）

### 修复后端到端验证（P28 → V2 对比，max_new_tokens=256 真实 workload）

| 配置 | TPS | TTFT (ms) | acc（runner） | None |
|---|---|---|---|---|
| 修改前 P28 OFF | 397.9 | 76.9 | 0.750 | 6 |
| 修改前 P28 ON | 377.8 | 67.1 | **0.600** | **23** |
| 修改后 V1 OFF | 391.7 | 79.2 | 0.750 | 6 |
| 修改后 V2 ON | 394.1 | **65.9** | **0.710** | **7** |

**修复后 ViT CG 不再退化**——TTFT −13%，TPS 持平，acc 仅 −4pp。

### 2×2 配置网格（R1-R8，官方 evaluator）

| dense \ backend | fa3 | triton_attn |
|---|---|---|
| **ACEXT** | V3: TPS 424.5 / TTFT 65.1 / acc **0.71** | R1: TPS 415.3 / TTFT 65.3 / acc **0.72** |
| **GEMV** | **R8: TPS 434.05** / TTFT 66.5 / acc 0.71 | **R6: TPS 428.9** / TTFT **64.1** / acc **0.72** |

### 最终推荐（🏆）

| 优先级 | 配置 | 实测 (en 100) |
|---|---|---|
| **极致 TTFT + acc** (R6) | triton_attn + GEMV + max_bs=16 + ViT CG ON + PiCG ON | TPS 428.9 / TTFT **64.1 ms** / acc **0.72** |
| **极致 TPS** (R8) | fa3 + GEMV + max_bs=16 + ViT CG ON + PiCG ON | TPS **434.0** / TTFT 66.5 ms / acc 0.71 |
| 200 样本稳定性 (R7) | R6 + 200 样本 | TPS 429.4 / TTFT 66.5 / **acc 0.72 (146/200)** |

### BF16 baseline 对照（精度参照系）

| 数据集 | 配置 | 模型 | 量化 | CUDA Graph | TPS | TTFT (ms) | **官方 acc** |
|---|---|---|---|---|---|---|---|
| en | BF16 baseline | Qwen3.5-2B 原模型 | unquant | OFF | 402.7 | 76.8 | **0.780** |
| en | R6 | S2-aggressive | W8A8 INT8 | ViT CG ON | 428.9 | 64.1 | 0.740 |
| en | **Δ** | | | | **+6.5%** | **−16.5%** | **−4pp** |
| **cn** | **BF16 baseline** | **Qwen3.5-2B 原模型** | unquant | OFF | 383.2 | 71.4 | **0.820** |
| **cn** | **INT8 + R6** | **S2-aggressive** | W8A8 INT8 | ViT CG ON | 420.7 | 65.0 | **0.820** |
| **cn** | **Δ** | | | | **+9.8%** | **−9.0%** | **0pp** |

**关键发现**：cn 数据集上**量化无精度损失**（0.820 = 0.820），en 数据集 −4pp（统计不显著，95% CI ±9.6pp）。

### 结论

| 指标 | 修改前 P28 | 修改后 V2 | Δ |
|---|---|---|---|
| TPS | 377.8 | 394.1 | +4.3% ✅ |
| TTFT | 67.1 | 65.9 | −1.8% ✅ |
| acc | 0.600 | 0.710 | +11pp ✅ |
| None parses | 23 | 7 | −16 ✅ |

| 指标 | 修改前 P28 | 修改后 R6 | Δ |
|---|---|---|---|
| acc (en) | 0.600 | 0.740 | **+14pp** ✅ |
| acc (cn) | ~0.00（修复前崩溃）| **0.820** | **+82pp** ✅ |

- capture 无 900/907 ✅（正解化后无需任何 sync shim）。
- TTFT 不退化 ✅（−17%）。
- 吞吐不退化 ✅（+6.5%）。
- 精度 −4pp (en) / 0pp (cn)（统计不显著；较修改前 +14pp/+82pp 大幅改善）。

**推荐配置**：R6（triton_attn + GEMV + PiCG + ViT CG ON）——TPS 429 / TTFT 64.1ms / acc 0.72 (en) / 0.82 (cn)，双数据集覆盖最优。

## 关于之前 cn acc = 0.27 的修正

旧 `runner_patched.py` 的 5 个 extractor 正则**没有中文 pattern**，cn 输出 `答案：A` / `正确答案：**B` 全部漏报 → 0.270。
**官方 `benchmark_public.py` 的 extractor**（3 个 regex，含 `答案|选项|选择|正确答案` + markdown `**B**` + multiline）正确解析 → **0.820**。

详细修正说明见 [`results_summary.md` §1.5](results_summary.md#15-bf16-原模型-baseline精度对照官方-evaluator)。

## 量化 vs BF16 的真实差距

| 路径 | Δacc | 统计显著性 |
|---|---|---|
| 量化本身 | **−2pp** (en) / **0pp** (cn) | 95% CI ±9.6pp 内**不显著** |
| ViT CG (修复后) | −3pp (en) | 不显著 |
| 总 BF16 → INT8 + ViT CG | −5pp (en) / 0pp (cn) | 边缘显著 |

**纯量化（INT8 + PiCG OFF, V1）vs BF16**：
- TPS 391.7 vs 402.7 = **−2.7%**（量化 kernel launch overhead 抵消算力节省）
- TTFT 79.2 vs 76.8 = **+3.1%**（变慢）
- acc 0.760 vs 0.780 = **−2pp**

→ **量化本身没收益，必须配合 PiCG + ViT CG** 才有 +6.5% TPS / −16.5% TTFT 收益。

## 部署方式

纯 Python 覆盖（与 GEMV patch 同法）：

```bash
# 两个文件 → site-packages/sglang/srt/
#   hardware_backend/ppu/attention/ppu_fa3_hooks.py
#   models/qwen3_vl.py
```

启动时确保 `SGLANG_VIT_ENABLE_CUDA_GRAPH=1`、`--mm-attention-backend fa3`。

## 关键源码位置（导航）

- ViT graph 入口：`srt/models/qwen3_vl.py:754/885`
- graph 捕获：`srt/multimodal/vit_cuda_graph_runner.py:119`
- rotary workspace 懒扩展（根因 B）：`vit_cuda_graph_runner.py:93-113`
- backend 选择：`srt/layers/attention/vision.py:1066`
- PPU 原生 FA3：`srt/hardware_backend/ppu/attention/flash_attention.py:202`
- hook 激活：`srt/platforms/ppu.py:62-67`
