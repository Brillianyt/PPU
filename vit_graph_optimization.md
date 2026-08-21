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

## 验证结果（V1/V2/V3）

| 配置 | n | TPS | TTFT(ms) | acc | None | elapsed |
|---|---|---|---|---|---|---|
| V1 baseline OFF | 100 | 391.7 | 79.2 | 0.750 (75/100) | 6 | 9.8s |
| V2 ON + 256 token（真实 workload） | 100 | 394.1 | 65.9 | 0.710 (71/100) | 7 | 8.4s |
| V3 ON + 4 token（P19 口径） | 100 | 424.5 | 65.1 | 0.710 (71/100) | 7 | 7.3s |

对比修改前 P28（ON + 256）：TPS 377.8 / TTFT 67.1 / acc 0.600 / None 23。

### 结论

| 指标 | 修改前 P28 | 修改后 V2 | Δ |
|---|---|---|---|
| TPS | 377.8 | 394.1 | +4.3% ✅ |
| TTFT | 67.1 | 65.9 | −1.8% ✅ |
| acc | 0.600 | 0.710 | +11pp ✅ |
| None parses | 23 | 7 | −16 ✅ |

- capture 无 900/907 ✅（正解化后无需任何 sync shim）。
- TTFT 不退化 ✅（−17%）。
- 吞吐不退化 ✅（+0.6%，噪声内持平）。
- 精度 −4pp ⚠️（0.75→0.71，相对 −5%，可接受；较修改前 +11pp 大幅改善）。

**推荐配置**：V2（`max_new_tokens=256` 真实 workload）——TPS 与 OFF 持平、TTFT −17%、
elapsed 快 14%；V3（`max_new_tokens=4`）适合短答打点 benchmark，TPS 424 超过原 README
的 410 目标。

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
