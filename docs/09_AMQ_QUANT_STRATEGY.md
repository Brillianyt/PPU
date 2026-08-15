# 09. AMQ 量化策略记录

> 日期:2026-08-15
> 工具:QLean(主办方量化工具)
> 模型:Qwen3.5-2B(原始 BF16)
> 验证:transformers 反量化路径 + benchmark_public.py(20/30 样本)

---

## 1. 目标

在不损失 mmbench 精度的前提下,最大化 INT8 量化率(量化层数 / 总 Linear 层数),为推理后端(SGLang w8a8_int8 + DeepGEMM)提供高精度量化模型。

## 2. AMQ 敏感度分析

对 175 个 2D Linear 权重做 INT8 重建误差(`amq_results/sensitivity.json`),按层类型分组:

| 层类型 | mean 误差 | 敏感性 |
|---|---|---|
| `linear_attn.in_proj_a/b`(GDN 状态投影) | **0.0154** | 最高 |
| `self_attn.v_proj` | 0.0131 | 高 |
| `self_attn.k_proj` | 0.0128 | 中 |
| `self_attn.o_proj` / `mlp.down_proj` | ~0.011 | 低 |
| `self_attn.q_proj` | 0.0100 | 低 |
| `linear_attn.in_proj_qkv/z` | **0.0096** | 很低 |
| `mlp.gate/up` | **0.0092** | 最低 |

误差 > 0.015 的 23 层中,18 层是 `in_proj_a/b`,其余是 19/23 层的 `v/k_proj`。

## 3. 三种策略

| 策略 | recipe 文件 | int8 层数 | 保护(BF16)的层 |
|---|---|---|---|
| v3(保守) | `recipes/qwen3.5-2b_amq_qkv_T015_v3.yaml` | 42 | GDN 环路全部 + gate/up + 19/23 qkv |
| **S2(推荐)** | `recipes/qwen3.5-2b_amq_s2_aggressive.yaml` | **126** | in_proj_a/b + out_proj + 19/23 qkv |
| S3(全量化) | `recipes/qwen3.5-2b_amq_s3_full.yaml` | 186 | 仅 norms/embed/lm_head 等不可量化层 |

策略设计原则:
- **S2** 保护误差最高的 `in_proj_a/b`(recurrent 状态投影)、`out_proj`(GDN 环路输出)和 19/23 qkv(误差 > 0.015),其余全部 int8。
- **S3** 不做敏感层保护,作为精度边界对照。

## 4. 精度验证(transformers 反量化,权重 int8 → BF16,激活保持 BF16)

### 20 样本

| 策略 | en | cn |
|---|---|---|
| BF16 baseline | 80% (16/20) | 90% (18/20) |
| v3 | 80% | 90% |
| S2 | 80% | 90% |
| S3 | 80% | 85% ⚠️ |

### 30 样本(稳健性验证)

| 策略 | en | cn |
|---|---|---|
| BF16 baseline | 70.0% (21/30) | 83.3% (25/30) |
| S2 | 73.3% (22/30) | 83.3% (25/30) |
| S3 | 70.0% (21/30) | 83.3% (25/30) |

## 5. 结论

1. **2B 小模型对 INT8 量化不敏感**。最大重建误差 1.9%,在 mmbench 30 样本上,三种策略与 BF16 精度无统计显著差异(±1 题波动)。
2. **S3 全量化(186 层)在 30 样本上精度无损**,但 20 样本 cn 出现过 5pp 波动。为了私有大测试集的安全边际,**推荐 S2(126 层)**——保留最敏感层保护,量化率仍有 68%。
3. S2 与 S3 的 30 样本精度相同,差异只在风险偏好:激进选 S3,稳健选 S2。

## 6. 复现命令

```bash
# 量化(以 S2 为例)
qlean \
  --model_name Qwen/Qwen3.5-2B \
  --model_path /mnt/workspace/Qwen3.5-2B \
  --save_path /mnt/workspace/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --recipe /mnt/workspace/PPU/recipes/qwen3.5-2b_amq_s2_aggressive.yaml \
  --output-format INT8_W8A8 \
  --log-level WARNING

# 反量化精度验证(en 30)
python3 scripts/bench_quant.py \
  --bf16-model /mnt/workspace/Qwen3.5-2B \
  --quant-dir /mnt/workspace/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --output result_S2_en30.json \
  --num-samples 30
```

## 7. 文件清单

| 文件 | 用途 |
|---|---|
| `recipes/qwen3.5-2b_amq_qkv_T015_v3.yaml` | v3 保守策略 |
| `recipes/qwen3.5-2b_amq_s2_aggressive.yaml` | **S2 推荐策略** |
| `recipes/qwen3.5-2b_amq_s3_full.yaml` | S3 全量化边界 |
| `../../amq_results/sensitivity.json` | 175 层 INT8 重建误差 |
| `../../amq_results/result_S2_en30.json` 等 | 精度验证结果 |
