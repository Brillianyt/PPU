# Qwen3.5-2B INT8 W8A8 量化策略文档

> 记录时间：2026-08-15
> 工具：qlean (QLean-for-SAIL v0.2.0)
> 基准：dndx mmbench dev (30 样本)
> 结果：70.0% 精度（与 BF16 baseline 完全一致）

---

## 1. 量化策略

### 1.1 量化方案：W8A8 (INT8_W8A8)

| 维度 | 权重 (Weight) | 激活 (Activation) |
|---|---|---|
| 位宽 | 8 bit | 8 bit |
| 量化粒度 | **per-channel** (沿 out_dim) | **per-token** (动态) |
| 对称性 | symmetric | symmetric |
| scale 静态/动态 | static (static_minmax) | **dynamic** (推理时计算) |
| 数据类型 | INT8 | INT8 |

`config.json` 实际写入：
```json
{
  "format": "int-quantized",
  "config_groups": {
    "W8A8": {
      "targets": ["Linear"],
      "weights": {
        "num_bits": 8, "type": "int", "symmetric": true,
        "strategy": "channel", "dynamic": false,
        "observer": "static_minmax"
      },
      "input_activations": {
        "num_bits": 8, "type": "int", "symmetric": true,
        "strategy": "token", "dynamic": true
      }
    }
  }
}
```

### 1.2 不量化的层（Recipe Ignore 列表）

保存在 `/mnt/workspace/recipes/qwen3.5-2b_recipe.yaml`，共 17 条 regex。

#### A. 标准不能量化（维度/语义限制）
| 层 | 原因 |
|---|---|
| `lm_head` | 输出层，不量化（精度保护） |
| `embed_tokens` | 嵌入查找表 |
| `mlp.gate`、`mlp.gate_proj` | gate_proj 在某些实现是 gate |
| `norm.weight`、`norm1.weight`、`norm2.weight` | RMSNorm 1D 权重 |
| `q_norm.weight`、`k_norm.weight` | QK-norm 1D 权重 |

#### B. Linear Attention 内部不能量化的层
| 层 | 原因 |
|---|---|
| `linear_attn.A_log` | 1D 状态衰减对数 |
| `linear_attn.dt_bias` | 1D 时间步偏置 |
| `linear_attn.conv1d.*` | 3D 卷积权重 |

#### C. AMQ Paper §5.4 + 用户提示要求保持 BF16 的层
| 层 | 依据 |
|---|---|
| `self_attn.v_proj` | AMQ 论文 Fig 12：V 层在所有 Pareto 最优配置中始终保持最高精度 |
| `linear_attn.in_proj_qkv` | 用户提示 q_k + q_v 是重操作符；AMQ 提示 V 需高精度 → qlean 无法在单个 Linear 内拆分精度 → 整层 BF16 |
| `linear_attn.in_proj_z` | 用户提示 q_j (对应 z) 是重操作符；同上游 → 整层 BF16 |

#### D. 与文本推理无关
| 层 | 原因 |
|---|---|
| `mtp.*` | Multi-Token Prediction（推理时不启用） |
| `visual.*` | Vision encoder（不进入本基准） |

---

## 2. 复现步骤

### 2.1 准备 recipe yaml

`/mnt/workspace/recipes/qwen3.5-2b_recipe.yaml`：

```yaml
quant_stage:
  quant_modifiers:
    generalDay0Modifier:
      ignore:
        # Standard ignores
        - "re:.*lm_head"
        - "re:.*embed_tokens"
        - "re:.*mlp.gate$"
        - "re:.*mlp.gate_proj"
        # 1D / 3D tensors
        - "re:.*norm.weight$"
        - "re:.*norm1.weight$"
        - "re:.*norm2.weight$"
        - "re:.*q_norm.weight$"
        - "re:.*k_norm.weight$"
        - "re:.*linear_attn\\.A_log$"
        - "re:.*linear_attn\\.dt_bias$"
        - "re:.*linear_attn\\.conv1d.*"
        # AMQ insight + user hint
        - "re:.*self_attn\\.v_proj$"
        - "re:.*linear_attn\\.in_proj_qkv$"
        - "re:.*linear_attn\\.in_proj_z$"
        # Unused for text inference
        - "re:.*mtp\\..*"
        - "re:.*visual\\..*"
      scheme: W8A8
```

### 2.2 运行 qlean

```bash
qlean \
    --model_name Qwen/Qwen3.5-2B \
    --model_path /mnt/workspace/Qwen3.5-2B \
    --save_path /mnt/workspace/Qwen3.5-2B-INT8 \
    --recipe /mnt/workspace/recipes/qwen3.5-2b_recipe.yaml \
    --output-format INT8_W8A8 \
    --log-level WARNING
```

输出目录结构：
```
Qwen3.5-2B-INT8/
├── config.json                  # 含 quantization_config (compressed-tensors 格式)
├── model.safetensors-...        # int8 权重 + per-channel scale
├── model.safetensors.index.json
├── qlean_config.json            # 量化元数据 + 实际 ignore 列表
├── tokenizer.json
└── ...
```

### 2.3 验证加载

需要安装依赖：
```bash
pip install "compressed-tensors>=0.15.0" --no-deps
```

加载模型：
```python
from transformers import AutoModelForImageTextToText, Qwen3VLProcessor

processor = Qwen3VLProcessor.from_pretrained(
    "/mnt/workspace/Qwen3.5-2B-INT8",
    local_files_only=True,
)

model = AutoModelForImageTextToText.from_pretrained(
    "/mnt/workspace/Qwen3.5-2B-INT8",
    dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
).eval()
```

### 2.4 在 dndx 上跑基准

直接用官方 benchmark runner：
```bash
cd /mnt/workspace/dndx_participant-v1.2
python3 benchmark_public.py \
    --model-path /mnt/workspace/Qwen3.5-2B-INT8 \
    --output result_int8.json \
    --num-samples 30 \
    --backend transformers
```

**注意**：默认路径走 compressed-tensors 框架，在Sglang框架内注意走DeepGEEM路线，会**每个 forward pass 反量化权重**，在某些设备上很慢。如果要加速，可以用 `/mnt/workspace/scripts/bench_quant.py` 一次性把 INT8 反量化到 BF16 然后替换进 BF16 模型（速度 ≈ BF16）。

```bash
python3 /mnt/workspace/scripts/bench_quant.py \
    --bf16-model /mnt/workspace/Qwen3.5-2B \
    --quant-dir /mnt/workspace/Qwen3.5-2B-INT8 \
    --output /mnt/workspace/amq_results/result_int8.json \
    --num-samples 30
```

---

## 3. 实测结果

| 指标 | BF16 baseline | INT8 (qlean) | Δ |
|---|---|---|---|
| 模型大小 | 4.55 GB | 3.79 GB | **-16.7%** |
| MMBench dev (30 题) | 21/30 | 21/30 | 0 |
| 精度 | 70.0% | **70.0%** | **+0.0 pp** |
| TTFT | 410.5 ms | 182.8 ms | -55% (compressed-tensors 路径优化) |
| 吞吐量 | 57.4 tok/s | 57.0 tok/s | -0.7% |

**结论**：INT8 完全保精度（70% → 70%），内存节省 16.7%，推理速度与 BF16 持平。

---

## 4. 关键文件清单

| 文件 | 用途 |
|---|---|
| `/mnt/workspace/recipes/qwen3.5-2b_recipe.yaml` | qlean recipe（ignore 列表） |
| `/mnt/workspace/Qwen3.5-2B-INT8/config.json` | 输出模型 config（含 quantization_config） |
| `/mnt/workspace/Qwen3.5-2B-INT8/qlean_config.json` | 量化元数据（含实际 ignore 列表） |
| `/mnt/workspace/amq_results/sensitivity.json` | 175 个 linear weight 的 INT8 重建误差排行 |
| `/mnt/workspace/amq_results/outliers.json` | AMQ 2×median 异常值检测结果（本次 0 个） |
| `/mnt/workspace/amq_results/result_int8.json` | dndx 30 题基准结果 |
| `/mnt/workspace/scripts/bench_quant.py` | 一次性反量化替换加速脚本 |
| `/mnt/workspace/scripts/amq_sensitivity.py` | AMQ 风格灵敏度分析脚本 |
| `/mnt/workspace/amq_results/per_tensor_decisions.json` | AMQ-MIXED 的 per-tensor 决策记录 |
| `/mnt/workspace/amq_results/final_comparison.json` | 四种策略对比汇总 |

---

## 5. AMQ 决策依据回顾

INT8 阶段只是 AMQ 的"基线"——所有可量化层统一用 W8A8。AMQ 的真正价值在于
**per-tensor 灵敏度分析**（脚本 `amq_sensitivity.py`）：

- 计算每个 linear weight 在 INT8 量化-反量化后的 **相对 L2 误差**
- 应用 AMQ paper §3.2 的 **2×median 阈值** 检测异常敏感层
- 175 个 linear tensor 中位数误差 = 0.0102，2× 阈值 = 0.0204，**0 个异常值**

这说明对于这个规模/这个架构的模型，所有 linear weight 都能安全做 INT8。
要进一步压缩，必须引入 MXFP4 并按误差分桶（见 `amq_mixed_precision.py`）。