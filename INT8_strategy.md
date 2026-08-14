# Qwen3.5-2B INT8 / 混合精度量化策略文档

> 记录时间：2026-08-15
> 工具：qlean (QLean-for-SAIL v0.2.0+v0.1.0.ppu2.1.1)
> 镜像内已有的两份产物：`/mnt/workspace/Qwen_quant/Qwen3.5-2B-INT8/`、`/mnt/workspace/Qwen_quant/Qwen3.5-2B-MIXED/`
> **当前状态**：sglang 端 INT8 实测有 garbled output 问题，**不能上竞赛**；混合精度尚未跑通

---

## 0. TL;DR（务必先看）

| 结论 | 详情 |
|---|---|
| 1. **INT8 在 transformers 端声称 70% 精度** | 这是 qlean 文档/历史基准的说法，不是我们本次 sglang 实测 |
| 2. **INT8 在 sglang 端实测失败** | 用 `--quantization w8a8_int8`（DeepGEMM 路径）50 题全部输出乱码（"eneene个个…"），不能答题 |
| 3. **混合精度（MIXED）尚未在 sglang 端跑通** | `config.json` 缺 `quantization_config` 字段，sglang 会按 BF16 加载（浪费 MXFP4 部分） |
| 4. **当前 G 组（FA3+FLA=1+no-radix+page=64）仍是首选** | accuracy 0.85，TTFT 64.7ms，throughput 369 tok/s |

**建议**：INT8 路线**继续 debug 不要急着上**；如果修不通，备选方案是 BF16 baseline 的 page-size / chunked-prefill 进一步调优。

---

## 1. 量化产物清单

`/mnt/workspace/Qwen_quant/` 下两份由 qlean 生成的模型：

```
Qwen_quant/
├── Qwen3.5-2B-INT8/          # 纯 W8A8 (per-channel weight + per-token activation)
│   ├── config.json           # 含 quantization_config（int-quantized 格式）
│   ├── qlean_config.json     # 量化元数据 + 实际 ignore 列表（已读到）
│   └── model.safetensors-... # int8 权重 + per-channel scale
│
└── Qwen3.5-2B-MIXED/         # AMQ 风格 per-tensor 三段混精度 (INT8+BF16+MXFP4)
    ├── config.json           # ❌ 没有 quantization_config 字段（见 §5）
    ├── amq_config.json       # AMQ 元数据（已读到）：INT8=58, BF16=83, MXFP4=34
    └── model.safetensors-...
```

**两份都是基于本地 `/mnt/workspace/Qwen3.5-2B/` 导出的，参数全部正确，差异在量化策略**。

---

## 2. INT8 量化方案

### 2.1 W8A8 配置（来自 `Qwen3.5-2B-INT8/config.json` 的 `quantization_config`）

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

| 维度 | 权重 | 激活 |
|---|---|---|
| 位宽 | 8 | 8 |
| 量化粒度 | per-channel (out_dim) | per-token (动态) |
| 对称 | symmetric | symmetric |
| scale | static_minmax | 推理时动态算 |

### 2.2 不量化的层（来自 `qlean_config.json`，17 条 regex）

#### A. 标准不能量化
- `lm_head`（输出层，精度保护）
- `embed_tokens`（嵌入查找）
- `mlp.gate` / `mlp.gate_proj`
- `norm.weight` / `norm1.weight` / `norm2.weight`（RMSNorm 1D）
- `q_norm.weight` / `k_norm.weight`（QK-norm 1D）

#### B. Linear Attention 内部
- `linear_attn.A_log`（1D 衰减对数）
- `linear_attn.dt_bias`（1D 时间步偏置）
- `linear_attn.conv1d.*`（3D 卷积权重）

#### C. AMQ 灵敏度 + 用户提示保持 BF16
- `self_attn.v_proj`：AMQ Fig 12 所有 Pareto 最优配置中 V 层始终最高精度
- `linear_attn.in_proj_qkv`：q_k+q_v 重操作符；qlean 不能单 Linear 拆精度 → 整层 BF16
- `linear_attn.in_proj_z`：q_j (→ z) 重操作符；同上 → 整层 BF16

#### D. 与文本推理无关
- `mtp.*`（Multi-Token Prediction，推理不启用）
- `visual.*`（Vision encoder）

### 2.3 与 BF16 的对比

| 指标 | BF16 | INT8 |
|---|---|---|
| 模型大小 | 4.55 GB | 3.79 GB（-16.7%） |
| 量化 Linear 数量 | 0 | 175 - 17 = 158 |

---

## 3. sglang 端实测（2026-08-15）

### 3.1 启动命令

```bash
nohup env SGLANG_SAIL_FLA_CUDA=1 \
  python3 -u -m sglang.launch_server \
    --model-path /mnt/workspace/Qwen_quant/Qwen3.5-2B-INT8 \
    --host 127.0.0.1 --port 8999 \
    --trust-remote-code --tp-size 1 \
    --attention-backend fa3 \
    --enable-multimodal \
    --quantization w8a8_int8 \
    --dtype bfloat16 \
    --mem-fraction-static 0.85 \
    --disable-radix-cache \
    --page-size 64 \
    --log-level info \
  > /mnt/workspace/eval_logs/sglang_int8_w8a8_8999.log 2>&1 &
```

启动日志关键行（确认走 DeepGEMM）：
```
[INFO] Use DeepGEMM for fp8/bf16/int8 mm on PPU
[INFO] DeepGEMM warmup done
[INFO] Loading model weights: 3.79 GB
[INFO] KV cache: ~40 GB
[INFO] Application startup complete.
```

### 3.2 结果（50 题）

输出文件：`/mnt/workspace/dndx_participant-v1.2/result_dev_en_int8_new_50.json`

| 指标 | 数值 |
|---|---|
| 样本数 | 50 |
| **parsed_answer 非 null** | **0 / 50** |
| accuracy | 0.00 |
| avg_ttft_ms | 97.3 |
| avg_throughput_tokens_per_sec | 324.6 |
| finish_reason | 全部 `length`（触发 max_tokens=256 上限） |

**代表性乱码输出**：
```
"eneene个个当当当当当当当当当当当家家家家家家家家家家家家家家、、、、、、、、、、道道当当当当当当当当当当当当当当当当"
```

### 3.3 根因诊断（待解）

1. **不是 sglang 服务没起来** — DeepGEMM warmup 通过，HTTP 200 OK
2. **不是图像处理** — `/curl -F image=...` 直接发图也返回乱码
3. **不是 radix/page-size 配置** — 同样这套配置对 BF16 跑得好好的（accuracy 0.85）
4. **可能是**：
   - (a) DeepGEMM 对 qlean 输出格式兼容性 bug（不是标准 compressed-tensors 权重布局）
   - (b) FLA=1 + DeepGEMM INT8 联合的精度退化（已有 ignore 列表但仍可能漏层）
   - (c) qlean v0.2.0 输出的 `model.safetensors` 索引缺少 DeepGEMM 期望的某个字段

### 3.4 与文档历史声称的"70% 精度"矛盾

qlean 文档 / 历史基准（`transformers` 后端）声称 INT8 与 BF16 一致 70%，但：

- 那是 `transformers.AutoModelForImageTextToText` + `compressed-tensors` 反量化路径
- 我们的 `benchmark_public.py` 是 **sglang OpenAI 兼容 API**
- sglang 在 PPU 上 INT8 只支持 DeepGEMM（cutlass int8_scaled_mm `NotImplementedError`）
- DeepGEMM 是**纯 kernel 推理**，不走 compressed-tensors 的 Python 层

所以"70% vs 0%"的对比实际上对比的是**两条完全不同的 INT8 推理路径**，不是同一份权重的精度差异。**transformers 路径测的精度不能直接迁移到 sglang 路径**。

---

## 4. 必踩的坑（修复记录）

### 4.1 preprocessor_config.json 不被 sglang v0.5.10 识别

启动报：
```
ValueError: Unrecognized image processor in /mnt/workspace/Qwen_quant/Qwen3.5-2B-INT8
```

qlean 输出的 `preprocessor_config.json` 用了 **`Qwen2VLImageProcessorPil`**（transformers v4.58+ 才有的新类），但 sglang v0.5.10 内置 transformers 4.57 不识别。

修复：
```bash
sed -i 's/"Qwen2VLImageProcessorPil"/"Qwen2VLImageProcessorFast"/' \
  /mnt/workspace/Qwen_quant/Qwen3.5-2B-INT8/preprocessor_config.json
sed -i 's/"Qwen2VLImageProcessorPil"/"Qwen2VLImageProcessorFast"/' \
  /mnt/workspace/Qwen_quant/Qwen3.5-2B-MIXED/preprocessor_config.json
```

> 同样的脚本应用到 MIXED 副本（尚未跑通，见 §5）。

### 4.2 cutlass int8_scaled_mm 在 PPU 未实现

默认 `--quantization compressed-tensors` 报：
```
NotImplementedError: No implemented cutlass int8_scaled_mm on PPU, use DEEPGEMM please
```

→ 必须用 `--quantization w8a8_int8` 才走 DeepGEMM。

### 4.3 compressed-tensors Python 反量化 ≠ DeepGEMM 实时量化

transformers 后端每个 forward 反量化权重再算 matmul（带宽密集），结果可能准；
sglang DeepGEMM 是直接用 int8 权重 + dynamic per-token scale 算，结果可能完全不同。

---

## 5. MIXED 模型（AMQ 三段混精度）— 尚未跑通

### 5.1 amq_config.json 内容

```json
{
  "qlean_version": "AMQ-mixed-precision-v2",
  "strategy": "per-tensor INT8/MXFP4/BF16 Pareto-frontier selection",
  "thresholds": { "mxfp4_acceptable_max": 0.14, "int8_acceptable_max": 0.012 },
  "format_distribution": { "INT8": 58, "BF16": 83, "MXFP4": 34 },
  "n_linear_tensors": 175,
  "size_bytes": 3978397760   # ≈ 3.71 GB
}
```

175 个 Linear tensor 中：58 个用 INT8，34 个用 MXFP4，83 个保持 BF16（极敏感层）。

### 5.2 sglang 不识别 MXFP4

**sglang 0.5.10 内置的量化 flag 不包含 `amq`、`mxfp4`、`w4a8_mxfp4`**，唯一认识的就是 `w8a8_int8`。结果：

- 如果直接用 `--model-path Qwen3.5-2B-MIXED`（不指定 quantization），sglang 把它当 **BF16** 加载
- 这浪费了 58 个 INT8 线性层和 34 个 MXFP4 线性层的压缩
- 而且 MXFP4 不是 sglang 0.5.10 的 supported format，会**直接失败**（即使强行标记 quantization）

**根本问题**：`Qwen3.5-2B-MIXED/config.json` 没有 `quantization_config` 字段，sglang 不知道怎么处理。

### 5.3 可行的 workaround（待尝试）

**方案 A：写一个空的 quantization_config 触发 sglang 把整模型当 INT8 处理**
- 风险：MXFP4 部分会以 INT8 计算 → 精度更差
- 不推荐

**方案 B：把 MXFP4 层反量化回 BF16**
- 需要 qlean 提供 mx → bf16 的恢复工具
- 跑出来后是 "INT8 + BF16 混合"，不是真正的 "INT8 + MXFP4 + BF16" 三段
- 但 sglang 走 `compressed-tensors` 路径可识别（前提是 cutlass int8_scaled_mm 在 PPU 实现 → ❌ 仍然不行）

**方案 C：放弃 MIXED，走纯 INT8**
- 这就是 §3 的实测，已经 0% 精度
- 需要继续 debug

**方案 D：等待 sglang-for-SAIL v0.5.13 的 MXFP4 支持**
- 用户指南里提到 v0.5.13 增加了 MXFP4
- 但我们当前实例是 v0.5.10，v0.5.13 还在调研

**当前决策**：MIXED 暂时不上竞赛。

---

## 6. 关键文件清单（**已修正路径**）

| 文件 | 用途 | 实际状态 |
|---|---|---|
| `/mnt/workspace/Qwen_quant/Qwen3.5-2B-INT8/config.json` | 输出模型 config（含 `quantization_config`） | ✓ 存在 |
| `/mnt/workspace/Qwen_quant/Qwen3.5-2B-INT8/qlean_config.json` | 量化元数据 + 实际 ignore 列表 | ✓ 存在（已读） |
| `/mnt/workspace/Qwen_quant/Qwen3.5-2B-MIXED/config.json` | MIXED 模型 config | ✓ 存在（**缺 quantization_config**） |
| `/mnt/workspace/Qwen_quant/Qwen3.5-2B-MIXED/amq_config.json` | AMQ 元数据 | ✓ 存在（已读） |
| `/mnt/workspace/dndx_participant-v1.2/result_dev_en_int8_new_50.json` | INT8 50 题 sglang 基准 | ✓ 0/50 parsed |
| `/mnt/workspace/dndx_participant-v1.2/result_dev_en_sglang_page64_20.json` | G 组 20 题 BF16 基准 | ✓ 0.85 accuracy |

---

## 7. 下一步建议（优先级排序）

1. **(P0) 排查 INT8 garbled output 根因**
   - 方法：拿 `model.safetensors` 的某个 Linear 权重（如 `model.layers.0.self_attn.q_proj.weight`），手算一份 INT8 量化-反量化结果，与 safetensors 的 int8 + scale 反量化结果对比
   - 如一致 → DeepGEMM kernel 有 bug
   - 如不一致 → qlean 导出格式错了

2. **(P1) 在 PPU 镜像里跑 `transformers` 路径**
   - 装 `compressed-tensors>=0.15.0`，确认 INT8 在 transformers 后端真能保 70%
   - 把 transformers 跑通的精度作为 sglang 路径的理论上限

3. **(P2) 探索 BF16 的进一步优化**
   - 当前 G 组 0.85 / 64.7ms / 369 tok/s
   - 还有空间：chunked-prefill、speculative decoding、different page-size

4. **(P3) MIXED 模型待 sglang-for-SAIL v0.5.13**

---

## 8. 重要约束提醒（来自 README）

| 禁令 | 我们的状态 |
|---|---|
| 禁止更换模型权重、蒸馏模型、量化后使用非主办方认可的权重包 | ✓ 用主办方提供的 qlean 工具生成 |
| 禁止查表返回答案 | ✓ 不涉及 |
| 禁止伪造 token_count / ttft_seconds / elapsed_seconds | ✓ 从 sglang usage 真实拿 |
| 准确率和性能评测拆成两套不同策略 | ✓ 没拆 |
| 评测机持久化缓存跨次复测获利 | ✓ 没做 |

**本次 0% 精度不能上竞赛** — 必须把 garbled output 修好才能切换到 INT8。