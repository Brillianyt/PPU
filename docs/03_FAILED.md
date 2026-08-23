# 03. 否决实验

> 按「为什么否决」分类,不是按时间。**只保留对复测有信号的失败**——噪音实验、
> 早期探索过的方案不在此处。

## F1. EAGLE/NEXTN 投机解码 — **崩溃**

**症状**: sglang server 启动后 ~12 秒崩
```
File "sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py", line 339
  retrieve_parent_token.int(),
AttributeError: 'NoneType' object has no attribute 'int'
```

**根因**: Qwen3.5-2B 是 hybrid GDN 模型,linear_attention 层是 recurrent 状态机,
**没有显式 parent token 概念**。EAGLE/NEXTN verify 阶段需要 `retrieve_parent_token`
做 parent tracking,sglang 0.5.13+ppu2.1.1 的 `gdn_backend.target_verify()`
没为 hybrid 模型正确填 parent token,递归核期望非 None,直接 `.int()` 崩。

**试过的参数组合**(全部崩):
- `--speculative-algorithm EAGLE --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3`
- `--speculative-algorithm EAGLE + --speculative-draft-model-path ./Qwen3.5-2B`
- `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`

**FLA=off 绕过**:能跑但全维度劣于 G 组:

| 配置 | acc | TTFT | tput |
|---|---|---|---|
| G 组 (FLA=on) | 0.85 | 64.7 ms | 369 tok/s |
| NEXTN + FLA=off | 0.85 | 87.3 ms (+35%) | 275 tok/s (−25%) |

**结论**:
- bug 与具体参数无关 —— `target_verify` 进 GDN 递归核之前就把 `retrieve_parent_token`
  留成 `None`
- 短回答场景 (mmbench 256 token) 投机节省本来就少,decode 步数有限
- **Qwen3.5 + sglang 0.5.13 上 EAGLE/NEXTN 完全不可用**,即使带 MTP 头也不能

## F2. INT8 W8A8 量化 (sglang 0.5.10) — **garbage 输出**

**症状**: 模型输出乱码,多数请求 4-13 token 就停,重复问题文本
```
"B. Does Madelyn's snowboard slide down a hill in less time when it has a layer of wax or when it does not have a layer of wax?"
```

**根因**: sglang 0.5.10 的 `w8a8_int8.py` 在 PPU 上有 bug。

**修复**: 升级到 sglang **0.5.13** (修复)。0.5.13 后 INT8 输出正常,但有 ~5pp acc
损失(详见 [`02_RESULTS.md`](02_RESULTS.md) §2.4 真伪量化损失分解)。

## F3. DALI 图像预处理 — **0.72ms/张,无空间**

| 操作 | 耗时 |
|---|---|
| HF Qwen2VLImageProcessorFast (已在 GPU) | 0.72 ms/张 |
| DALI pipeline (替换) | 1.5 ms/张 (含 H2D / CPU↔GPU) |

**结论**: 主路径 `Qwen2VLImageProcessorFast` 已是 GPU 实现,不需要 DALI。

## F4. transformers 4.57.1 直接推理 — **不可用**

- `ValueError: Transformers does not recognize this architecture` (`model_type=qwen3_5`)
- 模型目录无 `modeling_*.py`,`trust_remote_code` 也无济于事
- 升级 transformers 受网络/时间约束 (本任务不允许 30 s+ 等待)

**结论**: 走 sglang 后端是唯一选择(`--backend sglang`)。

## F5. S3-full 全量化 (186 层) — **acc 太低**

| 方案 | acc | TTFT | tput |
|---|---|---|---|
| S3-full + GEMV | 0.650 | 104.0 ms | 372.3 tok/s |
| S3-full 无 GEMV | 0.650 | 80.4 ms | 273.3 tok/s |

**根因**: 误差最高的 GDN 层也被量化 → 破坏 recurrent 状态。

**结论**: **不能提交**。S2-aggressive (126 层量化, 49 层 BF16 保护) 是上竞赛的唯一量化选项。

## F6. S0/v3 (42 层 INT8) — **端到端无收益**

- microbench 上 GEMV 比 Triton 快 2.15× (13.0 μs vs 27.9 μs)
- 端到端 benchmark 与不量化持平
- **根因**: 覆盖率太低,GEMV 节省吃不下 wrapper overhead
- **改进**: S2 (126 层) + GEMV 才有端到端 +15% tput

## F7. AMQ 三段混精度 (INT8 + MXFP4 + BF16) MIXED — **sglang 加载直接崩**

**症状**: sglang 0.5.10 加载 MIXED 报 `AssertionError`,shape 不匹配。

```
File "sglang/srt/layers/linear.py", line 735, in weight_loader
    assert param_data.shape == loaded_weight.shape
AssertionError
```

**根因**:
- MXFP4 是 4-bit packed,每 byte 含 2 个值。Linear layer 期望 `2048×6144=12,582,912`
  个 bf16 值,safetensors 里只有 `6,291,456` 个 uint8 (每个 unpack 成 2 个 MXFP4)。
  shape `(2048, 6144)` vs `(2048, 3072)` → 不匹配
- sglang 0.5.10 PPU **不支持** `mxfp4`、`mixed_precision_w4` 量化路径

**结论**: **MIXED 暂时不上竞赛**。需等 sglang-for-SAIL v0.5.13+ 的 MXFP4 支持(本仓
已升级 0.5.13,但实测 MIXED 配置仍与 sglang quant registry 不兼容)。

更详细的 sglang 0.5.10 量化方案清单见 `../INT8_strategy.md` §5`。

## F8. qlean ignore 列表 vs sglang fused layers 不兼容 — **0% acc 根因**

**症状**: 用纯 INT8 模型 (`Qwen3.5-2B-INT8/`)、`--quantization w8a8_int8` 跑 50 题
**全部 garbage output**(0/50 解析成功)。

**根因**(2026-08-15 发现的最终根因):

| 层 | qlean 想保留 BF16 的原因 |
|---|---|
| `self_attn.v_proj` | AMQ Fig 12 V 层最敏感 |
| `linear_attn.in_proj_qkv` | q_k + q_v 重操作符 |
| `linear_attn.in_proj_z` | q_j (→ z) 重操作符 |
| `linear_attn.conv1d` | 3D 张量无法量化 |
| `mlp.gate_proj` | 部分实现需精度 |

但 sglang 的 `qwen3_5.py` **硬合并**这些 shard:
```python
"qkv_proj":   ["q_proj", "k_proj", "v_proj"],   # self-attn
"gate_up_proj": ["gate_proj", "up_proj"],       # MLP
"in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],   # linear-attn
```

qlean 配方要 `v_proj` BF16、`q_proj/k_proj` INT8 → **矛盾** → 整 fused layer 无法
决定。sglang 推断走"q/k INT8 + v 强制 INT8"路径(把 v_proj 从 BF16 重新量化),丢失
v_proj 的精度保护,加上可能的 dtype 不匹配处理 bug → 整个模型精度塌方到 0%。

**三条出路**:
| 方案 | 精度 | INT8 收益 | 实施成本 |
|---|---|---|---|
| A. 全量化(含 v_proj/in_proj_qkvz 等) | -2 ~ -5 pp | 最大化 | qlean recipe 删 v_proj ignore,**已采纳(→ S2-aggressive)** |
| B. fused layer 全 BF16 | ±0 pp | 最小化 | config.json 加 ignore |
| C. sglang 支持 per-shard fused quantization | ±0 pp | 最大化 | 改 sglang 源码 |

**采纳方案 A**,接受精度损失换可用加速。完整 5 条 / 6 条路径决策树见
`../INT8_strategy.md` §10。
