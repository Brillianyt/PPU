# 02. 探索过程（按时间顺序）

## Phase 0：环境探测

```
- ls /mnt/workspace/dndx_participant-v1.2
- cat README.md                  → 了解竞赛规则 + 接口契约
- cat requirements.txt
- ls ./Qwen3.5-2B               → 4.19 GB safetensors
- cat Qwen3.5-2B/config.json     → Qwen3_5ForConditionalGeneration
- python3 -c "import sglang; print(sglang.__version__)" → 0.5.10+v0.3.0.ppu2.1.1
- which asys                     → /usr/local/PPU_SDK/asight/bin/asys (v2.1.1)
- nvidia-smi                     → 注：PPU 不是 NVIDIA GPU，nvidia-smi 不工作；用 PPU 自带工具
```

**关键发现**：
1. transformers 4.57.1 不支持 `model_type=qwen3_5`，baseline 路径走不通
2. sglang 里有 `sglang/srt/configs/qwen3_5.py` 但依赖更新的 transformers
3. 评测必须通过 sglang 后端

## Phase 1：sglang baseline（triton attention）

```bash
python3 -u -m sglang.launch_server \
  --model-path ./Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend triton --enable-multimodal \
  --dtype bfloat16 --mem-fraction-static 0.85
```

启动观察：
- 加载权重 1.04s（4.19 GB）
- Mamba cache：1.24 GB conv + 35.26 GB ssm
- KV cache：40.58 GB（`#tokens=3,545,083`，bf16）
- CUDA Graph 抓取：36 个 bs，**49.74 s**（很慢）
- 总启动 ~60 s

20 样本结果（`result_dev_en_sglang_20.json`）：
- accuracy: 0.80（16/20）
- avg_ttft_ms: 419.0（前 5 个 ~50ms 命中 radix prefix cache，之后 ~340ms 平台）
- avg_throughput_tokens_per_sec: 284.5
- 1 个 3074ms 离群点（qid=344，可能首次触发某个 bs 的 CUDA Graph 重捕）

TTFT 分布：min 42.5 / p25 96.8 / p50 346.0 / p75 354.9 / p95 3073.9 / max 3073.9

## Phase 2：fa3 attention

```bash
# 把 --attention-backend triton 改成 fa3
```

启动观察：
- `Linear attention kernel backend: decode=triton, prefill=triton`（fa3 不支持 GDN，linear 层仍走 triton）
- CUDA Graph 抓取 **3.97s**（vs triton 49.74s，启动快 12 倍）
- 启动日志明确显示 GDN 部分仍 triton

结果（`result_dev_en_sglang_fa3_20.json`）：
- accuracy: **0.85**（+5pp）
- avg_ttft_ms: 313.2（-25%）
- throughput: 279.4（基本不变）
- TTFT max: 762ms（vs 3074ms）

→ fa3 把 full attention 路径加速，但 GDN 仍走 triton → TTFT 仍有 ~330ms 平台

## Phase 3：SGLANG_SAIL_FLA_CUDA=1（重大突破）

动机：SGLang-for-SAIL 文档说该开关默认 True，但 `sglang/srt/environ.py` 里实际默认 `EnvBool(False)`。

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server ...  # 其余同 Phase 2
```

启动日志关键行：
```
[2026-08-14 16:56:29] USE PPU SAIL CUDA FLA kernel: fused_sigmoid_gating_delta_rule_forward_k_last_packed
```

Phase 2 没设这个 env var 时 GDN dispatcher 仍标 `TritonGDNKernel`。

结果（`result_dev_en_sglang_fla1_20.json`）：
- accuracy: 0.85（持平）
- avg_ttft_ms: **64.3 ms**（-80% vs Phase 2）
- throughput: 280.1（持平）
- TTFT 平台消失，曲线非常紧凑（max 79ms）

→ GDN 路径从 triton 切到 PPU 专用 CUDA kernel，单 sample prefill 直接打到亚 100ms

## Phase 4：--disable-radix-cache

动机：mmbench 样本唯一、prompt 模板略相似，radix cache 收益接近零，副作用是 lookup 开销 + 改变调度行为。

```bash
# 在 Phase 3 基础上加 --disable-radix-cache
```

联动：sglang 在 `disable_radix_cache=True` 时把 `disable_overlap_schedule` 自动设为 `False`，prefill/decode 重叠调度打开 → decode 不再等 prefill 完全结束。

结果（`result_dev_en_sglang_fla1_noradix_20.json`）：
- accuracy: 0.85
- avg_ttft_ms: 69.0（轻微 +5ms 因 radix lookup 消失 + overlap schedule 调度粒度变粗）
- throughput: **369.4 tok/s**（+32%）
- TTFT max: 80ms（仍紧凑）

## Phase 5：--page-size 64（最终配置 G 组）

动机：397B 手册 §2.2.4 PD 分离 D 节点用 `--page-size 64`（vs sglang 默认 1），KV cache 按 64-token 粒度管理减少碎片。

```bash
# 在 Phase 4 基础上加 --page-size 64
```

结果（`result_dev_en_sglang_page64_20.json`）：
- accuracy: 0.85
- avg_ttft_ms: **64.7 ms**（-6.2% vs Phase 4，KV metadata 减少）
- throughput: 369.0 tok/s（基本不变）

→ 这就是最终推荐配置

## Phase 6：投机解码尝试（全部失败）

### EAGLE
```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -m sglang.launch_server ... \
  --speculative-algorithm EAGLE --speculative-num-steps 2 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 3
```
启动 12 秒后崩：
```
File "sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py", line 339
    retrieve_parent_token.int(),
AttributeError: 'NoneType' object has no attribute 'int'
```

### EAGLE + MTP draft
试 F2 组显式 `--speculative-draft-model-path ./Qwen3.5-2B`（Qwen3.5-2B 带 MTP 权重：`text_config.mtp_num_hidden_layers=1`），同样崩。

### NEXTN（用户要求参数）
```bash
--speculative-algorithm NEXTN --speculative-num-steps 3 \
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```
同样崩。

### NEXTN + FLA=off（绕过 bug）
关掉 SGLANG_SAIL_FLA_CUDA，走 Triton 路径，`HAS_EAGLE_TREE_CUSTOM_ATTN_MASK = retrieve_parent_token is not None` 让核自己分 None 走无 tree-mask 分支 → 能跑。

结果（`result_dev_en_nextn_flaoff_20.json`）：
- accuracy: 0.85（持平）
- avg_ttft_ms: 87.3（+35%）
- throughput: 275.5 tok/s（-25%）

→ 投机解码对 2B 模型在短回答场景（mmbench 单选 256 token）下性价比为负，关闭。

根因（见 EXPERIMENT.md §15）：
- EAGLE/NEXTN verify 阶段需要 `retrieve_parent_token` 做 parent tracking
- Qwen3.5 hybrid GDN 层是 recurrent 状态机，没有显式 parent token
- sglang 的 `gdn_backend.target_verify()` 没为 hybrid 模型正确填 parent token
- 这是 sglang 0.5.10+ppu2.1.1 的 bug，与 num_steps/topk/draft_tokens 取值无关

## Phase 7：INT8 W8A8 量化（失败）

```bash
python3 -u -m sglang.launch_server \
  --model-path ./quant/Qwen3.5-2B-W8A8 \
  --host 127.0.0.1 --port 9001 \
  --trust-remote-code --tp-size 1 \
  --attention-backend triton \
  --enable-multimodal \
  --quantization w8a8_int8 \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --page-size 64 \
  --disable-radix-cache
```

启动：
- DeepGEMM warmup ~10s（16384 tokens 扫一遍）
- CUDA Graph 抓取 12.36s（36 个 bs）

结果（`result_dev_en_int8_triton_20.json` & `result_dev_en_int8_triton_50.json`）：
- accuracy: 0.75 (20 样本), 0.66 (50 样本) ← **-19pp**
- 远低于 2% 阈值，不可接受

通过对比 bf16 + Triton 单独实验确认：accuracy 下降是量化本身的问题，不是 Triton backend。

模型实际行为：输出是乱码，多数 4-13 tokens 就停，重复问题文本或乱码。

## Phase 8：DALI 图像预处理（否决）

动机：用户提出"TTFT 总共 63ms，CPU 处理个 10ms，但 GPU 不一样"。

实际探测：
- sglang 用的是 HF `Qwen2VLImageProcessorFast`（torch-based）
- 在 GPU 上跑，0.72ms/张（实测）
- vs prefill 64ms 中占比 ~1%
- 切换到 DALI 需要重写 sglang processor 路径，ROI 太低

→ 否决。

## Phase 9：Sliding window attention（否决）

动机：用户问 sliding window 能否继续加速。

实际探测：
- Qwen3.5 是 hybrid **linear_attention** + full_attention，不是 sliding window
- linear_attention（GDN）已经是 recurrent 计算，没有 KV cache 线性增长问题
- sliding window 是 full attention 的子集，对 linear_attention 不适用
- 否决。

## Phase 10：长 timeout（否决）

动机：曾经想过加 `--watchdog-timeout 3600 --dist-timeout 3600` 防推理 stall。

实际：用户明确"不允许长 timeout"（违反公平性原则，会被主办方判作弊）。

→ 移除长 timeout 参数，wrapper 默认不传任何 timeout 标志。

## Phase 11：算子 profile（关键洞察）

工具：asys v2.1.1

```bash
# 60s GPU kernel trace
asys attach -t hggc -s process-tree -o profile_results/g_group_60s.asysrep --duration 60 <scheduler_pid>

# 导出
asys stats -r ppu_op_type_sum -f csv -o . profile_results/g_group_60s.asysrep
asys stats -r ppu_op_sum -f csv -o . profile_results/g_group_60s.asysrep
asys stats -r ppu_op_kernel_breakdown -f csv -o . profile_results/g_group_60s.asysrep
asys stats -r hggc_kern_exec_trace -f csv -o . profile_results/g_group_60s.asysrep  # 巨大
```

G 组 profile 关键数据（60s, 10 mmbench 请求）：

| OP Type | Total % | Total ms | Avg ns | 次数 |
|---|---|---|---|---|
| Summary | 100.0 | 7057.4 | 8,345 | 845,699 |
| GEMM | 73.4 | 5181.9 | 19,414 | 266,915 |
| Native | 21.3 | 1501.2 | 2,795 | 537,066 |
| FlashAttention | 4.9 | 342.9 | 12,338 | 27,792 |
| Triton | 0.4 | 31.4 | 2,254 | 13,926 |

Top 3 kernel（占 73.4% GEMM 中的 72.7%）：
- 56.1% 168k calls `void gemvt_op<cutlass::bfloat16_t, ..., 32, 4, ...>` 23,452ns avg
- 11.6% 55k calls `void gemvt_op<cutlass::bfloat16_t, ..., 64, 2, ...>` 14,782ns avg
-  5.0% 41k calls `void gemvt_op<cutlass::bfloat16_t, ..., 128, 1, ...>`  8,439ns avg

→ **GEMV（线性层 decode 阶段）是决定性瓶颈**。

## Phase 12：INT8 profile 对比

同样方法 profile INT8 服务器（15MB raw trace）：

| Metric | bf16 G 组 | INT8+DeepGEMM | Δ |
|---|---|---|---|
| 总 kernel 时间 | 7057.4ms | 1004.4ms | -85.8% |
| Kernel launches | 845,699 | 97,771 | -88.4% |
| GEMM  share % | 73.4 | 61.4 | -12 pp |
| GEMM  total | 5181.9ms | 616.7ms | -88.1% |
| GEMM  launches | 266,915 | 21,677 | -91.9% |
| GEMM  **avg/call** | 19,414ns | 28,450ns | +46.5% ⚠️ |

⚠️ INT8 GEMM avg 看起来更慢是因为：
- bf16: 几乎全 decode GEMV (1 token)
- INT8: 主要 prefill GEMM (batch)，因为模型乱码输出短，请求提前结束
- 总算力压力只有 bf16 的 14%

但更重要的是：
- INT8 服务器 kernel 名字从 `gemvt_op<cutlass::bfloat16_t>` 变成 `deep_gemm::GemmKernel<MmaMultistage>`
- 确认 DeepGEMM 路径被命中
- 但仍有 2.7% `gemvt_op cutlass bf16` 残留（lm_head + GDN.b/a 未量化）

## Phase 13：GEMV → 模型组件定位

通过 `hggc_kern_exec_trace` 导出每个 kernel launch 的 grid/block：

Top GEMV (tile 32×4) 共 168,713 launches，按 grid 拆分：

| Grid | Shape (推测) | 模型组件 | Launches |
|---|---|---|---|
| 512×1×1 | (2048, 512) | self_attn.k_proj / v_proj（6 层 ×2） | 55,464 |
| 2048×1×1 | (2048, 2048) | self_attn.q_proj + o_proj + GDN.out_proj | 41,598 |
| 3072×1×1 | (2048, 3072) | GDN.in_proj 拆 4 块 OR MLP gate/up 拆 2 块 | 55,464 |
| 62080×1×1 | (2048, 62080) | lm_head 拆 4 块（vocab=248320/4） | 2,321 |
| 1280×1×1 | (?, 1280) | 未知（可能是 vision encoder） | 13,866 |

→ 量化优先级：q_proj/o_proj/out_proj (24 层都有) + k/v_proj (6 层) + lm_head

## Phase 14：DEEPGEMM 状态

读源码确认：

```python
# /usr/local/lib/python3.12/site-packages/sglang/srt/environ.py
SGLANG_SAIL_DEEPGEMM_DENSE = EnvBool(True)   # 默认 True
SGLANG_SAIL_DEEPGEMM_MOE = EnvBool(True)     # 默认 True
```

```python
# /usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/w8a8_int8.py
if envs.SGLANG_SAIL_DEEPGEMM_DENSE.get() and _is_ppu and bias is None:
    # 走 DeepGEMM 路径
```

```python
# /usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/unquant.py
def apply(self, layer, x, bias=None):
    if use_intel_amx_backend(layer): ...
    elif _use_aiter: ...
    return F.linear(x, layer.weight, bias)  # bf16 默认走 PyTorch F.linear → cutlass bf16 GEMV
```

→ G 组（bf16 + UnquantizedLinearMethod）**不使用 DEEPGEMM**。DeepGEMM 仅在 INT8 量化路径生效。

即使切到 INT8，DeepGEMM 也只覆盖 73% GEMV 里的部分（lm_head/GDN.in_proj.b+a 仍走 cutlass bf16）。