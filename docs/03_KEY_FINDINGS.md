# 03. 关键发现汇总

## F1. SGLANG_SAIL_FLA_CUDA=1 是最大加速点

**问题**：Qwen3.5-2B 是 hybrid linear_attention (GDN) + full_attention。
- fa3 不支持 GDN，linear 层默认走 triton community kernel
- 官方文档说 `SGLANG_SAIL_FLA_CUDA` 默认 True，但代码里实际 `EnvBool(False)`

**解决**：在 `evaluation_wrapper.py._SGLangServer._launch()` 强制设 `env["SGLANG_SAIL_FLA_CUDA"] = "1"`

**效果**：
- prefill TTFT: 340ms → 64ms（-80%）
- accuracy 不变（0.85）
- throughput 不变（280 tok/s）

**位置**：`/usr/local/lib/python3.12/site-packages/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
读源码时发现 dispatcher 根据 env 选：
- True → `fla.fused_sigmoid_gating_delta_rule_forward_k_last_packed`（PPU 专用 CUDA）
- False → Triton community kernel

## F2. --disable-radix-cache 隐含 --disable_overlap_schedule=False

**问题**：mmbench 样本唯一，radix cache 收益接近零。

**意外联动**：sglang 在 `disable_radix_cache=True` 时把 `disable_overlap_schedule` 也自动设为 False，prefill/decode 重叠调度打开。

**效果**：
- throughput: 280 → 369 tok/s（+32%）
- TTFT 轻微 +5ms（可忽略）

**结论**：在样本唯一、prompt 模板略相似的评测场景下，关 radix 几乎总是好的。

## F3. --page-size 64 是 KV cache 粒度微调

**问题**：KV cache 按 1-token 粒度管理（默认），metadata 开销大。

**解决**：用 `--page-size 64`（按 64-token 粒度管理），参考 397B 手册 §2.2.4。

**效果**：
- TTFT: 69 → 64.7ms（-6%）
- throughput 持平

**结论**：单样本 ~1500 token 场景下，page=64 是更优选择。

## F4. GEMV 是瓶颈（73% 时间）

**算子 profile（asys 60s, G 组）**：

| OP Type | Total % | Total ms | Avg ns | Instances |
|---|---|---|---|---|
| **GEMM** | **73.4%** | 5181.9 | 19,414 | 266,915 |
| Native | 21.3% | 1501.2 | 2,795 | 537,066 |
| FlashAttention | 4.9% | 342.9 | 12,338 | 27,792 |
| Triton | 0.4% | 31.4 | 2,254 | 13,926 |

**Top 3 kernel**（占 GEMM 中的 72.7%）：
- `void gemvt_op<cutlass::bfloat16_t, ..., 32, 4, ...>` 23,452ns avg, 168k calls (56.1%)
- `void gemvt_op<cutlass::bfloat16_t, ..., 64, 2, ...>` 14,782ns avg, 55k calls (11.6%)
- `void gemvt_op<cutlass::bfloat16_t, ..., 128, 1, ...>`  8,439ns avg, 41k calls (5.0%)

→ 全部是 bf16 cutlass GEMV（线性层 decode 阶段），cutlass 自动选不同 tile。

## F5. G 组 (bf16) 不使用 DeepGEMM

**结论**：sglang-fla 的 bf16 路径走 `UnquantizedLinearMethod.apply()` → `F.linear()` → cutlass bf16 GEMV。

**DeepGEMM 触发条件**（必须同时满足）：
1. 模型用 INT8 量化（`--quantization w8a8_int8`）
2. PPU 上 (`_is_ppu`)
3. Linear 没有 bias
4. `SGLANG_SAIL_DEEPGEMM_DENSE=True`（默认 True，但只在量化路径生效）

**位置**：`/usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/w8a8_int8.py`:
```python
if envs.SGLANG_SAIL_DEEPGEMM_DENSE.get() and _is_ppu and bias is None:
    # 走 DeepGEMM
```

`/usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/unquant.py`:
```python
def apply(self, layer, x, bias=None):
    ...
    return F.linear(x, layer.weight, bias)  # bf16 默认路径
```

## F6. INT8 W8A8 量化被否决

**问题**：理论上 W8A8 应该让 GEMV 内存带宽需求减半 → 2× 加速。

**实测**：
- accuracy: 0.85 (bf16) → 0.75 (20 样本 INT8) → 0.66 (50 样本 INT8)
- **-19pp**，远超 2% 阈值
- 模型输出乱码，多数 4-13 tokens 就停，重复问题文本

**根因**：
- 量化方案太激进（per-channel INT8 + lm_head 也量化）
- 不是 Triton backend 问题（bf16+Triton 实测 0.80 accuracy）

**对比 INT8 vs bf16 kernel profile**：
- INT8 kernel 名字从 `gemvt_op<cutlass::bfloat16_t>` → `deep_gemm::GemmKernel<MmaMultistage>`
- DeepGEMM 路径被确认命中
- 但仍有 2.7% cutlass bf16 GEMV 残留（lm_head + GDN.b/a 未量化）

## F7. EAGLE/NEXTN 投机解码在 hybrid GDN 模型上崩溃

**崩溃点**：`fused_sigmoid_gating_recurrent.py:339` → `retrieve_parent_token.int()` → `AttributeError: 'NoneType' object has no attribute 'int'`

**根因**：
- EAGLE/NEXTN verify 阶段需要 `retrieve_parent_token` 做 parent tracking
- Qwen3.5 hybrid GDN 层是 recurrent 状态机，没有显式 parent token
- sglang 0.5.10+ppu2.1.1 的 `gdn_backend.target_verify()` 没为 hybrid 模型正确填 parent token

**bug 与具体参数无关**：试过 `--speculative-num-steps 2/3`、`--speculative-eagle-topk 1`、`--speculative-num-draft-tokens 3/4`，全部崩。

**FLA=off 绕过**：让核走 Triton 路径，`HAS_EAGLE_TREE_CUSTOM_ATTN_MASK = retrieve_parent_token is not None` 让核自己分 None 走无 tree-mask 分支 → 能跑但 TTFT +35%, throughput -25%（投机收益 < 多花的 draft 推理时间）。

**结论**：投机解码对 2B 模型在短回答场景（mmbench 单选 256 token）下性价比为负。

## F8. CUDA Graph 抓取是隐形启动成本

- triton attention：CUDA Graph 抓取 49.74 s（36 个 bs）
- fa3 attention：CUDA Graph 抓取 3.97 s（同样 36 个 bs，快 12×）

→ 即使没有 inference 请求，启动一次 sglang server 也至少 10-60s 等待 CUDA Graph 抓取。

## F9. PPU profiler 是 asys（不是 nsys/nvprof）

```bash
# 抓 GPU kernel trace
asys attach -t hggc -s process-tree -o <file> --duration <s> <pid>

# 导出报告
asys stats -r ppu_op_type_sum -f csv -o . <file>.asysrep
asys stats -r ppu_op_sum -f csv -o . <file>.asysrep
asys stats -r ppu_op_kernel_breakdown -f csv -o . <file>.asysrep
asys stats -r hggc_kern_exec_trace -f csv -o . <file>.asysrep  # 巨大但有 grid
```

限制：
- Ftrace 不可用（CPU 调度 trace 关）
- Perf 不可用（CPU backtrace 关）
- 只能用 process-tree 模式
- `--callstack-group-config` 仅 device_memory_usage 支持
- `--python-backtrace hggc` + `--hggcbacktrace hggc` 可拿到 Python frame（但只对 memory trace 关联）

## F10. GEMV → 模型组件可通过 grid 拆分定位

**方法**：`asys stats -r hggc_kern_exec_trace` 导出每个 kernel launch 的 GridXYZ，按 grid 分组即可对应 Linear 层形状：

| Grid | Weight 形状 | 模型组件 |
|---|---|---|
| 512×1×1 | (2048, 512) | self_attn.k_proj / v_proj（6 层 full attn ×2） |
| 2048×1×1 | (2048, 2048) | self_attn.q_proj + o_proj + GDN.out_proj（24 层 ×~3） |
| 3072×1×1 | (2048, 3072) | GDN.in_proj 拆 4 块 或 MLP gate/up 拆 2 块 |
| 62080×1×1 | (2048, 62080) | lm_head 拆 4 块（vocab=248320/4） |
| 1280×1×1 | (?, 1280) | 未知（可能是 vision encoder） |

→ **量化优先级**：
1. self_attn.q_proj + o_proj + GDN.out_proj (2048×2048) — 24 层都有，最高 ROI
2. self_attn.k_proj + v_proj (2048×512) — 6 层 full attn
3. GDN.in_proj (2048→~8376) — 18 层 linear attn，in_proj 输出含 b/a (16 维) 不易量化
4. lm_head (2048→248320) — 每 token 一次但 vocab 太大

## F11. 规则红线（不可碰）

| 红线 | 我们的应对 |
|---|---|
| 换权重 / 蒸馏 | 用官方 `./Qwen3.5-2B`，INT8 量化包由主办方提供（`./quant/Qwen3.5-2B-W8A8`） |
| 查表返回答案 | 所有答案由模型生成，不维护 hash 表 |
| 硬编码公开集答案 | 不维护任何 sample_id → answer 映射 |
| 联网 | 不调用外部 API / 搜索 |
| 伪造 metrics | `token_count` 用 `usage.completion_tokens`，TTFT 用 stream 首个 chunk 时间 |
| 拆两套策略 | 用同一 wrapper，同一 prompt 模板，仅 sglang 配置不同 |
| 持久化缓存 | 不写任何持久化 cache 文件 |
| 长 timeout | 不传 `--watchdog-timeout` / `--dist-timeout`，用默认 |