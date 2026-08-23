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

## F9. HGGC mma 路径 for INT8 GEMV — **架构决策(放弃,不是失败)**

> 实验目录:`/mnt/workspace/ISA/src/kernels/gemv_int8_hggc.cu`(已弃)
> 调研笔记:`/mnt/workspace/ISA/notes/HGGC_PPU001_NOTES.md` 12 章

**背景**: 看过 HGGC 编程指南 245 页,INT8 mma `m16n16k16` / `m8n32k16` / `m32n8k16`
看起来是优化 INT8 GEMV 的标准武器;用 SAIL `awmma::fragment` 直接走张量核,理论上
比 SIMT 快 ~8×。

**尝试**: 写 `gemv_int8_hggc.cu` 用 `awmma::fragment<matrix_a, 16, 16, 16, int8_t, row_major>`
+ `awmma::fragment<matrix_b, 16, 16, 16, int8_t, col_major>` + `awmma::mma_sync(...)`。
模板化 K=2048 → 128 个 mma step。

**两次崩溃**:
1. **编译期**: `awmma` 头文件 + `<torch/extension.h>` + nvcc 13 + g++ 13 cmath
   constexpr 冲突(`isgreaterequal` / `isinf` / `isnan` 都被 HGGC 重定义为 `__device__ constexpr`),
   触发 "subexpression not valid in a constant expression" 20+ 错误。
2. **运行期**: 即便把 HGGC 头文件从 torch include path 隔离出来,`awmma::fragment`
   构造时若拿到的 row_major 数据 layout 不严格 256-bit 对齐就 segfault。

**关键决策(2026-08-23 决定放弃 mma 路线)**:

| 证据 | 含义 |
|---|---|
| deep_gemm 自己不用 mma 跑 m=1 GEMV(`gemvt.cuh` 是 SIMT + 4×int4 向量化 + cp.async 多 stage) | 业界经验:m=1 GEMV 走 mma 浪费 15/16 计算(M 维 16 但只用 1) |
| Phase 3.0 naive(SIMD-only)microbench 26.78μs 已经和 deep_gemm 31.73μs 持平 | mma 不是 GEMV 瓶颈 |
| microbench 显示 HBM 有效带宽只到 peak 4.5% | 瓶颈是 HBM latency 和 launch overhead,不是算力 |

**结论**: 跳过 HGGC mma 路径,直接走 **HBM-direct SIMT**(Phase 3.2 v10)。
v10 = 6.7 μs vs deep_gemm 30.3 μs = 4.5×,**比预期的 mma 加速还多**(因为不用
mma 后 layout 更简单,lane 间 coalesce 完美)。

**教训(写进 AGENTS.md)**: PPU kernel 任务"用 HGGC 不用 NVIDIA"指的是**PTX 指令层**(
用 `ppu.cp.async.*` 不用 `cp.async.*`,用 `awmma` 不用 `wmma`),**不是"必须用 mma"**。
m=1 GEMV 是反 mma 的任务,要按 deep_gemm 自己的做法走 SIMT。

## F10. v10 dispatcher layout bug — **dispatcher 第一版的 reverse-transpose**

**症状**: 启用 v10 fast path 后模型输出 garbage (e.g. `"Fated/Uglanyation/MaxFulFeyndropPolly"`),
TPS 411.6 → 353.0 (-14%),acc 0.73 → 0.70。

**根因**: sglang `apply()` 调用 `gemv_int8_i8i8bf16(x_q, x_scale, layer.weight.t(), ...)`。
`layer.weight` 存为 `(K, N)` row-major(在 `process_weights_after_loading` 里
`.t()` 转置过)。所以 `layer.weight.t()` 给我们 shape `(N, K)` 但 strides `(1, K)` —
**column-major 的 (N, K) view**。

v10 第一个版本的 dispatcher 写了:
```python
def _isa_v10_get_w_nk(W_kn):
    return W_kn.t().contiguous()  # ← BUG: 实际 B 已经是 (N, K) shape 了!
```
`.t().contiguous()` 又做了一次 reverse,把 `(N, K)` view 变回 `(K, N)` row-major。
kernel 拿到错的 shape,`TORCH_CHECK(W.size(1) == K, ...)` 抛 "K mismatch" →
exception caught → 静默 fallback 到 deep_gemm → deep_gemm 又因为 B 非 contiguous
assert 失败 → 输出 buffer 残留上次 decode 的值 → 模型 garbage。

**修复**:
```python
def _isa_v10_get_w_nk(B):
    cached = B.contiguous()  # ← 正确: 直接 contiguous,不要 .t()
    _ISA_V10_W_CACHE[B.data_ptr()] = cached
    return cached
```

**端到端验证**:
| 配置 | TPS | acc |
|---|---|---|
| baseline | 411.6 | 0.73 |
| v10 bug | 353.0 | 0.70 |
| **v10 fixed** | **421.1** | **0.73** |

**教训**: kernel .cu / wrapper .cpp 验证 correctness 不够,dispatcher 的 layout
转换是另一层容易出错的地方。集成时必须有 end-to-end 测试,不能只 unit test。

完整 dispatch + cache + CUDA-graph 安全性分析见 `ISA/docs/05_INTEGRATION.md`。

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
