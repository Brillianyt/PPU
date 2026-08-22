# 04. 5 个核心优化详解

> 本文件只讲**做了什么 + 为什么 + 怎么布署**。数字 / 启用收益见
> [`02_RESULTS.md`](02_RESULTS.md)。历史诊断细节见
> [`../INT8_strategy.md`](../INT8_strategy.md) 和
> [`../vit_graph_optimization.md`](../vit_graph_optimization.md)。

## 4.1 S2-aggressive INT8 量化 (W8A8)

### 思路

对 175 个 Linear 层做 INT8 **逐层重建误差**敏感性分析 → 量化**误差最低**的层,
保留**误差最高**的层为 BF16,得到 Pareto 最优精度/吞吐折中。

### 量化目标

```yaml
# qwen3.5-2b_recipe.yaml (核心片段)
quant_stage:
  quant_modifiers:
    generalDay0Modifier:
      scheme: W8A8
      weights:        # per-channel int8, static, observer=memoryless_minmax
        num_bits: 8; type: int; symmetric: true; strategy: channel; observer: memoryless_minmax
      input_activations:  # per-token int8, dynamic
        num_bits: 8; type: int; symmetric: true; strategy: token; dynamic: true
      # 不量化的层 (17 条 regex)
      ignore:
        - "re:.*lm_head"
        - "re:.*embed_tokens"
        - "re:.*norm\\.weight$"
        - "re:.*norm1\\.weight$"
        - "re:.*norm2\\.weight$"
        - "re:.*q_norm\\.weight$"
        - "re:.*k_norm\\.weight$"
        - "re:.*mlp\\.gate$"
        - "re:.*linear_attn\\.A_log$"
        - "re:.*linear_attn\\.dt_bias$"
        - "re:.*linear_attn\\.conv1d.*"
        - "re:.*mtp\\..*"
        - "re:.*visual\\..*"
        # AMQ 高误差层保留 BF16:
        - "re:.*linear_attn\\.in_proj_a$"   # recurrent state 投影
        - "re:.*linear_attn\\.in_proj_b$"
        - "re:.*linear_attn\\.out_proj$"   # GDN 环路输出
        - "re:.*layers\\.19\\.self_attn\\.(q_proj|k_proj|v_proj)$"
        - "re:.*layers\\.23\\.self_attn\\.(q_proj|k_proj|v_proj)$"
        # S2-aggressive: gate/up + in_proj_qkv/z 放回 INT8 (误差最低)
```

### 策略选择

| 策略 | 量化层数 | acc (en 100, S2 vs BF16 对照) | 备注 |
|---|---|---|---|
| v3 (低覆盖) | 42 | 持平 | 端到端无收益 |
| **S2-aggressive (推荐)** | **126** | **0.74 vs 0.78 = −4pp 不显著** | R6 / R8 用的就是这个 |
| S3-full (激进) | 186 | 0.65 | 否决,破坏 recurrent 状态 |

### 关键 pitfall (为什么 INT8 在 sglang 0.5.10 会 0% acc)

见 [`03_FAILED.md`](03_FAILED.md) F8 — qlean 的 v_proj BF16 保护与 sglang fused
layer 矛盾的根因分析。S2-aggressive 是绕开这条矛盾的"接受量化精度损失换可用加速"方案。

## 4.2 GEMV decode 分派 (M-aware kernel + prefill BF16 反量化)

### 思路

Qwen3.5-2B decode 阶段 M = 1..8(token 数),prefill 阶段 M = 100..1024(图片 token 数)
是极端不同的工作负载。两条路径用不同算子:

| 阶段 | M | 路径 | 算子 |
|---|---|---|---|
| decode (single-sample) | M ≤ 8 | **INT8 GEMV** | DeepGEMM `Gemvt<int8_t>`(复用 `gemvt.cuh`, PPU AIU 异步搬运 + int4 向量化 + multistage) |
| prefill (vision encoder + multi-token) | M > 8 | **BF16 GEMM** | `F.linear` 临时反量化权重 (懒缓存),避开 INT8 GEMM 在大 M 下的劣势 |

### 入口

`deep_gemm.gemv_int8_int8_bf16_nt` 在 PPU 上**比** `gemm_int8_int8_bf16_nt` 更快
(M = 1 microbench: **13.0 μs vs 27.9 μs, 2.15×**),且**消除** Python dispatch 开销。
**数值与 GEMM bit 级一致** (max_diff = 0)。

### 分派代码

`sglang/srt/layers/quantization/w8a8_int8.py` (patch B):

```python
# w8a8_int8.py apply()
if SGLANG_SAIL_PREFILL_BF16 and _is_ppu and bias is None and x.shape[0] > M_MAX:
    # prefill: 反量化 + BF16 GEMM (快,避开 INT8 GEMM 在大 M 下的劣势)
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

### 部署位置

`PPU/gemv_optimization/sglang-for-sail/python/sglang/srt/`:
- `environ.py` — 注册 SGLANG_SAIL_USE_GEMV / GEMV_M_MAX / WARMUP_ON_LOAD 等 env
- `layers/quantization/w8a8_int8.py` — 应用层分派 + prefill BF16 反量化
- `layers/deep_gemm_wrapper/{compile_utils,entrypoint}.py` — 暴露 `gemv_nt_i8i8bf16`
- `../../../DeepGEMM-for-sail/deep_gemm/{__init__.py,jit_kernels/{__init__.py,gemv_int8.py}}` — GEMV JIT

理论背景 + 算法细节见 [`../GEMV调研/手写INT8-GEMV-嵌入sglang-sail指南.md`](../GEMV调研/手写INT8-GEMV-嵌入sglang-sail指南.md)。

## 4.3 FLA_CUDA — PPU GDN/Mamba 原生 kernel

### 思路

Qwen3.5-2B 含 18 层 GDN (linear attention / Mamba style),**不能**用标准 scaled
dot-product attention。PPU fork 自带 FLA_CUDA 加速 GDN 递归核。

### 实测收益

从 triton_attn 切到 fa3 + FLA:
- GDN 内核从 Triton baseline 改 PPU 原生
- **TTFT 从 313 ms 跌到 64 ms(−80%)** —— FLA 是本方案首要收益来源
- 不开 FLA = TTFT 多 3 ms,不开 = GPU 时间多 33%

### 强制开启

`SGLANG_SAIL_FLA_CUDA=1` 由 PPU fork 的 `_handle_ppu_backends:1368-1369`
自动设;wrapper 也会设。**不要关,关掉会让 TTFT +3 ms**。

## 4.4 ViT CUDA Graph 修复 (qwen3_vl + fa3 hook)

### 问题:开 ViT CG 后精度 −15pp

Qwen3.5-2B 复用 Qwen3-VL 的 ViT (Vision Transformer)。sglang PiCG capture 时遇
**两个**根因,叠加后 acc 从 0.78 掉到 0.60,cn 数据集直接崩到 0。

#### 根因 A: ViT attention 没走 PPU 原生 FA3

- PPU 报 sm80 → `_determine_attention_backend` 默认落 `triton_attn`(通用 Triton,
  非 PPU 优化)。
- 切 `fa3` 后,`vision.py` 里的 `flash_attn_varlen_func` 来自
  `sglang.jit_kernel.flash_attention`(community FA3 / sgl_kernel),**不是 PPU 原生**。
- PPU 原生 FA3 只通过 `ppu_fa3_hooks.py` 接到 decode 的 `flashattention_backend`,
  **没接到** ViT 的 `VisionFlash3Attention`。

#### 根因 B: Qwen3-VL 缺 `max_context_len`,rotary workspace 懒扩展破坏 graph(精度 −15pp 主因)

- `Qwen3VLMoeVisionModel` 没有 `max_context_len`(Qwen2.5-VL 有)
- `ViTCudaGraphRunner._ensure_sin_cos_ws` 在 `None` 时首次只按第一个图像 token 数
  分配 rotary sin/cos 共享 buffer
- 评测中图像尺寸不同 → 更大 S 触发懒扩展 → buffer 重分配(新地址)
- 已 capture graph 记录旧地址,replay 前 `copy_` 写新地址,但 graph 按地址读旧地址
  → rotary cos/sin 错乱 → ViT 特征错 → 模型输出退化成 `!!!!`

### 修复(两个 patch)

**P1: ViT attention → PPU 原生 `flash_attn_interface`**

在 `ppu_fa3_hooks.py` 追加 `@plugin_hook`:
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

配合启动参数 `--mm-attention-backend fa3` / `triton_attn`。

**P2: Qwen3-VL 补 `max_context_len`**

在 `models/qwen3_vl.py` 的 `Qwen3VLMoeVisionModel.__init__` 加一行:
```python
self.max_context_len = (
    vision_config.num_position_embeddings * self.spatial_merge_unit
)  # = 2304 * 4 = 9216
```

`9216` 覆盖 spatial_merge 前最大 patch token 数(96×96),`sin_cos_ws` 首次分配
足够大,**永不扩展**,graph 地址保持稳定。

### 修复效果

| 指标 | 修复前 P28 (ViT CG ON) | 修复后 V2 (ViT CG ON) | Δ |
|---|---|---|---|
| TTFT | 67.1 ms | 65.9 ms | −1.8% ✓ |
| TPS | 377.8 | 394.1 | +4.3% ✓ |
| acc (en) | 0.600 | 0.710 | +11pp ✓ |
| acc (cn) | 0.000 (崩溃) | 0.820 | +82pp ✓ |
| None parses | 23 | 7 | −16 ✓ |

**结论**: 修复前 ViT CG 是**纯负优化**(精度 −15pp),正解化后变为
**纯正优化**(TTFT −13%,TPS 持平,acc 仅 −4pp 统计不显著)。

完整根因 + 调试记录见 [`../vit_graph_optimization.md`](../vit_graph_optimization.md)。

## 4.5 Inductor RoPE hook (F 类 patch)

### 问题:启动 ~30s 后崩

跑 R6 / R8 路径 + ViT CG 时,**启动约 30 秒后**崩:
```
BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture
  unless the CPU tensor is pinned. Please use tensor.pin_memory() or allocate the
  tensor with pin_memory=True.
```
栈底在 `torch._inductor.fx_passes.fuse_attention._get_sfdp_patterns`。

### 根因

`sglang/srt/layers/rotary_embedding/utils.py` 把 `apply_rotary_pos_emb_native`
装饰成 `@torch.compile(dynamic=True, ...)`。ViT 的 `VisionAttention.forward` 调
用它 → 整个 ViT forward 走 torch.compile → inductor。CUDA Graph capture 时第一
次跑这个函数,inductor 触发 `_sfdp_init` → `_get_sfdp_patterns`,pattern init
里会试 `torch.empty(..., device='cuda')` + `torch.empty_strided(device='cpu',
pin_memory=True)`。

**PPU 是 CUDA 兼容层**,但 **CUDA Graph capture 的语义边界** 跟真 NVIDIA 卡不
完全一致 —— inductor 的 SDPA pattern init + autotune scratch 路径正好落在
PPU 不兼容的一段。

### 修复:RoPE 包装成 `torch.library.custom_op`

在 `inductor_hook_PPU/hook/ppu_rotary_hooks.py` 实现:

```python
@torch.library.custom_op("sglang::apply_rotary_pos_emb", mutates_args=(), device_types="cuda")
def apply_rotary_pos_emb_ppu(q, k, cos, sin, unsqueeze_dim=1):
    # 字节级复用 upstream native 的数学(去掉 @torch.compile)
    orig_q_dtype = q.dtype
    q_f = q.float(); k_f = k.float()
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    q_embed = (q_f * cos) + (_rotate_half(q_f) * sin)
    k_embed = (k_f * cos) + (_rotate_half(k_f) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)

@torch.library.register_fake("sglang::apply_rotary_pos_emb")
def apply_rotary_pos_emb_ppu_fake(q, k, cos, sin, unsqueeze_dim=1):
    # Shape inference only —— 返回 q, k 本身(与真实输出同 shape/dtype)
    return q, k

# sglang plugin hook 注册(仿 ppu_fa3_hooks.py 的写法)
@plugin_hook("sglang.srt.layers.rotary_embedding.utils.apply_rotary_pos_emb",
             type=HookType.REPLACE)
def _ppu_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    return apply_rotary_pos_emb_ppu(q, k, cos, sin, unsqueeze_dim)
```

**为什么有效**: `register_fake` 让 inductor 走 symbolic shape inference —— 不 trace
函数体,所以不触发 `_sfdp_init`,capture 安全。

完整设计与消融见 [`../inductor_hook_PPU/README.md`](../inductor_hook_PPU/README.md)。

## 4.6 ACEXT vs GEMV 决策

### ACEXT

- PPU 厂商针对 810E 深度优化的 INT8 GEMM/MOE 库(类似 cuBLAS)
- 来源是 PPU sgl_kernel 内置 `sgl_kernel.int8_gemm`,**无需 source patch**
- 比 GEMV **+2.4%** (423 vs 418 tok/s)
- 与 GEMV **互斥**:开 ACEXT 后 GEMV 不再触发

### GEMV

- 自研 kernel,基于 DeepGEMM `gemvt.cuh`
- 优势:可独立控制 + 支持 CUDA Graph capture
- 适合科研发布与可控迭代

### 实际跑的路径 ⚠️

注意 [`01_ENVIRONMENT.md`](01_ENVIRONMENT.md) §1.11:
**`SGLANG_SAIL_USE_ACEXT_CUDA=0` 在 wrapper 默认路径下无效** ——
PPU fork 的 `_handle_ppu_backends()` (server_args.py:1349-1355) 会强制设
`SGLANG_SAIL_USE_ACEXT_CUDA=True`。所以 wrapper 默认启起来跑的就是
**ACEXT 路径 + GEMV-WARMUP_ON_LOAD 预热 GEMV JIT 供 fallback**。

**这是为什么 R6 / R8 标称数字 实际是 ACEXT 配置**:
- R8 TPS 434 ≈ ACEXT (423 tok/s + ViT CG 贡献)
- R6 TPS 429 ≈ ACEXT (423 tok/s + triton_attn backend 略慢 1.2%)

**复测 takeaway**: 想看真 GEMV 性能,需显式走 `is_acext_disabled` 路径,本仓库
没 commit 这条(env var 被 PPU fork 覆盖)。

历史决策脉络见 `../conversations/KEY_DECISIONS.md`(决策 §38: ACEXT vs GEMV in CG)。

## 4.7 环境变量总表

### 比赛提交件 (R6 / R8) 必设

| 变量 | 推荐值 | 设置方式 | 作用 |
|---|---|---|---|
| `PPU_SDK` | `/usr/local/PPU_SDK` | wrapper 默认 | PPU SDK 路径,触发 platform init |
| `SGLANG_SAIL_USE_GEMV` | 1 | wrapper 默认 | decode 小 M 走自研 GEMV |
| `SGLANG_SAIL_GEMV_M_MAX` | 8 | wrapper 默认 | GEMV/GEMM 分派阈值 |
| `SGLANG_SAIL_PREFILL_BF16` | 1 | wrapper 默认 | prefill 反量化走 BF16 GEMM(关掉 TTFT +1-2 ms)|
| `SGLANG_SAIL_FLA_CUDA` | 1 | PPU fork 自动 + wrapper | PPU GDN CUDA kernel(关掉 TTFT +3 ms)|
| `SGLANG_SAIL_GEMV_WARMUP_ON_LOAD` | 1 | wrapper 默认 | 每层加载后自动 warmup |
| `SGLANG_VIT_ENABLE_CUDA_GRAPH` | 1 | wrapper 默认 | ViT CG capture(前提:patch E + F 已装)|
| `SGLANG_CUDA_GRAPH_MAX_BS` | 16 | wrapper 默认 | CUDA Graph max batch size |
| `SGLANG_PIECEWISE_CUDA_GRAPH_MAX_TOKENS` | 256 / 4 | wrapper / R8 文档 | piecewise CG token 上限(本仓库 VLM 路径下被 auto-disable)|
| `SGLANG_EXTRA_ARGS` | `--mm-attention-backend triton_attn` (R6)/`fa3` (R8) | 用户设 | mm attention backend 切换 |

### PPU fork 默认开 (无需用户设)

见 [`01_ENVIRONMENT.md`](01_ENVIRONMENT.md) §1.11 (3):`SGLANG_SAIL_FLA_CUDA`、`SGLANG_SAIL_DEEPGEMM_DENSE`、`SGLANG_SAIL_DEEPGEMM_MOE`、`SGLANG_DISABLE_CUSTOM_ALL_REDUCE`。

### 已记录无效的开关

| 变量 | 用户意图 | 实际效果 |
|---|---|---|
| `SGLANG_SAIL_USE_ACEXT_CUDA=0` | 走 GEMV | 仍走 ACEXT(PPU fork 覆盖)|
| `--enable-piecewise-cuda-graph` | PiCG 开 | VLM 工作负载下自动 disable |
| `--enforce-piecewise-cuda-graph` | 强开 PiCG | scheduler 死(capture stream assert)|

## 4.8 修改文件清单 (更新用)

R6 / R8 路径依赖下面这些文件改动(全部纯 Python 覆盖):

```
PPU/
├── gemv_optimization/
│   ├── DeepGEMM-for-sail/
│   │   └── deep_gemm/
│   │       ├── __init__.py
│   │       └── jit_kernels/
│   │           ├── __init__.py
│   │           └── gemv_int8.py
│   └── sglang-for-sail/
│       └── python/sglang/srt/
│           ├── environ.py
│           ├── hardware_backend/ppu/attention/
│           │   ├── ppu_fa3_hooks.py
│           │   └── (其他 PPU 自带)
│           ├── layers/
│           │   ├── quantization/
│           │   │   └── w8a8_int8.py
│           │   └── deep_gemm_wrapper/
│           │       ├── compile_utils.py
│           │       └── entrypoint.py
│           ├── models/
│           │   ├── qwen3_5.py
│           │   ├── qwen3_5_mtp.py
│           │   └── qwen3_vl.py
│           └── speculative/
│               └── eagle_worker_v2.py
└── inductor_hook_PPU/
    └── hook/
        └── ppu_rotary_hooks.py
```

一键部署命令见 [`01_ENVIRONMENT.md`](01_ENVIRONMENT.md) §1.10。
