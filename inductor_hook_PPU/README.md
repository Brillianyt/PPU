# PPU RoPE / inductor Hook — 设计与根因

> **范围**：本文档说明 **为什么需要这个 hook、它解决什么、怎么实现的**。
> 部署 / 验证 / 回滚步骤 → 见 `USAGE.md`。

---

## 1. 工作目的

让 **ViT CUDA Graph capture 在 PPU 上稳定通过**，从而恢复 sglang 端到端推理（fa3 + GEMV + PiCG + ViT CG 组合）。

之前 PiCG 模式下，benchmark 在 ViT CG capture 时崩溃，栈底在 inductor 的
`_get_sfdp_patterns` → `triton_heuristics.copy_args_to_cpu_if_needed`（pin_memory 路径）。
**真正要做的是给 RoPE 找一个 PPU 的 cuda-等价 hook**，让 inductor 的 SDPA 模式匹配不走到
PPU 不安全的那条路。

---

## 2. 根因

`sglang/srt/layers/rotary_embedding/utils.py:73` 把 `apply_rotary_pos_emb_native` 装饰成
`@torch.compile(dynamic=True, backend=get_compiler_backend())`。ViT 的
`VisionAttention.forward`（`vision.py:1260`）调用它，于是整个 ViT forward 走
torch.compile → inductor。CUDA Graph capture 时第一次跑这个函数，inductor 触发
`_sfdp_init` → `_get_sfdp_patterns`，pattern init 里会试
`torch.empty(..., device='cuda')` 和 `torch.empty_strided(device='cpu', pin_memory=True)`——
**CUDA Graph capture 上下文里这些路径在 PPU 后端不安全**，所以崩溃。

**为什么 PPU 不安全**：PPU 是 T-Head 真武的 CUDA-兼容卡——所有 `torch.cuda.*` API 都能调，
但 **CUDA Graph capture 的语义边界**跟真 NVIDIA 卡不完全一致。inductor 的 SDPA pattern init
+ autotune scratch 路径正好落在 PPU 不兼容的那一段。

---

## 3. 设计思路：参考 PPU 自带 flashinfer / FA3 的做法

本地两个 PPU fork 已经处理过同类问题——把 op 注册成 inductor 看不到的 opaque custom-op：

| 库 | 文件 | 关键模式 |
|---|---|---|
| **FA3** | `/usr/local/lib/python3.12/site-packages/flash_attn_interface.py:40,132` | `@torch.library.custom_op` + `@torch.library.register_fake` |
| **flashinfer** | `/usr/local/lib/python3.12/site-packages/flashinfer/rope.py:47,74,90,123,…` | 同上（PPU fork 把 `register_custom_op` 实现成空 stub） |

```python
@torch.library.custom_op("flash_attn_3_ops::_flash_attn_forward", ...)
def _flash_attn_forward(q, k, v, ...):
    ...real kernel call...

@torch.library.register_fake("flash_attn_3_ops::_flash_attn_forward")
def _flash_attn_forward_fake(q, k, v, ...):
    ...symbolic empty tensors, no real work...
```

**注册 fake op 让 inductor 走 symbolic shape inference**——不 trace 函数体，所以不触发
`_sfdp_init`，capture 安全。

### 这就是"cuda-等价"的正解

在真 NVIDIA 卡上，inductor 会编译 `@torch.compile` 装饰的函数；
在 PPU 上没法这样编译（上面那条路径不安全），但通过 custom-op slot 让 inductor 看不到
函数体，达成**内存等价**——同样不进 function body、同样不触碰 pin_memory 路径，
但 RoPE 实际算的数学结果 bit-identical。

---

## 4. 实现

`hook/ppu_rotary_hooks.py`：

```python
@torch.library.custom_op("sglang::apply_rotary_pos_emb", mutates_args=(), device_types="cuda")
def apply_rotary_pos_emb_ppu(q, k, cos, sin, unsqueeze_dim=1):
    # 字节级复用 upstream native 的数学（去掉 @torch.compile）
    orig_q_dtype = q.dtype
    q_f = q.float(); k_f = k.float()
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    q_embed = (q_f * cos) + (_rotate_half(q_f) * sin)
    k_embed = (k_f * cos) + (_rotate_half(k_f) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)

@torch.library.register_fake("sglang::apply_rotary_pos_emb")
def apply_rotary_pos_emb_ppu_fake(q, k, cos, sin, unsqueeze_dim=1):
    # Shape inference only — 返回 q, k 本身（与真实输出同 shape/dtype）
    return q, k

# sglang plugin hook 注册（仿 ppu_fa3_hooks.py 的写法）
@plugin_hook("sglang.srt.layers.rotary_embedding.utils.apply_rotary_pos_emb",
             type=HookType.REPLACE)
def _ppu_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    return apply_rotary_pos_emb_ppu(q, k, cos, sin, unsqueeze_dim)
```

`mutates_args=()` 表示 op 不修改入参（RoPE 写新 tensor 给调用方）。

### Hook 影响面

替换 `sglang.srt.layers.rotary_embedding.utils.apply_rotary_pos_emb` 这一个 alias 后，
所有调用点都走新路径：

- **ViT**：`sglang/srt/layers/attention/vision.py:1260`（崩溃源头）
- **LM 文本**：`sglang/srt/layers/rotary_embedding/base.py:110` 的 `apply_rotary_emb_wrapped`
- **LM 文本**：`sglang/srt/layers/rotary_embedding/mrope.py:572` 的 `_apply_rotary_emb_wrapped`

三处都被同时修复。

---

## 5. 验证（已跑过，无 benchmark 数字）

| 项 | 结果 |
|---|---|
| Hook 模块 import | OK，`torch.ops.sglang.apply_rotary_pos_emb` 已注册 |
| Real impl forward（realistic ViT shape `[1024, 16, 128]` bf16） | OK |
| 数值与 upstream native 比对 | **bit-identical**（max diff = 0.0） |
| `torch.compile` 包一层 | OK，inductor 不再崩 |
| **CUDA Graph capture + replay**（**原崩溃场景**） | **OK** |
| `cos.size(-1)*2 == head_size` cat 路径（`vision.py:1257`）| OK |
| 平台 init 加载 hook 后 apply_hooks 替换 alias | OK，`apply_rotary_pos_emb` 已被替换为 `_ppu_apply_rotary_pos_emb` |

具体的部署步骤、验证脚本、常见问题见 `USAGE.md`。

---

## 6. 与之前 suppress_errors patch 的对比

| | suppress_errors（已撤） | ppu_rotary_hooks（本方案） |
|---|---|---|
| 修的位置 | `vit_cuda_graph_runner.py` | `rotary_embedding/utils.py` 的 alias |
| 怎么修 | 吞掉 inductor 抛出的异常 | 把函数换成 inductor 看不到的形式（custom-op） |
| 性能影响 | 退化成每次 eager 重跑，无 inductor 优化 | 仍走 inductor 的 graph capture 路径，但 RoPE 不进 inductor body |
| 是否治本 | 否 | **是** |
| 是否需要修改 vendor 代码 | 是 | **否**（只动 hook + 平台 init 一行 import） |

---

## 7. 已知 trade-off：hook 把 RoPE 拆成多个图节点

当前 hook 实现保留 upstream native 的 eager 数学（mul、add、cat 等），**没有**写 fused
PPU CUDA kernel。后果：

- CUDA Graph replay 时，RoPE 占 **~6 个图节点**（mul + add + cat + mul + add + cast 各一次）
- 之前 `@torch.compile` native 路径在真 NVIDIA 卡上是 1 个 fused triton kernel 节点
- PPU 上 `@torch.compile` 因 `_sfdp_init` 不安全走 fallback（应该也是多节点），所以当前
  实现的 launch overhead 不一定比 native 路径差——但**确定没赢**

### 改进方向（按工作量递增）

| 方案 | 工作量 | 收益 |
|---|---|---|
| **A. 现状**：6 个 eager kernel 节点 | 0 | crash 修复 + capture 安全 |
| B. 写 fused PPU CUDA kernel（`load_inline` JIT） | ~100 行 CUDA + 调试 | replay 时 RoPE launch 从 6 降到 1 |
| C. 复用 PPU fork 里已有的 fused kernel（如果有） | 需要查 `sgl_kernel` / `flashinfer` | 同 B 但 0 维护成本 |

如果 perf 是首要目标，走 B 或 C；如果稳定性 + 可读性优先，保持 A。

---

## 8. 文件清单

```
inductor_hook_PPU/
├── README.md                     # 本文件：设计与根因
├── USAGE.md                      # 部署 / 验证 / 回滚（推荐）
├── PROGRESS.md                   # 进度速记
├── analysis/                     # 根因 + PPU 验证笔记
├── docs/                         # T-Head / flashinfer / FA3 参考资料摘录
└── hook/
    └── ppu_rotary_hooks.py       # 主 hook 实现（已部署到 site-packages）
```
