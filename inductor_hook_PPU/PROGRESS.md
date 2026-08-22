# PPU RoPE Hook 工作进度（截至当前）

> **范围**：今日工作总结 + 工作方法论（不包含 benchmark 数字）。
> 详细设计与部署 → 见 `README.md` / `USAGE.md`。

---

## 已完成 ✅

### 1. 根因定位
- ViT CG capture 崩溃栈：`_create_graph` → `VisionAttention.forward` → `apply_rotary_pos_emb` → `@torch.compile` → inductor → `_get_sfdp_patterns` → pin_memory 路径
- PPU 是 cuda-兼容加速卡，CUDA Graph capture 上下文里 inductor 的 SDPA pattern init + autotune scratch 不安全

### 2. PPU 语义验证
- `torch.cuda.is_available()` 返回 True，设备名 "PPU-ZW810E"，capability `(8, 0)`
- CUDA Graph capture 本身 OK，`pin_memory=True` 在 CPU tensor 上 OK
- 无 PPU device namespace（不能用 `torch.device('ppu')`）

### 3. 正确方案选定
- 参考 PPU 自带的 FA3 fork（`flash_attn_interface.py:40,132`）和 flashinfer（`flashinfer/rope.py`）的模式
- `@torch.library.custom_op` + `@torch.library.register_fake` —— 让 inductor 把 RoPE 当成 opaque op
- **不修改 vendor sglang 代码**

### 4. Hook 实现（`hook/ppu_rotary_hooks.py`）
- 自定义 op 名：`sglang::apply_rotary_pos_emb`
- 真函数：复用 upstream native 的数学，无 `@torch.compile`
- Fake 函数：返回 `(q, k)` 给 inductor 做 shape inference
- 通过 `@plugin_hook(REPLACE)` 替换 `apply_rotary_pos_emb` alias

### 5. Hook 部署
- 复制到 `/usr/local/lib/python3.12/site-packages/sglang/srt/hardware_backend/ppu/attention/ppu_rotary_hooks.py`
- 在 `/usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py:68` 加 import

### 6. 数值 + Capture 验证（无 benchmark 数字，详见 `USAGE.md` §4 / §5）
- 跟 upstream native bit-identical（bf16 max diff = 0.0）
- `torch.compile` wrap 不崩
- **CUDA Graph capture + replay 成功**（原崩溃场景）
- realistic ViT shape `[1024, 16, 128]` bf16 + `cos.size(-1)*2 == head_size` cat 路径都通过

---

## 工作方法论总结（针对 inductor + RoPE 在 PPU 上的正确启动）

### 思路

遇到 PPU 上的 CUDA Graph capture 崩溃，**不要用 suppress_errors 治标**——那只让系统进
eager fallback，丢掉所有 PiCG / ViT CG 加速。要找到上游 op 在 capture 时触发的哪条不安全
路径，把它从 inductor 的 trace 范围里拿出来。

### 做法

1. **抓栈底**：崩溃栈底通常在 `_get_sfdp_patterns` / `copy_args_to_cpu_if_needed` /
   `triton_heuristics` 这类名字——这些都是 inductor 的 pattern init 路径。
2. **找上游 op**：沿着栈往上找到第一个 sglang 自己的 op（这里是 `apply_rotary_pos_emb`）。
3. **参考 PPU 自带库**：FA3 fork / flashinfer fork 都已经把同类 op 用 `custom_op` + `register_fake`
   注册过，照抄模式即可。
4. **写 hook**：把函数注册成 inductor 看不到的 opaque op，保持 fake shape inference 一致。
5. **不要改 vendor 代码**：用 sglang 的 `@plugin_hook(REPLACE)` 机制替换 alias，只动 hook + 一行 import。

### 验证清单（部署必跑）

| 步 | 命令 / 做法 | 期望 |
|---|---|---|
| 1 | `python3 -c "import torch; print(torch.ops.sglang.apply_rotary_pos_emb)"` | schema 输出，无 NameError |
| 2 | 单元 forward（realistic ViT shape） | q_out/k_out shape/dtype 正确 |
| 3 | 数值对比 `apply_rotary_pos_emb_ppu` vs `apply_rotary_pos_emb_native` | max diff = 0.0 |
| 4 | `torch.cuda.graph(...)` 包一层 forward + replay | capture + replay 均无异常 |
| 5 | 启 server，发 10+ multimodal 请求 | server 不崩，所有请求完成 |
| 6 | grep server log `pin_memory\|_get_sfdp_patterns\|capture stream is not set` | 0 命中 |

### 不要做的事

- ❌ 改 `vit_cuda_graph_runner.py` 加 suppress_errors——治标不治本，丢 CG 加速
- ❌ 改 `qwen3_vl.py` / `ppu_fa3_hooks.py` / 其他 vendor sglang 文件（用户已 patch 过）
- ❌ 改 `rotary_embedding/utils.py`——会污染 vendor 代码，且未来升级时 merge conflict
- ❌ 把 logs 写到 `/mnt/workspace/PPU/logs/`——用 symlink 到 `/tmp/ppu_logs/` 替代
- ❌ 开 `--enforce-piecewise-cuda-graph`——runtime recompile 会触发
  `cuda_piecewise_backend.py:171` 的 stream assert，scheduler 直接死

---

## 文件位置

- Hook 源：`/mnt/workspace/PPU/inductor_hook_PPU/hook/ppu_rotary_hooks.py`
- 部署位置：`/usr/local/lib/python3.12/site-packages/sglang/srt/hardware_backend/ppu/attention/ppu_rotary_hooks.py`
- 详细 README（设计 + 根因）：`/mnt/workspace/PPU/inductor_hook_PPU/README.md`
- 部署 + 验证 + 回滚：`/mnt/workspace/PPU/inductor_hook_PPU/USAGE.md`
