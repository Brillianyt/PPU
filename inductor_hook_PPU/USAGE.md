# PPU RoPE / inductor Hook — 使用文档

> **范围**：本文档说明 **如何部署、验证、回滚** `ppu_rotary_hooks.py`，**不包含 benchmark 数字**。
> 关于 root cause / 设计细节 / 为什么这样写 → 见 `README.md`。

---

## 1. 这个 hook 解决什么问题

`apply_rotary_pos_emb_native` 在 `sglang/srt/layers/rotary_embedding/utils.py` 上装饰了
`@torch.compile(dynamic=True, ...)`。在 ViT 的 CUDA Graph capture 流程里，第一次调用该函数
会触发 inductor → `_sfdp_init` → `_get_sfdp_patterns` → `copy_args_to_cpu_if_needed`（pin_memory 路径），
**在 PPU 后端这个路径不安全，capture 直接失败**。

PPU 是 T-Head 真武的 CUDA-兼容卡，所有 cuda API 都能调，但 **CUDA Graph capture 的语义边界**
跟真 NVIDIA 卡不完全一致——上面那条 pin_memory / autotune scratch 路径正好落在不兼容区。

Hook 把 RoPE 注册成 inductor 的 **opaque custom-op**，让 inductor 看不到函数体，不去 trace，也就不会
触发 `_sfdp_init`。PiCG / ViT CG capture 因此能顺利通过。

---

## 2. 文件清单

```
inductor_hook_PPU/
├── README.md                 # 设计与根因分析（推荐先读）
├── USAGE.md                  # 本文件：部署 / 验证 / 回滚
├── PROGRESS.md               # 今日工作进度速记
├── docs/                     # T-Head / flashinfer / FA3 参考摘录
└── hook/
    └── ppu_rotary_hooks.py   # hook 源码（已部署到 site-packages）
```

部署后的运行副本：

```
/usr/local/lib/python3.12/site-packages/sglang/srt/hardware_backend/ppu/attention/ppu_rotary_hooks.py
/usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py:64-68   # import 触发点
```

---

## 3. 部署步骤

### 3.1 复制 hook 到 PPU 扩展目录

```bash
cp hook/ppu_rotary_hooks.py \
  /usr/local/lib/python3.12/site-packages/sglang/srt/hardware_backend/ppu/attention/
```

这个目录是 PPU 自带的 attention hook 扩展目录（已经放了 `ppu_fa3_hooks.py` /
`ppu_dsa_hooks.py` / `ppu_flashmla_hooks.py`），hook 文件按相同命名规范放进去即可。

### 3.2 在 PPU platform init 注册 hook 加载

编辑 `/usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py`，
在文件末尾的 `if "PPU_SDK" in os.environ:` 块里加一行：

```python
import sglang.srt.hardware_backend.ppu.attention.ppu_dsa_hooks       # noqa: F401
import sglang.srt.hardware_backend.ppu.attention.ppu_fa3_hooks       # noqa: F401
import sglang.srt.hardware_backend.ppu.attention.ppu_flashmla_hooks  # noqa: F401
import sglang.srt.hardware_backend.ppu.attention.ppu_rotary_hooks   # noqa: F401  ← 新加
```

`PPU_SDK` 环境变量由 PPU runtime 设置——只要 server 在 PPU 上启动，`import` 就会被触发，
`@plugin_hook` 装饰器随之生效。

### 3.3 验证 hook 已生效

启动 server 后随便发个请求，然后：

```bash
# 1. 检查 RoPE alias 是否被替换
python3 -c "
from sglang.srt.layers.rotary_embedding import utils
print('apply_rotary_pos_emb =', utils.apply_rotary_pos_emb.__module__,
      utils.apply_rotary_pos_emb.__name__)
"
# 期望输出包含 '_ppu_apply_rotary_pos_emb'

# 2. 检查 custom-op 是否注册
python3 -c "
import torch
print(torch.ops.sglang.apply_rotary_pos_emb)
"
# 期望输出 schema 不报错

# 3. server log 里搜索
grep -i "rotary\|custom_op\|plugin_hook" <server_log>
# 应该看到 hook 模块被 import 一次（无 warning / 无 error）
```

---

## 4. 验证 capture 不再崩

### 4.1 单元验证（推荐先跑这个）

```python
import torch
from sglang.srt.hardware_backend.ppu.attention import ppu_rotary_hooks  # noqa

# Realistic ViT shape: [tokens, heads, head_dim]
q = torch.randn(1024, 16, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn_like(q)
cos = torch.randn(1024, 128, dtype=torch.bfloat16, device="cuda")
sin = torch.randn_like(cos)

# 1. Forward 跑通
q_out, k_out = ppu_rotary_hooks.apply_rotary_pos_emb_ppu(q, k, cos, sin)
assert q_out.shape == q.shape
assert q_out.dtype == q.dtype

# 2. CUDA Graph capture + replay 不崩
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    q2, k2 = ppu_rotary_hooks.apply_rotary_pos_emb_ppu(q, k, cos, sin)
g.replay()
torch.cuda.synchronize()
print("capture+replay OK")
```

如果这个跑得过，server 上的 PiCG / ViT CG capture 也应该过。

### 4.2 E2E 验证

```bash
# 用现有 wrapper 启 server，正常发请求
SGLANG_SAIL_USE_ACEXT_CUDA=0 python3 /mnt/workspace/PPU/benchmark_public.py \
  --dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --backend sglang --num-samples 10 --warmup-samples 1 \
  --output /tmp/hook_smoke_test.json
# 期望：server 启动顺利 + 10/10 sample 完成 + 无 inductor / pin_memory / capture 报错
```

之前 R8 配置（`--enforce-piecewise-cuda-graph`）会在 ViT capture 阶段崩栈底在
`_get_sfdp_patterns`。hook 部署后这条栈不应该再出现。

---

## 5. 常见问题与排查

### Q1：还是崩在 `_get_sfdp_patterns` / `pin_memory`

确认 hook 模块确实被 import 了：

```bash
grep "ppu_rotary_hooks" /usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py
# 应该有这一行（注释也行）
```

如果 import 行在但没生效，检查 `PPU_SDK` 是否设置：

```python
import os; print(os.environ.get("PPU_SDK"))
# 应该非 None
```

### Q2：启动报 `RuntimeError: cannot import name 'plugin_hook'`

sglang 版本太旧，`sglang/srt/plugins/hook_registry.py` 不存在。hook 需要 sglang 0.5.x
及以上。

### Q3：Custom-op 重复注册

如果 server log 报 `Trying to register an operator (sglang::apply_rotary_pos_emb) with
the same name as an existing operator`，说明 hook 被 import 多次。检查 `platforms/ppu.py`
的 `import` 行没有被复制粘贴。

### Q4：想看 RoPE 实际调用多少次

```bash
SGLANG_DEBUG_GEMV=0 SGLANG_SAIL_FLA_CUDA=1 \
  python3 -c "
import torch
from sglang.srt.hardware_backend.ppu.attention import ppu_rotary_hooks
torch.library.opcheck(ppu_rotary_hooks.apply_rotary_pos_emb_ppu, ...)
"
```

### Q5：性能比 baseline 差

如果 hook 部署后 perf 跟之前 baseline 出现差距，可能原因：

1. **Hook 拆成 6 kernel**——当前实现把 RoPE 拆成 mul/add/cat/mul/add，每个 graph replay
   付 6 次 launch overhead。如果有需要可以再写一个 fused PPU CUDA kernel 把这 6 步合并成 1 个。
   实现方法见 `README.md §7` 的"未来工作"小节。

2. **PiCG enforce 模式**——如果用 `--enforce-piecewise-cuda-graph`，runtime recompile 会
   触发 `cuda_piecewise_backend.py:171` 的 stream assert。**不要开 enforce**，让 PiCG 自然
   fallback 到 eager 即可。

3. **MM cache 配置**——`SGLANG_VLM_CACHE_SIZE_MB` 控制 multimodal embedding cache，
   `SGLANG_MM_FEATURE_CACHE_MB` 控制 IPC transport pool。两者对单 worker / unique-image
   benchmark 影响都不大，按 wrapper 当前值即可。

---

## 6. 回滚

如果 hook 部署后出现回归（性能、准确率、稳定性任何一项），按下面顺序回滚：

### 6.1 临时禁用（不改文件）

```bash
# 把 import 行注掉（最快）
sed -i 's/^.*ppu_rotary_hooks.*$/# &/' \
  /usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py
```

重启 server。RoPE alias 会回退到 `apply_rotary_pos_emb_native`（带 `@torch.compile`），
也就是回到 hook 之前的状态——同时也会回到原来的崩溃。

### 6.2 完整删除

```bash
rm /usr/local/lib/python3.12/site-packages/sglang/srt/hardware_backend/ppu/attention/ppu_rotary_hooks.py
sed -i '/ppu_rotary_hooks/d' \
  /usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py
```

### 6.3 回滚到旧版（如果有备份）

```bash
# 如果改 ppu.py 之前有备份
cp /path/to/backup/ppu.py \
  /usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py
```

---

## 7. 自检 checklist

部署前 / 部署后 / 每次升级 sglang 后各跑一次：

- [ ] `hook/ppu_rotary_hooks.py` 与 `site-packages/.../ppu_rotary_hooks.py` 内容一致
- [ ] `platforms/ppu.py` 第 68 行（最后一行 import）存在
- [ ] `python3 -c "import torch; print(torch.ops.sglang.apply_rotary_pos_emb)"` 不报错
- [ ] 单元验证脚本（§4.1）跑过：forward OK + capture+replay OK
- [ ] E2E 验证脚本（§4.2）跑过：10/10 sample 完成
- [ ] server log 里没有 `pin_memory` / `_get_sfdp_patterns` / `capture stream is not set` 相关报错

---

## 8. 跟其他 PPU hook 的关系

`ppu_rotary_hooks.py` 跟 sglang PPU fork 已有的三个 hook **完全独立**，互不依赖：

| Hook | 替换的算子 | 触发场景 |
|---|---|---|
| `ppu_fa3_hooks.py` | flash-attn 3 forward/backward | 主 attention |
| `ppu_dsa_hooks.py` | DeepSeek-style sparse attention | 备用 attention |
| `ppu_flashmla_hooks.py` | FlashMLA | MLA 路径 |
| **`ppu_rotary_hooks.py`** | **RoPE** | **RoPE（在 attention forward 里）** |

可以单独启用或禁用任意一个，不影响其他。
