# Patch 同步流程

> **TL;DR** — 本仓 `gemv_optimization/` 是 sglang site-packages 的**镜像源**,
> 真正的运行版本通过 `cp -rv` 部署到 `/usr/local/lib/python3.12/site-packages/`。
> 本文档说明改动从镜像源到部署的完整流程,以及反向把 site-packages 上
> 临时修改**同步回**镜像源的步骤。

---

## 1. 仓库目录结构(约定)

```
PPU/
├── evaluation_wrapper.py           # ← 不在 gemv_optimization/ 下,直接编辑
├── README_REPRODUCE.md             # 主入口文档(SSOT)
├── docs/                           # 架构/复现/同步说明
│   ├── 07_PPU_FRAMEWORK.md         # PPU-fused kernel 框架
│   ├── 08_SYNC.md                  # ← 本文件
│   └── ...
├── result_repro/                   # benchmark 输出 JSON(每次跑分时增量写入)
├── logs/                           # server / wrapper 日志(每次启动新增)
│
├── gemv_optimization/              # ← sglang site-packages 改动镜像源
│   ├── sglang-for-sail/python/sglang/
│   │   ├── srt/
│   │   │   ├── environ.py
│   │   │   ├── layers/
│   │   │   │   ├── deep_gemm_wrapper/{compile_utils,entrypoint}.py
│   │   │   │   └── quantization/w8a8_int8.py          ← [G 类] w8a8_int8 加 v10 ISA fast path
│   │   │   ├── models/{qwen3_vl.py, qwen3_5.py, ...}
│   │   │   ├── speculative/eagle_worker_v2.py
│   │   │   ├── hardware_backend/ppu/attention/ppu_fa3_hooks.py
│   │   │   ├── compilation/                            ← [G 类] 新建目录
│   │   │   │   ├── ppu_kernels.py                       ← [G 类] 框架核心
│   │   │   │   ├── pass_manager.py                      ← [G 类] monkey-patch 装 pass
│   │   │   │   └── backend.py                           ← [G 类] inductor_config 默认值
│   │   │   └── platforms/ppu.py
│   │   └── ...
│   └── DeepGEMM-for-sail/deep_gemm/
│       ├── __init__.py
│       └── jit_kernels/{__init__.py, gemv_int8.py}

# ↑ v10 kernel 不在这里 ↑ — 它在独立的 Python package:
#   /mnt/workspace/ISA/src/sglang_kernel_v10/
# 通过 `pip install .` 安装成独立 wheel,sglang 不再 JIT load。
# 镜像源里不再有 gemv_int8_async.{cu,cpp},2026-08-23 删除。
│
└── inductor_hook_PPU/              # RoPE hook(不在 gemv_optimization/,独立 module)
    └── hook/ppu_rotary_hooks.py
```

---

## 2. 正向部署:镜像源 → site-packages(日常使用)

参考 `README_REPRODUCE.md §2.4`,完整命令:

```bash
cd /mnt/workspace/PPU

# 1.1 — A/B/C/E + G 类:sglang 改动(共 12 文件)
cp -rv gemv_optimization/sglang-for-sail/python/sglang/* \
        /usr/local/lib/python3.12/site-packages/sglang/

# 1.2 — D:DeepGEMM 3 文件 (GEMV op)
cp -rv gemv_optimization/DeepGEMM-for-sail/deep_gemm/* \
        /usr/local/lib/python3.12/site-packages/deep_gemm/

# 1.3 — F:Inductor RoPE hook(独立路径)
cp -v inductor_hook_PPU/hook/ppu_rotary_hooks.py \
      /usr/local/lib/python3.12/site-packages/sglang/srt/hardware_backend/ppu/attention/ppu_rotary_hooks.py

# 1.4 — F 注册到 platforms/ppu.py(一次性)
grep -q "ppu_rotary_hooks" \
  /usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py \
  || sed -i '/ppu_flashmla_hooks/a\    import sglang.srt.hardware_backend.ppu.attention.ppu_rotary_hooks  # noqa: F401' \
        /usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py

# 1.5 — 清 __pycache__
rm -rf /usr/local/lib/python3.12/site-packages/sglang/srt/{layers/quantization,layers/deep_gemm_wrapper,models,hardware_backend/ppu/attention,platforms,compilation}/__pycache__
rm -rf /usr/local/lib/python3.12/site-packages/deep_gemm/{,jit_kernels}/__pycache__

# 1.6 — 落地检查(参考 README §3.1)
```

回滚整轮改动:

```bash
# 用 pip 重装 sglang + deep_gemm(还原到原厂版)
pip install --force-reinstall --no-deps sglang==0.5.13+v0.1.0.ppu2.1.1
pip install --force-reinstall --no-deps deep_gemm
# 然后从 §1.6 检查确认全 rollback
```

---

## 3. 反向同步:site-packages → 镜像源(改动回写)

> **何时需要这个** — 在 site-packages 上做了临时 patch(比如在
> `evaluation_wrapper.py` 上加了 env var 默认值、debug print 等),需要把改动
> 回写到 `gemv_optimization/` 镜像源,提交进 git。

### 3.1 标准流程(用 `diff` 做最小化对比)

```bash
# Step 1:确认改动文件清单
cd /mnt/workspace/PPU
# 把 site-packages 当前版本拷到临时目录
rm -rf /tmp/sglang_site_now
mkdir -p /tmp/sglang_site_now
cp -r /usr/local/lib/python3.12/site-packages/sglang /tmp/sglang_site_now/

# Step 2:逐文件 diff,只同步**确有改动**的
for f in $(find gemv_optimization/sglang-for-sail -type f -name '*.py'); do
    rel=${f#gemv_optimization/sglang-for-sail/python/sglang/}
    site=/tmp/sglang_site_now/$rel
    if ! diff -q "$f" "$site" >/dev/null 2>&1; then
        echo "DIFF: $rel"
    fi
done

# Step 3:把改动文件拷回镜像源
# 例:pass_manager.py 改了
cp /usr/local/lib/python3.12/site-packages/sglang/srt/compilation/pass_manager.py \
   gemv_optimization/sglang-for-sail/python/sglang/srt/compilation/pass_manager.py

# Step 4:site-packages 上有但镜像源没有的新文件(比如新建的 ppu_kernels.py)
for f in $(find /usr/local/lib/python3.12/site-packages/sglang -name '*.py' \
              -newer gemv_optimization/sglang-for-sail/python/sglang/__init__.py); do
    rel=${f#/usr/local/lib/python3.12/site-packages/sglang/}
    dst=gemv_optimization/sglang-for-sail/python/sglang/$rel
    if [ ! -f "$dst" ]; then
        echo "NEW: $rel"
        mkdir -p "$(dirname $dst)"
        cp "$f" "$dst"
    fi
done

# Step 5:验证 — 重新跑 README §3.1 的 patch 落地检查
```

### 3.2 快速流程(只针对本次会话的几处改动)

本次会话做了 4 类 site-packages 改动(A 类 wrapper + G 类框架)。反向同步命令:

```bash
cd /mnt/workspace/PPU

# A:wrapper 默认值 — 已在 /mnt/workspace/PPU/evaluation_wrapper.py(直接 git diff 就看得到)
#    如果只是改了 wrapper,直接 commit evaluation_wrapper.py 即可

# G:框架 — 3 个 sglang 文件 + 1 个新文件
mkdir -p gemv_optimization/sglang-for-sail/python/sglang/srt/compilation

cp /usr/local/lib/python3.12/site-packages/sglang/srt/compilation/ppu_kernels.py \
   gemv_optimization/sglang-for-sail/python/sglang/srt/compilation/

cp /usr/local/lib/python3.12/site-packages/sglang/srt/compilation/pass_manager.py \
   gemv_optimization/sglang-for-sail/python/sglang/srt/compilation/

cp /usr/local/lib/python3.12/site-packages/sglang/srt/compilation/backend.py \
   gemv_optimization/sglang-for-sail/python/sglang/srt/compilation/

cp /usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/w8a8_int8.py \
   gemv_optimization/sglang-for-sail/python/sglang/srt/layers/quantization/
```

### 3.3 提交前清单

```bash
git status gemv_optimization/ evaluation_wrapper.py docs/ result_repro/
# 期望看到:
#   modified:   gemv_optimization/sglang-for-sail/python/sglang/srt/compilation/backend.py
#   modified:   gemv_optimization/sglang-for-sail/python/sglang/srt/compilation/pass_manager.py
#   modified:   gemv_optimization/sglang-for-sail/python/sglang/srt/layers/quantization/w8a8_int8.py
#   new file:   gemv_optimization/sglang-for-sail/python/sglang/srt/compilation/ppu_kernels.py
#   modified:   evaluation_wrapper.py
#   modified:   README_REPRODUCE.md
#   modified:   docs/07_PPU_FRAMEWORK.md (新加)
#   modified:   docs/08_SYNC.md (新加)
#   new file:   result_repro/result_dev_en_r8_inductor_100.json
#   new file:   result_repro/result_dev_en_r8_final_100.json
#   new file:   result_repro/result_dev_en_r8_autotune_100.json

# 跑一遍落地检查 + smoke benchmark(20 样本),确认改动没破环
bash /mnt/workspace/PPU/README_REPRODUCE.md §3.1   # 实际不会执行,只是记着要做
PPU_SDK=/usr/local/PPU_SDK python3 benchmark_public.py \
  --dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --backend sglang --num-samples 20 --warmup-samples 8 \
  --output /tmp/smoke_check.json
# 期望:accuracy 0.7±0.1,TTFT median ~63ms,total elapsed ~12s(20 样本)
```

---

## 4. 跨机器同步

把 PPU 整个仓库 rsync 到目标机器后:

```bash
# 目标机器
cd /mnt/workspace/PPU
git pull                                    # 拉最新改动
pip install --force-reinstall --no-deps sglang==0.5.13+v0.1.0.ppu2.1.1   # 还原原厂
pip install --force-reinstall --no-deps deep_gemm

# 然后跑正向部署(§2)
bash -c "$(cat README_REPRODUCE.md | grep -A 30 'cp -rv gemv_optimization')"
# 或者手工执行 §2 的命令块

# 最后跑一次完整 100 样本验证(对标 result_repro/*.json)
PPU_SDK=/usr/local/PPU_SDK python3 benchmark_public.py \
  --dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --backend sglang --num-samples 100 --warmup-samples 8 \
  --output result_repro/result_dev_en_r8_100.json

# 对比 result_repro/{result_dev_en_r8_100.json, result_dev_en_r8_final_100.json}
# 期望:数字在 R8 spec ±5% 范围内(详见 README §3.2)
```

---

## 5. 常见 sync 陷阱

### 5.1 site-packages 上有,但镜像源没有的文件

```bash
# 发现方法
find /usr/local/lib/python3.12/site-packages/sglang -name '*.py' \
     -newer /mnt/workspace/PPU/gemv_optimization/sglang-for-sail/python/sglang/__init__.py
```

例:`ppu_kernels.py` 是新建文件,镜像源 `gemv_optimization/sglang-for-sail/.../compilation/` 目录也是新建的。

### 5.2 镜像源有,但 site-packages 没有的文件(过期)

```bash
find /mnt/workspace/PPU/gemv_optimization/sglang-for-sail -name '*.py' \
     -newer /usr/local/lib/python3.12/site-packages/sglang/__init__.py
```

要同步(再次跑 §2 部署命令)。

### 5.3 __pycache__ 干扰 diff

site-packages 上的 `__pycache__/*.pyc` 可能让 `diff` 报噪音。部署前先清(§2 step 1.5),diff 前先清镜像源 `find ... -name __pycache__ -exec rm -rf {} +`。

### 5.4 sglang 版本漂移

如果 `pip install --upgrade sglang` 拉了新版本,镜像源会过期。要么:
- 把镜像源当 freeze,锁定 sglang 版本(本次会话就是这个做法)
- 把镜像源当成 git patch,接受随版本漂移

本次 R8 复现依赖 `sglang==0.5.13+v0.1.0.ppu2.1.1`,其它版本未测试。

---

## 6. 2026-08-23 sync round: v10 prebuilt integration + layout bug fix

**改动:** 把 v10 INT8 GEMV fast path 从 `torch.utils.cpp_extension.load`
(JIT 编译,每次冷启动延迟 + 破坏 CUDA graph capture)改成正经 wheel:
`/mnt/workspace/ISA/src/sglang_kernel_v10/`,`pip install .` 装成独立 Python
package。同时修了 dispatcher 的 layout bug(`_isa_v10_get_w_nk` 之前做错了
`.t().contiguous()`,应该是 `.contiguous()`)。

**端到端验证:**

| 配置 | TPS | acc |
|---|---|---|
| baseline (no v10) | 411.6 | 0.73 |
| v10 with bug | 353.0 (-14%) | 0.70 |
| **v10 fixed** | **421.1 (+2.3%)** | **0.73** |

**这次同步做了什么:**
1. ✅ `sglang_kernel_v10/` wheel 安装到 site-packages(独立 package,不污染 sglang)
2. ✅ `w8a8_int8.py` 镜像源 = 当前部署版 (含 `_ISA_V10_ENABLE` / `_ISA_V10_W_CACHE` / `_isa_v10_get_w_nk` / `_isa_load_fast_path` 移除)
3. ✅ 删除镜像源里的过时 JIT artifact: `gemv_int8_async.{cu,cpp}` 和 `sglang/sglang_kernel_v10/` 旧 stub
4. ✅ `docs/05_INTEGRATION.md`(ISA 仓)记录 dispatcher 集成 + layout bug + cache invariants + CG 安全性
5. ✅ `docs/08_SYNC.md`(本节)记录这次 sync 的来源和 diff 范围

**安装步骤:**
```bash
# 一次性:装 v10 wheel
cd /mnt/workspace/ISA/src/sglang_kernel_v10 && pip install .

# 一次性:deploy sglang dispatcher patch
cp -v /mnt/workspace/PPU/gemv_optimization/sglang-for-sail/python/sglang/srt/layers/quantization/w8a8_int8.py \
      /usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/w8a8_int8.py

# 验证
python3 -c "
import torch
from sglang_kernel_v10 import gemv_int8_async
from sglang.srt.layers.quantization.w8a8_int8 import gemv_int8_i8i8bf16, _ISA_V10_ENABLE
print('v10 enabled:', _ISA_V10_ENABLE)
A = torch.randint(-128, 127, (1, 2048), dtype=torch.int8, device='cuda')
W = torch.randint(-128, 127, (2048, 8192), dtype=torch.int8, device='cuda')  # (K, N)
As = torch.ones(1, dtype=torch.float32, device='cuda')
Bs = torch.ones(8192, dtype=torch.float32, device='cuda')
out = torch.empty(1, 8192, dtype=torch.bfloat16, device='cuda')
gemv_int8_i8i8bf16(A, As, W, Bs, out)
ref = (A.float() @ W.float()).to(torch.bfloat16)
print('max diff:', (out.float() - ref.float()).abs().max().item())  # expect 0.0
"
```

**回滚:**
```bash
# 关掉 v10 fast path (回 baseline)
sed -i 's/^_ISA_V10_ENABLE = True/_ISA_V10_ENABLE = False/' \
    /usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/w8a8_int8.py
# 完全卸载 wheel
pip uninstall sglang_kernel_v10
```

---

## 7. 2026-08-23 最终汇报 (Session Summary)

### 7.1 本次会话做了什么

| 阶段 | 产出 |
|---|---|
| Phase 0 | 调研 HGGC PDF → `ISA/docs/03_HARDWARE.md` + `ISA/notes/HGGC_PPU001_NOTES.md` |
| Phase 1 | GEMV profile → deep_gemm 32μs / 4.5% HBM 利用率 (`ISA/docs/01_PROFILE.md`) |
| Phase 2 | PDK 架构设计 (`ISA/docs/02_ARCHITECTURE.md`) |
| Phase 3.0 | naive SIMD baseline = 26.78μs vs deep_gemm 31.73μs |
| Phase 3.1 | HGGC mma 路径 — **否决**(`03_FAILED.md` F9,deep_gemm 自己不用 mma for m=1) |
| Phase 3.2 | v10 HBM-direct kernel = 6.7μs vs deep_gemm 30.3μs = **4.5×** microbench |
| Phase 3.2.1 | cp.async 双 buffer — **未跑赢 v10**(微 bench +9%,shmem bank conflicts) |
| 集成 v0 | JIT hack — 反例(坏做法) |
| 集成 v1 | 独立 wheel (`sglang_kernel_v10/`) + dispatcher patch — **正解** |
| 修 bug | F10 dispatcher `.t().contiguous()` reverse-transpose → `.contiguous()` 修好 |
| 端到端 | 100 样本 Ablation A vs v10,公式评分分析 |

### 7.2 数字一览 (mmbench_dev_en 100, 2026-08-23, 当前 wrapper 配置)

| 方案 | TPS | TTFT | acc | 评分(40/30/30) |
|---|---|---|---|---|
| baseline (deep_gemm, 无 v10) | 411.6 | 72.7 | 0.73 | **SSOT 提交件** |
| + ISA v10 (opt-in) | 421.1 | 75.3 | 0.73 | +0.4% (基本持平) |

### 7.3 决策记录

| 决策 | 结论 | 文档位置 |
|---|---|---|
| **v10 默认配置** | **off**,opt-in(`SGLANG_ISA_V10_ENABLE=1` 启用) | `README.md` + `02_RESULTS.md` §2.1 |
| HGGC mma 路径 | 跳过,深_gemm 也不用 | `03_FAILED.md` F9 |
| v10 kernel 形态 | HBM-direct SIMT(无 cp.async, 无 mma) | `04_OPTIMIZATIONS.md` §4.6 |
| 集成模式 | 独立 wheel(`pip install .`),非 JIT hack | `04_OPTIMIZATIONS.md` §4.6.3 |
| `_ISA_V10_W_CACHE` | 永久 cache,key = `B.data_ptr()`,每层 16MB 一次性 contiguous | `05_INTEGRATION.md`(ISA 仓) |
| R6/R8 历史 baseline | **不可复现**,作废;新 SSOT = 2026-08-23 实测 | `02_RESULTS.md` §2.1 |
| Phase 3.3 persistent kernel | **待做**(预期 +20-60% TPS),当前最高优 | `notes/TODO.md`(ISA) |

### 7.4 待办 (Phase 3.3 优先级)

1. **Persistent kernel**: 1 launch 串起 200+ GEMV,消 launch overhead。预期 TPS 500-700
2. **Per-row As 多 token 支持**: 当前只 M=1,M>1 走 fallback
3. **多 K 模板**: 当前只 K=2048,K=6144 走 fallback
4. **TTFT 回归根因**: 3.6% 是 first-request 噪音还是 v10 真拖慢? Phase 3.3 一并分析

