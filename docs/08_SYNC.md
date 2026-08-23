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
│   │   │   │   └── quantization/w8a8_int8.py          ← [B 类] INT8 注入 + M-aware 分派
│   │   │   ├── models/{qwen3_vl.py, qwen3_5.py, ...}
│   │   │   ├── speculative/eagle_worker_v2.py
│   │   │   ├── hardware_backend/ppu/attention/ppu_fa3_hooks.py
│   │   │   └── platforms/ppu.py
│   │   └── ...
│   └── DeepGEMM-for-sail/deep_gemm/
│       ├── __init__.py
│       └── jit_kernels/{__init__.py, gemv_int8.py}
│
└── inductor_hook_PPU/              # RoPE hook(不在 gemv_optimization/,独立 module)
    └── hook/ppu_rotary_hooks.py
```

---

## 2. 正向部署:镜像源 → site-packages(日常使用)

参考 `README_REPRODUCE.md §2.4`,完整命令:

```bash
cd /mnt/workspace/PPU

# 1.1 — A/B/C/E/F 类:sglang 改动(纯 Python 覆盖)
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

# Step 4:site-packages 上有但镜像源没有的新文件(比如某个新建文件)
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

### 3.2 快速流程(针对本次会话的几处改动)

本次会话改动示例(pass_manager + w8a8_int8,纯修改非新文件)。反向同步命令:

```bash
cd /mnt/workspace/PPU

# 改动过的文件逐个 cp 回镜像源
cp /usr/local/lib/python3.12/site-packages/sglang/srt/compilation/pass_manager.py \
   gemv_optimization/sglang-for-sail/python/sglang/srt/compilation/
cp /usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/w8a8_int8.py \
   gemv_optimization/sglang-for-sail/python/sglang/srt/layers/quantization/
# wrapper 默认值 — 已在 /mnt/workspace/PPU/evaluation_wrapper.py(直接 git diff 就看得到)
```

### 3.3 提交前清单

```bash
git status gemv_optimization/ evaluation_wrapper.py docs/ result_repro/
# 期望看到:
#   modified:   gemv_optimization/sglang-for-sail/python/sglang/srt/compilation/pass_manager.py
#   modified:   gemv_optimization/sglang-for-sail/python/sglang/srt/layers/quantization/w8a8_int8.py
#   deleted:    gemv_optimization/sglang-for-sail/python/sglang/srt/compilation/ppu_kernels.py
#   deleted:    docs/07_PPU_FRAMEWORK.md
#   modified:   README_REPRODUCE.md
#   modified:   docs/08_SYNC.md
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

例:若某次会话新建了 `qwen3_5.py`,镜像源 `gemv_optimization/sglang-for-sail/.../models/` 里也要补上同名文件。

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
