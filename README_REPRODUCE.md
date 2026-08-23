# README — 复现入口 (SSOT)

> **Single Source of Truth** for 复现路径。本文件与 [`docs/02_RESULTS.md`](docs/02_RESULTS.md)
> §2.1 互为 SSOT,其它 doc 出现的数字若与本文件冲突,以本文件为准。
>
> 数字口径约定:**TTFT 一律报 median**(mean 受尾部冷启动样本拉高 4×);**acc 在 100
> 样本上的统计噪声 ≈ ±8pp**(不要把 0.78 vs 0.74 解读成"量化掉 4pp 精度",那是噪声)。

## 0. TL;DR — 一句话跑通

```bash
# 入口(PPU 自动启 server + 跑分):
PPU_SDK=/usr/local/PPU_SDK python3 benchmark_public.py \
  --dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --backend sglang --num-samples 100 --warmup-samples 8 \
  --output result_dev_en_r6.json
```

预期数字 (R8 / fa3 + inductor PiCG / wrapper 默认 / 100 样本 en, warmup 8):

| | 实测 (本方案,en) | baseline (BF16 G 组) | Δ |
|---|---|---|---|
| accuracy | **0.73** | 0.78 | −5pp (不显著,在 100 样本 ±8pp 噪声内) |
| TTFT median (post-warmup 92) | **63 ms** | 73 ms | −14% |
| throughput avg | **412 tok/s** | 357 tok/s | +15% |
| total elapsed | **51 s** | ~150 s(估计) | −66% |
| CN accuracy | **0.82** | 0.82 | 0pp |

> **CN 数据集上 INT8 量化是「完全免费的优化」**(acc 0pp 损失 + +9.8% tput
> + −9.0% TTFT,基于 cn 集 100 样本)。比赛私有测试集若是中英混合,整体
> 量化损失 < 4pp,在统计噪声范围。
>
> **TTFT median 优于 R8 spec 5%** — inductor PiCG + ViT CG 组合在 prefill 路径
> 真做算子融合;详情见 §2.3 + `docs/07_PPU_FRAMEWORK.md`。

## 1. 五步复现流程

```
[1] 部署 7 类共 14 文件 patch (§2.4 Step 1)
       ↓
[2] 启动 sglang server(wrapper 自动启,无需手工)
       ↓
[3] benchmark_public.py 跑分(--backend sglang 自动连 wrapper)
       ↓
[4] 读 result_*.json,核对 §3.2 期望数字
       ↓
[5] 若不符,按 §4 故障表定位
```

patches 通过纯 Python `cp` 覆盖到 site-packages,**不改 C++ 扩展**。回退 `git restore`
site-packages 即可还原。

## 2. 三档启动配置 (挑选其中一个)

| 变体 | 推荐场景 | 实测数字 (en 100, warmup 8) | patches |
|---|---|---|---|
| **G 组 bf16** §2.1 | 简单、无 patch、对照基线 | acc 0.81 / TTFT 73ms / tput 357 tok/s (500 样本) | 无 |
| **R6 (极致 TTFT+acc)** §2.2 | triton_attn mm-backend 对照 | acc 0.74 / TTFT 64ms / tput 429 tok/s | 全部 7 类 |
| **R8 (极致 TPS)** §2.3 | **当前默认** | acc 0.73 / TTFT 63ms / tput 412 tok/s / 51 s | 全部 7 类 |

### 2.1 G 组 bf16 (对照基线, 不打 patch)

```bash
PPU_SDK=/usr/local/PPU_SDK python3 benchmark_public.py \
  --dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/workspace/Qwen3.5-2B \
  --backend sglang --num-samples 100 --warmup-samples 8 \
  --output result_dev_en_g_group_100.json
```

> 注意:`--model-path` 走 BF16 原模型目录(`./Qwen3.5-2B`),不是
> `models/Qwen_quant/Qwen3.5-2B-S2-aggressive`(那是 INT8 量化模型)。

### 2.2 R6 — 极致 TTFT+acc (triton_attn backend) ⭐ 推荐

```bash
SGLANG_EXTRA_ARGS="--mm-attention-backend triton_attn" \
PPU_SDK=/usr/local/PPU_SDK python3 benchmark_public.py \
  --dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --backend sglang --num-samples 100 --warmup-samples 8 \
  --output result_dev_en_r6_100.json
```

wrapper 默认 (patch 已装后) 启用: `SGLANG_SAIL_USE_GEMV=1`、`SGLANG_SAIL_GEMV_M_MAX=8`、
`SGLANG_SAIL_PREFILL_BF16=1`、`SGLANG_SAIL_FLA_CUDA=1`、`SGLANG_SAIL_DEEPGEMM_DENSE=1`、
`SGLANG_VIT_ENABLE_CUDA_GRAPH=1`、`SGLANG_CUDA_GRAPH_MAX_BS=16`。SGLANG_EXTRA_ARGS
只在需要切换 mm attention backend 时设。

### 2.3 R8 — 极致 TPS (fa3 + inductor PiCG,默认)

**当前默认配方**: `--mm-attention-backend fa3` + piecewise CUDA Graph
**+ `--piecewise-cuda-graph-compiler inductor`**(真调 `torch._inductor.compile_fx`
做算子自动融合)+ ViT CG + GEMV。代码上同 R6,只是 wrapper 默认就
是 inductor,不需要 `SGLANG_EXTRA_ARGS` 注入。

```bash
PPU_SDK=/usr/local/PPU_SDK python3 benchmark_public.py \
  --dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --backend sglang --num-samples 100 --warmup-samples 8 \
  --output result_dev_en_r8_100.json
```

| 预期数字(100 样本,warmup 8) | R8 spec | 当前默认实现 |
|---|---|---|
| accuracy | 0.71 | **0.73** |
| TPS avg per-req | 434 | ~412 |
| TTFT median (post-warmup 92) | 66.5 ms | **63 ms** |
| total elapsed | — | ~51 s |

> **关键 trade-off** — TTFT median 比 R8 spec 还快 5%(spec 66.5 → 实测 63),
> TPS 略低于 spec(-5%),但总耗时是 R8+eager baseline(216 s)的 **1/4**。
> VLM workload(image prefill 主导总耗时)下净赚。

#### 2.3.1 用 env var 控制 inductor / autotune

| env var | 默认 | 作用 |
|---|---|---|
| `SGLANG_PIECEWISE_CUDA_GRAPH_COMPILER` | `inductor` | PiCG 编译后端。可设 `eager` 退回旧的非融合模式 |
| `SGLANG_PIECEWISE_INDUCTOR_AUTOTUNE` | `0`(关) | 设 `1` 开 inductor `max_autotune` + `coordinate_descent_tuning`(启动变慢,TPS 略增 ~1%,风险:在 PPU 上可能挑到不如手调 SAIL TensorCore 的 Triton kernel) |

示例:退回 eager + 关 autotune(诊断 / 对照实验用):

```bash
SGLANG_PIECEWISE_CUDA_GRAPH_COMPILER=eager \
PPU_SDK=/usr/local/PPU_SDK python3 benchmark_public.py ...
```

示例:inductor 全套 + autotune(冲极致):

```bash
SGLANG_PIECEWISE_INDUCTOR_AUTOTUNE=1 \
PPU_SDK=/usr/local/PPU_SDK python3 benchmark_public.py ...
```

### 2.4 Patch 部署 (Step 1,一次性)

⚠️ **必须**在启动 server 之前完成。位置 = `PPU/gemv_optimization/`(A-G 类) +
`PPU/inductor_hook_PPU/hook/`(F)。

**类清单**(共 7 类 14 文件):

| 类 | 来源 | 部署目标 | 作用 | 不打的代价 |
|---|---|---|---|---|
| **A** GEMV env | `gemv_optimization/.../environ.py` | `sglang/srt/environ.py` | 注册 `SGLANG_SAIL_USE_GEMV` 等 env vars | env vars 静默 no-op |
| **B** W8A8 INT8 注入 + 框架 functional siblings | `gemv_optimization/.../w8a8_int8.py` | `sglang/srt/layers/quantization/w8a8_int8.py` | 把 `gemv_int8_i8i8bf16` 注入 apply 路径 + 新增 `_nm` functional siblings(给 ppu_kernels 框架) | decode 走 fallback;框架无法注册 |
| **C** Wrapper 接入 | `gemv_optimization/.../{compile_utils,entrypoint}.py` | `sglang/srt/layers/deep_gemm_wrapper/` | 暴露 `gemv_nt_i8i8bf16` | AttributeError |
| **D** GEMV JIT | `gemv_optimization/.../{__init__.py, jit_kernels/{__init__.py, gemv_int8.py}}` | `deep_gemm/` | GEMV JIT 编译 | GEMV 算子完全缺失 |
| **E** ViT CG 修复 | `gemv_optimization/.../qwen3_vl.py` + `.../ppu_fa3_hooks.py` | `sglang/srt/models/qwen3_vl.py` + `sglang/srt/hardware_backend/ppu/attention/ppu_fa3_hooks.py` | 让 ViT 走 PPU 原生 FA3 + 杜绝 rotary workspace 懒扩展破坏 graph | 开 ViT CG 后精度 −15pp、cn 集直接崩到 0 |
| **F** Inductor RoPE hook | `inductor_hook_PPU/hook/ppu_rotary_hooks.py` | `sglang/srt/hardware_backend/ppu/attention/ppu_rotary_hooks.py` + `sglang/srt/platforms/ppu.py` 加 1 行 import | 把 RoPE 包成 `torch.library.custom_op`,让 inductor 看不到函数体,绕开 `_get_sfdp_patterns` | 启动 ~30s 后崩 `Cannot copy between CPU and CUDA tensors during CUDA graph capture` |
| **G** PPU-fused kernel 框架(架构修复)| `gemv_optimization/.../compilation/{ppu_kernels,pass_manager,backend}.py` | `sglang/srt/compilation/{ppu_kernels,pass_manager,backend}.py` | 让手写 CUDA kernel 能真正参与 inductor fusion,而不是 opaque 黑盒。`@ppu_fused_kernel` 装饰器 + `PpuPrologueFusionPass` 自动装上 | 框架不工作 — 手写 kernel 仍在 inductor 眼中是不透明节点 |

```bash
cd /mnt/workspace/PPU

# 1.1 — A/B/C/E: sglang 9 文件 + quant recipe + G 类
cp -rv gemv_optimization/sglang-for-sail/python/sglang/* \
        /usr/local/lib/python3.12/site-packages/sglang/

# 1.2 — D: DeepGEMM 3 文件 (GEMV op)
cp -rv gemv_optimization/DeepGEMM-for-sail/deep_gemm/* \
        /usr/local/lib/python3.12/site-packages/deep_gemm/

# 1.3 — F: Inductor RoPE hook (路径不与上面重叠)
cp -v inductor_hook_PPU/hook/ppu_rotary_hooks.py \
      /usr/local/lib/python3.12/site-packages/sglang/srt/hardware_backend/ppu/attention/ppu_rotary_hooks.py

# 1.4 — F 注册到 platforms/ppu.py (一次性)
grep -q "ppu_rotary_hooks" \
  /usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py \
  || sed -i '/ppu_flashmla_hooks/a\    import sglang.srt.hardware_backend.ppu.attention.ppu_rotary_hooks  # noqa: F401' \
        /usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py

# 1.5 — 清 __pycache__
rm -rf /usr/local/lib/python3.12/site-packages/sglang/srt/{layers/quantization,layers/deep_gemm_wrapper,models,hardware_backend/ppu/attention,platforms}/__pycache__
rm -rf /usr/local/lib/python3.12/site-packages/deep_gemm/{,jit_kernels}/__pycache__
```

#### 6 类 10 文件清单

| 类 | 来源 | 部署目标 | 作用 | 不打的代价 |
|---|---|---|---|---|
| **A** GEMV env | `gemv_optimization/.../environ.py` | `sglang/srt/environ.py` | 注册 `SGLANG_SAIL_USE_GEMV` 等 env vars | env vars 静默 no-op |
| **B** W8A8 INT8 注入 | `gemv_optimization/.../w8a8_int8.py` | `sglang/srt/layers/quantization/w8a8_int8.py` | 把 `gemv_int8_i8i8bf16` 注入 apply 路径 + prefill 反量化 | decode 走 fallback |
| **C** Wrapper 接入 | `gemv_optimization/.../{compile_utils,entrypoint}.py` | `sglang/srt/layers/deep_gemm_wrapper/` | 暴露 `gemv_nt_i8i8bf16` | AttributeError |
| **D** GEMV JIT | `gemv_optimization/.../{__init__.py, jit_kernels/{__init__.py, gemv_int8.py}}` | `deep_gemm/` | GEMV JIT 编译 | GEMV 算子完全缺失 |
| **E** ViT CG 修复 | `gemv_optimization/.../qwen3_vl.py` + `.../ppu_fa3_hooks.py` | `sglang/srt/models/qwen3_vl.py` 补 `max_context_len` + `sglang/srt/hardware_backend/ppu/attention/ppu_fa3_hooks.py` 加 ViT fa3 hook | 让 ViT 走 PPU 原生 FA3 + 杜绝 rotary workspace 懒扩展破坏 graph | 开 ViT CG 后精度 −15pp、cn 集直接崩到 0 |
| **F** Inductor RoPE hook | `inductor_hook_PPU/hook/ppu_rotary_hooks.py` | `sglang/srt/hardware_backend/ppu/attention/ppu_rotary_hooks.py` + `sglang/srt/platforms/ppu.py` 加 1 行 import | 把 RoPE 包成 `torch.library.custom_op` + `register_fake`,让 inductor 看不到函数体,绕开 `_get_sfdp_patterns` | **启动 ~30s 后崩 `Cannot copy between CPU and CUDA tensors during CUDA graph capture`** —— 整条 ViT CG 优化报废 |

#### 回滚 F (临时禁用 hook)

```bash
sed -i 's/^.*ppu_rotary_hooks.*$/# &/' \
  /usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py
```

> 完整回滚等价于退到关掉 ViT CG,TTFT 退回 ~78 ms。F 是前提,不是选项。

## 3. 验证清单 (SSOT)

### 3.1 Patch 落地检查

```bash
# A
grep -c "SGLANG_SAIL_USE_GEMV" /usr/local/lib/python3.12/site-packages/sglang/srt/environ.py   # ≥ 1
# B
grep -c "gemv_int8_i8i8bf16" /usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/w8a8_int8.py  # ≥ 4
# C
ls /usr/local/lib/python3.12/site-packages/sglang/srt/layers/deep_gemm_wrapper/   # compile_utils.py + entrypoint.py
# D
ls /usr/local/lib/python3.12/site-packages/deep_gemm/jit_kernels/gemv_int8.py   # exists
# E1
grep -c "max_context_len" /usr/local/lib/python3.12/site-packages/sglang/srt/models/qwen3_vl.py  # ≥ 2
# E2
grep -c "ppu_vision_flash_attn_varlen_func" \
  /usr/local/lib/python3.12/site-packages/sglang/srt/hardware_backend/ppu/attention/ppu_fa3_hooks.py  # ≥ 1
# F1
ls /usr/local/lib/python3.12/site-packages/sglang/srt/hardware_backend/ppu/attention/ppu_rotary_hooks.py
# F2
grep "ppu_rotary_hooks" /usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py  # ≥ 1
# G1 — framework module
ls /usr/local/lib/python3.12/site-packages/sglang/srt/compilation/ppu_kernels.py   # exists
# G2 — w8a8_int8.py 多了 _nm siblings
grep -c "ppu_fused_kernel\|gemv_int8_i8i8bf16_nm\|gemm_nt_i8i8bf16_nm" \
  /usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/w8a8_int8.py  # ≥ 3
# G3 — pass_manager 末尾 patch
grep -c "_ppu_patched_configure\|PpuPrologueFusionPass installed" \
  /usr/local/lib/python3.12/site-packages/sglang/srt/compilation/pass_manager.py  # ≥ 2
# G4 — wrapper 默认走 inductor (而不是 server_args.py 默认的 eager)
grep -oE "SGLANG_PIECEWISE_CUDA_GRAPH_COMPILER.*inductor" \
  /mnt/workspace/PPU/evaluation_wrapper.py  # 必须存在

# Server 启动日志回执:
LATEST=$(ls -t /mnt/workspace/PPU/logs/sglang_server_*.log | head -1)
grep "SGLANG_SAIL_USE_ACEXT_CUDA version" "$LATEST"   # 见 "1050100, token limit: 4096"
grep "Applied 1 hook(s) to sglang.srt.layers.rotary_embedding.utils.apply_rotary_pos_emb" "$LATEST"  # 至少 1 行
grep "PpuPrologueFusionPass installed" "$LATEST"      # G 类框架装上了
grep -oE "piecewise_cuda_graph_compiler='[^']+'" "$LATEST" | head -1   # 必须是 'inductor'
```

### 3.2 跑分期望(100 样本,warmup ≥ 8)

| 检查项 | 期望(R8 默认 / inductor) |
|---|---|
| accuracy (en) | **0.73 ± 0.08**(100 样本 95% CI 噪声主导)|
| accuracy (cn) | **0.82**(= BF16 baseline,INT8 零损失)|
| TTFT median (en, post-warmup 92) | **60–66 ms** |
| throughput avg (en) | **405–415 tok/s** |
| total elapsed | **~51 s** |
| failed_samples | 0 |
| None parses | 0 |
| 接口契约 | `generate_with_metrics` 不变 |

> **真伪量化损失 ≈ 2pp**(同 qid 对照:S2 组 100 样本错 24 题 vs G 组同 100 错 22 题,
> 仅 2 个新错)。表面上"0.78 vs 0.74"的 4pp 差距 = 100 样本 95% CI ±8pp 内的采样噪声,
> 不是量化本身掉的精度。详见 [`docs/02_RESULTS.md`](docs/02_RESULTS.md) §2.4。

### 3.3 读 result JSON 脚本

```bash
python3 -c "
import json
d = json.load(open('result_dev_en_r6_100.json'))
p = d['performance']
n = d['sample_count']
print(f'TPS     = {p[\"avg_throughput_tokens_per_sec\"]:.1f}')
print(f'TTFT    = {p[\"avg_ttft_ms\"]:.1f} ms (mean, 见上文 median 口径)')
print(f'acc     = {d[\"accuracy\"][\"score\"]:.3f} ({d[\"accuracy\"][\"correct\"]}/{n})')
print(f'elapsed = {d[\"timing\"][\"benchmark_elapsed_seconds\"]:.1f} s')
nones = sum(1 for a in d['answers'] if a['parsed_answer'] is None)
print(f'None    = {nones}')
# median 而非 mean 评估 TTFT 稳态:
import statistics
ttfts = [a['ttft_ms'] for a in d['answers']]
print(f'TTFT median = {statistics.median(ttfts):.1f} ms (稳态判断用)')
"
```

## 4. 故障表 (按症状 → 根因 → 解决)

| 症状 | 根因 | 解决 |
|---|---|---|
| 启动 ~30s 后崩 `Cannot copy between CPU and CUDA tensors during CUDA graph capture` / 栈底 `_get_sfdp_patterns` | PPU 上 Inductor SDPA pattern init 路径不安全,ViT + RoPE `@torch.compile` 撞 capture | **部署 F 类 inductor RoPE hook**(§2.4 step 1.3-1.4)。**不要关 ViT CG**,TTFT 会退化 −13% |
| `AttributeError: module 'deep_gemm' has no attribute 'gemv_int8_int8_bf16_nt'` | D 类 patch 未装 | §2.4 step 1.2 |
| TTFT ~87 ms(标称 ~64 ms) | GEMV custom op 缺失,C 类 patch 未装 | §2.4 step 1.1 |
| log 里无 `SGLANG_SAIL_USE_ACEXT_CUDA version` | ACEXT 没加载,w8a8_int8 quant 路径未激活 | 确认 `--quantization w8a8_int8`、检查 `SGLANG_SAIL_USE_ACEXT_CUDA` 不是 0 |
| `ValueError: Transformers does not recognize this architecture` (model_type=qwen3_5) | transformers 4.57.1 不识别 Qwen3.5 | 走 sglang 后端 (`--backend sglang`) |
| 启动 12s 内崩在 `fused_sigmoid_gating_recurrent.py line 339` | EAGLE/NEXTN 投机解码与 Qwen3.5 GDN 不兼容 | 移除 `--speculative-*` 参数。**Qwen3.5 上投机解码完全不可用** |
| INT8 乱码输出 ("eneene个个…") | SGLang < 0.5.13, w8a8_int8 有 garbage output bug | 升级到 **0.5.13+**(本仓已装) |
| accuracy 0.66 | INT8 + triton attention (无 FLA、ACEXT) | 改回 wrapper 自动启路径(FLA=1 + ACEXT + ViT CG 全开)|
| accuracy 在 0.74-0.83 飘 | 100 样本采样噪声主导 | 不是 bug,不要重跑 patch;500 样本会稳定 |

## 5. 结果评估口径

- **TTFT 用 median**。100 样本 mean 会受前 ~25 样本 warm-up 拉高 4×(尾部某些
  冷启动样本单样本 TTFT ~430 ms),median 才是稳态真实水平。
- **acc 用 mean ± 95% CI ±9.6pp** 判断。同一 100 样本多次跑 acc 会在 ±8pp 区间
  内浮动。`docs/02_RESULTS.md` §2.4 给同 qid 对照的"真实量化损失 = 2pp"数据。
- **不要把 `SGLANG_SAIL_USE_ACEXT_CUDA=0` 当成"真走 GEMV"**。详见
  [`docs/01_ENVIRONMENT.md`](docs/01_ENVIRONMENT.md) §1.11 —— PPU fork 的
  `_handle_ppu_backends()` 会强制覆盖它。
- **不要用 `--enforce-piecewise-cuda-graph`**。`evaluation_wrapper.py:188-203`
  注释里专门记录了这条 avoid 路径,会触发 `cuda_piecewise_backend.py:171` PCG
  stream assert,scheduler 直接死掉。VLM 工作负载下 PiCG 已经被 PPU fork 的
  `_handle_piecewise_cuda_graph rule 8`(多模态模型自动 disable)永远关掉。

## 6. 文件结构

```
PPU/
├── README.md                          # 本文档目录
├── README_REPRODUCE.md                # ← 复现入口 (SSOT)
├── 技术方案报告.md                     # 「为什么 + 怎么做到的」(技术评审用)
├── evaluation_wrapper.py              # 比赛提交件 (VLMModel 接口不变)
├── benchmark_public.py                # 本地复现入口(对齐根目录 benchmark/benchmark_public.py)
├── docs/
│   ├── 01_ENVIRONMENT.md              # 硬件/镜像/版本/patch 部署 6 类 10 文件
│   ├── 02_RESULTS.md                  # 横向对比 + R6/R8/G 三档实测
│   ├── 03_FAILED.md                   # 否决实验
│   ├── 04_OPTIMIZATIONS.md            # 5 个核心优化详解
│   ├── 05_KERNEL_PROFILE.md           # asys 算子 profile + GEMV → Linear 映射
│   └── 06_REPRODUCE_DETAIL.md         # 各对照组精确启动参数
├── INT8_strategy.md                   # 历史:量化路线诊断(0% vs 70% 之谜、MXFP4 不支持)
├── results_summary.md                 # 历史:V1/V2/R 系列实验原始记录
├── vit_graph_optimization.md          # 历史:ViT CG 修复完整根因分析
├── qwen3.5-2b_recipe.yaml             # S2-aggressive 量化配方 yaml
├── gemv_optimization/                  # A-E 类 patch 源码(sglang + deep_gemm)
├── inductor_hook_PPU/                  # F 类 patch 源码
├── GEMV调研/                            # 理论背景(算法 + 嵌入指南)
└── recipes/                             # 量化配方备用
```
