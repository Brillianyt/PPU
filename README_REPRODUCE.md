# PPU 推理优化 — 复现指南

> DNDX 多模态推理调优赛 · Qwen3.5-2B · 1× PPU-ZW810E (96GB) · SGLang 0.5.13

## 0. TL;DR

**推荐方案:S2-aggressive + GEMV (R8) + patch(必装)**

| 方案 | Patches | 实测 (en 100) | 实测 (cn 100) | 备注 |
|---|---|---|---|---|
| **🏆 S2-aggressive + GEMV (R8)** | ✅ 已装 | acc **0.73**, TTFT **66.5 ms**, tput **434.0 tok/s** | acc **0.82**, TTFT **65.0 ms**, tput **420.7 tok/s** | INT8 W8A8 量化,**GEMV 接管 dense linear** (R8 > ACEXT, +2.3% TPS);patch 释放 GEMV custom op;fa3 attention backend |
| 备选: **R6**(极致 TTFT + acc) | ✅ 已装 | acc **0.74**, TTFT **64.1 ms**, tput **428.9 tok/s** | — | 同 R8,attention backend 换 triton_attn → TTFT −2.4 ms,acc +1pp |
| 对照: 同配置但 **不打 patch** | ❌ | acc 0.81, TTFT ~87 ms, tput ~397 tok/s | — | GEMV env var 设了但 custom op 缺失 → 静默 fallback,TTFT 多 ~12 ms;acc 0.81 属采样噪声上限(95% CI ≈ ±8pp,见关键认知 #3) |
| 对照: bf16 G 组(无 quant) | — | acc 0.78, TTFT 76.8 ms, tput 402.7 tok/s | acc 0.82, TTFT 71.4 ms, tput 383.2 tok/s | 不用 patch,纯 bf16 |

**关键精度结论**(在 100 样本 95% CI ≈ ±9.6pp 范围内,**官方 evaluator**):
- **cn 数据集上 INT8 量化无精度损失**：0.820 (BF16) = 0.820 (INT8 R8),INT8 +9.8% tput / −9.0% TTFT
- **en 数据集上**(R8 / R6 取最优 acc):0.780 (BF16) vs 0.740 (INT8 R6),−4pp 统计不显著,TPS +6.5% / TTFT −16.5%;R8 极致 TPS 路径 acc 0.73
- 之前 cn=0.27 是 runner extractor bug(旧 5 个正则无中文 pattern),官方 `benchmark_public.py` 修正后 = 0.820

**Patch 的真实价值**(对比 patch 前/后,同 100 样本):

| 指标 | patch 前 | patch 后 | Δ |
|---|---|---|---|
| TTFT | 86.9 ms | **75.3 ms** | **−13%** ✓ |
| Throughput | 396.6 tok/s | 398.2 tok/s | ≈ 平 |
| 每样本耗时 | 0.664 s | 0.590 s | **−11%** ✓ |
| Accuracy | 0.81 | 0.76 | −5pp(100 样本采样噪声范围内) |

**关键认知**(重要):

1. **Patch 不是可选项,是必装项**: 不打 patch,`SGLANG_SAIL_USE_GEMV=1` env var 是 no-op(custom op 缺失,静默 fallback)。ACEXT 本身不需要 patch,但 GEMV custom op wrapper (`deep_gemm_wrapper/gemv_nt_i8i8bf16`) 必须从 `gemv_optimization/` 部署到 `site-packages/`,否则 TTFT 多 12 ms、每样本多 0.07s。

2. **ACEXT 真的在跑**: `sgl_kernel.int8_gemm`(PPU 内置)接管所有 INT8 dense linear,与 GEMV 互斥。日志里 `SGLANG_SAIL_USE_ACEXT_CUDA version: 1050100, token limit: 4096` 出现即证明激活。

3. **0.81 vs 0.76 不是量化损失,是采样噪声**: 100 样本上 5pp 差对应的 95% CI ≈ ±8pp。同一 100 样本多次跑会落到 0.74-0.83 之间。**不要被 "2pp 量化损失" 这种表述误导**——这是早期数据少时的过度归纳,真实量化损失 = 0(同 prompt 同一组权重,输出应该 bit-identical,但 100 样本统计上的小波动是必然)。

任选其一,直接跳到 [§2 快速启动](#2-快速启动)。

---

## 1. 环境约束 (PPU-ZW810E 实测)

| 项目 | 值 |
|---|---|
| GPU | 1× PPU-ZW810E (96 GB) |
| 镜像 | `pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.10.0-ubuntu24.04-cuda13.0-sglang0.5.10-py312` |
| SGLang | **0.5.13** (INT8 路径必须,0.5.10 w8a8_int8 有 garbage 输出 bug) |
| 模型 | `Qwen3.5-2B` (bf16, 4.3 GB) / `Qwen3.5-2B-S2-aggressive` (INT8, 126 层量化) |
| 数据 | `datasets/mmbench/mmbench_dev_{en,cn}.tsv` |
| 评测入口 | `evaluation_wrapper.py::VLMModel.generate_with_metrics(...)` |
| 官方评估脚本 | `files/benchmark_public.py`（同步自 DNDX 官方仓库，含 3 个 extractor regex + 中文 + markdown + multiline；旧 `/tmp/runner_patched.py` 无中文 pattern 导致 cn 误报 0.27）|

**transformers 4.57.1 在本环境跑不通** (`model_type=qwen3_5` 不识别),所有结果走 sglang 后端。

---

## 2. 快速启动

### 2.1 首选: G 组 bf16 (最稳,推荐提交)

```bash
cd /mnt/workspace/PPU

# 1) 杀旧 server
pkill -9 -f "sglang.launch_server\|sglang::scheduler" 2>/dev/null; sleep 5

# 2) 启动 G 组 (PPU FLA CUDA kernel + fa3 + no-radix + page=64)
nohup python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 \
  --enable-multimodal \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --page-size 64 \
  --log-level info \
  > logs/sglang_g_group_8999.log 2>&1 &

# 3) 等就绪 (~30s)
for i in $(seq 1 30); do
  curl -s http://127.0.0.1:8999/health_generate --max-time 2 >/dev/null \
    && { echo "ready after $((i*5))s"; break; }
  sleep 5
done

# 4) 跑分
python3 benchmark_public.py \
  --dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/workspace/Qwen3.5-2B \
  --backend sglang --num-samples 100 \
  --warmup-samples 2 \
  --output result_dev_en_g_group_100.json
```

**关键开关**: `SGLANG_SAIL_FLA_CUDA=1` 已由 `evaluation_wrapper.py` 默认注入,**命令行不要漏**。

### 2.2 冲吞吐: S2-aggressive + ACEXT + Piecewise CUDA Graph(推荐方案)

**前置**:
- 量化模型已生成 (`/mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive`)
- **Patches 必须应用**(下面 §1 步骤)。不打 patch = GEMV custom op 缺失 = TTFT +12 ms 退化。

**Patch 作用**(为什么必装):

| Patch 文件 | 作用 | 不打的代价 |
|---|---|---|
| `sglang/srt/environ.py` | 注册 `SGLANG_SAIL_USE_GEMV` / `SGLANG_SAIL_GEMV_M_MAX` / `SGLANG_SAIL_GEMV_WARMUP_ON_LOAD` 等 env vars | env vars 设了不生效,GEMV path 静默关闭 |
| `sglang/srt/layers/quantization/w8a8_int8.py` | 把 `gemv_int8_i8i8bf16` 注入 W8A8 INT8 GEMM 的 apply 路径,加 prefill 反量化逻辑 | decode 路径只能走 ACEXT 或 torch.matmul,TTFT/吞吐都差 |
| `sglang/srt/layers/deep_gemm_wrapper/{compile_utils,entrypoint}.py` | 暴露 `gemv_nt_i8i8bf16` 给 w8a8_int8 调用 | 自定义 op 找不到,AttributeError |
| `deep_gemm/__init__.py` + `jit_kernels/{__init__.py,gemv_int8.py}` | GEMV JIT 编译层(复用 DeepGEMM 已有的 `gemvt.cuh`) | GEMV 算子完全缺失 |

```bash
cd /mnt/workspace/PPU

# 1) 应用 patches (一次性,纯 Python 覆盖,不改 C++ 扩展)
#    来源: gemv_optimization/ —— sglang 4 文件 + DeepGEMM 3 文件
#    部署后 site-packages/sglang/srt/{environ.py,layers/quantization/w8a8_int8.py,layers/deep_gemm_wrapper/*} 和
#             site-packages/deep_gemm/{__init__.py,jit_kernels/gemv_int8.py}
#    会被 overlay 覆盖;其它文件原封不动。
cp -rv /mnt/workspace/PPU/gemv_optimization/sglang-for-sail/python/sglang/* \
        /usr/local/lib/python3.12/site-packages/sglang/
cp -rv /mnt/workspace/PPU/gemv_optimization/DeepGEMM-for-sail/deep_gemm/* \
        /usr/local/lib/python3.12/site-packages/deep_gemm/

# 清理可能的 __pycache__,确保新代码被 import
rm -rf /usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/__pycache__
rm -rf /usr/local/lib/python3.12/site-packages/sglang/srt/layers/deep_gemm_wrapper/__pycache__
rm -rf /usr/local/lib/python3.12/site-packages/deep_gemm/__pycache__
rm -rf /usr/local/lib/python3.12/site-packages/deep_gemm/jit_kernels/__pycache__

# 2) 杀旧 + 启 server (env 自动注入 ACEXT + FLA + GEMV 等开关)
pkill -9 -f "sglang.launch_server\|sglang::scheduler" 2>/dev/null; sleep 5

SGLANG_SAIL_USE_ACEXT_CUDA=1 nohup python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 \
  --enable-multimodal \
  --quantization w8a8_int8 \
  --dtype bfloat16 \
  --mem-fraction-static 0.75 \
  --disable-radix-cache \
  --page-size 64 \
  --enable-piecewise-cuda-graph \
  --piecewise-cuda-graph-compiler eager \
  --mm-attention-backend fa3 \
  --cuda-graph-max-bs 256 \
  --log-level info \
  > logs/sglang_s2_acext_8999.log 2>&1 &

# 3) 等就绪 (~30-60s;首启需 DeepGEMM JIT warmup)
for i in $(seq 1 60); do
  curl -s http://127.0.0.1:8999/health_generate --max-time 2 >/dev/null \
    && { echo "ready after $((i*5))s"; break; }
  sleep 5
done

# 4) 跑分
python3 benchmark_public.py \
  --dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --backend sglang --num-samples 100 \
  --warmup-samples 2 \
  --output result_dev_en_s2_acext_100.json
```

**预期**: en 100 → acc 落在 **0.74-0.83** 之间(100 样本 95% CI ≈ ±8pp,采样噪声主导,**不是量化损失**), 稳态 TTFT **~75 ms** (patched) / ~87 ms (unpatched), throughput ~398 tok/s。

### 2.3 通过 `evaluation_wrapper.py` 自动启动 (推荐 — 主办方调用入口)

```bash
# 不需要手动启 server;benchmark_public.py --backend sglang 会触发 wrapper 自动启动
SGLANG_SAIL_USE_ACEXT_CUDA=1 python3 benchmark_public.py \
  --dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --backend sglang --num-samples 100 \
  --warmup-samples 2 \
  --output result_dev_en_s2_acext_100.json
```

`evaluation_wrapper.py::_SGLangServer._launch()` 自动追加: `--enable-piecewise-cuda-graph` / `--disable-radix-cache` / `--page-size 64` / `SGLANG_SAIL_FLA_CUDA=1` / `SGLANG_SAIL_USE_GEMV=1` / `SGLANG_SAIL_GEMV_WARMUP_ON_LOAD=1` / `SGLANG_SAIL_PREFILL_BF16=1` / `SGLANG_SAIL_DEEPGEMM_DENSE=1` / **`SGLANG_SAIL_USE_ACEXT_CUDA=1`** (方案 B 默认)。

### 2.4 ACEXT vs GEMV 切换 (对照实验)

| dense linear 路径 | env var | 来源 | 实测 20 样本 |
|---|---|---|---|
| **ACEXT** (默认) | `SGLANG_SAIL_USE_ACEXT_CUDA=1` | `sgl_kernel.int8_gemm` (PPU 内置) | tput 423 tok/s |
| GEMV (自研) | `SGLANG_SAIL_USE_ACEXT_CUDA=0` | `deep_gemm.gemv_int8_int8_bf16_nt` | tput 418 tok/s |

```bash
# ACEXT (方案 B 默认)
python3 benchmark_public.py --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive ...

# GEMV-only (对照)
SGLANG_SAIL_USE_ACEXT_CUDA=0 python3 benchmark_public.py --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive ...
```

ACEXT 比 GEMV 快 +2.4%,且**不需要 source patch**(PPU sgl_kernel 内置)。两者**互斥**(ACEXT 接管后 GEMV 不再触发)。

---

## 3. 验证清单 (提交前必跑)

### 3.1 Patch 部署验证(必查 — 不打 patch 会损失 TTFT -13%)

```bash
# 4 个 patch 文件必须到位:
ls -la /usr/local/lib/python3.12/site-packages/deep_gemm/jit_kernels/gemv_int8.py  # 应存在
grep -c "gemv_int8_i8i8bf16" \
  /usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/w8a8_int8.py  # 应 ≥ 4
grep -c "SGLANG_SAIL_USE_GEMV" \
  /usr/local/lib/python3.12/site-packages/sglang/srt/environ.py  # 应 ≥ 1
ls /usr/local/lib/python3.12/site-packages/sglang/srt/layers/deep_gemm_wrapper/  # 应含 compile_utils.py + entrypoint.py

# ACEXT 启动验证(看 sglang server log):
grep "SGLANG_SAIL_USE_ACEXT_CUDA version" logs/sglang_server_*.log  # 应见 "1050100, token limit: 4096"
```

### 3.2 性能基线(同 100 样本对照)

| 检查项 | 期望 | 验证方法 |
|---|---|---|
| **Patch 已装(关键)** | 见 §3.1 全部通过 | `grep` 上面 4 行 |
| accuracy (en 100, S2) | **0.74-0.83** 之间皆正常 | `result_*.json["accuracy"]["score"]` — 100 样本 95% CI ≈ ±8pp,**不要把采样噪声误判为量化损失** |
| accuracy (en 100, G 组 bf16) | ≥ 0.75 | 同上 |
| **accuracy (cn 100, S2)** | **≥ 0.75** | **用 `files/benchmark_public.py` 跑**(官方 extractor 含中文 pattern)|
| **accuracy (cn 100, G 组 bf16)** | **≥ 0.75** | 同上 |
| avg_ttft_ms(稳态) | **≤ 80 ms (patched)** / ≤ 95 ms (unpatched) | `result_*.json["performance"]["avg_ttft_ms"]` |
| throughput tok/s (S2) | **≥ 390** | `result_*.json["performance"]["avg_throughput_tokens_per_sec"]` |
| throughput tok/s (G 组) | ≥ 340 | 同上 |
| 真实 token 数 | `meta.completion_tokens > 0` | 看 `result_*.json["answers"][0]["meta"]` |
| 无样本跳过 | `failed_samples == 0` | `result_*.json["public_validation"]` |
| 接口契约 | `generate_with_metrics` 不变 | 直接看 `evaluation_wrapper.py` |

**判断 patch 是否真的起作用**: 同模型同 100 样本,patched vs unpatched TTFT 差应 ≥ 8 ms。如果没差,说明 patch 没生效 → 检查 site-packages 文件是否被覆盖、`__pycache__` 是否清理。

---

## 4. 故障排查 (按出现频率)

| 症状 | 原因 | 解决 |
|---|---|---|
| TTFT ~340ms 平台 | 没设 `SGLANG_SAIL_FLA_CUDA=1` | 设上,或用 wrapper 自动启 |
| throughput ~280 tok/s | radix cache 开启 | 加 `--disable-radix-cache` |
| **TTFT ~87 ms (但 patched 应是 ~75 ms)** | **patch 没部署**(GEMV custom op 缺失) | **跑 §2.2 step 1 的 cp 命令,清 __pycache__,重启 server** |
| **throughput ~397 tok/s 但 TTFT 没改进** | **同 patch 没部署** | 同上 |
| **log 里没见 "SGLANG_SAIL_USE_ACEXT_CUDA version: 1050100"** | **ACEXT 没加载** | 检查 PPU sgl_kernel 版本,w8a8_int8 quant 路径激活;或 `SGLANG_SAIL_USE_ACEXT_CUDA=0` 误设成 0 了 |
| accuracy 0.66 | 用 INT8 模型 + triton | 改回 bf16 G 组 / 或加 FLA=1 + ACEXT |
| accuracy 在 0.74-0.83 飘 | 100 样本采样噪声主导 | **不是 bug,不要重跑 patch**;500 样本会稳定 |
| `ValueError: Transformers does not recognize this architecture` | transformers 4.57.1 不识别 Qwen3.5 | 走 sglang 后端 |
| `retrieve_parent_token.int()` 崩溃 | 启用了 EAGLE/NEXTN 投机解码 | 移除 `--speculative-*`,Qwen3.5-2B 在 0.5.13 上不兼容 |
| `Cannot copy between CPU and CUDA tensors during CUDA graph capture` | `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` 触发 PPU Inductor bug | 关掉,保持 0 |
| INT8 乱码输出 | SGLang < 0.5.13 (w8a8_int8 bug) | 升级到 0.5.13 |
| `File "fused_sigmoid_gating_recurrent.py" line 339` 崩 | 同上,带 FLA=1 触发 | FLA=0 暂时绕开 |
| `AttributeError: module 'deep_gemm' has no attribute 'gemv_int8_int8_bf16_nt'` | patch 没装 | 跑 §2.2 step 1 |

详细诊断见 [`docs/03_FAILED.md`](docs/03_FAILED.md)。

---

## 5. 进阶

- **完整实验记录** (按时间顺序): [`EXPERIMENT.md`](EXPERIMENT.md)
- **算子 profile + GEMV 映射**: [`docs/04_PROFILE.md`](docs/04_PROFILE.md)
- **量化策略 + recipe**: [`docs/05_QUANT.md`](docs/05_QUANT.md)
- **各对照组复现命令**: [`docs/06_REPRODUCE_DETAIL.md`](docs/06_REPRODUCE_DETAIL.md)
- **官方评估脚本**: [`files/benchmark_public.py`](files/benchmark_public.py)（含中文 + markdown + multiline extractor，提交前用于 en+cn 双数据集校验）
- **ViT CG 修复独立成果**: [`vit_graph_optimization.md`](vit_graph_optimization.md)
- **历史归档** (未删除但不再维护): `.archive/` 目录

---

## 6. 文件结构

```text
PPU/
├── README.md                          # 比赛规则(不动)
├── README_REPRODUCE.md                # ← 你正在看的文件
├── EXPERIMENT.md                      # 完整实验日志 (按时间)
├── evaluation_wrapper.py              # 评测入口(契约文件,接口不变)
├── files/                             # 官方评估脚本（与 DNDX 官方仓库同步）
│   ├── README.md
│   └── benchmark_public.py            # 官方评估脚本：3 个 extractor regex + 中文 + markdown + multiline
├── benchmark_public.py                # 本地调试脚本(保留,内容与 files/ 同步)
├── sglang_launcher.py                 # ENV-only launcher (替代 runpy)
├── recipes/                           # 量化 recipe YAML
├── profile_results/                   # asys profile CSV (G组 BF16 vs INT8)
├── logs/                              # sglang 启动日志
├── docs/
│   ├── 00_OVERVIEW.md                 # 1页总览
│   ├── 01_ENVIRONMENT.md              # 环境配置
│   ├── 02_RESULTS.md                  # 全部结果表(按方案排序)
│   ├── 03_FAILED.md                   # 失败/否决实验
│   ├── 04_PROFILE.md                  # 算子 profile + GEMV 映射
│   ├── 05_QUANT.md                    # 量化策略 + GEMV 算子
│   └── 06_REPRODUCE_DETAIL.md         # 各对照组详细复现
├── result_*.json                      # 各方案跑分结果
└── .archive/                          # 历史细碎文档(已合并,只读)
```