# PPU 推理优化 — 复现指南

> DNDX 多模态推理调优赛 · Qwen3.5-2B · 1× PPU-ZW810E (96GB) · SGLang 0.5.13

## 0. TL;DR

| 指标 | 推荐方案 | 实测 (100/500 样本 dev) |
|---|---|---|
| **首选 (高准确率,无 patch)** | G 组 bf16 | en acc **0.78** (同 100) / 0.81 (500), TTFT 73ms, **tput 357 tok/s** |
| **冲吞吐 (能接受 2pp acc 损失)** | S2-aggressive + ACEXT + pCG | en acc **0.76** (同 100), TTFT 78ms (稳态), **tput 410 tok/s** |

**真实量化损失 = 2pp**(同一 100 样本对照)。吞吐量 +15%。净 trade-off: −2pp acc 换 +15% tput。

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

### 2.2 冲吞吐: S2-aggressive + ACEXT + Piecewise CUDA Graph

**前置**: 量化模型已生成 (`/mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive`)。

```bash
cd /mnt/workspace/PPU

# 1) 应用 patches (一次性)
cp -rv /mnt/workspace/patches/sglang/* /usr/local/lib/python3.12/site-packages/sglang/
cp -rv /mnt/workspace/patches/deep_gemm/* /usr/local/lib/python3.12/site-packages/deep_gemm/

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

**预期**: en 100 → acc ≈ 0.76-0.80, 稳态 TTFT ~78 ms, throughput ~400 tok/s。

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

| 检查项 | 期望 | 验证方法 |
|---|---|---|
| accuracy (en 100) | ≥ 0.80 (G 组) / ≥ 0.75 (S2) | `result_*.json["accuracy"]["score"]` |
| avg_ttft_ms (稳态) | ≤ 80 ms | 看 p25,不看均值 (均值被 cache warm-up 拉高) |
| throughput tok/s | ≥ 350 (G 组) / ≥ 380 (S2) | `result_*.json["performance"]["avg_throughput_tokens_per_sec"]` |
| 真实 token 数 | `meta.completion_tokens > 0` | 看 `result_*.json["answers"][0]["meta"]` |
| 无样本跳过 | `failed_samples == 0` | `result_*.json["public_validation"]` |
| 接口契约 | `generate_with_metrics` 不变 | 直接看 `evaluation_wrapper.py` |

---

## 4. 故障排查 (按出现频率)

| 症状 | 原因 | 解决 |
|---|---|---|
| TTFT ~340ms 平台 | 没设 `SGLANG_SAIL_FLA_CUDA=1` | 设上,或用 wrapper 自动启 |
| throughput ~280 tok/s | radix cache 开启 | 加 `--disable-radix-cache` |
| accuracy 0.66 | 用 INT8 模型 + triton | 改回 bf16 G 组 / 或加 FLA=1 + ACEXT |
| `ValueError: Transformers does not recognize this architecture` | transformers 4.57.1 不识别 Qwen3.5 | 走 sglang 后端 |
| `retrieve_parent_token.int()` 崩溃 | 启用了 EAGLE/NEXTN 投机解码 | 移除 `--speculative-*`,Qwen3.5-2B 在 0.5.13 上不兼容 |
| `Cannot copy between CPU and CUDA tensors during CUDA graph capture` | `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` 触发 PPU Inductor bug | 关掉,保持 0 |
| INT8 乱码输出 | SGLang < 0.5.13 (w8a8_int8 bug) | 升级到 0.5.13 |
| `File "fused_sigmoid_gating_recurrent.py" line 339` 崩 | 同上,带 FLA=1 触发 | FLA=0 暂时绕开 |

详细诊断见 [`docs/03_FAILED.md`](docs/03_FAILED.md)。

---

## 5. 进阶

- **完整实验记录** (按时间顺序): [`EXPERIMENT.md`](EXPERIMENT.md)
- **算子 profile + GEMV 映射**: [`docs/04_PROFILE.md`](docs/04_PROFILE.md)
- **量化策略 + recipe**: [`docs/05_QUANT.md`](docs/05_QUANT.md)
- **各对照组复现命令**: [`docs/06_REPRODUCE_DETAIL.md`](docs/06_REPRODUCE_DETAIL.md)
- **历史归档** (未删除但不再维护): `.archive/` 目录

---

## 6. 文件结构

```text
PPU/
├── README.md                          # 比赛规则(不动)
├── README_REPRODUCE.md                # ← 你正在看的文件
├── EXPERIMENT.md                      # 完整实验日志 (按时间)
├── evaluation_wrapper.py              # 评测入口(契约文件,接口不变)
├── benchmark_public.py                # 本地调试脚本
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