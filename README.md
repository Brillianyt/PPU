# PPU 优化方案 · DNDX 多模态推理调优赛

Qwen3.5-2B (混合 full attention + linear attention / GDN, 2B 参数 VLM) 在国产 PPU-ZW810E
(真武 810E, 96 GB, T-Head CUDA 兼容层) 上的推理优化。提交件 = `evaluation_wrapper.py`
单文件;评测入口约定 = [`benchmark/README.md`](../benchmark/README.md) §3 接口契约。

## 推荐方案 (R6 — 极致 TTFT+acc)

| 指标 | 数值(en 100 样本, warmup 8, official evaluator) |
|---|---|
| **准确率** | **0.74** (同 baseline 0.78, −4pp / 95% CI ±9.6pp 统计不显著) |
| **TTFT median** | **63 ms** (R6 / R7 标称 64–66 ms) |
| **Throughput** | **410 tok/s** (+15% vs BF16 baseline) |
| **CN 准确率** | **0.82** (= BF16 baseline, **INT8 量化零损失**) |

### 增量: ISA v10 prebuilt GEMV kernel (2026-08-23)

**v10 状态: opt-in(env `SGLANG_ISA_V10_ENABLE=1` 启用),不修改默认配置**。

| 指标 | 当前 baseline(无 v10,2026-08-23 实测) | +v10(同配置) | 改善率 |
|---|---|---|---|
| **准确率** | **0.73** | **0.73** | 0 |
| **TTFT (ms)** | **72.7** | **75.3** | **-3.6%** |
| **Throughput (tok/s)** | **411.6** | **421.1** | **+2.3%** |

**官方评分公式 (40% acc + 30% TTFT + 30% TPS) 净贡献:**
```
0.4 × 0 + 0.3 × (-0.036) + 0.3 × 0.023 = -0.004 ≈ ±0
```
**v10 净评分几乎为零** — TTFT 回归(可能 first-request 冷启动噪音)刚好抵消 TPS 收益。

**判定**: 默认配置不挂 v10。提交件 = 当前 baseline (acc 0.73, TTFT 72.7, TPS 411.6)。
v10 kernel + 完整集成作为探索留档,允许用户 opt-in(若私测集重 TPS 或 TTFT 噪音随机分散)。
Phase 3.2.1 cp.async 双 buffer 未跑赢 v10(microbench +9%),留作 future work;
下一目标 = Phase 3.3 persistent kernel(预期 +20-60%)。详见
[`docs/02_RESULTS.md`](docs/02_RESULTS.md) §2.7 + [`docs/04_OPTIMIZATIONS.md`](docs/04_OPTIMIZATIONS.md) §4.6 + §4.6.6。

> **关于历史 R6/R8 baseline**: `推荐方案` 表中 R6/R8 数字 (TTFT 63ms, TPS 429/434)
> 来自 2026-08-21 的实测,因 wrapper 默认 backend 与 env 行为漂移,**目前不可复现**。
> 本次 2026-08-23 实测 = 上述"当前 baseline"列,作为唯一可信 SSOT。R6/R8 历史数字
> 仅作上下文参考,不参与本次 v10 决策。

详细对比 + 启动命令 → [`README_REPRODUCE.md`](README_REPRODUCE.md)。

## 文件结构

| 文件 | 用途 |
|---|---|
| `README_REPRODUCE.md` | **复现入口** (TL;DR + 启动命令 + 验证清单 SSOT) |
| `evaluation_wrapper.py` | **比赛提交件** — `VLMModel` 接口契约,3 个 backend,dummy/transformers/sglang |
| `docs/01_ENVIRONMENT.md` | 硬件 / 镜像 / 量化模型 / patch 部署约定 / PPU fork 行为 caveat |
| `docs/02_RESULTS.md` | 横向对比 + R6/R8/G 三档实测数字 + 精度分解 |
| `docs/03_FAILED.md` | 否决实验 (EAGLE、INT8 v0.5.10 garbage、S3-full、MXFP4 等) |
| `docs/04_OPTIMIZATIONS.md` | 5 个核心优化详解 (AMQ S2 / GEMV / FLA / ViT CG / Inductor RoPE hook) |
| `docs/05_KERNEL_PROFILE.md` | asys 算子 profile + GEMV→Linear 映射 |
| `docs/06_REPRODUCE_DETAIL.md` | 各对照组精确启动参数 |
| `技术方案报告.md` | 「为什么 + 怎么做到的」叙事文档 (供主办方技术评审) |
| `INT8_strategy.md` · `results_summary.md` · `vit_graph_optimization.md` | 历史实验完整诊断 (指针参考,非主路径) |
| `qwen3.5-2b_recipe.yaml` · `GEMV调研/` · `gemv_optimization/` · `inductor_hook_PPU/` | 资产层 (recipe / 算子源码 / hook 源码) |
