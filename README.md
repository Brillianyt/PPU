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

> **当前提交件基线 (2026-08-23 实测)**: acc **0.73**, TTFT median **72.7 ms**, TPS **411.6**。
> 上表 R6 数字 (TTFT 63ms, TPS 429/434) 来自 2026-08-21 的实测,因 wrapper 默认
> backend 与 env 行为漂移,**目前不可复现**,仅作上下文参考。以 2026-08-23 实测
> 为唯一可信 SSOT。

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
