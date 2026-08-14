# DNDX v1.2 完整工作日志

> 目的：在工作仓库不持久的情况下，把"如何认识环境 → 如何实验 → 最终推荐"的每一步固化下来，便于 git clone 后立即复现。

## 目录结构

| 文件 | 内容 |
|---|---|
| `docs/00_OVERVIEW.md` | 本文件 — 总体路径 |
| `docs/01_ENVIRONMENT.md` | 镜像、硬件、版本、模型、数据集 |
| `docs/02_DISCOVERY.md` | 启动 → baseline → 各 flag 探索过程（按时间顺序） |
| `docs/03_KEY_FINDINGS.md` | 关键技术发现（FLA_CUDA、radix、page-size、deepgemm） |
| `docs/04_PROFILE.md` | asys 算子 profile + GEMV → 组件定位 |
| `docs/05_FAILED_EXPERIMENTS.md` | EAGLE/NEXTN/INT8/DALI/长 timeout 等失败/否决项 |
| `docs/06_FINAL_CONFIG.md` | 最终推荐配置 + 一键启动脚本 |
| `docs/07_PROFILE_TO_KERNEL_MAP.md` | 每个 GEMV 算子对应到模型 Linear 层 |
| `docs/08_REPRODUCE.md` | 复现所有实验的命令清单 |

## 整体路径

```
1. 环境认识：1× PPU-ZW810E (96GB), sglang 0.5.10+ppu2.1.1, Qwen3.5-2B bf16, mmbench_dev_en
2. Baseline：transformers 不可用（版本太老不认识 qwen3_5），强制走 sglang 后端
3. sglang baseline (triton attention)：TTFT 419ms, throughput 284 tok/s, accuracy 0.80
4. 探索 1：fa3 attention → TTFT -25% (313ms), accuracy +5pp
5. 探索 2：+ SGLANG_SAIL_FLA_CUDA=1 → TTFT -85% (64ms!) ← 重大突破
6. 探索 3：+ --disable-radix-cache → throughput +30% (369 tok/s)
7. 探索 4：+ --page-size 64 → TTFT 再 -6%, throughput 持平 ← 最终配置 G 组
8. 失败：EAGLE/NEXTN 投机解码 → hybrid GDN 模型崩溃（parent_token None）
9. 失败：INT8 W8A8 量化 → accuracy -19pp, 不可用
10. 失败：DALI 图像预处理 → HF Qwen2VLImageProcessorFast 已在 GPU，0.72ms/张无空间
11. 失败：长 timeout（watchdog-timeout/dist-timeout 3600）→ 违反规则
12. 算子 profile：asys 60s trace → GEMV 73.4%, Native 21.3%, FA 4.9%, Triton 0.4%
13. GEMV 定位：grid 拆分 → 5 个 Linear 类别被定位
14. 结论：G 组是这台机器 + 这个 sglang + bf16 模型下的 SOTA
```

## 关键产出文件

- `evaluation_wrapper.py` — 唯一提交物，sglang 后端，默认 G 组配置
- `EXPERIMENT.md` — 完整实验日志（章节 1-18 + 新增 19-23）
- `docs/` — 详细的工作日志
- `profile_results/*.csv` — 算子 profile 数据（不含大 .asysrep）
- `result_dev_en_*.json` — 各次 benchmark 结果