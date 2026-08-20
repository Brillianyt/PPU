# 00. 总览

> DNDX 多模态推理调优赛 · Qwen3.5-2B · PPU-ZW810E

## 探索路径 (一段话总结)

```
sglang triton baseline (TTFT 419ms, tput 284 tok/s, acc 0.80)
  └─→ + fa3 attention          →  TTFT 313ms, acc 0.85
       └─→ + FLA_CUDA=1 (PPU 原生 GDN kernel)  →  TTFT 64ms (突破!)
            └─→ + --disable-radix-cache        →  tput 369 tok/s
                 └─→ + --page-size 64           →  TTFT 64.7ms ← G 组 (历史最优 bf16)

INT8 W8A8 量化 (单独路径):
  └─→ + DeepGEMM (sglang < 0.5.13 garbage)  →  acc 0.66, 否决
       └─→ + 0.5.13 修复 + FLA + ACEXT + Piecewise CUDA Graph  →  tput 410 tok/s (但 acc 0.76)
```

## 推荐配置 (二选一)

| 优先级 | 方案 | 预期指标 (en 100) |
|---|---|---|
| **首选** | G 组 bf16 (简单,无 patch) | acc 0.81 / TTFT 73ms / tput 357 tok/s |
| 冲吞吐 | S2-aggressive + ACEXT + pCG | acc 0.76 / TTFT 78ms 稳态 / tput 410 tok/s |

详见 [`../README_REPRODUCE.md`](../README_REPRODUCE.md) §2 快速启动。

## 文档地图

| 文件 | 内容 |
|---|---|
| [`../README_REPRODUCE.md`](../README_REPRODUCE.md) | 5 分钟复现入口 (TL;DR + 启动命令) |
| `01_ENVIRONMENT.md` | 镜像/硬件/版本/模型/数据集 |
| `02_RESULTS.md` | 全部实验结果 (按方案排序的对比表) |
| `03_FAILED.md` | 否决/失败的实验 (投机解码、量化错误等) |
| `04_PROFILE.md` | asys 算子 profile + GEMV → Linear 映射 |
| `05_QUANT.md` | S2/S3 量化策略 + GEMV decode 算子细节 |
| `06_REPRODUCE_DETAIL.md` | 各对照组详细复现命令 |
| [`../EXPERIMENT.md`](../EXPERIMENT.md) | 完整实验日志 (按时间顺序,原始 1838 行) |
| [`../.archive/`](../.archive/) | 历史细碎文档 (只读,不再维护) |