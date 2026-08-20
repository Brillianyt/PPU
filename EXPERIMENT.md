# 实验记录 (按时间顺序)

> **本文件是历史日志**,不重复 README/docs 中的总结。
> - 推荐配置: [`README_REPRODUCE.md`](README_REPRODUCE.md) §2
> - 结果汇总: [`docs/02_RESULTS.md`](docs/02_RESULTS.md)
> - 失败实验: [`docs/03_FAILED.md`](docs/03_FAILED.md)

---

## 时间线 (2026-08-13 → 2026-08-20)

| 阶段 | 关键动作 | 结果 |
|---|---|---|
| **§1-3** | 启动 sglang, baseline | triton baseline: acc 0.80, TTFT 419 ms, tput 284 tok/s |
| **§4-7** | attention → fa3 → FLA_CUDA=1 | acc 0.85, TTFT **64 ms (-85%)**, tput 280 tok/s |
| **§8-9** | + no-radix-cache | tput **369 tok/s (+32%)**, TTFT +5 ms |
| **§10** | G 组最终 (加 page=64) | TTFT 64.7 ms, tput 369 tok/s |
| **§11-12** | + chunked-prefill=2048 | 与 G 等价 (单样本 ~1500 token 不切) |
| **§13-15** | EAGLE/NEXTN 投机解码 | **崩溃** (GDN parent_token None) |
| **§16** | NEXTN + FLA=off 绕开 | 能跑但 TTFT +35%, tput -25% |
| **§17** | G 组 500 样本验证 | acc 0.810, TTFT 73.3 ms, tput 357 tok/s |
| **§18-23** | INT8 W8A8 (sglang 0.5.10) | **acc 0.66 garbage**,否决 |
| **§24-27** | 升级 sglang 0.5.13, INT8 + DeepGEMM autotune | 500 样本 acc 0.808, TTFT 79 ms, tput 358 tok/s |
| **§28** | BF16 GEMV vs INT8 DeepGEMM M=1 microbench | INT8 pad M=16, 浪费 94%, 端到端打平 |
| **§29** | INT8 + Triton GEMV patch | microbench -2.5×, 端到端持平 |
| **§30** | 直接修改 sglang 源码注入 Triton GEMV | CUDA Graph 抓 bs=256,patch 不触发,回退 |
| **§31-35** | AMQ 量化策略 v3/S2/S3, GEMV + prefill_BF16 | S2+GEMV+PBF16: 20 样本 acc 0.80, TTFT 72 ms, tput 389 tok/s |
| **§36** | S2 + GEMV + Piecewise CUDA Graph | 20 样本 acc 0.85, TTFT 86 ms, tput **418 tok/s** |
| **§37** | + ACEXT (替换 GEMV) | 20 样本 acc 0.85, TTFT 82 ms, tput **423 tok/s** |
| **§38** (本次) | 100 样本 en/cn 大样本验证 | en acc 0.76 (A 偏差), cn acc 0.82, tput 410/362 tok/s |

---

## 关键节点回溯

### §4-7: FLA_CUDA 突破

- 动机: fa3 不支持 GDN linear-attention,默认走 triton community kernel,TTFT ~340 ms 平台期
- 文档说 `SGLANG_SAIL_FLA_CUDA` 默认 True,**源码 `EnvBool(False)`**
- 在 `evaluation_wrapper.py._launch()` 强制设 `env["SGLANG_SAIL_FLA_CUDA"] = "1"`
- 启动日志确认:
  ```
  USE PPU SAIL CUDA FLA kernel: fused_sigmoid_gating_delta_rule_forward_k_last_packed
  ```
- **效果**: TTFT 平台消失,稳态 64 ms

### §8-9: radix cache 关闭

- mmbench 样本唯一 + prompt 模板相似 → radix cache 收益 ≈ 0
- **意外联动**: `disable_radix_cache=True` → sglang 自动设 `disable_overlap_schedule=False`,打开 prefill/decode 重叠调度
- **效果**: tput +32% (280 → 369)

### §17: G 组 (历史最优 bf16)

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/Qwen3.5-2B \
  --attention-backend fa3 --enable-multimodal \
  --dtype bfloat16 --mem-fraction-static 0.85 \
  --disable-radix-cache --page-size 64
```

500 样本实测: **acc 0.810, TTFT 73.3 ms, tput 357 tok/s**

### §24-27: INT8 500 样本验证

| 路径 | acc | TTFT | tput |
|---|---|---|---|
| §24 INT8+triton (无 FLA) | 0.800 | 293.8 ms | 363 tok/s |
| §25 INT8+fa3+FLA (tuner 同跑) | 0.750 | 103.7 ms | 324 tok/s |
| §26 INT8+fa3+FLA+**tuned** | 0.808 | **78.9 ms** | 358 tok/s |
| **§27 G 组 bf16** | **0.810** | **73.3 ms** | 357 tok/s |

**结论**: 量化在 2B 模型上几乎无收益 (acc -0.2pp, TTFT 反而 +5-7ms, tput 持平)

### §30: Triton GEMV 注入失败

详细见 [`docs/03_FAILED.md`](docs/03_FAILED.md) F7。

核心: sglang 抓 CUDA Graph 用 bs=256,`x.shape[0]` 恒等于 256。patch 在 capture 阶段只 fire 2 次后永不触发。

### §36-37: S2 + GEMV + ACEXT 突破

```bash
SGLANG_SAIL_USE_ACEXT_CUDA=1 python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --attention-backend fa3 --enable-multimodal \
  --quantization w8a8_int8 \
  --dtype bfloat16 --mem-fraction-static 0.75 \
  --page-size 64 --disable-radix-cache \
  --enable-piecewise-cuda-graph \
  --piecewise-cuda-graph-compiler eager \
  --mm-attention-backend fa3 \
  --cuda-graph-max-bs 256
```

20 样本实测: **acc 0.85, TTFT 82 ms, tput 423 tok/s**

### §38 (本次): 100 样本验证

详见 [`docs/02_RESULTS.md`](docs/02_RESULTS.md) §2.3。

**核心发现 (重要修正)**:
- S2 100 = G 500 的前 100 个 qid (qid 241-307,完全同一批)
- **同一 100 样本上的真实对比**: G 组 acc 0.780, S2 acc 0.760,**真实量化损失只有 2pp**
- 双方都错 22 题,S2 仅新增 2 题(qid 306: C→A,qid 488: B→A)
- 之前"0.81 vs 0.76 = 5pp 损失"的判断**不严谨**:0.81 是 500 样本里后 400 题加权,题目相对简单
- **净 trade-off**: −2pp acc 换 +15% tput (410 vs 357 tok/s)

---

## 配置文件历史

| 时间 | 文件 | 关键修改 |
|---|---|---|
| 2026-08-13 | `evaluation_wrapper.py` | 新增 sglang backend |
| 2026-08-13 | `evaluation_wrapper.py` | `_launch()` 默认 G 组三件套 (fa3 + FLA + no-radix) |
| 2026-08-14 | `evaluation_wrapper.py` | G 组加 `--page-size 64` |
| 2026-08-15 | `evaluation_wrapper.py` | INT8 + `--quantization w8a8_int8` 默认 |
| 2026-08-16 | `evaluation_wrapper.py` | + piecewise CUDA Graph + GEMV env vars |
| 2026-08-19 | `evaluation_wrapper.py` | + ACEXT env var (默认关) |
| 2026-08-20 | `evaluation_wrapper.py` | 移除 `--enable-torch-compile` (Inductor PPU bug) |

## 历史 result 文件清理

2026-08-20 清理了 6 个低质量样本 (TTFT>100 且 tput<300):

```
removed: result_dev_en_sglang_20.json          (triton baseline, 已淘汰)
removed: result_dev_en_sglang_fa3_20.json      (fa3 无 FLA, 已淘汰)
removed: result_dev_en_sglang_smoke.json       (冒烟测试)
removed: result_dev_en_int8_fa3_fla1_500.json  (tuner 抢占 GPU, 数据失真)
removed: result_dev_cn_int8_fa3_fla1_500.json  (同上)
removed: result_dev_en_int8_s2_mtp_20.json     (S2+MTP 失败)
```

详见 [`docs/03_FAILED.md`](docs/03_FAILED.md) F8。

---

## 历史归档

完整 1838 行原始日志备份在 `git log` + `.archive/` 目录。需要回溯请翻 `.archive/02_DISCOVERY.md` (探索时间线) 或 `.archive/05_FAILED_EXPERIMENTS.md` (失败记录原文)。