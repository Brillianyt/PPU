# 02. 完整实验结果

> 按方案分类。已过滤 TTFT>100 且 tput<300 的低质量样本。

## 2.1 横向对比表 (代表性结果)

| 方案 | 模型 | attention | 关键开关 | 样本数 | acc | TTFT (ms) | tput (tok/s) |
|---|---|---|---|---|---|---|---|
| **基线 A** (sglang triton) | bf16 | triton | — | 20 | 0.80 | 419.0 | 284.5 |
| **B** | bf16 | fa3 | — | 20 | 0.85 | 313.2 | 279.4 |
| **C** | bf16 | fa3 | +FLA=1 | 20 | 0.85 | 64.3 | 280.1 |
| **D** | bf16 | fa3 | +FLA=1 +no-radix | 20 | 0.85 | 69.0 | 369.4 |
| **⭐ G** (推荐 bf16) | bf16 | fa3 | +FLA=1 +no-radix +page=64 | 20 | 0.85 | 64.7 | 369.0 |
| E | bf16 | fa3 | G + chunked=2048 | 20 | 0.85 | 69.9 | 375.3 |
| **§27 G 组 500 (大样本验证)** | bf16 | fa3 | G 组全套 | **500** | **0.810** | **73.3** | **356.7** |
| §26 INT8 tuned 500 | INT8 | fa3 | +FLA=1 + DeepGEMM autotune | 500 | 0.808 | 78.9 | 358.1 |
| **⭐ S2+ACEXT+pCG (冲吞吐)** | S2-INT8 | fa3 | +ACEXT=1 +pCG | **100** | **0.760** | 78.2 (稳态) | **410.6** |
| S2+ACEXT+pCG (cn) | S2-INT8 | fa3 | +ACEXT=1 +pCG | 100 | 0.820 | 78.8 (稳态) | 361.7 |
| S2+ACEXT+pCG (20) | S2-INT8 | fa3 | +ACEXT=1 +pCG | 20 | 0.850 | 82.3 | 423.2 |

## 2.2 同 500 样本的最终对决 (BF16 G组 vs INT8 tuned)

| 维度 | §26 INT8 + fa3 + FLA=1 + tuned | **§27 G 组 bf16** | 差异 |
|---|---|---|---|
| en acc | 0.808 | **0.810** | +0.2 pp |
| en TTFT (稳态) | 78.9 ms | **73.3 ms** | -7% ⬇️ |
| en tput | 358.1 tok/s | 356.7 tok/s | 持平 |
| cn acc | 0.822 | **0.834** | +1.2 pp |
| cn TTFT (稳态) | 79.5 ms | **75.2 ms** | -5% ⬇️ |
| cn tput | 330.5 tok/s | 331.2 tok/s | 持平 |

**结论 (500 样本实测)**:
- 量化(G组 vs INT8 tuned)准确率几乎持平 (-0.2 ~ -1.2 pp)
- BF16 TTFT 反而略快 5-7%
- 吞吐量持平
- **BF16 是 2B 模型的最优解** (无量化刚需,96 GB 显存装得下 4.3 GB 模型)

## 2.3 INT8 + ACEXT + pCG 100 样本结果

| 数据集 | acc | TTFT 均值 | TTFT 稳态 (p25) | tput 均值 | tput 中位 |
|---|---|---|---|---|---|
| en 100 | 0.760 (76/100) | 197.2 | 78.2 | 410.6 | 397.9 |
| cn 100 | 0.820 (82/100) | 204.0 | 78.8 | 361.7 | 366.7 |

> ⚠️ TTFT 均值被前 ~25 样本 cache warm-up 拉高 (首批 ~340 ms),p25 才是稳态真实水平。

### 真实量化损失评估 (同 100 样本对比)

**S2 100 = G 500 的前 100 个 qid (qid 241-307,完全同一批)**:

| 方案 | 错题数 | acc | 真实量化损失 |
|---|---|---|---|
| G 组 bf16 (同 100) | 22 | 0.780 | — |
| S2 量化 (同 100) | 24 | 0.760 | **+2 样本 = 2pp** |

**错题重叠**:
- 双方都错: **22 个** (模型本身在 bf16 就不会做)
- 仅 G 错 S2 对: **0 个**
- 仅 S2 错 G 对: **2 个** (qid 306: C→A、qid 488: B→A — 这才是真正量化引入的错误)

**结论**:
- **真实量化损失只有 2pp**,不是表面看 0.76 vs 0.81 的 5pp
- "0.81 vs 0.76" 的差距主要来自 **数据偏差**:
  - G 组 500 样本分桶: 桶1 (0.78) / 桶2 (0.76) / 桶3 (0.76) / 桶4 (0.86) / 桶5 (0.89)
  - 整体 0.81 = 5 桶加权平均,**后两桶题目较易**
  - S2 测的恰好是较难的桶1
- 量化引入的 2 个新错都是 **A 选项偏差** (S2 倾向选 A),但只在 2/100 题上显形,不是所有题都偏 A

## 2.4 全部有效 result 文件

```bash
$ ls /mnt/workspace/PPU/result_*.json | xargs -I{} basename {}

result_dev_cn_g_bf16_500.json             # G 组 bf16 cn 500   → acc 0.834, ttft 75.2, tput 331.2
result_dev_cn_int8_cur_20.json            # INT8 + triton cn 20 → acc 0.850, ttft 260.1, tput 331.4
result_dev_cn_int8_fa3_fla1_20.json       # INT8 + fa3 + FLA cn 20 → acc 0.850, ttft 68.8, tput 309.5
result_dev_cn_int8_fa3_fla1_tuned_500.json# INT8 + tuned cn 500 → acc 0.822, ttft 79.5, tput 330.5
result_dev_cn_s2_acext_100.json           # S2+ACEXT cn 100    → acc 0.820, ttft 78.8 (稳态), tput 361.7
result_dev_en_bf16_triton_20.json         # bf16 + triton en 20 → acc 0.800, ttft 64.3, tput 380.9
result_dev_en_g_bf16_500.json             # G 组 bf16 en 500   → acc 0.810, ttft 73.3, tput 356.7
result_dev_en_int8_cur_20.json            # INT8+triton en 20  → acc 0.800, ttft 293.8, tput 362.8
result_dev_en_int8_dg_20.json             # INT8+DeepGEMM en 20 → acc 0.750, ttft 63.8, tput 307.6
result_dev_en_int8_fa3_fla1_100.json      # INT8+fa3+FLA en 100 → acc 0.750, ttft 180.8, tput 350.6
result_dev_en_int8_fa3_fla1_20.json       # INT8+fa3+FLA en 20 → acc 0.750, ttft 103.6, tput 324.2
result_dev_en_int8_fa3_fla1_tuned_500.json# INT8+fa3+FLA+tuned en 500 → acc 0.808, ttft 78.9, tput 358.1
result_dev_en_int8_gemv_20.json           # INT8+GEMV en 20    → acc 0.700, ttft 75.7, tput 353.6
result_dev_en_int8_new_50.json            # INT8 v2 en 50      → acc 0.000 (崩)
result_dev_en_int8_s2_acext_20.json       # S2+ACEXT en 20     → acc 0.850, ttft 86.7, tput 422.9
result_dev_en_int8_s2_acext_v12_20.json   # S2+ACEXT v12 en 20 → acc 0.850, ttft 82.3, tput 423.2
result_dev_en_int8_s2_cg_20.json          # S2+CUDA Graph en 20 → acc 0.850, ttft 84.8, tput 413.0
result_dev_en_int8_s2_gemv_20.json        # S2+GEMV en 20      → acc 0.750, ttft 106.7, tput 388.0
result_dev_en_int8_s2_gemv_cg_20.json     # S2+GEMV+CG en 20   → acc 0.850, ttft 84.2, tput 418.5
result_dev_en_int8_s2_gemv_cg_v9_20.json  # S2+GEMV+CG v9 en 20→ acc 0.850, ttft 84.2, tput 418.5
result_dev_en_int8_s2_pbf16_20.json       # S2+prefill_bf16 en 20 → acc 0.800, ttft 72.1, tput 389.4
result_dev_en_int8_s3full_gemv_20.json    # S3-full+GEMV en 20 → acc 0.650, ttft 104.0, tput 372.3
result_dev_en_int8_s3full_nogemv_20.json  # S3-full en 20      → acc 0.650, ttft 80.4, tput 273.3
result_dev_en_int8_triton_20.json         # INT8+triton en 20  → acc 0.750, ttft 101.6, tput 415.0
result_dev_en_int8_triton_50.json         # INT8+triton en 50  → acc 0.660, ttft 217.3, tput 439.9
result_dev_en_nextn_flaoff_20.json        # NEXTN+FLA off en 20→ acc 0.850, ttft 87.3, tput 275.5
result_dev_en_qwen_origin_G_baseline_v2_20.json # 原始 baseline (debug)
result_dev_en_qwen_origin_baseline_recheck_20.json # 原始 baseline recheck
result_dev_en_s2_acext_100.json           # S2+ACEXT en 100    → acc 0.760, ttft 78.2 (稳态), tput 410.6
result_dev_en_sglang_chunked2048_20.json  # chunked=2048 en 20 → acc 0.850, ttft 69.9, tput 375.3
result_dev_en_sglang_fa1_20.json          # FLA=1 en 20        → acc 0.850, ttft 64.3, tput 280.1
result_dev_en_sglang_fa1_noradix_20.json  # FLA=1+no-radix en 20 → acc 0.850, ttft 69.0, tput 369.4
result_dev_en_sglang_page64_20.json       # +page=64 en 20     → acc 0.850, ttft 64.7, tput 369.0
```

> 6 个低质量样本 (TTFT>100 且 tput<300) 已删除,详见 `03_FAILED.md`。

## 2.5 关键结论

1. **首选方案** (§17 / §27 验证, 500 样本稳定): **G 组 bf16**
   - 启动命令 5 行, 无 source patch, 无量化模型
   - 准确率 0.78-0.81 (不同样本段波动), TTFT 73 ms, tput 357 tok/s

2. **冲吞吐方案** (§29 / §30 / 本次实测, 100 样本对照): **S2-aggressive + ACEXT + Piecewise CUDA Graph**
   - 需 source patch + 量化模型 + ACEXT env
   - 与 G 组在**同一 100 样本**上的真实对比:
     - G 组 acc 0.780, S2 acc 0.760
     - **真实量化损失 = 2/100 = 2pp**(双方共错 22 题,S2 新增 2 题)
   - 吞吐量 +15% (410 vs 357 tok/s)
   - **净 trade-off**: −2pp acc 换 +15% tput

3. **绝对不要做**: 投机解码 (EAGLE/NEXTN) — sglang 0.5.13 在 Qwen3.5 GDN 上直接崩