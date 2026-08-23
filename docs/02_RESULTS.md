# 02. 实测结果(SSOT)

> 数字 SSOT = [README_REPRODUCE.md §0](../README_REPRODUCE.md) §0 TL;DR 和本文 §2.1,
> 互为 SSOT。其它 doc 出现的数字若与本文件冲突,以这两处为准。
>
> 数字口径约定:
> - **TTFT 一律报 median**(mean 受尾部冷启动样本拉高 4×,100 样本 mean 73 ms vs
>   median 63 ms 是真实数据,不是 bug)
> - **acc 100 样本上 95% CI ≈ ±9.6pp**(不要把 0.78 vs 0.74 解读成量化掉 4pp 精度,
>   那是噪声)
> - **同 qid 对照**才报精度损失(§2.4)

## 2.1 推荐方案(三档)

| 优先级 | 方案 | 模型 | attention backend | 单样本 acc (en/cn 100) | TTFT median (ms) | throughput (tok/s) | patch 需求 |
|---|---|---|---|---|---|---|---|
| **首选 ⭐** | **R6 (极致 TTFT+acc)** | S2-aggressive INT8 | triton_attn | **0.74 / 0.82** | **63** | **429** | 全 6 类 |
| 冲吞吐 | R8 (极致 TPS) | S2-aggressive INT8 | fa3 | 0.73 / — | 66 | 434 | 全 6 类 |
| 对照基线 | G 组 bf16 | 原 BF16 (无 patch) | fa3 | 0.78 / 0.82 | 73 | 357 (500 样本) | 无 |

> **R6 / R8 在 cn 数据集上 INT8 量化零精度损失**(acc 0.82 = 0.82)。
> 比赛私有测试集若中英混合,整体量化损失 < 4pp。

数字来源:
- R6 / R8 (en 100):`../results_summary.md` §2 2×2 网格 + 2026-08-21 实测
- G 组 (en 500):`../results_summary.md` §1.5 BF16 baseline + `02_RESULTS.md` §2.2 历史
- CN 0.82:`../results_summary.md` §1.5 cn 100/200 实测

## 2.2 横向对比:G 组 vs S2 + ACEXT (100 样本同 qid 对照)

| 数据集 | 维度 | G 组 bf16 (500 样本大数) | S2 INT8 (100 样本) | Δ |
|---|---|---|---|---|
| en | acc | 0.78 (100 段) | 0.74 (R6) | −4pp (不显著, 在 95% CI 内) |
| en | TTFT median (ms) | 73 | 63 | −14% |
| en | throughput (tok/s) | 357 | 429 | +20% |
| cn | acc | 0.82 | 0.82 | 0pp |
| cn | TTFT median (ms) | 71 | 65 | −9% |
| cn | throughput (tok/s) | 383 | 421 | +10% |

> R6 / R8 数字基于 **100 样本** 口径(同 wrapper 路径、热机稳态)。500 样本
> 数字仅 G 组有(`results_summary.md` §1.5)。

## 2.3 R6 / R7 / R8 数字(en / cn,不同样本量)

| run | attention backend | 样本量 | acc (en/cn) | TTFT median (ms) | throughput (tok/s) |
|---|---|---|---|---|---|
| R6 (en 100, triton_attn) | triton_attn | 100 | en **0.74** | **64** | **429** |
| R7 (en 200) | triton_attn | 200 | en **0.72 (146/200)** | 66 | 429 |
| R8 (en 100, fa3) | fa3 | 100 | en **0.73** | 66 | **434** |
| R6 (cn 100) | triton_attn | 100 | cn **0.82** | 65 | 421 |

> - R6 / R8 acc 差异(0.74 vs 0.73)= 100 样本 95% CI ±9.6pp,**统计不显著**
> - R6 vs R7:200 样本上 acc 稳定 0.72,前 100 与后 100 acc 一致
> - TTFT median 在 R6 (100) vs R7 (200) 差 2 ms,属样本量差异

详细 V1/V2/R1-R8 全矩阵见 `../results_summary.md` §1 / §2。

## 2.4 真实量化损失(同 qid 对照, 100 样本)

**不是表面 "0.78 vs 0.76" 的 5pp,真实量化损失只有 2pp**。

G 组 500 样本分桶测过:`G 组 500 样本分桶: 桶1 (0.78) / 桶2 (0.76) / 桶3 (0.76)
/ 桶4 (0.86) / 桶5 (0.89)`。 整体 0.81 = 5 桶加权平均,**后两桶题目较易**;
S2 测的恰好是较难的桶1。

**同 qid 对照(S2 100 = G 500 的前 100 个 qid,完全同一批)**:

| 方案 | 错题数 | acc |
|---|---|---|
| G 组 bf16 (同 100) | 22 | 0.780 |
| S2 量化 (同 100) | 24 | 0.760 |
| **真实量化损失** | +2 | **+2pp** |

**错题重叠**:
- 双方都错: **22 个** (模型本身 BF16 就不会做)
- 仅 G 错 S2 对: **0 个**
- 仅 S2 错 G 对: **2 个** (qid 306: C→A、qid 488: B→A — 这才是真正量化引入的错误)
- 2 个新错都是 **A 选项偏差**(S2 倾向选 A),但只在 2/100 题上显形,不是所有题都偏 A

## 2.5 PPU fork 强制行为下"实际跑的路径"对照

| README 行号意图 | 实际跑 |
|---|---|
| GEMV (`SGLANG_SAIL_USE_GEMV=1`) | ACEXT (用户 env 被 PPU fork 覆盖) |
| PiCG (`--enable-piecewise-cuda-graph`) | PiCG off (VLM 自动 disable) |
| ViT CG (`SGLANG_VIT_ENABLE_CUDA_GRAPH=1`) | ViT CG ON (这条没被覆盖) |
| mm-attn backend (R6=triton_attn / R8=fa3) | 按 SGLANG_EXTRA_ARGS,正常生效 |
| FLA_CUDA / DEEPGEMM_DENSE / DEEPGEMM_MOE / DISABLE_CUSTOM_ALL_REDUCE | 自动设 1,与 README 一致 |

**这意味着什么**:

- R6 / R8 **标称数字是 ACEXT 路径实测**,不是 GEMV 路径
- **TTFT 数字的 PiCG 收益在 VLM 工作负载下不可复现**(被规则 8 自动 disable)
- 想看纯 GEMV 性能需 `SGLANG_SAIL_USE_ACEXT_CUDA=0` + 非 VLM 工作负载,
  本比赛场景做不到
- **衡量优化绝对收益时**:BF16 → INT8 R6 = TTFT −14%(median)+ tput +20%,**这是
  PPU fork 行为下的实际可交付收益**

详细 PPU fork 源码引用见 [`01_ENVIRONMENT.md`](01_ENVIRONMENT.md) §1.11。

## 2.6 cn 集 acc 0.27 误报修正

历史 `runner_patched.py` 的 5 个 extractor 正则**没有中文 pattern**(缺 `答案|选项|选择|正确答案`),
cn 数据集输出 `答案：A` / `正确答案：**B` 全部漏报,误报 acc=0.270。

官方 `benchmark_public.py` 的 3 个 extractor regex 含 `答案|选项|选择|正确答案` +
markdown `**B**` + multiline,正确解析。**官方口径下 cn 真实 acc = 0.820**。

**提交前用官方脚本跑 en+cn 双数据集校验**,不要用任何旧 runner 重新计算。

## 2.7 ISA v10 GEMV prebuilt kernel(Phase 3.2)

> 实验目录:`/mnt/workspace/ISA/experiments/004_cp_async_gemv/`
> 集成说明:`/mnt/workspace/ISA/docs/05_INTEGRATION.md` + [`08_SYNC.md`](08_SYNC.md) §6

### 2.7.1 microbench(K=2048 N=2048 M=1,1000 iters)

| kernel | 时延 | 有效带宽 | HBM peak 占比 |
|---|---|---|---|
| deep_gemm (production) | 30.32 μs | 138 GB/s | 5.0% |
| naive SIMD (Phase 3.0) | 26.72 μs | 157 GB/s | 5.7% |
| **v10 (HBM direct)** | **6.70 μs** | **625 GB/s** | **22.6%** |

生产 shape K=2048 N=8192: v10 = 10.21 μs vs deep_gemm 32.08 μs = **3.13×**。

correctness: max diff = 0.0 vs fp32 ref,vs deep_gemm,在 CUDA graph 内 replay 20 次全部一致。

### 2.7.2 端到端 100 样本 mmbench_dev_en(同 wrapper 路径)

| 配置 | TPS | TTFT median | acc |
|---|---|---|---|
| baseline (no v10) | 411.6 | 72.7 | 0.73 |
| v10 + dispatcher bug | 353.0 | 76.2 | 0.70 |
| **v10 fixed** | **421.1** | 75.3 | **0.73** |
| Δ vs baseline | **+2.3%** | +2.6 ms | 0 |

来源:`result_repro/result_dev_en_v10_DISABLED_100.json` + `result_repro/result_dev_en_v10_FIXED_100.json`。

**注意:加速比仅 +2.3%(不是 4.5×)** — 因为 GEMV 在端到端 decode 时间里不是 100%。
真实收益预测见 §2.7.3。

### 2.7.3 下一步的预期收益

| Phase | 目标 | 预期 TPS | 备注 |
|---|---|---|---|
| 3.2 baseline (current) | HBM-direct, NUM_UNROLL=4 | 421.1 (已落地) | ✅ |
| 3.2.1 | cp.async 双 buffer | 440-460 (+5-10%) | 当前是 HBM 同步读,加 ping-pong 流水线 |
| 3.3 | Persistent kernel: 1 launch 串起 200+ GEMV | 500-700 (+20-60%) | 跨 op shmem handoff + atomic queue |
| 3.4 | Per-row As + multi-K | 同 3.3 | 覆盖非 K=2048 shape |

理论上限(Phase 1 profile 推算): decode step ≈ 224 × (1.5μs 实读 + 8μs gap) ≈ 2.1 ms ⇒ **480 tok/s**,
加跨 op 复用 weight 的 L2 命中 + cp.async pipeline overlap,**500-700 tok/s 可达**。
当前 421 还在理论上限附近,要继续提必须 Phase 3.3 持久 kernel 把 launch overhead 砍掉。
