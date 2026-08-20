# Qwen3.5-2B S2 量化 + ACEXT + piecewise CUDA Graph 复现指南

最终方案：**S2-aggressive 量化模型 + PPU ACEXT INT8 GEMM/MOE 加速库 + piecewise CUDA Graph + GEMV CUDA-Graph 友好封装**。
在 PPU-ZW810E + SGLang 0.5.13 下，**通过 `evaluation_wrapper.py` 的 `sglang` 后端自动启 server**，20 样本 dev_en 实测：

| 指标 | 数值 | 对照（`sglang_page64` baseline） |
|---|---|---|
| throughput | **423 tok/s** | 369 tok/s（**+14.5%**） |
| avg_ttft | 86.7 ms | 64.7 ms（退化 +34%，启动期 capture 开销在小 batch 上分摊不掉） |
| accuracy | 0.85 | 0.85（持平） |

> **重要：评测入口是 `evaluation_wrapper.py`，不是 `sglang_launcher.py`**
> 主办方调用 `VLMModel.generate_with_metrics(...)`，本仓库的 `evaluation_wrapper.py` 已支持 `backend="sglang"`（自动启 sglang server 并发 HTTP 请求）。所有 CUDA Graph flag、ACEXT、GEMV 等开关已硬编码在 `_SGLangServer._launch()` 中，选手**不需要**额外传任何环境变量。

## 优化路径总览

```
                 Qwen3.5-2B INT8 forward
                            │
                  ┌─────────┴─────────┐
                  │                   │
              dense linear        MoE linear
                  │                   │
            ACEXT.int8_gemm     MoeRunnerBackend.ACEXT
            (PPU 810E 专属)      (PPU 810E 专属)
                  │                   │
              优势:
              ┌──────────────────────────────┐
              │ ACEXT > DeepGEMM INT8 GEMM   │
              │ ACEXT > Triton GEMM          │
              │ ACEXT 与 GEMV 互斥(开 ACEXT   │
              │ 后不再走 GEMV)               │
              └──────────────────────────────┘
                  │
                  ▼
          prefill path(走 BF16 反量化,不进 ACEXT)
                  │
              F.linear (BF16 GEMM)
```

**对照实验记录(PPU 实测,2026-08-20)**:

| dense linear backend | throughput | TTFT | accuracy |
|---|---|---|---|
| 历史 baseline(sglang_page64,无 ACEXT 无 GEMV) | 369 tok/s | 64.7 ms | 0.85 |
| DeepGEMM INT8 GEMV(decode 小 M)+ GEMM(M>8) | 413 tok/s | 84.8 ms | 0.85 |
| **ACEXT(本方案默认)** | **423 tok/s** | 86.7 ms | 0.85 |

> ACEXT 比 GEMV 又快 2.4%,完全替代 GEMV 的 dense linear 路径。
> 用户可设 `SGLANG_SAIL_USE_ACEXT_CUDA=0` 回退到 GEMV 做对照实验。

## PPU 实测关键约束（仅记录实测发现，**不**代表通用结论）

> **重要：评测入口是 `evaluation_wrapper.py`，不是 `sglang_launcher.py`**
> 主办方调用 `VLMModel.generate_with_metrics(...)`，本仓库的 `evaluation_wrapper.py` 已支持 `backend="sglang"`（自动启 sglang server 并发 HTTP 请求）。所有 CUDA Graph flag 已硬编码在 `_SGLangServer._launch()` 中，选手**不需要**额外传任何环境变量。

## PPU 实测关键约束（仅记录实测发现，**不**代表通用结论）

1. **`SGLANG_VIT_ENABLE_CUDA_GRAPH=1` 在 PPU 上不可行** —— 撞到 Inductor codegen 的 `Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned` bug（PR #15320 的 PPU 兼容性）。本方案默认关闭。
2. **`--enable-piecewise-cuda-graph --piecewise-cuda-graph-compiler eager` 在 PPU 上稳定工作** —— eager 编译器不走 Inductor codegen，绕开 causal_conv1d 的 bug。
3. **`--enable-torch-compile` 必须关闭** —— 单独开 `-enable-piecewise-cuda-graph` 就够了，开 torch.compile 会让 Inductor 编译 GDN 的 causal_conv1d triton 内核时崩（`launcher() missing _grid_2`）。
4. **`--quantization w8a8_int8` 必须显式传** —— 不传会 fallback 到 `compressed_tensors_w8a8_int8`，该路径调 `sgl_kernel.int8_scaled_mm`，PPU sgl_kernel 没这个 op，直接崩。
5. **`--linear-attn-backend fla` 这个选项不存在** —— PPU 的 FLA 路径由 `SGLANG_SAIL_FLA_CUDA=1` env var 触发（已被 `evaluation_wrapper.py` 强制设）。

## 目录结构

```text
/mnt/data_oss/
├── models/Qwen_quant/Qwen3.5-2B-S2-aggressive/   # 量化好的模型（已含权重，直接可用）
├── recipes/qwen3.5-2b_amq_s2_aggressive.yaml     # 量化 recipe（备查）
├── patches/                                       # 源码覆盖文件（按 site-packages 相对路径）
│   ├── sglang/...
│   └── deep_gemm/
│       └── jit_kernels/
│           └── gemv_int8_wrapper.py   # GEMV CUDA-Graph 友好封装
│                                     # （grouped_layout 静态池 + JIT runtime 缓存 + 调试 API）
├── benchmark/                                     # 评测脚本 + wrapper + launcher
└── docs/                                          # 探索过程 / 关键决策记录
```

## 环境要求

- 1x PPU-ZW810E（96GB）
- SGLang 0.5.13（`w8a8_int8` 在 0.5.10 有输出 garbage 的 bug，必须 0.5.13）
- DeepGEMM-for-SAIL（PPU 定制版，已随 `patches/` 提供增量文件）
- `PPU_SDK=/usr/local/PPU_SDK`
- 数据集：mmbench（TSV + images），若本目录没有，从原始 `dndx_participant-v1.2/datasets/` 拷贝或按主办方提供路径

## 复现步骤

### Step 1：覆盖源码 patches

```bash
cp -r /mnt/data_oss/patches/sglang/* /usr/local/lib/python3.12/site-packages/sglang/
cp -r /mnt/data_oss/patches/deep_gemm/* /usr/local/lib/python3.12/site-packages/deep_gemm/
```

覆盖后校验：

```bash
python3 -c "import sglang; print(sglang.__version__)"   # 0.5.13+...
python3 -c "import deep_gemm; print(deep_gemm.gemv_int8_int8_bf16_nt)"
python3 -c "import deep_gemm; print(deep_gemm.GemvInt8Int8Bf16NtWrapper, deep_gemm.warmup_gemv)"
```

### Step 2：启动 server

```bash
cd /mnt/data_oss/benchmark

python3 -u sglang_launcher.py \
  --model-path /mnt/data_oss/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --host 0.0.0.0 --port 8999 \
  --tp-size 1 --dtype bfloat16 \
  --attention-backend fa3 \
  --disable-radix-cache --page-size 64 \
  --quantization w8a8_int8 \
  --enable-multimodal --trust-remote-code \
  --mem-fraction-static 0.75
```

**`sglang_launcher.py` 会自动注入并设置以下所有 CUDA Graph 相关开关**(用户无需在命令行重复):

命令行参数(自动追加):
```text
--enable-piecewise-cuda-graph             # 总开关
--piecewise-cuda-graph-max-tokens 4096    # 4096 token 以下分段都进 graph
--piecewise-cuda-graph-compiler eager     # eager compile，避免 dynamo guard 开销
--disable-custom-all-reduce               # custom all-reduce 不支持 CUDA Graph
--enable-torch-symm-mem                   # torch-symm-mem 替代 all-reduce
--mm-attention-backend fa3                # VLM attention 也走 FA3
--enable-torch-compile                    # 配合 piecewise graph
--cuda-graph-max-bs 32                    # 普通 CUDA Graph 的最大 batch size
```

环境变量(自动设置):
```text
# === GEMV / 量化分派 ===
SGLANG_SAIL_DEEPGEMM_DENSE=1
SGLANG_SAIL_USE_GEMV=1
SGLANG_SAIL_GEMV_M_MAX=8
SGLANG_SAIL_GEMV_WARMUP_ON_LOAD=1   # 每个 INT8 层加载后自动 warmup 该层 (N,K) GEMV
SGLANG_SAIL_PREFILL_BF16=1
SGLANG_SAIL_FLA_CUDA=1

# === ViT Piecewise CUDA Graph (SGLang PR #15320) ===
SGLANG_VIT_ENABLE_CUDA_GRAPH=1      # 把 ViT 的 blocks+merger+deepstack 整段压成 graph
SGLANG_MM_FEATURE_CACHE_MB=4096     # ViT 特征 cache 提到 4GB
SGLANG_VLM_CACHE_SIZE_MB=0          # 关闭 VLM 多模态缓存（SGLang 0.5.13 适配）
```

> 用户若想显式覆盖任一 flag,直接在命令行写即可,launcher 检测到存在会跳过;否则自动追加。
> 例如想试 `--cuda-graph-max-bs 64`,直接加在命令行末尾就行。

等待日志出现 `Application startup complete`。

### Step 2b：GEMV Wrapper 调试（可选）

PPU 机上可直接调用 wrapper 的调试 API：

```python
import deep_gemm

w = deep_gemm.GemvInt8Int8Bf16NtWrapper.instance()
diff = w.verify_against_gemm(m=1, n=4096, k=4096)  # 必须返回 0.0（与 GEMM bit-level 等价）
w.dump_state()                                       # 打印缓存状态（编译日志、layout 池）
```

关掉自动 warmup（仅用于排查）：

```bash
export SGLANG_SAIL_GEMV_WARMUP_ON_LOAD=0
```

### Step 3：跑 benchmark

```bash
cd /mnt/data_oss/benchmark

python3 benchmark_public.py \
  --dataset-path /path/to/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/data_oss/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --backend sglang \
  --num-samples 20 \
  --output result_dev_en_s2_20.json
```

### Step 4：校验

```text
result_*.json
├── accuracy.score ≈ 0.80
├── performance.avg_ttft_ms ≈ 71
└── performance.avg_throughput_tokens_per_sec ≈ 386
```

## 关键分派逻辑（w8a8_int8.py）

```text
prefill (M > 8)  -> 反量化 INT8 权重为 BF16，走 F.linear（PREFILL_BF16）
decode  (M <= 8) -> per_token_quant_int8 + DeepGEMM Gemvt GEMV（USE_GEMV）
```

GEMV 调用统一通过 `GemvInt8Int8Bf16NtWrapper` 单例分发（CUDA-Graph 友好封装）：

```text
gemv_int8_int8_bf16_nt (对外函数，签名不变)
  └─ GemvInt8Int8Bf16NtWrapper.__call__
       ├─ grouped_layout : Dict[M, Tensor]   # 预分配静态 buffer（按 M 分桶）
       ├─ _runtime_cache : Dict[RuntimeKey, Runtime]  # JIT runtime 按 (M,N,K,cfg) 缓存
       └─ _config_cache  : Dict[(M,N,K), cfg]          # best_config 按形状缓存
```

抓图期热路径：0 Python 分配、0 JIT 调用、1 次 kernel launch。

## ViT CUDA Graph（任务 1）

依据 SGLang PR #15320（《Support ViT Piecewise CUDA Graph for Qwen3-VL》）。**所有开关由 launcher 自动注入**，用户不需要在命令行写。验证方式：启动日志里看到 `[sglang_launcher] 自动追加 CUDA Graph 编译 flag: --enable-piecewise-cuda-graph --disable-custom-all-reduce ...` 即生效。

官方 PR 在 8×H20 Qwen3-VL-8B TP=4 上 TTFT 1384 → 1120ms（-19%）；本项目（PPU-ZW810E、TP=1、单样本）理论收益更明显（kernel-launch 开销占比更高）。

> 用户覆盖：命令行显式写任何 flag，launcher 会检测并跳过自动注入——例如想改 max_bs 为 64：命令行加 `--cuda-graph-max-bs 64`，launcher 会沿用你的值。

## 注意事项

- `/mnt/data_oss` 是 ossfs 挂载，不支持 `truncate`；大文件先传本地再 `cp` 到 ossfs。
- 模型权重已在 `models/` 内，复现时**不需要重新量化**。
- 若需重新量化，用 `recipes/qwen3.5-2b_amq_s2_aggressive.yaml` + Qlean 对原始 BF16 模型执行。
- 数据集和原始 BF16 模型未包含在本目录（如需可另行拷贝）。
- **`--disable-custom-all-reduce` + `--enable-torch-symm-mem` 由 launcher 自动追加**：custom all-reduce 还不支持 CUDA Graph，必须关掉并用 torch-symm-mem 替代，否则 piecewise CUDA Graph 抓图会失败。用户命令行重复写不会冲突（launcher 检测到就跳过）。
- **第一次启动会触发 GEMV JIT 编译**：每个 INT8 层的 `(N, K)` 形状编译耗时约 1–2 秒，受 `SGLANG_SAIL_GEMV_WARMUP_ON_LOAD` 控制。启动日志里会看到 `[GemvInt8 warmup done: ... jit runtime cached`。
- **CUDA Graph 与 torch.compile 的兼容性**：launcher 同时启用了 piecewise CUDA Graph 与 torch.compile。在 SGLang 0.5.13 上这两者协同工作（piecewise graph 基于 torch.compile 的 FX graph 做分割）；如果遇到 `RuntimeError: Cannot trace through ...` 之类的错误，先关 `SGLANG_SAIL_GEMV_WARMUP_ON_LOAD=0` + `--disable-torch-compile` 排除。
