# 06. 各对照组精确复现命令

> 快速复现见 [`../README_REPRODUCE.md`](../README_REPRODUCE.md) §2(SSOT)。
> 本文件是**精确启动参数 + patch 步骤**——和 README_REPRODUCE 配合使用。

## 6.1 共同前置

```bash
# 杀旧 server
pkill -9 -f "sglang.launch_server\|sglang::scheduler\|sglang::detokenizer" 2>/dev/null
sleep 5

# 准备目录
mkdir -p /mnt/workspace/PPU/logs
mkdir -p /mnt/workspace/PPU/result_repro
```

## 6.2 G 组 BF16 baseline (无 patch)

> 无需 patch 部署,直接启就行。这是对照基线。

```bash
PPU_SDK=/usr/local/PPU_SDK \
SGLANG_SAIL_FLA_CUDA=1 \
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
  > /mnt/workspace/PPU/logs/sglang_g_group_8999.log 2>&1 &

# 等就绪
for i in $(seq 1 30); do
  curl -s http://127.0.0.1:8999/health_generate --max-time 2 >/dev/null \
    && { echo "ready after $((i*5))s"; break; }
  sleep 5
done

# 跑分(用评价入口)
python3 /mnt/workspace/PPU/benchmark_public.py \
  --dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/workspace/Qwen3.5-2B \
  --backend sglang --num-samples 100 --warmup-samples 8 \
  --output /mnt/workspace/PPU/result_repro/g_group_100.json
```

**预期**: acc 0.78 / TTFT median 73 ms / throughput 357 tok/s (500 样本统计稳态)。

## 6.3 Patch 部署 (6 类共 10 文件, 必装)

> 详见 [`01_ENVIRONMENT.md`](01_ENVIRONMENT.md) §1.10。复制 A-F 完整块即可。

```bash
cd /mnt/workspace/PPU

# A/B/C/E — sglang 9 文件
cp -rv gemv_optimization/sglang-for-sail/python/sglang/* \
        /usr/local/lib/python3.12/site-packages/sglang/

# D — DeepGEMM 3 文件
cp -rv gemv_optimization/DeepGEMM-for-sail/deep_gemm/* \
        /usr/local/lib/python3.12/site-packages/deep_gemm/

# F — Inductor RoPE hook (1 文件 + platforms/ppu.py 一行 import)
cp -v inductor_hook_PPU/hook/ppu_rotary_hooks.py \
      /usr/local/lib/python3.12/site-packages/sglang/srt/hardware_backend/ppu/attention/ppu_rotary_hooks.py
grep -q "ppu_rotary_hooks" /usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py \
  || sed -i '/ppu_flashmla_hooks/a\    import sglang.srt.hardware_backend.ppu.attention.ppu_rotary_hooks  # noqa: F401' \
        /usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py

# 清 __pycache__
rm -rf /usr/local/lib/python3.12/site-packages/sglang/srt/{layers/quantization,layers/deep_gemm_wrapper,models,hardware_backend/ppu/attention,platforms}/__pycache__
rm -rf /usr/local/lib/python3.12/site-packages/deep_gemm/{,jit_kernels}/__pycache__
```

### 验证 6 类 patch

```bash
# A
grep -c "SGLANG_SAIL_USE_GEMV" /usr/local/lib/python3.12/site-packages/sglang/srt/environ.py   # ≥ 1
# B
grep -c "gemv_int8_i8i8bf16" /usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/w8a8_int8.py  # ≥ 4
# C — 文件存在
ls /usr/local/lib/python3.12/site-packages/sglang/srt/layers/deep_gemm_wrapper/{compile_utils,entrypoint}.py
# D — 文件存在
ls /usr/local/lib/python3.12/site-packages/deep_gemm/jit_kernels/gemv_int8.py
# E1
grep -c "max_context_len" /usr/local/lib/python3.12/site-packages/sglang/srt/models/qwen3_vl.py  # ≥ 2
# E2
grep -c "ppu_vision_flash_attn_varlen_func" \
  /usr/local/lib/python3.12/site-packages/sglang/srt/hardware_backend/ppu/attention/ppu_fa3_hooks.py  # ≥ 1
# F1
ls /usr/local/lib/python3.12/site-packages/sglang/srt/hardware_backend/ppu/attention/ppu_rotary_hooks.py
# F2
grep "ppu_rotary_hooks" /usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py   # ≥ 1
```

**行为陷阱(影响结果解读,见 [`01_ENVIRONMENT.md`](01_ENVIRONMENT.md) §1.11)**:

| 行为 | 影响 |
|---|---|
| PPU fork 默认开 ACEXT (`SGLANG_SAIL_USE_ACEXT_CUDA=True`) | 实际跑的是 ACEXT 路径,不是 README 行号上的 GEMV 路径 |
| PPU fork 在 VLM 模型下自动 `disable_piecewise_cuda_graph=True` | PiCG 收益不可在 VLM 工作负载上复现 |
| 评估 TTFT 用 **median** | 100 样本 mean 73-280 ms 是尾部冷启动拉高的;median 63 ms 才是稳态 |

## 6.4 R6 / R8 启动 (走 wrapper)

> 比赛提交件 = `evaluation_wrapper.py`,**不需要手工启 server**,让 wrapper 在
> `_SGLangServer._launch()` 里自动启。

### R6 (极致 TTFT+acc,triton_attn backend) ⭐ 推荐

```bash
SGLANG_EXTRA_ARGS="--mm-attention-backend triton_attn" \
PPU_SDK=/usr/local/PPU_SDK \
python3 -u /mnt/workspace/PPU/benchmark_public.py \
  --dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --backend sglang --num-samples 100 --warmup-samples 8 \
  --output /mnt/workspace/PPU/result_repro/r6_en_100.json
```

**预期**(en 100,warmup 8):
- accuracy: **0.74** (0.78 baseline, −4pp 不显著)
- TTFT median: **63 ms**
- throughput: **429 tok/s**

### R8 (极致 TPS,fa3 backend)

```bash
# wrapper 默认 attention backend = fa3, SGLANG_EXTRA_ARGS 可以省略
PPU_SDK=/usr/local/PPU_SDK \
python3 -u /mnt/workspace/PPU/benchmark_public.py \
  --dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --backend sglang --num-samples 100 --warmup-samples 8 \
  --output /mnt/workspace/PPU/result_repro/r8_en_100.json
```

**预期**(en 100,warmup 8):
- accuracy: **0.73**
- TTFT median: **66 ms**
- throughput: **434 tok/s**

### 手工启 server 命令(无需 wrapper)

如需跳过 wrapper 直接手工启(便于调试):

```bash
cd /mnt/workspace/PPU

SGLANG_EXTRA_ARGS="--mm-attention-backend triton_attn" \
PPU_SDK=/usr/local/PPU_SDK \
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 \
SGLANG_CUDA_GRAPH_MAX_BS=16 \
SGLANG_SAIL_USE_ACEXT_CUDA=0 \
SGLANG_SAIL_USE_GEMV=1 \
SGLANG_SAIL_GEMV_M_MAX=8 \
SGLANG_SAIL_GEMV_WARMUP_ON_LOAD=1 \
SGLANG_SAIL_PREFILL_BF16=1 \
SGLANG_SAIL_FLA_CUDA=1 \
SGLANG_SAIL_DEEPGEMM_DENSE=1 \
nohup python3 -u -m sglang.launch_server \
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
  --mm-attention-backend triton_attn \
  --log-level info \
  > /mnt/workspace/PPU/logs/sglang_r6_8999.log 2>&1 &
```

> ⚠️ `SGLANG_SAIL_USE_ACEXT_CUDA=0` 实际无效(PPU fork 强制设为 1),见
> `01_ENVIRONMENT.md` §1.11。

## 6.5 CN 数据集对照 (R6 同配置, 验证 INT8 零精度损失)

```bash
SGLANG_EXTRA_ARGS="--mm-attention-backend triton_attn" \
PPU_SDK=/usr/local/PPU_SDK \
python3 -u /mnt/workspace/PPU/benchmark_public.py \
  --dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_cn.tsv \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --backend sglang --num-samples 100 --warmup-samples 8 \
  --output /mnt/workspace/PPU/result_repro/r6_cn_100.json
```

**预期**: acc **0.82**(= BF16 baseline,**INT8 量化零精度损失**)。

## 6.6 算子 profile 复现 (asys)

> 见 [`05_KERNEL_PROFILE.md`](05_KERNEL_PROFILE.md) §5.1 完整流程。

快速复用:
```bash
SCHED_PID=$(pgrep -f "sglang::scheduler" | head -1)
asys attach -t hggc -s process-tree --force-overwrite true \
  -o /mnt/workspace/PPU/profile_results/60s.asysrep --duration 60 $SCHED_PID

# trace 期间并发 50 个请求(同 6.4 evaluator 期间跑即可)
asys stats -r ppu_op_kernel_breakdown -f csv --force-overwrite true -o . \
  /mnt/workspace/PPU/profile_results/60s.asysrep
```

## 6.7 历史对照表 (100 样本)

> 数字均来自 [`02_RESULTS.md`](02_RESULTS.md) §2.2 / §2.3。复现命令同上,仅数据集或样本量不同。

| 编号 | 模型 | attention | samples | acc | TTFT | tput | 备注 |
|---|---|---|---|---|---|---|---|
| G-baseline | bf16 | fa3 | 100 | 0.78 | 73 ms | 357 tok/s | 对照基线,无 patch |
| G-baseline-500 | bf16 | fa3 | 500 | 0.81 | 73 ms | 357 tok/s | 500 样本稳态 |
| R6 | S2-INT8 | triton_attn | 100 | 0.74 | 63 ms | 429 tok/s | ⭐ 推荐 |
| R7 | S2-INT8 | triton_attn | 200 | 0.72 | 66 ms | 429 tok/s | 200 样本稳定性 |
| R8 | S2-INT8 | fa3 | 100 | 0.73 | 66 ms | 434 tok/s | 极致 TPS |
| R6-cn | S2-INT8 | triton_attn | 100 | **0.82** | 65 ms | 421 tok/s | cn 上 INT8 零精度损失 |
