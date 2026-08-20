# 06. 各对照组详细复现命令

> 快速复现见 [`../README_REPRODUCE.md`](../README_REPRODUCE.md) §2。
> 本文件是每个对照组的精确启动参数 + 预期指标。

## 6.1 共同前置

```bash
pkill -9 -f "sglang.launch_server\|sglang::scheduler" 2>/dev/null; sleep 5
mkdir -p /mnt/workspace/PPU/logs
```

## 6.2 bf16 路径

### A. sglang triton baseline

```bash
python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend triton \
  --enable-multimodal \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  > logs/sglang_triton_8999.log 2>&1 &
```

预期: acc 0.80, TTFT 419 ms, tput 284 tok/s (20 样本)。

### B. fa3 (无 FLA)

```bash
python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 \
  --enable-multimodal \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  > logs/sglang_fa3_8999.log 2>&1 &
```

预期: acc 0.85, TTFT 313 ms, tput 279 tok/s (20 样本)。

### C. fa3 + FLA=1

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 \
  --enable-multimodal \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  > logs/sglang_fa1_8999.log 2>&1 &
```

预期: acc 0.85, TTFT 64 ms, tput 280 tok/s (20 样本)。

### D. fa3 + FLA=1 + no-radix

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 \
  --enable-multimodal \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  > logs/sglang_fa1_noradix_8999.log 2>&1 &
```

预期: acc 0.85, TTFT 69 ms, tput 369 tok/s (20 样本)。

### ⭐ G. 推荐配置 (fa3 + FLA=1 + no-radix + page=64)

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 \
  --enable-multimodal \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --page-size 64 \
  > logs/sglang_g_group_8999.log 2>&1 &
```

预期: acc 0.85, TTFT 64.7 ms, tput 369 tok/s (20 样本);500 样本: acc 0.810, TTFT 73.3 ms, tput 357 tok/s。

### E. G + chunked-prefill-size 2048

```bash
# 在 G 组命令基础上追加:
--chunked-prefill-size 2048
```

预期: acc 0.85, TTFT 69.9 ms, tput 375 tok/s (20 样本,几乎与 G 等价)。

## 6.3 INT8 路径

### 前置: 应用 patches

```bash
cp -rv /mnt/workspace/patches/sglang/* /usr/local/lib/python3.12/site-packages/sglang/
cp -rv /mnt/workspace/patches/deep_gemm/* /usr/local/lib/python3.12/site-packages/deep_gemm/
```

### INT8 §26 (fa3 + FLA=1 + DeepGEMM tuned)

```bash
# Step 1: 启动 server
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-AMQ-QKV-T015 \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 --enable-multimodal \
  --quantization w8a8_int8 --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --page-size 64 --disable-radix-cache \
  --log-level info > logs/int8_fa3_fla1.log 2>&1 &

# Step 2: 等就绪,跑一些请求预热
# ... 跑 ~30 个请求 ...

# Step 3: SIGINT 给 scheduler (不是 launch_server!),dump tunner 候选
SCHED_PID=$(pgrep -f "sglang::scheduler")
kill -INT $SCHED_PID

# Step 4: 跑 tuner (与 serving 并行)
python3 -m sglang.tune_deepgemm_int8 --tp 1 > /tmp/tuner.log 2>&1 &

# Step 5: tuner 完成 (~15-30 分钟) 后,重启 server,跑分
```

预期 (500 样本): acc 0.808-0.822, TTFT 78 ms, tput 358 tok/s。

### ⭐ S2+ACEXT+pCG (推荐冲吞吐方案)

```bash
SGLANG_SAIL_USE_ACEXT_CUDA=1 nohup python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 --enable-multimodal \
  --quantization w8a8_int8 \
  --dtype bfloat16 --mem-fraction-static 0.75 \
  --page-size 64 --disable-radix-cache \
  --enable-piecewise-cuda-graph \
  --piecewise-cuda-graph-compiler eager \
  --mm-attention-backend fa3 \
  --cuda-graph-max-bs 256 \
  --log-level info > logs/sglang_s2_acext_8999.log 2>&1 &
```

预期 (100 样本): en acc 0.76 / cn acc 0.82, TTFT 稳态 78 ms, tput 410 tok/s。

## 6.4 跑分

```bash
python3 benchmark_public.py \
  --dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/workspace/Qwen3.5-2B \
  --backend sglang --num-samples 20 \
  --warmup-samples 2 \
  --output result_dev_en_g_group_20.json
```

中文换 `--dataset-path /mnt/workspace/datasets/mmbench/mmbench_dev_cn.tsv`。

## 6.5 算子 profile 复现

```bash
# 找 scheduler PID
SCHED_PID=$(pgrep -f "sglang::scheduler" | head -1)

# 60s trace
asys attach -t hggc -s process-tree --force-overwrite true \
  -o profile_results/monitor_60s.asysrep --duration 60 $SCHED_PID

# 必须并发请求,否则 trace 数据很少
nohup python3 -u -c "
import requests, csv, random, concurrent.futures
random.seed(20260625)
with open('/mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv') as f:
    samples = list(csv.DictReader(f, delimiter='\t'))
def run(i):
    s = samples[i]
    requests.post('http://127.0.0.1:8999/v1/chat/completions',
        json={'model': 'Qwen3.5-2B', 'messages': [{'role':'user','content':[
            {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,xxx'}},
            {'type':'text','text':s['question']}]}],
            'max_tokens':256,'temperature':0,'stream':False}, timeout=120)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(run, range(50)))
" &
wait $!

# 导出汇总
asys stats -r ppu_op_type_sum -f csv --force-overwrite true -o . profile_results/monitor_60s.asysrep
asys stats -r ppu_op_kernel_breakdown -f csv --force-overwrite true -o . profile_results/monitor_60s.asysrep
```

## 6.6 复现 GEMV 定位分析

```bash
python3 << 'EOF'
import csv
target = "void gemvt_op<cutlass::bfloat16_t, float, cutlass::bfloat16_t, int4, int4, 32, 4, 1, 2, 1, 1"
grids = {}
with open('profile_results/g_group_60s_hggc_kern_exec_trace.csv') as f:
    for row in csv.DictReader(f):
        if target in row['Kernel Name']:
            g = row['GridXYZ']
            grids[g] = grids.get(g, 0) + 1
print("Top GEMV grid distribution:")
for g, c in sorted(grids.items(), key=lambda x: -x[1]):
    print(f"  grid={g}: {c}")
EOF
```