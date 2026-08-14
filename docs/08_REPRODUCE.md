# 08. 完整复现指南

> 按这个文档可以在新机器上完整复现所有实验结果。

## 8.1 准备

### 环境假设

- 1× PPU-ZW810E（96GB）
- 镜像 `pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.10.0-ubuntu24.04-cuda13.0-sglang0.5.10-py312`
- 模型 `./Qwen3.5-2B` 已下载
- 数据集 `./datasets/mmbench/mmbench_dev_en.tsv` 已下载
- INT8 量化包 `./quant/Qwen3.5-2B-W8A8/`（可选）

### 验证

```bash
python3 --version                                    # 3.12
python3 -c "import sglang; print(sglang.__version__)" # 0.5.10+v0.3.0.ppu2.1.1
which asys                                            # /usr/local/PPU_SDK/asight/bin/asys
asys --version                                        # v2.1.1_20260725-15d8b9d
```

## 8.2 一键启动 + benchmark

```bash
cd /mnt/workspace/dndx_participant-v1.2
bash docs/06_FINAL_CONFIG.md#6.4 中的 run_g_group.sh
```

或手动：

```bash
# 1) 杀旧 server
pkill -9 -f sglang 2>/dev/null || true
sleep 5

# 2) 启动 G 组
nohup python3 -u -m sglang.launch_server \
  --model-path ./Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 \
  --enable-multimodal \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --page-size 64 \
  --log-level info \
  > logs/sglang_g_group_8999.log 2>&1 &

# 3) 等就绪
for i in $(seq 1 30); do
  if curl -s http://127.0.0.1:8999/health_generate --max-time 2 > /dev/null; then
    echo "ready after $((i*5))s"
    break
  fi
  sleep 5
done

# 4) 跑 benchmark
python3 benchmark_public.py \
  --dataset-path ./datasets/mmbench/mmbench_dev_en.tsv \
  --model-path ./Qwen3.5-2B \
  --backend sglang --num-samples 100 \
  --output result_dev_en_g_group_100.json
```

## 8.3 复现各对照组（按时间顺序）

### A. triton baseline

```bash
SGLANG_SAIL_FLA_CUDA=0 python3 -u -m sglang.launch_server ... \
  --attention-backend triton ... &
python3 benchmark_public.py --backend sglang --num-samples 20 \
  --output result_dev_en_sglang_20.json
```

### B. fa3

```bash
SGLANG_SAIL_FLA_CUDA=0 python3 -u -m sglang.launch_server ... \
  --attention-backend fa3 ... &
python3 benchmark_public.py --backend sglang --num-samples 20 \
  --output result_dev_en_sglang_fa3_20.json
```

### C. fa3 + FLA=1

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server ... \
  --attention-backend fa3 ... &
python3 benchmark_public.py --backend sglang --num-samples 20 \
  --output result_dev_en_sglang_fla1_20.json
```

### D. fa3 + FLA=1 + no-radix

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server ... \
  --attention-backend fa3 --disable-radix-cache ... &
python3 benchmark_public.py --backend sglang --num-samples 20 \
  --output result_dev_en_sglang_fla1_noradix_20.json
```

### E. D + chunked-prefill-size 2048

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server ... \
  --attention-backend fa3 --disable-radix-cache --chunked-prefill-size 2048 ... &
python3 benchmark_public.py --backend sglang --num-samples 20 \
  --output result_dev_en_sglang_chunked2048_20.json
```

### G. D + page-size 64（最终）

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server ... \
  --attention-backend fa3 --disable-radix-cache --page-size 64 ... &
python3 benchmark_public.py --backend sglang --num-samples 20 \
  --output result_dev_en_sglang_page64_20.json
```

### INT8（量化对比）

```bash
python3 -u -m sglang.launch_server \
  --model-path ./quant/Qwen3.5-2B-W8A8 \
  --host 127.0.0.1 --port 9001 \
  --trust-remote-code --tp-size 1 \
  --attention-backend triton \
  --enable-multimodal \
  --quantization w8a8_int8 \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --page-size 64 \
  --disable-radix-cache &

python3 benchmark_public.py --backend sglang --num-samples 50 \
  --output result_dev_en_int8_triton_50.json
```

### NEXTN（应崩）

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server ... \
  --attention-backend fa3 --disable-radix-cache --page-size 64 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
# 应在 12 秒内崩：AttributeError: 'NoneType' object has no attribute 'int'
```

## 8.4 复现算子 profile

```bash
# 找 scheduler PID
SCHED_PID=$(ps aux | grep "sglang::scheduler" | grep -v grep | awk '{print $2}')

# 60s trace
asys attach -t hggc -s process-tree --force-overwrite true \
  -o profile_results/g_group_60s.asysrep --duration 60 $SCHED_PID

# 并发请求（必须，否则 trace 数据很少）
nohup python3 -u -c "
import requests, csv, random, concurrent.futures
random.seed(20260625)
with open('datasets/mmbench/mmbench_dev_en.tsv') as f:
    samples = list(csv.DictReader(f, delimiter='\t'))
def run(i):
    s = samples[i]
    r = requests.post('http://127.0.0.1:8999/v1/chat/completions', json={...}, timeout=120)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(run, range(50)))
" &

# 等 asys 完成
wait $!

# 导出
asys stats -r ppu_op_type_sum -f csv --force-overwrite true -o . profile_results/g_group_60s.asysrep
asys stats -r ppu_op_sum -f csv --force-overwrite true -o . profile_results/g_group_60s.asysrep
asys stats -r ppu_op_kernel_breakdown -f csv --force-overwrite true -o . profile_results/g_group_60s.asysrep
asys stats -r hggc_kern_exec_trace -f csv --force-overwrite true -o . profile_results/g_group_60s.asysrep  # 巨大
```

## 8.5 复现 GEMV 定位分析

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

## 8.6 验证清单

复现后必须验证：

| 指标 | 期望值 | 验证方法 |
|---|---|---|
| accuracy | ≥ 0.83 (推荐 G 组 0.85) | `result_*.json["accuracy"]["score"]` |
| avg_ttft_ms | ≤ 75 ms | `result_*.json["performance"]["avg_ttft_ms"]` |
| throughput | ≥ 350 tok/s | `result_*.json["performance"]["avg_throughput_tokens_per_sec"]` |
| backend | "sglang" | `result_*.json["backend"]` |
| token_count 真实 | usage.completion_tokens > 0 | `result_*.json["answers"][0]["meta"]["completion_tokens"]` |
| 无离群点 | TTFT max < 200ms | 自己看分布 |

## 8.7 故障排查

| 症状 | 原因 | 解决 |
|---|---|---|
| `ValueError: Transformers does not recognize this architecture` | transformers 4.57.1 太老 | 走 sglang 后端 |
| sglang 启动 60+ 秒 | CUDA Graph 抓取 | 正常，等待 |
| TTFT ~340ms 平台 | `SGLANG_SAIL_FLA_CUDA=0` | 设 `SGLANG_SAIL_FLA_CUDA=1` |
| throughput ~280 tok/s | radix cache 开启 | 加 `--disable-radix-cache` |
| accuracy 0.66 | 用 INT8 模型 | 改回 bf16 |
| `retrieve_parent_token.int()` 崩溃 | 用 EAGLE/NEXTN | 移除 `--speculative-*` 参数 |
| asys 报 "Ftrace is not available" | PPU 限制 | 忽略，用 `-s process-tree` |
| asys 报 "Perf is not available" | PPU 限制 | 忽略 |