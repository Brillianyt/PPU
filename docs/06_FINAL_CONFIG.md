# 06. 最终推荐配置（G 组）

## 6.1 启动命令

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server \
  --model-path ./Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 \
  --enable-multimodal \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --page-size 64 \
  --log-level info
```

## 6.2 `evaluation_wrapper.py` 默认已锁定 G 组配置

`_SGLangServer._launch()` 默认开启：
- `--attention-backend fa3`
- `env["SGLANG_SAIL_FLA_CUDA"] = "1"`（关键！）
- `--disable-radix-cache`
- `--page-size 64`

可调环境变量：

```bash
# 覆盖 attention backend
SGLANG_ATTENTION_BACKEND=triton python3 benchmark_public.py ...

# 覆盖 TP 大小（仅当有多卡）
SGLANG_TP_SIZE=2 ...

# 覆盖 dtype（注意：INT8 需要 --quantization flag）
SGLANG_DTYPE=float16 ...

# 覆盖 mem-fraction-static
SGLANG_MEM_FRACTION=0.90 ...

# 强制开启 radix（不推荐）
SGLANG_FORCE_RADIX_CACHE=1 ...

# 注入任意额外参数
SGLANG_EXTRA_ARGS="--max-running-requests 32" ...
```

## 6.3 最终成绩单

| 配置 | accuracy | avg_ttft_ms | throughput tok/s | TTFT max |
|---|---|---|---|---|
| A. triton（baseline） | 0.80 | 419.0 | 284.5 | 3074 ms |
| B. fa3 | 0.85 | 313.2 | 279.4 | 762 ms |
| C. fa3 + FLA=1 | 0.85 | 64.3 | 280.1 | 79 ms |
| D. fa3 + FLA=1 + no-radix | 0.85 | 69.0 | 369.4 | 80 ms |
| **G. D + page=64** ⭐ 推荐 | 0.85 | **64.7** | 369.0 | 74 ms |
| E. D + chunked=2048 | 0.85 | 69.9 | 375.3 | 86 ms |
| F / F2 / NEXTN | — | crashed | — | — |

**对比 baseline (A)**：
- TTFT: **-84.6 %**
- Throughput: **+29.7 %**
- Accuracy: **+5 pp**

## 6.4 启动 + 监控脚本

```bash
#!/bin/bash
# run_g_group.sh
set -e
cd /mnt/workspace/dndx_participant-v1.2

# 杀旧 server
pkill -9 -f sglang 2>/dev/null || true
sleep 5

# 启动 G 组服务器（后台）
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

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# 等就绪
for i in $(seq 1 30); do
  if curl -s http://127.0.0.1:8999/health_generate --max-time 2 > /dev/null; then
    echo "ready after $((i*5))s"
    break
  fi
  sleep 5
done

# 跑 benchmark
python3 benchmark_public.py \
  --dataset-path ./datasets/mmbench/mmbench_dev_en.tsv \
  --model-path ./Qwen3.5-2B \
  --backend sglang --num-samples 100 \
  --output result_dev_en_g_group_100.json
```

## 6.5 profile 监控（可选）

```bash
# 找 scheduler PID
SCHED_PID=$(ps aux | grep "sglang::scheduler" | grep -v grep | awk '{print $2}')
echo "Scheduler PID: $SCHED_PID"

# 60s 抓 trace
asys attach -t hggc -s process-tree --force-overwrite true \
  -o profile_results/monitor_60s.asysrep --duration 60 $SCHED_PID

# 导出汇总
asys stats -r ppu_op_type_sum -f csv --force-overwrite true -o . profile_results/monitor_60s.asysrep
asys stats -r ppu_op_kernel_breakdown -f csv --force-overwrite true -o . profile_results/monitor_60s.asysrep
```

## 6.6 验证清单（提交前必跑）

- [ ] `python3 benchmark_public.py --backend sglang --num-samples 20` → accuracy ≥ 0.83
- [ ] `python3 benchmark_public.py --backend sglang --num-samples 100` → accuracy ≥ 0.83, throughput ≥ 350 tok/s, TTFT ≤ 75ms
- [ ] `python3 benchmark_public.py --backend sglang --num-samples 50 --dataset-path ./datasets/mmbench/mmbench_dev_cn.tsv` → 中文 accuracy 正常
- [ ] 检查 sglang server 日志无崩溃
- [ ] 检查 `result_*.json` 里 `meta.backend == "sglang"` 且 `prompt_tokens` / `completion_tokens` 来自 sglang usage（不是估算）
- [ ] `evaluation_wrapper.py` 接口契约不变
- [ ] 不传长 timeout 参数
- [ ] 不维护任何 sample_id → answer 缓存