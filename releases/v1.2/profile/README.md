# 原始 asys profile 数据（.asysrep）

> `asys attach -t hggc -s process-tree --duration 60 <sglang_scheduler_pid>` 抓的原始 GPU kernel trace。
> 这些是 `.gitignore` 不收的原始数据，单独放到这里便于复现分析。

## 文件

| 文件 | 大小 | 配置 | 时长 | 备注 |
|---|---|---|---|---|
| `g_group_60s.asysrep` | 87 MB | G 组（FA3 + FLA=1 + no-radix + page=64） | 60s | **主分析对象**，本仓库所有 profile CSV 都从这里导出 |
| `int8_60s.asysrep` | 15 MB | INT8 对比组 | 60s | 量化对照，导出对比 kernel 分布 |

## 怎么读

```bash
# 0) 安装 asys（在 DSW 镜像里已预装）
asys --version   # v2.1.1

# 1) 列出 stats
asys stats -r ppu_op_type_sum -f csv -o . g_group_60s.asysrep
asys stats -r ppu_op_sum -f csv -o . g_group_60s.asysrep
asys stats -r ppu_op_kernel_breakdown -f csv -o . g_group_60s.asysrep
asys stats -r hggc_kern_exec_trace -f csv -o . g_group_60s.asysrep

# 2) 在 kernel 粒度分析 GEMV 分布（grid=2048 → q_proj/o_proj/GDN.out_proj）
python3 << 'PY'
import csv
target = "void gemvt_op<cutlass::bfloat16_t, float, cutlass::bfloat16_t, int4, int4, 32, 4, 1, 2, 1, 1"
grids = {}
with open('g_group_60s_hggc_kern_exec_trace.csv') as f:
    for row in csv.DictReader(f):
        if target in row['Kernel Name']:
            grids[row['GridXYZ']] = grids.get(row['GridXYZ'], 0) + 1
for g, c in sorted(grids.items(), key=lambda x: -x[1]):
    print(f"grid={g}: {c}")
PY
```

## 怎么重新生成

```bash
# 1) 启动 sglang（G 组配置）
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server \
  --model-path ./Qwen3.5-2B \
  --attention-backend fa3 --enable-multimodal \
  --disable-radix-cache --page-size 64 \
  --mem-fraction-static 0.85 --dtype bfloat16 &

# 2) 找 scheduler PID
SCHED_PID=$(ps aux | grep "sglang::scheduler" | grep -v grep | awk '{print $2}')

# 3) asys 抓 60s（必须有并发请求，否则 trace 数据很少）
nohup python3 -c "
import requests, csv, random, concurrent.futures
random.seed(20260625)
with open('datasets/mmbench/mmbench_dev_en.tsv') as f:
    samples = list(csv.DictReader(f, delimiter='\t'))
def run(i):
    s = samples[i]
    requests.post('http://127.0.0.1:8999/v1/chat/completions',
        json={'model': 'Qwen3.5-2B', 'messages': [{'role':'user','content':[
            {'type':'text','text':s.get('question','')[:200]},
            {'type':'image_url','image_url':{'url': f'data:image/jpeg;base64,...'}}
        ]}], 'max_tokens': 64}, timeout=120)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(run, range(50)))
" &

asys attach -t hggc -s process-tree --force-overwrite true \
  -o my_trace.asysrep --duration 60 $SCHED_PID
```

## 对应的已分析 CSV

本仓库已包含从这两个 asysrep 导出的关键 CSV（在 `profile_results/` 根目录）：
- `g_group_60s_ppu_op_sum.csv` ← OP 类型聚合
- `g_group_60s_ppu_op_kernel_breakdown.csv` ← kernel 分解
- `g_group_60s_ppu_time_util.csv` ← 时间利用率
- `int8_60s_ppu_op_sum.csv` ← INT8 对照
- `int8_60s_ppu_op_kernel_breakdown.csv`
