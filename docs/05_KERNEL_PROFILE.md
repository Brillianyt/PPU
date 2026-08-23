# 05. 算子 profile + GEMV 映射

> 本文件目标:解释**G 组 BF16 baseline + ACEXT/R6** 在 PPU 上跑的算子分布,
> 让复测者能复现并理解性能数字的来源。
>
> 不含比赛路径上的具体优化决策(见 [`04_OPTIMIZATIONS.md`](04_OPTIMIZATIONS.md))。

## 5.1 复现 profile 流程

### 5.1.1 启用 ASYS 跟踪 (60 秒 trace)

```bash
# 找 sglang scheduler PID (单 worker 时通常 = launch_server 子进程)
SCHED_PID=$(pgrep -f "sglang::scheduler" | head -1)
echo "scheduler PID: $SCHED_PID"

# 60 秒 GPU kernel trace
asys attach -t hggc -s process-tree --force-overwrite true \
  -o /mnt/workspace/PPU/profile_results/monitor_60s.asysrep \
  --duration 60 $SCHED_PID
```

### 5.1.2 跑并发请求 (trace 才有数据)

trace 期间必须跑请求,否则 trace 数据极少:

```bash
nohup python3 -u -c "
import requests, csv, random, concurrent.futures
random.seed(20260625)
with open('/mnt/workspace/datasets/mmbench/mmbench_dev_en.tsv') as f:
    samples = list(csv.DictReader(f, delimiter='\t'))
def run(i):
    s = samples[i]
    requests.post('http://127.0.0.1:8999/v1/chat/completions',
        json={'model':'Qwen3.5-2B','messages':[{'role':'user','content':[
            {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,xxx'}},
            {'type':'text','text':s['question']}]}],
            'max_tokens':256,'temperature':0,'stream':False}, timeout=120)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(run, range(50)))
" &
wait $!
```

### 5.1.3 导出报告

```bash
asys stats -r ppu_op_type_sum -f csv --force-overwrite true -o . \
  /mnt/workspace/PPU/profile_results/monitor_60s.asysrep
asys stats -r ppu_op_kernel_breakdown -f csv --force-overwrite true -o . \
  /mnt/workspace/PPU/profile_results/monitor_60s.asysrep
asys stats -r hggc_kern_exec_trace -f csv --force-overwrite true -o . \
  /mnt/workspace/PPU/profile_results/monitor_60s.asysrep   # 巨大,但有 grid
```

## 5.2 G 组 BF16 (500 samples) profile 摘要

> 数据来源:`../profile_results/g_group_500.asysrep` (已存在,但 profile_results
> 目录不再在 docs/ 中被显式引用,需要时直接读)

| OP Type | Total (ns) | Total Time (%) | Instances | Avg (ns) | Med (ns) |
|---|---|---|---|---|---|
| **GEMM** | 3.96 × 10⁹ | **56.07%** | 168713 | 23 452 | 19 200 |
| **GEMV (gemvt_op)** | (内嵌在 GEMM 中) | — | ~266 000 | ~13 000 | — |
| Element-wise / 杂项 | 剩余占比 | ~30% | — | — | — |
| Attention | (内嵌) | ~10% | — | — | — |

PPU 反向 266 k GEMV calls / 60 s ≈ **4400 calls/s** —— 这反映了 CUDA Graph 在
省时间,所有 decode 路径是 GEMV 占主导。

## 5.3 GEMV 算子 grid → Linear 层映射

> Top GEMV kernel = `gemvt_op<cutlass::bfloat16_t, float, cutlass::bfloat16_t, int4, int4, 32, 4, 1, 2, 1, 1, ...>`

通过 grid 拆分 + kernel 时间精确定位,Top GEMV **实际上混合了 5 个不同 Linear 层**:

| Grid | Weight 形状 | 模型组件 | Launches (60s) |
|---|---|---|---|
| **2048×1×1** | (2048, 2048) | `self_attn.q_proj` + `o_proj` + `GDN.out_proj` | 41 598 |
| **512×1×1** | (2048, 512) | `self_attn.k_proj` + `v_proj` (full attn 6 层 ×2) | 55 464 |
| **3072×1×1** | (2048, 3072) | `GDN.in_proj` 拆 4 块或 MLP gate/up 拆 2 块 | 55 464 |
| **62080×1×1** | (2048, 62080) | `lm_head` 拆 4 块 (vocab=248320/4) | 2 321 |
| **1280×1×1** | (?, 1280) | 未知小分量(可能是 vision encoder)| (少量) |

> **谜团 `1280×1×1`** 不对应已知的 Linear 形状,可能是 vision encoder 内的
> 隐藏 projection;**目前不影响交付**,记录在案待后续调研。

### 复现脚本

```python
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

## 5.4 GEMV vs GEMM microbench (解码路径决定性证据)

| 算子 | M | 耗时 | 加速比 |
|---|---|---|---|
| DeepGEMM `gemv_int8_int8_bf16_nt` (INT8 GEMV) | 1 | **13.0 μs** | **2.15×** baseline |
| Triton fast path (FP16/BF16 GEMM, single token) | 1 | 27.9 μs | 1.0× (baseline) |
| ACEXT `int8_gemm` (INT8 GEMM, decode) | 1 | ~14 μs | ~2× |

> GEMV 在 M = 1..8 区间比 GEMM 快 2× 来自消除 Python dispatch overhead。数值
> 精度 bit-level 一致 (max_diff = 0)。M ≥ 16 时 PPU 张量核强制 BLOCK_M=16,
> GEMV 不再有优势,GC dispatch 路径反而有 padding 开销,需切 ACEXT 或 torch。

## 5.5 profile 复现小贴士

- **ASYS 启动开销 ~3 s**,前 5 秒 trace 数据当作 warm-up 丢掉
- **`hggc_kern_exec_trace` CSV 体积巨大**,单 run ~500 MB;分析用 grep 裁剪
- 跑 60 s trace + 50 并发请求是经验最稳态,小于这个数据会稀疏
- profile 期间禁止 background tuner(tuner 抢占 GPU,把 trace 数据搞乱)

## 5.6 ISA v10 GEMV 算子 profile(Phase 3.2 baseline)

> 实验目录:`/mnt/workspace/ISA/experiments/004_cp_async_gemv/`
> 源码:`/mnt/workspace/ISA/src/kernels/gemv_int8_async.cu`

### 5.6.1 v10 vs deep_gemm microbench

| kernel | K=N=2048 M=1 | K=2048 N=8192 M=1 |
|---|---|---|
| deep_gemm (production) | 30.32 μs | 32.08 μs |
| naive SIMD (Phase 3.0) | 26.72 μs | — |
| **v10 (HBM direct)** | **6.70 μs** | **10.21 μs** |
| speedup vs deep_gemm | **4.5×** | **3.13×** |

正确性:v10 vs fp32 ref **max diff = 0.0**;v10 vs deep_gemm **max diff = 0.0**;
CUDA-graph 内 20 次 replay 全部 max diff = 0.0。

### 5.6.2 v10 算子 profile(K=2048 N=2048 M=1)

| 步骤 | μs | 占比 |
|---|---|---|
| HBM read (8 KB total → 256 B/cycle × 64 SM ÷ 1 block) | 1.5 | 22% |
| compute (16 字节 × 64 lane · 32 lanes / SM × 64 SM → 16 KB / cycle) | 1.0 | 15% |
| warp reduce + 写 bf16 | 0.5 | 7% |
| **launch overhead** | **3.7** | **55%** |

launch overhead 是大头 → **Phase 3.3 persistent kernel 的主要动机**(单 launch
串起 200+ GEMV,把这个 3.7 μs 平摊到所有 GEMV 上)。

### 5.6.3 HBM 有效带宽利用率

| kernel | 有效带宽 | HBM peak (2.7 TB/s) |
|---|---|---|
| deep_gemm | 138 GB/s | **5.0%** |
| naive SIMD | 157 GB/s | 5.7% |
| **v10** | **625 GB/s** | **22.6%** |

v10 距 HBM peak 还差 4×,Phase 3.2.1 cp.async 双 buffer 应该能把 HBM latency
hide 掉一截(目标: 35-45% peak)。Phase 3.3 持久 kernel 跨 op 复用 weight,理论
可把整个 decode step 推到 ~60% peak → 500-700 tok/s。

### 5.6.4 复现命令

```bash
cd /mnt/workspace/ISA/src/sglang_kernel_v10
pip install .                       # 装 v10 wheel
python3 /mnt/workspace/ISA/experiments/004_cp_async_gemv/bench.py
```

或直接 unit test:
```python
import torch
from sglang_kernel_v10 import gemv_int8_async
A = torch.randint(-128, 127, (1, 2048), dtype=torch.int8, device='cuda')
W = torch.randint(-128, 127, (8192, 2048), dtype=torch.int8, device='cuda')
As = torch.ones(1, dtype=torch.float32, device='cuda')
Bs = torch.ones(8192, dtype=torch.float32, device='cuda')
out = torch.empty(1, 8192, dtype=torch.bfloat16, device='cuda')
gemv_int8_async(A, W, As, Bs, out)
```
