# 04. 算子 Profile（asys）

## 4.1 工具

```bash
asys --version
# T-Head Asight Systems version v2.1.1_20260725-15d8b9d
which asys
# /usr/local/PPU_SDK/asight/bin/asys
```

主要子命令：
- `asys attach -t hggc -s process-tree -o <file> --duration <s> <pid>` — 抓 GPU kernel trace
- `asys stats -r <report> -f csv -o . <file>.asysrep` — 导出报告

## 4.2 可用 reports

```
ppu_op_type_sum           # 按 op type 汇总
ppu_op_sum                # 按 op 细分
ppu_op_kernel_breakdown   # 每个 kernel 名称细分
ppu_op_trace              # 每个 op 的时间线
hggc_kern_exec_trace      # 每个 kernel launch（含 grid/block/PID/timing）
hggc_ppu_kern_sum         # 按 kernel 汇总
hggc_ppu_kern_gb_sum      # 按 grid+block 汇总
device_memory_usage_detail # 设备内存使用（需 callstack 配置）
hgtx_*                    # 需要应用主动 push HGTX range（sglang 不推）
```

## 4.3 关键限制

- **Ftrace 不可用**：CPU 调度 trace 关闭
- **Perf 不可用**：CPU backtrace 关闭
- **OS sampling 不可用**：只能用 `process-tree` 模式
- **callstack-group-config**：仅 `device_memory_usage_detail` 支持，kernel trace 不带 Python frame
- **Python frame**：可用 `--python-backtrace hggc` + `--hggcbacktrace hggc` 抓，但只对 memory trace 关联

## 4.4 G 组 60s profile（baseline 之后的最终配置）

```bash
SCHED_PID=$(ps aux | grep "sglang::scheduler" | grep -v grep | awk '{print $2}')
asys attach -t hggc -s process-tree -o profile_results/g_group_60s.asysrep --duration 60 $SCHED_PID

# 同时并发请求压 GPU
nohup python3 -c "
import requests, csv, random, concurrent.futures
random.seed(20260625)
with open('datasets/mmbench/mmbench_dev_en.tsv') as f:
    samples = list(csv.DictReader(f, delimiter='\t'))
def run(i):
    s = samples[i]
    r = requests.post('http://127.0.0.1:9001/v1/chat/completions', json={...})
    return r.json()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(run, range(50)))
" &
```

G 组 profile 关键数据（10 个 mmbench 请求，60 秒）：

| OP Type | Total % | Total ms | Avg ns | 次数 |
|---|---|---|---|---|
| **Summary** | **100.0** | 7057.4 | 8,345 | 845,699 |
| **GEMM** | **73.4** | 5181.9 | 19,414 | 266,915 |
| Native | 21.3 | 1501.2 | 2,795 | 537,066 |
| FlashAttention | 4.9 | 342.9 | 12,338 | 27,792 |
| Triton | 0.4 | 31.4 | 2,254 | 13,926 |

Top 5 kernels（占 GEMM 72.7%）：

| % | calls | avg ns | kernel name |
|---|---|---|---|
| 56.1 | 168,713 | 23,452 | `void gemvt_op<cutlass::bfloat16_t, ..., 32, 4, 1, 2, 1, 1, ...>` |
| 11.6 | 55,464 | 14,782 | `void gemvt_op<cutlass::bfloat16_t, ..., 64, 2, ...>` |
| 5.0 | 41,598 | 8,439 | `void gemvt_op<cutlass::bfloat16_t, ..., 128, 1, ...>` |
| 0.3 | 288 | 80,230 | `gemm_ktype0_..._tile64x256x32x64x64x2_...` |
| 0.2 | 288 | 36,849 | `gemm_ktype0_..._tile64x256x32x64x64x2_splitk2_...` |

Top GEMV 按 grid 拆分：

| Grid | launches | 推测 weight 形状 | 模型组件 |
|---|---|---|---|
| 512×1×1 | 55,464 | (2048, 512) | k_proj / v_proj |
| 2048×1×1 | 41,598 | (2048, 2048) | q_proj / o_proj / GDN.out_proj |
| 1280×1×1 | 13,866 | (?, 1280) | 未知小分量 |
| 3072×1×1 | 55,464 | (2048, 3072) | GDN.in_proj 拆 4 块 / MLP gate-up 拆 2 块 |
| 62080×1×1 | 2,321 | (2048, 62080) | lm_head 拆 4 块 |

Native 21.3% 主要构成（从 kernel_breakdown）：
- `FusedAddRMSNormKernel` (prefill + decode 各层 RMSNorm)
- `silu_and_mul` / `fused_silu_mul` (LLaMA-style SwiGLU)
- `causal_conv1d_fwd_kernel` (GDN 的卷积扫描)

FlashAttention 4.9% 全是 `FlashAttnFwdSm80`。

Triton 0.4%：`_triton_mrope_forward_fused`（MRoPE）+ 部分 GDN fallback。

## 4.5 INT8 60s profile（对比）

```bash
# 杀掉 G 组，启动 INT8 服务器
nohup python3 -u -m sglang.launch_server \
  --model-path ./quant/Qwen3.5-2B-W8A8 \
  --host 127.0.0.1 --port 9001 \
  --trust-remote-code --tp-size 1 \
  --attention-backend triton \
  --enable-multimodal \
  --quantization w8a8_int8 \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --page-size 64 \
  --disable-radix-cache \
  > logs/sglang_int8_9001.log 2>&1 &

# 等 ready，profile 同样 60s
asys attach -t hggc -s process-tree --force-overwrite true \
  -o profile_results/int8_60s.asysrep --duration 60 <int8_scheduler_pid>
```

INT8 profile 关键数据（80 个请求，但模型输出乱码，多数 4-13 tokens 就停）：

| OP Type | Total % | Total ms | Avg ns | 次数 |
|---|---|---|---|---|
| Summary | 100.0 | 1004.4 | 10,272 | 97,771 |
| GEMM | 61.4 | 616.7 | 28,450 | 21,677 |
| Native | 37.6 | 377.5 | 5,113 | 73,814 |
| Triton | 1.0 | 10.2 | 4,473 | 2,280 |

Top kernels（INT8）：
- 11.4% 3687 calls 31,117ns `gemm_ktype0_..._tile64x128x32x32x64x2_...`
- 5.4% 216 calls 251,727ns `void cutlass::Kernel<deep_gemm::GemmKernel<aiu::gemm::threadblock::MmaMultistage...>`
- 4.0% 78 calls 509,714ns `gemm_ktype0_..._tile16x64x16x16x128x4_...`
- 3.8% 1152 calls 32,921ns `gemm_ktype0_..._tile128x128x32x64x64x2_...`
- 3.2% 1320 calls 23,974ns `deep_gemm::GemmKernel<...>`
- 2.5% 58 calls 435,327ns `gemvt_op<cutlass::bfloat16_t, ...>` ← **残留 bf16 GEMV**
- 0.2% 846 calls 2,860ns `gemvt_op<cutlass::bfloat16_t, ...>` ← **残留 bf16 GEMV**

## 4.6 对比总结

| Metric | bf16 G 组 | INT8+DeepGEMM | Δ |
|---|---|---|---|
| 总 kernel 时间 | 7057.4 ms | 1004.4 ms | -85.8% |
| Kernel launches | 845,699 | 97,771 | -88.4% |
| GEMM share % | 73.4 | 61.4 | -12.0 pp |
| GEMM total | 5181.9 ms | 616.7 ms | -88.1% |
| GEMM launches | 266,915 | 21,677 | -91.9% |
| GEMM **avg/call** | 19,414 ns | 28,450 ns | +46.5% ⚠️ |
| Native share % | 21.3 | 37.6 | +16.3 pp |
| Native total | 1501.2 ms | 377.5 ms | -74.9% |
| Native avg/call | 2,795 ns | 5,113 ns | +82.9% |

⚠️ **直接对比 avg/call 不公平**：
- bf16: 几乎全 decode GEMV (1 token)
- INT8: 主要 prefill GEMM (batch)，因为模型乱码输出短

**真实算力压力**：INT8 只有 bf16 的 14%（因输出短），所以 1004ms vs 7057ms 不能解读为 7× 加速。

## 4.7 导出文件

```
profile_results/
├── g_group_60s.asysrep                          # 90 MB 原始 trace（gitignored）
├── g_group_60s_ppu_op_type_sum.csv              # OP type 汇总
├── g_group_60s_ppu_op_sum.csv                   # 每 op 细分（86 行）
├── g_group_60s_ppu_op_kernel_breakdown.csv      # 每 kernel 名称细分
├── g_group_60s_hggc_kern_exec_trace.csv         # 377 MB 每个 kernel launch（gitignored）
├── g_group_60s_ppu_time_util.csv                # GPU 时间利用率
├── int8_60s.asysrep                             # 15 MB 原始 trace（gitignored）
├── int8_60s_ppu_op_sum.csv                      # 196 KB
├── int8_60s_ppu_op_kernel_breakdown.csv         # 196 KB
├── int8_op_type_sum.csv                         # 773 B
├── g_group_30s_backtrace.asysrep                # 10 MB backtrace 实验（gitignored）
├── g_group_30s_pybt.asysrep                     # 43 MB pybt 实验（gitignored）
├── g_group_30s_pybt_hggc_kern_exec_trace.csv    # 122 MB（gitignored）
└── g_group_30s_pybt_ppu_op_trace.csv            # 183 MB（gitignored）
```