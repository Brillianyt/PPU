# 07. GEMV → 模型 Linear 层完整映射

> 通过 asys hggc_kern_exec_trace 的 grid/block 拆分，把每个 GEMV kernel 反推到模型具体 Linear 层。

## 7.1 方法

```bash
# 1. profile 60s
asys attach -t hggc -s process-tree -o profile_results/g_group_60s.asysrep --duration 60 <scheduler_pid>

# 2. 导出每个 kernel launch
asys stats -r hggc_kern_exec_trace -f csv --force-overwrite true -o . profile_results/g_group_60s.asysrep
# → g_group_60s_hggc_kern_exec_trace.csv (377 MB, gitignored)

# 3. 按 grid 分组 + kernel 模板
python3 << 'EOF'
import csv
target = "void gemvt_op<cutlass::bfloat16_t, float, cutlass::bfloat16_t, int4, int4, 32, 4, 1, 2, 1, 1"
grids = {}
durs = []
with open('profile_results/g_group_60s_hggc_kern_exec_trace.csv') as f:
    for row in csv.DictReader(f):
        if target in row['Kernel Name']:
            g = row['GridXYZ']
            grids[g] = grids.get(g, 0) + 1
            durs.append(int(row['Kernel Dur (ns)']))
print(f"Total: {sum(grids.values())}")
for g, c in sorted(grids.items(), key=lambda x: -x[1]):
    print(f"  grid={g}: {c}")
EOF
```

## 7.2 完整 grid 分布（top GEMV tile 32×4）

```
grid=512×1×1     55,464 launches    # (2048, 512) Linear
grid=2048×1×1    41,598 launches    # (2048, 2048) Linear
grid=1280×1×1    13,866 launches    # 未知小分量
grid=3072×1×1    55,464 launches    # (2048, 3072) Linear (GDN.in_proj 拆 4 块 / MLP 拆 2 块)
grid=62080×1×1    2,321 launches    # lm_head 拆 4 块
─────────────────────────────────
total           168,713 launches
```

## 7.3 逐个映射

### Grid 2048×1×1 → q_proj / o_proj / GDN.out_proj

```python
# Qwen3.5-2B config.json
text_config = {
    "num_attention_heads": 8,
    "head_dim": 256,
    "num_key_value_heads": 2,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
}

# hidden_size = 2048

# Full attention layer (6 层，每 4 层一次，full_attention_interval=4)
self_attn.q_proj: Linear(2048 → 8*256=2048)        # GEMV (2048, 2048) ✓
self_attn.k_proj: Linear(2048 → 2*256=512)         # GEMV (2048, 512)
self_attn.v_proj: Linear(2048 → 2*256=512)         # GEMV (2048, 512)
self_attn.o_proj: Linear(2048 → 2048)              # GEMV (2048, 2048) ✓

# GDN linear attention layer (18 层)
GDN.in_proj:  Linear(2048 → 4*2048 + 16 + 16)      # GEMV (2048, 8224)
                                                     # 拆为 4 块可能: q, k, v, z 各 2048
                                                     # 或 3 块: q+k, v+z, b+a
GDN.out_proj: Linear(2048 → 2048)                   # GEMV (2048, 2048) ✓

# MLP (所有 24 层)
mlp.gate_proj: Linear(2048 → 6144)                 # GEMV (2048, 6144)
mlp.up_proj:   Linear(2048 → 6144)                 # GEMV (2048, 6144)
mlp.down_proj: Linear(6144 → 2048)                  # GEMV (2048, 6144) (转置后形状一样)
```

24 层 forward pass 中，每层有 **~3 个 (2048, 2048) GEMV**：
- 6 层 full attn × (q_proj + o_proj) = 12 个
- 18 层 GDN × (out_proj) = 18 个
- 合计 ~30 个 per forward

41,598 / 30 ≈ **1387 个 forward passes per 60s** ≈ 23 forward/sec（合理）

### Grid 512×1×1 → k_proj / v_proj

只在 6 层 full attention 有，per forward 12 个：
- 6 层 × (k_proj + v_proj) = 12 个

55,464 / 12 ≈ **4622 个 forward passes per 60s** ≈ 77 forward/sec

⚠️ 这里两个 grid (2048 和 512) 给出不同的 forward pass count（23 vs 77），可能是统计/截断误差。

### Grid 3072×1×1 → GDN.in_proj 拆 4 块 OR MLP gate/up 拆 2 块

可能性 A：GDN.in_proj 拆 4 块，每块 2048 输出
- 18 层 × 4 块 = 72 per forward
- 55,464 / 72 ≈ 770 forward passes
- 但 GDN.in_proj 的实际输出是 ~8376（含 b/a），不太可能整除 3072

可能性 B：MLP gate/up 拆 2 块，每块 3072 输出
- 24 层 × 2 (gate+up) × 2 块 = 96 per forward  
- 55,464 / 96 ≈ 577 forward passes
- 6144 = 2 × 3072 ✓ 数学上匹配
- 但 6144 通常不拆

可能性 C：GDN.in_proj 包含 q+k+v+z 4 块，每块 2048 输出（不含 b+a）
- 4 × 2048 = 8192 (接近 8376)
- 18 层 × 4 块 = 72 per forward
- 55,464 / 72 ≈ 770 forward passes

最可能：**A 或 C**（GDN.in_proj 拆分）

### Grid 62080×1×1 → lm_head 拆 4 块

```python
vocab_size = 248320
248320 / 4 = 62080  ✓
```

每个 decode token 调一次 lm_head，被 CUDA graph 拆为 4 个 kernel launch：
- 1 forward × 1 token × 4 = 4 per forward
- 但有连续 batching：假设 batch=8，8 × 4 = 32 per forward
- 2,321 / 32 ≈ 73 forward passes per 60s（远低于前面的 23-77）

⚠️ 数量级不对。可能 lm_head 在 decode batch 中并不是每 token 都全跑（部分用 speculative 或 cache）。

实际上：CUDA Graph 抓的是 batched decode，lm_head 可能被合并为 1 个 launch。

### Grid 1280×1×1 → 未知小分量

1280 = 16 × 80 = 16 heads × 80 dim? 但 Qwen3.5 GDN head_dim = 128。
1280 也可能 = 5 × 256 (full attention per-head partial)
或 vision encoder（hidden_size=1024, 1280 ≠ 1024 × n）

**推测**：vision encoder 的某个 Linear（如 `text_model.vision_tower.blocks.*.attn.qkv` 是 3×hidden=3072），但 1280 ≠ 3072。可能 vision encoder 的 `mlp.fc1` 拆 2 块 = 4096/2 = 2048，也不对。

最大可能：**1280 = 16 heads × 80 dim**（某个 partial projection）或 **1280 = 1024 + 256**（residual 加法后的 Linear）

→ 不确定，需要源码深挖或 GPU 上实测排除。

## 7.4 量化 ROI 排名

| 组件 | 形状 | GEMV calls | 单 call 时间 | 量化收益 |
|---|---|---|---|---|
| **q_proj + o_proj + GDN.out_proj** | (2048, 2048) | 41,598 | 23,452ns | ⭐⭐⭐⭐⭐ 最高 ROI |
| **k_proj + v_proj** | (2048, 512) | 55,464 | ~14,000ns | ⭐⭐⭐⭐ |
| **GDN.in_proj** | (2048, ~8224) | 55,464 | ~32,000ns | ⭐⭐⭐ 难量化（含 b/a） |
| **mlp.gate/up/down** | (2048, 6144) | (推测 ~55k 包含在 GEMM 但被更大 kernel 覆盖) | 70,000ns | ⭐⭐⭐ |
| **lm_head** | (2048, 62080) | 2,321 | ~441,000ns | ⭐⭐ 单次延迟大但频次低 |

但所有这一切都受制于：**W8A8 INT8 量化会掉 19pp 准确率 → 当前不可用**。

只有当有更好的量化方案（如 W4A16 权重量化 + smoothquant 激活量化 + 重新校准）才有可能既不掉点又加速。

## 7.5 不在 GEMM 里的时间

| OP | 占比 | 时间 |
|---|---|---|
| FlashAttention (full attention) | 4.9% | 342.9 ms |
| RMSNorm + SwiGLU + causal_conv1d (Native) | 21.3% | 1501.2 ms |
| MRoPE + Triton fallback | 0.4% | 31.4 ms |

- FlashAttention 4.9%：全部 `FlashAttnFwdSm80`，fa3 已经是最优，没空间
- Native 21.3%：RMSNorm、SiLU、causal_conv1d 都是 elementwise，理论上可以融合到 GEMM 里省掉，但需要改 sglang 源码
- Triton 0.4%：MRoPE 不可替代

→ 如果有 sglang 源码修改能力，可以考虑 **fuse RMSNorm into previous GEMM**（写一个 RMSNormLinear 算子），理论上能省 5-10% 时间。