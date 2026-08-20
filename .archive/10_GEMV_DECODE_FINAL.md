# 10. GEMV Decode 算子与最终量化方案(终章)

> 日期:2026-08-16
> 目标:Qwen3.5-2B INT8 量化 + PPU 原生 GEMV decode 算子 + prefill 反量化,端到端吞吐 +8.8%,TTFT 接近 baseline,准确率持平。

---

## 1. 最终成果(§36)

| 配置 | tok/s | TTFT(ms) | accuracy |
|---|---|---|---|
| §27 baseline(AMQ-QKV + 无 GEMV) | 358 | ~63 | 0.75 (15/20) |
| §35 S2 + GEMV | 388 | 106.7 | 0.75 |
| **§36 S2 + GEMV + PREFILL_BF16** | **389.4** | **72.1** | **0.80 (16/20)** |

三项收益:
- 吞吐 **+8.8%**(358 → 389)
- TTFT **大幅回落**(107 → 72,接近 baseline 63)
- 准确率持平(0.80 vs 0.75,+1 样本,需大样本确认是否真实提升)

## 2. 三个关键优化点

### 2.1 AMQ 量化策略(S2)

- 工具:QLean(主办方工具)
- 策略:`qwen3.5-2b_amq_s2_aggressive.yaml`
- int8 层 126 个,保护误差最高的敏感层(`in_proj_a/b`、`linear_attn.out_proj`、19/23 qkv)
- 精度:30 样本与 BF16 无损;比 S3 全量化(186 层)更稳

### 2.2 GEMV Decode 算子

- 内核:复用 DeepGEMM `gemvt.cuh` 的 `Gemvt<int8_t>`(PPU 原生 AIU 异步搬运 + multistage)
- 入口:`deep_gemm.gemv_int8_int8_bf16_nt`
- 微基准:M=1 时 **13.0us vs Triton fast path 27.9us,2.15×**
- 端到端价值:高 INT8 覆盖率下绕开 `gemm_nt_i8i8bf16` 的 Python dispatch 开销,S3-full 上 **+36%**(273→372)
- 数值与 GEMM **bit 级一致**

### 2.3 Prefill BF16 反量化(方向 B)

- 问题:int8 GEMM 在 prefill(M 大)比 BF16 GEMM 慢(+46%),量化层越多 prefill 越慢
- 方案:prefill(M>8)时把 int8 权重临时反量化成 BF16,走 `F.linear`(BF16 GEMM);decode(M≤8)仍走 int8 GEMV
- 权重始终只存 int8 一份,反量化结果懒缓存
- 效果:TTFT 从 106.7ms 回落到 72.1ms

## 3. 分派逻辑(核心代码)

```python
# w8a8_int8.py apply()
if SGLANG_SAIL_PREFILL_BF16 and _is_ppu and bias is None and x.shape[0] > M_MAX:
    # prefill: 反量化 + BF16 GEMM(快)
    w_bf16 = layer._w_bf16_cache or dequant(layer.weight)
    return F.linear(x, w_bf16)

x_q, x_scale = per_token_quant_int8(x)

if SGLANG_SAIL_DEEPGEMM_DENSE and _is_ppu and bias is None:
    if SGLANG_SAIL_USE_GEMV and x_q.shape[0] <= M_MAX:
        # decode: int8 GEMV(Gemvt,快)
        gemv_int8_i8i8bf16(...)
    else:
        gemm_nt_i8i8bf16(...)
```

## 4. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `SGLANG_SAIL_USE_GEMV` | False | decode 小 M 走自研 GEMV |
| `SGLANG_SAIL_GEMV_M_MAX` | 8 | GEMV/GEMM 分派阈值 |
| `SGLANG_SAIL_PREFILL_BF16` | False | prefill 反量化走 BF16 GEMM |

## 5. 启动命令(最终)

```bash
export PPU_SDK=/usr/local/PPU_SDK
export SGLANG_SAIL_DEEPGEMM_DENSE=1
export SGLANG_SAIL_USE_GEMV=1
export SGLANG_SAIL_GEMV_M_MAX=8
export SGLANG_SAIL_PREFILL_BF16=1
export SGLANG_SAIL_FLA_CUDA=1

python3 -m sglang.launch_server \
  --model-path /mnt/workspace/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --host 0.0.0.0 --port 8999 \
  --tp-size 1 --dtype bfloat16 \
  --attention-backend fa3 \
  --disable-radix-cache --page-size 64 \
  --quantization w8a8_int8 \
  --enable-multimodal --trust-remote-code \
  --mem-fraction-static 0.85
```

## 6. 修改文件清单

见 `PPU/gemv_optimization/`(按原仓库路径保留):

- `DeepGEMM-for-sail/deep_gemm/jit_kernels/gemv_int8.py`(新增)
- `DeepGEMM-for-sail/deep_gemm/jit_kernels/__init__.py`(导出)
- `DeepGEMM-for-sail/deep_gemm/__init__.py`(导出)
- `sglang-for-sail/python/sglang/srt/layers/quantization/w8a8_int8.py`(GEMV custom op + apply 分派 + prefill 反量化)
- `sglang-for-sail/python/sglang/srt/environ.py`(3 个环境变量)
- `sglang-for-sail/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`(GEMV kernel 枚举 + warmup executor)
- `sglang-for-sail/python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`(gemv_nt_i8i8bf16 wrapper)

## 7. 探索历程(来时路)

1. **AMQ 敏感度分析**:175 层 INT8 重建误差,`in_proj_a/b` 最敏感(0.0154)
2. **量化策略**:v3(42 层)→ S2(126 层,推荐)→ S3(186 层,全量化)
3. **发现**:SGLang 0.5.10 的 w8a8_int8 在 PPU 有 bug(输出 garbage),0.5.13 修复
4. **GEMV 探索**:微基准 Gemvt 13us vs Triton 27.9us(2.15×),但 v3 低覆盖下端到端无收益
5. **关键转折**:S3 高覆盖下 GEMV +36%,证明 GEMV 价值在"高 int8 覆盖率下绕开 dispatch"
6. **方向 B**:prefill 反量化 BF16,解决 int8 GEMM 在 prefill 慢的问题,三项全胜

## 8. 待验证

- 准确率 0.80 是真实提升还是 ±1 样本波动(需 50/100 样本确认)
- 更大样本下的吞吐/TTFT 稳定性
