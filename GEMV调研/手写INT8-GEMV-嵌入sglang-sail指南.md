# 手写 INT8 GEMV 算子嵌入 sglang-sail + DeepGEMM 集成指南

> 依据:sglang-for-sail v0.5.13 源码(`w8a8_int8.py`、`deep_gemm_wrapper/entrypoint.py`、`int8_kernel.py`)+ 前述 GEMV 调研。
> 目标:自研 INT8 GEMV 内核(M=1/小 M 的 int8×int8 矩阵向量乘,输出 BF16),"最完美"地嵌入 sglang-sail 推理链路,与 DeepGEMM 共存。

---

## 1. 定位:你的 GEMV 该接在哪

sglang-sail 的 W8A8 INT8 线性层执行链(源码核实):

```
W8A8Int8LinearMethod.apply(layer, x, bias)
 ├─ CPU/AMX 路径
 ├─ x_q, x_scale = per_token_quant_int8(x)          # 激活恒为 int8 动态 per-token
 ├─ [PPU + no bias + SGLANG_SAIL_DEEPGEMM_DENSE] → gemm_nt_i8i8bf16(x_q, x_scale, weight.t(), weight_scale, out)
 ├─ [SGLANG_SAIL_USE_ACEXT_CUDA]                   → acext_int8_gemm(...)
 └─ fallback                                       → sgl_kernel.int8_scaled_mm
```

- **DeepGEMM 的 `gemm_nt_i8i8bf16` 在 decode(M 很小)时并非最优**(它按 GEMM tile 调优);你的 GEMV 就是为 M=1/小 M 场景补这块。
- **最完美嵌入点 = `apply()` 里 deep_gemm 分支的前面**:M 小于阈值时走你的 GEMV,其余走 DeepGEMM。这样**零模型改动**自动覆盖所有 decode 线性层(qkv_proj / o_proj / gate / up / down),权重布局零转换直接复用。
- 不需要碰 MQA logits(那是 DeepSeek-V4 索引器专用,和你的场景无关)。

## 2. 数据/权重契约(必须与 w8a8_int8 完全一致)

| 张量 | 形状 | dtype | 说明 |
| --- | --- | --- | --- |
| 激活 `x_q` | (M, K),M=1/小 | int8 | 复用 `int8_kernel.per_token_quant_int8(x)` 生成 |
| 激活 scale `x_scale` | (M,) | fp32 | per-token 动态 |
| 权重 `w_q` | (N, K) **行主序** | int8 | `layer.weight`(fused 模块 N = Σ partition 宽) |
| 权重 scale `w_scale` | (N,) | fp32 | per-channel(`layer.weight_scale`) |
| 输出 `out` | (M, N) | bf16 | 结果 |

- 对齐建议:N、K 按 16 对齐(DeepGEMM/acext 均要求);若模型 K 不满足,kernel 内 pad 或权重预处理时 pad。
- fused 模块注意:`qkv_proj` 的 N 是 q+k+v 宽度之和,`weight_scale` 也是按 partition 拼好的 per-channel——你的 GEMV 把 N 当一维处理即可,无需拆。

## 3. 集成代码骨架(基于真实源码)

### 3.1 新增自定义算子(模仿 `gemm_nt_i8i8bf16` 的注册模式)

新建 `sglang/srt/layers/gemv_wrapper.py`:

```python
from sglang.srt.utils.custom_op import register_custom_op

def gemv_int8_i8i8bf16_fake(A, As, B, Bs, C):
    # torch.compile / CUDA graph 需要的 fake 实现:只声明副作用,不计算
    return

@register_custom_op(mutates_args=["C"], fake_impl=gemv_int8_i8i8bf16_fake)
def gemv_int8_i8i8bf16(A, As, B, Bs, C):
    # A: (M,K) int8; As: (M,) fp32; B: (N,K) int8 行主序; Bs: (N,) fp32; C: (M,N) bf16
    my_gemv_int8_kernel(A, As, B, Bs, C)   # ← 你的手写 kernel(绑定/调 .so 或 triton)
```

要点:
- `@register_custom_op(mutates_args=["C"], fake_impl=...)` 与现有 `gemm_nt_i8i8bf16` 完全同款,天然支持 torch.compile 与 CUDA graph 捕获;
- kernel 内部**不要有 Python 分支/动态 alloc**,保持图捕获友好;
- 若你的 kernel 是 CUDA/PPU 扩展(.so),在此处做 pybind 调用。

### 3.2 在 `w8a8_int8.py` 的 `apply()` 加分派(推荐形态)

```python
if envs.SGLANG_SAIL_DEEPGEMM_DENSE.get() and _is_ppu and bias is None:
    if (
        envs.SGLANG_SAIL_USE_GEMV.get()
        and x_q.shape[0] <= envs.SGLANG_SAIL_GEMV_M_MAX.get()
    ):
        out = torch.empty((x_q.shape[0], layer.weight.t().shape[0]),
                          dtype=torch.bfloat16, device=x_q.device)
        gemv_int8_i8i8bf16(x_q, x_scale, layer.weight.t(), layer.weight_scale, out)
    else:
        out = torch.empty((x_q.shape[0], layer.weight.t().shape[0]),
                          dtype=torch.bfloat16, device=x_q.device)
        gemm_nt_i8i8bf16(x_q, x_scale, layer.weight.t(), layer.weight_scale, out)
    return out.to(x.dtype)
```

### 3.3 环境变量(仿 `SGLANG_SAIL_DEEPGEMM_DENSE` 模式,加进 `sglang/srt/environ.py`)

```
SGLANG_SAIL_USE_GEMV: bool = 0     # 默认关,开=decode 小 M 走自研 GEMV
SGLANG_SAIL_GEMV_M_MAX: int = 8    # M 阈值,超过回 DeepGEMM
```

默认关闭 + 阈值回退 = 风险隔离,出问题只需关环境变量。

## 4. 集成要求清单(逐项核对)

| 项 | 要求 |
| --- | --- |
| 权重格式 | int8 行主序 + fp32 per-channel scale,直接消费 w8a8_int8 权重(零转换) |
| 激活量化 | 复用 `per_token_quant_int8`(保持与 deep_gemm 路径数值一致) |
| custom op | `register_custom_op(mutates_args=["C"], fake_impl=...)`,支持 torch.compile/CUDA graph |
| 对齐 | N/K %16(必要时 pad),与 acext/deep_gemm 契约一致 |
| CUDA graph | decode 捕图时无 Python 控制流、无内部动态 alloc |
| TP | 逐层调用,`weight_scale` 的 TP 切分已由 `ChannelQuantScaleParameter` 加载,天然兼容 |
| bias | 先支持 `bias=None`(deep_gemm 分支同款限制);有 bias 回退 deep_gemm |
| 回退 | env 默认关;M 超阈值/异常自动回 `gemm_nt_i8i8bf16` |
| 数值验证 | 与 `gemm_nt_i8i8bf16` 对同一权重/输入做逐位对比(M=1,2,4,8 × 典型 N,K) |
| UT | 加入 sglang 测试(仿 `test_attention.py` / w8a8 相关 UT) |

## 5. 验证步骤

1. 单元验证:单卡、固定权重,`gemv_int8_i8i8bf16` vs `gemm_nt_i8i8bf16` 输出 bit 级对齐(M=1/2/4/8);
2. 开关验证:`SGLANG_SAIL_USE_GEMV=1` 启动 Qwen3.5-2B,日志确认 decode 层走 GEMV;
3. CUDA graph 验证:开启 `--cuda-graph-max-bs` 场景不炸、可捕获;
4. 精度:量化模型精度与 deep_gemm 路径一致(应逐位一致,因为是同一 int8 数值);
5. 性能:对比 decode TTFT/TPOT,确认 GEMV 在 M=1 场景优于 `gemm_nt_i8i8bf16`(这才是手写意义所在)。

## 6. 注意事项

- **810E 只有 INT8**:你的算子只做 int8×int8→bf16;FP8/FP4 GEMV 留给 M890(可日后按 `gemm_nt_f8f8bf16` 同款扩展)。
- **别动 MQA logits 路径**:那是 DeepSeek-V4 索引器专用,与 Qwen3.5-2B 无关;若未来要做,直接包装 `deep_gemm.*_mqa_logits` 即可(接口见 GEMV 调研笔记 §3.2)。
- **若想进 wrapper 体系**:也可以把 GEMV 加进 `deep_gemm_wrapper/entrypoint.py`(配 `DeepGemmKernelType` + `compile_utils` hook + `deep_gemm_tuner`),与 `gemm_nt_i8i8bf16` 并列——更"原生",但工程量更大;先按 §3 的自定义 op 路线落地,再考虑升级。
- 与上游升级的兼容:改动集中在 `w8a8_int8.py`(本地 fork 文件),升级 sglang 时注意合入冲突。
