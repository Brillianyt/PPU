# PPU-fused Kernel 框架

> **TL;DR** — 之前手写 CUDA kernel(INT8 GEMV/GEMM)在 inductor 眼中是 **opaque
> 黑盒**,只能在它周围做点对点融合。本框架把这件事工程化了:用
> `@ppu_fused_kernel` 装饰器一站式把 kernel 注册成 `torch.library` custom
> op **+** `torch._inductor.ExternKernelChoice`,并通过 post-grad pass
> 让 inductor 真正看得见(能 fuse prologue/epilogue)。当下没有注册 `*_bias`
> 等融合 op,所以 pass 是 no-op — 但**架构就绪**,下一个手写 CUDA kernel
> 进来就能自动获得 inductor 融合机会。

---

## 1. 为什么需要这个框架

之前的架构(只到 R8 spec):

```
PyTorch model (FX graph)
├── torch.nn.functional.rms_norm      ← inductor 可见,会被 fusion
├── cast(int8)                         ← inductor 可见,会被 fusion
├── gemv_int8_i8i8bf16(...)            ← inductor opaque,黑盒
├── bias_add                           ← inductor 可见,会被 fusion
└── residual                           ← inductor 可见,会被 fusion
```

inductor 能 fuse 1 和 2、4 和 5,但中间的 GEMV 是图断点 — fusion 机会被切断。

加完框架之后:

```
PyTorch model (FX graph)
├── rms_norm         ─┐
├── cast(int8)        │  ← prologue:被 inductor fuse 进一个 Triton kernel
├── gemv_int8_nm      ← extern call(sglang::gemv_int8_i8i8bf16)
└── bias_add         ─┘  ← epilogue:被 inductor fuse 进下一个 Triton kernel

注册了 gemv_int8_i8i8bf16_nm_bias 之后:
└── gemv_int8_nm_bias ← 单个 extern call 包了 GEMV + bias_add
                         inductor 把它当 1 个 op,做 prologue/epilogue 调度
```

---

## 2. 模块结构

```
gemv_optimization/sglang-for-sail/python/sglang/srt/compilation/
├── ppu_kernels.py       # 框架核心:装饰器 + Registry + Pass
├── pass_manager.py       # PostGradPassManager monkey-patch,装 PpuPrologueFusionPass
└── backend.py            # SGLangBackend.inductor_config 默认 = inductor(可 env 覆盖)
```

加上量化层一处扩展:

```
gemv_optimization/sglang-for-sail/python/sglang/srt/layers/quantization/
└── w8a8_int8.py          # 加了 gemv_int8_i8i8bf16_nm / gemm_nt_i8i8bf16_nm
                          # (mutating 旧 API + functional 新 API 并存)
```

---

## 3. 公共 API

### 3.1 `@ppu_fused_kernel` — 一站式注册

```python
from sglang.srt.compilation.ppu_kernels import ppu_fused_kernel

@ppu_fused_kernel(
    name="my_op_nm",                          # custom-op name (→ torch.ops.sglang.my_op_nm)
    output=("out", torch.bfloat16),            # (arg_name, dtype) — 声明返回新 tensor
    output_shape_from="A",                     # 用哪个输入 arg 的 shape 决定输出 shape
    cpp_symbol="sglang::my_op",                # C++ symbol,给 inductor extern path
    epilogue_fusion=("bias_add",),             # 声明可 fusion 的 epilogue 模式
)
def my_op_nm(A, B, C, D) -> torch.Tensor:
    """Functional 实现:分配 out,调手写 CUDA kernel,返回。"""
    out = torch.empty(A.shape[0], D.shape[0], dtype=torch.bfloat16, device=A.device)
    my_cuda_kernel(A, B, C, D, out)
    return out
```

装饰器**自动做三件事**:

1. 注册成 `torch.ops.sglang.<name>.default`(shape/dtype propagation)
2. 注册成 `torch._inductor.ExternKernelChoice`(inductor 调度时知道这是已知 extern)
3. 加进 `PpuKernelRegistry`,post-grad pass 启动时被发现

### 3.2 `PpuKernelRegistry` — 全局注册表

```python
from sglang.srt.compilation.ppu_kernels import PpuKernelRegistry

# 所有已注册 kernel
for name, k in PpuKernelRegistry.all().items():
    print(name, k.cpp_symbol, k.epilogue_fusion)

# 只看声明了 epilogue_fusion 的
for k in PpuKernelRegistry.by_pattern():
    print(k.name, k.epilogue_fusion)
```

### 3.3 `PpuPrologueFusionPass` — post-grad 模式重写 pass

自动通过 `PostGradPassManager` 接入(无需手装):

```python
from sglang.srt.compilation.pass_manager import PostGradPassManager
mgr = PostGradPassManager()
mgr.configure()
# → 自动包含 PpuPrologueFusionPass(只要至少 1 个 kernel 注册了 epilogue_fusion)
```

识别规则(每条 epilogue pattern):

| pattern | 识别 | 重写 |
|---|---|---|
| `bias_add` | `%y = <kernel>; %z = aten.add(%y, %bias)` | `%z = <kernel>_bias(..., %bias)` |
| `residual` | 同上,语义化为 residual add(暂同 bias_add) | 同上 |

新增 pattern:在 `_EPILOGUE_PATTERNS` 字典加一行 + 写对应的 `_rewrite_to_fused` 分支。

---

## 4. 已注册的 kernel 清单

通过 `w8a8_int8.py` 的 `@ppu_fused_kernel` 装饰器:

| name | cpp_symbol | epilogue_fusion | 状态 |
|---|---|---|---|
| `gemv_int8_i8i8bf16_nm` | `sglang::gemv_int8_i8i8bf16` | `bias_add` | ✓ functional 注册成功 |
| `gemm_nt_i8i8bf16_nm` | `sglang::gemm_nt_i8i8bf16` | `bias_add` | ✓ functional 注册成功 |

两个 kernel 都注册成 `ExternKernelChoice`,在 inductor scheduler 里是已知 extern op。

**没注册的真融合 op**(待后续工作):

| name | 应该做的事 | 当前状态 |
|---|---|---|
| `gemv_int8_i8i8bf16_nm_bias` | GEMV + bias_add 合一 CUDA kernel(走 SAIL TensorCore + epilogue bias_add) | ❌ 没写 — 需要重写 SAIL TensorCore 级 kernel |
| `gemm_nt_i8i8bf16_nm_bias` | 同上(GEMM 版) | ❌ 没写 |

PpuPrologueFusionPass 会自动跳过没注册的真融合 op(debug log 提示),**不伪造 stub**。

---

## 5. 使用方式

### 5.1 默认 — 框架自动工作

```bash
# 跟普通 R8 启动一样,框架被动生效(ExternKernelChoice 已注册)
PPU_SDK=/usr/local/PPU_SDK python3 benchmark_public.py \
  --model-path /mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive \
  --backend sglang --num-samples 100 --warmup-samples 8 \
  --output result_dev_en_r8_100.json
```

启动日志会显示:

```
INFO pass_manager.py: PpuPrologueFusionPass installed (covers 2 kernels)
INFO ppu_kernels.py: ppu_fused_kernel: registered gemv_int8_i8i8bf16_nm as ExternKernelChoice (symbol=sglang::gemv_int8_i8i8bf16)
INFO ppu_kernels.py: ppu_fused_kernel: registered gemm_nt_i8i8bf16_nm as ExternKernelChoice (symbol=sglang::gemm_nt_i8i8bf16)
```

### 5.2 关掉 framework(回退 R8 spec 行为)

如果你想**完全回退**到 R8 spec 的非框架行为(framework 也别跑),把两类 patch 不部署即可:

- 不要 cp `gemv_optimization/.../compilation/{ppu_kernels,pass_manager,backend}.py`
- 不要 cp `w8a8_int8.py`(会让 `gemv_int8_i8i8bf16_nm` 不存在但 GEMV 旧 API 还在)

注意:`evaluation_wrapper.py` 的 `piecewise_cuda_graph_compiler=inductor` 默认值
**和 framework 无关** — 即使 framework 没装,`inductor` 还是默认(只是没有
prologue-fusion 增强)。要同时回退 inductor:

```bash
SGLANG_PIECEWISE_CUDA_GRAPH_COMPILER=eager \
PPU_SDK=/usr/local/PPU_SDK python3 benchmark_public.py ...
```

### 5.3 加新 hand-written CUDA kernel

Step 1:用 `@ppu_fused_kernel` 装饰器包一层

```python
@ppu_fused_kernel(
    name="my_op_nm",
    output=("out", torch.bfloat16),
    output_shape_from="A",
    cpp_symbol="sglang::my_op",
    epilogue_fusion=("bias_add",),  # 声明后可被 pattern pass 融合
)
def my_op_nm(A, B, C, D):
    out = torch.empty(A.shape[0], D.shape[0], dtype=torch.bfloat16, device=A.device)
    my_cuda_kernel(A, B, C, D, out)
    return out
```

Step 2:把对应真融合 op(例如 `my_op_nm_bias`)也注册成 custom op + ExternKernelChoice。
这就是手写 CUDA kernel 真正参与的入口 — pass 会自动重写 `my_op_nm + add` → `my_op_nm_bias`。

Step 3:把这两段代码加到 `gemv_optimization/sglang-for-sail/python/sglang/...`,
同步流程见 `docs/08_SYNC.md`。

---

## 6. 局限与未来工作

### 6.1 已知的架构 trade-off(非 bug)

- **Decode path 在 wrapper 里多 ~10μs/step 的 Python 开销**(`assert_size_stride` /
  `assert_alignment` / `empty_strided_cuda` / `get_raw_stream`),导致 TPS 比
  eager baseline 低 ~5%。
- 这跟 vLLM、TensorRT-LLM 同样的设计取舍 — prefill 收益 >> decode 损失,
  对 VLM workload(image prefill 主导总耗时)净赚。
- 根本性修复:decode path 走 eager、prefill path 走 inductor(架构级改动)。

### 6.2 max_autotune 当前是 opt-in

`backend.py` 默认 `max_autotune=False`,因为:
- 启动变慢(每个 shape benchmark 几套 Triton kernel)
- 在 PPU 这种非 NVIDIA 架构上,autotune 可能挑到不如手调 SAIL TensorCore 的 kernel
- 当前实测效果中性(TPS +0.6%,TTFT -2%)

要开:`SGLANG_PIECEWISE_INDUCTOR_AUTOTUNE=1`

### 6.3 真正"自然接轨"的下一公里

要把 `gemv_nm_bias` 真融合 CUDA kernel 写出来,需要:

1. 写一份 SAIL TensorCore + mma + epilogue bias_add 的 INT8 GEMV CUDA kernel
2. 注册成 `torch.ops.sglang.gemv_int8_i8i8bf16_nm_bias`
3. PpuPrologueFusionPass 自动发现 + 自动重写 `gemv_nm + add` → `gemv_nm_bias`
4. inductor 把 `gemv_nm_bias` 作为 extern kernel 调用,生成更紧凑的 wrapper

这一步是真正性能收益的来源,但需要重写 SAIL TensorCore 级 CUDA 代码,不是
"调一下"能完成的。需要 PPU kernel 工程师用 profiling + autotune 工具产出。

---

## 7. 设计决策记录

| 决策 | 选择 | 理由 |
|---|---|---|
| Mutating 还是 functional? | **functional(返回新 tensor)** | mutating op inductor 没法 fuse prologue/epilogue |
| 框架装在哪个 module? | `sglang/srt/compilation/ppu_kernels.py` | 跟 pass_manager、backend 同层,编译关注点 |
| 注册时机? | import 时 decorator 立即注册 | 启动时就绪,无 lazy init 复杂度 |
| Pass 装入 PostGradPassManager 的方式? | pass_manager.py 末尾 monkey-patch `configure()` | 避免 circular import(ppu_kernels 需要 pass_manager 的 SGLangInductorPass 基类) |
| 找不到 fused op 时 pass 行为? | debug log + skip(不插 stub) | 工程化 — 不假装不存在的东西存在 |
| max_autotune 默认值? | False(opt-in) | 安全 — inductor 在 PPU 上可能挑到不如手调 kernel |
| 跟旧 mutating API 兼容? | 完全兼容(mutating 旧 API 保留) | 不破坏现有 caller |
| framework 是 opt-in 还是默认装? | **默认装**(装饰器一旦 import 立即注册,pass 启动时自动发现) | "需要"才装 = 没人会记得装,默认装 + 缺 fused op 时 no-op 是最稳的设计 |

---

## 8. ISA v10 GEMV — 本框架的"下一个 kernel"参考实现

> ISA Phase 3.2 v10 (`/mnt/workspace/ISA/src/sglang_kernel_v10/`) 是按本框架模式
> 集成的**手写 CUDA kernel**——走的是工程化正解路径(`pip install .` wheel 而非
> JIT hack)。虽然当前 `mutating` API 集成,没用到本框架的 `@ppu_fused_kernel`
> 装饰器层,但 ISA v10 展示了**集成到 sglang decode path 的完整工程闭环**:
>
> | 环节 | ISA v10 的做法 | 与本框架的关系 |
> |---|---|---|
> | kernel 形态 | `gemv_int8_async_impl` 函数化模板 | 未来用 `@ppu_fused_kernel` 包装即可 |
> | dispatcher | `w8a8_int8.gemv_int8_i8i8bf16` 一个 if-else | 与现有 GEMV/GEMM 分派共用 |
> | 集成模式 | `pip install sglang_kernel_v10` wheel | 与本框架的 install-time compile 模式一致 |
> | cache 策略 | `_ISA_V10_W_CACHE[B.data_ptr()]` 永久 | 不影响框架的 cache 模型 |
> | CUDA Graph 兼容 | **预编译 wheel**(无 JIT 编译) | 与框架要求一致 — 不能在 CUDA Graph capture 期间编译 |
>
> **端到端结果**(100 样本 mmbench_dev_en, opt-in):
> - TPS 411.6 → **421.1** (+2.3%)
> - TTFT 72.7 → 75.3 (avg, +3.6%)
> - acc 0.73 = 0.73 (无变化)
> - 官方公式净评分:约 **-0.4%** (微负,TTFT 回归抵消 TPS 收益)
>
> **判定**: 默认关闭, opt-in (`SGLANG_ISA_V10_ENABLE=1`)。若 Phase 3.3 persistent
> kernel 上线,把 v10 函数化集成进持久 kernel(消 launch overhead 同时保留
> HBM-direct SIMT 路径),净评分会回正。详见 `技术方案报告_完整版.md §4.4 / §5.5 / §12 F9-F10 / §13.3`。

---

## 9. 参考

- `sglang/srt/compilation/pass_manager.py` — PostGradPassManager 基类
- `sglang/srt/compilation/inductor_pass.py` — SGLangInductorPass 基类
- `torch._inductor.select_algorithm.ExternKernelChoice` — extern kernel 注册原语
- `torch.fx.replace_pattern` — 模式重写工具(暂未用,future work)
- PPU RoPE hook(`inductor_hook_PPU/hook/ppu_rotary_hooks.py`) — custom_op 包装的参考实现
