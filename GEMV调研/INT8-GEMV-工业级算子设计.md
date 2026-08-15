# INT8 GEMV 工业级算子设计(DeepGEMM-for-sail 原生实现)

> 目标:为 M=1/小 M 的 decode 场景,在 DeepGEMM-for-sail 内新增与 `gemm_int8_int8_bf16_nt` 同等级的
> `gemv_int8_int8_bf16_nt`,使用 PPU 原生 AIU/TSM 数据结构,接入 JIT 编译/缓存/tuner 体系,
> 再由 SGLang w8a8_int8 按 M 阈值分派。

---

## 1. 结构位置(与 GEMM 并列,不是外面打补丁)

```
DeepGEMM-for-sail/
├── deep_gemm/
│   ├── __init__.py                          # 导出 gemv_int8_int8_bf16_nt
│   ├── jit_kernels/
│   │   ├── gemm_int8.py                     # 现有 GEMM JIT(Python 层)
│   │   └── gemv_int8.py                     # ★ 新增 GEMV JIT(Python 层)
│   └── include/deep_gemm/
│       ├── int8_gemm.cuh                    # 现有 GEMM CUDA 模板
│       └── int8_gemv.cuh                    # ★ 新增 GEMV CUDA 模板
└── csrc/jit_kernels/impls/
    ├── int8_gemm.hpp                        # 现有 GEMM C++ Runtime(pybind)
    └── int8_gemv.hpp                        # ★ 新增 GEMV C++ Runtime(pybind)

SGLang:
python/sglang/srt/layers/quantization/w8a8_int8.py
  └─ apply(): 在 deep_gemm 分支前, M <= SGLANG_SAIL_GEMV_M_MAX 时走 gemv_int8_int8_bf16_nt
```

**关键**:GEMV 不进 `m_grouped_gemm`、不进 MoE、不进 MQA logits——它是 Dense Linear 的 M=1 特化,
与 `DenseINT8GemmCutlass3Runtime` 同层,但用**独立调度**。

## 2. 数据契约(与 w8a8_int8 完全一致,零转换)

| 张量 | 形状 | dtype | 布局 |
|---|---|---|---|
| `x_q` | (M, K) | int8 | 行主序 |
| `x_scale` | (M,) | fp32 | per-token |
| `w_q` | (N, K) | int8 | **行主序** |
| `w_scale` | (N,) | fp32 | per-channel |
| `out` | (M, N) | bf16 | 行主序 |

数值语义:`out[m,n] = (Σ_k x_q[m,k]·w_q[n,k]) · x_scale[m] · w_scale[n]`,int32 累加,与 GEMM 逐位一致。

## 3. GEMV 计算结构(为什么不用 MMA)

GEMM 的 AIU MMA atom 是 **16×16×32**(`InstructionShape`)。M=1 时,16 行的 MMA 只算 1 行,
15/16 的 Tensor Cell 算力浪费——这正是当前 int8 GEMM 在 decode 低效的根因。

**GEMV 的正确计算结构 = 向量化标量点积,不用 MMA:**

- **一个 warp 处理一行**(N 维一行)
- warp 内 32 线程沿 K 维并行,每线程用 `int8x16`(128bit 向量)加载并做 SIMD 点积
- 累加在 **int32 寄存器**(避免 float 精度损失,与 GEMM 累加器一致)
- K 维循环后 warp 内 reduce,乘 scale,写 1 个 bf16

## 4. PPU 原生数据结构(AIU 异步搬运 + TSM)

- 权重 `w_q`(N×K)由 **AIU 异步搬运**:HBM → TSM(shared memory),分块流水。
- 激活 `x_q`(M×K,decode 时 K 小)直接驻留寄存器/TSM。
- 采用 **multistage 流水**:AIU 搬下一块 K 时,当前块做点积(与 `MainloopPPUAiuA8W8` 同思想)。
- swizzle 布局贴合 AIU 128bit 读取(与 `SmemLayoutAtom` 同约定)。

## 5. Tile 调度(参考 BF16 gemvt_op 的分档)

BF16 gemvt 已有 `32×4 / 64×2 / 128×1` 三档(按 N 大小挑)。int8 GEMV 采用同一思想:

| N 档 | block_n | warps/block | 每 warp 行数 |
|---|---|---|---|
| N ≤ 512 | 128 | 4 | 1 行/warp(block 覆盖 4 行) |
| 512 < N ≤ 2048 | 256 | 8 | 1 行/warp |
| N > 2048 | 512 | 16 | 1 行/warp |

grid = ceil(N / block_n),每个 threadblock 独立处理一段 N。

## 6. 高级编译与融合位置

1. **JIT 模板化**:K、block_n、warps、num_stages 全部编译期常量(与 `gemm_int8.py` 同款 `jit_tuner.compile_and_tune`)。
2. **Epilogue 融合点**:`LinearCombination` 里 alpha=`x_scale[m]`,beta=0;per-channel `w_scale[n]` 由 epilogue 的 `scale_b` 指针消费——与 GEMM 完全同构。
3. **tuner 接入**:`deep_gemm_tuner` 增加 `gemv_int8` 的 `(N,K)` 配置缓存,复用 `DG_CACHE_DIR`。
4. **SGLang 分派**:`M <= SGLANG_SAIL_GEMV_M_MAX`(默认 8)走 GEMV,否则 GEMM;env 默认关,风险隔离。

## 7. 与 GEMM 的关系(调度群体位置)

- **prefill / 大 M** → `DenseINT8GemmCutlass3Runtime`(Tensor Cell MMA,2D tile)
- **decode / M ≤ 8** → `INT8GemvRuntime`(SIMD 点积,1D tile)
- 两者共享:权重格式、scale 语义、JIT 缓存目录、`ProfilingInterface` instrument、`get_best_configs` 入口。

## 8. 落地文件清单

| 文件 | 内容 |
|---|---|
| `deep_gemm/include/deep_gemm/int8_gemv.cuh` | GEMV CUDA 模板(AIU 搬运 + SIMD 点积 + epilogue) |
| `deep_gemm/jit_kernels/gemv_int8.py` | Python JIT 层(模板填充 + tuner 接入) |
| `csrc/jit_kernels/impls/int8_gemv.hpp` | C++ Runtime(pybind + launch) |
| `deep_gemm/__init__.py` | 导出 `gemv_int8_int8_bf16_nt` |
| `sglang/.../w8a8_int8.py` | apply() 加 M 阈值分派 |
| `sglang/.../environ.py` | 加 `SGLANG_SAIL_USE_GEMV` / `SGLANG_SAIL_GEMV_M_MAX` |
