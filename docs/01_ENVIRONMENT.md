# 01. 环境

> SSOT 数字(精度的 0.78 baseline、-2pp 量化损失 等)= 上一级
> [README_REPRODUCE.md §0](../README_REPRODUCE.md)。
> 本文件只讲**环境与部署约束**,不复述数字。

## 1.1 硬件

| 项目 | 值 |
|---|---|
| 机器 | 1× PPU-ZW810E (真武 810E, T-Head) |
| HBM | 96 GB |
| HBM 带宽 | **~2.7 TB/s peak** (1.8 GHz × 6144-bit × 2 DDR / 8);实测可达 ~1 TB/s on typical GEMV |
| 计算 ABI | CUDA 兼容层(sgl_kernel / flash_attn_interface / DeepGEMM 等都跑通) |
| profiler | `asys` v2.1.1 (`/usr/local/PPU_SDK/asight/bin/asys`) |
| 关键驱动力 | `SGLANG_SAIL_USE_ACEXT_CUDA` 接入 PPU `sgl_kernel.int8_gemm`,见 §1.11 |

### 1.1a PPU001 详细硬件参数(从 `torch.cuda.get_device_properties(0)` + SDK 头文件)

| 参数 | 值 | 备注 |
|---|---|---|
| 设备名 | PPU-ZW810E | T-Head 自研 |
| **compute capability** | **8.0** | 跟 NVIDIA A100 同一代,**但带 SAIL 自有 ISA 扩展** |
| **SM 数** | **64** | 比 A100 少(108),决定 persistent worker pool 大小 |
| warp size | 32 | 跟 NVIDIA 一致 |
| regs/SM | 131,072 (128 KB) | 每个 thread ≤ 256 registers |
| **shared memory/SM** | **256 KB**(2 组 × 128 B/cycle) | 较大 — 适合 weight tile 缓存。**注意双 buffer 必须 128B 步长对齐** |
| shared memory/block | 256 KB | 单 block 可全占 shmem |
| max threads/SM | 2048 | = 8 blocks × 256 threads |
| **L2 cache** | **64 MB** | 很大 — 跨请求 weight 复用机会大 |
| memory clock | 1.8 GHz | GDDR6X 等级 |
| memory bus | 6144-bit | 6×1024-bit channel |
| 编译器 | gcc 13.3.0 + nvcc 13.0.0 | 注意 g++ 13 `cmath` constexpr vs HGGC `__device__ isinf/isnan` 冲突,见 `03_FAILED.md` F9 |

**SAIL ISA 扩展(PPU001 独有,需用 HGGC SDK 头文件):**
- `ppu.cp.async.cg.shared.global [%0], [%1], 16` — 异步 global→shared 拷贝(替代 NVIDIA `cp.async`)
- `ppu.cp.async.commit_group` / `ppu.cp.async.wait_group N` — 异步组管理
- `awmma::fragment<...>` — 公开 mma API(signed char, row_major / col_major)
- `ppu.tc01.mma.sync.aligned.m16n16k16` — Tensor Cell mma(PPU0015 不支持)

来源:`/mnt/workspace/ISA/docs/03_HARDWARE.md` + `/mnt/workspace/ISA/notes/HGGC_PPU001_NOTES.md`(12 章, 10KB,从 HGGC 编程指南 245 页 + 示例 68 页提取)。

## 1.2 镜像与软件版本

```text
镜像  pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.10.0-ubuntu24.04-cuda13.0-sglang0.5.10-py312

torch         2.10.0+v0.1.0.ppu2.1.1
transformers  4.57.1    ← 太老,不识别 qwen3_5,只能走 sglang 后端
sglang        0.5.13+v0.1.0.ppu2.1.1   ← 必须 0.5.13+(0.5.10 w8a8_int8 garbage 输出)
sglang-kernel 0.4.1+v0.1.0.ppu2.1.1
openai        2.6.1
python        3.12
```

> transformers 4.57.1 报 `ValueError: Transformers does not recognize this architecture`
> (`model_type=qwen3_5`),且模型目录无 `modeling_*.py`,`trust_remote_code` 也无济于事。
> **所有结果走 sglang 后端**。

## 1.3 模型

| 路径 | 用途 |
|---|---|
| `/mnt/workspace/Qwen3.5-2B` | BF16 原模型 (4.55 GB safetensors 单文件) |
| `/mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive` | **S2-aggressive INT8 量化 (3.79 GB)** — R6 / R8 用此 |
| `arch` | `Qwen3_5ForConditionalGeneration` (hybrid full + linear attn) |
| `hidden_size` | 2048 |
| `num_hidden_layers` | 24 (6 full + 18 GDN, full_attention_interval=4) |
| `quantization_config` 格式 | W8A8: weight per-channel int8 / activation per-token dynamic int8 |

> 历史备份 INT8 / MIXED (AMQ 三段) 在 `/mnt/workspace/Qwen_quant/Qwen3.5-2B-{INT8,MIXED}/`,
> 上不了竞赛 —— 详见 `../INT8_strategy.md` 与 [`03_FAILED.md`](03_FAILED.md) F2/F5。

## 1.4 数据集

```text
/mnt/workspace/datasets/mmbench/
├── mmbench_dev_en.tsv   (~4325 样本,公开自测)
└── mmbench_dev_cn.tsv   (中文版,同样大小)
```

每条: `index, image (base64), question, hint, A, B, C, D, answer, category, l2-category`。

`benchmark_public.py` 的 prompt 模板 (en):
> "Solve this single-choice question. Your response must make one final choice among A/B/C/D clearly. You may include one short reason."

cn 模板:见 `benchmark_public.py:117-118`(含「答案/选项/正确」等约束)。

**约束**:`max_new_tokens=256`, `temperature=0.0`, `top_p=1.0`。

## 1.5 评测接口契约 (不可改)

```python
class VLMModel:
    def generate_with_metrics(
        self, *, image, prompt: str, choices: dict[str, str],
        generation_config: GenerationConfig, sample_id: str,
    ) -> GenerationResult: ...

@dataclass
class GenerationResult:
    text: str
    token_count: int
    ttft_seconds: float
    elapsed_seconds: float
    meta: dict
```

3 个 backend: `dummy` / `transformers` / `sglang`(默认)。`benchmark_public.py` 通过
实例化 `VLMModel(model_path, backend=...)` 触发。

## 1.6 比赛禁止条例 (摘自 [`../../benchmark/README.md`](../../benchmark/README.md))

- 禁止更换模型权重 / 蒸馏 / 主办方不认可的量化包
- 禁止按 `sample_id` / 题号 / 图片 hash / 题干 hash 查表
- 禁止硬编码公开集或疑似测试集答案
- 禁止联网调用外部模型 / API / 搜索
- **禁止伪造 `token_count` / `ttft_seconds` / `elapsed_seconds`**
- 禁止把准确率评测和性能评测拆成两套策略
- 禁止在评测机器上写持久化缓存跨次复测获利

## 1.7 性能数据采集约定

- **TTFT**:流式模式下,从请求发到 sglang 到第一个 content chunk 到达的时间
  (我们的 wrapper 用 `TextIteratorStreamer` 实测)。
- **token_count**:必须用 sglang 返回的 `usage.completion_tokens`,**不能**根据 text 长度估算。
- **elapsed_seconds**:从请求发到流结束的 wall clock。
- **TTFT 评估用 median**:100 样本 mean 受尾部冷启动样本拉高 4×(PPU fork 在 VLM
  工作负载下自动 disable PiCG,见 §1.11)。

## 1.8 PPU 关键工具

```bash
# GPU kernel trace (60s)
asys attach -t hggc -s process-tree -o <file.asysrep> --duration 60 <PID>

# 报告导出
asys stats -r ppu_op_type_sum -f csv -o . <file>.asysrep
asys stats -r ppu_op_kernel_breakdown -f csv -o . <file>.asysrep
asys stats -r hggc_kern_exec_trace -f csv -o . <file>.asysrep  # 巨大,但有 grid

# DeepGEMM JIT 预编译(可选,提速首推理)
python3 -m sglang.compile_deep_gemm
```

## 1.9 量化模型细节 (S2-aggressive)

- **175 → 126 个 Linear** 走 INT8 W8A8,保留误差最高的敏感层 (`in_proj_a/b`、
  `out_proj`、19/23 层 qkv) 为 BF16。
- **Recipe 文件**:`../qwen3.5-2b_recipe.yaml`(根目录)。完整量化策略见
  [`04_OPTIMIZATIONS.md`](04_OPTIMIZATIONS.md) §4.1 与 `INT8_strategy.md`。
- **生成工具**:qlean (QLean-for-SAIL v0.2.0+v0.1.0.ppu2.1.1),主办方认可的量化工具,
  不违反比赛禁止条例第 1 条。
- 模型大小:4.55 GB → 3.79 GB (−16.7%)。eval 时占用 ~30 GB KV cache +
  3.79 GB 权重,在 96 GB 显存内有余量。

## 1.10 Patch 部署约定 (启动 VLM/CG 路径强制)

PPU 镜像里 sglang 0.5.13 出厂不带任何针对性 patch。R6 / R8 路径必须打以下
**6 类共 10 个文件**。来源固定:

| 类 | 文件路径 (位于 `../`) | 部署目标 |
|---|---|---|
| A | `gemv_optimization/sglang-for-sail/python/sglang/srt/environ.py` | `site-packages/sglang/srt/environ.py` |
| B | `gemv_optimization/.../w8a8_int8.py` | `site-packages/sglang/srt/layers/quantization/w8a8_int8.py` |
| C | `gemv_optimization/.../layers/deep_gemm_wrapper/{compile_utils,entrypoint}.py` | `site-packages/sglang/srt/layers/deep_gemm_wrapper/` |
| D | `gemv_optimization/DeepGEMM-for-sail/deep_gemm/{__init__.py, jit_kernels/{__init__.py, gemv_int8.py}}` | `site-packages/deep_gemm/` |
| E | `gemv_optimization/.../models/qwen3_vl.py` + `.../hardware_backend/ppu/attention/ppu_fa3_hooks.py` | `site-packages/sglang/srt/models/qwen3_vl.py` (补 `max_context_len`) + `site-packages/.../attention/ppu_fa3_hooks.py` (新 ViT fa3 hook) |
| F | `inductor_hook_PPU/hook/ppu_rotary_hooks.py` | `site-packages/sglang/srt/hardware_backend/ppu/attention/ppu_rotary_hooks.py` + `site-packages/sglang/srt/platforms/ppu.py` 一行 `import` |

**一键部署**(纯 Python 覆盖,不改 C++ 扩展):

```bash
cd /mnt/workspace/PPU

# A/B/C/E (sglang 9 文件,含 qwen3_5 / qwen3_5_mtp / eagle_worker_v2)
cp -rv gemv_optimization/sglang-for-sail/python/sglang/* \
        /usr/local/lib/python3.12/site-packages/sglang/

# D (DeepGEMM 3 文件)
cp -rv gemv_optimization/DeepGEMM-for-sail/deep_gemm/* \
        /usr/local/lib/python3.12/site-packages/deep_gemm/

# F (1 文件,路径不与上面重叠)
cp -v inductor_hook_PPU/hook/ppu_rotary_hooks.py \
      /usr/local/lib/python3.12/site-packages/sglang/srt/hardware_backend/ppu/attention/ppu_rotary_hooks.py

# F 注册到 platforms/ppu.py (一次性)
grep -q "ppu_rotary_hooks" /usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py \
  || sed -i '/ppu_flashmla_hooks/a\    import sglang.srt.hardware_backend.ppu.attention.ppu_rotary_hooks  # noqa: F401' \
        /usr/local/lib/python3.12/site-packages/sglang/srt/platforms/ppu.py

# 清理 __pycache__
rm -rf /usr/local/lib/python3.12/site-packages/sglang/srt/{layers/quantization,layers/deep_gemm_wrapper,models,hardware_backend/ppu/attention,platforms}/__pycache__
rm -rf /usr/local/lib/python3.12/site-packages/deep_gemm/{,jit_kernels}/__pycache__
```

**回滚**:`git restore` site-packages(单仓库 update)即可还原。

## 1.11 PPU fork 启动自动行为 (会覆盖用户 CLI,影响 TTFT 复现)

`sglang/srt/server_args.py:1349-1355 (_handle_ppu_backends)` 与
`:1417 (_handle_piecewise_cuda_graph rule 8)` 在 PPU 平台上有 **3 个自动行为**,
会让 wrapper 默认启服务的实测数字与 README 标称数字有出入:**对结果解读有影响**。

### (1) `SGLANG_SAIL_USE_ACEXT_CUDA` 默认 True,即使设 0 也不生效

PPU fork 在用户没显式禁用时会强制开启 ACEXT (`sgl_kernel.int8_gemm`) 接管
dense linear。`SGLANG_SAIL_USE_GEMV=1` 仍会预热 deep_gemm 的 GEMV JIT 内核供
fallback,但**运行时 dense 路径跑的是 ACEXT**。R8 标称数字 (TPS 434) 实际就是
这条 ACEXT 路径的实测,GEMV 数字 (~418) 是无 PiCG 的纯 dense linear 对照口径。

```python
# sglang/srt/server_args.py:1349-1355 (_handle_ppu_backends)
if is_ppu():
    if not envs.SGLANG_SAIL_USE_ACEXT_CUDA.is_set():
        envs.SGLANG_SAIL_USE_ACEXT_CUDA.set(True)  # ← 强制覆盖
    if envs.SGLANG_SAIL_USE_ACEXT_CUDA.get():
        check_acext_version_compatibility()
```

### (2) `disable_piecewise_cuda_graph=True` 是 VLM 工作负载的必然结果

`sglang/srt/server_args.py:1417` 的第 8 条规则 "Multimodal / VLM models" **触发
`--enable-piecewise-cuda-graph` 被自动 disable**。`PiCG → TTFT −13%` 收益在
multimodal 路径下不可复现。

- 稳态 TTFT median **63–66 ms** 仍成立 (R6 / R7 标称 64–66 ms)
- 但 **100 样本 mean 会被尾部冷启动样本拉到 73–280 ms**
- 评估 TTFT 时**用 median 不用 mean**
- 不要尝试 `--enforce-piecewise-cuda-graph` 强行开 PiCG,会触发
  `cuda_piecewise_backend.py:171` 的 `PCG capture stream is not set`,scheduler
  直接死掉。`../evaluation_wrapper.py:188-203` 注释里专门记录了这条 avoid 路径。

```python
# sglang/srt/server_args.py:1417
# 8. Multimodal / VLM models
if self.get_model_config().is_multimodal:
    self.disable_piecewise_cuda_graph = True   # ← multimodal 必 disable
```

### (3) 默认开启的 4 个 env (无需用户设)

| env | 用途 | 设置位置 |
|---|---|---|
| `SGLANG_SAIL_FLA_CUDA=1` | PPU GDN/Mamba 原生 CUDA kernel | `_handle_ppu_backends:1368-1369` |
| `SGLANG_SAIL_DEEPGEMM_DENSE=1` | DeepGEMM INT8 dense 路径 | `_handle_ppu_backends:1357-1358` |
| `SGLANG_SAIL_DEEPGEMM_MOE=1` | DeepGEMM INT8 MoE 路径 | `_handle_ppu_backends:1359-1360` |
| `SGLANG_DISABLE_CUSTOM_ALL_REDUCE=1` | PPU 不支持 custom all-reduce | `_handle_ppu_backends:1361-1363` |

> 这些自动设置与 README 推荐值一致,不会影响结果解读。但**用户想覆盖**时
> 这些 env var 仍生效 (set 行为只跳过默认值,用户显式 set 不被覆盖)。

### 复测关键 takeaway

| 用户以为发生的 | 实际发生 |
|---|---|
| `SGLANG_SAIL_USE_ACEXT_CUDA=0` + `SGLANG_SAIL_USE_GEMV=1` → 走 GEMV | **走 ACEXT** (用户 env 被 PPU fork 覆盖) |
| `--enable-piecewise-cuda-graph` → PiCG on | **PiCG off** (multimodal 自动 disable) |
| `SGLANG_EXTRA_ARGS=--mm-attention-backend fa3` → fa3 attention | **fa3 attention** (这条**没**被覆盖,正常生效) |
| `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` → ViT CG capture | **ViT CG ON** (前提是 patch F 已装) |
