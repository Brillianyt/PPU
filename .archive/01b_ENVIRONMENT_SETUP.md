# 01b. 环境调通全过程（最耗时阶段）

> 这一段是整个工作中最费时的：从一个全新镜像到能跑出一个 baseline 数，经历了多个"卡住 → 排查 → 突破"的循环。把所有排查细节都保留下来。

## 0. 起点：拿到的工作目录

```bash
# 工作目录
/mnt/workspace/dndx_participant-v1.2

# 目录内容（最初）
README.md              # 5500 字节，包含比赛规则和接口契约
requirements.txt       # 32 字节
benchmark_public.py    # 10KB，本地自测脚本
evaluation_wrapper.py  # 19KB，待优化对象（接口契约）
datasets/mmbench/      # mmbench_dev_en.tsv 和 mmbench_dev_cn.tsv
Qwen3.5-2B/            # 模型权重目录
```

## 1. 第一步：理解 README

读 README 拿到关键约束：

| 关键点 | 内容 |
|---|---|
| 比赛目的 | 在不换权重/不修评测规则/不伪造结果下提升推理表现 |
| 评测入口 | `VLMModel.generate_with_metrics()` 不可改 |
| 评估指标 | accuracy + TTFT + throughput |
| 公开自测 | 只用 mmbench_dev_en.tsv 和 mmbench_dev_cn.tsv |
| 禁止条例 | 查表/换权重/联网/伪造/双策略/持久化缓存 |

→ 把这些都先记下来，避免后面违规。

## 2. 第二步：环境探测

```bash
python3 --version                          # 3.12
pip list | grep -E "torch|transformers|sglang"
which asys
ls /usr/local/PPU_SDK/                    # PPU 工具链
nvidia-smi                                  # PPU 不是 NVIDIA，无输出
ls /sys/class/drm | grep card              # DRM 设备不一定有 PPU
```

发现：
- Python 3.12
- torch 2.10.0+ppu2.1.1（PPU 版本）
- transformers **4.57.1** ← 关键
- sglang 0.5.10+v0.3.0.ppu2.1.1
- asys v2.1.1（PPU profiler）

## 3. 第三步：看模型是什么

```bash
cat Qwen3.5-2B/config.json
```

```json
{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "text_config": {
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 24,
    "vocab_size": 248320,
    "tie_word_embeddings": true,
    ...
  },
  "vision_config": {...}
}
```

→ VLM，hybrid linear_attention + full_attention，不是纯文本。

## 4. 第四步：试 baseline transformers 后端（第一次卡住）

```bash
python3 benchmark_public.py \
  --dataset-path ./datasets/mmbench/mmbench_dev_en.tsv \
  --model-path ./Qwen3.5-2B \
  --backend transformers --num-samples 20
```

报错：
```
ValueError: Transformers does not recognize this architecture "Qwen3_5ForConditionalGeneration"
```

排查：
1. 看 `evaluation_wrapper.py` 里的 transformers 后端 → 用 `AutoModelForImageTextToText.from_pretrained(...)`
2. 找 `trust_remote_code=True` 是否能解决 → **不行**，因为本地没有 `modeling_*.py`
3. 看 sglang 怎么用 qwen3.5 → `cat /usr/local/lib/python3.12/site-packages/sglang/srt/configs/qwen3_5.py`：
   ```python
   from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5Config
   # 注释提到需要更新的 transformers
   ```
4. 升级 transformers → 受限于时间和网络约束，**放弃**。

**结论**：transformers 后端在当前环境不可用，必须走 sglang。

## 5. 第五步：看 `evaluation_wrapper.py` 现状

```bash
cat evaluation_wrapper.py
```

文件结构：
- `GenerationConfig` dataclass
- `GenerationResult` dataclass
- `VLMModel` class
  - `__init__(model_path, backend='auto')` 自动选 backend
  - `generate_with_metrics(...)` 接口契约

现有 backend：
- `dummy`（smoke test）
- `transformers`（不可用）
- **没有 sglang 后端** ← 必须新增

## 6. 第六步：第一次启动 sglang server（卡 CUDA Graph 抓取 50 秒）

```bash
python3 -u -m sglang.launch_server \
  --model-path ./Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend triton \
  --enable-multimodal \
  --dtype bfloat16 \
  --mem-fraction-static 0.85
```

观察启动日志（关键行）：
```
[INFO] Loading model weights: 4.19 GB
[INFO] Allocating Mamba cache: 1.24 GB conv + 35.26 GB ssm
[INFO] KV cache: 40.58 GB (#tokens=3,545,083, bf16)
[INFO] Capturing CUDA graph (bs=256): 1/36 ...
...
[INFO] Capturing CUDA graph (bs=1): 36/36 done
[INFO] Application startup complete.
```

**启动 50+ 秒**，CUDA Graph 抓取 36 个 batch size 是主要耗时。

但**起来了**！

## 7. 第七步：写 sglang 后端 wrapper

按 OpenAI 兼容 API 写：

```python
class _SGLangServer:
    def __init__(self, model_path):
        # 1) 启动 subprocess 跑 sglang.launch_server
        # 2) 等待 /health_generate 200
        # 3) 发一次 warmup 请求（无图 chat, max_tokens=4）
        # 4) atexit 注册关进程

class VLMModel:
    def __init__(self, model_path, *, backend='auto', ...):
        if backend == 'sglang':
            self._load_sglang_backend()
        # ...

    def _generate_with_sglang(self, *, image, prompt, generation_config):
        # 1) PIL → JPEG(base64) → data URL
        # 2) requests.post stream=True + stream_options.include_usage
        # 3) 第一个 content chunk 时间 = TTFT
        # 4) usage.completion_tokens = token_count
```

注意：**禁止伪造 token_count / ttft_seconds**，必须从 sglang usage 拿真实值。

## 8. 第八步：第一次跑通 benchmark（20 样本）

```bash
python3 benchmark_public.py \
  --dataset-path ./datasets/mmbench/mmbench_dev_en.tsv \
  --model-path ./Qwen3.5-2B \
  --backend sglang --num-samples 20 \
  --output result_dev_en_sglang_20.json
```

结果：
- accuracy: 0.80（16/20）
- avg_ttft_ms: 419.0
- avg_throughput_tokens_per_sec: 284.5

但 TTFT 分布很怪：
```
min 42.5 / p25 96.8 / p50 346.0 / p75 354.9 / p95 3073.9 / max 3073.9
```

前 5 个 ~50ms（radix prefix cache 命中），之后 ~340ms 平台，1 个 3074ms 离群点。

## 9. 第九步：分析离群点 + 找优化方向

排查 3074ms 离群点：
- qid=344 的请求异常长
- 可能首次触发某个 bs 的 CUDA Graph 重捕
- 或后台 autotune

排查 340ms 平台：
- 不是 attention（用 triton 也是 ~300ms+）
- 不是 KV cache miss
- 不是 mmbench 数据特征（prompt 模板一样）

→ 锁定 GDN linear_attention 路径。

## 10. 第十步：尝试 fa3 attention

```bash
# 只改 --attention-backend triton → fa3
```

启动日志变化：
```
# 之前
Linear attention kernel backend: decode=triton, prefill=triton

# 之后 (fa3)
Linear attention kernel backend: decode=triton, prefill=triton  # GDN 仍 triton
```

CUDA Graph 抓取时间：**49.74s → 3.97s**（启动快 12 倍，但 GDN 仍 triton）

结果：
- accuracy: 0.85（+5pp）
- avg_ttft_ms: 313.2（-25%）
- throughput: 279.4（持平）

→ fa3 把 full attention 部分加速了，但 GDN 仍是瓶颈。

## 11. 第十一步：发现 SGLANG_SAIL_FLA_CUDA 隐藏开关

读 `sglang/srt/environ.py`：
```python
SGLANG_SAIL_FLA_CUDA = EnvBool(False)  # 实际默认 False（文档说 True 是错的）
```

读 `sglang/srt/layers/attention/hybrid_linear_attn_backend.py`：
- 根据这个开关选 kernel
- True → `fla.fused_sigmoid_gating_delta_rule_forward_k_last[_packed]`（PPU 专用 CUDA）
- False → Triton community kernel

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -m sglang.launch_server ... --attention-backend fa3 ...
```

启动日志：
```
[2026-08-14 16:56:29] USE PPU SAIL CUDA FLA kernel: fused_sigmoid_gating_delta_rule_forward_k_last_packed
```

结果：
- accuracy: 0.85
- **avg_ttft_ms: 64.3**（-80%）
- throughput: 280.1
- TTFT max: 79ms（紧凑）

→ **重大突破**：TTFT 平台从 ~340ms 跌到 ~64ms。

## 12. 第十二步：--disable-radix-cache 提升 throughput

mmbench 样本唯一，radix cache 收益接近零。试关：

```bash
# 在 fa3 + FLA=1 基础上加 --disable-radix-cache
```

联动：sglang 在 `disable_radix_cache=True` 时把 `disable_overlap_schedule` 自动设为 `False`：
- prefill/decode 重叠调度打开
- decode 不再等 prefill 完全结束

结果：
- accuracy: 0.85
- avg_ttft_ms: 69.0（轻微 +5ms，因 radix lookup 消失）
- **throughput: 369.4 tok/s**（+32%）

## 13. 第十三步：--page-size 64 微调

```bash
# 在 Phase 12 基础上加 --page-size 64
```

KV cache 按 64-token 粒度管理（vs 默认 1），减少 metadata。

结果：
- accuracy: 0.85
- **avg_ttft_ms: 64.7ms**（-6.2%）
- throughput: 369.0（持平）

→ **最终配置 G 组确定**

## 14. 整体时间线回顾

| 阶段 | 内容 | 用时 |
|---|---|---|
| 0→3 | 读 README、看镜像版本、看模型 | 5 分钟 |
| 4 | 试 transformers baseline → 失败（版本太老） | 10 分钟 |
| 5 | 读 wrapper 现状 | 5 分钟 |
| 6 | 第一次启动 sglang | 5 分钟（含 50s 启动等待） |
| 7 | 写 sglang wrapper | 20 分钟 |
| 8 | 第一次跑通 benchmark | 5 分钟 |
| 9 | 分析离群点 + 找方向 | 15 分钟 |
| 10 | 试 fa3 | 5 分钟 |
| 11 | 试 SGLANG_SAIL_FLA_CUDA=1（突破） | 20 分钟（需要读源码找开关） |
| 12 | 试 --disable-radix-cache | 5 分钟 |
| 13 | 试 --page-size 64 | 5 分钟 |
| **总计** | **从 0 到 baseline→优化** | **~100 分钟** |

## 15. 关键经验教训

1. **transformers 版本坑**：永远先查 `AutoModel.*.from_pretrained` 是否支持 `model_type`，不要直接试。
2. **sglang 启动慢**：CUDA Graph 抓取是主要耗时，fa3 比 triton 快 12 倍。
3. **隐藏开关要去读源码**：SGLang-for-SAIL 文档和实际代码可能不一致（如 `SGLANG_SAIL_FLA_CUDA`）。
4. **PPU profiler 是 asys 不是 nsys/nvprof**：`asys attach` 是 GPU kernel trace。
5. **算子 profile 是找瓶颈的关键**：73% GEMV 告诉我们量化是方向，但 INT8 准确率不达标 → 死路。
6. **CUDA Graph capture 是隐形启动成本**：36 个 bs 的 capture 一次启动 50s，跑 benchmark 前就要预留。