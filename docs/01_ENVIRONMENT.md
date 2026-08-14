# 01. 环境认识

## 1.1 镜像与硬件

```
镜像：pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.10.0-ubuntu24.04-cuda13.0-sglang0.5.10-py312
机器：1× PPU-ZW810E（真武 810E，96 GB HBM）
CPU：未详查（不重要，GPU bound）
PPU 算子：PPU-ZW810E 自带 HBM 约 1 TB/s 量级（从 GEMV 时间反推）
```

## 1.2 软件版本

```
torch            : 2.10.0+v0.1.0.ppu2.1.1
transformers     : 4.57.1         ← 关键：太老，不认识 model_type=qwen3_5
sglang           : 0.5.10+v0.3.0.ppu2.1.1
sglang-kernel    : 0.4.1+v0.1.0.ppu2.1.1
openai           : 2.6.1
asys (PPU profiler): v2.1.1_20260725-15d8b9d
Python           : 3.12
```

## 1.3 模型

```
路径：./Qwen3.5-2B
架构：Qwen3_5ForConditionalGeneration（hybrid full_attention + linear_attention）
类型：VLM（vision encoder + LLM decoder）
精度：bf16
权重大小：4.19 GB（safetensors 单文件）
vocab_size：248320
hidden_size：2048
intermediate_size：6144（MLP）
num_hidden_layers：24（hybrid: 6 full attn + 18 GDN linear attn，full_attention_interval=4）
num_attention_heads：8 (full)
head_dim：256
num_key_value_heads：2 (GQA)
linear_num_key_heads：16（GDN）
linear_key_head_dim：128（GDN）
linear_conv_kernel_dim：4（causal_conv1d before recurrence）
tie_word_embeddings：True（lm_head 共享 embed）
mtp_num_hidden_layers：1（带 MTP 权重，但无法用 — 见 05）
```

## 1.4 数据集

```
datasets/mmbench/mmbench_dev_en.tsv  （公开自测集，~4325 样本）
datasets/mmbench/mmbench_dev_cn.tsv  （中文版，结构同）

每条：index, image (base64), question, hint, A, B, C, D, answer, category, l2-category
prompt 模板（benchmark_public.py）：
  "Solve this single-choice question. Your response must make one final choice among A/B/C/D clearly. You may include one short reason."
  + "Question: ..." + A/B/C/D 选项
约束：max_new_tokens=256, temperature=0.0, top_p=1.0
```

## 1.5 评测接口契约（不可改）

```python
class VLMModel:
    def generate_with_metrics(self, *, image, prompt, choices, generation_config, sample_id) -> GenerationResult

@dataclass
class GenerationResult:
    text: str
    token_count: int
    ttft_seconds: float
    elapsed_seconds: float
    meta: dict
```

## 1.6 主办方禁止条例（重要，必须遵守）

- 禁止更换模型权重 / 蒸馏 / 非官方认可量化包
- 禁止根据 sample_id / 题号 / 图片 hash / 题干 hash 查表
- 禁止硬编码公开集或疑似测试集答案
- 禁止联网调用外部模型 / API / 搜索
- **禁止伪造 token_count / ttft_seconds / elapsed_seconds**
- 禁止把准确率评测和性能评测拆成两套不同策略
- 禁止在评测机器上写持久化缓存以跨次复测获利

## 1.7 性能数据采集点

- TTFT：从请求发到 sglang 到第一个 content chunk 到达（stream 模式下）
- token_count：必须用 sglang 返回的 `usage.completion_tokens`，不能根据 text 长度估算
- elapsed_seconds：从请求发到流结束的 wall clock

## 1.8 关键 PPU 工具

```
asys attach -t hggc -s process-tree -o <file> --duration <s> <PID>  # GPU kernel trace
asys stats -r ppu_op_type_sum -r ppu_op_sum -r ppu_op_kernel_breakdown ...
python3 -m sglang.compile_deep_gemm ...                              # 预编译 DeepGEMM
python3 -m sglang.launch_server ...                                  # 主入口
```