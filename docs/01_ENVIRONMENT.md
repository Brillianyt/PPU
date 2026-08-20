# 01. 环境认识

## 1.1 硬件

| 项目 | 值 |
|---|---|
| 机器 | 1× PPU-ZW810E (真武 810E) |
| HBM | 96 GB |
| HBM 带宽 | ~1 TB/s (从 GEMV 耗时反推) |
| PPU profiler | `asys` v2.1.1 (`/usr/local/PPU_SDK/asight/bin/asys`) |

## 1.2 镜像与软件版本

```
镜像：pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.10.0-ubuntu24.04-cuda13.0-sglang0.5.10-py312

torch            : 2.10.0+v0.1.0.ppu2.1.1
transformers     : 4.57.1            ← 太老,不识别 qwen3_5,本地 transformers 不可用
sglang           : 0.5.13+v0.1.0.ppu2.1.1    ← 必须 0.5.13+ (0.5.10 w8a8_int8 输出 garbage)
sglang-kernel    : 0.4.1+v0.1.0.ppu2.1.1
openai           : 2.6.1
Python           : 3.12
```

## 1.3 模型

| 项目 | 值 |
|---|---|
| 路径 (bf16) | `./Qwen3.5-2B` |
| 路径 (INT8) | `/mnt/workspace/models/Qwen_quant/Qwen3.5-2B-S2-aggressive` |
| 架构 | `Qwen3_5ForConditionalGeneration` (hybrid full + linear attn) |
| 精度 | bf16 (4.3 GB safetensors 单文件) |
| vocab_size | 248320 |
| hidden_size | 2048 |
| num_hidden_layers | 24 (6 full + 18 GDN, full_attention_interval=4) |
| mtp_num_hidden_layers | 1 (有 MTP 权重但不可用 — 见 `03_FAILED.md`) |

## 1.4 数据集

```
datasets/mmbench/mmbench_dev_en.tsv  (~4325 样本,公开自测)
datasets/mmbench/mmbench_dev_cn.tsv  (中文版)

每条: index, image (base64), question, hint, A, B, C, D, answer, category, l2-category
prompt 模板 (benchmark_public.py):
  en: "Solve this single-choice question. Your response must make one final choice among A/B/C/D clearly. You may include one short reason."
  cn: "请完成这道单选题。请给出你认为正确的选项,并可附带一句简短理由。答案必须明确,且只能对应 A/B/C/D 中的一个选项。"
约束: max_new_tokens=256, temperature=0.0, top_p=1.0
```

## 1.5 评测接口契约 (不可改)

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

## 1.6 主办方禁止条例 (必须遵守)

- 禁止更换模型权重 / 蒸馏 / 非官方认可量化包
- 禁止根据 sample_id / 题号 / 图片 hash / 题干 hash 查表
- 禁止硬编码公开集或疑似测试集答案
- 禁止联网调用外部模型 / API / 搜索
- **禁止伪造 token_count / ttft_seconds / elapsed_seconds**
- 禁止把准确率评测和性能评测拆成两套不同策略
- 禁止在评测机器上写持久化缓存以跨次复测获利

## 1.7 性能数据采集

- **TTFT**: 流式模式下,从请求发到 sglang 到第一个 content chunk 到达的时间
- **token_count**: 必须用 sglang 返回的 `usage.completion_tokens`,**不能**根据 text 长度估算
- **elapsed_seconds**: 从请求发到流结束的 wall clock

## 1.8 PPU 关键工具

```bash
# GPU kernel trace (60s)
asys attach -t hggc -s process-tree -o <file> --duration 60 <PID>

# 导出报告
asys stats -r ppu_op_type_sum -f csv -o . <file>.asysrep
asys stats -r ppu_op_sum -f csv -o . <file>.asysrep
asys stats -r ppu_op_kernel_breakdown -f csv -o . <file>.asysrep
asys stats -r hggc_kern_exec_trace -f csv -o . <file>.asysrep  # 巨大,但有 grid

# DeepGEMM JIT 预编译
python3 -m sglang.compile_deep_gemm
```

## 1.9 sglang 后端为什么是唯一选择

- transformers 4.57.1 不识别 `model_type=qwen3_5`
- 模型目录无 `modeling_*.py`,`trust_remote_code` 也无济于事
- 升级 transformers 受网络/时间约束 (本任务不允许 30s+ 等待)
- sglang 内部 `sglang/srt/configs/qwen3_5.py` 已支持,但 transformers 旧版不支持
- **结论**: 所有结果走 sglang 后端