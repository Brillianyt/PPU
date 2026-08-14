# 05. 失败 / 否决实验完整记录

> 每个失败都写出根因，避免重蹈覆辙。

## 5.1 EAGLE 投机解码

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -m sglang.launch_server ... \
  --speculative-algorithm EAGLE --speculative-num-steps 2 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 3
```

**症状**：启动 12 秒后崩

```
File "sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py", line 339, in fused_sigmoid_gating_delta_rule_update
    retrieve_parent_token.int(),
AttributeError: 'NoneType' object has no attribute 'int'
[Received sigquit from a child process. It usually means the child failed.]
```

**根因**：
- EAGLE verify 阶段需要 `retrieve_parent_token` 做 parent tracking
- Qwen3.5-2B hybrid GDN 层是 **recurrent** 状态机，没有显式 parent token
- sglang 的 `gdn_backend.target_verify()` 没为 hybrid 模型正确填 parent token
- `hybrid_linear_attn_backend.py` 只在 `topk > 1` 时填 `retrieve_parent_token`，topk=1 留成 None

**结论**：sglang 0.5.10+ppu2.1.1 与 Qwen3.5-2B hybrid GDN 不兼容，无解。

## 5.2 EAGLE + MTP draft

```bash
# Qwen3.5-2B 自带 MTP 权重 (text_config.mtp_num_hidden_layers=1)
# safetensors 有 15 个 mtp.* 键
python3 -m sglang.launch_server ... \
  --speculative-algorithm EAGLE --speculative-draft-model-path ./Qwen3.5-2B
```

**症状**：同样崩，同样的栈尾。

**结论**：MTP 头当 draft 也走 EAGLE verify 路径，bug 一样。

## 5.3 NEXTN（用户指定参数）

```bash
--speculative-algorithm NEXTN --speculative-num-steps 3 \
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

**症状**：崩在 `fused_sigmoid_gating_recurrent.py:339`，与 EAGLE 同。

**结论**：bug 与 `num_steps` / `num_draft_tokens` / `topk` 取值无关。

## 5.4 NEXTN + FLA=off（绕过 bug 但性能劣化）

```bash
SGLANG_SAIL_FLA_CUDA=0 python3 -m sglang.launch_server ... \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

**原理**：
- FLA-off 走 Triton community kernel 路径
- `HAS_EAGLE_TREE_CUSTOM_ATTN_MASK = retrieve_parent_token is not None` 让核自己分 None 走无 tree-mask 分支

**结果**（`result_dev_en_nextn_flaoff_20.json`）：

| 配置 | accuracy | avg_ttft_ms | throughput (tok/s) |
|---|---|---|---|
| G 组（FLA=on, 无投机） | 0.85 | 64.7 | 369.0 |
| NEXTN + FLA=off | 0.85 | 87.3 (+35%) | 275.5 (-25%) |

**根因**：
- FLA-off 让 prefill 从 SAIL CUDA 退到 Triton，prefill 阶段本身就慢
- topk=1 的 draft 接受率低（MTP 头只预测 1 token，模型概率分布靠不住）
- 投机节省的 decode 步数 < 多花的 draft 推理时间

**结论**：投机解码对 2B 模型在短回答场景（mmbench 单选 256 token）下性价比为负。

## 5.5 INT8 W8A8 量化

```bash
python3 -u -m sglang.launch_server \
  --model-path ./quant/Qwen3.5-2B-W8A8 \
  --attention-backend triton \
  --quantization w8a8_int8 \
  ...
```

**DeepGEMM warmup**：~10s 扫 16384 tokens，然后 CUDA Graph 抓取 12.36s。

**结果**：
- accuracy: 0.75 (20 样本) → 0.66 (50 样本) ← **-19pp**
- 远低于 2% 阈值
- 模型输出乱码：多数 4-13 tokens 就停，重复问题文本

**排除 backend 因素**：bf16 + Triton 实测 accuracy 0.80，与 INT8 的 0.66 差距主要是量化。

**结论**：量化方案太激进（per-channel INT8 + lm_head 也量化），不能用。

## 5.6 DALI 图像预处理

**用户反馈**："DAIL我觉得在一定程度上是可以有效果的，因为TTFT总共63ms，CPU处理个10ms，但GPU就不一样了"

**实际探测**：
- sglang 用 HF `Qwen2VLImageProcessorFast`（torch-based）
- 在 GPU 上跑（不是 CPU）
- 实测：0.72ms/张
- vs 64ms TTFT 占比 ~1%

**否决理由**：
1. sglang 没有 DALI processor 入口，要重写 processor 路径
2. 即使切到 DALI 节省 0.5ms，对 64ms TTFT 收益 < 1%
3. ROI 太低

## 5.7 Sliding window attention

**用户反馈**："sliding window亦可以试试能不能继续加速"

**否决理由**：
- Qwen3.5 是 hybrid **linear_attention**（GDN）+ full_attention
- linear_attention 已经是 recurrent 计算，没有 KV cache 线性增长问题
- sliding window 是 full attention 的子集，对 linear_attention 不适用
- 模型架构不支持 sliding window config

## 5.8 长 timeout

**用户反馈**："不允许长 timeout"

**否决理由**：
- 违反公平性原则
- 主办方禁止"在评测机器上写入持久化缓存以跨次复测获利，除非主办方明确允许"
- 长 timeout 不算严格违规，但属于灰色地带

**实施**：wrapper 默认不传 `--watchdog-timeout` 或 `--dist-timeout`。

## 5.9 Vision encoder 量化 / 图像预处理优化

**探测**：
- vision encoder 有 24 层 ViT hidden=1024
- 接收 mmbench 图像，输出 ~1500 token 视觉 embedding
- 占 prefill 阶段 ~30%（粗估）

**未尝试原因**：
- sglang 内部 vision encoder 已高度优化
- 修改 vision encoder 需要改 sglang 源码
- ROI 不确定

## 5.10 NVLink / 多卡并行

**探测**：
- 机器只有 1× PPU-ZW810E，无 NVLink 多卡
- TP>1 无意义

## 5.11 用户给过的反馈纠正

| 用户反馈 | 我们的应对 |
|---|---|
| "你参考的过多了吧，你只用看这几个参数" | 收敛到用户指定的 4 个 NEXTN 参数 |
| "你并没有更多的采样看看是不是所有图片都很小" | 完整数据集采样，不只看 100 张 |
| "DAIL我觉得在一定程度上是可以有效果的" | 重新探索，但实测 ROI 太低仍否决 |
| "不允许长 timeout" | 移除所有长 timeout 参数 |
| "@quant/Qwen3.5-2B-W8A8/ 量化后的小模型来了，第一你并没有更多的采样看看是不是所有图片都很小，第二既然是推理加速竞赛，只要能加速就值得探索" | 认真跑 INT8，确认 0.66 accuracy 后正式否决 |