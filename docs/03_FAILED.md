# 03. 失败 / 否决实验

> 按"为什么否决"分类,不是按时间。

## F1. EAGLE / NEXTN 投机解码 — **崩溃**

**症状**: sglang server 启动后 12 秒内崩溃

```
File "sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py", line 339, in fused_sigmoid_gating_delta_rule_update
    retrieve_parent_token.int(),
AttributeError: 'NoneType' object has no attribute 'int'
```

**根因**:
- EAGLE/NEXTN verify 阶段需要 `retrieve_parent_token` 做 parent tracking
- Qwen3.5-2B 是 hybrid GDN 模型,linear_attention 层是 recurrent 状态机,**没有显式的 parent token 概念**
- sglang 0.5.13+ppu2.1.1 的 `gdn_backend.target_verify()` 没为 hybrid 模型正确填 parent token
- `fla.fused_sigmoid_gating_delta_rule_update()` 接口期望非 None,直接 .int() 崩

**试过的参数组合** (全部崩):
```
--speculative-algorithm EAGLE --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3
--speculative-algorithm EAGLE + --speculative-draft-model-path ./Qwen3.5-2B
--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

**绕过 (FLA=off)**: 能跑但全维度劣于 G 组
| 配置 | acc | TTFT | tput |
|---|---|---|---|
| G 组 (FLA=on) | 0.85 | 64.7 ms | 369 tok/s |
| NEXTN + FLA=off | 0.85 | 87.3 ms (+35%) | 275 tok/s (-25%) |

**结论**:
- bug 与具体参数无关 — `target_verify` 进 GDN 递归核之前就把 `retrieve_parent_token` 留成 `None`
- 短回答场景 (mmbench 256 token) 投机节省本来就少,decode 步数有限
- **Qwen3.5 + sglang 0.5.13 上 EAGLE/NEXTN 完全不可用**,即使带 MTP 头也不能

## F2. INT8 W8A8 量化 (sglang 0.5.10) — **garbage 输出**

**症状**: 模型输出乱码,多数请求 4-13 token 就停,重复问题文本

```
"B. Does Madelyn's snowboard slide down a hill in less time when it has a layer of wax or when it does not have a layer of wax?"
```

**根因**: sglang 0.5.10 的 `w8a8_int8.py` 在 PPU 上有 bug

**修复**: 升级到 sglang **0.5.13** (修复)。0.5.13 后 INT8 输出正常,但有 ~5pp acc 损失 (见 `02_RESULTS.md`)。

## F3. DALI 图像预处理 — **0.72ms/张,无空间**

| 操作 | 耗时 |
|---|---|
| HF Qwen2VLImageProcessorFast (已在 GPU) | 0.72 ms/张 |
| DALI pipeline (替换) | 1.5 ms/张 (含 H2D/CPU↔GPU) |

**结论**: 主路径 `Qwen2VLImageProcessorFast` 已是 GPU 实现,不需要 DALI。

## F4. 长 timeout 参数 — **违反规则**

试过 `--watchdog-timeout 3600 --dist-timeout 3600` — 主办方明确禁止,直接放弃。

## F5. transformers 4.57.1 直接推理 — **不可用**

- `ValueError: Transformers does not recognize this architecture` (model_type=qwen3_5)
- 模型目录无 `modeling_*..py`,`trust_remote_code` 也无济于事
- 升级 transformers 受网络/时间约束 (本任务不允许 30s+ 等待)
- **结论**: 走 sglang 后端是唯一选择

## F6. 单层量化 (v3, 42 层 INT8) — **端到端无收益**

- microbench 上 GEMV 比 Triton 快 2.15× (13.0us vs 27.9us)
- 端到端 benchmark 与不量化持平
- **根因**: 覆盖率太低,GEMV 节省吃不下 wrapper overhead
- **改进**: S2 (126 层) + S3 (186 层) + GEMV 才有端到端 +36%

## F7. 手动修改 sglang 源码注入 Triton GEMV — **无效**

**§30 详细**: 把 Triton GEMV 直接写进 `w8a8_int8.py` 和 `compressed_tensors_w8a8_int8.py`,期望捕获 M=1 decode。

**问题**: sglang 抓 CUDA Graph 时用 bs=256,**`x.shape[0]` 恒等于 256,从不等 1**。patch 在 capture 阶段一次性 fire 2 次后就再也不调用 Python。

**扩展到 M=2..16**: 用 `tl.dot` 写 small-M kernel,但 PPU 张量核强制 BLOCK_M=16,M=1/2/4/8 全 pad 到 16,**反而比 acext 慢 1.1-3.3×**。

**回退**: 已恢复 sglang 源码 (sha256 验证),无残留。

**教训**: sglang PPU INT8 推理瓶颈在 prefill + 大 batch decode (M >= 256),不是 M=1 路径。CUDA Graph 完全封死 M=1。

## F8. 一致性低质量 result (已删除)

TTFT>100 ms **且** tput<300 tok/s 的样本被过滤掉 (噪声或冷启期数据):

| 文件 | acc | TTFT | tput | 原因 |
|---|---|---|---|---|
| `result_dev_en_sglang_20.json` | 0.80 | 419.0 | 284.5 | triton baseline (历史) |
| `result_dev_en_sglang_fa3_20.json` | 0.85 | 313.2 | 279.4 | fa3 但无 FLA (历史) |
| `result_dev_en_sglang_smoke.json` | 0.80 | 288.7 | 280.3 | 冒烟测试 |
| `result_dev_en_int8_fa3_fla1_500.json` | 0.81 | 149.8 | 115.6 | tuner 抢占 GPU (跑分时后台 tunner 仍在跑) |
| `result_dev_cn_int8_fa3_fla1_500.json` | 0.82 | 121.2 | 112.7 | 同上 |
| `result_dev_en_int8_s2_mtp_20.json` | 0.80 | 100.9 | 243.6 | S2 + MTP (尝试失败) |

## F9. S3-full 全量化 (186 层) — **acc 太低**

| 方案 | acc | TTFT | tput |
|---|---|---|---|
| S3-full + GEMV | 0.650 | 104.0 | 372.3 |
| S3-full 无 GEMV | 0.650 | 80.4 | 273.3 |

- 误差最高的 GDN 层也被量化 → 破坏 recurrent 状态
- **不能提交**

## F10. Qwen origin baseline recheck — **acc 0**

`result_dev_en_qwen_origin_*_20.json` 都是 0% 准确率,疑似 transformers 路径上崩或路径错误,只作为 debug 标记。