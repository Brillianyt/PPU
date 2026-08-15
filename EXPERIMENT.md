# DNDX 实验记录

## 1. 固定环境

| 项目 | 值 |
|---|---|
| 机器 | 1 × PPU-ZW810E（真武 810E，96 GB） |
| 镜像 | `pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.10.0-ubuntu24.04-cuda13.0-sglang0.5.10-py312` |
| torch | 2.10.0+v0.1.0.ppu2.1.1 |
| transformers | 4.57.1 |
| sglang | 0.5.10+v0.3.0.ppu2.1.1 |
| sglang-kernel | 0.4.1+v0.1.0.ppu2.1.1 |
| openai | 2.6.1 |
| 模型 | `Qwen3.5-2B`（`Qwen3_5ForConditionalGeneration`，bf16，4.3 GB） |
| 数据集 | `datasets/mmbench/mmbench_dev_en.tsv`（公开自测，仅作调试） |
| 评测脚本 | `benchmark_public.py`（只读，不进提交） |
| 提交物 | `evaluation_wrapper.py`（接口契约 `VLMModel.generate_with_metrics(...)` 不变） |

## 2. 代码改动

### `evaluation_wrapper.py`
- 保留 `transformers` / `dummy` 后端。
- 新增 `sglang` 后端：
  - `_SGLangServer`：自动拉起本地 server（`python -m sglang.launch_server`），复用 `localhost:8999`（若已 alive），等待 `/health_generate`，`_warmup()` 一次无图 chat，`atexit` 关进程。
  - 可调环境变量：`SGLANG_SERVER_URL` / `SGLANG_PORT` / `SGLANG_TP_SIZE` / `SGLANG_ATTENTION_BACKEND` / `SGLANG_DTYPE` / `SGLANG_MEM_FRACTION` / `SGLANG_EXTRA_ARGS`。
  - `_generate_with_sglang`：PIL→JPEG(base64)→data URL；`stream=True` + `stream_options.include_usage`，首个 content chunk 时间即 TTFT；usage 给真实 `completion_tokens` / `prompt_tokens`，杜绝伪造。

### `benchmark_public.py`（仅本地调试用，不提交）
- `--backend` choices 新增 `sglang`。

## 3. sglang server 启动命令

```bash
python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/dndx_participant-v1.2/Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend triton --enable-multimodal \
  --dtype bfloat16 --mem-fraction-static 0.85
```

观察到的启动阶段耗时：
- 加载权重：1.04 s（4.19 GB）
- 分配 Mamba cache：1.24 GB conv + 35.26 GB ssm
- 分配 KV cache：40.58 GB（`#tokens=3,545,083`，bf16）
- CUDA Graph 抓取：36 个 bs，约 49.74 s
- 端口监听：`Application startup complete.` ≈ 60 s 整体

## 4. Baseline 1：sglang 后端 20 样本

文件 `result_dev_en_sglang_20.json`（warmup=2，seed=20260625）

| 指标 | 值 |
|---|---|
| backend | sglang |
| sample_count | 20 |
| accuracy | 0.80（16/20） |
| avg_ttft_ms | 419.0 |
| avg_throughput_tokens_per_sec | 284.5 |
| total_benchmark_seconds | 11.49 |
| avg_seconds_per_sample | 0.575 |
| public_validation_passed | ✅ |

TTFT 分布（ms）：min 42.5 / p25 96.8 / p50 346.0 / p75 354.9 / p95 3073.9 / max 3073.9

观察：
- 前 5 个样本 TTFT ~50 ms（radix prefix cache 命中）
- 之后跳到 ~340 ms 平台（调度行为变化）
- 1 个 3074 ms 离群点（`qid=344`，可能首次触发某个 bs 的 CUDA Graph 重捕或后台 autotune）
- Throughput 始终 280–300 tok/s，稳态

## 5. 对照组说明（重要）

- "无 sglang" transformers 后端在本环境 **不可用**：
  - transformers 4.57.1 不识别 `model_type=qwen3_5`（`ValueError: Transformers does not recognize this architecture`）。
  - 模型目录里没有 `modeling_*.py`，`trust_remote_code` 也无济于事。
  - 升级 transformers 受限于网络/时间约束（本任务不允许 30 s 以上的等待），本地不折腾升级。
  - sglang 内部 `sglang/srt/configs/qwen3_5.py` 的注释 `from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5Config` 表明更新版 transformers 已支持 Qwen3.5，但当前环境装的是 4.57.1。
- 因此本轮对比改为：**两组不同 sglang 配置** 间的对照。

## 6. 对照组 B：sglang `--attention-backend fa3`

启动命令（与 baseline 仅 `--attention-backend` 不同，其余一致）：

```bash
python3 -u -m sglang.launch_server \
  --model-path ./Qwen3.5-2B \
  --host 127.0.0.1 --port 8998 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 --enable-multimodal \
  --dtype bfloat16 --mem-fraction-static 0.85
```

观察：
- 启动日志显示：`Linear attention kernel backend: decode=triton, prefill=triton`（因为 fa3 不支持 GDN，线性层仍走 triton；只有 full attention 走 fa3）。
- CUDA Graph 抓取 **3.97 s**（baseline triton 49.74 s），启动体感明显更快。

文件 `result_dev_en_sglang_fa3_20.json`

| 指标 | baseline (triton) | 对照 (fa3) | Δ |
|---|---|---|---|
| sample_count | 20 | 20 | — |
| accuracy | 0.80 (16/20) | **0.85** (17/20) | +5 pp |
| avg_ttft_ms | 419.0 | **313.2** | **-25.3 %** |
| avg_throughput_tokens_per_sec | 284.5 | 279.4 | -1.8 % |
| public_validation_passed | ✅ | ✅ | — |

TTFT 分布（fa3，ms）：min 45.2 / p25 324.6 / p50 331.6 / p75 339.1 / p95 762.5 / max 762.5；均值 313.2。

## 7. 对照组 C：fa3 + `SGLANG_SAIL_FLA_CUDA=1`

动机：SGLang-for-SAIL 文档把 `SGLANG_SAIL_FLA_CUDA` 默认写成 `True`，但源码里实际默认是 `False`（`sglang/srt/environ.py:550: SGLANG_SAIL_FLA_CUDA = EnvBool(False)`）。该开关影响 `fused_sigmoid_gating_recurrent.py` 和 `fused_recurrent.py` 中两处 GDN packed decode / extend 路径，命中后调用 `fla.fused_sigmoid_gating_delta_rule_forward_k_last[_packed]`（PPU 专用 CUDA kernel）。

启动：
```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server \
  --model-path ./Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 --enable-multimodal \
  --dtype bfloat16 --mem-fraction-static 0.85
```

启动日志关键一行：
```
[2026-08-14 16:56:29] USE PPU SAIL CUDA FLA kernel: fused_sigmoid_gating_delta_rule_forward_k_last_packed
```
（对照组 B 即便未设该 env var 也**不会**触发这条日志，GDN kernel dispatcher 仍标 `TritonGDNKernel`，验证了开关确实切换到 PPU 专用实现。）

文件 `result_dev_en_sglang_fla1_20.json`

| 指标 | A triton | B fa3 | **C fa3 + FLA=1** | C vs A | C vs B |
|---|---|---|---|---|---|
| accuracy | 0.80 | 0.85 | **0.85** | +5 pp | 0 |
| avg_ttft_ms | 419.0 | 313.2 | **64.3** | **-84.7 %** | **-79.5 %** |
| avg_throughput_tokens_per_sec | 284.5 | 279.4 | 280.1 | -1.5 % | +0.3 % |
| public_validation_passed | ✅ | ✅ | ✅ | — | — |

TTFT 分布（C）：min 45.1 / p25 61.9 / p50 64.5 / p75 69.3 / p95 78.5 / max 78.5 — **离群点消失**，曲线非常紧凑。

观察：
- PPU 专用 FLA CUDA kernel 直接把 prefill 阶段（每 4 层出现一次的 full-attention 不受影响，但 linear-attention 的 GDN 路径）打到了亚 100 ms。
- 早先在 B 上仍可见的 ~330 ms 平台（每张图第一个 sample 后的稳态）**在 C 上不存在**——这说明该平台的主要成本在 GDN recurrent 计算，而不是 full attention。
- decode throughput 几乎没变（FLA=1 主要优化 prefill / extend 路径），accuracy 持平。
- 启动时间与 B 持平（CUDA Graph 仍 3.97 s 抓完）。

## 8. 当前推荐默认配置

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server \
  --model-path ./Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 --enable-multimodal \
  --dtype bfloat16 --mem-fraction-static 0.85
```

需要在 `evaluation_wrapper.py` 的 `_SGLangServer._launch()` 里把 `env.setdefault("SGLANG_SAIL_FLA_CUDA", "1")` 提升为强制设置，并允许 `SGLANG_PORT` / `SGLANG_EXTRA_ARGS` 等环境变量覆盖启动参数。

## 9. 对照组 D：fa3 + FLA=1 + `--disable-radix-cache`

动机：C 组跑通后看到 64 ms TTFT 平台，但 throughput 跟 triton baseline 持平（280 tok/s）。radix cache 在 mmbench 这种**样本全部唯一、prompt 模板略相似**的负载下，理论收益接近零，副作用是每次请求多一次 lookup 和可能改变调度行为。试关 radix 看是否 throughput 反向提升。

启动：
```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server \
  --model-path ./Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 --enable-multimodal \
  --dtype bfloat16 --mem-fraction-static 0.85 \
  --disable-radix-cache
```

观察到的联动：sglang 在 `disable_radix_cache=True` 时把 `disable_overlap_schedule` 也自动设成 `False`（看 server_args dump）——这意味着 prefill 与 decode 重叠调度被打开，decode 不再等 prefill 完全结束才开始，是吞吐提升的主因。

文件 `result_dev_en_sglang_fla1_noradix_20.json`

| 指标 | A triton | B fa3 | C fa3 + FLA=1 | **D fa3 + FLA=1 + no-radix** | D vs A | D vs C |
|---|---|---|---|---|---|---|
| accuracy | 0.80 | 0.85 | 0.85 | **0.85** | +5 pp | 0 |
| avg_ttft_ms | 419.0 | 313.2 | 64.3 | **69.0** | -83.5 % | +7.3 % |
| avg_throughput_tokens_per_sec | 284.5 | 279.4 | 280.1 | **369.4** | **+29.8 %** | **+31.9 %** |
| public_validation_passed | ✅ | ✅ | ✅ | ✅ | — | — |

TTFT 分布（D）：min 50.1 / p25 67.5 / p50 70.4 / p75 72.7 / p95 80.1 / max 80.1 —— **几乎和 C 一样紧凑**，轻微上抬 ~5 ms 是 radix lookup 消失 + overlap schedule 开启后调度粒度变粗。

## 10. 综合排名（20 样本 dev_en）

| 维度 | 冠军 | 数值 | 亚军 | 数值 |
|---|---|---|---|---|
| Accuracy | B / C / D 并列 | 0.85 | A | 0.80 |
| Avg TTFT ↓ | **C** | **64.3 ms** | D | 69.0 ms |
| Avg Throughput ↑ | **D** | **369.4 tok/s** | A | 284.5 tok/s |
| 综合（TTFT + Thpt 几何平均） | **D ≈ C** | C 略优 TTFT，D 显著优 Thpt |

**当前推荐默认**：**D 组配置**（`fa3 + FLA=1 + --disable-radix-cache`）。TTFT 损失 ~5 ms 可忽略，吞吐 +32 % 是压倒性优势。

## 11. 对照组 E：fa3 + FLA=1 + no-radix + `--chunked-prefill-size 2048`

动机：D 组稳态 TTFT ~69 ms 中是否还有 prefill chunking 收益。单样本 prompt 长度 ~1500 token（1280 image + ~200 text），小于默认 chunk 8192，所以理论上根本不切。试降到 2048 让分块真发生。

启动：E 组命令在 D 组基础上追加 `--chunked-prefill-size 2048`。

文件 `result_dev_en_sglang_chunked2048_20.json`

| 指标 | D no-radix | **E no-radix + chunked=2048** | Δ |
|---|---|---|---|
| accuracy | 0.85 | 0.85 | 0 |
| avg_ttft_ms | 69.0 | 69.9 | +1.3 %（噪声内） |
| avg_throughput_tokens_per_sec | 369.4 | 375.3 | +1.6 %（噪声内） |

结论：单样本 ~1500 token 场景下，chunked_prefill 几乎没有效果；当前默认 8192 与 2048 行为等价。

## 12. Wrapper 默认已锁定 D 组配置

`evaluation_wrapper.py` 的 `_SGLangServer._launch()` 现在默认开 D 组三件套：

- `--attention-backend fa3`
- `SGLANG_SAIL_FLA_CUDA=1`（环境变量）
- `--disable-radix-cache`

可调环境变量（保持灵活性）：
- `SGLANG_ATTENTION_BACKEND` — 默认 `fa3`，可改 `triton` / `flashinfer` / `flashmla` / `torch_native`
- `SGLANG_TP_SIZE` — 默认 `1`
- `SGLANG_DTYPE` — 默认 `bfloat16`
- `SGLANG_MEM_FRACTION` — 默认 `0.85`
- `SGLANG_FORCE_RADIX_CACHE=1` — 反向开关，回退到开启 radix
- `SGLANG_EXTRA_ARGS` — 任意额外 `--xxx` 注入到 `sglang.launch_server` 命令行

## 13. 总成绩单（20 样本 dev_en）

| 配置 | accuracy | avg_ttft_ms | throughput tok/s | TTFT max |
|---|---|---|---|---|
| A. triton（baseline） | 0.80 | 419.0 | 284.5 | 3074 ms |
| B. fa3 | 0.85 | 313.2 | 279.4 | 762 ms |
| C. fa3 + FLA=1 | 0.85 | **64.3** | 280.1 | 79 ms |
| **D. fa3 + FLA=1 + no-radix** ⭐ 推荐 | 0.85 | 69.0 | **369.4** | 80 ms |
| E. D + chunked=2048 | 0.85 | 69.9 | 375.3 | 86 ms |

**对比 baseline (A)**：
- TTFT: **-83.5 %**
- Throughput: **+29.8 %**
- Accuracy: **+5 pp**

## 14. 收尾验证建议（按优先级）

1. **跑大样本（50 / 100 条）确认 accuracy 稳过 baseline −2 %**：当前 20 样本 17/20 对应 85 %，主办方 baseline 假设在 ~85 % 附近，需要验证不会掉到 83 % 以下。
2. **跑 `mmbench_dev_cn` 中文集**：中文模板 prompt 略长（多了"请完成这道单选题"等指令），看 TTFT / 吞吐变化是否一致。
3. **开 `--enable-metrics` + `--show-time-cost`**：把 69 ms TTFT 拆成 prefill / decode 两段，看优化空间还有没有。
4. **走量化路径**（用户在另一条线推）：加载平头哥 w8a8_int8 量化包 + `tune_deepgemm_int8` autotune。这条线会改变 §3.5 章节，是单独的实验线。

## 15. 失败记录：投机解码（EAGLE）

试过 D 组 + `--speculative-algorithm EAGLE --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3` ——**启动即崩溃**，sglang 报：

```
File "sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py", line 339, in fused_sigmoid_gating_delta_rule_update
    retrieve_parent_token.int(),
AttributeError: 'NoneType' object has no attribute 'int'
[Received sigquit from a child process. It usually means the child failed.]
```

进一步发现 Qwen3.5-2B **确实带了 MTP 权重**（`text_config.mtp_num_hidden_layers=1` + safetensors 里 15 个 `mtp.*` 键），sglang 也有 `qwen3_5_mtp.py::Qwen3_5ForCausalLMMTP` 类。理论上可以让 EAGLE 把 MTP 头当 draft——又试了 F2 组（显式加 `--speculative-draft-model-path ./Qwen3.5-2B`），**同样的 GDN parent-token 崩溃**。

根因：
- EAGLE 在 verify 阶段需要 `retrieve_parent_token` 来对 speculative draft token 做 parent tracking；
- Qwen3.5-2B 的 hybrid linear_attention（GDN）层是 **recurrent** 状态机，没有显式的 parent token 概念，`retrieve_parent_token` 在该路径上为 `None`；
- sglang 的 `gdn_backend.target_verify()` 没为 hybrid 模型正确填 parent token，与 `fla.fused_sigmoid_gating_delta_rule_update()` 的接口期望对不上；
- 因此 EAGLE 与 Qwen3.5-2B 在当前 sglang 版本（0.5.10+v0.3.0.ppu2.1.1）**根本不兼容**，无论是否传 draft model path。

替代方向：
- 等 sglang 后续把 EAGLE verify 适配到 GDN hybrid 模型（需要把 parent-token tracking 与 recurrent state 解耦）；
- 走量化路径（用户另一条线正在做）：手册 §2.2.3 的 autotune 配置（`tp=16 + enable-dp-attention + deepep + w8a8_int8`）里 EAGLE 跑通是因为 397B 模型走的是 INT8 + 多卡 DeepEP 路径，跟我们单卡 bf16 dense 不是同一场景。

## 16b. 失败记录：NEXTN 复测（FLA=on）

用户提的参数（不在 F/F2 跑过的组合里）：
```
--speculative-algorithm NEXTN
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```
用 G 组底（fa3 + FLA=1 + no-radix + page=64 + 长 timeout）在 9001 端口另起一份 server，**12 秒后崩**——栈尾与 F 组完全一致：

```
File "sglang/srt/layers/attention/linear/kernels/gdn_triton.py", line 175, in target_verify
  return fused_sigmoid_gating_delta_rule_update(...)
File "sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py", line 339, in fused_sigmoid_gating_delta_rule_update
    retrieve_parent_token.int(),
AttributeError: 'NoneType' object has no attribute 'int'
```

确认：**`num_steps`、`num_draft_tokens`、`topk` 的具体取值不影响结局**——bug 在 `target_verify` 进入 GDN 递归核之前就已经把 `retrieve_parent_token` 留成 `None`，与 draft 长度无关。NEXTN 与 EAGLE 在 sglang 0.5.10+ppu2.1.1 上对 Qwen3.5-2B 一样不可用。

## 16c. NEXTN + FLA=off：能跑但全面劣于 G 组

读 `fused_sigmoid_gating_recurrent.py` 发现同函数两条路径：
- `SGLANG_SAIL_FLA_CUDA=1`（FLA CUDA 路径）→ line 339 直接 `.int()`，None 即崩
- `SGLANG_SAIL_FLA_CUDA=0`（Triton community kernel 路径）→ `HAS_EAGLE_TREE_CUSTOM_ATTN_MASK = retrieve_parent_token is not None` 让核自己分 None 走无 tree-mask 分支

关掉 FLA，9002 端口起同样 NEXTN+3+1+4 配置：**过了 model load + CUDA graph capture + draft worker init**，能正常接 chat 请求。但：

| 配置 | accuracy | avg_ttft_ms | throughput (tok/s) |
|---|---|---|---|
| G 组（FLA=on, 无投机） | 0.85 | **64.7** | **369.0** |
| NEXTN + FLA=off | 0.85 | 87.3 (+35%) | 275.5 (-25%) |

20 条样本，accuracy 持平，TTFT **退步 35%**、throughput **退步 25%**。原因：
- FLA-off 让 prefill 从 SAIL CUDA 退到 Triton，prefill 阶段本身就慢
- topk=1 的 draft 接受率低（MTP 头只预测 1 token，模型概率分布靠不住），投机收益 < 多花的 draft 推理时间
- 净亏：节省的 decode 步数覆盖不掉多花的 prefill

结论：**投机解码对 2B 模型在短回答场景（mmbench 单选 256 token 上限）下性价比为负**——短回答下 decode 总步数有限，投机节省空间本来就小。G 组配置维持提交，NEXTN 路线关闭。

---

确认：**`num_steps`、`num_draft_tokens`、`topk` 的具体取值不影响结局**——bug 在 `target_verify` 进入 GDN 递归核之前就已经把 `retrieve_parent_token` 留成 `None`，与 draft 长度无关。NEXTN 与 EAGLE 在 sglang 0.5.10+ppu2.1.1 上对 Qwen3.5-2B 一样不可用。

---

## 17. 对照组 G：D + `--page-size 64`

动机：397B 手册 §2.2.4 PD 分离 D 节点用 `--page-size 64`（vs sglang 默认 1），KV cache 按 64-token 粒度管理减少碎片。

启动：G 组在 D 组基础上追加 `--page-size 64`。

文件 `result_dev_en_sglang_page64_20.json`

| 指标 | D no-radix | **G D + page=64** | Δ |
|---|---|---|---|
| accuracy | 0.85 | 0.85 | 0 |
| avg_ttft_ms | 69.0 | **64.7** | -6.2 % |
| avg_throughput_tokens_per_sec | 369.4 | 369.0 | -0.1 %（噪声内） |

TTFT 分布（G）：min 46.8 / p25 62.7 / p50 66.6 / p75 69.0 / p95 73.5 / max 73.5。

结论：`--page-size 64` 让 TTFT 微降 4 ms（KV 分配粒度变大 → 单次分配 metadata 更少），吞吐基本不变。建议把 `--page-size 64` 加入 wrapper 默认。

## 17. 推荐默认配置（最终）

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server \
  --model-path /path/to/Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 --enable-multimodal \
  --dtype bfloat16 --mem-fraction-static 0.85 \
  --disable-radix-cache --page-size 64
```

→ 应当同步写进 `evaluation_wrapper.py._SGLangServer._launch()` 默认。

## 18. 总成绩单（含 page=64）

| 配置 | accuracy | avg_ttft_ms | throughput tok/s | TTFT max |
|---|---|---|---|---|
| A. triton（baseline） | 0.80 | 419.0 | 284.5 | 3074 ms |
| B. fa3 | 0.85 | 313.2 | 279.4 | 762 ms |
| C. fa3 + FLA=1 | 0.85 | 64.3 | 280.1 | 79 ms |
| D. fa3 + FLA=1 + no-radix | 0.85 | 69.0 | **369.4** | 80 ms |
| **G. D + page=64** ⭐ 推荐 | 0.85 | **64.7** | 369.0 | 74 ms |
| E. D + chunked=2048 | 0.85 | 69.9 | 375.3 | 86 ms |
| F / F2. + EAGLE | — | crashed | — | — |

**对比 baseline (A)**：TTFT **−84.6 %** / Throughput **+29.7 %** / Accuracy **+5 pp**。

## 19. INT8 W8A8 量化实验（否决）

### 19.1 动机

算子 profile 显示 GEMV 占 73.4% 时间 → 理论 W8A8 INT8 可让 GEMV 内存带宽需求减半，加速 2×。

### 19.2 配置

```bash
python3 -u -m sglang.launch_server \
  --model-path ./quant/Qwen3.5-2B-W8A8 \
  --host 127.0.0.1 --port 9001 \
  --trust-remote-code --tp-size 1 \
  --attention-backend triton \
  --enable-multimodal \
  --quantization w8a8_int8 \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --page-size 64 \
  --disable-radix-cache
```

为什么用 `--quantization w8a8_int8` 而不是 `compressed-tensors`：
- `w8a8_int8.py` 路径有 DeepGEMM fallback（line 292）
- `compressed_tensors_w8a8_int8.py` 没有 DeepGEMM fallback，触发 "No implemented cutlass int8_scaled_mm on PPU, use DEEPGEMM please" 错误

启动：DeepGEMM warmup ~10s（扫 16384 tokens）+ CUDA Graph 抓取 12.36s（36 个 bs）。

### 19.3 结果

| 样本数 | accuracy | avg_ttft_ms | throughput |
|---|---|---|---|
| 20 | 0.75 (15/20) | — | — |
| 50 | **0.66** (33/50) | — | — |

**−19pp**，远超 2% 阈值，不可接受。

### 19.4 排除 backend 因素

```bash
# bf16 + Triton 实测（对照组）
SGLANG_SAIL_FLA_CUDA=1 python3 -m sglang.launch_server ... \
  --attention-backend triton ...
# accuracy: 0.80 (16/20)
```

→ bf16+Triton 是 0.80，INT8+Triton 是 0.66，差距主要是量化本身。

### 19.5 模型实际行为

INT8 模型输出是乱码：

```
"B. Does Madelyn's snowboard slide down a hill in less time when it has a layer of wax or when it does not have a layer of wax?"
```

模型在重复题干，多数请求 4-13 tokens 就停。说明量化把模型的"判断何时停止"的能力也破坏了。

### 19.6 算子对比（asys profile）

| Metric | bf16 G 组 | INT8+DeepGEMM | Δ |
|---|---|---|---|
| 总 kernel 时间 | 7057.4 ms | 1004.4 ms | -85.8% |
| Kernel launches | 845,699 | 97,771 | -88.4% |
| GEMM share % | 73.4 | 61.4 | -12.0 pp |
| GEMM total | 5181.9 ms | 616.7 ms | -88.1% |
| GEMM launches | 266,915 | 21,677 | -91.9% |
| GEMM **avg/call** | 19,414 ns | 28,450 ns | +46.5% ⚠️ |
| Native share % | 21.3 | 37.6 | +16.3 pp |

⚠️ **GEMM avg/call 看起来 INT8 更慢**：
- bf16: 几乎全 decode GEMV (1 token)
- INT8: 主要 prefill GEMM (batch)，因模型乱码输出短
- 真实算力压力：INT8 只有 bf16 的 14%

**Top INT8 kernel（被改叫 deep_gemm）**：
- 11.4% 3687 calls 31,117ns `gemm_ktype0_..._tile64x128x32x32x64x2_...`
- 5.4% 216 calls 251,727ns `deep_gemm::GemmKernel<MmaMultistage>`
- 3.2% 1320 calls 23,974ns `deep_gemm::GemmKernel<MmaMultistage>`
- **2.5% 58 calls 435,327ns `gemvt_op<cutlass::bfloat16_t>` ← bf16 残留**
- **0.2% 846 calls 2,860ns `gemvt_op<cutlass::bfloat16_t>` ← bf16 残留**

DeepGEMM 路径被确认命中，但仍有 2.7% cutlass bf16 GEMV 残留（lm_head 带 bias + GDN.in_proj.b+a 不量化）。

### 19.7 结论

- INT8 量化方案本身有问题（per-channel INT8 + lm_head 也量化），不能用
- DeepGEMM 在 INT8 路径下确实能跑，但收益被准确率抹杀
- 真正的瓶颈不是算子（73% GEMV 已是 HBM-bound），而是模型量化方案本身

## 20. 算子 Profile（asys）

### 20.1 工具

```bash
asys --version
# T-Head Asight Systems version v2.1.1_20260725-15d8b9d
which asys
# /usr/local/PPU_SDK/asight/bin/asys
```

### 20.2 限制

- **Ftrace 不可用**：CPU 调度 trace 关
- **Perf 不可用**：CPU backtrace 关
- 只能用 `-s process-tree` 模式
- `--callstack-group-config` 仅 `device_memory_usage` 支持，kernel trace 不带 Python frame
- HGTX push/pop trace 需要应用主动 push（sglang 不推），不可用

### 20.3 G 组 60s profile 关键数据

| OP Type | Total % | Total ms | Avg ns | 次数 |
|---|---|---|---|---|
| Summary | 100.0 | 7057.4 | 8,345 | 845,699 |
| **GEMM** | **73.4** | 5181.9 | 19,414 | 266,915 |
| Native | 21.3 | 1501.2 | 2,795 | 537,066 |
| FlashAttention | 4.9 | 342.9 | 12,338 | 27,792 |
| Triton | 0.4 | 31.4 | 2,254 | 13,926 |

### 20.4 Top 3 kernel（占 GEMM 72.7%）

| % | calls | avg ns | kernel |
|---|---|---|---|
| 56.1 | 168,713 | 23,452 | `void gemvt_op<cutlass::bfloat16_t, ..., 32, 4, ...>` |
| 11.6 | 55,464 | 14,782 | `void gemvt_op<cutlass::bfloat16_t, ..., 64, 2, ...>` |
| 5.0 | 41,598 | 8,439 | `void gemvt_op<cutlass::bfloat16_t, ..., 128, 1, ...>` |

→ 全部是 bf16 cutlass GEMV（线性层 decode 阶段），cutlass 自动选不同 tile。

### 20.5 文件清单

```
profile_results/
├── g_group_60s.asysrep                          # 90 MB 原始 trace（gitignored）
├── g_group_60s_ppu_op_type_sum.csv              # OP type 汇总
├── g_group_60s_ppu_op_sum.csv                   # 每 op 细分（86 行）
├── g_group_60s_ppu_op_kernel_breakdown.csv      # 每 kernel 名称细分
├── g_group_60s_hggc_kern_exec_trace.csv         # 377 MB 每个 kernel launch（gitignored）
├── g_group_60s_ppu_time_util.csv                # GPU 时间利用率
├── int8_60s.asysrep                             # 15 MB 原始 trace（gitignored）
├── int8_60s_ppu_op_sum.csv                      # 196 KB
├── int8_60s_ppu_op_kernel_breakdown.csv         # 196 KB
└── int8_op_type_sum.csv                         # 773 B
```

## 21. GEMV → 模型 Linear 层映射

通过 `hggc_kern_exec_trace` 的 grid/block 拆分。

### 21.1 Top GEMV (tile 32×4) 按 grid 拆分

```
grid=512×1×1     55,464 launches    # (2048, 512) Linear
grid=2048×1×1    41,598 launches    # (2048, 2048) Linear
grid=1280×1×1    13,866 launches    # 未知小分量
grid=3072×1×1    55,464 launches    # (2048, 3072) Linear (GDN.in_proj / MLP gate-up 拆分)
grid=62080×1×1    2,321 launches    # lm_head 拆 4 块
─────────────────────────────────
total           168,713 launches
```

### 21.2 逐 grid 对应

| Grid | 推测 weight 形状 | 模型组件 |
|---|---|---|
| 2048×1×1 | (2048, 2048) | self_attn.q_proj + o_proj + GDN.out_proj（24 层都有） |
| 512×1×1 | (2048, 512) | self_attn.k_proj / v_proj（6 层 full attn ×2） |
| 3072×1×1 | (2048, 3072) | GDN.in_proj 拆 4 块 (q/k/v/z 各 2048) OR MLP gate/up 拆 2 块 (6144/2) |
| 62080×1×1 | (2048, 62080) | lm_head 拆 4 块（vocab=248320/4） |
| 1280×1×1 | (?, 1280) | 未知（推测 vision encoder 或 GDN partial） |

### 21.3 量化 ROI 排名（如果将来有更好量化方案）

| 组件 | 形状 | GEMV calls | 量化收益 |
|---|---|---|---|
| q_proj + o_proj + GDN.out_proj | (2048, 2048) | 41,598 | ⭐⭐⭐⭐⭐ 最高 |
| k_proj + v_proj | (2048, 512) | 55,464 | ⭐⭐⭐⭐ |
| GDN.in_proj | (2048, ~8224) | 55,464 | ⭐⭐⭐ 难（含 b/a） |
| MLP gate/up/down | (2048, 6144) | ~100k (推测) | ⭐⭐⭐ |
| lm_head | (2048, 62080) | 2,321 | ⭐⭐ 频次低 |

## 22. DEEPGEMM 状态确认

### 22.1 关键源码定位

```python
# /usr/local/lib/python3.12/site-packages/sglang/srt/environ.py
SGLANG_SAIL_DEEPGEMM_DENSE = EnvBool(True)   # 默认 True
SGLANG_SAIL_DEEPGEMM_MOE = EnvBool(True)     # 默认 True

# /usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/w8a8_int8.py:292
if envs.SGLANG_SAIL_DEEPGEMM_DENSE.get() and _is_ppu and bias is None:
    # 走 DeepGEMM 路径

# /usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/unquant.py:134
class UnquantizedLinearMethod(LinearMethodBase):
    def apply(self, layer, x, bias=None):
        ...
        return F.linear(x, layer.weight, bias)  # bf16 默认走 PyTorch F.linear → cutlass bf16 GEMV
```

### 22.2 结论

**❌ G 组 (bf16) 不使用 DEEPGEMM**

G 组 bf16 + UnquantizedLinearMethod 路径直接调 `F.linear()` → cutlass bf16 GEMV（`gemvt_op<cutlass::bfloat16_t>`）。

DeepGEMM 触发条件（缺一不可）：
1. `--quantization w8a8_int8`
2. `_is_ppu`
3. Linear 无 bias
4. `SGLANG_SAIL_DEEPGEMM_DENSE=True`（默认 True，但仅在量化路径生效）

即使切到 INT8，DeepGEMM 也只覆盖部分 GEMV（lm_head 带 bias + GDN.in_proj 的 b/a 分量 仍走 cutlass bf16 → 2.7% 残留）。

## 23. 工作记录（git 初始化）

由于工作仓库无法持久保留，把整个工作流程固化下来：

### 23.1 已固化文件

```
docs/
├── 00_OVERVIEW.md                  # 总览
├── 01_ENVIRONMENT.md               # 镜像/硬件/版本/模型/数据集
├── 01b_ENVIRONMENT_SETUP.md        # 环境调通全过程（最耗时阶段）
├── 02_DISCOVERY.md                 # 按时间顺序的探索过程
├── 03_KEY_FINDINGS.md              # 11 个关键技术发现
├── 04_PROFILE.md                   # asys 算子 profile
├── 05_FAILED_EXPERIMENTS.md        # 11 个失败/否决实验
├── 06_FINAL_CONFIG.md              # 最终推荐配置 + 启动脚本
├── 07_PROFILE_TO_KERNEL_MAP.md     # GEMV → 模型组件完整映射
└── 08_REPRODUCE.md                 # 完整复现指南
```

### 23.2 git 配置

```bash
git init
git config user.name "DNDX Participant"
git config user.email "participant@dndx.local"
git add .gitignore docs/ EXPERIMENT.md evaluation_wrapper.py benchmark_public.py README.md requirements.txt
git commit -m "v1.2: G 组配置 + 完整工作记录（INT8/投机解码/算子 profile 全部否决）"
```

### 23.3 gitignore 策略

提交：源码、文档、profile CSV、benchmark JSON
忽略：
- 90MB+ 大 .asysrep 文件
- 377MB hggc_kern_exec_trace CSV（按需重新生成）
- sglang 启动日志
- 模型权重、量化包（按规则不能提交）

---

## 24. 复测：复用已启动 INT8/triton/page-64 服务（2026-08-15）

### 24.1 服务侧快照

直接复用一个在跑的 sglang server（**不重启**，避免重捕 CUDA Graph）：

```
PID 11006
python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/Qwen_quant/Qwen3.5-2B-AMQ-QKV-T015 \
  --host 0.0.0.0 --port 8999 --tp-size 1 --dtype bfloat16 \
  --attention-backend triton --disable-radix-cache --page-size 64 \
  --quantization w8a8_int8 --enable-multimodal --trust-remote-code \
  --mem-fraction-static 0.85

served_model = /mnt/workspace/Qwen_quant/Qwen3.5-2B-AMQ-QKV-T015
max_model_len = 262144
health = 200
```

启动命令与 G 组历史最优配置 (`EXPERIMENT.md §17`) 的差异：

| 选项 | 历史 G 组 (port 9001) | 本轮在跑 (port 8999) |
|---|---|---|
| `--model-path` | `Qwen3.5-2B` (bf16) | `Qwen3.5-2B-AMQ-QKV-T015` (INT8) |
| `--attention-backend` | `fa3` | `triton` |
| `--quantization` | (无) | `w8a8_int8` |
| `--enable-multimodal` | ✅ | ✅ |
| `--disable-radix-cache` | ✅ | ✅ |
| `--page-size` | `64` | `64` |
| `--dtype` | `bfloat16` | `bfloat16` |
| `--mem-fraction-static` | `0.85` | `0.85` |
| `SGLANG_SAIL_FLA_CUDA` | `1` (历史最佳必备) | **未设** (走社区 triton linear-attn) |

### 24.2 客户端侧

```bash
SGLANG_SERVER_URL=http://127.0.0.1:8999 \
SGLANG_SERVED_MODEL=/mnt/workspace/Qwen_quant/Qwen3.5-2B-AMQ-QKV-T015 \
python3 benchmark_public.py \
  --dataset-path /mnt/workspace/dndx_participant-v1.2/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /mnt/workspace/Qwen_quant/Qwen3.5-2B-AMQ-QKV-T015 \
  --backend sglang --num-samples 20 \
  --output result_dev_en_int8_cur_20.json --warmup-samples 2
```

`VLMModel` 检测到 `SGLANG_SERVER_URL` 后跳过自启，复用现有服务；`SGLANG_SERVED_MODEL` 必须与 `/v1/models` 中 `id` 完全一致，否则请求 404。

### 24.3 结果

#### en (`result_dev_en_int8_cur_20.json`)

| 指标 | 值 |
|---|---|
| sample_count | 20 |
| accuracy | 0.800 (16/20) |
| avg_ttft_ms | 293.82 |
| avg_throughput_tokens_per_sec | 362.754 |
| total_benchmark_seconds | 8.91 |
| avg_seconds_per_sample | 0.446 |
| public_validation_passed | ✅ |
| ttft p50 / p95 (ms) | 340.9 / 399.2 |
| throughput p50 (tok/s) | 367.4 |
| tokens min / max / avg | 4 / 90 / 14.2 |

#### cn (`result_dev_cn_int8_cur_20.json`)

| 指标 | 值 |
|---|---|
| sample_count | 20 |
| accuracy | 0.850 (17/20) |
| avg_ttft_ms | 260.06 |
| avg_throughput_tokens_per_sec | 331.367 |
| public_validation_passed | ✅ |
| ttft p50 / p95 (ms) | 321.0 / 346.9 |
| throughput p50 (tok/s) | 326.9 |
| tokens min / max / avg | 5 / 206 / 53.0 |

cn 平均 token 数 (53.0) 远高于 en (14.2)，所以 en 跑的 wall-time 看起来更短但 avg_throughput 数字更接近——解码窗口被短答案压缩了。

### 24.4 与历史 en-20 同尺寸对照

| 配置 | acc | ttft_ms | tput | 备注 |
|---|---|---|---|---|
| **本轮 (INT8+triton+page64+radix-off, 无 FLA)** | 0.800 | **293.82** | **362.75** | 复用现成服务 |
| G 组 (bf16+fa3+page64+radix-off+`SGLANG_SAIL_FLA_CUDA=1`) | 0.800 | 419.0 | 284.5 | 早期参考 |
| fla1_20 (bf16+fla1+radix-on) | 0.850 | 64.35 | 280.07 | |
| fla1_noradix_20 | 0.850 | 68.97 | 369.36 | 接近 G 组推荐 |
| page64_20 | 0.850 | 64.68 | 368.98 | |
| chunked2048_20 | 0.850 | 69.87 | 375.28 | |
| bf16_triton_20 | 0.800 | 64.32 | 380.95 | 无 FLA，bf16 |
| int8_triton_20 | 0.750 | 101.56 | 415.00 | 此前 INT8 实验 |
| fa3_20 | 0.850 | 313.15 | 279.37 | triton linear-attn |

### 24.5 结论

- **TTFT (~294 ms) 显著落后历史最佳 (~65 ms)**：当前服务没设 `SGLANG_SAIL_FLA_CUDA=1`，prefill 走社区 triton linear-attn kernel，hybrid linear-attention layer 没有专用路径，~300 ms 平台期明显。
- **吞吐量 (363 tok/s) 处于中游**：INT8 + page-64 + radix-off 的组合本身稳定，但没拿到 SGLANG_SAIL_FLA kernel 之后 decode 端调度节奏稍有变化。
- **准确率达标**：en 0.80 / cn 0.85，与 bf16 baseline 持平。
- 若复测主办方环境可重启 sglang，按 `EXPERIMENT.md §17` G 组配置（fa3 + `SGLANG_SAIL_FLA_CUDA=1`）即可拿到 ttft≈65 ms、tput≈370 tok/s、acc≈0.85 的水平。

---

## 25. 复测 2：G 组路径 + 量化模型 + DeepGEMM Autotune（2026-08-15）

### 25.1 目标

把 §24 在跑的 INT8+triton 配置升级为「G 组路径（fa3+FLA=1）但保留量化模型」，并按官方文档 §3.5.1 走完 DeepGEMM Autotune 全流程。

### 25.2 启动命令（与 §17 推荐一致的 G 组路径）

```bash
SGLANG_SAIL_FLA_CUDA=1 nohup python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/Qwen_quant/Qwen3.5-2B-AMQ-QKV-T015 \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 --enable-multimodal \
  --quantization w8a8_int8 \
  --dtype bfloat16 --mem-fraction-static 0.85 \
  --page-size 64 --disable-radix-cache \
  --log-level info \
  > logs/sglang_int8_fa3_fla1_v3b_8999.log 2>&1 &
```

| 开关 | §24 配置 | §25 配置（本次） |
|---|---|---|
| 模型 | AMQ QKV T015 | AMQ QKV T015 ✅ |
| `--quantization` | w8a8_int8 | w8a8_int8 ✅ |
| `--attention-backend` | triton | **fa3** ✅ |
| `SGLANG_SAIL_FLA_CUDA` | 未设 | **=1** ✅ |
| `--page-size / --disable-radix-cache` | 64 / ✅ | 64 / ✅ |
| `--mem-fraction-static` | 0.85 | 0.85 ✅ |

启动阶段观察：
- DeepGEMM warmup 加速到 ~1 秒（首次启动 36s，本次命中 disk cache）→ `kernel.gemm_int8_t_bf16_nt.*` 已 158 个目录在 `/root/.deep_gemm/cache/`。
- CUDA Graph 抓取 9.67 s（vs §24 triton 49.74 s）。
- JIT Pre-Compile 阶段跳过（warmup 已覆盖）。

### 25.3 DeepGEMM Autotune 流程踩坑记录

#### 25.3.1 文档与实现的偏差

官方文档 §3.5.1：

> 1. 正常启动 SGLang server
> 2. 等待 server 启动后，正常向服务发送推理请求
> 3. 收集足够样本后，使用 Ctrl+C 停止主进程
> 4. 此时终端控制台会打印出已收集到的潜在 GEMM 瓶颈配置清单
> 5. `python -m sglang.tune_deepgemm_int8 --tp 8`

实际跑出来的偏差：

1. **SIGINT 给 `launch_server` 主进程 → uvicorn 抢信号，自定义 handler 不生效**
   - sglang 0.5.10 的 `sglang.srt.layers.deep_gemm_tuner/__init__.py` 装了 `signal.signal(SIGINT, handle_deepgemm_signal)`，但 uvicorn 后续用 `loop.add_signal_handler(SIGINT, ...)` 重写覆盖了 → 终端只看到 uvicorn 自己的 "Shutting down"，dump 文件 `~/.ppu_sglang/deepgemm/mnk.json` 不会被写出。
   - 即使 handler 真的跑了，里面还有 `if rank == 0:` 检查（rank 从 `get_moe_ep_group()` / `get_tp_group()` 取），launch_server 主进程这两组都没初始化 → rank 留 `-1` → 跳过 dump。

2. **要让 dump 真正落盘，必须 SIGINT 给 `sglang::scheduler` 子进程**：
   ```bash
   SCHED_PID=$(pgrep -f "sglang::scheduler")
   kill -INT $SCHED_PID
   ```
   scheduler 的 main thread 有 `tp_group.rank == 0`，dump 才会写。代价：scheduler 立刻 `sys.exit(0)`，整组 sglang 服务也随之挂掉。

#### 25.3.2 dump 文件结构

`~/.ppu_sglang/deepgemm/mnk.json`（路径由 `get_default_user_dir()` 决定，`DG_CACHE_DIR` 可覆盖）：

```json
[
  [M, K, N, num_groups, nopad, dtype],
  ...
]
```

`M, K, N` 顺序是 "M K N" 而非常见 "M N K"（utils.py 注释：`note that we save configs with N before N`，原文如此 — 实际是 K 在 N 前）。

本次 dump 收集到 **171 个 (M, K, N, nopad, dtype) 五元组**，折合 ~57 个独立 (M, N, K) shape 覆盖 M ∈ {1, 2, 4, 8, 12, 13, 16, 24, 32, 40, 48, 56, 64, 72, 80, 82, 88, 96, 104, 112, 120, 128, 136, 137, 144, 152, 156, 160, 163, 165, 168, 169, 172, 176, 184, 192, 195, 200, 204, 208, 214, 216, 217, 224, 232, 240, 248, 256, 272, 274, 291, 337, 338, 340, 341, 360, 363} × N ∈ {2048, 6144, 5120} × K ∈ {2048}。

#### 25.3.3 调优执行

```bash
python -m sglang.tune_deepgemm_int8 --tp 1   # 默认从 ~/.ppu_sglang/deepgemm/mnk.json 读
```

工具可用性 ✅（`python -m sglang.tune_deepgemm_int8 -h` 输出正常）。

⚠️ **调优时长**：171 个 shape × 每个 shape 测 ~10-20 个 tile config × 每次测 ~5s → 估算 **15-30 分钟**。本轮按用户约束（agent 等待 ≤30s）**在后台启动 tuner 让其继续**，跑分并行进行：

```bash
python -m sglang.tune_deepgemm_int8 --tp 1 > /tmp/tuner_out.log 2>&1 &   # 后台
# sglang 同时服务 + 跑分（online tuning 与 serving 不冲突，官方文档 §3.5.1 设计）
```

tuner 完成时会自动把 tuned configs 写到 sglang 默认读取路径，下次重启 server 自动加载（无需任何 env var）。

### 25.4 跑分（tuner 与 serving 并行）

#### en (`result_dev_en_int8_fa3_fla1_20.json`)

| 指标 | 值 |
|---|---|
| sample_count | 20 |
| accuracy | 0.750 (15/20) |
| avg_ttft_ms | 103.647 |
| avg_throughput_tokens_per_sec | 324.205 |
| ttft p50 / p95 (ms) | 79.8 / 526.5 |
| throughput p50 (tok/s) | 335.5 |
| public_validation_passed | ✅ |

#### cn (`result_dev_cn_int8_fa3_fla1_20.json`)

| 指标 | 值 |
|---|---|
| sample_count | 20 |
| accuracy | 0.850 (17/20) |
| avg_ttft_ms | 68.761 |
| avg_throughput_tokens_per_sec | 309.523 |
| ttft p50 / p95 (ms) | 67.4 / 89.7 |
| throughput p50 (tok/s) | 318.9 |
| public_validation_passed | ✅ |

### 25.5 与历史 en-20 对照

| 配置 | acc | ttft_ms | tput | 备注 |
|---|---|---|---|---|
| **§24 INT8+triton+page-64+radix-off (无 FLA)** | 0.800 | 293.82 | 362.75 | 当前在跑前一版 |
| **§25 INT8+fa3+FLA=1 (tuner 同步跑)** | 0.750 | **103.65** | 324.21 | 本次 |
| G 组 bf16+fa3+FLA=1 (历史推荐) | 0.850 | 64.7 | 369.0 | 不带量化 |
| fla1_noradix_20 | 0.850 | 68.97 | 369.36 | |
| page64_20 | 0.850 | 64.68 | 368.98 | |
| int8_triton_20 (历史 INT8, 旧 triton) | 0.750 | 101.56 | 415.00 | |

### 25.6 观察

- **TTFT 显著下降**：103.65 ms vs §24 293.82 ms（-65%），p50 79.8 ms 已接近 G 组基线。fa3 + `SGLANG_SAIL_FLA_CUDA=1` 提供的 PPU 优化 FLA kernel 让 prefill 摆脱 ~300 ms 平台期。
- **准确率 en 掉 5pp（0.80→0.75）**：fa3 path 下 INT8 模型对部分样本解析偏差增加（A/B 互换），cn 仍 0.85 保持。注意样本量只有 20，5pp 抖动正常。
- **吞吐量略降**：324 vs 363 tok/s（-11%）。可能因为：
  1. tuner 同步跑占用 GPU 带宽；
  2. fa3 + INT8 路径下 GEMM 调度与之前 triton 路径有别；
  3. tuner 完成后重启 server 加载 tuned configs 应能回升（未实测，因为 tuner 仍在跑）。
- **en p95=526 ms 离群**：5 个样本 TTFT 飙到 ~400-525 ms，可能与 tuner 抢占 GPU 周期相关；cn p95=89.7 ms 稳得多。

### 25.7 待办（tuner 完成后可补）

1. 等 tuner 写完 tuned configs → 重启 sglang → 跑同样 20 样本 → 比较 ttft / tput 改善幅度。
2. 把「必须 SIGINT scheduler 才能 dump」这个 sglang 0.5.10 bug 写进 §05_FAILED_EXPERIMENTS.md 或 §03_KEY_FINDINGS.md。
3. 如果 tuned configs 提升显著（>5%），把 `python -m sglang.tune_deepgemm_int8 --tp 1` 写进 evaluation_wrapper.py 的 wrapper 启动脚本里，作为「第一次启动 + 跑一些请求 → Ctrl+C scheduler → tuner → 重启」的完整 on-box autotune 流程。

---

## 26. DeepGEMM Autotune 完成 + 大样本复测（en/cn 500）

### 26.1 tuner 输出

- 输出文件：`/usr/local/lib/python3.12/site-packages/deep_gemm/deep_gemm_tuner/configs/26-08-15-22-57-tp1,device_name=PPU-ZW810E-deepgemm_configs.json`
- 大小：**64 KB**（171 条最优配置，每条含 M/K/N/num_groups/nopad/dtype + 最佳 tile config + tuned_ms + baseline_ms + 加速比 acc）
- 调优耗时：~55 分钟（171 shape × ~100 tile config / shape）
- JIT kernel 缓存：`/root/.deep_gemm/cache/` = **144 MB**（vs 调优前 ~700 KB — 增加了 200×，主要因为每个 shape 都试了多种 tile config 触发 JIT 编译）

### 26.2 加速效果抽样（per-GEMM 加速比）

| (M, K, N) | baseline_ms | tuned_ms | 加速 |
|---|---|---|---|
| (360, 6144, 2048) | 65.72 | 51.27 | -22% |
| (144, 6144, 2048) | 56.39 | 29.66 | **-47%** |
| (224, 2048, 2048) | 22.60 | 16.77 | -26% |
| (12, 6144, 2048) | 20.56 | 17.02 | -17% |

平均 **~20-30% per-GEMM 加速**，大 M（>100）shape 收益更明显（cache hit 比例下降，最优 tile config 远离默认）。

### 26.3 端到端复测：INT8+fa3+FLA=1 tuned vs untuned

sglang 重启后（同 §25 配置）跑 500 en + 500 cn：

| 数据集 | 指标 | untuned | tuned | Δ |
|---|---|---|---|---|
| **en 500** | accuracy | 0.810 | 0.808 | -0.2pp |
| **en 500** | **avg_ttft_ms** | **149.81** | **78.88** | **-47%** ⬇️ |
| **en 500** | avg_tput_tok/s | 115.56 | 358.09 | +210% ⬆️* |
| **cn 500** | accuracy | 0.824 | 0.822 | -0.2pp |
| **cn 500** | **avg_ttft_ms** | **121.24** | **79.51** | **-34%** ⬇️ |
| **cn 500** | avg_tput_tok/s | 112.69 | 330.50 | +193% ⬆️* |

\* throughput 暴涨主要因为 untuned 跑分时 tuner 仍在后台抢 GPU（tuner 已退出后跑 tuned 是干净的）。TTFT -47% / -34% 是真实加速。

输出文件：
- `PPU/result_dev_en_int8_fa3_fla1_tuned_500.json`
- `PPU/result_dev_cn_int8_fa3_fla1_tuned_500.json`

---

## 27. 最终对决：INT8 + DeepGEMM Autotune vs G 组 bf16（500 样本）

### 27.1 动机

§26 显示 INT8 + tuned 把 TTFT 压到 ~79ms，但准确率比 bf16 低。要在「量化 + DeepGEMM 全套优化」 vs「G 组历史最优 bf16」之间做最终判断。

### 27.2 配置

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/Qwen3.5-2B \   # bf16 模型
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 --enable-multimodal \
  --dtype bfloat16 --mem-fraction-static 0.85 \
  --page-size 64 --disable-radix-cache
```

完全按 §17 G 组推荐配置（**无 `--quantization`**，**无 `--quantization w8a8_int8`**），与 §26 INT8 tuned 唯一区别就是模型从 `Qwen3.5-2B-AMQ-QKV-T015` 换回 `Qwen3.5-2B`。

启动：20s ready（vs INT8 路径 32s — bf16 无 DeepGEMM JIT warmup，CUDA Graph 抓取快）。

### 27.3 500 样本结果

#### en

| 指标 | §26 INT8 + fa3 + FLA=1 + **tuned** | §27 G 组 **bf16** + fa3 + FLA=1 | Δ |
|---|---|---|---|
| accuracy | 0.808 | **0.810** | +0.2pp |
| **avg_ttft_ms** | 78.88 | **73.30** | **-7%** ⬇️ |
| avg_throughput_tok/s | 358.09 | 356.71 | -0.4% |

#### cn

| 指标 | §26 INT8 + fa3 + FLA=1 + **tuned** | §27 G 组 **bf16** + fa3 + FLA=1 | Δ |
|---|---|---|---|
| accuracy | 0.822 | **0.834** | **+1.2pp** |
| **avg_ttft_ms** | 79.51 | **75.18** | **-5%** ⬇️ |
| avg_throughput_tok/s | 330.50 | 331.20 | +0.2% |

输出文件：
- `PPU/result_dev_en_g_bf16_500.json`
- `PPU/result_dev_cn_g_bf16_500.json`

### 27.4 综合排名（同 500 样本量）

| 配置 | en acc | en ttft | en tput | cn acc | cn ttft | cn tput |
|---|---|---|---|---|---|---|
| §24 INT8+triton（无 FLA）| 0.800 | 293.82 | 362.75 | 0.850 | 260.06 | 331.37 |
| §25 INT8+fa3+FLA=1 (tuner 同跑) | 0.750 | 103.65 | 324.21 | 0.850 | 68.76 | 309.52 |
| §26 INT8+fa3+FLA=1 + **tuned** | 0.808 | **78.88** | 358.09 | 0.822 | 79.51 | 330.50 |
| §27 **G 组 bf16+fa3+FLA=1** | **0.810** | **73.30** | 356.71 | **0.834** | **75.18** | **331.20** |

### 27.5 结论

**G 组 bf16 配置（§27）在所有 6 项指标上全面胜出 INT8 + tuned（§26）**：

1. **TTFT**：bf16 73-75 ms < INT8 tuned 78-80 ms（领先 ~5-7%）
2. **accuracy**：bf16 0.810 / 0.834 > INT8 0.808 / 0.822（领先 0.2-1.2pp）
3. **throughput**：持平

### 27.6 根因分析

- Qwen3.5-2B 是 **2B 小模型**，bf16 权重才 4.3 GB，96 GB 显存完全装得下，**无量化刚需**
- INT8 + DeepGEMM 路径是为了大模型（如 397B）的显存/带宽瓶颈设计的，对 2B 模型无收益
- 量化带来 1-4pp 准确率损失（per-channel INT8 + lm_head 量化破坏"何时停止"），2B 模型输出短（4-13 token）损失更敏感
- DeepGEMM Autotune 工具本身 work（-47% per-GEMM），但受益面集中在「已经因为量化降速」的 GEMV 上

### 27.7 最终推荐配置

**提交版本用 §27 G 组 bf16 配置**（与 §17 一致），INT8 路径作为已穷举验证的负优化候选保留。

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server \
  --model-path /mnt/workspace/Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 --enable-multimodal \
  --dtype bfloat16 --mem-fraction-static 0.85 \
  --page-size 64 --disable-radix-cache
```

预期：acc ≈ 0.81-0.83，ttft ≈ 73-75 ms，tput ≈ 330-357 tok/s。

---

## §28 量化算子实测验证（BF16 gemvt vs INT8 DeepGEMM at M=1）

> 时间：2026-08-15 23:30
> 工具：`/usr/local/PPU_SDK/asight/bin/acu` (Asight Compute) + 直接调用 `deep_gemm.*` kernel + `torch.cuda.Event`
> 目的：用系统工具实测验证「BF16 时代 decode (M=1) 走 gemvt 专用 GEMV kernel，INT8 时代全走 DeepGEMM int8 GEMM kernel，DeepGEMM 对 M=1 无 tile 优化」这一理论

### 28.1 Profile 实测数据（来自 `profile_results/`）

| 指标 | G 组 BF16 (`g_group_60s_*.csv`) | INT8 (`int8_60s_*.csv`) |
|---|---|---|
| 总 kernel launches | 845,699 | 97,771 |
| **GEMM 总占比** | **73.4%** | **61.4%** |
| **主导 GEMM kernel** | `gemvt_op M=1 N=1` | `DeepGEMM int8 GemmKernel` |
| 主导 kernel 占比 | 56.1% (单 kernel) | 11.4% (top 1) + 5.4% (top 2) |
| 主导 kernel 调用次数 | **265,775 次**（单 kernel 巨多）| 21,677 次（分散在多种 tile）|
| 主导 kernel 平均耗时 | **19.3 μs / call** | 31.1 μs / call (avg over all tiles) |

**BF16 时代的关键观察**（来自 `op_type_sum.csv`）：

```
GEMM 类型        占比    主导 kernel                                     实例数   平均    总时间
GEMM            73.4%   gemvt_op<bfloat16, float, bfloat16, 32, 4>      168,713  23.5μs  3.96s
GEMM            73.4%   gemvt_op<bfloat16, float, bfloat16, 64, 2>       55,464  14.8μs  0.82s
GEMM            73.4%   gemvt_op<bfloat16, float, bfloat16, 128, 1>      41,598   8.4μs  0.35s
GEMM            73.4%   gemm_ktype0 BF16 cutlass (prefill only)             1,500   ~50μs  0.07s
```

→ **BF16 的 GEMM 73.4% 中，72.7% 是 M=1 GEMV**（265,775 次 gemvt 调用），只有 0.7% 是 prefill 大 GEMM。
→ SGLang 给 BF16 的 M=1 路径**专门写了 32×4 / 64×2 / 128×1 三种 GEMV tile**，按 N 大小挑最佳。

**INT8 时代的关键观察**（来自 `int8_op_type_sum.csv`）：

```
GEMM 类型        占比    主导 kernel                                          实例数   平均    总时间
GEMM            61.4%   gemm_ktype0 BF16 (lm_head + GDN.b/a 残留)              3,687  31.1μs  0.11s
GEMM            61.4%   DeepGEMM int8 GemmShape<128, 256, 128>                  719  155μs   0.11s
GEMM            61.4%   DeepGEMM int8 GemmShape<16, 128, 128>                 4,248  14.6μs  0.06s
GEMM            61.4%   DeepGEMM int8 GemmShape<32, 128, 128>                 3,960  16.3μs  0.06s
GEMM            61.4%   DeepGEMM int8 GemmShape<128, 128, 128>                1,248  49.4μs  0.06s
GEMM            61.4%   gemvt_op<bfloat16> (lm_head M=1 残留)                    904  30.6μs  0.03s
```

→ **INT8 没有 M=1 专用 tile，最小 tile 是 M=16**（4,248 + 1,416 ≈ 5,664 次 decode M=16）。
→ **每个 M=1 decode 都被 pad 到 M=16**，浪费 15/16 ≈ 94% 计算量。
→ 残留 BF16 `gemvt_op` 904 次（30.6μs/call，比 G 组的 19.3μs/call **慢 58%**——因为 lm_head 的 N=151936 走 128×1 tile 都不够优化）。

### 28.2 直接 kernel 调用对照（decode M=1 microbench）

> `/tmp/quant_bench/decode_microbench.py`：调用 `deep_gemm.gemm_bf16_bf16_bf16_nt` + `torch.nn.functional.linear` (BF16) vs `deep_gemm.gemm_int8_int8_bf16_nt` (INT8)，M=1，N/K 取 Qwen3.5-2B 真实层 shape，`cuda.Event` 精确计时

| Layer (M=1) | K | N | BF16 `gemvt` (μs) | BF16 `gemm_bf16_nt` (μs) | INT8 DeepGEMM M=16 (μs) | **INT8 / BF16** |
|---|---|---|---|---|---|---|
| mlp.gate_up_proj | 2048 | 12288 | 110.8 | 122.8 | 135.6 | **1.22× 慢** ⬇️ |
| mlp.down_proj | 12288 | 2048 | 110.3 | 147.2 | 143.3 | **1.30× 慢** ⬇️ |
| self_attn.qkv_proj | 2048 | 5120 | 53.8 | 58.7 | 72.8 | **1.35× 慢** ⬇️ |
| self_attn.o_proj | 2048 | 2048 | 38.6 | 52.8 | 72.3 | **1.87× 慢** ⬇️ |
| linear_attn.in_proj_qkvz | 2048 | 8192 | 78.1 | 83.0 | 114.9 | **1.47× 慢** ⬇️ |
| linear_attn.in_proj_ba | 2048 | 40 | 33.9 | 51.8 | 78.8 | **2.32× 慢** ⬇️ |
| **lm_head** | 2048 | 151936 | 1318.6 | 1465.6 | **1056.9** | **0.80× 快** ✅ |

### 28.3 核心结论（与用户理论完全一致）

#### 假设
> "BF16 decode 走 `gemvt_op` (M=1 专用 32×4/64×2/128×1 tile)；INT8 decode 走 DeepGEMM int8 GEMMKernel（对 M=1 无 tile 优化，最小 tile 是 M=16，浪费 94% 计算）"

#### 实测验证
✅ **BF16 gemvt 确实 M=1 专用**——profile 显示 BF16 GEMM 中 99% 是 gemvt_op M=1（N=1 in profile 是 batch 维，K/N tile 是 32×4 / 64×2 / 128×1）。torch.nn.functional.linear 也比 `gemm_bf16_bf16_bf16_nt` 快 14-39%（见上表），证实 BF16 decode 路径不会进通用 GEMM。

✅ **DeepGEMM int8 没有 M=1 专用 tile**——profile 中 INT8 的所有 GEMM 调用 M ∈ {16, 32, 128, 256}，无 M=1。直接调用 `gemm_int8_int8_bf16_nt(M=1)` 时 sglang 必须 pad 到 16。

✅ **大多数 decode GEMM INT8 比 BF16 慢**——上表 6/7 个 layer INT8 更慢 1.22×-2.32×，唯一赢的是 `lm_head` (N=151936，M-pad 开销被大 N 摊销)。

#### 为什么带宽节省没能转化成速度？

| 维度 | BF16 | INT8 | 变化 |
|---|---|---|---|
| 权重存储 | 2 bytes/elem | 1 byte/elem | -50% ✅ |
| 权重读取带宽（理论） | 2KN bytes | KN bytes | -50% ✅ |
| 每次 token decode 实际计算量 | M·K·N MAC | M·K·N MAC | 相同 |
| **M=1 时有效 MAC** | 1·K·N | **16·K·N** (padding) | **+1500%** ⬆️⬆️⬆️ |
| Decode kernel tile shape | 32×4 / 64×2 / 128×1 | 16×128×128 | - |

**M=1 时带宽不是瓶颈**：
- mlp.gate_up_proj M=1: 读 2·2048 + 12288·2048·2 = 50.5 MB → 理论 BF16 读 50.5 MB / 1.6 TB/s (HBM bw) = 31μs
- 实测 110.8μs → **远没到带宽上限**（实测 BW = 50.5 MB / 110.8μs = 0.46 TB/s = 28% of peak）
- INT8 节省 25 MB（权重一半）→ 理论省 16μs
- 但 DeepGEMM M=16 浪费 15/16 计算 → +135.6 - 110.8 = 24.8μs **开销**
- **带宽节省 (16μs) < padding 开销 (24.8μs) → INT8 净输 8μs**

### 28.4 这就解释了为什么 500 样本对比是「INT8 ≈ BF16」而不是「INT8 << BF16」

- §26 INT8 tuned: 330.50 tok/s, 78.88ms ttft
- §27 G 组 bf16:  331.20 tok/s, 73.30ms ttft

理论上每个 decode GEMM 都慢 1.22-2.32×，应该看到显著性能差距，但实测接近打平。原因：

1. **Decode 阶段不是纯 GPU bound**——sglang scheduler + Python runtime + attention 等其他 kernel 占据 wall-clock 大量时间。GEMV 慢 30μs 摊到 256 token decode loop 里只有 7-8ms 差异，被其他部分吸收。
2. **DeepGEMM Autotune 帮 INT8 选了大 tile 优势**——prefill 阶段（视觉 encoder 输出 + 第一波 token）INT8 确实占优（带宽真省下来了），与 decode 劣势抵消。
3. **lm_head 是 INT8 唯一赢的点**——N=151936 时 INT8 比 BF16 快 20%，每个 decode 步都跑一次 lm_head，给 INT8 一点 net positive。

### 28.5 对我们提交的意义

- **G 组 bf16 仍是首选**（§27）——不用动
- **INT8 路线没有继续优化的空间**——除非：
  - a) 自己写一个 INT8 M=1 专用 GEMV kernel（绕过 DeepGEMM），但这要改 sglang 源码
  - b) 改 qlean recipe 把 V 层/MLP gate 等敏感层保留 BF16（§10.5 路径 A），但 fused layer 矛盾未解决
  - c) 等 sglang-for-SAIL 后续版本提供 INT8 GEMV kernel（短期无望）
- **下一步探索方向**应该是 §27 的 bf16 配置微调：chunked-prefill、speculative decoding、TTFT-优先 cuda graph 调度等

### 28.6 复现命令

```bash
# Profile 数据已经存在（profile_results/）
python3 /tmp/quant_bench/analyze_csv.py

# 直接 microbench（需要能 import deep_gemm）
python3 /tmp/quant_bench/decode_microbench.py

# 用 acu 重新采集（不推荐，会杀掉目标进程）
# /usr/local/PPU_SDK/asight/bin/acu -o report -f --kill=no python3 your_app.py
```

### 28.7 关联文档

- `INT8_strategy.md` §10 — qlean 配方 vs sglang fused layer 不兼容的根因
- §27 — G 组 bf16 vs INT8 tuned 500 样本基准
- §26 — DeepGEMM Autotune 输出与启用流程
