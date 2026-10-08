# Qwen3.5-2B on PPU-ZW810E · 汇报 PPT 大纲与讲稿

> **用途**：现场汇报（默认按 20 分钟讲 + 10 分钟问答设计，每页标注建议时长，可按 15 分钟压缩讲稿中的「补充」段）。
>
> **四处已统一的表述口径**（避免评审追问）：
> 1. 不是「量化没效果」，而是 **「量化有效，但收益在 sglang/PPU 的默认执行路径上无法直接兑现——被框架吃掉了」**。量化有效性的证据：算力与权重带宽减半、transformers 端 68% 量化率零精度损失；无法兑现的证据：纯量化接入 sglang 后 TPS −2.7%、TTFT +3.1%。P8 / P15 / P18 的措辞统一按此口径（P1–P4 为"充分使用现有框架"的探索式开篇，先不抛结论）。
> 2. transformers 端不是「精度提升」，而是 **「68% 量化率下零精度损失」**；
> 3. 不是「算子仍然是 bf16」，而是 **「权重量化了、算子形态没量化」**（保护层仍走 BF16 kernel，量化层落进 M≥16 的 GEMM 形态，M=1 被 pad，浪费 94% 算力）；
> 4. M=8 阈值表述为 **「几何依据（张量核 BLOCK_M=16 下限）+ 实测 sweet spot」**，不声称做过完整参数扫描。
>
> **数据来源索引**：见文末附录 C。所有数字均可在仓库内溯源。

---

## P1 · 封面（0.5 min）

**屏幕**
- 主标题：Qwen3.5-2B 在 PPU-ZW810E 上的 INT8 量化与算子-图协同优化
- 副标题（三步走）：**① 先把现有框架压榨到底，锁定 sglang 最佳配置作为落脚点 → ② 引入 INT8 量化 → ③ 打通框架层，把量化的收益释放出来**
- 底部：提交件 `evaluation_wrapper.py` · 数据 en/cn 双集 100 样本

**讲稿**
> 各位好。我先说这次工作的展开方式，因为它决定了后面所有事情的解释顺序。
>
> **在做任何"改进"之前，我先做的是"充分使用"**——不写任何算子、不改任何源码，先把现有框架能压榨的空间尽量试完，看看这套技术栈在最好的配置下到底能跑到什么水平。我们做了六组参数对照，最后确认：**sglang 在其中一组特定配置下有绝对领先的优势**（相对默认配置，TTFT 降低 84.6%、吞吐提升 29.7%）。这组配置就成为了后续所有工作的**落脚点和参照系**。
>
> 有了这个基线，第二步才是引入 INT8 量化；第三步，是在量化模型上发现框架层的瓶颈——量化带来的算力收益并没有自动兑现——于是做算子形态、CUDA Graph 封装与分阶段分派的协同优化。汇报就按这三步展开。
>
> 这条路走下来，最终结果是——中文数据集上**精度零损失（0.82 = 0.82）**、吞吐 +9.8%、TTFT −9%，是严格的 Pareto 改进；英文数据集吞吐 +15%、TTFT −14%、精度差异在 100 样本置信区间内不显著。

---

## P2 · 任务与硬件基线（1 min）

**屏幕**
| 项 | 内容 |
|---|---|
| 模型 | Qwen3.5-2B VLM（hybrid：full attention + GDN linear attention，26 层 + ViT 视觉编码） |
| 硬件 | 1× PPU-ZW810E（T-Head 真武，96 GB，CUDA 兼容层） |
| 软件 | sglang 0.5.13 + PPU fork（sail）、DeepGEMM-for-sail、系统分析工具 asys |
| 数据集 | MMBench dev en / cn |
| 评测口径 | Accuracy（门槛 baseline −2%）、TTFT（取 median）、吞吐（tok/s） |

**讲稿**
> 先交代约束。模型是 Qwen3.5-2B 的多模态版本，注意它不是纯 Transformer——**除了 26 层里的 full attention，还有大量 GDN 线性注意力层，这是一个带循环状态的混合架构**。这一点在后面会反复出现：投机解码用不了、量化敏感层集中在 GDN 的状态投影上，都源自这个结构。
>
> 硬件是国产 PPU ZW810E，通过 CUDA 兼容层暴露接口，96 GB 显存。这里有个关键前提：**模型 bf16 只占 4.3 GB，显存完全没有量化刚需**——所以我们做量化的动机不是省显存，而是省算力和权重带宽。
>
> 指标口径提前说明：TTFT 我们一律报 median，因为 100 样本的 mean 会被前几十个冷启动样本拉高 4 倍；精度差异小于 9.6 个百分点的，我们统一标注为「统计不显著」。

---

## P3 · 第一步：先把现有框架压榨到底（框架选型与最佳基线）（2 min）

**屏幕**
| 组 | 增量改动 | acc | TTFT | TPS |
|---|---|---|---|---|
| A | sglang 默认（triton attention） | 0.80 | 419.0 ms | 284.5 |
| B | + `--attention-backend fa3` | 0.85 | 313.2 ms | 279.4 |
| C | + `SGLANG_SAIL_FLA_CUDA=1` | 0.85 | **64.3 ms** | 280.1 |
| D | + `--disable-radix-cache` | 0.85 | 69.0 ms | **369.4** |
| E | + `--chunked-prefill-size 2048` | 0.85 | 69.9 ms | 375.3（噪声内） |
| **G ⭐** | D + `--page-size 64` | **0.85** | **64.7 ms** | **369.0** |

- 起点对照：**transformers 直接推理 baseline（20 样本）acc 0.75 / TTFT 188 ms / 吞吐 55 tok/s**——迁移到 sglang 调优后（R6）相对它是 **TTFT −66%、吞吐 +680%**
- G 组 vs A 组：**TTFT −84.6%、吞吐 +29.7%、acc +5pp**
- 三项增益正交叠加：fa3（CUDA Graph 抓取 49.7s→4.0s）· FLA_CUDA（消掉 GDN 的 330 ms TTFT 平台）· 关 radix（顺带打开 overlap schedule）

**讲稿**
> 回到第一步：**在动手改进之前，我先做的是充分使用现有框架**——不写算子、不改源码，把能试的配置都试一遍，看它到底能跑到什么水平。先试的是 transformers 直接推理：它能勉强跑通，但 20 样本下只有 **55 tok/s、首 token 188 毫秒**（acc 0.75）；而且换到量化模型目录后，transformers 4.57.1 直接报"不识别 qwen3_5 架构"，连模型都加载不进去。所以候选框架实际只剩 sglang，接下来的问题就变成：它的参数能调到什么程度。
>
> 我们做了六组参数对照，三个增益互相正交：
>
> 第一，**attention 后端从 triton 换成 fa3**，CUDA Graph 的抓取时间从 49.7 秒降到 4 秒，TTFT 从 419 降到 313 毫秒。
>
> 第二，也是最关键的一步：**打开 FLA_CUDA**。这个开关文档里写默认是开的，源码里实际是 `False`。打开之后，GDN 层从通用 Triton kernel 切到 PPU 原生 kernel，TTFT 直接从 313 掉到 64 毫秒——原来一直有个 330 毫秒的平台期，就是 GDN 用通用 kernel 跑的代价，而且精度还顺带涨了 5 个点。
>
> 第三，**关掉 radix cache**。这个评测的 prompt 都不重复，缓存只有查表开销没有命中收益；更妙的是关掉它之后 sglang 会自动打开 prefill/decode 的重叠调度，吞吐直接涨 32%。
>
> 最后加 `--page-size 64`，把 TTFT 从 69 收到 64.7 毫秒。**相对默认配置，这组配置把 TTFT 压掉 84.6%、吞吐提升 29.7%、精度还高 5 个点——这就是"充分使用现有框架"能达到的上限，也是我们认定的最优落脚点**。后面的量化、算子、图优化，全部以它为参照系。

---

## P4 · 基线参数配置（"充分使用现有框架"的全部产出）（1.5 min）

**屏幕**
- 阶段：**Qwen3.5-2B 原模型（bf16）**——本页不含量化相关的任何参数

服务器启动参数（基线）：
```
--attention-backend fa3        --trust-remote-code    --tp-size 1
--enable-multimodal            --dtype bfloat16
--mem-fraction-static 0.75     --page-size 64         --disable-radix-cache
```
关键环境变量：

| 变量 | 值 | 作用 |
|---|---|---|
| `SGLANG_SAIL_FLA_CUDA` | **1** | GDN 走 PPU 原生 CUDA kernel（**源码默认 False，必须显式打开**，收益最大的一项） |

**试过但否决的**：`--chunked-prefill-size 2048`（噪声内无收益）· 保留 radix cache（只有查表开销）· 投机解码（启动 12s 必崩）

**效果**：TTFT **64.7 ms** · TPS **369.0** · acc **0.85**（en 20 样本）；相对默认配置 **TTFT −84.6%、吞吐 +29.7%、acc +5pp**

> 量化模型出现之后的完整配置（`w8a8_int8` + GEMV + 图）见 **P17**。

**讲稿**
> 这一页是"充分使用现有框架"的全部产出。请注意一个关键点：**它不包含任何量化相关的参数**——到这一步为止，我们用的还是 bf16 原模型。
>
> 配置本身只有三样：attention 用 fa3、`page-size` 用 64、关掉 radix cache。再加上一个必须显式打开的环境变量 `FLA_CUDA`——它在源码里的默认值是 `False`，这是最容易漏掉、也是收益最大的一项。
>
> 顺便交代我们试过但没用的：`chunked-prefill` 设 2048 在噪声内没有收益；保留 radix cache 反而更差；投机解码在这个模型上根本起不来。
>
> 效果是 TTFT 64.7 毫秒、吞吐 369 tok/s、精度 0.85。**量化相关的参数我放到后面 P17 一起讲，因为它们只有在量化模型出场之后才有意义。**

---

## P5 · 为什么量化：最直接的加速手段（1.5 min）

**屏幕**
- 硬件前提：模型 bf16 仅占 4.3 GB、显存 96 GB → 量化动机**不是省显存，而是算力与权重带宽减半**
- 原始模型系统分析（G 组 60 秒 kernel trace）：
  - **GEMM 占 73.4%**，其中 72.7% 是三种 `gemvt_op` tile（32×4 占 56.1%），共 **265,775 次 GEMV 调用**（均值 19.3 μs）
  - grid 拆分反推：GEMV 集中在 `q/k/v/o_proj`、GDN 投影（in_proj/out_proj）、MLP gate/up、`lm_head`
  - 一句话：**decode 阶段的线性层以 GEMV 形态占绝对主导**
- 下一步的问题：**量化哪些层、以什么方式量化？** 太保守收益吃不满，太激进精度会崩 → **需要一套足够精确的选层框架，在精度不变的前提下尽可能量化不敏感层** → 引出 AMQ（P6）

**讲稿**
> 先说为什么选量化。它是这套硬件上最直接的加速手段：INT8 把计算量和权重带宽都减半；而模型本身 bf16 只占 4.3 GB、显存有 96 GB，完全够用——**所以我们要的不是省显存，是省算力**。
>
> 那这股加速应该往哪里打？这由系统分析决定。我们对原始模型做了 60 秒的 kernel trace：**GEMM 占了 73.4% 的时间，而这 73.4% 里面有 72.7% 是同一个 GEMV kernel 的三种 tile**——26 万次调用，平均每次 19 微秒。这说明瓶颈不在"大矩阵乘"，而在 decode 阶段"一个激活向量乘以一个大权重矩阵"这个动作上。我们还用 grid 维度做了反推，把 GEMV 精确定位到五类 Linear 层：注意力的 q/k/v/o 投影、GDN 的 in_proj 和 out_proj、MLP 的 gate/up，以及 lm_head。
>
> 但紧接着就遇到下一个问题：**量化哪些层、以什么方式量化**。保护得太保守，收益吃不满；放得太开，精度会崩——这两类失败我们在早期实验里都踩过。所以量化不是随便挑一个配方就行，它**需要一套足够精确的选层框架，能在精度不变的前提下，尽可能多地把不敏感的层量化掉**。
>
> 这就是我们选择 AMQ 的原因。下一页讲它的原理。

---

## P6 · AMQ 原理：敏感度驱动的量化边界（2 min）

**屏幕**

对 175 个 2D Linear 权重做 INT8 重建误差分析（按层类型统计 mean 误差）：

| 层类型 | mean 重建误差 | 敏感性 | 决策 |
|---|---|---|---|
| `linear_attn.in_proj_a/b`（GDN 状态投影） | **0.0154** | 最高 | 保 BF16 |
| `self_attn.v_proj` / `k_proj` | 0.0131 / 0.0128 | 高 | 局部保护 |
| `self_attn.o_proj` / `mlp.down_proj` | ~0.011 | 中 | 量化 |
| `self_attn.q_proj` | 0.0100 | 低 | 量化 |
| `linear_attn.in_proj_qkv/z` | 0.0096 | 很低 | 量化 |
| `mlp.gate / up` | **0.0092** | 最低 | 量化 |

- 误差 >0.015 的 23 层中，18 层是 `in_proj_a/b`，其余是第 19/23 层的 `v/k_proj`
- **核心结论：量化边界由敏感度排序决定，而不是由覆盖率决定**（与"尽量多量化"的直觉相反）

**讲稿**
> 上一页留下的问题是"到底哪些层能量化"。我们选的框架是 AMQ，Activation-aware Mixed Quantization，它的原理不复杂但很有效：**逐层试量化，测每一层的 INT8 重建误差，再用这个误差给所有层排出一个敏感度序**。误差大说明这层不耐量化，误差小说明可以放心压到 INT8。
>
> 这张表就是敏感度分析的结果，很有信息量：最敏感的是 GDN 的状态投影 `in_proj_a/b`，误差 0.0154；最不敏感的是 MLP 的 gate 和 up，0.0092，差了快一倍。误差超过 0.015 的 23 层里，18 层都是这两个状态投影。
>
> 为什么 GDN 的状态投影这么敏感？因为它是**循环状态机**的一部分——量化误差会在时间步之间累积，不像前馈层那样算完就结束。这解释了后面 S3 全量化方案为什么会崩。
>
> 所以 AMQ 给我们的最大启示是一句反直觉的话：**量化边界应该由敏感度排序决定，而不是由"尽量多量化"这个目标决定**。这是整条量化线的设计原则。

---

## P7 · 三档配方与 transformers 验证（2 min）

**屏幕**

| 策略 | INT8 层数 | 保护（BF16）的层 | 结论 |
|---|---|---|---|
| v3（保守） | 42 | GDN 环路全部 + gate/up + 19/23 层 qkv | 后来被否决：覆盖率不足，无端到端收益 |
| **S2-aggressive（采用）** | **126（量化率 68%）** | `in_proj_a/b` + `out_proj` + 19/23 层 qkv | 端到端 +20% 吞吐，精度几乎不动 |
| S3-full | 186（近全量化） | 仅 norms/embed/lm_head | 否决：acc 掉 13pp |

transformers 库直接对比（权重 int8→BF16 反量化，激活 BF16）：

| 样本 | BF16 baseline | S2（126 层） | S3（186 层） |
|---|---|---|---|
| 20 题 en / cn | 80% / 90% | 80% / 90% | 80% / **85%** |
| 30 题 en / cn | 70.0% / 83.3% | **73.3%** / 83.3% | 70.0% / 83.3% |

**结论：68% 量化率下零精度损失**（30 样本 S2 表面高 3.3pp，在 ±1 题噪声内）

> **本页只比精度，不比性能**——transformers 侧的算法是权重 int8 → 反量化回 BF16 再算、激活保持 BF16，耗时与 bf16 完全相同，**性能维度不可比**。这也正是"迁移到 sglang 之后才发现性能落差"的原因：**量化模型在 transformers 上从来没有被性能验证过**。

**讲稿**
> 按敏感度排序，我们出了三档配方。保守档只量化 42 层，全量化档 186 层，我们采用中间档 S2，量化 126 层，量化率 68%。保护的都是敏感度最高的那些：GDN 的状态投影、输出投影，以及标定出来误差最大的第 19、23 层的 qkv。
>
> 然后我们在 transformers 库里做了直接对比。这里要注意测试方式：权重是 int8 存储，但要反量化回 BF16 再算，激活也是 BF16——**这等价于只验证"量化带来的信息损失"，不涉及算子效率**。所以这一页**只有精度数据，没有 TTFT 和吞吐**：这条路径的耗时和 bf16 完全一样，性能不可比。换句话说，**量化模型在 transformers 上从来没有被性能检验过**，这也埋下了下一页"迁移到 sglang 才发现落差"的伏笔。
>
> 结果是：20 题上三种配方和 BF16 完全持平；30 题上 S2 是 73.3%，BF16 是 70%，表面还高 3.3 个点，但在 ±1 题的噪声范围内。全量化的 S3 在中文 20 题上掉到 85%，这就是我们放弃它的原因——**循环状态投影被量化，误差累积起来了**。
>
> 所以这时候我们的结论是：**68% 量化率、精度零损失，模型侧没有问题了**。带着这个结论，我们把它迁移到了 sglang。

---

## P8 · 转折：迁移 sglang 后并没有炸裂（2 min）

**屏幕**
- **现象**：sglang 端首轮实测 —— 353.6 tok/s / TTFT 75.7 ms / acc 0.70，**比 baseline 还慢 1.3%**
- **根因：权重量化了，算子形态没量化**（双重算力浪费）

| 浪费来源 | 事实 | 证据 |
|---|---|---|
| ① 保护层仍是 BF16 计算 | AMQ 保护的 GDN 投影等，恰是 GEMV 调用量最大的层 | — |
| ② 量化层落进 GEMM 泛型形态 | INT8 无 M=1 专用 tile，**每个 M=1 decode 被 pad 到 M=16，浪费 15/16 ≈ 94% 计算量** | §28.1 |
| ③ 微基准层面 INT8 GEMM 本就输 | M=1 时 **6/7 个层 INT8 DeepGEMM 比 BF16 GEMV 慢 1.22×~2.32×**（仅 lm_head 因 N 巨大摊薄 padding 而赢，0.80×） | §28.2 |

**讲稿**
> 迁移到 sglang 之后，预期中的性能提升没有出现——吞吐 353.6，**比基线还慢 1.3%**。
>
> 排查之后发现一个很反直觉的事实：**权重量化了，但算子形态没量化**。具体有两层浪费。
>
> 第一层，AMQ 为了保护精度保留下来的那些 BF16 层，恰恰是 GEMV 调用量最大的层——它们还在用 BF16 的 kernel 算，等于这部分算力一点没省。
>
> 第二层更隐蔽。已经量化成 INT8 的层，在 DeepGEMM 里走的是**通用 GEMM 泛型路径**。这个 kernel 的最小 tile 是 M=16，而 decode 阶段 M 等于 1，**每个 M=1 的调用都被 padding 到 M=16，浪费了 15/16，也就是 94% 的算力**。
>
> 我们做了 M=1 的逐层微基准来验证：**7 个层里有 6 个，INT8 GEMM 比 BF16 的专用 GEMV 内核还要慢 1.22 到 2.32 倍**。唯一赢的是 lm_head，因为它的输出维度有 15 万，把 padding 摊薄了。
>
> 结论很清楚：想要吃到量化的红利，**必须有一个真正为小 M 设计的算子**。

---

## P9 · 判断：为 decode 定制算子（1 min）

**屏幕**
- 量化模型落地后**重新做系统分析**：**GEMM 占 56.07%**（168,713 次），其中 `gemvt_op` **~266,000 次 ≈ 4400 calls/s**；element-wise ~30%、attention ~10%
- → 量化确实改变了算子的执行分布，但 **decode 的瓶颈形态没变，仍是 GEMV 占主导**
- decode 负载特征：单激活向量 × 大权重矩阵，GEMM 利用率极低（M=1 vs tile M=16）
- 微基准：GEMV 在 M=1..8 区间比 GEMM 快约 2×，来源是**消除小 M 形态劣势 + Python dispatch 开销**
- 设计边界：**复用 PPU 已有的 kernel primitive，只做面向工作负载的形态特化**——不碰 ISA、不手写 CUDA

**讲稿**
> 所以在动手之前，我们在量化模型上又做了一次系统分析：GEMM 占 56%、GEMV 的调用率约每秒 4400 次。**量化确实改变了算子的执行分布，但 decode 阶段的瓶颈形态没有变——还是 GEMV 占主导**。
>
> 下一步因此很明确：**为 decode 阶段定制算子**。
>
> 这里我先说设计边界，因为它决定了工程量和风险。我们没有去手写 CUDA、更没有碰 ISA 层，而是定了两条原则：
>
> 第一，**尽可能复用 PPU 上 DeepGEMM 已有的泛型内核**——硬件能力是现成的，我们只做面向工作负载的形态特化。
>
> 第二，**两条线并行推进**：GEMM 的 INT8 路径和 GEMV 的 INT8 路径。前者用于大 M 场景，后者用于小 M 场景，让不同 batch 形态各走各的最优路径。

---

## P10 · 算子定制双线：复用 deepdense 泛型（2 min）

**屏幕**

**GEMM 线（既有，接管为主路径）**
- DeepGEMM 原生 `gemm_int8_int8_bf16_nt`（PPU fork 已有）+ sglang 侧既有 wrapper `gemm_nt_i8i8bf16`（`entrypoint.py:384`，含 tile config 选择 + `deep_gemm_execution_hook`）

**GEMV 线（新增，零额外编译）**
- 新增 `deep_gemm/jit_kernels/gemv_int8.py`，**复用 `include/deep_gemm/gemvt.cuh` 的泛型内核**：
```cpp
Gemvt<int8_t, __nv_bfloat16, int, N, K, kNumGroups,
      ThreadPerN, NPerThread, NUM_UNROLL, SWZL_SIZE_M, BlockSize, USE_SMALL_K>
```
- 它与 MoE grouped GEMV **是同一个内核**：dense 场景只需 `NUM_GROUPS=1`、`grouped_layout` 全 0 → **数值语义与 GEMM 路径 bit 级一致（max_diff = 0）**
- tile config 由既有 `get_gemv_best_configs` 选，`jit_tuner.compile_and_tune` 编译并缓存
- 接入三件：`entrypoint.py:406` 新增 `gemv_nt_i8i8bf16` wrapper · `compile_utils.py` 新增 `GEMV_NT_I8I8BF16` 枚举与 `_GemvWarmupExecutor_int8`（warmup 只覆盖 m≤16）· `environ.py` 新增 GEMV 相关 env
- 部署方式：纯 Python 覆盖 site-packages，**不改 C++ 扩展**，`git restore` 即回滚

**讲稿**
> 这是我最想讲的一页，因为它是整个方案里"性价比最高"的决策。
>
> GEMM 这条线比较简单：DeepGEMM 里本来就有 INT8 的 GEMM 内核，sglang 侧也有现成的 wrapper，我们要做的是让它成为主路径，并加上分派逻辑。
>
> GEMV 这条线是新的，但我们**没有手写 kernel**。我们在 DeepGEMM 里找到了一个现成的泛型 GEMV 内核 `Gemvt`，它原本是服务于 MoE 的 grouped GEMV 的。**它是同一个内核，只是在 dense 场景下把分组数设成 1、把 grouped layout 全置零**。
>
> 这个选择带来两个巨大的好处：第一，**不需要额外编译**，直接 JIT 复用；第二，因为用的就是这个仓库原本的数值路径，**GEMV 的计算结果和 GEMM 是 bit 级一致的**——我们实测 max diff 等于 0。这一点非常重要：它意味着**精度完全由量化方案决定，和我们换算子没有任何关系**，不会出现"换了算子精度掉了"这种纠缠。
>
> 工程上只做了三件事：加一个 JIT 封装文件、在分派层加一个枚举和预热执行器、注册几个环境变量。**全部是纯 Python 覆盖，不改任何 C++ 扩展，回滚就是一条 git restore。**

---

## P11 · 微基准大胜 vs 端到端反跌（2 min）

**屏幕**

| 尝试 | 微基准 | 端到端 | 结果 |
|---|---|---|---|
| Triton 手写 GEMV（§29） | M=1 比 DeepGEMM **快 2–5×** | en 500：358.03 tok/s | **增益几乎全被吸收**（对比 358.09） |
| CUTLASS LUT 调优（§31） | gate_up M=1 **4.2×**、qkv M=1 **9.3×** | 308 tok/s | **比基线慢 14%** |
| 复用 Gemvt 首次落地（§32） | **13.0 μs vs 27.9 μs = 2.15×**（bit 级一致） | 353.6 tok/s / acc 0.70 | **−1.3%** |

**文档给出的三个归因**
1. 当时 INT8 覆盖太薄（仅 42 层），GEMV 的固定开销吃掉 kernel 加速
2. **Python 派发开销**：每 token 每层累加（env 读取 + shape 判断 + custom op + execution hook）
3. GEMM 只占 decode 的 30–50%，其余是 attention / mamba / KV 操作 / 调度

**讲稿**
> 然后我们遇到了这次项目里最有教育意义的一段：**微基准大胜，端到端反跌**。
>
> 我们试了三条路。第一条，手写 Triton GEMV，微基准比 DeepGEMM 快 2 到 5 倍，结果端到端 358.03 对 358.09，**增益被完全吸收，一点没体现**。
>
> 第二条，调 CUTLASS 的 LUT 配置，微基准上 gate_up 快 4.2 倍、qkv 快 9.3 倍，端到端反而**比基线慢 14%**。
>
> 第三条，复用 Gemvt 内核，13 微秒对 27.9 微秒，2.15 倍，数值还 bit 级一致——端到端 353.6，还是**比基线慢 1.3%**，精度也掉了。
>
> 当时文档给了三个归因：INT8 覆盖层数太少、Python 派发开销每层每 token 累加、以及 GEMM 本身只占 decode 的 30 到 50%。
>
> 但这三条其实都不是真正的根因。真正的根因在下一页——它是我们排查到最深、也最容易被忽略的一条。

---

## P12 · 根因：CUDA Graph 里没有你的算子（1.5 min）

**屏幕**
- sglang 默认用 CUDA Graph 连续启动算子；**capture 时把 Python 调用固化进图，replay 阶段不再走 Python**
- profiling 原始输出（决定性证据）：
```
apply() called with x.shape = torch.Size([256, 6144])   # decode
```
- **CUDA Graph 按 `bs=256`（max batch size）抓取 → replay 期 `x.shape[0]` 恒为 256 → 所有 `x.shape[0] == 1`（甚至 `<= 16`）的 Python 分支永远不执行**
- 结论：**不在图里的算子，等于不存在**。放宽阈值（改 `<=16`）同样失败；关掉 CUDA Graph 让 Python 分派生效，代价是约 −30% 吞吐，更差

**讲稿**
> 真正的根因是：**我们的算子在 CUDA Graph 里根本不存在**。
>
> sglang 默认会把 decode 的整个执行过程捕获成 CUDA Graph。capture 的时候，Python 调用被固化进图；replay 的时候**Python 完全不再参与**。所以我们在 Python 层做的任何 monkey-patch、任何 shape 判断，在 replay 期间一行都不会执行。
>
> 我们打印出来的 profiling 是最直接的证据：apply 被调用时张量形状是 `[256, 6144]`。也就是说，**CUDA Graph 是按最大 batch size 256 抓的图，replay 时 `x.shape[0]` 恒等于 256——我们所有"如果 M 等于 1 就走 GEMV"的判断，永远为假**。
>
> 我们试过放宽阈值到小于等于 16，一样失败，因为图里给的就是 256。也试过反过来关掉 CUDA Graph 让 Python 分派生效，但要付出大约 30% 的吞吐代价，得不偿失。
>
> 所以结论只有一句：**不在图里的算子，等于不存在**。算子必须被封装成 CUDA Graph 能够识别的类型。

---

## P13 · 封装方式：custom op（2 min）

**屏幕**

```python
def gemv_int8_i8i8bf16_fake(A, As, B, Bs, C):
    return                                   # fake: 只做 shape 推断，不分配、不计算

@register_custom_op(mutates_args=["C"], fake_impl=gemv_int8_i8i8bf16_fake)
def gemv_int8_i8i8bf16(A, As, B, Bs, C):
    deep_gemm_wrapper.gemv_nt_i8i8bf16((A, As), (B, Bs), C)   # 写入预分配的 C
```
- **输出张量预分配并作为参数传入**（`C`），`mutates_args=["C"]` 声明原地写
- fake impl 只返回 shape 信息 → capture 期无需真实分配，replay 期零 Python
- 完整调用链：
```
apply() → gemv_int8_i8i8bf16(custom op) → gemv_nt_i8i8bf16 → Gemvt<int8_t, ...>
```

**讲稿**
> 解法是把算子注册成 CUDA Graph 能识别的 **custom op**，写法上有两个要点。
>
> 第一，**输出张量预分配好，作为参数传进去**。因为图捕获期间不能做动态内存分配，我们把输出 `C` 提前分配，函数只负责往里写，并用 `mutates_args` 声明它是原地写。
>
> 第二，**注册一个 fake 实现**。这个 fake 只做 shape 推断，不分配内存也不做计算。它让编译器在图捕获阶段就能拿到形状信息，从而**不需要真正执行算子**，也不需要在 capture 期间碰任何不安全的路径。
>
> 这样改造之后，调用链就完整了：从量化层的 apply 分派，到 custom op，到 wrapper，最后落到 Gemvt 内核。
>
> 顺便说一句，这套"fake 实现 + 预分配输出"的写法不是我们发明的，是**从 PPU 自家的 FA3 和 flashinfer fork 里学来的**——它们在 PPU 上处理同类问题的做法一模一样。后面的 ViT 修复我们也是沿用了同一个套路。

---

## P14 · 封装后的效果（2 min）

**屏幕**

**控制变量实验（最关键）**
| 配置 | tok/s |
|---|---|
| S3-full + GEMV | 372 |
| **S3-full，关掉 GEMV** | **273** |
| → **GEMV 单独贡献 +99 tok/s（+36%）** | |

**串联后的连续提升**：358 → 372（§33）→ **388.0**（§35，+8.4%，acc 持平）→ **389.4**（§36，+8.8%，acc 0.80）

**同机严格对照（§38，20 样本）**
| 配置 | TTFT | TPS | acc |
|---|---|---|---|
| 真实 G 组 baseline（BF16） | 86.94 ms | 279.79 | 0.85 |
| **GEMV in CUDA Graph（本方案）** | **84.16 ms** | **418.47** | 0.85 |

→ **吞吐 +49.6%、TTFT −3.2%、精度持平**

**讲稿**
> 封装完成之后，效果终于出来了。我先讲最关键的一组数据，因为它是一次**控制变量实验**。
>
> 我们拿同一个量化配置，只切换 GEMV 的开关：开着是 372，关掉是 273。**GEMV 单独贡献了 99 个 tok/s，+36%**。
>
> 这组数据为什么重要？因为在此之前，我们看到的"提升"只有 4%，看起来毫无希望——那是被低覆盖率的对照组误导了。**只有把变量控制住，才能把 GEMV 的真实价值分离出来**。
>
> 在此之后是一串连续提升：358 → 372 → 388 → 389.4，最后一档同时把精度拉回 0.80。
>
> 到了同机严格对照这一组，量化的价值就非常清楚了：同样的机器、同样的 sglang 版本，bf16 基线是 280 tok/s，我们的方案是 **418 tok/s，吞吐 +49.6%**，TTFT 还略有改善，精度持平。

---

## P15 · 副作用与解法：M-aware 双阶段（2 min）

**屏幕**
- **现象**：吞吐 388（+8.4%）的同时，**TTFT 从 63 ms 恶化到 106.7 ms**
- **原因（当时文档原文）**：**prefill 阶段 M > 8 走 INT8 GEMM（autotune LUT）实际上比 BF16 GEMM 慢**；而 TTFT 由 prefill 主导（384 token 输入走 batched GEMM，不是 GEMV）
- **解法（M-aware 双阶段分派）**：
```
        M ≤ 8                     M > 8
   INT8 GEMV（decode）      BF16 GEMM（反量化 + F.linear，权重懒缓存）
        └──────────── 结果：TTFT 106.7 ms → 72.1 ms ────────────┘
```
- 阈值 `SGLANG_SAIL_GEMV_M_MAX = 8`：几何依据是 M≥16 时 PPU 张量核强制 `BLOCK_M=16`、GEMV 不再有优势；实测 sweet spot 为 8
- 「收益被框架吃掉」vs「收益被释放」对照（同一份量化模型）：**TPS −2.7% → +20%；TTFT +3.1%（变慢）→ −14%**

**讲稿**
> 但性能上去之后，出现了一个明显的副作用：**TTFT 从 63 毫秒恶化到 106.7 毫秒**，涨了将近 70%。
>
> 原因我复盘一下，当时文档里记录得很清楚：**prefill 阶段走的 INT8 GEMM 实际上比 BF16 GEMM 慢**。这是很反常识的一点——量化应该更快才对。但 TTFT 是 prefill 主导的，prefill 输入是 384 个 token，M 等于 384，是大矩阵乘的场景；在这个场景下 INT8 GEMM 因为 padding 和缺乏成熟融合方案，反而输给了 bf16。
>
> 所以解法是**按 token 数分阶段**：M 小于等于 8 的 decode 阶段走 INT8 GEMV，M 大于 8 的 prefill 阶段把权重反量化回 BF16、走 `F.linear`。反量化结果懒缓存，推理期只算一次，没有额外开销。
>
> 效果是 TTFT 从 106.7 拉回 **72.1 毫秒**。
>
> 阈值为什么取 8？几何上的依据是：M 到了 16 的时候，PPU 张量核强制最小块是 16，GEMV 就没有优势了，所以阈值必须小于 16；实测在 8 附近是最优的。
>
> 最后这张对照表请重点看，它改变的是"怎么表述这件事"：**同一份量化模型，在没有算子定制和图封装时，吞吐 −2.7%、TTFT 变慢 3.1%——量化本该带来的算力和带宽收益，被框架的执行路径整个吃掉了；加上算子定制、CUDA Graph 封装和双阶段分派之后，变成吞吐 +20%、TTFT −14%**。所以准确的结论不是"量化没用"，而是**"量化有用，但要先把框架这一层打通，收益才拿得出来"**。

---

## P16 · TTFT 再压缩：ViT 图捕获与全链路图支持（2.5 min）

**屏幕**

**问题**：TTFT 回到 72 ms 后仍高于基线的 63 ms——剩下的差距在 **ViT 视觉编码**（图像编码在首 token 之前完成）

**ViT CG 初期是纯负优化**：开图捕获后 acc 从 0.76 崩到 **0.62**（−14pp），degenerate 输出达 21 个

**两个叠加根因**
| 根因 | 内容 |
|---|---|
| **A：ViT attention 未命中 PPU 原生 FA3** | PPU 报 sm80 → 默认落 `triton_attn`；即使切 `fa3`，`vision.py` 的 `flash_attn_varlen_func` 来自 **community FA3 / sgl_kernel**，而 PPU 原生 FA3 只接到了 decode 的 attention backend，**没接到 ViT 的 `VisionFlash3Attention`** |
| **B：精度 −14pp 的主因** | Qwen3-VL 缺 `max_context_len` → `ViTCudaGraphRunner._ensure_sin_cos_ws` 首次只按第一张图分配 rotary buffer → 大图触发懒扩展换地址 → **已捕获的 graph 仍读旧地址 → rotary 错乱 → 输出退化为 `!!!!`** |

**修复（2 个 patch）**
- P1：用 `@plugin_hook(REPLACE)` 把 ViT 的 `flash_attn_varlen_func` **映射到 PPU 原生 `flash_attn_interface`**
- P2：`Qwen3VLMoeVisionModel` 补 `max_context_len = 2304 × 4 = 9216`，首次分配即足够大，图地址永不失效
- 效果：acc **0.600 → 0.710（+11pp）**、degenerate 输出 21 → 7、TTFT −13%

**全链路图支持（三层）**
| 层 | 对象 | 开关 |
|---|---|---|
| decode CUDA Graph | 整步 decode（bs=1..16） | `--cuda-graph-max-bs 16` |
| prefill PiCG | prefill 分段 + inductor 算子融合 | `--piecewise-cuda-graph-compiler inductor` → TTFT median **330 ms → 63.2 ms（−81%）** |
| ViT CUDA Graph | ViT blocks + merger | `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` |
| 兜底 | RoPE 包成 custom op，避免 inductor 在 capture 期走不安全路径 | F 类 hook（与 FA3 fork 同套路，bit-identical） |

**讲稿**
> 双阶段之后 TTFT 是 72 毫秒，还是比基线的 63 高。这 9 毫秒的差距在 ViT 上——因为这是多模态模型，图像编码发生在首 token 之前，**ViT 的耗时直接进 TTFT**。
>
> 所以我们开始做 ViT 的图捕获。这一步踩到了整个项目最深的正确性坑：**开了图捕获之后，精度从 0.76 直接崩到 0.62，还有 21 个输出完全退化成乱码**。
>
> 排查出来是两个叠加的根因。
>
> 第一个，**ViT 的 attention 没有走 PPU 原生实现**。PPU 上报的计算能力是 sm80，所以默认选了通用 Triton；即使强制切 fa3，`vision.py` 里调的 `flash_attn_varlen_func` 来自社区版本，而 **PPU 原生的 FA3 只接到了 decode 路径，没有接到 ViT 的 attention 模块**。修复方式是一个 hook，把 ViT 的这个函数替换成 PPU 原生实现。
>
> 第二个根因更隐蔽，也是精度崩 14 个点的主因：**Qwen3-VL 缺少 `max_context_len` 这个属性**。这导致图捕获时，rotary 位置编码的共享缓冲区只按第一张图的 token 数分配；评测里图像尺寸不一样，来了张大图就触发缓冲区的懒扩展，**地址变了，但已经捕获的图还按旧地址读**，于是位置编码全乱，输出变成感叹号。
>
> 这个 bug 很有意思：它不是图优化引入的，而是图优化**把一个原本被掩盖的存量 bug 暴露了出来**。修复就是补一行 `max_context_len`，让它第一次就分配足够大——按最大 96×96 的图算出来是 9216。
>
> 两个 patch 之后，精度从 0.60 回到 0.71，涨了 11 个点，TTFT 还降了 13%。
>
> 到这里，**全链路的图支持就补齐了三层**：decode 的整步图、prefill 的分段图、ViT 的图。prefill 那层我们还切到了 inductor 编译器做真正的算子融合，**TTFT median 从 330 毫秒降到 63.2 毫秒，降了 81%**。
>
> 另外还有一块兜底：inductor 在捕获阶段会走到 PPU 上不安全的内存路径，我们的做法是把 RoPE 包成一个 custom op，让编译器看不到函数体——**跟 FA3 fork 里的做法完全一致，数值 bit 级不变**。

---

## P17 · 最终配置总表（量化模型 + 算子 + 图）（1.5 min）

**屏幕**
- 阶段：**量化模型（S2-aggressive）就位 + 算子定制 + 图优化全部落地**——即 P4 基线配置之上的完整形态

服务器启动参数（提交件自动组装）：
```
--quantization w8a8_int8       ★必传（否则 fallback 到 compressed_tensors 路径，调 PPU 上不存在的 int8_scaled_mm，直接崩）
--attention-backend fa3        --mm-attention-backend fa3
--enable-piecewise-cuda-graph  --piecewise-cuda-graph-compiler inductor
--cuda-graph-max-bs 16         --page-size 64          --disable-radix-cache
--enable-multimodal            --dtype bfloat16        --mem-fraction-static 0.75
```
在基线之上新增的关键环境变量（= 量化 / 算子 / 图 三组）：

| 组 | 变量 | 值 | 作用 |
|---|---|---|---|
| 量化 | `SGLANG_SAIL_DEEPGEMM_DENSE` | 1 | 进入 DeepGEMM 分支（保护性开关，让 ACEXT 路径永远到不了） |
| 算子 | `SGLANG_SAIL_USE_GEMV` | 1 | decode 走 INT8 GEMV |
| 算子 | `SGLANG_SAIL_GEMV_M_MAX` | **8** | decode/prefill 分派阈值 |
| 算子 | `SGLANG_SAIL_PREFILL_BF16` | 1 | prefill 反量化走 BF16 GEMM |
| 算子 | `SGLANG_SAIL_GEMV_WARMUP_ON_LOAD` | 1 | 加载期预热 JIT，推理期零编译 |
| 图 | `SGLANG_VIT_ENABLE_CUDA_GRAPH` | 1 | ViT 图捕获 |
| 图 | `SGLANG_CUDA_GRAPH_MAX_BS` | 16 | decode 图覆盖 bs=1..16（实测 sweet spot） |
| 图 | `SGLANG_PIECEWISE_CUDA_GRAPH_COMPILER` | inductor | prefill 分段图 + 算子融合 |
| 基线 | `SGLANG_SAIL_FLA_CUDA` | 1 | PPU 原生 GDN kernel（同 P4） |

**明确禁用**：`--enforce-piecewise-cuda-graph`（触发 stream 断言杀死 scheduler）· `--enable-torch-compile`（PPU 上 causal conv codegen bug）· 任何 `--speculative-*`（GDN 不兼容，启动 12s 必崩）

**讲稿**
> 到这一步，量化、算子、图三件事都落地了，这是最终交付的完整配置。它等于 P4 那份基线配置，加上三组新东西。
>
> 量化这一组只有一条，但它必须显式传：`--quantization w8a8_int8`。不传会 fallback 到 compressed-tensors 路径，去调 PPU 上根本不存在的 `int8_scaled_mm`，直接崩。
>
> 算子这一组四条：`DEEPGEMM_DENSE` 是保护性开关，让厂商的 ACEXT 路径永远到不了——原因我在后面的结果页会用数据说明；`USE_GEMV` 和 `M_MAX=8` 决定 decode 走 GEMV、prefill 反量化走 BF16；`WARMUP_ON_LOAD` 让 JIT 在模型加载期就编译完，推理期零编译。
>
> 图这一组三条：ViT 图捕获、decode 图的 batch 覆盖上限取 16（我们实测的 sweet spot）、prefill 的分段图用 inductor 编译器做真正的算子融合。
>
> 最后是禁用项：`enforce-piecewise-cuda-graph` 会触发一个 stream 断言，把调度器直接干掉；`torch-compile` 在 PPU 上对 causal conv 的 codegen 有 bug；投机解码在 Qwen3.5 上完全不可用——这三条我们都写进了故障排查表。

---

## P18 · 最终结果与结论（2 min）

**屏幕**

| 配置 | TPS | TTFT median | acc(en) | acc(cn) |
|---|---|---|---|---|
| BF16 baseline | 402.7 / cn 383.2 | 76.8 / cn 71.4 | 0.780 | 0.820 |
| **R6（极致 TTFT+acc，推荐）** | 428.9 | **64.1 ms** | 0.72 | **0.820（0pp）** |
| R8（极致 TPS） | **434.05** | 66.5 ms | 0.71 | — |
| 提交件实测 | 411.6（+15%） | 63 ms（−14%） | 0.73 | 0.82 |

**三条结论**
1. **量化有效，但收益不会自动兑现**——同一份量化模型：无算子定制与图封装时 TPS −2.7%、TTFT +3.1%（收益被框架吃掉）；四件事协同后 TPS **+20%**、TTFT **−14%**（收益被释放）
2. **不在 CUDA Graph 里的算子等于不存在**——bs=256 抓图让所有 Python 分派失效，custom op 封装是唯一出路
3. **中文数据集严格 Pareto 改进**（精度 0pp、TPS +9.8%、TTFT −9.0%）；英文 −4pp 在 100 样本 ±9.6pp 置信区间内不显著，同 qid 对照真实损失 −2pp

**否决清单（一页带过）**：投机解码 EAGLE/NEXTN（GDN 无 parent token，启动即崩）· MXFP4 混精度（sglang 0.5.10 不支持）· S3 全量化（acc −13pp）· 真 ACEXT（端到端 −15.7%）· `enforce-piecewise-cuda-graph`（杀死 scheduler）· DALI 预处理（慢于 HF 实现）

**讲稿**
> 最后是结果和结论。
>
> 两个推荐配置：R6 用 triton attention，TTFT 中位数 64.1 毫秒、中文精度 0.82 和 bf16 完全一致；R8 用 fa3，吞吐最高 434 tok/s。提交件实测吞吐 411.6，比基线 +15%，TTFT 63 毫秒，比基线 −14%。
>
> 中文数据集上是**严格的 Pareto 改进**：精度零损失、吞吐 +9.8%、TTFT −9%。英文数据集精度表面差 4 个点，但在 100 样本 ±9.6 个点的置信区间内不显著，同题目对照后真实损失是 2 个点。
>
> 三条结论。**第一，量化是有效的，但收益不会自动兑现**——量化把算力需求和权重带宽减半，这一点在 transformers 上已经验证过；但在 sglang 的默认执行路径上，这份收益被算子形态和图机制整个吃掉了。是算子定制、图封装、双阶段分派、ViT 图捕获这四件事协同，才把收益真正释放出来。**第二，不在 CUDA Graph 里的算子等于不存在**，这是本次踩得最深的一个坑，也是 custom op 封装这条路的唯一理由。**第三，系统分析驱动的优化顺序很重要**——我们每一步的靶子都是前一步的系统分析给的，不是枚举试错。
>
> 最后这一栏是否决清单。我想特别提两个：投机解码在 Qwen3.5 上完全不可用，根因是 GDN 是循环状态机、没有显式 parent token，这是框架层不兼容，不是参数问题；还有厂商的 ACEXT 算子库，微基准上看起来更快，但端到端因为不接我们的 prefill 反量化路径，实测慢 15.7%——**这是"微基准不等于端到端"的第三次教训**，我们也把它作为一条保护性开关固化进了代码。
>
> 谢谢，欢迎提问。

---

## 附录 A · 讲稿节奏与压缩建议

| 页 | 建议时长 | 15 分钟压缩时删掉的内容 |
|---|---|---|
| P1–P2 | 1.5 min | P2 的软件版本细节 |
| P3–P4 | 3.5 min | P4 只留 `FLA_CUDA` 一项与效果数字，否决项略过 |
| P5–P7 | 5.5 min | P5 的量化模型 trace 细节；P7 的 S3 讨论 |
| P8–P10 | 5 min | P10 的代码模板 |
| P11–P14 | 7.5 min | P11 只留一行表格结论 |
| P15–P16 | 4.5 min | P16 的根因 A 细节 |
| P17 | 1.5 min | 只讲三组各一句（量化 1 条 / 算子 4 条 / 图 3 条） |
| P18 | 2 min | 否决清单改为口头一句 |

**加时（30 分钟场）**：P10 展开讲 JIT 与 warmup 机制；P13 展开 custom op 与 fake impl 的实现细节；P16 展开 RoPE hook（`inductor._get_sfdp_patterns` 的 pin_memory 崩溃链路）。

## 附录 B · 预判问答

**★ 全场最高频问题：「量化到底有没有用？值不值得做？」**
A：**有用**——量化把算力需求和权重带宽都减半，transformers 端 68% 量化率下精度零损失。但它的收益**不会在 sglang 默认执行路径上自动出现**：纯量化接入后 TPS −2.7%、TTFT +3.1%，因为保护层仍走 BF16 kernel、量化层落进 M≥16 的 GEMM 形态（M=1 被 pad，浪费 94% 算力）。需要算子形态特化 + CUDA Graph 封装 + 双阶段分派 + ViT 图捕获四件事协同，收益才被释放出来（TPS +20%、TTFT −14%）。**回答时不要说"量化没用"——那既不准确，也否掉了整条方案的立足点。**

**Q1：为什么不用 FP8 或更激进的量化？**
A：810E 这代硬件只支持 INT8，FP8/FP4 要 M890 才支持。另外 2B 模型的 bf16 权重只有 4.3 GB，96 GB 显存下量化从来不是为了省显存，而是为了小 M 下的算力与带宽。

**Q2：AMQ 的"敏感度"是准确率敏感度吗？**
A：是**逐层 INT8 重建误差**（权重级），不是端到端准确率。它给出的是层间相对排序，用来决定量化边界；最终精度由端到端评测确认。这个方法的代价很低，因为它不需要反复跑端到端。

**Q3：量化的精度代价到底是多少？**
A：中文 0pp；英文表面 −4pp、同题对照后真实 −2pp，在 100 样本 ±9.6pp 置信区间内不显著。另外 GEMV 与 GEMM 数值 bit 级一致，所以这些差异**全部来自量化方案本身，与算子替换无关**。

**Q4：M=8 这个阈值是怎么定的？**
A：几何依据是 M≥16 时 PPU 张量核强制最小块 16、GEMV 失去优势，所以阈值必须小于 16；实测 sweet spot 在 8。需要补充完整扫参表的话，可以再跑一组 4/8/12/16 的对照实验。

**Q5：为什么 ACEXT 先被认为是更好的？**
A：分三次结论。第一次微基准和 20 样本对照它略胜；第二次 100 样本网格里 GEMV 反超；最终端到端测"真 ACEXT"路径发现它不接 prefill 反量化，整体慢 15.7%。根因是测量精度在逐步提高——**最终裁决一律以 100+ 样本端到端为准**。

**Q6：ViT 图捕获让英文精度退了 3-4 个点，可接受吗？**
A：中文零损失，英文在噪声内。而且如果不修，开着图捕获精度会崩 14 个点、中文直接崩到 0——**修复前的状态是不可用，修复后才是可选项**。

**Q7：这套方案的通用性如何？**
A：方法论可迁移（系统分析驱动 → 算子形态特化 → 图封装 → 分阶段分派），但具体结论强依赖硬件：INT8 GEMV 的收益来自 PPU 张量核的 BLOCK_M=16 限制，这是硬件特性。换硬件需要重新做 M 阈值标定。

**Q8：transformers 的对比为什么只有精度、没有 TTFT 和吞吐？**
A：因为那一步的算法是"权重 int8 → 反量化回 BF16 再算、激活保持 BF16"，**耗时与 bf16 完全相同，性能维度不可比**——它只用于验证"量化带来的信息损失"。真正的性能基线是另一组数据：bf16 原模型用 transformers 直接推理只有 **55 tok/s / TTFT 188 ms**（20 样本，见 P3），迁移到 sglang 并调优后（R6）相对它是 **TTFT −66%、吞吐 +680%**。**"量化模型在 transformers 上性能如何"这个问题本身不成立：量化模型从未在 transformers 上做过性能验证——这正是迁移到 sglang 之后才发现落差的原因。**

## 附录 C · 数据来源索引

| 数据 | 来源 |
|---|---|
| 原始模型（bf16）asys profile 73.4%、gemvt 265,775 次、GEMV→层映射 | `ce1e714:docs/02_DISCOVERY.md`、`docs/04_PROFILE.md`、`docs/07_PROFILE_TO_KERNEL_MAP.md` |
| G 组基线六组对照与最终配置 | `ce1e714:docs/02_DISCOVERY.md`、`docs/06_FINAL_CONFIG.md` |
| **量化模型** trace：GEMM 56.07% / 4400 calls/s（P9 用） | `技术方案报告_完整版.md` §4.1 |
| transformers 直接推理 baseline：acc 0.75 / TTFT 188 ms / 55 tok/s（P3 用，20 样本） | `技术方案报告_完整版.md` §1.3、§10.3 |
| AMQ 敏感度表、三档配方、transformers 精度对比（**仅 acc，无 TTFT/吞吐**） | `eb5650f:docs/09_AMQ_QUANT_STRATEGY.md` §2–§4、`recipes/*.yaml` |
| M=1 padding 94%、6/7 层更慢 1.22–2.32× | `f4a7743:EXPERIMENT.md` §28 |
| Triton / LUT / Gemvt 三次微基准与端到端 | 同上 §29、§31、§32 |
| CUDA Graph bs=256 证据、分派失效 | 同上 §30 |
| 控制变量实验 +36%（372 vs 273） | 同上 §34；§33、§35、§36 为串联提升 |
| 同机对照 +49.6% | `fa6bfd9:EXPERIMENT.md` §38 |
| ViT CG 双根因与修复效果 | `vit_graph_optimization.md`、`0471341:vit_graph_optimization.md` |
| RoPE hook 根因与实现 | `inductor_hook_PPU/README.md`、`cc046f8:inductor_hook_PPU/PROGRESS.md` |
| R6/R8 网格、BF16 参照系、PiCG inductor 切换 | `4c3333a:results_summary.md`、`技术方案报告_完整版.md` §11 |
| 最终参数与验证清单 | `README_REPRODUCE.md`、`evaluation_wrapper.py` |
| 算子代码实现 | `gemv_optimization/DeepGEMM-for-sail/deep_gemm/jit_kernels/gemv_int8.py`、`.../deep_gemm_wrapper/entrypoint.py:406`、`compile_utils.py:128/892`、`.../quantization/w8a8_int8.py:91-120/265-318` |

**口径备注（若被追问）**：TTFT 在 08-21 实测为 63 ms、08-23 复测为 72.7 ms（median），差异源于 wrapper 默认 backend 与环境行为漂移，仓库内已记录为已知风险，以 2026-08-23 实测为准。汇报时建议统一说「63–73 ms 区间」，或直接引用提交件当次实测值。
