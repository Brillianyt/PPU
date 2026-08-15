# GEMV 调研笔记(flytiger-eco / sail 栈)

> 调研日期:2026-08-15。入口文档:https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=79(= FlashMLA-for-sail Release 用户指南)
> 原始材料见同目录 `raw/`。

---

## 1. 结论概要

- flytiger 文档中**没有以 "GEMV" 命名的独立文档**;GEMV 相关内容散落在四类材料里:**FlashMLA(文档 79,入口页)、DeepGEMM-for-sail README、vLLM-sail 的 `ppu_deep_gemm.py` 包装层、社区 DeepGEMM README**。
- GEMV 在 sail 推理栈里对应三类场景(见 §2),其中文档/源码里**明确命名**的是 DeepGEMM 的 **MQA logits kernel 族**(`*_mqa_logits` / `*_paged_mqa_logits`),即注意力"查询-键"打分(闪电索引器/MLA decode 用),本质就是 GEMV 型内核。
- FlashMLA 的解码路径(QK/PV)也是 GEMV 主导,但文档从"优化手段"角度描述,未用 GEMV 一词。
- ACTLIZE(CUTLASS 体系)与 FlashInfer-for-sail 指南中**无 GEMV 字样**。

## 2. GEMV 在 sail 推理栈中的三类场景

| 场景 | 形态 | sail 对应实现 |
| --- | --- | --- |
| ① 线性层 decode | batch≈1 的逐 token 投影(QKV/o/gate/up/down) | 走 GEMM 内核的 M=1 路径(如 vLLM `PPUInt8ScaledMMLinearKernel` 处理小 M);无专门 GEMV 文档 |
| ② Attention QK 打分 | 单查询 × KV → logits(索引器 / MQA / GQA decode) | **DeepGEMM `*_mqa_logits`(非 paged)+ `*_paged_mqa_logits`(paged)** |
| ③ MLA decode | MLA 的 QK/PV(解码逐 token) | **FlashMLA-for-sail**(dense decoding、DSA sparse prefill/decode) |

## 3. 各来源的 GEMV 相关内容

### 3.1 文档 79 = FlashMLA-for-sail(入口页,GEMV 型解码内核)

面向 MLA 架构大模型推理解码:提供 dense decoding、DeepSeek Sparse Attention prefill/decoding 的 PPU 内核。PPU 专项优化(原文要点):
- **AIU + TSM Swizzle 数据通路**:Q/K/V 与 KV cache 分块经 AIU 异步搬运,写入 TSM/shared memory 时配合 swizzle 布局,贴合后续 MMA 操作数读取;
- **PPU Tensor Cell 映射**:QK 与 PV 分块 MMA 映射到 Tensor Cell,含 FP16/BF16 与 FP8 KV cache→BF16 计算路径;
- **Warp interleave / CrossCut 调度**:dense MLA 解码按序列长度/batch/设备能力选 CrossCut 与 warp interleave,QK、softmax、PV、写回在不同 warp group 交错推进;
- **Tile 与寄存器驻留调优**:按 `seqlen_q`、head 数、page block、top-k、KV cache 形态选 block shape/warp 数/stage 数/MMA atom/Q 驻留策略;
- **Metadata + Split-KV 负载均衡**、**Sparse / FP8 KV cache 路径**(token-level sparse、extra KV、attn sink、top-k length;FP8 KV 读入反量化→BF16 计算)。

(MLA 解码的单查询 QK/PV 即 GEMV 主导;文档从内核优化角度描述)

### 3.2 DeepGEMM-for-sail README(MQA logits = 明确的 GEMV 族内核)

> "We provide Multi-Query Attention (MQA) logits kernels that compute the query-key logits used by the LLM attention indexer. Both a non-paged variant (for contiguous KV) and a paged variant (for a paged KV cache with a block table) are supported, across BF16, INT8, FP8, and FP4 precisions."

- 非 paged:`deep_gemm.bf16_mqa_logits` / `int8_mqa_logits` / `fp8_mqa_logits` / 统一 `fp8_fp4_mqa_logits`。输入:query、key-value(量化变体带 scales)、per-token weights、`cu_seq_len_k_start` / `cu_seq_len_k_end` 范围张量 → 返回 masked logits。
- paged:`deep_gemm.bf16_paged_mqa_logits` / `int8_paged_mqa_logits` / `fp8_paged_mqa_logits` / `fp8_fp4_paged_mqa_logits`。操作融合 KV cache + `block_table` + 调度 metadata(变长序列);先调 `deep_gemm.get_paged_mqa_logits_metadata` 计算调度元数据。
- 精度覆盖:BF16 / INT8 / FP8 / FP8+FP4——即 **GEMV 也支持量化**(INT8/FP8 GEMV)。

另有 **HyperConnection (HC) prenorm GEMM**(`deep_gemm.tf32_hc_prenorm_gemm`,融合 prenorm 平方和归约到 TF32 GEMM)与工具函数(`set_num_sms`、`get_m_alignment_for_contiguous_layout`、`get_tma_aligned_size`、`get_col_major_tma_aligned_tensor` 等,AIU 对齐相关)。

### 3.3 社区 DeepGEMM README(V3.2 MQA kernels 示例)

- "2025.09.28: DeepGEMM now supports scoring kernels (weighted ReLU MQA logits) for the lightning indexer for DeepSeek v3.2."
- 内核族两个版本:non-paged(prefill)/paged(decode);以 `fp8_mqa_logits` 为例给出 6 个输入(`q` E4M3 `[seq_len, num_heads, head_dim]`、`kv` E4M3 等);paged 版本见 `tests/test_attention.py`。
- 即:GEMV 族 MQA scoring 内核服务于 **DeepSeek-V3.2/V4 的闪电索引器(Lightning Indexer)**。

### 3.4 vLLM-sail 包装层(`vllm/utils/ppu_deep_gemm.py`,已存 raw)

对外暴露的 GEMV 族函数(包装 DeepGEMM):
- `bf16/fp8/int8/fp8_fp4_mqa_logits`(非 paged)+ `fp8/int8/fp8_fp4_paged_mqa_logits`(paged)
- `get_paged_mqa_logits_metadata`、`get_mk_alignment_for_contiguous_layout`
- 另有 `fp8_mqa_logits_torch` / `fp8_paged_mqa_logits_torch`(torch 回退实现)
- 使用方:`vllm/models/deepseek_v4/`(ppu/ 子目录、sparse_mla.py、attention.py、compressor.py)→ DeepSeek-V4 系模型的 sparse MLA + 索引器路径在 PPU 上通过该包装层调用。

### 3.5 FlashInfer / ACTLIZE / FlashAttention

- **FlashInfer-for-sail** v0.6.12 指南:安装类文档,提及 PPU 适配的高性能 attention kernel 与 channelwise RMSNorm kernel;**未展开 GEMV**(社区 FlashInfer 本身有 decode/GEMV kernels,但 sail 指南未提)。
- **ACTLIZE**(CUTLASS 体系)用户指南:v1.0.0 分支 README 无 gemv/vector 字样,聚焦 GEMM。
- **FlashAttention-for-sail**:支持 MQA/GQA、`flash_attn_with_kvcache` 增量解码路径(GQA/MQA decode 与 GEMV 相关,但文档未用 GEMV 一词)。

## 4. GEMV × 量化

- GEMV 族内核支持量化精度:DeepGEMM MQA logits 覆盖 **BF16 / INT8 / FP8 / FP8+FP4**;
- 硬件约束(810E):**只有 INT8**;FP8/FP4 仅 M890(与前面量化调研一致);
- FlashMLA:FP8 KV cache 读入后反量化→BF16 计算(减轻 KV cache 带宽压力,810E 上可用 FP8 KV cache,因它是存储压缩而非计算);
- 量化 GEMV(如 `int8_mqa_logits`)在 decode 阶段的收益:索引器 QK 打分是 decode 热点,INT8 化可降带宽/计算。

## 5. 与你的 Qwen3.5-2B 的关系

- Qwen3.5-2B 的 linear_attn 层是 **GatedDeltaNet**(in_proj_a/in_proj_b),**不使用** 闪电索引器/MQA logits;**MQA logits(GEMV 族)是 DeepSeek-V3.2/V4 系模型专用**;
- 如果你的目标是让 Qwen3.5-2B 的 decode 更快,更相关的 GEMV 型路径是 **FlashMLA 解码内核**(MLA)或 FlashAttention `flash_attn_with_kvcache`(MQA/GQA);Qwen3.5-2B 目前走 FA3 后端,不涉及 MQA logits。
- 若后续要研究 DeepSeek-V4 系在 810E 上的索引器性能,则直接看 `int8_mqa_logits`/`int8_paged_mqa_logits` 这条量化 GEMV 链路。

## 6. 原始材料清单(raw/)

| 文件 | 内容 |
| --- | --- |
| `79-FlashMLA-for-SAIL-完整指南.md` | 入口文档(文档 79),MLA 解码内核优化 |
| `FlashMLA-for-sail-README.md` | 上游 README |
| `DeepGEMM-for-sail-README.md` | **MQA logits / paged MQA logits / HC prenorm GEMM API 原文** |
| `DeepGEMM-社区-README.md` | 社区 DeepGEMM,V3.2 MQA kernels 说明 |
| `vllm-sail-ppu_deep_gemm.py` | vLLM-sail 包装层(含全部 GEMV 族函数) |
| `FlashInfer-for-SAIL-v0.6.12指南.md` / `FlashInfer-用户指南-版本索引.md` | FlashInfer-sail 指南 |
| `FlashAttention-用户指南.md` | FlashAttention-sail 指南 |
| `ACTLIZE-用户指南.md` | ACTLIZE 指南(无 GEMV) |
