# FlashAttention-for-SAIL 用户指南

FlashAttention 是一套针对 PPU 内存层次结构优化的高效注意力实现，通过重新设计 HBM 与片上 SRAM 之间的数据搬运方式，避免在 HBM 中存储完整的中间注意力矩阵，在保持精确计算的同时大幅降低长序列场景下的显存占用并提升吞吐。

主要能力：

- 支持因果掩码（causal mask）与滑动窗口（local）注意力。
- 支持变长序列与 cross-attention。
- 支持 Multi-Query Attention（MQA）与 Grouped-Query Attention（GQA）。
- 支持基于 KV Cache（含分页 KV Cache）的增量解码，并提供面向推理的 `flash_attn_with_kvcache` 快速路径。
- 支持 ALiBi、RoPE 位置编码及 dropout。
- 支持确定性的反向传播。
- 提供可直接替换标准注意力的 Python API（`flash_attn_func` 等），head 维度支持至 256。

代码仓库：https://github.com/t-head/flash-attention-for-sail

详细使用说明请参阅仓库 README。

本系列文档在平头哥（T-Head）提供的 PPU 原生 flash-attention-for-sail 版本基础上，提供面向社区 CUDA 软件栈的兼容解决方案。在 CUDA 兼容场景下，构建前需要先对 flash-attention-for-sail 及其依赖的 ACTLIZE 完成 CUDA 兼容转换，转换方法请参考 [CUDAFY-for-SAIL](/docs_center/doc_detail/index.html?projectId=6&chapterId=66)。

## 版本列表

### [flash-attention-for-sail v2.8.2 用户指南](/docs_center/doc_detail/index.html?projectId=6&documentId=74)

本指南介绍 flash-attention-for-sail v2.8.2 的版本能力、PPU 后端优化和构建/接口参考入口。

- 面向 SAIL SDK Release 版本构建。
- 基于社区 FlashAttention 2.8.2 版本适配。
- 构建前需使用 [CUDAFY-for-SAIL](/docs_center/doc_detail/index.html?projectId=6&chapterId=66) 完成 CUDA 兼容转换。

### [flash-attention-for-sail v2.7.4 用户指南](/docs_center/doc_detail/index.html?projectId=6&documentId=73)

本指南介绍 flash-attention-for-sail v2.7.4 的版本能力、PPU 后端优化和构建/接口参考入口。

- 面向 SAIL SDK Release 版本构建。
- 基于社区 FlashAttention 2.7.4.post1 版本适配。
- 构建前需使用 [CUDAFY-for-SAIL](/docs_center/doc_detail/index.html?projectId=6&chapterId=66) 完成 CUDA 兼容转换。

### [flash-attention-for-sail v2.7.2 用户指南](/docs_center/doc_detail/index.html?projectId=6&documentId=72)

本指南介绍 flash-attention-for-sail v2.7.2 的版本能力、PPU 后端优化和构建/接口参考入口。

- 面向 SAIL SDK Release 版本构建。
- 基于社区 FlashAttention 2.7.2 版本适配。
- 构建前需使用 [CUDAFY-for-SAIL](/docs_center/doc_detail/index.html?projectId=6&chapterId=66) 完成 CUDA 兼容转换。
