# FlashMLA-for-sail Release 用户指南

本指南在平头哥（T-Head）提供的 PPU 原生 FlashMLA-for-sail 版本基础上，提供面向社区 CUDA 软件栈的兼容解决方案。PPU 原生仓库无法天然适配社区已有的 CUDA 软件栈，因此在 CUDA 兼容场景下，构建前需要先对 FlashMLA-for-sail 及其依赖的 ACTLIZE 完成 CUDA 兼容转换，转换方法请参考 [CUDAFY-for-SAIL](/docs_center/doc_detail/index.html?projectId=6&chapterId=66)。

## 1. 概述

FlashMLA-for-sail 是 FlashMLA 在真武 PPU（Parallel Process Unit）平台上的适配版本，面向 MLA（Multi-head Latent Attention）架构的大模型推理解码场景，提供 dense decoding、DeepSeek Sparse Attention prefill / decoding 等 PPU 内核能力。该版本保留上游常用 Python API 习惯，并针对 PPU 的访存、TSM 布局、Tensor Cell、warp 调度和 Split-KV 任务分配做了专项优化。

支持范围请参阅 [README「Introduction」](https://github.com/t-head/FlashMLA-for-sail#introduction) 和 [README「Requirements」](https://github.com/t-head/FlashMLA-for-sail#requirements) 章节。

## 2. PPU 后端扩展与优化

FlashMLA-for-sail 围绕数据搬运、共享存储布局、矩阵计算、warp 分工、metadata 调度和编译后端编排做了专项优化。

- **AIU + TSM Swizzle 数据通路**：Q/K/V 与 KV cache 分块数据通过 AIU 接入异步搬运，并在写入 TSM / shared memory 时配合 swizzle 布局，使数据进入共享存储后更贴合后续 MMA 操作数读取方式，减少额外重排和访存冲突。
- **PPU Tensor Cell 映射**：QK 与 PV 的矩阵乘加采用分块 MMA 方式组织，并映射到 PPU Tensor Cell 指令；同时配合 accumulator layout 转换，覆盖 FP16/BF16 计算路径以及 FP8 KV cache 读入后的 BF16 计算路径。
- **Warp interleave / CrossCut 调度**：dense MLA 解码路径会根据序列长度、batch 和设备能力选择 CrossCut 与 warp interleave 方案，将 QK、softmax、PV 和输出写回在不同 warp group 间交错推进，提升解码阶段的并行度和流水利用率。
- **Tile 与寄存器驻留调优**：针对不同 `seqlen_q`、head 数、page block、top-k 和 KV cache 形态选择 block shape、warp 数、stage 数、MMA atom 形态和 Q 寄存器驻留策略，适配 dense decoding、sparse prefill 和 sparse decoding 的不同访存/计算比例。
- **Metadata + Split-KV 负载均衡**：运行前通过 metadata 预计算 batch 内 tile 划分和 split 数量，解码 kernel 根据 metadata 分发任务，用于改善长短序列、变长 batch、top-k sparse attention 和多 split 场景下的负载均衡。
- **Sparse / FP8 KV cache 路径优化**：DSA 解码路径支持 token-level sparse attention、extra KV、attn sink、top-k length 等场景；FP8 KV cache 读入后完成反量化并进入 BF16 计算，减少 KV cache 带宽压力。
- **编译选项辅助后端编排**：构建参数中启用 PPU/AIU 支持，并配置寄存器数量、矩阵地址下沉、异步地址下沉、load/store 地址下沉、warpage 和虚拟寄存器重排等后端优化选项，帮助编译器更好地编排访存、寄存器使用和计算流水。

## 3. 环境要求

环境要求请参阅 [README「Requirements」](https://github.com/t-head/FlashMLA-for-sail#requirements) 章节。

## 4. 构建与安装

FlashMLA-for-sail 支持源码构建和 wheel 包安装。构建前需先使用 CUDAFY-for-SAIL 的转换脚本，对 FlashMLA-for-sail 及其依赖的 ACTLIZE 完成 CUDA 兼容转换：

```bash
# 转换 FlashMLA-for-sail 源码树
python3 cudafy.py flashmla /path/to/FlashMLA

# 转换依赖的 ACTLIZE include 目录，FlashMLA-for-sail 对应 ACTLIZE 0.8.0
python3 cudafy.py actlize --version=0.8.0 /path/to/actlize/include
```

转换脚本的支持版本、参数说明和使用注意事项请参阅 [CUDAFY-for-SAIL Release 用户指南](/docs_center/doc_detail/index.html?projectId=6&documentId=66)。

转换完成后，具体构建方式、依赖安装和测试命令请参考项目仓库 README。

## 5. 编程接口与使用

FlashMLA-for-sail 保持与项目 README 中描述的一致使用方式。MLA decoding、Sparse Attention prefill / decoding、FP8 KV cache、metadata 和 sparse indices 等接口说明与示例请参考项目仓库 README。

## 6. 已知问题

- 以项目 README 和发行说明中的说明为准。

## 7. 更多信息

详细的编程接口、使用示例和测试说明请参阅项目仓库文档。

- CUDA 兼容转换工具：[CUDAFY-for-SAIL 用户指南](/docs_center/doc_detail/index.html?projectId=6&chapterId=66)。
