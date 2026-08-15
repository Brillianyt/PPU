# ACTLIZE 用户指南

actlize 是面向真武 PPU（Parallel Process Unit）平台的矩阵计算与算子加速基础库。该项目基于 CUTLASS 的高性能矩阵计算体系进行 PPU 适配，重点围绕 GEMM 等核心计算路径，提供更贴近 PPU 架构特性的底层实现能力。

在 SAIL AI 软件生态中，actlize 更偏向基础设施层：它不直接面向最终模型业务，而是为上层算子库、注意力计算库、推理组件和相关示例工程提供可复用的 PPU 计算内核与实现参考。

## 技术定位

actlize 的核心目标是在 PPU 硬件上承载高性能矩阵计算，尤其是围绕 AIU tensor cell 操作对 GEMM 执行路径进行适配与优化。相比通用示例代码，仓库中的 PPU 专属实现更关注 PPU 数据搬运、矩阵乘累加、流水组织和后处理等环节的协同。

该项目主要服务于需要在真武 PPU 上获得高效矩阵计算能力的底层软件场景，例如上层算子库适配、PPU kernel 开发参考，以及依赖特定 actlize 分支能力的生态项目补全。

## 分支指南

actlize GitHub 仓库以版本分支组织不同阶段的 PPU 适配内容。不同分支面向的 CUTLASS 基线、适用项目和能力范围有所差异；具体构建命令、环境要求、功能清单、示例代码和性能说明请以对应分支 README 为准。

### [actlize v0.5.0 用户指南](/docs_center/doc_detail/index.html?projectId=6&documentId=62)

v0.5.0 是基于 CUTLASS 2.5.0 的早期 PPU 适配分支，主要用于支撑依赖该版本的历史生态项目。该分支除早期 PPU GEMM 路径外，也保留了卷积相关适配说明。

### [actlize v0.8.0 用户指南](/docs_center/doc_detail/index.html?projectId=6&documentId=63)

v0.8.0 是基于 CUTLASS 3.4.1 的 PPU 适配分支，主要围绕 PPU GEMM、AIU tensor cell 操作以及相关上层生态项目的配套需求展开。

### [actlize v1.0.0 用户指南](/docs_center/doc_detail/index.html?projectId=6&documentId=64)

v1.0.0 是基于 CUTLASS 3.6.0 的主线 PPU 适配分支，面向 ppu001/ppu0015 提供更完整的能力参考，并包含更系统的 README 文档入口、API 说明和 Profiler 相关内容。

## 相关文档

- 用于支持框架依赖的转换脚本：[CUDAFY-for-SAIL 用户指南](/docs_center/doc_detail/index.html?projectId=6&chapterId=66)。
