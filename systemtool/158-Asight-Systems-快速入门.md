# Asight Systems 快速入门

本文将引导您快速完成一次完整的性能分析流程：从采集数据到查看报告。

在本文中，将运行 PPU 程序的设备称为 **目标机（Target）**，将查看报告的设备称为 **主机（Host）**。

Asight Systems 的基本使用流程如下：

1. 在 **目标机（Target）** 上使用 **asys 命令行工具**采集性能数据并生成报告。
2. 在 **主机（Host）** 上使用 **Asight Systems GUI** 打开报告并进行性能分析。

**前提条件：**

+ 目标机已配备 PPU 设备，且驱动已正确安装。
+ 目标机已安装 asys 命令行工具。如尚未安装，请参见 [安装指南](/docs_center/doc_detail/index.html?projectId=1&chapterId=157)。
+ 主机已安装 Asight Systems GUI 工具。

## 1. 采集性能数据

在目标机上执行 `asys profile` 命令，启动目标应用程序并采集跟踪数据，完成后自动生成跟踪报告。

命令格式：

```bash
asys profile [options] <application> [application args]
```

以下示例演示如何对目标程序 `test` 进行数据采集，并将报告输出为 `report.asysrep`：

```bash
asys profile -o report test
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `-o report` | 指定输出报告文件名（生成的文件为 `report.asysrep`） |
| `test` | 要进行性能分析的目标应用程序 |

执行该命令后，终端将输出类似如下信息：

```text
[Asight]: Profile executed.

[Asight]: Start trace service, this may take a while...
[Asight]: Starting target app: ./test
[Asight]: Target application pid: 125290, pgrp:125290

[Asight]: Collecting data...

[Asight]: All target application created processes are terminated.
[Asight]: Trace written into the output file: report.asysrep
```

默认情况下，asys 会采集 `hggc,hgtx,acdnn,acblas` 跟踪项。如需指定其他跟踪项，可使用 `-t` 选项，例如：

```bash
asys profile -t hggc,hgtx -o report test
```

asys 支持多种跟踪项与灵活的采集控制方式，详细信息请参见 [asys 命令行工具](/docs_center/doc_detail/index.html?projectId=1&chapterId=159#l80Hg)。

!!! tip
    推荐将报告文件的尺寸控制在 200MB 以内。要减少报告尺寸，可以指定 `-s none` 选项关闭 CPU 采样，请参见 [CPU 采样](/docs_center/doc_detail/index.html?projectId=1&chapterId=159#CU82J)；也可以控制采样时长，请参见 [控制采集过程](/docs_center/doc_detail/index.html?projectId=1&chapterId=159#Oz0mt)。

## 2. 分析报告

asys 生成的报告文件使用 `.asysrep` 作为文件后缀。报告文件是独立的，可以从目标机拷贝到主机上，使用 Asight Systems GUI 打开并进行分析。

Asight Systems GUI 支持以下方式打开报告：

+ 主菜单打开：File -> Open...
+ 将文件拖拽（Drag & Drop）到 Asight Systems GUI 主窗口
+ 右键菜单打开：在 Project Explorer 空白区域点击右键 -> Open...
+ 通过命令行打开：`asys-ui report.asysrep`

!!! note "注意"
    + 用于查看报告的 Asight Systems GUI 版本不应低于生成报告时使用的 asys 工具版本。
    + 如果使用较旧版本的 GUI 打开新版本工具生成的报告，GUI 会给出相应提示。

有关 GUI 的更多功能和使用说明，请参见 [使用 Asight Systems GUI 查看报告](/docs_center/doc_detail/index.html?projectId=1&chapterId=159#cEMJf)。

## 3. 下一步

完成上述流程后，您已经掌握了 Asight Systems 的基本使用方法。如需了解更多高级功能，请参阅以下文档：

+ [使用指南](/docs_center/doc_detail/index.html?projectId=1&chapterId=159) — 详细介绍 asys 命令行工具与 GUI 的各项功能
+ [常见问题](/docs_center/doc_detail/index.html?projectId=1&chapterId=160) — 使用过程中的高频问题与解决方案
