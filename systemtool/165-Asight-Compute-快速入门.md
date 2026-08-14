# Asight Compute 快速入门

本文将引导您快速完成一次完整的 Kernel 性能分析流程：从采集数据到查看报告。

在本文中，将运行 PPU 程序的设备称为 **目标机（Target）**，将查看报告的设备称为 **主机（Host）**。

Asight Compute 的基本使用流程如下：

1. 在 **目标机（Target）** 上使用 **acu 命令行工具**采集性能数据并生成报告。
2. 在 **主机（Host）** 上使用 **Asight Compute GUI** 打开报告并进行性能分析。

**前提条件：**

+ 目标机已配备 PPU 设备，且驱动已正确安装。
+ 目标机已安装 acu 命令行工具。如尚未安装，请参见 [安装指南](/docs_center/doc_detail/index.html?projectId=1&chapterId=164)。
+ 主机已安装 Asight Compute GUI 工具。

## 1. 采集性能数据

在目标机上使用 **acu** 命令行工具启动目标应用程序，并在应用程序运行期间采集性能数据。应用程序结束后，acu 会自动输出 profiling 报告文件。

命令格式：

```bash
acu [options] <application> [application args]
```

以下示例演示如何采集目标程序中所有 Kernel 启动的性能数据：

```bash
acu -o test_report -f python test_linear.py
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `-o test_report` | 指定输出报告名称（无需指定文件后缀名） |
| `-f` | 强制覆盖已存在的报告文件 |
| `python test_linear.py` | 要进行性能分析的目标应用程序 |

应用程序运行结束后，acu 会生成对应的报告文件：`test_report.acurep`。


!!! note
    如果 acu 运行时报错：`device is not ready for profiling`，请参考 [报错：device is not ready for profiling](/docs_center/doc_detail/index.html?projectId=1&chapterId=167#J5Ul3) 处理。

acu 还提供了多种高级选项，用于精细控制采集范围和内容：

+ **过滤 Kernel**：通过 `-c`、`-s`、`-k` 等选项按启动次数或名称过滤采集的 Kernel。详见：[过滤 Kernel](/docs_center/doc_detail/index.html?projectId=1&chapterId=166#NjM0a)
+ **指定采集指标**：通过 `--set`、`--section`、`--metrics` 等选项指定采集的性能指标集合。详见：[指定采集的 Metrics 集合](/docs_center/doc_detail/index.html?projectId=1&chapterId=166#Uu8pH)
+ **Replay 模式**：当采集的 Metrics 超过硬件计数器限制时，acu 会自动通过多次执行应用获取完整数据。详见：[Replay 模式](/docs_center/doc_detail/index.html?projectId=1&chapterId=166#UziBy)
+ **终端输出**：不指定 `-o` 时，acu 默认在终端输出 `details` 页面。可通过 `--page=<Page>` 指定输出页面，当前支持：`details`、`raw`

## 2. 分析报告

acu 生成的报告文件使用 `.acurep` 作为文件后缀。报告文件是独立的，可以从目标机拷贝到主机上，使用 Asight Compute GUI 打开并查看分析结果。

Asight Compute GUI 支持以下方式打开报告：

+ 主菜单打开：File -> Open...
+ 将文件拖拽（Drag & Drop）到 Asight Compute GUI 主窗口
+ 右键菜单打开：在 Project Explorer 空白区域点击右键 -> Open...
+ 通过命令行打开：`acu-ui report.acurep`

!!! note
    + 用于查看报告的 Asight Compute GUI 版本不应低于生成报告时使用的 acu 工具版本。
    + 如果使用较旧版本的 GUI 打开新版本工具生成的报告，GUI 会给出相应提示。

有关 GUI 的更多功能和使用说明，请参见 [Asight Compute GUI 工具](/docs_center/doc_detail/index.html?projectId=1&chapterId=166#XOXk5)。

## 3. 下一步

完成上述流程后，您已经掌握了 Asight Compute 的基本使用方法。如需了解更多高级功能，请参阅以下文档：

+ [使用指南](/docs_center/doc_detail/index.html?projectId=1&chapterId=166) — 详细介绍 acu 命令行工具与 GUI 的各项功能
+ [常见问题](/docs_center/doc_detail/index.html?projectId=1&chapterId=167) — 使用过程中的高频问题与解决方案
