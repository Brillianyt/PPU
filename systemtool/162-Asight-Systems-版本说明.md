# Asight Systems 版本说明

## 新增改动

### 1. asys stats 统计规则增强
  - 新增 `PPU operator type summary` 统计规则，支持分析算子类型占比、列出热点算子
  - 支持 `PPU operator trace` 跟踪导出，导出每算子关联的 PPU 活动跟踪
  - 新增 `PPU kernel summary compare` 对比规则，比较两个报告的 PPU kernel 汇总结果
  - 新增 `HGTX PPU projection summary compare` 对比规则，比较两个报告的 HGTX→PPU 投影汇总结果
  - 新增 `PPU operator summary compare` 对比规则，比较两个报告的算子汇总结果
  - 新增 `PPU operator type summary compare` 对比规则，比较两个报告的算子类型汇总结果

### 2. PPU operator summary 统计功能增强
  - 支持识别更多算子
  - 支持自定义算子识别规则

### 3. HGTX 关联 PPU 活动统计增强
  - 支持 `range-include` 和 `range-exclude` 选项，指定 HGTX 黑/白名单
  - `PPU 算子统计规则` 支持 `order-by` 选项，指定排序方式

### 4. asys stats 命令参数与输出增强
  - 指定统计报告类型时支持对逗号 `,` 和冒号 `:` 使用 `\` 转义
  - 未指定类型时：
    - 默认在终端打印简要统计结果
    - 输出详细统计结果到文件

### 4. 统计输出样式优化
  - 调整 `asys stats` 和 `analysis` 子命令 column 样式，使输出结果更加易读
  - 调整 `asys stats` 默认生效的统计规则

### 5. PCCL Desynchronization Summary 增强
  - 新增 P90 指标统计，显示大部分 PCCL 通信不同步率优于此门限

### 6. 报告对比功能扩展
  - 新增 `asys compare` 子命令，比较两个报告文件
    - 支持显示精简结果
    - 输出详细比较数据到 CSV 文件
  - 新增 `PCCL desynchronization summary compare` 对比规则
  - 新增 `PPU time utilization compare` 对比规则
  - 新增 `PPU Operator Breakdown` 与 `PPU Operator Kernel Breakdown` 统计规则
  - 新增 `Breakdown Compare` 对比规则，比较性能分解结果
  - 新增比较视图（Compare View），可选择任意两个 `.asysrep` 报告对算子、kernel 性能进行对比

### 7. PyTorch 算子采集与分析支持
  - 采集 PyTorch 算子跟踪
  - 支持通过 HGTX 标记 PyTorch 算子的执行时间
  - 支持采集 PyTorch 算子参数信息

### 8. PPU operator unit test 规则（实验特性）
  - 支持批量生成 PyTorch 算子测试用例代码

### 9. Function View 功能增强
  - 支持选择目标线程，仅对该线程进行分析

### 10. 时间线显示改进
  - tooltip 支持切换显示完整时间线名字
  - 选中时间线时高亮效果和边框颜色加强
  - 支持统一高度比例尺显示 PPU metrics 与 CPU metrics 时间线
  - 当 HGTX 行层数较多时，标题名字保持居中显示
  - Timeline View 更新 range filter 时，自动同步 Heap Memory View 内容

### 11. Events View 优化
  - 默认使用 list 模式，并提升操作性能

### 12. asys profile 新增功能
  - 新增 `--stats` 选项，报告生成后打印简要统计结果到终端，并输出详细结果到文件
  - 新增 `--format` 选项支持 `xlsx` 格式输出
