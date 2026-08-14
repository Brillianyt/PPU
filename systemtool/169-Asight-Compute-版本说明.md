# Asight Compute 版本说明

## 新增改动

### 1. 新增 PM Sampling 功能
  - 可在 Details Page 中查看任意时刻的硬件指标变化

### 2. acu 新增统计系统功能
  - 系统可通过基于 Python 脚本的内置规则或自定义规则，汇总、计算性能数据，将结果导出到文件，或直接显示在终端

### 3. acu 命令行参数增强
  - `--csv` 参数支持与 `--page` 联动，输出格式受 `--page` 参数控制
  - 新增 `--csv-file <文件>` 参数，支持将 csv 数据直接输出到文件中

### 4. 加强 Details Page 的 tooltip 显示
  - 集成 tab 与表格等内容，丰富信息展示样式

### 5. 提升 baseline 对比体验
  - 当 current kernel 为 baseline 时，不再参与对比

### 6. 统一 Memory 相关 metrics 的命名
  - 按 Metric 命名规范修正一致性

### 7. 改进 kernel 选项错误提示
  - 使用 `--kernel-name` 或 `--kernel-id` 时，如遇到 "No Kernels" 错误，acu 会自动打印出当前程序中所有可用的 kernel 名称
