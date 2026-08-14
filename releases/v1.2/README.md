# v1.2 Release Artifacts

比赛 G 组配置（FA3 + FLA=1 + no-radix + page=64）相关备份。

## 成绩单

| 指标 | 数值 |
|---|---|
| accuracy | 0.85 |
| avg_ttft_ms | 64.7 |
| throughput tok/s | 369.0 |

## 文件

- `dndx-v1.2.bundle` (280 KB) — git bundle，包含完整 git 历史（master 分支所有对象），单文件可用 `git clone` 还原
- `dndx-v1.2-working-tree.tar.gz` (322 KB) — 不含 .git / 模型权重 / 巨大 trace / 测试数据集的工作树快照

## 还原方法

```bash
# 从 bundle 还原（保留完整 git 历史）
git clone dndx-v1.2.bundle dndx_participant-v1.2
cd dndx_participant-v1.2
git remote set-url origin https://github.com/Brillianyt/PPU.git

# 从 tar 还原（只是文件，不带 git 历史）
tar -xzf dndx-v1.2-working-tree.tar.gz
```

## 不包含的内容

- 模型权重（`Qwen3.5-2B/`）
- INT8 量化包（`quant/`）
- 测试数据（`datasets/mmbench/*.tsv`）
- 巨大原始 trace（`.asysrep` 等，可通过 asys 重新抓取）
