# 环境备份 — 完整还原所需的所有信息

> **关键认知**：DSW 实例 + CPFS 只能保护"数据"，但环境本身（Python 包、asys、PPU SDK、sglang PPU patch）全部依赖**镜像**。
> **没有镜像，requirements.txt 再详细也复现不了**。

## 1. 镜像（Source of Truth）

```
pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.10.0-ubuntu24.04-cuda13.0-sglang0.5.10-py312
```

⚠️ 这个 tag **不可再生**：里面包含阿里自研的 sglang PPU patch（+v0.3.0.ppu2.1.1）、deep_gemm PPU 版、deep_ep、fla 等私域 wheel，**不能从 PyPI 装**。

镜像丢失的应对：
- 联系阿里赛事组重发镜像
- 自行 docker pull 备份：`docker pull <上面这个 tag> && docker save -o llm-ppu-v2.1.1.tar <tag>`

## 2. Python 环境

- Python: 3.12.3
- pip: 24.0
- 总包数: **305**（`requirements-frozen.txt` 完整列表）
- PPU 专用包: **27 个**（`ppu-specific-packages.txt`）
- 关键 PPU 包：
  - `sglang==0.5.10+v0.3.0.ppu2.1.1`
  - `torch==2.10.0` + `cuda-python==13.0.0+v0.1.0.ppu2.1.1`
  - `deep_gemm==1.0.0+v0.3.0.ppu2.1.1`（INT8 GEMM）
  - `deep_ep==1.0.0+v0.2.0.ppu2.1.1`
  - `fla==1.0.0+v0.1.0.ppu2.1.1`（linear attention）
  - `cpm_kernels==1.0.11+v0.1.0.ppu2.1.1`

## 3. 系统层

| 项 | 值 |
|---|---|
| OS | Ubuntu 24.04.2 LTS (Noble Numbat) |
| 内核 | Linux 5.10.134 |
| PPU Driver | 1.3.2-d7f5a2 |
| HGGC Version | 13.0 |
| PPU SDK | /usr/local/PPU_SDK/ |
| asys (profiler) | v2.1.1_20260725-15d8b9d |

## 4. 启动 sglang server 的精确命令（G 组，最终配置）

```bash
SGLANG_SAIL_FLA_CUDA=1 python3 -u -m sglang.launch_server \
  --model-path ./Qwen3.5-2B \
  --host 127.0.0.1 --port 8999 \
  --trust-remote-code --tp-size 1 \
  --attention-backend fa3 \
  --enable-multimodal \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --page-size 64 \
  --log-level info
```

**关键环境变量**（不设会回到 419ms TTFT）：
- `SGLANG_SAIL_FLA_CUDA=1` ← 文档与代码默认值不符，源码里是 False，必须显式设

## 5. 文件清单

| 文件 | 内容 |
|---|---|
| `requirements-frozen.txt` | 305 个包的精确版本（pip freeze） |
| `ppu-specific-packages.txt` | 27 个 PPU 专用包（**不能从 PyPI 装**） |
| `system-info.txt` | Python/asys/sglang/torch/PPU/系统版本 |

## 6. 还原步骤（新 DSW 实例 + 镜像挂 CPFS）

```bash
# 1) 新建 DSW 实例时选官方镜像
#    镜像: pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.10.0-ubuntu24.04-cuda13.0-sglang0.5.10-py312
#    存储挂载: CPFS -> /mnt/workspace
#
# 2) 实例起来后
cd /mnt/workspace
git clone https://github.com/Brillianyt/PPU.git dndx_participant-v1.2
cd dndx_participant-v1.2

# 3) 验证环境（无需重装 pip，镜像自带）
python3 -c "import sglang; print(sglang.__version__)"   # 0.5.10+v0.3.0.ppu2.1.1
python3 -c "import deep_gemm; print(deep_gemm.__file__)"

# 4) 下载模型权重（如果 CPFS 上没缓存）
huggingface-cli download Qwen/Qwen3.5-2B --local-dir ./Qwen3.5-2B
# 或从公开下载源:
# wget https://your-mirror/Qwen3.5-2B.tar.gz && tar -xzf Qwen3.5-2B.tar.gz

# 5) 下载测试集
mkdir -p datasets/mmbench
wget -O datasets/mmbench/mmbench_dev_en.tsv https://your-mirror/mmbench_dev_en.tsv
wget -O datasets/mmbench/mmbench_dev_cn.tsv https://mirror/mmbench_dev_cn.tsv

# 6) 启动 + benchmark
bash docs/06_FINAL_CONFIG.md 中的 run_g_group.sh
```

## 7. 备份 vs 复现：什么是不能复现的

| 类别 | 备份状态 | 复现难度 |
|---|---|---|
| 镜像（base） | ❌ 没有 docker save | **如果镜像 tag 失效就废了** |
| PPU SDK + asys | ✅ 在镜像里 | 跟着镜像 |
| Python 全部包 | ✅ requirements-frozen.txt | 跟着镜像，**不需重装** |
| Qwen3.5-2B 权重 | ❌ 不在 git（4.3G） | 重新下载 |
| mmbench 测试集 | ❌ 不在 git（250M） | 重新下载 |
| 工作代码 | ✅ GitHub 67 文件 | git clone |
| 最终配置 | ✅ docs + wrapper 代码 | git clone |
| 评测结果 | ✅ result_*.json | 跟着 git clone |
| profile 汇总 | ✅ profile_results/*op_sum.csv | 跟着 git clone |
| 巨大 profile trace | ❌ .gitignore | 用 asys 重新抓 60s |
