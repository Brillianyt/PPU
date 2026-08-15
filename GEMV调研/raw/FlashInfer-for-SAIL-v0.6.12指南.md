# FlashInfer-for-SAIL v0.6.12 用户指南

## 1. 简介

FlashInfer-for-SAIL 是为平头哥自研 AI 加速芯片深度定制和优化的推理算子库。本算子库在 Flashinfer 社区核心的推理算子库基础上，针对真武 PPU 的硬件架构特点，深度集成了真武 PPU 特有的高性能算子（如高效的 attention kernel 和 channelwise RMSNorm kernel 等），以最大化发挥真武 PPU 的计算与显存带宽优势。FlashInfer-for-SAIL v0.6.12 基于社区 v0.6.12 版本适配，支持直接通过 PyPI 源和 Docker 镜像安装使用，也支持从源码编译后使用。

## 2. 安装指南

### 2.1. 使用镜像安装

推荐使用 SGLang-for-SAIL v0.5.13 Docker 镜像，该镜像中已预装了 v0.6.12 版本，可避免繁琐的基础环境配置。镜像获取及安装方法请参考 [SAIL Docker 下载中心](/download?businessType=DOCKER)及 [Docker 安装指南](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=2&documentId=137)。

**拉取镜像并启动容器示例：**
```bash
# 设置镜像名称
IMAGE_NAME="pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.11.0-ubuntu24.04-cuda13.0-sglang0.5.13-py312"

# 创建并启动容器
sudo docker run -it \
  --name ppu_sglang_0.5.13 \
  --privileged \
  --ipc=host \
  --device=/dev/alixpu_ctl \
  --device=/dev/alixpu \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --init \
  --shm-size=8g \
  -v /path/to/host/:/path/to/container/ \
  ${IMAGE_NAME} bash
```

### 2.2. PyPI 源安装

PyPI 源 wheel 包的下载及安装方法请参考 [SAIL PyPI 下载中心](/download?businessType=PIP)及 [PIP 源使用指南](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=2&documentId=138)。

使用以下命令从 PyPI 源安装：

```bash
pip install flashinfer_python==0.6.12 -i https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple
```

!!! note 

    * FlashInfer 集成 TVM-FFI 作为与底层框架的接入层。v0.6.12 版本运行时推荐使用 apache-tvm-ffi 0.1.6, 可使用命令 `pip install apache-tvm-ffi==0.1.6` 安装
    * FlashInfer-for-SAIL 暂不支持 FP8
    * 在 v0.2.x 版本，FlashInfer 的 whl 包名从 flashinfer 更改为 flashinfer-python，安装 FlashInfer v0.2.x 版本前请先卸载环境中的 flashinfer

### 2.3. 从源码编译

* Requirements 和社区 FlashInfer 相同；
* 编译前请确认环境中安装了 2.1 或以上版本 SAIL SDK;
* 推荐使用与 SAIL SDK 版本匹配的 SAIL Docker 镜像；
* 在编译前，需要通过以下命令对 FlashInfer 的子模块进行初始化;

```bash
git clone https://github.com/flytiger-eco/flashinfer-for-sail.git -b v0.6.12
cd flashinfer-for-sail/
git submodule update --init --recursive
```

* 编译命令:

```bash
pip install packaging setuptools build --force-reinstall --no-deps
pip install apache-tvm-ffi==0.1.6
pip install pynvml==13.0.1
export TORCH_CUDA_ARCH_LIST="8.0"
export FLASHINFER_CUDA_ARCH_LIST="8.0"
python -m flashinfer.aot
python -m build --no-isolation --wheel
pip install dist/*.whl
```

### 2.4. 安装验证
安装完成后，请验证 FlashInfer 是否已正确安装和配置。可以参考 Flashinfer 官方文档的[Verify Installation](https://github.com/flashinfer-ai/flashinfer/tree/v0.6.12#verify-installation)验证环境的正确性。

```bash
flashinfer show-config
```

## 3. 快速开始

参考 Flashinfer 的官方文档，使用 `single_decode_with_kv_cache` 进行 attention 推理：
```python
import torch
import flashinfer

# Single decode attention
q = torch.randn(32, 128, device="cuda", dtype=torch.float16)  # [num_qo_heads, head_dim]
k = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)  # [kv_len, num_kv_heads, head_dim]
v = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)

output = flashinfer.single_decode_with_kv_cache(q, k, v)
```
更多 API 和环境变量使用方法请参考[flashinfer.ai](https://docs.flashinfer.ai/)。

## 4. 已知问题
* 测试结果：

| UT Case     | Instance |
| :---: | :---: |
| Test Case 总数 | 361424 |
| Pass 总数 | 68803 |
| Xpassed 总数 | 0 |
| Xfail 总数 | 4726 |
| Fail 总数 | 25021 |
| Skip 总数 | 262874 |
| Pass Rate | 73.31% |

* 已知问题列表：

| UT Case      |  Fail Instance | Fail Reason |
| :---: | :---: | :---: |
| test_batch_attention.py      | 23039       | 没有对 mma 适配，sglang 暂时没有支持需求       |
| test_attention_sink.py    | 1336        | 用了 hopper 的特性        |
| test_cudnn_decode.py    | 12        | cudnnFinalize failedptrDesc->finalize() cudnn_status: CUDNN_STATUS_BAD_PARAM，cudnn 版本兼容问题        |
| test_group_gemm.py     | 26        | AssertionError(AssertionError: Unknown cases: sf.dtype=torch.float32) 框架侧有替代方 triton/deepgemm      |
| test_trtllm_alltoall.py    | 6        | trtllm 底层库不支持 AssertionError: Process 0 failed with exit code 1；test_trtllm_moe_allreduce_fusion.py:443: AssertionError        |
| test_trtllm_gen_mla.py    | 1232        | trtllm 底层库不支持        |
| test_nvshmem.py    | 1        | 暂时没有接入 nvshmem        |
| 例如 test_attention_sink_blackwell.py/test_groupwise_scaled_gemm_fp8.py 等   | /        | 真武 810/810E 不支持 fp8 的计算与 blackwell 的一些特性，相关的 case fail 在预期之内        |
