# Asight Systems 常见问题

## 1. 减小报告尺寸
推荐将采集的报告尺寸控制在 200MB 以内，可以采用以下办法减小报告尺寸：

+ 关闭 CPU 采样：使用选项`-s none`
+ 延迟启动采集，控制采样时间：使用`-y`和`-d`
+ 通过事件触发采集：使用`-c`指定触发方式
+ 通过`--trace-rotation`选项配合`--duration`选项进行报告文件轮转采集
+ 采集类似 mpirun 方式启动应用时，使用`--process-rank-include`指定只采集部分 Rank 进程的跟踪数据

对于已采集到的大尺寸报告，可在 Asight Systems GUI 中通过`File` -> `Open Advanced...`自定义查看报告的部分内容：

+ 约束加载部分报告的比例（如只加载前 50%的跟踪数据）
+ 不解析指定类型的跟踪数据（如忽略 CPU 调度信息跟踪数据）

!!! note "注意"
    使用加载部分报告功能打开配置`capture range`的报告文件时，或者打开采集时间较短的报告文件时，若部分加载比例较低，可能打开报告后无跟踪数据，请尝试扩大部分加载的比例。


## 2. 启动应用后采集多份报告
在应用启动后，可通过以下方法采集多份报告：

可通过`asys launch`启动应用，随后通过`asys start/stop`启动采集，例如：

```bash
asys launch -t hggc cuda_test
asys start
asys stop
asys start
asys stop
```

若应用程序中包含多个 capture range 的插桩，可以通过`--capture-range-end repeat-shutdown:N`多次触发采集，例如通过 capture range 事件触发生成 3 个报告文件：

```bash
asys profile --capture-range-end repeat-shutdown:3 python test_linear.py 
```



可通过`asys profile`子命令的`--duration`和`--trace-rotation`选项，通过报告文件轮转的方式，周期地生成报告文件，例如每 30 秒生成一个报告文件，保留最近的 3 份报告文件：

```bash
asys profile --duration 30 --trace-rotation 3 --output test_report_%t python test_linear.py 
```



可通过`asys attach`子命令通过多次附着到目标应用的方式，采集多份报告，例如多次采集 PID 为 94644 的进程跟踪数据，可多次执行命令：

```bash
asys attach -o report_%i 94644
```

## 3. 通过事件触发跟踪采集
asys 支持以事件触发的方式进行采集，例如：

```bash
asys profile -c hgtx -p msg python test_linear.py 
```

事件支持 NVTX 与 profilerStart/Stop 两种标记，下面给出了一些添加事件标记的例子：

**PyTorch 框架中增加 NVTX 范围信息：**

```python
import torch.cuda.nvtx as nvtx

nvtx.range_push("msg")
...
nvtx.range_pop()
```

**C/C++应用增加 NVTX 标记：**

```cpp
#include <nvtx3/nvToolsExt.h>

nvtxRangePushA("msg");
...
nvtxRangePop();
```

**PyTorch 框架中增加 profilerStart/Stop 标记：**

```python
torch.cuda.cudart().cudaProfilerStart()
...
torch.cuda.cudart().cudaProfilerStop()
```

## 4. 采集 OSRT 信息查看 CPU 线程挂起的原因
通过采集 OSRT 信息可以查看 CPU 线程挂起的原因。

指定采集 OSRT 跟踪：跟踪选项-t 参数增加 osrt

+ 可通过--osrt-threshold 选项指定采集门限
+ 可通过--osrt-backtrace-threshold 指定调用栈回溯门限

## 5. 使用 asys 采集 mpirun 等多机多卡应用
对于需要在多个计算节点的多个 PPU 上运行的应用程序，运行时常常通过`mpirun`/`deepspeed`等方式进行启动，asys 采集此类应用程序时，可通过以下方式采集跟踪数据，以`mpirun`应用为例：

+ 使用多节点跟踪采集模式（推荐）：
    - 通过 asys 启动`mpirun`应用，通过`--multi-node-mode mpirun`使能多节点跟踪采集模式，每个节点生成一份报告文件
+ 不使用多节点采集模式：
    - 单节点方式：通过 asys 启动`mpirun`应用，本节点的所有进程跟踪数据记录到一份报告文件，其他节点不采集跟踪数据
    - 多节点方式：通过`mpirun`应用启动 asys，每个节点的每个应用（通常对应一个 Rank / 一个 PPU）跟踪数据记录到一个报告文件

单节点方式和多节点方式的使用方法举例如下，采集单节点跟踪信息，汇总生成一份报告文件：

```bash
asys profile mpirun -np 4 cuda_test
```

多个节点采集，每个应用程序生成一份报告文件，可使用`-o`选项`%h`和`%p`根据 host name 和 PID 生成报告名称，避免报告重名问题。可使用`-o`选项通过环境变量生成报告名称，例如通过`mpirun`启动的应用程序，`-o`选项可通过`%q{OMPI_COMM_WORLD_RANK}`在报告名称中包含 Rank ID。

```bash
mpirun -np 4 asys profile -o report_%h_%p cuda_test
```

在使用单节点方式采集跟踪数据时，由于所有进程的跟踪数据记录到同一份报告文件，跟踪数据量可能巨大，若希望只采集部分进程的跟踪数据，可通过`--process-rank-include`选项指定需要采集的进程的 Rank ID，多个 Rank ID 之间通过逗号`,`拼接，例如只采集 Rank ID 为`0`和`2`的进程的跟踪数据：

```bash
asys profile --process-rank-include 0,2 mpirun -np 4 cuda_test
```

可通过`--process-ppu-include`选项指定采集的 PPU 设备 ID 列表，多个 PPU 设备 ID 之间通过逗号`,`拼接，例如只采集使用 PPU 设备 ID 为`0`和`2`的进程的跟踪数据：

```bash
asys profile --process-ppu-include 0,2 mpirun -np 4 cuda_test
```



## 6. 采集包含 fork 的应用程序
若发现采集的应用程序的子进程跟踪缺失，可能由于应用程序调用了`fork`函数，且`fork`后没有执行`exec`启动其他应用程序，asys 默认在`fork`后`exec`前不采集跟踪数据，若希望采集`fork`后子进程的跟踪数据，需要通过`--trace-fork-before-exec`使能此场景的跟踪采集：

```bash
asys profile --trace-fork-before-exec true cuda_test
```

## 7. 没有采集到设备内存使用信息
应用程序可能仅在初始化阶段申请设备内存，后续运行过程中无内存申请、释放动作，请检查 asys 选项：

+ 若存在`-c`或`--capture-range`选项，或存在`-y`或`--delay`选项，尝试删除以抓取启动阶段跟踪数据
+ 若使用`asys start`或者`asys attach`方式在应用运行过程中采集，尝试更换为`asys profile`方式采集包含应用程序初始化阶段的跟踪数据

## 8. 安装 SSH 服务
Ubuntu 操作系统安装 SSH 服务步骤参考如下：

```bash
# install openssh service
sudo apt-get update
sudo apt-get install openssh-server

# edit ssh service config to modify SSH port / environment / authentication
# vi /etc/ssh/sshd_config

# restart ssh service
sudo service ssh restart
```



Centos 操作系统安装 SSH 服务步骤参考如下：

```bash
# install openssh service
sudo yum install openssh-server

# edit ssh service config to modify SSH port / environment / authentication
# vi /etc/ssh/sshd_config

# restart ssh service
sudo systemctl restart sshd.service
```

使用 Docker 场景需配置转发 Docker 内 SSH 端口到主机端口，以 SSH 使用 22 端口为例，若转发到主机的 50022 端口，需在 Docker 启动命令中添加如下选项：

```bash
--expose=22 -p 50022:22
```

## 9. 清理 asys 运行环境
若发现 asys 运行异常，请尝试如下步骤清理 asys 运行环境：

+ 运行`asys sessions list`检查是否有长时间未结束的 session，可通过`asys shutdown --session xxx`命令停止相关 session
+ 运行`ps -aux | grep traced | grep -v grep |  awk '{print $2}' | xargs kill`停止 asys 相关后台服务后，再次尝试运行 asys
+ 重新安装 T-Head SAIL SDK

## 10. asys 采集完成应用未停止
在指定`--duration`或者`--capture-range`选项时，在采集停止时，asys 默认发送`SIGTERM`停止应用，若应用程序中包含`SIGTERM`的处理且未及时停止，可能导致应用程序 asys 采集结束后无法停止，请尝试：

+ 指定选项`--kill 9`，asys 将发送不可拦截的信号`SIGKILL`停止应用

## 11. 采集 Tensorflow Eager 模式应用跟踪不完整
若发现 asys 采集 Tensorflow Eager 模式应用的跟踪数据不完整，可添加选项` --flush-on-context-synchronize true`，使能 asys 在每次 HGGC context / stream 同步时保存数据，配置后 asys 采集性能开销将会增大：

```bash
asys profile --flush-on-context-synchronize true cuda_test
```

## 12. Linux 上开启 Timeline View 的 Alt 平移快捷键
在 Linux 平台上，如果 Alt + 滚轮被全局缩放功能占用，可以通过以下办法关闭全局缩放，释放 Alt 键：

在终端中，执行：

```bash
xfconf-query -c xfwm4 -p /general/zoom_desktop -s false
```

可以用`xfconf-query -c xfwm4 -p /general/zoom_desktop`来检查其值是否为 false

## 13. 采集 Heap 内存使用量和进程内存占用量不同
asys 支持通过如下选项采集进程在 CPU 侧的内存使用：

+ `--heap-memory-usage`：采集进程的`动态分配内存使用量`，比如通过`malloc` / `new`等方式动态分配的内存
+ `--host-memory-sampling`：采样进程的`实际物理内存使用量`，比如通过 Linux 的`top`命令查看进程的`RES`列内存占用大小

asys 采集到的进程的`动态分配内存使用量`和`实际物理内存使用量`受到如下因素影响，内存使用量可能有较大差异：

+ 受到 C 库内存缓存优化机制以及内存碎片化的影响，内存可能被进程持有而不是被操作系统回收，导致`实际物理内存使用量`显著高于`动态分配内存使用量`
+ 受到操作系统内存分页机制和惰性内存分配策略影响，通过`malloc`等方式分配但未访问的内存不会分配实际物理内存，导致`实际物理内存使用量`显著低于`动态分配内存使用量`

## 14. 没有采集到使用 Ray 框架应用的跟踪数据
asys 的默认采集模式不支持采集 Ray 框架应用的跟踪数据，需要通过`--multi-node-mode ray`选项使能多节点跟踪采集模式，每个节点生成一份报告文件：

```bash
asys profile --multi-node-mode ray -f true -o demo_vllm bash eval_llm_infer.sh
```

## 15. CUDA GPU 环境使用 asys 采集跟踪
asys 支持在 CUDA GPU 环境采集跟踪，使用方式和在 PPU 环境采集跟踪相同，大部分 asys 特性均可在 CUDA GPU 环境生效，例如：

+ 支持映射 CUDA graph 创建阶段 NVTX 到 CUDA graph 执行阶段
+ 支持采集 CUDA graph 节点在 GPU 上执行信息，并汇总显示 CUDA graph 整体生命周期
+ 支持 python function trace 采集所有 python 函数的执行信息
+ 支持 attach 方式采集跟踪数据

在 CUDA GPU 环境中，可以单独安装 asight 工具，或者使用 T-Head SAIL SDK 中包含的 asight 工具，单独安装 asight 工具的方法为：

1. 前往 PPU artifactory 页面下载 asight 安装包
2. 解压 asight 安装包，以`asight_ubuntu2404.tar.gz`举例，执行`tar -xf asight_ubuntu2404.tar.gz`
3. 进入解压生成的`asight`文件夹，执行`source envsetup.sh`，即可开始使用 asys 工具

使用 SDK 中的 asight 工具的方法为：

1. 进入 SDK 下`asight`目录，执行`source envsetup.sh`，即可开始使用 asys 工具
2. 注意通过`source`执行的并非`SDK`文件夹根目录下的`envsetup.sh`，请注意分辨

可分别采集 GPU 和 PPU 环境应用的 asysrep 报告，通过`asys compare`输出比较结果，例如：

```bash
asys compare gpu_report.asysrep ppu_report.asysrep
```

!!! note "注意"
    在采集 asysrep 报告前，建议使能 PPU 算子 HGTX range 标注相关功能：

    1. 执行`export PPU_LIB_PERF_INSTRUMENT=1`配置环境变量，使能基础框架 PPU 算子 HGTX range 标注功能
    2. asys 添加选项`--pytorch autograd-shapes-hgtx`，使能 pytorch 算子 HGTX range 标注功能


在 CUDA GPU 环境不支持的特性列举如下：

+ video 跟踪
+ pccl 跟踪
+ PPU metrics 采样
+ HGGC memory usage 跟踪
+ CUDA event 参数采集和依赖关系查询

