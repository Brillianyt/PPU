# Asight Systems 安装指南

## 1. 获取 asys 命令行工具

<a id="ZFzMg"></a>
### 1.1. 配置环境变量

asys 命令行工具包含在 T-Head SAIL SDK 中，获取 SDK 请参见[SDK 使用指南](https://developer.t-head.cn/docs_center/doc_detail/index.html?projectId=39&chapterId=194)。安装完成后，进入 SDK 目录，执行以下命令配置所需环境变量：

```bash
source envsetup.sh
```

### 1.2. 检查运行环境

环境变量配置完成后，执行以下命令查看当前安装的 asys 版本信息：

```bash
asys -v
```

执行以下命令检查当前环境是否满足 asys 跟踪采样要求：

```text
$ asys status

[Asight]: Profiling Environment Check
[Asight]: Root privilege: enabled: OK
[Asight]: Linux Kernel Paranoid Level = 2
[Asight]: Linux Distribution = Ubuntu: OK
[Asight]: Linux Kernel Version = 5.10.134-13.al8.x86_64: OK
[Asight]: Linux perf_event_open syscall available: OK
[Asight]: Linux Ftrace Support: available: OK
[Asight]: Glibc version requires 2.39 and above, Current version 2.39: OK 
[Asight]: SDK Installed: OK(0.0.0-000000)
[Asight]: CPU Profiling Environment: OK
[Asight]: Profiling Environment Check: OK
```

若命令显示采样环境检查通过，即可使用 asys 采集应用跟踪数据。

!!! note "注意"
    部分配置的检查失败不影响 asys 命令行工具的正常使用，但会影响部分功能：

    + `root` 权限检查失败将无法进行 CPU 采样及其调用栈采集。
    + `perf_event_open` syscall 检查失败将无法使用调用栈采集。如果使用 Docker，可在 `docker run` 命令中增加参数 `--pid=host --privileged=true` 来开启此功能。
    + `Ftrace` 检查失败将无法使用 CPU 采样。如果使用 Docker，可在 `docker run` 命令中增加参数 `--privileged=true -v /sys/kernel/debug:/sys/kernel/debug --pid=host` 来挂载 Ftrace 目录，同时需确认宿主机上的 `/sys/kernel/debug` 目录不为空。

## 2. 获取 Asight Systems GUI 工具

Asight Systems GUI 工具安装包单独发布，支持以下操作系统：

+ Windows 10 / Windows 11
+ macOS 10.15 及以上版本

请前往[下载页面](/download/index.html?type=TOOLS)获取安装包，根据操作系统选择对应的格式：

+ Windows 系统请选择 `.msi` 安装包
+ macOS 系统请选择 `.dmg` 安装包
