# Asight Systems 已知问题

+ asys 采集过程中通过 Ctrl + C 中断采集，采集结束阶段跟踪可能不会写入报告文件
+ asys 不支持在 Linux Kernel 4.1 以下的操作系统中采集 CPU 调用栈信息
+ asys 采集过程可能导致 HGGC stream synchronize 耗时明显变长
+ 应用程序编译需包含`-pthread`选项，否则可能导致 asys 运行崩溃
+ asys attach 可能导致某些应用卡住或者崩溃
+ asys attach 不支持采集在 attach 之前已经实例化的 legacy HGGC graph 相关跟踪信息
+ asys 采集 RDMA 网卡数据吞吐量低于实际值
+ asys 仅支持在 ftrace 时钟设置为 boottime 时钟的环境采集 CPU 线程切换信息
+ asys 采集使能 PyTorch Profiler 功能的应用时，PyTorch Profiler 将无法采集到数据

---

