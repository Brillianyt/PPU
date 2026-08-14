# Asight Compute 常见问题

## 1. 减少测试环境差异

可通过锁定设备频率，减少设备调频对 acu 采样的影响：

```bash
# 锁定设备频率到1.5GHz
ppu-smi -lpc 1500

# 解除频率锁定
ppu-smi -rpc
```

## 2. 目标应用已经结束，但 acu 没有收到目标应用退出消息而卡死
有些目标应用程序在启动后，会 fork 出很多的子进程，在目标程序主进程退出后，还有子进程一直存在不退出，acu 此时就会一直等待子进程，看到的现象会误以为 acu 卡死。

**确认此场景方法：**

1. 安装 pstree 工具
2. 在 acu 出现卡死现象后，在另一个终端中输入命令：`pstree <acu_pid>`，确认是否有目标应用 fork 出的进程出现在了此命令的列表中。如果有，就说明有 fork 出的子进程不退出。

**规避方法：**

在 acu 命令中使用 `--wait primary` 参数，让 acu 仅等待目标应用的主进程退出。

<a    id="J5Ul3"></a>

## 3. 报错：Device is not ready for profiling
这个报错大概率是 PPU 性能数据采集资源 PCM 被其他应用占用，导致 acu 无法采集。可能是其它 asys 或者 acu 应用正在采集，也可能是其它采集 PPU 运行指标的监控程序正在后台运行。

1. 请运行 `sudo lsof /dev/alixpu` 和`dmesg --level=err,crit,alert,emerg`命令，查询哪个应用在占用 PPU PCM
2. 请运行` ppu-smi`，查询当前环境的 Driver Version（KMD 版本）
    1. 如果 KMD 版本低于 1.4.0，需要手动停止步骤 1 查询到的 PCM 占用程序，重新运行 acu
    2. 如果 KMD 版本大于或者等于 1.4.0，可以用 PPU-SMI 临时关闭其它应用的采集功能。请参考下面命令，更详细说明请参考[设置性能监控输出状态](/ppu-smi/user-guide.md#gpm_state)

```bash
#查询性能监控GPM stream状态，如果PCM被占用，acu无法采集时，期望返回ENABLED
ppu-smi gpm -g

#临时禁止GPM stream功能，让acu可正常采集
ppu-smi gpm -s DISABLED

#查询性能监控采集状态，期望返回DISABLED
ppu-smi gpm -g

#此时acu应该可以正常使用

#acu采集完成后，恢复GPM stream状态
ppu-smi gpm -s ENABLED

#查询性能监控采集状态，期望返回ENABLED
ppu-smi gpm -g
```

## 4. LLC 向 DRAM 写入数据偏少
如下图，memory D2H 的行为中 L2 向 LLC 写入了 64M 数据，但 LLC 仅向 DRAM 写入了 45.88M 数据。

造成这种数据不匹配的情况，是因为驱动层在将数据写入 LLC 后，就认为已经到 DRAM 了，随即结束 kernel 的执行。kernel 执行结束后，counter 的采集也会停止。但 LLC 向 DRAM 写入数据不受外部的控制，时机不确定，所以 LLC 向 DRAM 写入数据的 counter 会少一部分记录。

![](https://thead-marketing-public.oss-cn-hangzhou.aliyuncs.com/1784125009747/9ca52604758d6ea54af947ae00629d4b/llcdram_1.png)



