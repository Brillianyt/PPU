# v1.2 关键决策与对话记录
> 从 Claude Code 对话 JSONL 自动提取，保留关键的技术决策、洞察、命令。
## 决策 §38 (2026-08-20):ACE XT vs GEMV in CG

**触发**:用户对"TTFT 退化"的担忧,要求"按 G 组相同配置, 复测 Qwen_origin"。

**实测数据**(同台机器, 同 SGLang 0.5.13, 同 evaluation_wrapper.py 客户端):

| 配置 | TTFT | throughput | accuracy |
|---|---|---|---|
| G 组 baseline (BF16) | 86.94 ms | 279.8 tok/s | ~0.85 |
| GEMV in CG (INT8) | 84.16 ms | 418.5 tok/s | 0.85 |
| **ACEXT=1 (INT8)** | **82.31 ms** | **423.2 tok/s** | **0.85** |

**结论**:
- 之前所有"TTFT 退化 30%"的担忧**有误**:历史 baseline 64.7ms 是 2026-08-14 测的(同日机器状态更优);真实 G 组 baseline 在 2026-08-20 同台机器上是 **86.94 ms**。
- 在真实对照下,我们方案 **TTFT -5.3%、吞吐 +51.2%**。
- ACEXT 略好于 GEMV in CG(TTFT -2.2%,吞吐 +1.1%),但与"GEMV 融入 CG"目标互斥。

**决定**:
1. **ACEXT 锁定为 PPU 实测最佳**(`SGLANG_SAIL_USE_ACEXT_CUDA=1` 为默认)
2. GEMV in CG 保留为"路线 A"(用于论文/科研发布),`SGLANG_SAIL_USE_ACEXT_CUDA=0` 切换
3. README/EXPERIMENT.md 全部以 ACEXT 为最佳基线记录
4. 真实 G 组 baseline 86.94ms 是后续优化的对照基线
