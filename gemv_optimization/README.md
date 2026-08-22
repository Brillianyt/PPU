# GEMV Decode 算子优化(§36 最终方案)

本目录包含 Qwen3.5-2B INT8 量化 + PPU 原生 GEMV decode 算子 + prefill 反量化的全部源码修改。

## 目录结构(按原仓库路径保留)

```
gemv_optimization/
├── DeepGEMM-for-sail/
│   └── deep_gemm/
│       ├── __init__.py                        # 导出 gemv_int8_int8_bf16_nt
│       └── jit_kernels/
│           ├── __init__.py                    # 导出
│           └── gemv_int8.py                   # ★ 新增:GEMV JIT 层,复用 Gemvt
└── sglang-for-sail/
    └── python/sglang/srt/
        ├── environ.py                         # +3 环境变量
        ├── hardware_backend/
        │   └── ppu/attention/
        │       └── ppu_fa3_hooks.py           # ViT CG: vision.flash_attn_varlen_func → PPU 原生 FA3
        ├── models/
        │   └── qwen3_vl.py                    # ViT CG: Qwen3-VL 补 max_context_len
        └── layers/
            ├── quantization/
            │   └── w8a8_int8.py               # GEMV custom op + apply 分派 + prefill 反量化
            └── deep_gemm_wrapper/
                ├── compile_utils.py           # GEMV 枚举 + warmup executor
                └── entrypoint.py              # gemv_nt_i8i8bf16 wrapper
```

## 部署方式

不改 C++ 扩展,纯 Python 覆盖即可:

```bash
# DeepGEMM 3 个文件 → site-packages/deep_gemm/
# SGLang 4 个文件 → site-packages/sglang/srt/
```

## 依赖的内核

`gemv_int8.py` 复用 DeepGEMM 仓库已存在的 `include/deep_gemm/gemvt.cuh`(泛型 GEMV,支持 int8 + scale),无需额外编译。

## 完整设计文档

见 `../docs/10_GEMV_DECODE_FINAL.md`

## 附：ViT CUDA Graph 正解化（新增，与 GEMV 正交）

本目录同时归档 ViT CUDA Graph 正解化的两处 sglang 源码修改，与 GEMV decode
优化相互独立、可叠加：

- `sglang-for-sail/python/sglang/srt/hardware_backend/ppu/attention/ppu_fa3_hooks.py`
  （P1）：新增 `@plugin_hook`，把 `vision.flash_attn_varlen_func` 接到 PPU 原生
  `flash_attn_interface`，使 ViT graph 的 `fa3` backend 真正走 PPU 原生 FA3。
- `sglang-for-sail/python/sglang/srt/models/qwen3_vl.py`
  （P2）：`Qwen3VLMoeVisionModel` 补 `max_context_len`，避免 rotary 共享 workspace
  懒扩展导致已 capture 的 graph 地址失效（精度 −15pp 的根因）。

完整根因、验证数据与部署说明见 `../vit_graph_optimization.md`。
