from __future__ import annotations

import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, cast

import torch
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper

# ISA Phase 3.2 fast path (K=2048, M<=8):
# Imports the prebuilt sglang_kernel_v10 wheel (.so built at install time,
# not JIT-loaded at first call). This is the proper integration pattern —
# see /mnt/workspace/ISA/src/sglang_kernel_v10/README.md.
#
# Why not torch.utils.cpp_extension.load: it triggers a compile on first
# call, which both adds TTFT latency AND breaks CUDA graph capture (the
# capture happens on the very first forward pass, while JIT is still
# compiling). The prebuilt .so is loaded into the module table at import
# time, before any forward / capture.
try:
    import sglang_kernel_v10 as _isa_v10
    _ISA_V10_AVAILABLE = True
    # Lazy-register v10 ops on sglang's torch.ops.sglang namespace so
    # inductor / ppu_fused_kernel can see them. Safe no-op if sglang
    # isn't imported yet (e.g. w8a8_int8.py imported before sglang).
    try:
        from sglang_kernel_v10.torch_ops import register_sglang_alias
        register_sglang_alias()
    except Exception:
        pass
except ImportError:
    _isa_v10 = None
    _ISA_V10_AVAILABLE = False

# ABLATION A: temporarily disable v10 fast path to isolate regression source.
# Default OFF (2026-08-23 决策: 净评分接近 0, TTFT 回归抵消 TPS 收益, 详见
# PPU/docs/02_RESULTS.md §2.1 + README.md)。
# 设 True 启用 v10 fast path;False 走 deep_gemm baseline。
_ISA_V10_ENABLE = False

# Per-data_ptr cache of (N, K) row-major weight buffers for the v10 fast path.
#
# IMPORTANT — layout convention (was wrong in initial integration, model
# produced garbage):
#   sglang stores layer.weight as (K, N) row-major (transposed at
#   process_weights_after_loading). Then apply() calls
#       gemv_int8_i8i8bf16(x_q, x_scale, layer.weight.t(), ...)
#   so the dispatcher receives B = layer.weight.t() which has SHAPE (N, K)
#   but COLUMN-MAJOR strides (the underlying memory is still (K, N)
#   row-major). Our kernel reads W as (N, K) row-major for coalesced access.
#   So we cache B.contiguous() (which copies the (N, K) view into row-major).
#   DO NOT call B.t().contiguous() — that gives (K, N) row-major, which the
#   kernel would read as if it were (N, K) → garbage output.
_ISA_V10_W_CACHE: dict[int, torch.Tensor] = {}


def _isa_v10_get_w_nk(B: torch.Tensor) -> torch.Tensor:
    """Return a (N, K) row-major copy of an (N, K)-shape, column-major view.

    B is `layer.weight.t()` — shape (N, K), strides (1, K). The kernel
    needs (N, K) row-major. We cache the contiguous() copy keyed by
    B.data_ptr() (which equals layer.weight.data_ptr() since .t() doesn't
    reallocate), so the copy happens ONCE per layer.
    """
    # layer.weight.data_ptr() == B.data_ptr() because .t() is a view, not a copy.
    # Cache on that so we don't keep a separate map.
    key = B.data_ptr()
    cached = _ISA_V10_W_CACHE.get(key)
    if cached is not None:
        return cached
    cached = B.contiguous()  # (N, K) row-major — what the kernel wants
    _ISA_V10_W_CACHE[key] = cached
    return cached
from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_host_cpu_arm64,
    is_ppu,
    set_weight_attrs,
    use_intel_amx_backend,
)
from sglang.srt.utils.custom_op import register_custom_op
from sglang.srt.utils.patch_torch import register_fake_if_exists

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput

import logging

logger = logging.getLogger(__name__)
ACTIVATION_SCHEMES = ["static", "dynamic"]

_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_cpu_arm64 = is_host_cpu_arm64()
_is_ppu = is_ppu()

if _is_cuda:
    from sgl_kernel import int8_scaled_mm

    @register_fake_if_exists("sgl_kernel::int8_scaled_mm")
    def _int8_scaled_mm_abstract(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias=None,
    ):
        M = mat_a.shape[-2]
        N = mat_b.shape[-1]
        return mat_a.new_empty((M, N), dtype=out_dtype)


if _is_ppu:
    from acext import int8_gemm as acext_int8_gemm

logger = logging.getLogger(__name__)


def gemm_nt_i8i8bf16_fake(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
) -> None:
    return


@register_custom_op(mutates_args=["C"], fake_impl=gemm_nt_i8i8bf16_fake)
def gemm_nt_i8i8bf16(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
) -> None:
    deep_gemm_wrapper.gemm_nt_i8i8bf16((A, As), (B, Bs), C)


def gemv_int8_i8i8bf16_fake(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
) -> None:
    return


@register_custom_op(mutates_args=["C"], fake_impl=gemv_int8_i8i8bf16_fake)
def gemv_int8_i8i8bf16(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
) -> None:
    # ISA fast path: K=2048, M==1 → our v10 kernel (3-4.5× vs deep_gemm).
    # The fast-path kernel is a prebuilt .so (sglang_kernel_v10 wheel),
    # NOT JIT-loaded — so it can be safely used inside CUDA graph capture.
    #
    # B is in sglang's native (K, N) row-major layout (because layer.weight
    # is transposed at process_weights_after_loading). Our kernel wants
    # (N, K) row-major for coalesced access. We cache the transposed buffer
    # by B.data_ptr() so the (16MB for hidden=2048) memcpy happens ONCE per
    # layer, NOT on every decode step.
    #
    # Hot-path optimization: cache is pre-populated in
    # process_weights_after_loading, so the dict lookup hits 100% of the time.
    # We still do the dict lookup (microseconds) for safety against missing
    # entries; the common case is one dict.get + one registered op call.
    #
    # IMPORTANT (Phase 3.2 v10 + TTFT fix — this is the v10_TTFTFIX baseline,
    # TPS 417.4). Direct pybind call into the prebuilt v10 .so:
    #   - No registered-op wrapper overhead (Stream A regressed -4.5%)
    #   - No PpuKernelRegistry lookup (Stream C regressed -2-3%)
    #   - Mutating form (writes into caller-provided C) avoids the copy_
    #   that the functional alias requires.
    # The previous Stream A attempt (torch.ops.sglang.v10_gemv_int8_async +
    # C.copy_(out)) was reverted on 2026-08-23 after a 100-sample e2e
    # benchmark confirmed TPS 417 → 398 (-4.5%) regression. The functional
    # op + copy_ is too expensive on mmbench's many short completions
    # (each small completion pays the extra alloc+copy).
    if _ISA_V10_ENABLE and A.shape[1] == 2048 and A.shape[0] == 1:
        key = B.data_ptr()
        W_nk = _ISA_V10_W_CACHE.get(key)
        if W_nk is None:
            # Cache miss (e.g. layer was re-quantized at runtime): rebuild
            W_nk = B.contiguous()
            _ISA_V10_W_CACHE[key] = W_nk
        # Mutating pybind call — writes bf16 directly into C.
        _isa_v10.gemv_int8_async(A, W_nk, As, Bs, C)
        return
    deep_gemm_wrapper.gemv_nt_i8i8bf16((A, As), (B, Bs), C)


def gemv_int8_fused_qkv_fake(
    A: torch.Tensor,
    Wq: torch.Tensor,
    Wk: torch.Tensor,
    Wv: torch.Tensor,
    As: torch.Tensor,
    Bs_q: torch.Tensor,
    Bs_k: torch.Tensor,
    Bs_v: torch.Tensor,
    out_q: torch.Tensor,
    out_k: torch.Tensor,
    out_v: torch.Tensor,
) -> None:
    # Inplace op (writes to out_q/out_k/out_v) — fake returns None for shape
    # propagation. Shapes of the mutating args are checked at dispatch time.
    return


@register_custom_op(
    mutates_args=["out_q", "out_k", "out_v"],
    fake_impl=gemv_int8_fused_qkv_fake,
)
def gemv_int8_fused_qkv(
    A: torch.Tensor,
    Wq: torch.Tensor,
    Wk: torch.Tensor,
    Wv: torch.Tensor,
    As: torch.Tensor,
    Bs_q: torch.Tensor,
    Bs_k: torch.Tensor,
    Bs_v: torch.Tensor,
    out_q: torch.Tensor,
    out_k: torch.Tensor,
    out_v: torch.Tensor,
) -> None:
    """Fused Q+K+V INT8 GEMV (Phase 3.4).

    Computes out_q = A @ Wq, out_k = A @ Wk, out_v = A @ Wv in one kernel
    launch. The v10 ``fused_qkv`` kernel shares the A load + write
    bandwidth across Q/K/V — microbench gives **1.40× speedup** vs 3
    separate v10 calls at N=2048 (self-attn Q/K/V) with identical outputs
    (max diff = 0 vs fp32 ref and inside CUDA graph capture).

    Shape contract:
      - A:      (1, 2048) int8
      - Wq/Wk/Wv: each (N, 2048) int8 row-major, **same N** (MVP constraint)
      - As:     (1,)       fp32 (per-token activation scale)
      - Bs_*:   (N,)       fp32 (per-channel weight scales, one per projection)
      - out_*:  (1, N)     bf16 contiguous (written in-place)

    Trigger conditions (all must hold):
      1. v10 kernel wheel is installed AND enabled
      2. A.shape == (1, 2048)
      3. Wq.shape == Wk.shape == Wv.shape   (symmetric N — kernel MVP)

    Fallback (any condition fails): three separate ``gemv_int8_i8i8bf16``
    calls. This reuses the existing v10 single-GEMV fast path + cache, so
    no functionality is silently broken; we just lose the 1.40× fused speedup.

    Note on Qwen3.5-2B self-attn: the QKVParallelLinear layout has
    N_q = 4096 (8 heads × head_dim=256, doubled for attn_output_gate) but
    N_k = N_v = 512 (2 kv heads × head_dim=256) — asymmetric, so condition 3
    fails and the dispatcher falls back. The dispatcher is registered and
    ready for future symmetric-N models (or after relaxing the MVP constraint
    in the kernel).
    """
    if (
        _ISA_V10_ENABLE
        and _ISA_V10_AVAILABLE
        and A.shape[1] == 2048
        and A.shape[0] == 1
        and Wq.shape == Wk.shape == Wv.shape
    ):
        # Convert sglang's (N, K)-shape column-major views into (N, K)
        # row-major copies. _isa_v10_get_w_nk caches by data_ptr() so the
        # 16MB memcpy happens at most once per weight tensor.
        W_nk_q = _isa_v10_get_w_nk(Wq)
        W_nk_k = _isa_v10_get_w_nk(Wk)
        W_nk_v = _isa_v10_get_w_nk(Wv)
        _isa_v10.fused_qkv(
            A,
            W_nk_q, W_nk_k, W_nk_v,
            As,
            Bs_q, Bs_k, Bs_v,
            out_q, out_k, out_v,
        )
        return

    # Fallback: 3 separate v10 GEMVs (each goes through the existing
    # gemv_int8_i8i8bf16 dispatcher — reuses the cache + v10 fast path).
    gemv_int8_i8i8bf16(A, As, Wq, Bs_q, out_q)
    gemv_int8_i8i8bf16(A, As, Wk, Bs_k, out_k)
    gemv_int8_i8i8bf16(A, As, Wv, Bs_v, out_v)


# ---------------------------------------------------------------------------
# Functional (non-mutating) variants — inductor-fusable.
# ---------------------------------------------------------------------------
#
# The mutating API above is kept for backward compatibility and for call sites
# that pre-allocate ``out`` and reuse the same buffer across iterations.
# The functional variants below are registered via the PPU-kernels framework
# so that torch.compile / inductor treats them as a known extern kernel and
# can fuse a Triton prologue / epilogue around them (see
# :mod:`sglang.srt.compilation.ppu_kernels`).
#
# During CUDA graph capture the ``torch.empty`` allocation is amortized
# (sglang reuses the same output buffer across iterations), so the extra
# allocation cost is negligible in practice.

from sglang.srt.compilation.ppu_kernels import ppu_fused_kernel  # noqa: E402


@ppu_fused_kernel(
    name="gemv_int8_i8i8bf16_nm",
    output=("C", torch.bfloat16),
    output_shape_from="A",
    cpp_symbol="sglang::gemv_int8_i8i8bf16",
    epilogue_fusion=("bias_add",),
)
def gemv_int8_i8i8bf16_nm(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
) -> torch.Tensor:
    """Functional INT8 GEMV. Returns a new bf16 tensor of shape ``(A.shape[0], B.shape[0])``."""
    C = torch.empty(
        (A.shape[0], B.shape[0]),
        dtype=torch.bfloat16,
        device=A.device,
    )
    deep_gemm_wrapper.gemv_nt_i8i8bf16((A, As), (B, Bs), C)
    return C


@ppu_fused_kernel(
    name="gemm_nt_i8i8bf16_nm",
    output=("C", torch.bfloat16),
    output_shape_from="A",
    cpp_symbol="sglang::gemm_nt_i8i8bf16",
    epilogue_fusion=("bias_add",),
)
def gemm_nt_i8i8bf16_nm(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
) -> torch.Tensor:
    """Functional INT8 GEMM. Returns a new bf16 tensor of shape ``(A.shape[0], B.shape[0])``."""
    C = torch.empty(
        (A.shape[0], B.shape[0]),
        dtype=torch.bfloat16,
        device=A.device,
    )
    deep_gemm_wrapper.gemm_nt_i8i8bf16((A, As), (B, Bs), C)
    return C


class W8A8Int8Config(QuantizationConfig):
    """Config class for W8A8 Quantization.

    - Weight: static, per-channel, symmetric
    - Activation: dynamic, per-token, symmetric
    """

    def __init__(self, quant_config: Dict[str, Any] = {}):
        super().__init__()
        self.quant_description = quant_config
        self.is_dynamic = quant_config.get("is_dynamic", False)
        ignore = cast(List[str], quant_config.get("ignore", []))
        self.ignore = ignore if ignore is not None else []
        packed_modules_mapping = quant_config.get("packed_modules_mapping", {})
        self.packed_modules_mapping = (
            packed_modules_mapping if packed_modules_mapping is not None else {}
        )

        self.mix_layer = quant_config.get("mix_layer", None)

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_name(self) -> str:
        return "w8a8_int8"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        filenames = []
        return filenames

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W8A8Int8Config:
        return cls(config)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if should_ignore_layer(
                prefix, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
            ):
                return UnquantizedLinearMethod()
            return W8A8Int8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return W8A8Int8MoEMethod(self)
        return None

    def is_layer_skipped(
        self, prefix: str, fused_mapping: Mapping[str, List[str]] = MappingProxyType({})
    ):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = (
                    self.quant_description[shard_prefix + ".weight"] == "FLOAT"
                )

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision."
                    )
        else:
            is_skipped = self.quant_description[prefix + ".weight"] == "FLOAT"

        assert is_skipped is not None
        return is_skipped

    def get_scaled_act_names(self) -> List[str]:
        return []


class W8A8Int8LinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: W8A8Int8Config):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu:
            if _is_cpu_amx_available:
                _amx_process_weight_after_loading(layer, ["weight"])
            elif _is_cpu_arm64:
                layer.weight = Parameter(layer.weight.data, requires_grad=False)
            else:
                assert False, "W8A8Int8LinearMethod on CPU only works on AMX or Arm64"
        else:
            layer.weight = Parameter(layer.weight.t(), requires_grad=False)
            # ISA v10 fast path: pre-populate the (N, K) row-major cache for this
            # layer so the first decode step doesn't pay a 16MB memcpy per layer
            # (was the main cause of TTFT regression in initial integration).
            # The cache key is layer.weight.data_ptr() (== dispatcher B.data_ptr()
            # because .t() is a view, not a copy).
            if _ISA_V10_AVAILABLE and _ISA_V10_ENABLE:
                W_nk = layer.weight.t().contiguous()  # (N, K) row-major
                _ISA_V10_W_CACHE[layer.weight.data_ptr()] = W_nk
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        if use_intel_amx_backend(layer) or _is_cpu_arm64:
            return torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
                x,
                layer.weight,
                layer.weight_scale,
                bias,
                x.dtype,
                True,  # is_vnni
            )

        # 方向 B: prefill(M 大)可选走 BF16 反量化 GEMM,
        # 避开 int8 GEMM 在 prefill 比 BF16 GEMM 慢的问题。
        # 权重仍以 int8 存储,反量化结果懒缓存(推理期权重不变)。
        if (
            envs.SGLANG_SAIL_PREFILL_BF16.get()
            and _is_ppu
            and bias is None
            and x.shape[0] > envs.SGLANG_SAIL_GEMV_M_MAX.get()
        ):
            w_bf16 = getattr(layer, "_w_bf16_cache", None)
            if w_bf16 is None:
                w_bf16 = (
                    layer.weight.t().float() * layer.weight_scale
                ).bfloat16().contiguous()
                layer._w_bf16_cache = w_bf16
            return torch.nn.functional.linear(x, w_bf16)

        x_q, x_scale = per_token_quant_int8(x)

        if envs.SGLANG_SAIL_DEEPGEMM_DENSE.get() and _is_ppu and bias is None:
            out = torch.empty(
                (x_q.shape[0], layer.weight.t().shape[0]),
                dtype=torch.bfloat16,
                device=x_q.device,
            )
            if (
                envs.SGLANG_SAIL_USE_GEMV.get()
                and x_q.shape[0] <= envs.SGLANG_SAIL_GEMV_M_MAX.get()
            ):
                gemv_int8_i8i8bf16(x_q, x_scale, layer.weight.t(), layer.weight_scale, out)
            else:
                gemm_nt_i8i8bf16(x_q, x_scale, layer.weight.t(), layer.weight_scale, out)
            return out.to(x.dtype)

        if envs.SGLANG_SAIL_USE_ACEXT_CUDA.get():
            return acext_int8_gemm(
                x_q, layer.weight.t(), layer.weight_scale, x_scale, bias, x.dtype
            )

        x_q_2d = x_q.view(-1, x_q.shape[-1])
        x_scale_2d = x_scale.view(-1, x_scale.shape[-1])
        output_shape = [*x_q.shape[:-1], layer.weight.shape[1]]

        output = int8_scaled_mm(
            x_q_2d,
            layer.weight,
            x_scale_2d,
            layer.weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )

        return output.view(output_shape)


class W8A8Int8MoEMethod(FusedMoEMethodBase):
    """MoE method for INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: W8A8Int8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        tp_size = get_tensor_model_parallel_world_size()

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])
        else:
            layer.w13_weight = Parameter(layer.w13_weight, requires_grad=False)
            layer.w2_weight = Parameter(layer.w2_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        from sglang.srt.layers.moe.utils import (
            get_moe_a2a_backend,
        )

        self.moe_runner_config = moe_runner_config
        moe_runner_backend = get_moe_runner_backend()
        # only PPU deepep / deepgemm support int8
        if (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and _is_ppu
            and (
                get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake()
            )
        ):
            moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
        elif moe_runner_backend.is_auto():
            if (
                envs.SGLANG_SAIL_DEEPGEMM_MOE.get()
                and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            ):
                moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
            elif envs.SGLANG_SAIL_USE_ACEXT_CUDA.get():
                moe_runner_backend = MoeRunnerBackend.ACEXT
            else:
                moe_runner_backend = MoeRunnerBackend.TRITON
        if moe_runner_backend.is_acext():
            import sglang.srt.layers.moe.moe_runner.acext  # noqa: F401 – triggers @register_fused_func
        if moe_runner_backend.is_deep_gemm():
            import sglang.srt.layers.moe.moe_runner.ppu_deepgemm_moe  # noqa: F401 – triggers @register_fused_func
        if (
            moe_runner_backend.is_acext()
            or moe_runner_backend.is_triton()
            or moe_runner_backend.is_deep_gemm()
        ):
            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
        else:
            # TODO(cwan): refactor other backends
            pass

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_int8_w8a8=True,
            per_channel_quant=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPLLDispatchOutput,
            DeepEPNormalDispatchOutput,
        )

        x = dispatch_output.hidden_states
        if isinstance(dispatch_output, DeepEPLLDispatchOutput) or isinstance(
            dispatch_output, DeepEPNormalDispatchOutput
        ):
            topk_output = (dispatch_output.topk_weights, dispatch_output.topk_ids)
        else:
            topk_output = dispatch_output.topk_output

        if use_intel_amx_backend(layer) or _is_cpu_arm64:
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            topk_ids = topk_ids.int()
            x, topk_weights = apply_topk_weights_cpu(
                self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace See [Note] inplace should be False in fused_experts.
                CPUQuantMethod.INT8_W8A8,
                layer.w13_weight_scale,  # w1_scale
                layer.w2_weight_scale,  # w2_scale
                None,  # w1_zp
                None,  # w2_zp
                None,  # block_size
                None,  # w1 bias
                None,  # w3 bias
                None,  # alpha
                None,  # limit
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)

        if self.runner.runner_backend.is_deep_gemm():
            quant_info = DeepGemmMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                use_int8=True,
                per_channel_quant=True,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                block_shape=None,
            )
        else:
            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                use_int8_w8a8=True,
                per_channel_quant=True,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a13_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )
        return self.runner.run(dispatch_output, quant_info)
