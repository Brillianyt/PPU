# DeepGEMM 稠密 INT8 GEMV(decode, M <= 8)
# 复用 DeepGEMM 现成的泛型 GEMV kernel `gemvt.cuh` 的 `Gemvt<int8_t, ...>`。
# 它与 MoE grouped GEMV 用同一内核,但 NUM_GROUPS=1、grouped_layout 全 0,
# 数值语义与 gemm_int8_int8_bf16_nt 完全一致。

import torch
from typing import Tuple

from .tuner import jit_tuner
from .gemm import get_gemv_best_configs
from .utils import get_num_sms

includes_gemv = ('"deep_gemm/gemvt.cuh"', )

template_gemv = """
using namespace deep_gemm;

constexpr auto N = {N}, K = {K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto ThreadPerN = {ThreadPerN};
constexpr auto NPerThread = {NPerThread};
constexpr auto NUM_UNROLL = {NUM_UNROLL};
constexpr auto SWZL_SIZE_M = {SWZL_SIZE_M};
constexpr auto USE_SMALL_K = {USE_SMALL_K};
constexpr auto BlockSize = {BlockSize};

using gemm_v = Gemvt<int8_t, __nv_bfloat16, {acc_type}, N, K, kNumGroups,
                     ThreadPerN, NPerThread, NUM_UNROLL, SWZL_SIZE_M, BlockSize, USE_SMALL_K>;

gemm_v::run(out, grouped_layout,
            m, lhs, rhs,
            stream,
            lhs_scales, rhs_scales);
"""

arg_defs = (
    ('lhs', torch.int8),          # [M, K]
    ('rhs', torch.int8),          # [N, K] 行主序
    ('out', torch.bfloat16),      # [M, N]
    ('grouped_layout', torch.int32),  # [M], dense 全 0
    ('m', int),
    ('stream', torch.cuda.Stream),
    ('lhs_scales', torch.float),  # [M]
    ('rhs_scales', torch.float),  # [N]
)


def gemv_int8_int8_bf16_nt(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
) -> None:
    """稠密 INT8 GEMV(M=1..8 特化),复用 Gemvt 内核。

    lhs=(x_q[M,K] int8, x_scale[M] fp32)
    rhs=(w_q[N,K] int8 行主序, w_scale[N] fp32)
    out=[M,N] bf16
    """
    x, x_scale = lhs
    w, w_scale = rhs
    m, k = x.shape
    n, k_ = w.shape

    assert x.dtype == torch.int8 and w.dtype == torch.int8
    assert x_scale.dtype == torch.float32 and w_scale.dtype == torch.float32
    assert k == k_ and out.shape == (m, n)
    assert out.dtype == torch.bfloat16
    assert x.is_contiguous() and w.is_contiguous()
    assert x_scale.is_contiguous() and w_scale.is_contiguous()
    assert out.is_contiguous()
    assert m <= 16, "gemv_int8_int8_bf16_nt 仅用于小 M(decode)"

    if m == 0 or n == 0:
        return

    num_groups = 1
    num_sms = get_num_sms()
    block_size, thread_per_n, num_unroll, swzl_size_m, n_per_thread, use_small_k = \
        get_gemv_best_configs(m, n, k, num_groups, num_sms, torch.int8)

    if thread_per_n == -1:
        # Gemvt 不支持该形状,回退 GEMM 由上层处理;这里直接报错提示
        raise RuntimeError(f"Gemvt does not support shape M={m}, N={n}, K={k}")

    grouped_layout = torch.zeros(m, dtype=torch.int32, device=x.device)

    args = (x, w, out, grouped_layout, m,
            torch.cuda.current_stream(), x_scale, w_scale)

    runtime = jit_tuner.compile_and_tune(
        name='gemv_int8_int8_bf16_nt',
        keys={
            'N': n, 'K': k, 'NUM_GROUPS': num_groups,
            'ThreadPerN': thread_per_n, 'NUM_UNROLL': num_unroll,
            'SWZL_SIZE_M': swzl_size_m, 'NPerThread': n_per_thread,
            'BlockSize': block_size, 'USE_SMALL_K': use_small_k,
            'acc_type': 'int',
        },
        space=(),
        includes=includes_gemv,
        arg_defs=arg_defs,
        template=template_gemv,
        args=args,
    )
    runtime(*args)
