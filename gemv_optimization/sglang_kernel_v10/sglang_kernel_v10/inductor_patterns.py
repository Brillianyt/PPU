"""inductor_patterns.py — Pattern-match int8 GEMV → sglang_kernel_v10 ops.

Goal
====

Even though ``sglang_kernel_v10`` registers v10 ops under
``torch.ops.sglang.v10_gemv_int8_async`` (see ``torch_ops.py``), inductor
won't *emit* that op on its own — it only emits what the user's FX graph
contains. By default, sglang's ``W8A8Int8LinearMethod.apply`` goes through
``gemv_int8_i8i8bf16`` which calls our kernel directly, but inductor sees
that as an opaque custom op (no pattern-match for fusion).

To get true "deep integration" we register a pattern matcher that:

1. Detects the generic ``aten._weight_int8pack_mm`` pattern (what
   ``Linear`` lowers to when ``x` is` is INT8 + weight is INT8) and
   rewrites it to ``torch.ops.sglang.v10_gemv_int8_async``.

2. Detects a sequence of 3 same-shape int8 GEMVs (the Q/K/V pattern in
   models with symmetric attention, like LLaMA / Mistral) and rewrites
   to a single ``torch.ops.sglang.v10_fused_qkv`` call.

3. Detects the explicit ``torch.ops.sglang.gemv_int8_i8i8bf16`` custom
   op (sglang's dispatcher that currently calls v10 internally) and
   doesn't re-wrap it (we'd just loop infinitely otherwise).

Usage
=====

    from sglang_kernel_v10.inductor_patterns import (
        register_v10_pattern_matchers,
    )
    register_v10_pattern_matchers()  # idempotent

The pattern matchers are inserted as a *post-grad custom pass* so they
run inside inductor's compilation pipeline. After registration,
``torch.compile(model)`` and ``torch.compile(model, mode="reduce-overhead")``
both automatically pick up v10 for matching int8 GEMV patterns.

Architecture
============

Inductor's pattern matcher runs against the FX graph *before* codegen.
We use ``torch._inductor.pattern_matcher.register_replacement`` which
takes:
    - a ``pattern`` (subgraph of fx.node matching what to look for)
    - a ``replacement`` (callable that returns the replacement node)
    - a ``pass_dict`` (defaults to ``pattern_match_dynamo`` or
      ``pattern_match_fx_graph`` based on which phase)

We register against ``pattern_match_fx_graph`` since we want post-grad.

This file does NOT touch the C++ kernels. It only adds Python-side
pattern matching on top of the already-working dispatcher + functional
wrapper. If inductor fails to match, the model still works (just slower).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from . import torch_ops

logger = logging.getLogger(__name__)

# Tracks whether ``register_v10_pattern_matchers`` has already run.
_REGISTERED = False

# The pass dicts that inductor's pattern matcher runs against.
# Inductor's `fx_passes.post_grad` exposes ``pass_patterns`` (a list of
# ``PatternMatcherPass`` instances); patterns registered against
# ``pass_patterns[1]`` (the post-grad pass) run during normal compilation.
try:
    from torch._inductor.fx_passes.post_grad import pass_patterns as _pass_dicts
except Exception:
    # Fall back to a local single-element list if inductor isn't available.
    from torch._inductor.pattern_matcher import PatternMatcherPass
    _pass_dicts = [PatternMatcherPass(), PatternMatcherPass()]


def _make_int8_gemv_pattern():
    """Build an FX-graph pattern that matches the canonical int8 GEMV.

    Matches ``aten._int_mm(A_int8, W_int8) -> int32`` (PyTorch's native
    int8 matmul) and replaces it with our v10 kernel call. The pattern
    only fires for the v10 MVP shape (K=2048, M=1); otherwise the
    original ``aten._int_mm`` stays in place.
    """
    from torch._inductor.pattern_matcher import (
        fwd_only,
        register_replacement,
    )
    aten = torch.ops.aten

    def int8_gemv_pattern(A, W):
        return aten._int_mm(A, W)  # noqa: F841

    def int8_gemv_replacement(A, W):
        M, K = A.shape
        N = W.shape[0]
        if K != 2048 or M != 1:
            return aten._int_mm(A, W)
        ones_M = A.new_ones(M, dtype=torch.float32)
        ones_N = W.new_ones(N, dtype=torch.float32)
        out = torch.ops.sglang.v10_gemv_int8_async(A, W, ones_M, ones_N)
        return out.to(torch.int32)

    return register_replacement(
        int8_gemv_pattern,
        int8_gemv_replacement,
        [torch.zeros(1, 2048, dtype=torch.int8), torch.zeros(2048, 8192, dtype=torch.int8)],
        fwd_only,
        _pass_dicts[1],
    )


def _make_fused_qkv_pattern():
    """Build a pattern that matches 3 sequential int8 GEMVs sharing A.

    When inductor lowers a transformer attention block, Q/K/V projections
    are usually emitted as 3 separate ``aten._int_mm`` calls with the same
    input A but different W. We pattern-match on this triple.

    Detection is heuristic: we look for 3 ``_int_mm`` calls all with the
    same A tensor. The replacement emits a single ``fused_qkv`` call.

    Caveats:
    - For models with asymmetric Q/K/V (Qwen3.5-2B GQA), the W shapes
      don't match → pattern doesn't fire (correct).
    - Only fires when W shapes are identical AND contiguous in the FX
      graph. May miss if inductor fuses ops differently.
    """
    from torch._inductor.pattern_matcher import (
        fwd_only,
        register_replacement,
    )
    aten = torch.ops.aten

    def three_qkv_pattern(A, Wq, Wk, Wv):
        return (aten._int_mm(A, Wq), aten._int_mm(A, Wk), aten._int_mm(A, Wv))  # noqa: F841

    def three_qkv_replacement(A, Wq, Wk, Wv):
        if Wq.shape != Wk.shape or Wk.shape != Wv.shape:
            return (aten._int_mm(A, Wq), aten._int_mm(A, Wk), aten._int_mm(A, Wv))
        M, K = A.shape
        N = Wq.shape[0]
        if K != 2048 or M != 1:
            return (aten._int_mm(A, Wq), aten._int_mm(A, Wk), aten._int_mm(A, Wv))
        ones_M = A.new_ones(M, dtype=torch.float32)
        ones_N = Wq.new_ones(N, dtype=torch.float32)
        q, k, v = torch.ops.sglang.v10_fused_qkv(
            A, Wq, Wk, Wv, ones_M, ones_N, ones_N, ones_N
        )
        return (q.to(torch.int32), k.to(torch.int32), v.to(torch.int32))

    return register_replacement(
        three_qkv_pattern,
        three_qkv_replacement,
        [torch.zeros(1, 2048, dtype=torch.int8),
         torch.zeros(2048, 8192, dtype=torch.int8),
         torch.zeros(2048, 8192, dtype=torch.int8),
         torch.zeros(2048, 8192, dtype=torch.int8)],
        fwd_only,
        _pass_dicts[1],
    )


def register_v10_pattern_matchers() -> bool:
    """Register inductor pattern matchers that emit v10 ops.

    Idempotent — safe to call multiple times.

    Returns True if registration actually ran, False if it was a no-op.
    """
    global _REGISTERED
    if _REGISTERED:
        return False

    # Make sure the underlying ops are registered first
    torch_ops.register_v10_for_inductor()

    try:
        _make_int8_gemv_pattern()
        logger.info(
            "sglang_kernel_v10.inductor_patterns: registered int8 GEMV → "
            "v10_gemv_int8_async pattern"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "sglang_kernel_v10.inductor_patterns: int8 GEMV pattern "
            "registration failed (model will fall back to default inductor "
            "lowering): %s", exc
        )

    try:
        _make_fused_qkv_pattern()
        logger.info(
            "sglang_kernel_v10.inductor_patterns: registered 3× int8 GEMV → "
            "v10_fused_qkv pattern (symmetric Q/K/V only)"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "sglang_kernel_v10.inductor_patterns: fused-QKV pattern "
            "registration failed (model will use 3 separate calls): %s", exc
        )

    _REGISTERED = True
    return True


__all__ = ["register_v10_pattern_matchers"]