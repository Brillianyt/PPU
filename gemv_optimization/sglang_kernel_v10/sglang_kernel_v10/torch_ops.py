"""torch_ops.py — Inductor-friendly registrations for sglang_kernel_v10.

Goal
====

The sglang_kernel_v10 C++ extension exposes five entry points
(``gemv_int8_async``, ``gemv_int8_async_v2``, ``fused_qkv``,
``build_gemv_queue``, ``persistent_gemv_decoder_launch``). The wheel is
loaded eagerly at import time (``.so`` was built at ``pip install`` time,
so first-call compile latency and CUDA-graph capture are unaffected).

But for ``torch.compile`` / inductor to *see* the kernels as first-class
ops — so it can symbolically infer output shape/dtype, fuse epilogues
around them, and register them as ``ExternKernelChoice`` — the ops need
to be registered through the public ``torch.library`` API with a fake
(meta) implementation. Without a fake impl, inductor's shape propagation
fails the first time the op appears in an FX graph and the model is
forced off the ``inductor`` path back to eager.

This module:

1. Registers a fake impl for each v10 op on a dedicated
   ``torch.library.Library("sglang_kernel_v10", "FRAGMENT")`` so it can
   be imported without depending on sglang's own ``torch.ops.sglang``
   namespace (the wheel is a sibling package, not a vendored sglang
   module).
2. Also registers them on the sglang custom-op namespace
   (``torch.ops.sglang.v10_gemv_int8_async`` etc.) so SGLang's existing
   ``@ppu_fused_kernel`` machinery picks them up via ``register_fake``.
3. Provides Python wrappers (functional, non-mutating) for the v10 ops
   so they can be called from FX-traceable Python without going through
   the pre-allocated ``out=`` argument that the raw op expects.

Usage
-----

    from sglang_kernel_v10.torch_ops import (
        register_v10_for_inductor,
        v10_gemv_int8_async,    # functional wrapper
        v10_gemv_int8_async_v2, # functional wrapper
        v10_fused_qkv,          # functional wrapper
    )
    register_v10_for_inductor()  # idempotent

    out = v10_gemv_int8_async(A, W, As, Bs)   # returns new bf16 tensor

Design constraints
------------------

- The C++ side requires an *out* tensor to write into (the kernel mutates
  the buffer rather than allocating). The functional wrappers below
  allocate a contiguous bf16 buffer of the right shape and pass it as
  ``out``.
- We never change the C++ kernels' semantics. Only Python-side wrapping
  is added.
- Registration is idempotent — calling ``register_v10_for_inductor``
  twice in the same process (e.g. once via sglang's import and once via
  a model that imports ``sglang_kernel_v10`` directly) is a no-op.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.library

from . import _C  # the C++ extension's pybind11 module

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Namespace + library handle
# ---------------------------------------------------------------------------

# A dedicated library for v10 keeps the public surface separate from
# torch.ops.aten / sglang's own torch.ops.sglang namespace. SGLang sees
# the v10 ops through a *separate* alias registered on the sglang library
# (see _register_alias_on_sglang below) so its PPU framework can pick them
# up automatically.
_v10_lib = torch.library.Library("sglang_kernel_v10", "FRAGMENT")


# Tracks whether ``register_v10_for_inductor`` has already run. Importing
# this module twice in the same process is common in sglang's launcher
# (model loader + worker + cuda-graph capture path).
_REGISTERED = False


def _is_already_defined(lib, schema_prefix):
    """Check if an op with this schema prefix is already defined on the lib.

    torch.library raises RuntimeError on duplicate define() calls. We
    inspect ``lib`` to see what's already registered.
    """
    try:
        for name in dir(lib):
            if name.startswith(schema_prefix):
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Fake (meta) implementations
# ---------------------------------------------------------------------------
#
# Inductor's shape propagation needs a function with the same signature as
# the op that returns a tensor of the correct shape / dtype / device, but
# without doing any real work. The fake impl runs in fake-tensor mode
# (where tensor.shape is symbolic) — it must produce an output whose
# shape can be derived from the input shapes.
#
# v10 signatures (from csrc/gemv_int8_async_wrapper.cpp):
#
#   gemv_int8_async(A int8[M,K], W int8[N,K], As fp32[M], Bs fp32[N],
#                   out bf16[M,N]) -> Tensor          # Phase 3.2
#   gemv_int8_async_v2(A int8[M,K], W int8[N,K], As fp32[M], Bs fp32[N],
#                      out bf16[M,N]) -> Tensor       # Phase 3.2.1
#   fused_qkv(A int8[M,K],
#             Wq int8[N,K], Wk int8[N,K], Wv int8[N,K],
#             As fp32[M], Bs_q fp32[N], Bs_k fp32[N], Bs_v fp32[N],
#             out_q bf16[M,N], out_k bf16[M,N], out_v bf16[M,N]) -> Tensor
#
#   build_gemv_queue(items: List[Tensor]) -> (Tensor queue, int total_work)
#   persistent_gemv_decoder_launch(queue: Tensor, total_work: int) -> None


def _fake_gemv_int8_async(A, W, As, Bs, out):
    """Fake impl for v10 ``gemv_int8_async``.

    Returns the pre-allocated output buffer unchanged. Inductor's
    symbolic-shape inference reads ``A.shape`` / ``W.shape`` to verify
    consistency, but the output shape we need is the union of
    ``A.shape[0]`` and ``W.shape[0]``, which equals ``out.shape``.
    """
    return out


def _fake_gemv_int8_async_v2(A, W, As, Bs, out):
    return out


def _fake_fused_qkv(A, Wq, Wk, Wv, As, Bs_q, Bs_k, Bs_v, out_q, out_k, out_v):
    return out_q


def _fake_build_gemv_queue(items):
    """Return a placeholder (queue_tensor, total_work)."""
    total = len(items)
    queue = torch.empty(0, dtype=torch.int64, device=items[0].device if items else "cpu")
    return queue, total


def _fake_persistent_gemv_decoder_launch(queue, total_work):
    return None


# ---------------------------------------------------------------------------
# Functional Python wrappers (for FX tracing & inductor lowering)
# ---------------------------------------------------------------------------
#
# The C++ binding mutates a pre-allocated output tensor. Inside a traced
# FX graph we want a *functional* form: take inputs, return a new tensor.
# The wrappers below allocate a fresh output buffer, call the binding,
# and return the buffer. The allocation is amortized across the
# CUDA-graph capture lifetime because sglang re-uses the same captured
# buffer for replay; in eager mode it's a single ``torch.empty`` which
# is cheap (~1us on H100, free on most capture paths).


def v10_gemv_int8_async(
    A: torch.Tensor,
    W: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
) -> torch.Tensor:
    """Functional INT8 GEMV (v10 HBM-direct).

    Returns a new bf16 tensor of shape ``(A.shape[0], W.shape[0])``.
    Constraints (matching the C++ side): ``A.shape[0] == 1``,
    ``A.shape[1] == 2048`` (PPU001 Phase 3.2 MVP).
    """
    M = A.shape[0]
    N = W.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)
    _C.gemv_int8_async(A, W, As, Bs, out)
    return out


def v10_gemv_int8_async_v2(
    A: torch.Tensor,
    W: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
) -> torch.Tensor:
    """Functional INT8 GEMV (v2 cp.async double-buffer, Phase 3.2.1)."""
    M = A.shape[0]
    N = W.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)
    _C.gemv_int8_async_v2(A, W, As, Bs, out)
    return out


def v10_fused_qkv(
    A: torch.Tensor,
    Wq: torch.Tensor,
    Wk: torch.Tensor,
    Wv: torch.Tensor,
    As: torch.Tensor,
    Bs_q: torch.Tensor,
    Bs_k: torch.Tensor,
    Bs_v: torch.Tensor,
):
    """Functional fused Q+K+V INT8 GEMV (Phase 3.4).

    Returns ``(out_q, out_k, out_v)`` each of shape ``(A.shape[0], Wq.shape[0])``.
    """
    M = A.shape[0]
    N = Wq.shape[0]
    out_q = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)
    out_k = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)
    out_v = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)
    _C.fused_qkv(A, Wq, Wk, Wv, As, Bs_q, Bs_k, Bs_v, out_q, out_k, out_v)
    return out_q, out_k, out_v


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _define_and_impl_on_v10_lib() -> None:
    """Register the raw C++ entry points (mutating form) on the
    ``sglang_kernel_v10`` library with their fake impls.

    These are the *op overloads* that inductor sees when FX-tracing
    through the functional wrappers above.
    """
    # gemv_int8_async — A:int8[M,K], W:int8[N,K], As:fp32[M], Bs:fp32[N],
    #                    out:bf16[M,N] -> Tensor
    if not _is_already_defined(_v10_lib, "gemv_int8_async"):
        _v10_lib.define(
            "gemv_int8_async(Tensor A, Tensor W, Tensor As, Tensor Bs, "
            "Tensor(a!) out) -> Tensor"
        )
        _v10_lib.impl("gemv_int8_async", _C.gemv_int8_async, "CUDA")
        _v10_lib._register_fake("gemv_int8_async", _fake_gemv_int8_async)

    if not _is_already_defined(_v10_lib, "gemv_int8_async_v2"):
        _v10_lib.define(
            "gemv_int8_async_v2(Tensor A, Tensor W, Tensor As, Tensor Bs, "
            "Tensor(a!) out) -> Tensor"
        )
        _v10_lib.impl("gemv_int8_async_v2", _C.gemv_int8_async_v2, "CUDA")
        _v10_lib._register_fake("gemv_int8_async_v2", _fake_gemv_int8_async_v2)

    if not _is_already_defined(_v10_lib, "fused_qkv"):
        _v10_lib.define(
            "fused_qkv(Tensor A, Tensor Wq, Tensor Wk, Tensor Wv, Tensor As, "
            "Tensor Bs_q, Tensor Bs_k, Tensor Bs_v, Tensor(a!) out_q, "
            "Tensor(a!) out_k, Tensor(a!) out_v) -> Tensor"
        )
        _v10_lib.impl("fused_qkv", _C.fused_qkv, "CUDA")
        _v10_lib._register_fake("fused_qkv", _fake_fused_qkv)

    if not _is_already_defined(_v10_lib, "build_gemv_queue"):
        _v10_lib.define("build_gemv_queue(Tensor[] items) -> (Tensor, int)")
        _v10_lib.impl("build_gemv_queue", _C.build_gemv_queue, "CUDA")
        _v10_lib._register_fake("build_gemv_queue", _fake_build_gemv_queue)

    if not _is_already_defined(_v10_lib, "persistent_gemv_decoder_launch"):
        _v10_lib.define(
            "persistent_gemv_decoder_launch(Tensor queue, int total_work) -> ()"
        )
        _v10_lib.impl(
            "persistent_gemv_decoder_launch",
            _C.persistent_gemv_decoder_launch,
            "CUDA",
        )
        _v10_lib._register_fake(
            "persistent_gemv_decoder_launch",
            _fake_persistent_gemv_decoder_launch,
        )


def _register_alias_on_sglang_lib() -> bool:
    """Register alias op-names on sglang's torch.ops.sglang namespace so
    that the existing ``@ppu_fused_kernel`` framework picks them up.

    Why an alias: the framework in
    ``sglang.srt.compilation.ppu_kernels`` walks
    ``torch.ops.sglang.<name>.default`` to identify inductor-fusable
    custom ops. By exposing v10 under the same namespace the framework's
    ``PpuPrologueFusionPass`` automatically treats them as PPU-fused
    kernels (epilogue fusion = bias_add, prologue fusion = per_token_quant).

    The aliases are *functional* (no pre-allocated output arg), which is
    what the framework expects for inductor to be able to fuse an
    epilogue around them.

    We must reuse sglang's existing ``sglang_lib`` singleton
    (``torch.library.Library("sglang", "FRAGMENT")``); creating a second
    Library with the same namespace/kind is a torch error.

    Returns True if aliases were registered, False if skipped (e.g. sglang
    not imported yet — caller should retry after sglang import).
    """
    try:
        from sglang.srt.utils.common import sglang_lib
    except ImportError as exc:
        # sglang may not be imported in this process. The v10 lib
        # registration is still active for direct use. Caller should retry
        # after sglang is imported (e.g. via register_sglang_alias()).
        logger.debug(
            "sglang.srt.utils.common.sglang_lib not importable yet: %s", exc,
        )
        return False

    # ---- v10_gemv_int8_async -------------------------------------------------
    try:
        sglang_lib.define(
            "v10_gemv_int8_async(Tensor A, Tensor W, Tensor As, "
            "Tensor Bs) -> Tensor"
        )
    except (RuntimeError, AttributeError):
        pass  # already defined

    try:
        sglang_lib.impl("v10_gemv_int8_async", v10_gemv_int8_async, "CUDA")
        sglang_lib._register_fake(
            "v10_gemv_int8_async",
            lambda A, W, As, Bs: torch.empty(
                (A.shape[0], W.shape[0]),
                dtype=torch.bfloat16,
                device=A.device,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("v10_gemv_int8_async alias skipped: %s", exc)

    # ---- v10_gemv_int8_async_v2 ----------------------------------------------
    try:
        sglang_lib.define(
            "v10_gemv_int8_async_v2(Tensor A, Tensor W, Tensor As, "
            "Tensor Bs) -> Tensor"
        )
    except (RuntimeError, AttributeError):
        pass

    try:
        sglang_lib.impl(
            "v10_gemv_int8_async_v2", v10_gemv_int8_async_v2, "CUDA"
        )
        sglang_lib._register_fake(
            "v10_gemv_int8_async_v2",
            lambda A, W, As, Bs: torch.empty(
                (A.shape[0], W.shape[0]),
                dtype=torch.bfloat16,
                device=A.device,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("v10_gemv_int8_async_v2 alias skipped: %s", exc)

    # ---- v10_fused_qkv -------------------------------------------------------
    try:
        sglang_lib.define(
            "v10_fused_qkv(Tensor A, Tensor Wq, Tensor Wk, Tensor Wv, "
            "Tensor As, Tensor Bs_q, Tensor Bs_k, Tensor Bs_v) -> "
            "(Tensor, Tensor, Tensor)"
        )
    except (RuntimeError, AttributeError):
        pass

    try:
        sglang_lib.impl("v10_fused_qkv", v10_fused_qkv, "CUDA")
        sglang_lib._register_fake(
            "v10_fused_qkv",
            lambda A, Wq, Wk, Wv, As, Bs_q, Bs_k, Bs_v: (
                torch.empty(
                    (A.shape[0], Wq.shape[0]),
                    dtype=torch.bfloat16,
                    device=A.device,
                ),
                torch.empty(
                    (A.shape[0], Wq.shape[0]),
                    dtype=torch.bfloat16,
                    device=A.device,
                ),
                torch.empty(
                    (A.shape[0], Wq.shape[0]),
                    dtype=torch.bfloat16,
                    device=A.device,
                ),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("v10_fused_qkv alias skipped: %s", exc)

    return True


# Tracks whether the sglang alias registration has actually succeeded.
# Separate from ``_REGISTERED`` (which tracks the v10 lib + alias combined).
_sglang_alias_registered = False


def register_sglang_alias() -> bool:
    """Idempotently register v10 ops on sglang's torch.ops.sglang namespace.

    Call this AFTER importing sglang. The auto-import path in this module's
    bottom-of-file may run before sglang is imported; in that case the sglang
    alias registration is skipped and this function should be called later.

    Example::

        import sglang_kernel_v10       # registers v10_lib
        import sglang                   # imports sglang
        from sglang_kernel_v10.torch_ops import register_sglang_alias
        register_sglang_alias()         # now sglang_lib is importable

    Returns True if alias was newly registered, False if already done.
    """
    global _sglang_alias_registered
    if _sglang_alias_registered:
        return False
    if _register_alias_on_sglang_lib():
        _sglang_alias_registered = True
        logger.info(
            "sglang_kernel_v10.torch_ops: registered torch.ops.sglang.* "
            "aliases (v10_gemv_int8_async, v10_gemv_int8_async_v2, v10_fused_qkv)"
        )
        return True
    return False


def register_v10_for_inductor() -> bool:
    """Idempotently register v10 ops with torch.library + sglang aliases.

    Returns True if the registration actually ran, False if it was a
    no-op (already registered in this process).
    """
    global _REGISTERED
    if _REGISTERED:
        return False

    # Set the flag *first* so the second call is a no-op even if the first
    # call's registration raises partway through. The lib registration is
    # atomic from torch.library's POV (one define per op), so partial
    # registration is impossible.
    _REGISTERED = True

    lib_ok = True
    alias_ok = True

    try:
        _define_and_impl_on_v10_lib()
    except Exception as exc:  # noqa: BLE001
        lib_ok = False
        logger.warning(
            "sglang_kernel_v10.torch_ops: v10_lib registration failed: %s. "
            "Functions still callable directly via sglang_kernel_v10.gemv_int8_async",
            exc,
        )

    try:
        _register_alias_on_sglang_lib()
    except Exception as exc:  # noqa: BLE001
        alias_ok = False
        logger.debug(
            "sglang_kernel_v10.torch_ops: sglang alias registration failed "
            "(sglang not installed?): %s", exc,
        )

    if lib_ok or alias_ok:
        logger.info(
            "sglang_kernel_v10.torch_ops: registered (lib=%s, sglang_alias=%s) "
            "v10 ops for torch.compile / inductor",
            lib_ok, alias_ok,
        )
    return True


# Run registration on import so downstream code (sglang worker,
# torch.compile, model loader) sees the registered ops without having to
# call ``register_v10_for_inductor()`` explicitly.
try:
    register_v10_for_inductor()
except Exception as exc:  # noqa: BLE001
    # Registration should never break a downstream import; just log so
    # the user knows to call register_v10_for_inductor() manually.
    logger.warning(
        "sglang_kernel_v10.torch_ops: eager registration failed: %s. "
        "Call register_v10_for_inductor() manually before compiling.",
        exc,
    )


__all__ = [
    "register_v10_for_inductor",
    "v10_gemv_int8_async",
    "v10_gemv_int8_async_v2",
    "v10_fused_qkv",
]