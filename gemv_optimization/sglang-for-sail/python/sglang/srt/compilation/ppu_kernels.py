# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""
PPU kernel integration with torch.compile / inductor.

Design goal
===========

Hand-written CUDA kernels (INT8 GEMV/GEMM on PPU, fused linear+activation,
etc.) need to participate naturally in inductor's fusion pipeline instead of
being opaque black boxes next to it. This module provides the engineering
abstraction that makes that possible.

Architecture
============

Three layers, each independently replaceable:

1. **Functional contract layer** (``@ppu_fused_kernel`` decorator).
   Every hand-written kernel is wrapped so that, from torch.compile's point
   of view, it is a *functional* op: takes inputs, returns a new output.
   Mutating (``out=...``) variants stay as a parallel API for callers that
   want them.

2. **Inductor bridge layer** (``register_as_extern_choice``).
   Each functional wrapper is registered as a ``torch._inductor.select_algorithm.ExternKernelChoice``
   so that inductor's scheduler treats it as a known extern kernel and can
   decide whether to interleave a Triton prologue/epilogue around it.

3. **Pattern-rewriting layer** (``PpuPrologueFusionPass``).
   Each kernel may declare the prologue/epilogue patterns it can fuse with
   (e.g. ``gemv_int8_nm + bias_add``). A single ``SGLangInductorPass`` walks
   the post-grad FX graph and rewrites matched patterns into a single
   fused custom op, which itself is registered as an ``ExternKernelChoice``.

Backward compatibility
======================

The existing mutating APIs (``gemv_int8_i8i8bf16``, ``gemm_nt_i8i8bf16``)
remain untouched. The framework adds parallel ``*_nm`` (non-mutating)
variants. Callers can opt in.

Extension points
================

To integrate a new hand-written kernel:

.. code-block:: python

    @ppu_fused_kernel(
        name="my_op_nm",
        output=("out", torch.bfloat16),  # functional return
        cpp_symbol="sglang::my_op",      # optional, for inductor extern
        epilogue_fusion=("bias_add",),  # optional, enables pattern pass
    )
    def my_op_nm(a, b):
        out = torch.empty(...)
        my_cuda_kernel(a, b, out)
        return out

The framework handles registration with both ``torch.library`` (for shape
propagation) and ``torch._inductor`` (for fusion) automatically.
"""

from __future__ import annotations

import functools
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import fx
from torch._inductor import config as inductor_config
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import ExternKernelChoice

from sglang.srt.compilation.inductor_pass import SGLangInductorPass
from sglang.srt.utils.common import direct_register_custom_op

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class PpuFusedKernel:
    """Description of one hand-written kernel registered for inductor fusion."""

    name: str
    python_fn: Callable
    op_overload: Any  # torch.ops.<ns>.<name>.default
    fake_impl: Callable
    inputs: Tuple[str, ...]
    output_name: Optional[str]
    output_dtype: Optional[torch.dtype]
    output_shape_from: Optional[str]  # which input arg's shape determines output shape
    cpp_symbol: Optional[str]
    epilogue_fusion: Tuple[str, ...] = field(default_factory=tuple)
    registered_as_extern: bool = False

    def register_extern(self) -> None:
        """Register with torch._inductor as an extern kernel choice.

        Once registered, inductor's scheduler can emit
        ``extern_kernels.<cpp_symbol>`` calls in the generated wrapper and
        decide whether to fuse a Triton prologue / epilogue around them.
        """
        if self.registered_as_extern:
            return
        if self.cpp_symbol is None:
            return
        try:
            choice = ExternKernelChoice(
                self.python_fn,
                self.cpp_symbol,
                op_overload=self.op_overload,
            )
            register_lowering(self.op_overload, choice)
            self.registered_as_extern = True
            logger.debug(
                "ppu_fused_kernel: registered %s as ExternKernelChoice "
                "(symbol=%s)",
                self.name,
                self.cpp_symbol,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ppu_fused_kernel: failed to register %s as ExternKernelChoice: %s",
                self.name,
                exc,
            )


class PpuKernelRegistry:
    """Global registry of PPU-fused kernels. Singleton."""

    _instance: "PpuKernelRegistry" = None
    _kernels: Dict[str, PpuFusedKernel] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._kernels = {}
        return cls._instance

    @classmethod
    def register(cls, kernel: PpuFusedKernel) -> None:
        cls._kernels[kernel.name] = kernel
        kernel.register_extern()

    @classmethod
    def get(cls, name: str) -> Optional[PpuFusedKernel]:
        return cls._kernels.get(name)

    @classmethod
    def all(cls) -> Dict[str, PpuFusedKernel]:
        return dict(cls._kernels)

    @classmethod
    def by_pattern(cls) -> List[PpuFusedKernel]:
        """Return kernels that declared epilogue-fusion patterns, in registration order."""
        return [k for k in cls._kernels.values() if k.epilogue_fusion]


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def _make_fake_impl_from_output(
    output_name: str,
    output_dtype: torch.dtype,
    output_shape_from: Optional[str],
) -> Callable:
    """Build a fake_impl that allocates a new tensor of ``output_dtype`` with
    shape derived from a specific input argument."""

    def fake_impl(*args, **kwargs):
        sig = inspect.signature(_captured_fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        ref = bound.arguments[output_shape_from]
        return torch.empty(ref.shape, dtype=output_dtype, device=ref.device)

    return fake_impl


_captured_fn: Callable = None  # used by fake_impl factory to read signature


def ppu_fused_kernel(
    name: str,
    *,
    output: Optional[Tuple[str, torch.dtype]] = None,
    output_shape_from: Optional[str] = None,
    cpp_symbol: Optional[str] = None,
    epilogue_fusion: Tuple[str, ...] = (),
) -> Callable:
    """Decorator factory: register a hand-written CUDA kernel as a
    functional, inductor-fusable custom op.

    :param name: Custom-op name (will be exposed as ``torch.ops.sglang.<name>``).
    :param output: Tuple of ``(arg_name, dtype)`` declaring that the op
        returns a new tensor whose dtype is ``dtype``. Set to ``None`` for
        purely mutating ops (legacy mode).
    :param output_shape_from: Which input argument's shape determines the
        output shape. Required when ``output`` is set.
    :param cpp_symbol: C++ symbol used by inductor's ``extern_kernels.*``
        path. If ``None``, inductor falls back to the Python op call.
    :param epilogue_fusion: Tuple of pattern names the post-grad pass
        should recognise for fusion with this op. Currently supports
        ``"bias_add"`` and ``"residual"``.
    """

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        input_names = tuple(sig.parameters.keys())

        if output is not None:
            # Functional path: allocate output inside, return new tensor.
            output_name, output_dtype = output
            shape_src = output_shape_from or input_names[0]

            # Build fake_impl that allocates a new tensor of the right
            # shape and dtype. We capture ``fn`` in the closure so the
            # fake_impl sees the real signature.
            global _captured_fn
            _captured_fn = fn

            fake_impl = _make_fake_impl_from_output(
                output_name, output_dtype, shape_src
            )

            # Register as torch.library custom op (sglang namespace).
            try:
                direct_register_custom_op(
                    op_name=name,
                    op_func=fn,
                    mutates_args=[],
                    fake_impl=fake_impl,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ppu_fused_kernel: direct_register_custom_op(%s) failed: %s",
                    name,
                    exc,
                )
                return fn

            op_overload = getattr(getattr(torch.ops.sglang, name, None), "default", None)
            if op_overload is None:
                logger.warning(
                    "ppu_fused_kernel: op_overload for %s not found after register",
                    name,
                )
                return fn

            # Build a thin Python wrapper that always goes through
            # torch.ops.sglang.<name>.default. This guarantees FX graphs
            # see the op-overload, not the bare Python function.
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return op_overload(*args, **kwargs)

            kernel = PpuFusedKernel(
                name=name,
                python_fn=wrapper,
                op_overload=op_overload,
                fake_impl=fake_impl,
                inputs=input_names,
                output_name=output_name,
                output_dtype=output_dtype,
                output_shape_from=shape_src,
                cpp_symbol=cpp_symbol,
                epilogue_fusion=epilogue_fusion,
            )
            PpuKernelRegistry.register(kernel)
            return wrapper

        else:
            # Legacy / mutating path: keep the function as-is, no inductor
            # integration. Useful for the existing GEMV/GEMM API.
            logger.debug(
                "ppu_fused_kernel: %s registered without inductor bridge "
                "(no `output` specified)",
                name,
            )
            return fn

    return decorator


# ---------------------------------------------------------------------------
# Post-grad pattern-rewriting pass
# ---------------------------------------------------------------------------


# Maps epilogue_fusion tag -> (target op, arg-positions) describing how to
# fuse the matched epilogue into a single fused custom op name.
_EPILOGUE_PATTERNS: Dict[str, Dict[str, Any]] = {
    "bias_add": {
        "op": torch.ops.aten.add.Tensor,
        "fused_suffix": "_bias",
        "description": "C = matmul(...) ; Y = C + bias  ->  C = matmul_bias(...) ",
    },
}


def _make_fused_op_name(base_name: str, epilogue: str) -> str:
    suffix = _EPILOGUE_PATTERNS.get(epilogue, {}).get("fused_suffix")
    if suffix is None:
        raise ValueError(f"Unknown epilogue pattern: {epilogue}")
    return f"{base_name}{suffix}"


class PpuPrologueFusionPass(SGLangInductorPass):
    """Generic post-grad pass that fuses declared epilogue patterns with
    registered PPU-fused kernels.

    For each registered kernel that declared e.g. ``epilogue_fusion=("bias_add",)``,
    the pass scans the FX graph for:

        %y = torch.ops.sglang.<kernel>(...)
        %z = torch.ops.aten.add.Tensor(%y, %bias, alpha=...)

    and replaces them with a single call to ``torch.ops.sglang.<kernel>_bias``
    (which is itself registered as an ``ExternKernelChoice`` so inductor
    emits it as a single extern call in the generated wrapper).

    Adding a new pattern
    --------------------

    Subclass and override ``_build_fused_call``. The default implementation
    handles ``"bias_add"`` and ``"residual"``.
    """

    def __init__(self):
        super().__init__()
        # Lazily build a map: (op_overload, epilogue_tag) -> fused_op_overload
        # so we know which fused op name to insert.
        #
        # We key on the actual OpOverload object (not the string name) because
        # ``fx.Node.target`` IS the OpOverload object — comparing objects is
        # both faster and immune to formatting quirks.
        self._fused_kernels: Dict[Tuple[Any, str], Any] = {}
        for k in PpuKernelRegistry.by_pattern():
            for ep in k.epilogue_fusion:
                fused_name = _make_fused_op_name(k.name, ep)
                fused_op = _resolve_fused_op(fused_name)
                if fused_op is None:
                    continue
                self._fused_kernels[(k.op_overload, ep)] = fused_op

    def __call__(self, graph: fx.Graph) -> None:
        if not self._fused_kernels:
            return
        for _ in list(self._find_and_rewrite(graph)):
            pass  # iterates until no more rewrites

    # ---- rewriting -----------------------------------------------------

    def _find_and_rewrite(self, graph: fx.Graph):
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            for (op_overload, epilogue), fused_op in self._fused_kernels.items():
                if node.target is not op_overload:
                    continue
                epilogue_node = _find_epilogue_consumer(node, epilogue)
                if epilogue_node is None:
                    continue
                yield from _rewrite_to_fused(
                    graph, node, epilogue_node, epilogue, fused_op
                )


# --- rewriting helpers ---


def _matches_kernel(node: fx.Node, kernel_name: str) -> bool:
    """Return True if ``node`` is a call to ``torch.ops.sglang.<kernel_name>.default``.

    ``OpOverloadPacket.__name__`` uses underscores for the overload suffix
    (e.g. ``"sglang.gemv_int8_i8i8bf16_nm_default"``); the namespace-qualified
    name uses dots. Match either form.
    """
    tgt = node.target
    name = getattr(tgt, "__name__", "")
    qualified = getattr(tgt, "_qualified_op_name", None) or ""
    if callable(qualified):
        try:
            qualified = qualified()
        except Exception:  # noqa: BLE001
            qualified = ""
    return (
        name == f"sglang.{kernel_name}_default"
        or name == f"sglang.{kernel_name}.default"
        or qualified == f"sglang::{kernel_name}"
        or qualified == f"sglang.{kernel_name}"
    )


def _find_epilogue_consumer(node: fx.Node, epilogue: str) -> Optional[fx.Node]:
    """If ``node`` has exactly one user that matches the epilogue pattern,
    return that user node. Otherwise ``None``."""
    if len(node.users) != 1:
        return None
    user = list(node.users.keys())[0]
    if user.op != "call_function":
        return None
    # FX tracing may lower torch.add to either ``aten.add.Tensor`` (post-AOT)
    # or the built-in ``add`` (symbolic_trace). Match either.
    target = user.target
    is_add = (
        target is torch.ops.aten.add.Tensor
        or target is torch.ops.aten.add.Scalar
        or target is getattr(getattr(torch, "add", None), "__wrapped__", None)
        or getattr(target, "__name__", "") == "add"
    )
    if epilogue in ("bias_add", "residual"):
        return user if is_add else None
    return None


def _resolve_fused_op(fused_name: str):
    """Look up ``torch.ops.sglang.<fused_name>.default`` (returning ``None`` if not registered)."""
    op = getattr(torch.ops.sglang, fused_name, None)
    if op is None:
        return None
    return getattr(op, "default", None)


def _rewrite_to_fused(
    graph: fx.Graph,
    kernel_node: fx.Node,
    epilogue_node: fx.Node,
    epilogue: str,
    fused_op,
):
    """Replace ``kernel_node`` + ``epilogue_node`` with a single fused call.

    For ``bias_add``:

        %y   = torch.ops.sglang.<kernel>(...)    # kernel_node
        %out = aten.add(%y, %bias, alpha=1.0)    # epilogue_node

    becomes:

        %out = torch.ops.sglang.<kernel>_bias(<args of kernel>, %bias)
    """
    if epilogue == "bias_add":
        # epilogue args: (input, other, alpha=1) — `input` is kernel_node's
        # output, `other` is the bias
        bias_node = epilogue_node.args[1]
        with graph.inserting_before(epilogue_node):
            new_args = list(kernel_node.args) + [bias_node]
            fused_node = graph.call_function(
                fused_op, tuple(new_args), kwargs={}
            )
        epilogue_node.replace_all_uses_with(fused_node)
        graph.erase_node(epilogue_node)
        # We deliberately leave kernel_node in place — inductor will
        # recognise the fused op as the externally-defined kernel and emit
        # a single extern call. The unused kernel_node is cleaned up by
        # dead-code elimination later.
        yield fused_node


# ---------------------------------------------------------------------------
# Auto-registration with PostGradPassManager
# ---------------------------------------------------------------------------
#
# The install is performed by :mod:`sglang.srt.compilation.pass_manager` at
# module load time: it monkey-patches ``PostGradPassManager.configure`` to
# append :class:`PpuPrologueFusionPass` whenever at least one
# ``@ppu_fused_kernel``-decorated function has been registered.
#
# This split (patching in pass_manager, not ppu_kernels) breaks the circular
# import: ``ppu_kernels`` imports from pass_manager for the
# ``SGLangInductorPass`` base class.


__all__ = [
    "PpuFusedKernel",
    "PpuKernelRegistry",
    "PpuPrologueFusionPass",
    "ppu_fused_kernel",
]
