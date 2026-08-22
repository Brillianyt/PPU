"""PPU RoPE / rotary embedding hook.

Why this hook exists
====================

``sglang.srt.layers.rotary_embedding.utils.apply_rotary_pos_emb_native`` is
decorated with::

    @torch.compile(dynamic=True, backend=get_compiler_backend())

During CUDA Graph capture (ViT's ``ViTCudaGraphRunner._create_graph`` →
``VisionAttention.forward`` → ``apply_rotary_pos_emb``), the first call into
the ``@torch.compile``-wrapped function triggers ``torch._dynamo`` → inductor
→ ``_sfdp_init`` → ``_get_sfdp_patterns`` → ``triton_heuristics.copy_args_to_cpu_if_needed``.

On PPU (a T-Head CUDA-compatible accelerator), inductor's SDPA pattern
match + autotune pin_memory scratch is not safe under
``torch.cuda.graph(graph)`` capture — the capture context rejects the
side-effects and the graph never finishes building.

PPU already ships the fix in its flash-attn fork (``flash_attn_interface.py``
lines 40 / 132):

    @torch.library.custom_op("flash_attn_3_ops::_flash_attn_forward", ...)
    def _flash_attn_forward(q, k, v, ...):
        ...real kernel call...

    @torch.library.register_fake("flash_attn_3_ops::_flash_attn_forward")
    def _flash_attn_forward_fake(q, k, v, ...):
        ...symbolic empty tensors, no real work...

By registering both a real op and a fake op, inductor treats the call as an
opaque CustomOp: it never traces into the function body, so ``_sfdp_init``
never runs and the pin_memory path is never taken during capture.  This is
the **memory-equivalent** of the upstream ``@torch.compile`` path: same
argument list, same output shapes, but the dispatch goes through the
custom-op slot instead of through inductor's pattern matcher.

We use exactly the same pattern for RoPE.  The math itself is elementwise
(no CUDA kernel needed — we just keep the upstream eager math), but the
**registration** through ``@torch.library.custom_op`` +
``@torch.library.register_fake`` is what makes ViT's CUDA Graph capture
safe on PPU.

Hook surface
============

* REPLACE ``sglang.srt.layers.rotary_embedding.utils.apply_rotary_pos_emb``
  with a custom-op-backed implementation.

  Note: the alias is assigned at module load to one of:
    * ``apply_rotary_pos_emb_npu``        (NPU)
    * ``torch.ops.sgl_kernel.apply_rotary_pos_emb_cpu``  (CPU+AMX)
    * ``apply_rotary_pos_emb_native``     (everything else, including PPU)
  PPU is none of the above, so the alias points to the compiled native
  version.  Hooking the alias swaps in our custom-op-backed version.

  Both the ViT path (``sglang/srt/layers/attention/vision.py:1260``) and
  the LM-side ``rotary_embedding/base.py`` ``_apply_rotary_emb_wrapped``
  call site route through the alias, so a single hook fixes every RoPE
  site that previously hit the inductor path.

Verified semantics vs. upstream
===============================

* Inputs identical: ``(q, k, cos, sin, unsqueeze_dim=1)``
* Outputs identical: ``(q_embed, k_embed)`` with the same shapes/dtypes
  as the upstream native version returns.
* Math identical to ``apply_rotary_pos_emb_native`` (the body is a
  copy of the upstream code, just without the ``@torch.compile``
  decorator).
* Numerically identical when both run on the same inputs.
"""

from __future__ import annotations

from typing import Tuple

import torch

# Custom-op name. Namespace "sglang::" mirrors what FA3 uses
# ("flash_attn_3_ops::_flash_attn_forward") and what flashinfer uses
# ("flashinfer::apply_rope"). The double-colon is the torch.library
# convention for ops registered from a python module.
_PPU_ROPE_OP = "sglang::apply_rotary_pos_emb"


# ---------------------------------------------------------------------------
# Real implementation (eager math, same body as upstream native)
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Identical to ``rotary_embedding.utils.rotate_half``; inlined so the
    hook module has no dependency on the hooked module's helpers.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.library.custom_op(_PPU_ROPE_OP, mutates_args=(), device_types="cuda")
def apply_rotary_pos_emb_ppu(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Real impl of RoPE for PPU.

    Body is byte-for-byte the same math as
    ``sglang.srt.layers.rotary_embedding.utils.apply_rotary_pos_emb_native``,
    just without the ``@torch.compile`` decorator.  Registered as a
    custom op so inductor sees it as opaque.
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q_f, k_f = q.float(), k.float()
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    q_embed = (q_f * cos) + (_rotate_half(q_f) * sin)
    k_embed = (k_f * cos) + (_rotate_half(k_f) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


# ---------------------------------------------------------------------------
# Fake implementation (for inductor symbolic tracing)
# ---------------------------------------------------------------------------
#
# Mirrors flash_attn_interface.py:132 — ``@torch.library.register_fake``.
# Inductor calls the fake impl to learn the output shapes/dtypes without
# executing real work.  As long as the fake returns tensors with the
# right shapes, inductor's pattern matcher never touches the real body,
# which is exactly what keeps ``_sfdp_init`` / pin_memory out of capture.


@torch.library.register_fake(_PPU_ROPE_OP)
def apply_rotary_pos_emb_ppu_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symbolic fake — same shapes as real output, no real computation.

    The real ``apply_rotary_pos_emb_ppu`` returns tensors shaped like
    ``q`` and ``k`` (possibly cast to a different dtype).  The fake
    returns ``q`` and ``k`` themselves — same shape, same dtype, same
    device — which is the minimum inductor needs to keep tracing without
    re-entering the function body.
    """
    return q, k


# ---------------------------------------------------------------------------
# Plugin hook registration
# ---------------------------------------------------------------------------
#
# Wire the custom-op-backed function into the sglang module alias so that
# every caller (ViT's ``VisionAttention.forward``, the LM-side
# ``_apply_rotary_emb_wrapped`` in ``rotary_embedding/base.py``, etc.)
# transparently dispatches through the custom-op slot.


from sglang.srt.plugins.hook_registry import HookType, plugin_hook  # noqa: E402


@plugin_hook(
    "sglang.srt.layers.rotary_embedding.utils.apply_rotary_pos_emb",
    type=HookType.REPLACE,
)
def _ppu_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """REPLACE hook: swap compiled RoPE for custom-op-backed PPU RoPE.

    Internally delegates to ``apply_rotary_pos_emb_ppu``, which is
    registered as ``sglang::apply_rotary_pos_emb`` via
    ``@torch.library.custom_op`` so inductor treats it as opaque.

    This is the cuda-equivalent of the upstream ``@torch.compile`` path:
    same signature, same math, but dispatch goes through the custom-op
    slot instead of inductor's SDPA pattern matcher.  On PPU that keeps
    ``_get_sfdp_patterns`` / ``copy_args_to_cpu_if_needed`` out of the
    ViT CUDA Graph capture path.
    """
    return apply_rotary_pos_emb_ppu(q, k, cos, sin, unsqueeze_dim)