# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/pass_manager.py

import logging

from torch import fx as fx

from sglang.srt.compilation.fix_functionalization import FixFunctionalizationPass
from sglang.srt.compilation.inductor_pass import (
    CustomGraphPass,
    InductorPass,
    SGLangInductorPass,
    get_pass_context,
)

logger = logging.getLogger(__name__)


class PostGradPassManager(CustomGraphPass):
    """
    The pass manager for post-grad passes.
    It handles configuration, adding custom passes, and running passes.
    It supports uuid for the Inductor code cache. That includes torch<2.6
    support using pickling (in .inductor_pass.CustomGraphPass).

    The order of the post-grad post-passes is:
    1. passes (constructor parameter)
    2. default passes (NoopEliminationPass, FusionPass)
    3. config["post_grad_custom_post_pass"] (if it exists)
    4. fix_functionalization
    This way, all passes operate on a functionalized graph.
    """

    def __init__(self):
        self.passes: list[SGLangInductorPass] = []

    def __call__(self, graph: fx.Graph):
        shape = get_pass_context().runtime_shape
        for pass_ in self.passes:
            if pass_.is_applicable_for_shape(shape):
                pass_(graph)

        # always run fix_functionalization last
        self.fix_functionalization(graph)

    def configure(
        self,
    ):
        self.pass_config = dict()
        self.fix_functionalization = FixFunctionalizationPass()

    def add(self, pass_: InductorPass):
        assert isinstance(pass_, InductorPass)
        self.passes.append(pass_)

    def uuid(self):
        """
        The PostGradPassManager is set as a custom pass in the Inductor and
        affects compilation caching. Its uuid depends on the UUIDs of all
        dependent passes and the pass config. See InductorPass for more info.
        """
        pass_manager_uuid = "fshdakhsa"
        state = {"pass_config": pass_manager_uuid, "passes": []}
        for pass_ in self.passes:
            state["passes"].append(pass_.uuid())
        state["passes"].append(self.fix_functionalization.uuid())
        return InductorPass.hash_dict(state)


# ---------------------------------------------------------------------------
# PPU prologue-fusion pass auto-install (runs once at module load)
# ---------------------------------------------------------------------------
# Patch ``PostGradPassManager.configure`` so that any PPU-fused kernels
# (registered via ``@ppu_fused_kernel`` in
# :mod:`sglang.srt.compilation.ppu_kernels`) are picked up automatically by
# sglang's piecewise CUDA graph / torch.compile pipeline.
#
# The lazy import inside the patched configure breaks the circular
# dependency: ``ppu_kernels`` imports from this module for
# :class:`SGLangInductorPass`.


def _ppu_patched_configure(self):
    """Replacement for :meth:`PostGradPassManager.configure` that also
    installs :class:`PpuPrologueFusionPass` when at least one
    ``@ppu_fused_kernel``-decorated function has been registered."""
    self.pass_config = dict()
    self.fix_functionalization = FixFunctionalizationPass()

    from sglang.srt.compilation.ppu_kernels import (
        PpuKernelRegistry,
        PpuPrologueFusionPass,
    )

    if PpuKernelRegistry.by_pattern() and not any(
        isinstance(p, PpuPrologueFusionPass) for p in self.passes
    ):
        self.passes.append(PpuPrologueFusionPass())
        logger.info(
            "PpuPrologueFusionPass installed (covers %d kernels)",
            len(PpuKernelRegistry.by_pattern()),
        )


# Apply the patch exactly once. Sentinel attribute on the function object
# so re-imports of pass_manager don't double-patch.
if not getattr(PostGradPassManager.configure, "_ppu_patched", False):
    PostGradPassManager.configure = _ppu_patched_configure
    PostGradPassManager.configure._ppu_patched = True
