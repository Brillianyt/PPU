#!/usr/bin/env bash
# install_v10.sh — 1-shot deployment of the ISA Phase 3.2 v10 GEMV kernel.
#
# What this does:
#   1. pip install the sglang_kernel_v10 wheel (builds _C.so at install time)
#   2. deploy the patched w8a8_int8.py dispatcher to sglang site-packages
#
# After this, sglang's decode path (M=1, K=2048) routes through our v10
# GEMV fast path, falling back to deep_gemm for other shapes.
#
# Idempotent: re-running is safe. Each step is reversible.
#
# To uninstall / rollback:
#   pip uninstall sglang_kernel_v10
#   cp gemv_optimization/sglang-for-sail/python/sglang/srt/layers/quantization/w8a8_int8.py \
#      /usr/local/lib/python3.12/site-packages/sglang/srt/layers/quantization/w8a8_int8.py
#   # Then set _ISA_V10_ENABLE = False at the top of the patched w8a8_int8.py

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPU_ROOT="$(cd "$HERE/../.." && pwd)"

SITE_PACKAGES="${SITE_PACKAGES:-/usr/local/lib/python3.12/site-packages}"
SGLANG_QUANT_DIR="$SITE_PACKAGES/sglang/srt/layers/quantization"
SGLANG_KERNEL_V10_DIR="$SITE_PACKAGES/sglang_kernel_v10"

echo "=================================================================="
echo "ISA Phase 3.2 v10 GEMV — install script"
echo "=================================================================="
echo "PPU root:      $PPU_ROOT"
echo "Kernel src:    $HERE"
echo "site-packages: $SITE_PACKAGES"
echo

# ---------------------------------------------------------------------------
# Step 1: build & install the wheel
# ---------------------------------------------------------------------------
echo "[1/3] Building & installing sglang_kernel_v10 wheel ..."
cd "$HERE"
pip install .
echo "      Installed at: $SGLANG_KERNEL_V10_DIR"
echo

# ---------------------------------------------------------------------------
# Step 2: deploy patched w8a8_int8.py dispatcher
# ---------------------------------------------------------------------------
W8A8_SRC="$PPU_ROOT/gemv_optimization/sglang-for-sail/python/sglang/srt/layers/quantization/w8a8_int8.py"
W8A8_DST="$SGLANG_QUANT_DIR/w8a8_int8.py"

if [ ! -f "$W8A8_SRC" ]; then
    echo "ERROR: mirror source not found: $W8A8_SRC" >&2
    echo "       Did the sync from site-packages back to mirror run? See 08_SYNC.md §3." >&2
    exit 1
fi

echo "[2/3] Deploying patched w8a8_int8.py ..."
# Use install -p to preserve attributes
install -p -m 0644 "$W8A8_SRC" "$W8A8_DST"
# Clear __pycache__ to avoid stale bytecode shadowing the patch
rm -rf "$SGLANG_QUANT_DIR/__pycache__"
echo "      Deployed at: $W8A8_DST"
echo

# ---------------------------------------------------------------------------
# Step 3: verify install
# ---------------------------------------------------------------------------
echo "[3/3] Verifying install ..."
# IMPORTANT: cd to /tmp before importing, otherwise Python prepends the
# current directory to sys.path and finds sglang_kernel_v10/__init__.py
# inside the PPU mirror (which has no _C.so) before site-packages'.
VERIFY_DIR=$(mktemp -d)
cd "$VERIFY_DIR"
python3 << 'PYEOF'
import sys, torch
try:
    import sglang_kernel_v10
    print(f'  ✓ sglang_kernel_v10 v{sglang_kernel_v10.__version__} import OK')
    print(f'      from: {sglang_kernel_v10.__file__}')
except ImportError as e:
    print(f'  ✗ sglang_kernel_v10 import failed: {e}')
    sys.exit(1)

try:
    from sglang.srt.layers.quantization.w8a8_int8 import (
        gemv_int8_i8i8bf16, _ISA_V10_ENABLE, _ISA_V10_AVAILABLE
    )
    print(f'  ✓ w8a8_int8 dispatcher import OK (v10_enabled={_ISA_V10_ENABLE}, available={_ISA_V10_AVAILABLE})')
except ImportError as e:
    print(f'  ✗ w8a8_int8 import failed: {e}')
    sys.exit(1)

# Correctness: K=2048 N=8192 M=1 with sglang-like (N, K) view layout
# (B is what sglang passes: layer.weight.t() = (N, K) shape, column-major)
torch.manual_seed(0)
A = torch.randint(-128, 127, (1, 2048), dtype=torch.int8, device='cuda')
W_storage_kn = torch.randint(-128, 127, (2048, 8192), dtype=torch.int8, device='cuda')
B = W_storage_kn.t()  # (8192, 2048) view, strides (1, 8192) — what sglang passes
As = torch.ones(1, dtype=torch.float32, device='cuda')
Bs = torch.ones(8192, dtype=torch.float32, device='cuda')
out = torch.empty(1, 8192, dtype=torch.bfloat16, device='cuda')
gemv_int8_i8i8bf16(A, As, B, Bs, out)
ref = (A.float() @ W_storage_kn.float()).to(torch.bfloat16)
diff = (out.float() - ref.float()).abs().max().item()
print(f'  ✓ correctness: dispatcher vs ref max diff = {diff} (expect 0.0)')

if diff > 0.5:
    print(f'  ✗ correctness FAILED — likely F10 dispatcher bug regressed')
    sys.exit(1)

print()
print('==================================================================')
print('Install complete. v10 fast path is active for K=2048 M=1 decode.')
print('==================================================================')
PYEOF
rm -rf "$VERIFY_DIR"