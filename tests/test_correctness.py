"""Correctness tests for the attention implementations.

In Phase 1 this locks the two baselines (naive vs SDPA) against each other and
establishes the shapes + tolerances the fused Triton kernel must later satisfy.
When the fused kernel lands, it gets added to ``IMPLS`` and tested by the same
parametrized suite for free.

Run on the GPU via:  ``modal run modal_app.py::run_tests``
"""

from __future__ import annotations

import itertools

import pytest
import torch

from forge.flash_attn import flash_attention
from forge.reference import make_qkv, naive_attention, sdpa_attention
from forge.utils import compare

# (batch, heads, seqlen, head_dim) — a spread of small/large, square/non-square.
SHAPES = [
    (1, 4, 128, 64),
    (2, 8, 512, 64),
    (4, 12, 1024, 64),
    (1, 16, 2048, 64),
]
DTYPES = [torch.float16, torch.bfloat16]

# Implementations under test, keyed by name. The fused Triton kernel is appended
# here in Phase 2 and inherits this whole test matrix.
IMPLS = {
    "naive": naive_attention,
    "fused": flash_attention,
}

# fp16/bf16 attention accumulates a lot of terms; these tolerances track what
# two legitimate fused/unfused implementations actually differ by.
TOL = {
    torch.float16: dict(atol=2e-2, rtol=2e-2),
    torch.bfloat16: dict(atol=4e-2, rtol=4e-2),
}


@pytest.mark.parametrize("shape,dtype,impl_name", list(itertools.product(SHAPES, DTYPES, IMPLS)))
def test_matches_sdpa(shape, dtype, impl_name):
    """Each implementation must match PyTorch SDPA within tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    q, k, v = make_qkv(*shape, dtype=dtype)
    ref = sdpa_attention(q, k, v, causal=True)
    out = IMPLS[impl_name](q, k, v, causal=True)

    res = compare(out, ref, **TOL[dtype])
    assert res.passed, (
        f"{impl_name} {shape} {dtype}: max_abs={res.max_abs_err:.4g} "
        f"max_rel={res.max_rel_err:.4g}"
    )


def test_fused_backward_wires_up():
    """The autograd seam must produce finite dQ/dK/dV of the right shape.

    Backward currently defers to PyTorch (Phase 2), so this guards the wiring,
    not the math — a fused backward kernel will harden it later.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    q, k, v = make_qkv(2, 4, 256, 64, dtype=torch.float16)
    q, k, v = (t.clone().requires_grad_(True) for t in (q, k, v))

    out = flash_attention(q, k, v, causal=True)
    out.sum().backward()

    for name, t in (("dQ", q.grad), ("dK", k.grad), ("dV", v.grad)):
        assert t is not None, f"{name} is None"
        assert t.shape == q.shape, f"{name} wrong shape"
        assert torch.isfinite(t).all(), f"{name} has non-finite values"
