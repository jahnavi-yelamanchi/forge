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


@pytest.mark.parametrize("shape", [(2, 4, 256, 64), (1, 8, 512, 64), (2, 12, 1024, 64)])
def test_fused_backward_matches_sdpa(shape):
    """The fused backward kernels must produce dQ/dK/dV matching SDPA autograd."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    q, k, v = make_qkv(*shape, dtype=torch.float16)
    grad = torch.randn_like(q)  # shared upstream gradient

    def grads_of(attn_fn):
        qi, ki, vi = (t.clone().requires_grad_(True) for t in (q, k, v))
        out = attn_fn(qi, ki, vi, causal=True)
        out.backward(grad)
        return qi.grad, ki.grad, vi.grad

    ref = grads_of(sdpa_attention)
    got = grads_of(flash_attention)

    for name, g, r in zip(("dQ", "dK", "dV"), got, ref):
        res = compare(g, r, atol=3e-2, rtol=3e-2)
        assert res.passed, f"{name} {shape}: max_abs={res.max_abs_err:.4g} max_rel={res.max_rel_err:.4g}"


def test_gpt2_fused_matches_sdpa():
    """Swapping the fused kernel into GPT-2 must not change the model's logits."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from forge.gpt2 import GPT, GPT2Config

    cfg = GPT2Config(n_layer=2, n_head=4, n_embd=256, block_size=256, vocab_size=1000)
    model = GPT(cfg).cuda().half().eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 128), device="cuda")

    with torch.no_grad():
        model.set_attention_backend("sdpa")
        ref = model(idx)
        model.set_attention_backend("fused")
        out = model(idx)

    res = compare(out, ref, atol=3e-2, rtol=3e-2)
    assert res.passed, f"GPT-2 logits diverge: max_abs={res.max_abs_err:.4g}"


@pytest.mark.parametrize("tokens", [1024, 4096])
def test_fused_linear_gelu_matches_torch(tokens):
    """Fused linear+GELU must match F.gelu(F.linear(x), 'tanh')."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    import torch.nn as nn
    import torch.nn.functional as F

    from forge.mlp import fused_linear_gelu

    K = 768
    x = torch.randn(tokens, K, device="cuda", dtype=torch.float16)
    lin = nn.Linear(K, 4 * K).cuda().half()

    ref = F.gelu(lin(x), approximate="tanh")
    out = fused_linear_gelu(x, lin.weight, lin.bias)

    res = compare(out, ref, atol=2e-2, rtol=2e-2)
    assert res.passed, f"fused MLP {tokens}: max_abs={res.max_abs_err:.4g}"
