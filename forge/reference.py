"""PyTorch attention baselines.

Two reference implementations of causal multi-head attention, both operating on
tensors of shape ``(batch, heads, seqlen, head_dim)``:

* :func:`naive_attention` — the textbook unfused path that materializes the full
  N×N score matrix in HBM. This is the baseline the fused Triton kernel must beat.
* :func:`sdpa_attention` — PyTorch's built-in ``scaled_dot_product_attention``,
  itself a fused/Flash implementation. Used as a fast, trusted correctness oracle
  and a strong reference point for benchmarks.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True):
    """Unfused causal attention: QKᵀ → mask → softmax → ·V.

    Materializes the (..., N, N) score matrix explicitly — this is exactly the
    HBM round-trip the fused kernel eliminates.
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, N, N)

    if causal:
        n = q.size(-2)
        mask = torch.triu(torch.ones(n, n, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True):
    """PyTorch's fused scaled-dot-product attention — correctness oracle."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def make_qkv(
    batch: int,
    heads: int,
    seqlen: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    seed: int = 0,
):
    """Make a random (Q, K, V) triple of shape (batch, heads, seqlen, head_dim)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (batch, heads, seqlen, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype, generator=gen)
    k = torch.randn(shape, device=device, dtype=dtype, generator=gen)
    v = torch.randn(shape, device=device, dtype=dtype, generator=gen)
    return q, k, v
