"""Fused FlashAttention-style forward kernel in Triton.

The whole point: compute ``softmax(QKᵀ · scale) · V`` without ever writing the
N×N score matrix to HBM. We tile the sequence and use the **online-softmax**
recurrence so each query block is reduced against streamed key/value blocks
entirely inside on-chip SRAM. See ``docs/design.md`` for the math.

Tensors are ``(batch, heads, seqlen, head_dim)``, contiguous.

The launch knobs ``BLOCK_M / BLOCK_N / num_warps / num_stages`` are exposed on
:func:`flash_attention` because they are exactly what the Phase-4 profiling loop
tunes (tile size = SRAM reuse, ``num_stages`` = software-pipelining depth).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_fwd_kernel(
    Q, K, V, O,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    N,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # One program = one block of BLOCK_M queries, for one (batch, head).
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # query row indices
    offs_n = tl.arange(0, BLOCK_N)                       # key col indices (per tile)
    offs_d = tl.arange(0, BLOCK_D)                       # head-dim indices

    # Base pointers into this (batch, head) slice.
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh
    o_base = O + off_b * stride_ob + off_h * stride_oh

    # Load this query block once; it stays resident in SRAM for the whole loop.
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)  # (BLOCK_M, BLOCK_D)

    # Online-softmax running state, kept in fp32 for stability:
    #   m_i = running row max, l_i = running denominator, acc = running output.
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Causal: a query at row i only attends to keys j <= i, so we never need key
    # blocks that start past this query block's last row.
    hi = (start_m + 1) * BLOCK_M if CAUSAL else N

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        key_idx = start_n + offs_n  # absolute key indices for this tile

        # K is loaded transposed (BLOCK_D, BLOCK_N) so tl.dot gives QKᵀ directly.
        k_ptrs = k_base + key_idx[None, :] * stride_kn + offs_d[:, None] * stride_kd
        v_ptrs = v_base + key_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=key_idx[None, :] < N, other=0.0)  # (BLOCK_D, BLOCK_N)
        v = tl.load(v_ptrs, mask=key_idx[:, None] < N, other=0.0)  # (BLOCK_N, BLOCK_D)

        qk = tl.dot(q, k) * sm_scale  # (BLOCK_M, BLOCK_N), fp32 accumulate

        # Mask invalid keys: out-of-range padding and (if causal) the future.
        valid = key_idx[None, :] < N
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= key_idx[None, :])
        qk = tl.where(valid, qk, float("-inf"))

        # --- online-softmax update -------------------------------------------
        m_ij = tl.max(qk, axis=1)            # this tile's row max
        m_new = tl.maximum(m_i, m_ij)        # new running max
        p = tl.exp(qk - m_new[:, None])      # rescaled probabilities for this tile
        alpha = tl.exp(m_i - m_new)          # factor to rescale prior state
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]  # finalize the softmax denominator
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N)


class FlashAttnFunc(torch.autograd.Function):
    """Autograd wrapper. Forward uses the Triton kernel; backward currently
    defers to PyTorch autograd (recompute). The ``backward`` body is the clean
    seam where a real fused backward kernel drops in later — nothing else needs
    to change."""

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, block_m, block_n, num_warps, num_stages):
        B, H, N, D = q.shape
        assert q.is_cuda and q.is_contiguous(), "expect contiguous CUDA tensors"
        assert D in (16, 32, 64, 128), f"head_dim {D} must be a power of two <= 128"

        o = torch.empty_like(q)
        grid = (triton.cdiv(N, block_m), B * H)
        _flash_fwd_kernel[grid](
            q, k, v, o,
            sm_scale,
            *q.stride(), *k.stride(), *v.stride(), *o.stride(),
            N,
            H=H,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=D,
            CAUSAL=causal,
            num_warps=num_warps, num_stages=num_stages,
        )

        ctx.save_for_backward(q, k, v)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        # Phase-2 fallback: recompute the forward under PyTorch autograd and let
        # it produce dQ, dK, dV. Correct and self-contained; to be replaced by a
        # fused Triton backward kernel without touching the public API.
        import torch.nn.functional as F

        q, k, v = ctx.saved_tensors
        with torch.enable_grad():
            qd = q.detach().requires_grad_(True)
            kd = k.detach().requires_grad_(True)
            vd = v.detach().requires_grad_(True)
            o = F.scaled_dot_product_attention(qd, kd, vd, is_causal=ctx.causal, scale=ctx.sm_scale)
            dq, dk, dv = torch.autograd.grad(o, (qd, kd, vd), do)
        # grads line up with forward()'s args; non-tensor args get None.
        return dq, dk, dv, None, None, None, None, None, None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    sm_scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Fused causal attention. Drop-in for :func:`forge.reference.naive_attention`.

    ``block_m/block_n/num_warps/num_stages`` are the tuning knobs profiled in
    Phase 4; the defaults are a sane A100 fp16 starting point.
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    return FlashAttnFunc.apply(
        q, k, v, causal, sm_scale, block_m, block_n, num_warps, num_stages
    )
