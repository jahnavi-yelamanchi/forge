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
def _attend(
    acc, l_i, m_i,
    q,
    k_base, v_base,
    stride_kn, stride_kd, stride_vn, stride_vd,
    offs_m, offs_n, offs_d,
    sm_scale, N,
    lo, hi,
    BLOCK_N: tl.constexpr,
    MASK: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Fold key/value tiles in ``[lo, hi)`` into the running softmax state.

    ``MASK=False`` is the fast path for tiles that are guaranteed in-range and
    (under causality) fully below the diagonal — no bounds check, no ``tl.where``.
    ``MASK=True`` handles the diagonal tile (causal compare) and any ragged tail
    (bounds check).
    """
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        key_idx = start_n + offs_n

        k_ptrs = k_base + key_idx[None, :] * stride_kn + offs_d[:, None] * stride_kd
        v_ptrs = v_base + key_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd
        if MASK:
            k = tl.load(k_ptrs, mask=key_idx[None, :] < N, other=0.0)
            v = tl.load(v_ptrs, mask=key_idx[:, None] < N, other=0.0)
        else:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)

        qk = tl.dot(q, k) * sm_scale  # (BLOCK_M, BLOCK_N), fp32
        if MASK:
            valid = key_idx[None, :] < N
            if CAUSAL:
                valid = valid & (offs_m[:, None] >= key_idx[None, :])
            qk = tl.where(valid, qk, float("-inf"))

        # online-softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _flash_fwd_kernel(
    Q, K, V, O, L,
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

    # Two-phase causal loop. A query at row i only attends to keys j <= i.
    #   * Full blocks  [0, diag)         are strictly below the diagonal for every
    #     row in this query block -> no masking needed (MASK=False fast path).
    #   * Diagonal block(s) [diag, end)  straddle the diagonal -> causal mask.
    # (Requires BLOCK_M % BLOCK_N == 0 so `diag` is BLOCK_N-aligned; enforced in
    # the wrapper.) Non-causal is a single masked pass over all keys.
    if CAUSAL:
        diag = start_m * BLOCK_M
        acc, l_i, m_i = _attend(
            acc, l_i, m_i, q, k_base, v_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_n, offs_d, sm_scale, N,
            0, diag, BLOCK_N, MASK=False, CAUSAL=True,
        )
        acc, l_i, m_i = _attend(
            acc, l_i, m_i, q, k_base, v_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_n, offs_d, sm_scale, N,
            diag, (start_m + 1) * BLOCK_M, BLOCK_N, MASK=True, CAUSAL=True,
        )
    else:
        acc, l_i, m_i = _attend(
            acc, l_i, m_i, q, k_base, v_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_n, offs_d, sm_scale, N,
            0, N, BLOCK_N, MASK=True, CAUSAL=False,
        )

    acc = acc / l_i[:, None]  # finalize the softmax denominator
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N)

    # Save the log-sum-exp per query row so the backward pass can rebuild the
    # softmax probabilities (P = exp(S - L)) without recomputing the row max.
    tl.store(L + off_bh * N + offs_m, m_i + tl.log(l_i), mask=offs_m < N)


# ---------------------------------------------------------------------------
# Backward pass
#
# FlashAttention backward recomputes the softmax probabilities tile-wise from the
# saved log-sum-exp (no N×N matrix in HBM, same as forward) and produces dQ, dK,
# dV. Using the identities (with S = scale·QKᵀ, P = softmax(S) = exp(S − L)):
#   dV = Pᵀ·dO,  dP = dO·Vᵀ,  D = rowsum(dO∘O),  dS = P∘(dP − D),
#   dQ = scale·dS·K,  dK = scale·dSᵀ·Q.
# dQ accumulates over key blocks while dK/dV accumulate over query blocks, so we
# use two kernels (parallel over query- vs key-blocks) to avoid atomics.
# ---------------------------------------------------------------------------


@triton.jit
def _bwd_preprocess(
    O, DO, Delta,
    stride_ob, stride_oh, stride_om, stride_od,
    N,
    H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Delta[b,h,m] = sum_d O·dO — the softmax-jacobian correction term."""
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b, off_h = off_bh // H, off_bh % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    base = off_b * stride_ob + off_h * stride_oh
    ptrs = base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o = tl.load(O + ptrs, mask=offs_m[:, None] < N, other=0.0).to(tl.float32)
    do = tl.load(DO + ptrs, mask=offs_m[:, None] < N, other=0.0).to(tl.float32)
    tl.store(Delta + off_bh * N + offs_m, tl.sum(o * do, axis=1), mask=offs_m < N)


@triton.jit
def _bwd_dkdv_kernel(
    Q, K, V, DO, DK, DV, L, Delta, sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    N,
    H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr, CAUSAL: tl.constexpr,
):
    """One key/value block; accumulate dK, dV by looping over query blocks."""
    start_n = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b, off_h = off_bh // H, off_bh % H
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = off_b * stride_qb + off_h * stride_qh
    k_base = off_b * stride_kb + off_h * stride_kh

    kv_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    k = tl.load(K + kv_ptrs, mask=offs_n[:, None] < N, other=0.0)  # (BLOCK_N, D)
    v = tl.load(V + kv_ptrs, mask=offs_n[:, None] < N, other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    # Causal: only queries m >= n contribute; start at the block containing n.
    lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M if CAUSAL else 0
    for start_m in range(lo, N, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        qd_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q = tl.load(Q + qd_ptrs, mask=offs_m[:, None] < N, other=0.0)
        do = tl.load(DO + qd_ptrs, mask=offs_m[:, None] < N, other=0.0)
        l_i = tl.load(L + off_bh * N + offs_m, mask=offs_m < N, other=0.0)
        delta = tl.load(Delta + off_bh * N + offs_m, mask=offs_m < N, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale               # (BLOCK_M, BLOCK_N)
        p = tl.exp(qk - l_i[:, None])
        keep = (offs_m[:, None] < N) & (offs_n[None, :] < N)
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(keep, p, 0.0)

        dv += tl.dot(tl.trans(p).to(do.dtype), do)           # Pᵀ·dO
        dp = tl.dot(do, tl.trans(v))                         # dO·Vᵀ
        ds = p * (dp - delta[:, None])                        # P∘(dP − D)
        dk += tl.dot(tl.trans(ds).to(q.dtype), q)            # dSᵀ·Q

    dk *= sm_scale
    tl.store(DK + kv_ptrs, dk.to(DK.dtype.element_ty), mask=offs_n[:, None] < N)
    tl.store(DV + kv_ptrs, dv.to(DV.dtype.element_ty), mask=offs_n[:, None] < N)


@triton.jit
def _bwd_dq_kernel(
    Q, K, V, DO, DQ, L, Delta, sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    N,
    H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr, CAUSAL: tl.constexpr,
):
    """One query block; accumulate dQ by looping over key/value blocks."""
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b, off_h = off_bh // H, off_bh % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = off_b * stride_qb + off_h * stride_qh
    k_base = off_b * stride_kb + off_h * stride_kh

    qd_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(Q + qd_ptrs, mask=offs_m[:, None] < N, other=0.0)
    do = tl.load(DO + qd_ptrs, mask=offs_m[:, None] < N, other=0.0)
    l_i = tl.load(L + off_bh * N + offs_m, mask=offs_m < N, other=0.0)
    delta = tl.load(Delta + off_bh * N + offs_m, mask=offs_m < N, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    hi = (start_m + 1) * BLOCK_M if CAUSAL else N
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(K + kv_ptrs, mask=offs_n[:, None] < N, other=0.0)
        v = tl.load(V + kv_ptrs, mask=offs_n[:, None] < N, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.exp(qk - l_i[:, None])
        keep = (offs_m[:, None] < N) & (offs_n[None, :] < N)
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(keep, p, 0.0)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)                      # dS·K

    dq *= sm_scale
    tl.store(DQ + qd_ptrs, dq.to(DQ.dtype.element_ty), mask=offs_m[:, None] < N)


class FlashAttnFunc(torch.autograd.Function):
    """Autograd wrapper: fused Triton kernels for both forward and backward."""

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, block_m, block_n, num_warps, num_stages):
        B, H, N, D = q.shape
        assert q.is_cuda and q.is_contiguous(), "expect contiguous CUDA tensors"
        assert D in (16, 32, 64, 128), f"head_dim {D} must be a power of two <= 128"
        assert block_m % block_n == 0, "two-phase causal loop needs BLOCK_M % BLOCK_N == 0"

        o = torch.empty_like(q)
        lse = torch.empty((B, H, N), device=q.device, dtype=torch.float32)  # log-sum-exp
        grid = (triton.cdiv(N, block_m), B * H)
        _flash_fwd_kernel[grid](
            q, k, v, o, lse,
            sm_scale,
            *q.stride(), *k.stride(), *v.stride(), *o.stride(),
            N,
            H=H,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=D,
            CAUSAL=causal,
            num_warps=num_warps, num_stages=num_stages,
        )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        ctx.block_m, ctx.block_n = block_m, block_n
        ctx.num_warps, ctx.num_stages = num_warps, num_stages
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        do = do.contiguous()
        B, H, N, D = q.shape
        bm, bn, causal, scale = ctx.block_m, ctx.block_n, ctx.causal, ctx.sm_scale

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

        _bwd_preprocess[(triton.cdiv(N, bm), B * H)](
            o, do, delta, *o.stride(), N, H=H, BLOCK_M=bm, BLOCK_D=D,
        )
        _bwd_dkdv_kernel[(triton.cdiv(N, bn), B * H)](
            q, k, v, do, dk, dv, lse, delta, scale,
            *q.stride(), *k.stride(), N,
            H=H, BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=D, CAUSAL=causal,
        )
        _bwd_dq_kernel[(triton.cdiv(N, bm), B * H)](
            q, k, v, do, dq, lse, delta, scale,
            *q.stride(), *k.stride(), N,
            H=H, BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=D, CAUSAL=causal,
        )
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
    num_stages: int = 3,
) -> torch.Tensor:
    """Fused causal attention. Drop-in for :func:`forge.reference.naive_attention`.

    ``block_m/block_n/num_warps/num_stages`` are the tuning knobs profiled in
    Phase 4. Defaults are the empirical A100 fp16 winner from the config sweep
    (see docs/profiling.md): 64×64 tiles keep occupancy high at head_dim=64, and
    ``num_stages=3`` deepens the software pipeline so the next K/V tile is
    prefetched while the current tile's matmuls run (up to 1.28× vs num_stages=2).
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    return FlashAttnFunc.apply(
        q, k, v, causal, sm_scale, block_m, block_n, num_warps, num_stages
    )
