"""Fused linear + GELU kernel for the GPT-2 FFN.

GPT-2's MLP is ``c_proj(gelu(c_fc(x)))``. The ``c_fc`` step is a matmul that
produces a 4×d-wide intermediate, which the activation then reads back from HBM
and rewrites. This module fuses the **GELU into the matmul epilogue**: each output
tile is activated in registers and written once, so the pre-activation never makes
a round-trip through HBM.

Scope mirrors the attention Phase-2 seam: the forward is a fused Triton GEMM;
backward recomputes via PyTorch autograd (a clean seam for a fused backward GEMM).
So this is an **inference-forward** optimization — benchmarked as such.

GELU uses the tanh approximation (as in nanoGPT), implemented with
``tanh(z) = 2·sigmoid(2z) − 1`` to avoid libdevice dependencies.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_GELU_C = 0.7978845608028654  # sqrt(2/pi)


@triton.jit
def _linear_gelu_kernel(
    X, W, Bias, Y,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Y = gelu(X @ W + bias), tiled GEMM with a GELU epilogue.

    Program ids are swizzled into GROUP_M-row super-blocks so nearby programs
    reuse the same K/W rows in L2 (the standard Triton GEMM ordering).
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # tanh-approx GELU, computed in fp32 in-register (the fusion payoff).
    # 0.7978845608028654 = sqrt(2/pi) (inlined; Triton can't read module globals).
    inner = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
    tanh = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    y = 0.5 * acc * (1.0 + tanh)

    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y.to(Y.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class FusedLinearGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, block_m, block_n, block_k, group_m, num_warps, num_stages):
        # x: (M, K); weight: (N, K) as stored by nn.Linear; bias: (N,)
        M, K = x.shape
        N = weight.shape[0]
        assert x.is_cuda and x.is_contiguous()
        wt = weight.t().contiguous()  # (K, N) for the GEMM
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
        _linear_gelu_kernel[grid](
            x, wt, bias, y,
            M, N, K,
            x.stride(0), x.stride(1),
            wt.stride(0), wt.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
            num_warps=num_warps, num_stages=num_stages,
        )
        ctx.save_for_backward(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, dy):
        # Recompute via autograd (clean seam for a future fused backward GEMM).
        x, weight, bias = ctx.saved_tensors
        with torch.enable_grad():
            xi = x.detach().requires_grad_(True)
            wi = weight.detach().requires_grad_(True)
            bi = bias.detach().requires_grad_(True)
            y = F.gelu(F.linear(xi, wi, bi), approximate="tanh")
            dx, dw, db = torch.autograd.grad(y, (xi, wi, bi), dy)
        return dx, dw, db, None, None, None, None, None, None


def fused_linear_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    group_m: int = 8,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    """gelu(x @ weightᵀ + bias) with the GELU fused into the matmul epilogue.

    ``x`` is 2D ``(tokens, in_features)``; ``weight``/``bias`` are an
    ``nn.Linear``'s parameters. Matches ``F.gelu(F.linear(x, w, b), 'tanh')``.
    Defaults are a standard A100 fp16 GEMM config (128×128 tiles, GROUP_M=8 for
    L2 reuse, 4-stage pipeline).
    """
    return FusedLinearGELU.apply(
        x, weight, bias, block_m, block_n, block_k, group_m, num_warps, num_stages
    )
