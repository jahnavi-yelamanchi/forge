"""Micro-benchmark for the fused linear+GELU FFN kernel.

Compares the fused Triton `c_fc + GELU` against PyTorch's
`F.gelu(F.linear(x), 'tanh')` (cuBLAS matmul + a fused GELU kernel) across token
counts, at GPT-2's d=768 -> 4d geometry. Reports latency, speedup, and the HBM
traffic the epilogue fusion saves on the 4d intermediate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge import utils
from forge.mlp import fused_linear_gelu

K = 768          # n_embd
N = 4 * K        # FFN hidden
TOKENS = [1024, 4096, 8192, 16384]  # batch*seqlen


def run_mlp_bench(dtype: torch.dtype = torch.float16) -> list[dict]:
    assert torch.cuda.is_available(), "CUDA required"
    utils.set_seed(0)
    dtype_bytes = torch.finfo(dtype).bits // 8
    rows = []

    for M in TOKENS:
        x = torch.randn(M, K, device="cuda", dtype=dtype)
        lin = nn.Linear(K, N).cuda().to(dtype)

        torch_fn = lambda: F.gelu(lin(x), approximate="tanh")
        fused_fn = lambda: fused_linear_gelu(x, lin.weight, lin.bias)

        # correctness gate before timing
        assert torch.allclose(fused_fn(), torch_fn(), atol=2e-2, rtol=2e-2)

        t_torch = utils.bench_ms(torch_fn, warmup=10, iters=50)
        t_fused = utils.bench_ms(fused_fn, warmup=10, iters=50)

        # Fusion avoids one read+write of the M×N intermediate through HBM.
        saved_gb = 2 * M * N * dtype_bytes / 1e9
        rows.append({
            "tokens": M,
            "torch_ms": round(t_torch, 4),
            "fused_ms": round(t_fused, 4),
            "speedup": round(t_torch / t_fused, 3),
            "intermediate_traffic_saved_gb": round(saved_gb, 4),
        })
        print(f"  M={M:6d}: torch {t_torch:7.3f} ms | fused {t_fused:7.3f} ms | "
              f"{t_torch / t_fused:.2f}x | saves {saved_gb:.3f} GB")

    return rows
