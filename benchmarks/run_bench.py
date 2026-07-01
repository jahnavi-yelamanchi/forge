"""Benchmark sweep: fused vs naive vs SDPA across batch sizes and seqlens.

Produces one row of metrics per (config, implementation):
latency, speedup vs the naive baseline, compute throughput (TFLOP/s), and HBM
traffic. The actual GPU run is driven from ``modal_app.py::run_bench``; this
module is import-clean so the timing logic lives next to the kernel code.

Two sweeps are tagged in the output:
  * ``seqlen`` — fixed batch/heads, vary sequence length (the headline story).
  * ``batch``  — fixed seqlen, vary batch size.
"""

from __future__ import annotations

import torch

from forge import utils
from forge.flash_attn import flash_attention
from forge.reference import make_qkv, naive_attention, sdpa_attention

# GPT-2 small geometry: 12 heads × 64 head-dim.
HEADS = 12
HEAD_DIM = 64

# (sweep_name, batch, heads, seqlen, head_dim)
SEQLEN_SWEEP = [("seqlen", 4, HEADS, n, HEAD_DIM) for n in (512, 1024, 2048, 4096)]
BATCH_SWEEP = [("batch", b, HEADS, 1024, HEAD_DIM) for b in (1, 2, 4, 8)]
CONFIGS = SEQLEN_SWEEP + BATCH_SWEEP

IMPLS = {
    "naive": naive_attention,
    "sdpa": sdpa_attention,
    "fused": flash_attention,
}

# Per-implementation HBM traffic model (see forge/utils.py). The naive path also
# streams the N×N score matrix; the fused/SDPA paths keep it on chip.
BYTES_FN = {
    "naive": utils.naive_attention_bytes,
    "sdpa": utils.fused_attention_bytes,
    "fused": utils.fused_attention_bytes,
}


def run_sweep(dtype: torch.dtype = torch.float16, warmup: int = 10, iters: int = 30) -> list[dict]:
    """Run the full sweep on the current CUDA device, return metric rows."""
    assert torch.cuda.is_available(), "CUDA required for benchmarking"
    utils.set_seed(0)
    dtype_bytes = torch.finfo(dtype).bits // 8
    rows: list[dict] = []

    for sweep, B, H, N, D in CONFIGS:
        q, k, v = make_qkv(B, H, N, D, dtype=dtype)
        flops = utils.attention_flops(B, H, N, D, causal=True)

        # Time the naive baseline first so we can express speedups against it.
        timings = {}
        for name, fn in IMPLS.items():
            ms = utils.bench_ms(lambda fn=fn: fn(q, k, v, causal=True), warmup=warmup, iters=iters)
            timings[name] = ms

        base = timings["naive"]
        for name in IMPLS:
            ms = timings[name]
            moved = BYTES_FN[name](B, H, N, D, dtype_bytes)
            rows.append(
                {
                    "sweep": sweep,
                    "impl": name,
                    "batch": B,
                    "heads": H,
                    "seqlen": N,
                    "head_dim": D,
                    "dtype": str(dtype).replace("torch.", ""),
                    "latency_ms": round(ms, 4),
                    "speedup_vs_naive": round(base / ms, 3),
                    "tflops": round(utils.tflops(flops, ms), 2),
                    "achieved_gbps": round(utils.achieved_gbps(moved, ms), 1),
                    "hbm_gb": round(moved / 1e9, 4),
                }
            )
            print(
                f"  [{sweep:6s}] {name:5s} B={B} N={N:5d}: "
                f"{ms:8.3f} ms  {base / ms:5.2f}x  {utils.tflops(flops, ms):6.1f} TFLOP/s"
            )

    return rows
