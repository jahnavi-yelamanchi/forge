"""Phase-4 profiling + autotuning of the fused kernel.

Two jobs:

1. :func:`sweep_configs` — grid-search the launch knobs (``BLOCK_M``, ``BLOCK_N``,
   ``num_warps``, ``num_stages``) at a representative shape and report the speedup
   of the best config over the untuned default. This is the tuning that actually
   makes the kernel faster; the knobs map directly to SRAM reuse (tile size),
   occupancy (warps), and software-pipelining / prefetch depth (stages).

2. :func:`profile_timeline` — run ``torch.profiler`` on the default vs best config
   and return the on-GPU kernel time, as profiler evidence for the writeup.

Invalid configs (e.g. tiles that overflow shared memory) are skipped gracefully.
"""

from __future__ import annotations

import itertools

import torch

from forge import utils
from forge.flash_attn import flash_attention
from forge.reference import make_qkv

# The untuned Phase-3 launch config, used as the speedup reference.
DEFAULT_CFG = dict(block_m=64, block_n=64, num_warps=4, num_stages=2)

# Search grid. head_dim=64 on A100 comfortably fits larger M tiles and deeper
# pipelines; we let the sweep confirm which actually wins.
BLOCK_M = [64, 128]
BLOCK_N = [32, 64, 128]
NUM_WARPS = [4, 8]
NUM_STAGES = [2, 3, 4]


def _configs():
    for bm, bn, w, s in itertools.product(BLOCK_M, BLOCK_N, NUM_WARPS, NUM_STAGES):
        yield dict(block_m=bm, block_n=bn, num_warps=w, num_stages=s)


def _time_cfg(q, k, v, cfg, warmup, iters) -> float | None:
    """Median latency for one config, or None if the config is invalid on this GPU."""
    fn = lambda: flash_attention(q, k, v, causal=True, **cfg)
    try:
        fn()  # triggers JIT compile; raises here if SMEM/regs don't fit
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - invalid launch configs are expected
        print(f"    skip {cfg}: {type(e).__name__}")
        return None
    return utils.bench_ms(fn, warmup=warmup, iters=iters)


def sweep_configs(shape, dtype=torch.float16, warmup=10, iters=30) -> dict:
    """Grid-search configs at one shape. Returns default/best timings + all rows."""
    B, H, N, D = shape
    q, k, v = make_qkv(B, H, N, D, dtype=dtype)

    default_ms = _time_cfg(q, k, v, DEFAULT_CFG, warmup, iters)
    print(f"  default {DEFAULT_CFG}: {default_ms:.4f} ms")

    rows = []
    for cfg in _configs():
        ms = _time_cfg(q, k, v, cfg, warmup, iters)
        if ms is None:
            continue
        rows.append({**cfg, "latency_ms": round(ms, 4), "speedup_vs_default": round(default_ms / ms, 3)})
        print(f"    {cfg}: {ms:.4f} ms  ({default_ms / ms:.2f}x)")

    best = min(rows, key=lambda r: r["latency_ms"])
    print(f"\n  BEST @ N={N}: {best}  ->  {best['speedup_vs_default']}x over default")
    return {
        "shape": {"batch": B, "heads": H, "seqlen": N, "head_dim": D},
        "default_ms": round(default_ms, 4),
        "best": best,
        "rows": rows,
    }


def profile_timeline(shape, cfg, dtype=torch.float16, active=20) -> dict:
    """torch.profiler CUDA-time for `cfg` at `shape`. Returns kernel-time summary."""
    from torch.profiler import ProfilerActivity, profile

    B, H, N, D = shape
    q, k, v = make_qkv(B, H, N, D, dtype=dtype)
    fn = lambda: flash_attention(q, k, v, causal=True, **cfg)
    for _ in range(10):  # warmup / compile
        fn()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(active):
            fn()
        torch.cuda.synchronize()

    total_us = sum(e.self_device_time_total for e in prof.key_averages())
    return {"cfg": cfg, "gpu_ms_per_call": round(total_us / active / 1000, 4)}
