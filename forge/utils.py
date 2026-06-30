"""Shared utilities: timing, FLOP/bandwidth accounting, seeding, tolerances.

These helpers are deliberately framework-light so the same functions drive
correctness tests, the benchmark sweep, and the profiling runs — giving every
phase of the project consistent, comparable metrics.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch

# A100-SXM4-40GB datasheet peaks, used to report "% of peak" in benchmarks.
A100_HBM_BANDWIDTH_GBPS = 1555.0  # GB/s, HBM2e
A100_FP16_TFLOPS = 312.0  # TFLOP/s, tensor-core fp16 w/ fp32 accumulate


def set_seed(seed: int = 0) -> None:
    """Seed torch (CPU + CUDA) for reproducible inputs across runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bench_ms(fn, *, warmup: int = 10, iters: int = 50) -> float:
    """Median latency of `fn` in milliseconds, timed with CUDA events.

    Uses CUDA events (not wall-clock) so we time GPU work, not Python/launch
    overhead, and takes the median over `iters` to resist outliers.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return statistics.median(times)


def attention_flops(batch: int, heads: int, seqlen: int, head_dim: int, *, causal: bool) -> float:
    """Forward FLOPs for one attention call.

    Two matmuls dominate: QKᵀ and P·V, each 2·B·H·N²·d FLOPs (mul+add), so
    4·B·H·N²·d total. Causal masking computes ~half the score matrix.
    """
    flops = 4.0 * batch * heads * seqlen * seqlen * head_dim
    return flops * 0.5 if causal else flops


def naive_attention_bytes(
    batch: int, heads: int, seqlen: int, head_dim: int, dtype_bytes: int = 2
) -> float:
    """HBM bytes the *naive* path moves: dominated by writing+reading the N×N
    score matrix to/from HBM (the cost fusion eliminates)."""
    qkvo = 4 * batch * heads * seqlen * head_dim * dtype_bytes
    scores = 2 * batch * heads * seqlen * seqlen * dtype_bytes  # write then read
    return qkvo + scores


def fused_attention_bytes(
    batch: int, heads: int, seqlen: int, head_dim: int, dtype_bytes: int = 2
) -> float:
    """HBM bytes the *fused* path moves: only Q, K, V in and O out — the score
    matrix lives in SRAM and never touches HBM."""
    return 4 * batch * heads * seqlen * head_dim * dtype_bytes


def tflops(flops: float, ms: float) -> float:
    """Convert FLOPs + latency(ms) to TFLOP/s."""
    return flops / (ms * 1e-3) / 1e12


def achieved_gbps(bytes_moved: float, ms: float) -> float:
    """Effective HBM bandwidth in GB/s from bytes moved + latency(ms)."""
    return bytes_moved / (ms * 1e-3) / 1e9


@dataclass
class CorrectnessResult:
    max_abs_err: float
    max_rel_err: float
    passed: bool


def compare(out: torch.Tensor, ref: torch.Tensor, *, atol: float, rtol: float) -> CorrectnessResult:
    """Compare a kernel output against a reference within (atol, rtol).

    Errors are computed in fp32 to avoid the comparison itself losing precision.
    """
    o, r = out.float(), ref.float()
    abs_err = (o - r).abs()
    max_abs = float(abs_err.max().item())
    max_rel = float((abs_err / (r.abs() + 1e-6)).max().item())
    passed = bool(torch.allclose(o, r, atol=atol, rtol=rtol))
    return CorrectnessResult(max_abs_err=max_abs, max_rel_err=max_rel, passed=passed)
