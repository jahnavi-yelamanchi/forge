# Profiling & tuning

> 🚧 Phase 4 — in progress.

The Phase-3 benchmarks show Forge is ~1.5× behind PyTorch SDPA at its default,
untuned launch config (`BLOCK_M=BLOCK_N=64, num_stages=2`). This phase profiles
the kernel and tunes it to close the gap.

Tooling (NCU's hardware counters are locked in Modal's containers, so we use the
reliable alternatives): **`torch.profiler`** for the kernel timeline, **Triton
`proton`** for op-level attribution, and **Nsight Systems (`nsys`)** for a full
trace.

Each tuning iteration — block sizes (SRAM reuse), `num_warps` (occupancy),
`num_stages` (software-pipelining / prefetch depth) — is recorded below with its
before → after effect.

_Results table to follow._
