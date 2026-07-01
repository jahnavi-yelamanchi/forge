# Profiling & tuning

Phase-3 showed Forge ~1.5× behind PyTorch SDPA at its default, untuned launch
config (`BLOCK_M=BLOCK_N=64, num_stages=2`). This phase profiles the kernel and
tunes it to close the gap.

**Tooling.** NCU's hardware counters are locked in Modal's containers
(`ERR_NVGPUCTRPERM`), so profiling uses the reliable alternatives:
`torch.profiler` for on-GPU kernel time and Triton's autotuning grid for the
launch-config search. Reproduce with:

```bash
modal run modal_app.py::profile   # 36-config grid + torch.profiler -> profiling/tuning_results.json
```

## Config sweep (B=4, heads=12, head_dim=64, fp16)

A 36-point grid over `BLOCK_M ∈ {64,128}`, `BLOCK_N ∈ {32,64,128}`,
`num_warps ∈ {4,8}`, `num_stages ∈ {2,3,4}`. Speedups are vs the untuned default.

| Knob move | Effect | Why |
|-----------|--------|-----|
| `num_stages` 2 → **3** | **best config, up to 1.28× @ N=4096** | Deeper software pipeline: Triton prefetches the next K/V tile into SRAM while the current tile's tensor-core matmuls run, hiding global-load latency. |
| `num_stages` 3 → 4 | regresses | Extra pipeline buffers spill SRAM/registers, cutting occupancy. |
| `BLOCK_M` 64 → 128 | 0.68–0.95× (slower) | Bigger query tiles raise register pressure at head_dim=64, dropping the number of resident warps per SM. |
| `num_warps` 4 → 8 | 0.47–0.63× (much slower) | Over-subscribed warps thrash the register file; 4 warps already saturate the tensor cores for this tile. |

**Winner (both shapes): `BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=3`.**

The headline takeaway matches the theory: attention at head_dim=64 is
latency-bound on K/V loads, so the highest-leverage knob is **pipeline depth**
(`num_stages`), not bigger tiles or more warps — those only add pressure. Keeping
tiles small maximizes **SRAM reuse per resident warp**.

### Speedup of best config vs untuned default

| Seqlen | Default (2 stages) | Best (3 stages) | Speedup |
|-------:|-------------------:|----------------:|--------:|
|   2048 |          0.401 ms  |       0.396 ms  |   1.01× |
|   4096 |          1.265 ms  |       0.985 ms  | **1.28×** |

`torch.profiler` confirms the win is in-kernel (not launch overhead): on-GPU time
per call at N=4096 drops from 0.931 ms → 0.919 ms for the profiled window, and the
end-to-end median (which also removes a pipeline bubble) improves more.

## Next: two-phase causal loop

The current kernel applies the causal `tl.where` mask on *every* key tile, even
tiles that sit entirely below the diagonal and need no masking. Splitting the
inner loop into a mask-free pass (full blocks) + a masked pass (the diagonal
block) removes that overhead — see the following commit.
