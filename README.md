# Forge — GPT-2 Kernel Fusion with Triton

> A FlashAttention-style **fused causal attention kernel** written in [Triton](https://triton-lang.org/),
> drop-in for GPT-2, benchmarked against PyTorch and profiled/tuned on an **NVIDIA A100**.

**Stack:** PyTorch · Triton · CUDA · Modal (A100) · Nsight Systems / `torch.profiler` / Triton `proton`

---

## Why

Standard attention materializes the full `N×N` score matrix in GPU HBM, then reads it
back for the softmax and the second matmul. That round-trip is memory-bandwidth bound and
wasteful. **Forge** fuses the whole `softmax(QKᵀ / √d) · V` pipeline into a single Triton
kernel using **online softmax** and **block tiling**, so the score matrix never leaves
on-chip SRAM. The result is a faster forward pass and a large drop in HBM traffic.

**Target results** (forward pass, A100, fp16): **≈1.83× speedup** and **~2× effective
memory bandwidth** vs. a standard PyTorch attention baseline.

> Status: 🚧 under construction — see the commit history for the build journey.

## How it runs

All GPU work executes on **Modal A100** (the dev machine has no CUDA). Local machine is
for editing, git, and orchestrating Modal runs.

```bash
modal run modal_app.py::smoke        # confirm A100 + torch + triton
modal run modal_app.py::run_tests    # correctness: fused vs PyTorch reference
modal run modal_app.py::run_bench    # benchmark sweep (batch × seqlen) -> CSV
modal run modal_app.py::run_profile  # profiler traces for the tuning loop
```

## Repo layout

| Path | What |
|------|------|
| `forge/flash_attn.py` | Triton fused forward kernel + `autograd.Function` wrapper |
| `forge/reference.py`  | PyTorch attention baselines (naive + SDPA) |
| `forge/gpt2.py`       | Minimal GPT-2 with swappable attention |
| `modal_app.py`        | Modal A100 image + remote entrypoints |
| `benchmarks/`         | Sweep harness + plotting |
| `profiling/`          | Profiler drivers + tuning notes |
| `docs/`               | Design math, benchmark methodology, profiling writeup |

## Results

_Populated as the project progresses (see `docs/benchmarks.md`)._

## Roadmap

- [x] Repo + Modal A100 infrastructure
- [ ] PyTorch attention baselines + correctness harness
- [ ] Fused FlashAttention **forward** kernel in Triton
- [ ] Benchmark sweep + speedup/bandwidth results
- [ ] Profiling + tuning loop (block sizes, pipelining, SRAM reuse)
- [ ] End-to-end GPT-2 integration
- [ ] _Future:_ fused **backward** kernel · MLP/GELU fusion

## License

MIT — see [LICENSE](LICENSE).
