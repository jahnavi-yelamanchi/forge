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

Measured on **NVIDIA A100-SXM4-40GB**, fp16, causal, GPT-2-small heads (full
methodology + more plots in [`docs/benchmarks.md`](docs/benchmarks.md)).

![Speedup vs sequence length](docs/assets/speedup_vs_seqlen.png)

| Seqlen | Naive | **Forge (fused, tuned)** | PyTorch SDPA¹ |
|-------:|------:|-------------------------:|--------------:|
|   1024 | 1.00× |                **7.20×** |         10.3× |
|   2048 | 1.00× |               **14.61×** |         19.6× |
|   4096 | 1.00× |               **18.91×** |         24.5× |

- **Up to 18.9× faster** than a naive PyTorch baseline, and **~33× less HBM
  traffic** at N=4096 — the fusion eliminates the N×N score-matrix round-trip.
- Compute throughput rises from ~6 TFLOP/s (bandwidth-bound naive) to
  **~115 TFLOP/s** (compute-bound fused).
- Phase-4 tuning (pipeline depth + two-phase causal loop) narrows the gap to
  PyTorch SDPA from ~1.5× to **~1.3×**. See [`docs/profiling.md`](docs/profiling.md).

**End-to-end**, dropped into a full GPT-2 (12L, B=8, T=1024, fp16), Forge gives a
**1.85× forward-pass speedup** over the naive baseline — matching PyTorch SDPA to
within 1%, with logits identical to 2.9e-3.

![GPT-2 end-to-end forward](docs/assets/e2e_forward.png)

¹ SDPA is NVIDIA's production FlashAttention-2 (hand-tuned CUDA); it's the ceiling
Forge chases.

## Roadmap

- [x] Repo + Modal A100 infrastructure
- [x] PyTorch attention baselines + correctness harness
- [x] Fused FlashAttention **forward** kernel in Triton — _17/17 tests vs SDPA (fp16+bf16, N≤2048)_
- [x] Benchmark sweep + speedup/bandwidth results — _up to 18.9× vs naive, ~33× less HBM traffic_
- [x] Profiling + tuning loop (pipeline depth + two-phase causal) — _gap to SDPA 1.5×→1.3×_
- [x] End-to-end GPT-2 integration — _1.85× forward speedup, logits match SDPA to 2.9e-3_
- [ ] _Future:_ fused **backward** kernel · MLP/GELU fusion

## License

MIT — see [LICENSE](LICENSE).
