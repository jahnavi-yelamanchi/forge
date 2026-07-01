# Forge — GPT-2 Kernel Fusion with Triton

> A FlashAttention-style **fused causal attention kernel** written in [Triton](https://triton-lang.org/),
> drop-in for GPT-2, benchmarked against PyTorch and profiled/tuned on an **NVIDIA A100**.

**Stack:** PyTorch · Triton · CUDA · Modal (A100) · `torch.profiler`

**TL;DR:** the fused kernel is **up to 18.9× faster** than a naive PyTorch
attention baseline and moves **~33× less HBM traffic**; dropped into a full
GPT-2 (fused **forward + backward**) it delivers a **1.83× forward** and **1.61×
training-step** speedup, **matching PyTorch's production SDPA (FlashAttention-2)
to within ~2%**.

![Speedup vs sequence length](docs/assets/speedup_vs_seqlen.png)

---

## Why

Standard attention materializes the full `N×N` score matrix in GPU HBM, then reads it
back for the softmax and the second matmul. That round-trip is memory-bandwidth bound and
wasteful. **Forge** fuses the whole `softmax(QKᵀ / √d) · V` pipeline into a single Triton
kernel using **online softmax** and **block tiling**, so the score matrix never leaves
on-chip SRAM. The result is a faster forward pass and a large drop in HBM traffic.
Full derivation in [`docs/design.md`](docs/design.md).

## Results

Measured on **NVIDIA A100-SXM4-40GB**, fp16, causal, GPT-2-small head geometry
(heads=12, head_dim=64). Full methodology + all plots in
[`docs/benchmarks.md`](docs/benchmarks.md).

### Kernel (attention in isolation)

| Seqlen | Naive | **Forge (fused, tuned)** | PyTorch SDPA¹ |
|-------:|------:|-------------------------:|--------------:|
|   1024 | 1.00× |                **7.20×** |         10.3× |
|   2048 | 1.00× |               **14.61×** |         19.6× |
|   4096 | 1.00× |               **18.91×** |         24.5× |

- **Up to 18.9× faster** than naive, and **~33× less HBM traffic** at N=4096 — the
  fusion eliminates the N×N score-matrix round-trip.
- Compute throughput rises from ~6 TFLOP/s (bandwidth-bound naive) to
  **~115 TFLOP/s** (compute-bound fused).
- [Profiling + tuning](docs/profiling.md) (pipeline depth + a two-phase causal
  loop) narrowed the gap to SDPA from ~1.5× to **~1.3×**.

### End-to-end (full GPT-2, 12L, B=8, T=1024, fp16)

| Backend | Forward | Train step (fwd+bwd) |
|---------|--------:|---------------------:|
| Naive (PyTorch) | 40.5 ms (1.00×) | 110.5 ms (1.00×) |
| PyTorch SDPA | 21.9 ms (1.85×) | 67.2 ms (1.64×) |
| **Forge (fused)** | **22.0 ms (1.83×)** | **68.7 ms (1.61×)** |

Swapping Forge's fused **forward + backward** into GPT-2 gives a **1.83× forward**
and **1.61× training-step** speedup, matching SDPA within ~2%. Logits differ from
SDPA by at most **2.9e-3**; gradients match SDPA autograd within fp16 tolerance.

![GPT-2 end-to-end forward](docs/assets/e2e_forward.png)

¹ SDPA is NVIDIA's production FlashAttention-2 (hand-tuned CUDA) — the ceiling Forge chases.

## How it runs

All GPU work executes on **Modal A100** (the dev machine has no CUDA); the local
machine only edits code, runs git, and orchestrates Modal.

```bash
# one-time local setup
python -m venv .venv && source .venv/bin/activate
pip install -e .            # local-side deps (modal, numpy, pandas, matplotlib)
modal token new             # if not already configured

# run on the A100
modal run modal_app.py               # smoke test: confirm A100 + torch + triton
modal run modal_app.py::run_tests    # correctness: fused vs PyTorch SDPA (18 tests)
modal run modal_app.py::bench        # benchmark sweep -> results.csv + plots
modal run modal_app.py::profile      # 36-config autotune + torch.profiler
modal run modal_app.py::e2e          # end-to-end GPT-2 fwd + training step
modal run modal_app.py::mlp          # fused linear+GELU FFN micro-benchmark
```

## Repo layout

| Path | What |
|------|------|
| `forge/flash_attn.py` | Triton fused forward **and backward** kernels + autograd wrapper |
| `forge/mlp.py`        | Fused linear+GELU FFN kernel (GEMM + activation epilogue) |
| `forge/reference.py`  | PyTorch attention baselines (naive + SDPA) |
| `forge/gpt2.py`       | Minimal GPT-2 with swappable attention + MLP backend |
| `forge/utils.py`      | Timing, FLOP/bandwidth accounting, tolerances |
| `modal_app.py`        | Modal A100 image + remote entrypoints |
| `benchmarks/`         | Sweep + end-to-end harness + plotting |
| `profiling/`          | Autotune grid + `torch.profiler` driver |
| `tests/`              | Correctness suite (fused vs SDPA, fp16+bf16) |
| `docs/`               | [design](docs/design.md) · [benchmarks](docs/benchmarks.md) · [profiling](docs/profiling.md) · [MLP fusion](docs/mlp-fusion.md) |

## Notes & caveats

- Benchmarks use **random weights** — we measure forward *latency* and
  *implementation consistency*, not model quality.
- Numbers are a single A100-SXM4-40GB, fp16, causal, median of timed CUDA-event
  runs; expect a few % run-to-run variance (the naive baseline is noisiest).
- NCU's hardware counters are locked in Modal's containers (`ERR_NVGPUCTRPERM`),
  so tuning relies on `torch.profiler` + a launch-config grid search rather than
  raw Nsight Compute.
- Both **forward and backward** are fused Triton kernels; gradients match SDPA
  autograd within fp16 tolerance.

## Roadmap

- [x] Repo + Modal A100 infrastructure
- [x] PyTorch attention baselines + correctness harness
- [x] Fused FlashAttention **forward** kernel in Triton — _tests vs SDPA (fp16+bf16)_
- [x] Benchmark sweep + speedup/bandwidth results — _up to 18.9× vs naive, ~33× less HBM traffic_
- [x] Profiling + tuning loop (pipeline depth + two-phase causal) — _gap to SDPA 1.5×→1.3×_
- [x] End-to-end GPT-2 integration — _1.83× forward speedup, logits match SDPA to 2.9e-3_
- [x] Fused **backward** kernel — _grads match SDPA, 1.61× training-step speedup_
- [x] MLP / GELU **epilogue fusion** — _correct; ~0.7× vs cuBLAS (compute-bound; [why](docs/mlp-fusion.md))_

## License

MIT — see [LICENSE](LICENSE).
