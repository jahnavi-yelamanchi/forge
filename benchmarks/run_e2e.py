"""End-to-end GPT-2 forward benchmark across attention backends.

Builds one GPT-2 (random fp16 weights) and runs the *same* forward pass with each
attention backend, so the only thing that changes is the attention kernel. Reports
per-backend forward latency, speedup vs the naive backend, and the logit
difference of the fused path vs SDPA (proving the swap is numerically safe).
"""

from __future__ import annotations

import torch

from forge import utils
from forge.gpt2 import GPT, GPT2Config


def _train_step_ms(model, idx) -> float | None:
    """Median forward+backward (training step) latency, or None on OOM."""
    def step():
        model.zero_grad(set_to_none=True)
        loss = model(idx).float().mean()
        loss.backward()

    try:
        return utils.bench_ms(step, warmup=3, iters=10)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def run_e2e(batch: int = 8, seqlen: int = 1024, dtype: torch.dtype = torch.float16) -> dict:
    assert torch.cuda.is_available(), "CUDA required"
    utils.set_seed(0)

    cfg = GPT2Config(block_size=seqlen)
    model = GPT(cfg).cuda().to(dtype)
    idx = torch.randint(0, cfg.vocab_size, (batch, seqlen), device="cuda")

    logits = {}
    fwd_ms = {}
    train_ms = {}
    for backend in ("naive", "sdpa", "fused"):
        model.set_attention_backend(backend)
        with torch.no_grad():
            logits[backend] = model(idx)
            fwd_ms[backend] = utils.bench_ms(lambda: model(idx), warmup=5, iters=20)
        train_ms[backend] = _train_step_ms(model, idx)
        tr = f"{train_ms[backend]:8.3f}" if train_ms[backend] else "    OOM"
        print(f"  {backend:5s}: fwd {fwd_ms[backend]:8.3f} ms | train {tr} ms")

    base_f = fwd_ms["naive"]
    base_t = train_ms["naive"]
    diff = (logits["fused"].float() - logits["sdpa"].float()).abs()

    def train_speedup(k):
        return round(base_t / train_ms[k], 3) if (base_t and train_ms[k]) else None

    result = {
        "config": {"n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd,
                   "batch": batch, "seqlen": seqlen, "dtype": str(dtype).replace("torch.", "")},
        "latency_ms": {k: round(v, 4) for k, v in fwd_ms.items()},
        "speedup_vs_naive": {k: round(base_f / v, 3) for k, v in fwd_ms.items()},
        "train_ms": {k: (round(v, 4) if v else None) for k, v in train_ms.items()},
        "train_speedup_vs_naive": {k: train_speedup(k) for k in train_ms},
        "fused_speedup_over_sdpa": round(fwd_ms["sdpa"] / fwd_ms["fused"], 3),
        "fused_vs_sdpa_max_abs": round(float(diff.max()), 4),
    }
    print(
        f"\n  forward: fused {result['speedup_vs_naive']['fused']}x vs naive | "
        f"train step: fused {result['train_speedup_vs_naive']['fused']}x vs naive"
    )
    return result
