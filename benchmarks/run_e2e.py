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


@torch.no_grad()
def run_e2e(batch: int = 8, seqlen: int = 1024, dtype: torch.dtype = torch.float16) -> dict:
    assert torch.cuda.is_available(), "CUDA required"
    utils.set_seed(0)

    cfg = GPT2Config(block_size=seqlen)
    model = GPT(cfg).cuda().to(dtype).eval()
    idx = torch.randint(0, cfg.vocab_size, (batch, seqlen), device="cuda")

    logits = {}
    latency = {}
    for backend in ("naive", "sdpa", "fused"):
        model.set_attention_backend(backend)
        logits[backend] = model(idx)
        latency[backend] = utils.bench_ms(lambda: model(idx), warmup=5, iters=20)
        print(f"  {backend:5s}: {latency[backend]:8.3f} ms")

    base = latency["naive"]
    # Numerical agreement of the fused kernel vs SDPA, over the full model stack.
    diff = (logits["fused"].float() - logits["sdpa"].float()).abs()
    rel = diff / (logits["sdpa"].float().abs() + 1e-3)

    result = {
        "config": {"n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd,
                   "batch": batch, "seqlen": seqlen, "dtype": str(dtype).replace("torch.", "")},
        "latency_ms": {k: round(v, 4) for k, v in latency.items()},
        "speedup_vs_naive": {k: round(base / v, 3) for k, v in latency.items()},
        "fused_speedup_over_sdpa": round(latency["sdpa"] / latency["fused"], 3),
        "fused_vs_sdpa_max_abs": round(float(diff.max()), 4),
        "fused_vs_sdpa_max_rel": round(float(rel.max()), 4),
    }
    print(
        f"\n  end-to-end: fused {result['speedup_vs_naive']['fused']}x vs naive, "
        f"logits vs SDPA max_abs={result['fused_vs_sdpa_max_abs']}"
    )
    return result
