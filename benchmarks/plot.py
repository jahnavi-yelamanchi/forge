"""Turn the benchmark CSV into publication-quality plots for docs/README.

Usage (run locally, after the sweep CSV exists):
    python benchmarks/plot.py benchmarks/results.csv docs/assets
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Consistent color per implementation across every figure.
COLORS = {"naive": "#9aa0a6", "sdpa": "#4285f4", "fused": "#ea4335"}
LABELS = {"naive": "Naive (PyTorch)", "sdpa": "PyTorch SDPA", "fused": "Forge (fused Triton)"}


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def plot_speedup_vs_seqlen(df: pd.DataFrame, out_dir: Path) -> None:
    s = df[df.sweep == "seqlen"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for impl in ("fused", "sdpa"):
        d = s[s.impl == impl].sort_values("seqlen")
        ax.plot(d.seqlen, d.speedup_vs_naive, "o-", color=COLORS[impl], label=LABELS[impl])
    ax.axhline(1.0, color=COLORS["naive"], ls="--", lw=1, label="Naive baseline")
    ax.set(xlabel="Sequence length", ylabel="Speedup vs naive (×)",
           title="Forward-pass speedup vs sequence length")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(s.seqlen.unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    _save(fig, out_dir, "speedup_vs_seqlen.png")


def plot_latency_vs_seqlen(df: pd.DataFrame, out_dir: Path) -> None:
    s = df[df.sweep == "seqlen"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for impl in ("naive", "sdpa", "fused"):
        d = s[s.impl == impl].sort_values("seqlen")
        ax.plot(d.seqlen, d.latency_ms, "o-", color=COLORS[impl], label=LABELS[impl])
    ax.set(xlabel="Sequence length", ylabel="Latency (ms, log)",
           title="Forward latency vs sequence length")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(s.seqlen.unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    _save(fig, out_dir, "latency_vs_seqlen.png")


def plot_tflops_vs_seqlen(df: pd.DataFrame, out_dir: Path) -> None:
    s = df[df.sweep == "seqlen"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for impl in ("naive", "sdpa", "fused"):
        d = s[s.impl == impl].sort_values("seqlen")
        ax.plot(d.seqlen, d.tflops, "o-", color=COLORS[impl], label=LABELS[impl])
    ax.set(xlabel="Sequence length", ylabel="Compute throughput (TFLOP/s)",
           title="Compute throughput vs sequence length")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(s.seqlen.unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    _save(fig, out_dir, "tflops_vs_seqlen.png")


def plot_hbm_traffic(df: pd.DataFrame, out_dir: Path) -> None:
    """HBM bytes moved: naive (with N×N scores) vs fused (Q,K,V,O only)."""
    s = df[df.sweep == "seqlen"]
    seqlens = sorted(s.seqlen.unique())
    naive = [s[(s.seqlen == n) & (s.impl == "naive")].hbm_gb.iloc[0] for n in seqlens]
    fused = [s[(s.seqlen == n) & (s.impl == "fused")].hbm_gb.iloc[0] for n in seqlens]
    x = range(len(seqlens))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - 0.2 for i in x], naive, 0.4, color=COLORS["naive"], label="Naive (incl. N×N scores)")
    ax.bar([i + 0.2 for i in x], fused, 0.4, color=COLORS["fused"], label="Forge (Q,K,V,O only)")
    for i, (nv, fs) in enumerate(zip(naive, fused)):
        ax.text(i, max(nv, fs), f"{nv / fs:.1f}× less", ha="center", va="bottom", fontsize=9)
    ax.set(xlabel="Sequence length", ylabel="HBM traffic (GB)",
           title="HBM traffic: fusion eliminates the N×N round-trip")
    ax.set_xticks(list(x))
    ax.set_xticklabels(seqlens)
    ax.legend()
    _save(fig, out_dir, "hbm_traffic.png")


def plot_e2e(json_path: str, out_dir: Path) -> None:
    """Grouped bars: GPT-2 forward and training-step latency per backend."""
    import json

    data = json.loads(Path(json_path).read_text())
    order = ["naive", "sdpa", "fused"]
    fwd = data["latency_ms"]
    fwd_sp = data["speedup_vs_naive"]
    train = data.get("train_ms", {})
    train_sp = data.get("train_speedup_vs_naive", {})
    x = range(len(order))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    fb = ax.bar([i - 0.2 for i in x], [fwd[k] for k in order], 0.4,
                color=[COLORS[k] for k in order], label="Forward")
    tb = ax.bar([i + 0.2 for i in x], [train.get(k) or 0 for k in order], 0.4,
                color=[COLORS[k] for k in order], alpha=0.55, hatch="//", label="Train (fwd+bwd)")

    for k, b in zip(order, fb):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{fwd[k]:.0f}\n{fwd_sp[k]:.2f}×", ha="center", va="bottom", fontsize=8)
    for k, b in zip(order, tb):
        if train.get(k):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{train[k]:.0f}\n{train_sp[k]:.2f}×", ha="center", va="bottom", fontsize=8)

    cfg = data["config"]
    ax.set(ylabel="Latency (ms)",
           title=f"GPT-2: {cfg['n_layer']}L, B={cfg['batch']}, T={cfg['seqlen']}, fp16")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[k] for k in order])
    ax.margins(y=0.18)
    ax.legend()
    _save(fig, out_dir, "e2e_forward.png")


def main(csv_path: str, out_dir: str) -> None:
    df = pd.read_csv(csv_path)
    out = Path(out_dir)
    print(f"Plotting from {csv_path} -> {out}/")
    plot_speedup_vs_seqlen(df, out)
    plot_latency_vs_seqlen(df, out)
    plot_tflops_vs_seqlen(df, out)
    plot_hbm_traffic(df, out)


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "docs/assets"
    main(csv, out)
