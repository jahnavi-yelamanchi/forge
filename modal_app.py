"""Modal entrypoints for Forge.

All GPU work runs here on an NVIDIA A100, since the dev machine has no CUDA.
Run any function with, e.g.:

    modal run modal_app.py::smoke

The local `forge/` package (plus tests/benchmarks/profiling) is mounted into the
image so remote functions import the same code you edit locally.
"""

from __future__ import annotations

import modal

# --- Image -----------------------------------------------------------------
# A CUDA-capable PyTorch + Triton image. Torch ships its own CUDA runtime, so we
# only need a plain base plus the pip deps. Nsight Systems (`nsys`) is added in
# the profiling phase.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3",
        "triton>=2.3",
        "numpy>=1.26",
        "pandas>=2.0",
        "matplotlib>=3.8",
        "pytest>=8.0",
    )
    # Mount the project source so remote code == local code.
    .add_local_python_source("forge")
    .add_local_python_source("benchmarks")
    .add_local_python_source("profiling")
    .add_local_dir("tests", remote_path="/root/tests")
)

app = modal.App("forge", image=image)

GPU = "A100"


@app.function(gpu=GPU)
def smoke() -> dict:
    """Confirm the A100 image is healthy: GPU visible, torch + triton import."""
    import torch
    import triton

    has_cuda = bool(torch.cuda.is_available())
    cap = torch.cuda.get_device_capability(0) if has_cuda else (None, None)

    # Tiny on-GPU matmul to prove the runtime actually works end to end.
    x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    y = float((x @ x).float().sum().item())

    # Return only plain primitives — the local client deserializes this and has
    # no torch installed, so no torch-typed objects may leak into the result.
    info = {
        "torch": str(torch.__version__),
        "triton": str(triton.__version__),
        "cuda_available": has_cuda,
        "device": str(torch.cuda.get_device_name(0)) if has_cuda else None,
        "capability": f"{cap[0]}.{cap[1]}" if has_cuda else None,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if has_cuda else False,
        "matmul_ok": bool(y == y),  # NaN check
    }

    print("=== Forge A100 smoke test ===")
    for k, v in info.items():
        print(f"  {k:18s}: {v}")
    return info


@app.function(gpu=GPU)
def run_tests() -> int:
    """Run the correctness suite on the A100. Returns pytest's exit code.

    pytest output is captured and re-emitted with a ``PYTEST|`` prefix so the
    summary survives Modal's progress spinner overwriting terminal lines.
    """
    import contextlib
    import io
    import sys

    import pytest

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = pytest.main(["-q", "/root/tests"])

    for line in buf.getvalue().splitlines():
        print(f"PYTEST| {line}")
    print(f"PYTEST| exit code: {int(code)}")
    sys.stdout.flush()
    return int(code)


@app.function(gpu=GPU)
def run_bench(dtype: str = "float16") -> list[dict]:
    """Run the benchmark sweep on the A100, return metric rows (primitives)."""
    import torch

    from benchmarks.run_bench import run_sweep

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    print(f"=== Forge benchmark sweep ({dtype}) ===")
    return run_sweep(dtype=torch_dtype)


@app.function(gpu=GPU, timeout=1800)
def run_profile() -> dict:
    """Autotune the kernel on the A100 across representative shapes."""
    from profiling.profile_kernel import profile_timeline, sweep_configs

    HEADS, HEAD_DIM = 12, 64
    out = {"sweeps": [], "timeline": []}
    for N in (2048, 4096):
        shape = (4, HEADS, N, HEAD_DIM)
        print(f"\n=== autotune sweep @ B=4 N={N} ===")
        res = sweep_configs(shape)
        out["sweeps"].append(res)
        # torch.profiler evidence: default vs best.
        from profiling.profile_kernel import DEFAULT_CFG

        best_cfg = {k: res["best"][k] for k in ("block_m", "block_n", "num_warps", "num_stages")}
        out["timeline"].append(
            {
                "seqlen": N,
                "default": profile_timeline(shape, DEFAULT_CFG),
                "best": profile_timeline(shape, best_cfg),
            }
        )
    return out


@app.local_entrypoint()
def profile():
    """`modal run modal_app.py::profile` -> autotune sweep + profiler timeline."""
    import json
    from pathlib import Path

    out = run_profile.remote()
    path = Path("profiling/tuning_results.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")
    for s in out["sweeps"]:
        n = s["shape"]["seqlen"]
        b = s["best"]
        print(f"  N={n}: best {b} -> {b['speedup_vs_default']}x over untuned default")


@app.local_entrypoint()
def bench(dtype: str = "float16"):
    """`modal run modal_app.py::bench` -> sweep on A100, write CSV + plots locally."""
    import csv
    from pathlib import Path

    rows = run_bench.remote(dtype=dtype)

    csv_path = Path("benchmarks/results.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {csv_path}")

    from benchmarks.plot import main as plot_main

    plot_main(str(csv_path), "docs/assets")
    print("Plots written to docs/assets/ ✅")


@app.local_entrypoint()
def main():
    """`modal run modal_app.py` -> run the smoke test and print the result."""
    result = smoke.remote()
    assert result["cuda_available"], "CUDA not available on the Modal worker!"
    print("\nSmoke test passed ✅")
