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
    """Run the correctness suite on the A100. Returns pytest's exit code."""
    import sys

    import pytest

    code = pytest.main(["-q", "/root/tests"])
    print(f"\npytest exit code: {int(code)}")
    sys.stdout.flush()
    return int(code)


@app.local_entrypoint()
def main():
    """`modal run modal_app.py` -> run the smoke test and print the result."""
    result = smoke.remote()
    assert result["cuda_available"], "CUDA not available on the Modal worker!"
    print("\nSmoke test passed ✅")
