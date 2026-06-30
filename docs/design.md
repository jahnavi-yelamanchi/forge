# Design — fused causal attention

## The problem with the naive path

Standard attention computes, for queries/keys/values `Q, K, V ∈ ℝ^{N×d}`:

```
S = (Q Kᵀ) / √d        # N×N scores
P = softmax(S)         # N×N probabilities  (row-wise, causal-masked)
O = P V                # N×d output
```

The `N×N` matrices `S` and `P` are the problem. For `N = 2048` that is ~4M
entries **per head**, and the naive implementation writes them to **HBM** (global
GPU memory) and reads them back twice — once for the softmax, once for `P V`.
Attention is therefore **memory-bandwidth bound**: the GPU spends its time moving
the score matrix, not doing math.

`forge/utils.py` makes this concrete: `naive_attention_bytes` counts the `2·N²`
score-matrix traffic, while `fused_attention_bytes` counts only `Q,K,V,O` — the
gap is what fusion removes.

## The fix: tiling + online softmax

We never materialize `S` or `P`. Instead we tile the sequence into blocks and
keep a **running** softmax so each query block is reduced against streamed
key/value blocks entirely inside on-chip **SRAM**.

For a query block, we walk key/value blocks `j = 1, 2, …` and maintain three
running quantities (all fp32, per query row):

- `m` — running row max of the scores seen so far
- `ℓ` — running sum of `exp(score − m)` (the softmax denominator)
- `acc` — running unnormalized output `Σ exp(score − m) · V`

When a new tile gives scores `s` with tile-max `m̃`, we update with the
**online-softmax recurrence** (Milakov & Gimelshein, 2018):

```
m_new = max(m, m̃)
α     = exp(m − m_new)          # rescale everything computed before
p     = exp(s − m_new)          # probabilities for the new tile
ℓ     = α·ℓ + Σ p
acc   = α·acc + p · V_tile
m     = m_new
```

After the last tile, `O = acc / ℓ`. The subtraction by `m` is what keeps `exp`
from overflowing in fp16/bf16; rescaling old state by `α` corrects for the fact
that the max can grow as we see more tiles. The result is **bit-for-tolerance
identical** to the naive softmax, with no `N×N` HBM traffic.

This is the FlashAttention algorithm (Dao et al., 2022); see the kernel in
`forge/flash_attn.py`.

## Causality

A query at row `i` may only attend to keys `j ≤ i`. Two optimizations follow:

1. **Block skipping** — a query block never needs key blocks that start past its
   last row, so the inner loop stops early (`hi = (start_m+1)·BLOCK_M`). This is
   the ~0.5× FLOP factor in `attention_flops`.
2. **Diagonal masking** — the one key block straddling the diagonal is masked
   element-wise (`offs_m ≥ key_idx`); fully-below-diagonal blocks need no mask.
   (Phase 4 can skip the `tl.where` on full blocks for a small speedup.)

## Mapping to the GPU

- **Grid:** `(ceil(N / BLOCK_M), batch·heads)` — one program per query block per
  (batch, head). Query blocks and heads are fully independent → embarrassingly
  parallel across the A100's SMs.
- **SRAM residency:** the query tile is loaded once and reused across all key
  tiles; `m, ℓ, acc` live in registers. Nothing round-trips to HBM mid-kernel.
- **Tunables** (profiled in Phase 4):
  - `BLOCK_M`, `BLOCK_N` — tile sizes → SRAM footprint and reuse.
  - `num_warps` — threads per program → occupancy.
  - `num_stages` — software-pipelining depth → overlap of global loads (K/V
    prefetch) with tensor-core math.

## Why it's faster

Same arithmetic, far less memory traffic. Eliminating the `N×N` HBM round-trip is
the source of both the **higher effective bandwidth** and the **forward-pass
speedup** measured in Phase 3.

## Numerics

Scores, the running max/sum, and the accumulator are fp32; only the `P·V` matmul
operands are cast back to fp16/bf16 to use tensor cores (standard FlashAttention
practice). Tolerances vs. PyTorch SDPA: `atol=rtol=2e-2` (fp16), `4e-2` (bf16) —
see `tests/test_correctness.py`.

## Backward (future)

`FlashAttnFunc.backward` currently recomputes gradients via PyTorch autograd — a
correct, self-contained seam. A fused Triton backward kernel (recomputing `S`/`P`
tile-wise from the saved `m`/`ℓ` statistics) drops in there without changing the
public API.
