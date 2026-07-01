"""Minimal GPT-2 with a swappable attention backend.

A compact nanoGPT-style GPT-2 whose self-attention can be routed through any of
the three implementations (``naive`` / ``sdpa`` / ``fused``) via one flag. This
lets us measure the fused kernel's effect on **end-to-end forward latency** of a
real model, not just the kernel in isolation, and confirm the model's logits are
unchanged when we swap in the Triton kernel.

Weights are random — we benchmark the forward pass and check impl-consistency,
not model quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.flash_attn import flash_attention
from forge.reference import naive_attention, sdpa_attention

ATTENTION_BACKENDS = {
    "naive": naive_attention,
    "sdpa": sdpa_attention,
    "fused": flash_attention,
}


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768  # head_dim = n_embd / n_head = 64


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # fused QKV projection
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn = "fused"  # backend key; flip via GPT.set_attention_backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # The Triton kernel wants contiguous (B, H, T, d) tensors.
        q, k, v = (t.contiguous() for t in (q, k, v))

        y = ATTENTION_BACKENDS[self.attn](q, k, v, causal=True)  # (B, H, T, d)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def set_attention_backend(self, name: str) -> None:
        """Route every attention layer through `name` (naive/sdpa/fused)."""
        assert name in ATTENTION_BACKENDS, name
        for block in self.blocks:
            block.attn.attn = name

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))
