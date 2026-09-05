from __future__ import annotations

import math

import torch
from torch import nn
from cs336_basics.utils import RMSNorm, RoPE, SwiGLU


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention used inside a Transformer block."""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.d_head, theta, max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        def split_heads(projection: torch.Tensor) -> torch.Tensor:
            return projection(x).view(
                batch_size, seq_len, self.num_heads, self.d_head
            ).transpose(1, 2)

        q = self.rope(split_heads(self.q_proj), token_positions)
        k = self.rope(split_heads(self.k_proj), token_positions)
        v = split_heads(self.v_proj)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        output = torch.softmax(scores, dim=-1) @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.output_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, max_seq_len, theta)
        self.ln1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_indices.shape
        positions = torch.arange(seq_len, device=in_indices.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x, positions)
        return self.lm_head(self.ln_final(x))
