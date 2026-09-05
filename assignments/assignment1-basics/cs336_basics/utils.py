import math

import torch
from torch import nn


class LinearModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype))
        std = 2 / (self.in_features + self.out_features) ** 0.5
        torch.nn.init.trunc_normal_(self.W, std=std, a = -3 * std, b = 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T



class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=self.device, dtype=self.dtype))
        std = 1
        torch.nn.init.trunc_normal_(self.embedding_matrix, std=std, a = -3 * std, b = 3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the final input dimension."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the normalization in float32 for numerical stability, then
        # return the result in the input dtype.
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float / rms * self.weight).to(input_dtype)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self, 
            Q: torch.Tensor, 
            K: torch.Tensor, 
            V: torch.Tensor, 
            mask: torch.Tensor | None = None
        ):
        """
        Given key (K), query (Q), and value (V) tensors, return
        the output of your scaled dot product attention implementation.

        Args:
            Q (Float[Tensor, " ... queries d_k"]): Query tensor
            K (Float[Tensor, " ... keys d_k"]): Key tensor
            V (Float[Tensor, " ... keys d_v"]): Values tensor
            mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
        Returns:
            Float[Tensor, " ... queries d_v"]: Output of SDPA
        """
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V)

def softmax(x: torch.Tensor, dim: int):
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

class RoPE(nn.Module):
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    def __init__(self, d_k: int, theta: float, max_seq_len: int):
        super().__init__()
        assert d_k % 2 == 0

        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        positions = torch.arange(max_seq_len, dtype=torch.float32)
        dim_indices = torch.arange(0, d_k, 2, dtype=torch.float32)
        inv_freq = theta ** (-dim_indices /d_k)
        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer("cos_cache", torch.cos(angles))
        self.register_buffer("sin_cache", torch.sin(angles))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        """
        x.shape:
            (..., sequence_length, d_k)

        token_positions.shape:
            (..., sequence_length)
        """
        if x.shape[-1] != self.d_k:
            raise ValueError(f"Expected last dimension {self.d_k}, got {x.shape[-1]}")

        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        x_even = x[...,0::2]
        x_odd = x[...,1::2]

        while cos.ndim < x_even.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd  = x_even * sin + x_odd * cos

        output = torch.empty_like(x)
        output[..., 0::2] = rotated_even
        output[..., 1::2] = rotated_odd
        return output



class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V)

    def forward(self, x, wq, wk, wv, wo) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        q = x @ wq.T
        k = x @ wk.T
        v = x @ wv.T

        q = q.view(batch_size, seq_len, self.num_heads, self.d_head)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_head)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device,
            )
        )
        mask = mask.unsqueeze(0).unsqueeze(0)
        out = self.attention(q, k, v, mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, seq_len, d_model)
        out = out @ wo.T
        return out

class MultiHeadAttentionWithRoPE(nn.Module):

    def __init__(self, d_model:int, n_heads:int, max_seq_len:int, theta:float,device=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RoPE(
            d_k=self.head_dim,
            theta=theta,
            max_seq_len=max_seq_len,
        )

    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        # d_k = self.head_dim
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9) #如果mask为0，则将对应位置的score设置为-1e9
        attn_weights = torch.softmax(scores, dim=-1) #对key这一维度进行softmax归一化
        return torch.matmul(attn_weights, V)

    def forward(self, x, wq, wk, wv, wo, token_positions)->torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        q = x @ wq.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        k = x @ wk.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        v = x @ wv.T # (batch_size, seq_len, d_model) @ (d_model, d_v) -> (batch_size, seq_len, d_v)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim) #view会优先切分最后一个维度，这和内存有关。
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        mask = torch.triu(torch.ones(seq_len, seq_len,dtype=torch.bool,device=x.device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

        out = self.attention(q, k, v, mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, seq_len, d_model)
        out = out @ wo.T
        return out
