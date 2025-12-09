import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # (B, n_heads, T, head_dim)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim**0.5)

        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ V  # (B, n_heads, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)
