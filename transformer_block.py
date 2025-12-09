from multi_head_attention import MultiHeadAttention
import torch
import torch.nn as nn


# d_ff — hidden size of feed-forward (typically 4× d_model)
# pre-norm more stable than post-norm
# x + ... residual connections help gradient flow
# GELU smoother than ReLU
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # residual + pre-norm
        x = x + self.ff(self.ln2(x))
        return x
