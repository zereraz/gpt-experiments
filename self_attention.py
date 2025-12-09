import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_model**0.5)

        T = x.size(1)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        return attn @ V
