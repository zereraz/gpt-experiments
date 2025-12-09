import torch
import torch.nn as nn
from embedding import Embeddings
from transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        d_ff: int,
    ):
        super().__init__()
        self.embed = Embeddings(vocab_size, d_model, max_seq_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        print(f"[gpt] Created: {n_layers} layers, d_model={d_model}, n_heads={n_heads}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) token ids

        x = self.embed(x)  # (B, T, d_model)
        for block in self.blocks:
            x = block(x)  # (B, T, d_model)
        x = self.ln_f(x)  # final layer norm
        logits = self.head(x)  # (B, T, vocab_size)
        return logits
