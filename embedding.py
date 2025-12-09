import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        return self.token_emb(x) + self.pos_emb(positions)
