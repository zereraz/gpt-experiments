from torch.utils.data import Dataset, DataLoader
import torch
from tokenizer import BPETokenizer


class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer: BPETokenizer, seq_len: int):
        self.tokens = tokenizer.encode(text)
        self.seq_len = seq_len
        print(f"[dataset] Created: {len(self.tokens)} tokens, seq_len={seq_len}")

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return torch.tensor(chunk, dtype=torch.long)
