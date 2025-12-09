import json


class BPETokenizer:
    vocab_size: int
    merges: dict[tuple[int, int], int]
    vocab: dict[int, bytes]

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

    def train(self, text: str) -> None:
        """build the vocab"""
        tokens = list(text.encode("utf-8"))
        num_merges = self.vocab_size - 256
        print(f"[tokenizer] Training: {len(text)} chars, {num_merges} merges")
        for i in range(num_merges):
            if (i + 1) % 100 == 0:
                print(f"[tokenizer] Merge {i + 1}/{num_merges}")
            pairs: dict[tuple[int, int], int] = {}
            for p in zip(tokens, tokens[1:]):
                pairs[p] = pairs.get(p, 0) + 1
            if not pairs:
                break

            best = max(pairs, key=pairs.get)

            new_id = 256 + i
            tokens = merge(tokens, best, new_id)
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
        print(f"[tokenizer] Done: vocab size {len(self.vocab)}")

    def encode(self, text: str) -> list[int]:
        """
        go through the pairs, look in the self.merges for the earliest one and then replace it in the token list
        """
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            pairs = set(zip(tokens, tokens[1:]))
            earliest = None
            for pair in pairs:
                if pair in self.merges:
                    if earliest is None or self.merges[pair] < self.merges[earliest]:
                        earliest = pair
            if earliest is None:
                break
            tokens = merge(tokens, earliest, self.merges[earliest])
        return tokens

    def decode(self, ids: list[int]) -> str:
        """go through the vocab"""
        return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """save tokenizer to json file"""
        merges_serialized = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        data = {"vocab_size": self.vocab_size, "merges": merges_serialized}

        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[tokenizer] Saved to {path}")

    def load(self, path: str) -> None:
        """load from path"""
        with open(path, "r") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]

        self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair, new_id in sorted(self.merges.items(), key=lambda x: x[1]):
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        print(f"[tokenizer] Loaded from {path}")


def merge(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """replace all occurrences of pair in tokens with new_id"""
    result = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result
