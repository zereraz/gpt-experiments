from tokenizer import BPETokenizer

SMALL_TEXT = "hello world " * 100
MEDIUM_TEXT = "the quick brown fox jumps over the lazy dog " * 1000
LARGE_TEXT = (
    open("README.md").read() if __import__("os").path.exists("README.md") else MEDIUM_TEXT * 10
)


def test_train_small(benchmark):
    def train():
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(SMALL_TEXT)

    benchmark(train)


def test_train_medium(benchmark):
    def train():
        tokenizer = BPETokenizer(vocab_size=500)
        tokenizer.train(MEDIUM_TEXT)

    benchmark(train)


def test_encode_speed(benchmark):
    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(MEDIUM_TEXT)
    benchmark(tokenizer.encode, MEDIUM_TEXT)
