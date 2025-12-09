from tokenizer import BPETokenizer, merge


def test_merge_basic():
    tokens = list("hello".encode("utf-8"))
    result = merge(tokens, ("l".encode("utf-8")[0], "l".encode("utf-8")[0]), 256)
    assert result == [104, 101, 256, 111]  # h, e, "ll", o


def test_merge_multiple_occurrences():
    tokens = [1, 2, 1, 2, 1, 2]
    result = merge(tokens, (1, 2), 256)
    assert result == [256, 256, 256]


def test_merge_no_match():
    tokens = [1, 2, 3]
    result = merge(tokens, (4, 5), 256)
    assert result == [1, 2, 3]


def test_tokenizer_train():
    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train("hello hello hello")
    print(tokenizer.merges)
    assert len(tokenizer.merges) == 4


def test_tokenizer_encode_decode():
    tokenizer = BPETokenizer(vocab_size=280)
    tokenizer.train("the quick brown fox jumps over the lazy dog")

    text = "the fox"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text
