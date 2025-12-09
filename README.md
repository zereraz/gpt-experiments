# Training from Scratch

Learning to build an LLM from scratch using just PyTorch.

## Goals

- Understand the fundamentals of language models
- Implement a BPE tokenizer from scratch
- Build transformer architecture step by step
- Train on real data

## Setup

```bash
uv sync --extra dev
```

## Development

```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Run tests
uv run pytest

# Run benchmarks
uv run pytest benchmark_tokenizer.py --benchmark-only
```
