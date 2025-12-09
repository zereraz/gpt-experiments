import torch
import time


def benchmark(device, size=2000, runs=10):
    x = torch.randn(size, size, device=device)

    # Warmup
    for _ in range(3):
        _ = x @ x.T

    if device == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = x @ x.T

    if device == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start
    print(f"{device}: {elapsed / runs * 1000:.2f} ms per matmul")


benchmark("cpu")
if torch.backends.mps.is_available():
    benchmark("mps")
