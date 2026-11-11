"""PyTorch matmul throughput on the same shapes as `cargo run --release -p rt-bench --bin matmul`.

Usage: python pytorch_matmul.py
"""
import statistics
import time

import torch

SHAPES = [(32, 64, 64), (64, 64, 64), (128, 128, 128), (256, 128, 256), (256, 256, 256),
          (512, 512, 512), (256, 256, 10), (1024, 1024, 1024)]


def median_micros(m, k, n):
    a, b = torch.rand(m, k) * 2 - 1, torch.rand(k, n) * 2 - 1
    out = torch.empty(m, n)
    iterations = int(min(max(4e8 / (2.0 * m * k * n), 5), 3000))
    for _ in range(iterations // 10 + 1):
        torch.matmul(a, b, out=out)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        torch.matmul(a, b, out=out)
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def main():
    default_threads = torch.get_num_threads()
    print(f"torch {torch.__version__}; {'shape (m,k,n)':<18} {'1T us':>10} {'1T GFLOP/s':>10} {'all us':>10} {'all GFLOP/s':>10}")
    for m, k, n in SHAPES:
        torch.set_num_threads(1)
        one = median_micros(m, k, n)
        torch.set_num_threads(default_threads)
        everything = median_micros(m, k, n)
        gflops = lambda micros: 2.0 * m * k * n / micros / 1e3
        print(f"{'':<11}{f'{m}x{k}x{n}':<18} {one:>10.1f} {gflops(one):>10.2f} {everything:>10.1f} {gflops(everything):>10.2f}")


if __name__ == "__main__":
    main()
