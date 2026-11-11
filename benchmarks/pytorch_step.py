"""PyTorch eager baseline for the same MLP + MSE + AdamW step; compare with `cargo run --release -p rt-bench`.

Usage: python pytorch_step.py [num_threads]
"""
import statistics
import sys
import time

import torch

WORKLOADS = [
    ("tiny   (batch 8,   4-32-32-2)", [4, 32, 32, 2], 8, 3000),
    ("small  (batch 32,  16-64-64-4)", [16, 64, 64, 4], 32, 1500),
    ("medium (batch 256, 128-256-256-10)", [128, 256, 256, 10], 256, 200),
]


def build(sizes):
    layers = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(torch.nn.Linear(a, b))
        if i + 2 < len(sizes):
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def main():
    if len(sys.argv) > 1:
        torch.set_num_threads(int(sys.argv[1]))
    print(f"torch {torch.__version__}, threads: {torch.get_num_threads()}")
    for name, sizes, batch, steps in WORKLOADS:
        torch.manual_seed(0)
        model, loss_fn = build(sizes), torch.nn.MSELoss()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x, target = torch.rand(batch, sizes[0]) * 2 - 1, torch.rand(batch, sizes[-1]) * 2 - 1

        def step():
            opt.zero_grad()
            loss_fn(model(x), target).backward()
            opt.step()

        for _ in range(steps // 10):
            step()
        micros = []
        for _ in range(steps):
            start = time.perf_counter()
            step()
            micros.append((time.perf_counter() - start) * 1e6)
        micros.sort()
        print(name)
        print(f"  pytorch eager  median {statistics.median(micros):>9.1f} us   p95 {micros[len(micros) * 95 // 100]:>9.1f} us")


if __name__ == "__main__":
    main()
