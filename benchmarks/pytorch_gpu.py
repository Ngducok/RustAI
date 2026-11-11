"""PyTorch CUDA baseline for `cargo run --release -p rt-bench --bin gpu`.

Tensors live on the GPU; timing synchronises the device, like the Rust benchmark.
Usage: python pytorch_gpu.py
"""
import statistics
import time

import torch

WORKLOADS = [
    ("medium (batch 256,  128-256-256-10)", [128, 256, 256, 10], 256),
    ("large  (batch 2048, 512-1024-1024-10)", [512, 1024, 1024, 10], 2048),
    ("xlarge (batch 4096, 1024-2048-2048-10)", [1024, 2048, 2048, 10], 4096),
]


def median_micros(rounds, warmup, fn):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def build(sizes):
    layers = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(torch.nn.Linear(a, b))
        if i + 2 < len(sizes):
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers).cuda()


def main():
    device = torch.device("cuda")
    print(f"torch {torch.__version__}, adapter: {torch.cuda.get_device_name(0)} (CUDA)")
    print("matmul (inputs resident on the device)")
    for n in (512, 1024, 2048, 4096):
        a, b = torch.rand(n, n, device=device) * 2 - 1, torch.rand(n, n, device=device) * 2 - 1
        micros = median_micros(15, 5, lambda: torch.matmul(a, b))
        print(f"  {n}x{n}x{n}: {micros:10.1f} us  {2.0 * n ** 3 / micros / 1e3:8.1f} GFLOP/s")
    print("training step: forward + MSE + backward + AdamW")
    for name, sizes, batch in WORKLOADS:
        torch.manual_seed(0)
        model, loss_fn = build(sizes), torch.nn.MSELoss()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x, target = torch.rand(batch, sizes[0], device=device) * 2 - 1, torch.rand(batch, sizes[-1], device=device) * 2 - 1

        def step():
            for _ in range(10):
                opt.zero_grad()
                loss_fn(model(x), target).backward()
                opt.step()

        micros = median_micros(15, 3, step) / 10
        print(f"  {name}")
        print(f"    pytorch cuda {micros:10.1f} us/step")


if __name__ == "__main__":
    main()
