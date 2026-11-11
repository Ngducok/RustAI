"""Runs the Rust and PyTorch benchmarks in interleaved rounds and renders charts for the README.

Usage: python tools/benchmark_report.py [rounds] [--reparse] [--out DIR] [--raw FILE]
Raw outputs are kept in docs/benchmark-raw.json; --reparse redraws from them without measuring.
Writes docs/images/benchmark-{matmul,training,gpu,ppo}.png (or into DIR). The GPU comparison is
skipped on machines without a wgpu adapter or without CUDA PyTorch.
The best round is reported because background load only ever makes a run slower.
"""
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
TORCH_VERSION, TORCH_THREADS = "", ""


def option(name, default):
    return Path(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


OUT = option("--out", OUT)
BG, PANEL, INK, MUTED, GRID = "#0d1117", "#161b22", "#e6edf3", "#8b949e", "#30363d"
COLORS = {"rust": "#3fb950", "torch": "#d29922", "tape": "#8b949e"}


def run(*args):
    done = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode != 0:
        raise RuntimeError(done.stderr or done.stdout)
    return done.stdout


def cargo(*args):
    return run("cargo", "run", "--release", "-q", "-p", "rt-bench", *args)


def parse_matmul(text):
    rows = {}
    for line in text.splitlines():
        found = re.search(r"(\d+x\d+x\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if found:
            rows[found.group(1)] = tuple(float(v) for v in found.groups()[1:])
    return rows


def parse_training(text, label_pattern):
    rows, workload = {}, None
    for line in text.splitlines():
        head = re.match(r"(tiny|small|medium)\b", line)
        if head:
            workload = head.group(1)
        found = re.match(rf"\s+({label_pattern})\s+median\s+([\d.]+)", line)
        if found and workload:
            rows.setdefault(workload, {})[found.group(1).strip()] = float(found.group(2))
    return rows


def parse_gpu(text):
    result, workload = {"matmul": {}, "train": {}}, None
    for line in text.splitlines():
        m = re.search(r"(\d+)x\d+x\d+:\s+([\d.]+) us\s+([\d.]+) GFLOP/s", line)
        if m:
            result["matmul"][m.group(1)] = (float(m.group(2)), float(m.group(3)))
        head = re.match(r"\s+(medium|large|xlarge)\b", line)
        if head:
            workload = head.group(1)
        step = re.search(r"(rust gpu|rust cpu|pytorch cuda).*?\s([\d.]+) us/step", line)
        if step and workload:
            result["train"].setdefault(workload, {})[step.group(1)] = float(step.group(2))
    return result


def parse_ppo(text):
    rates = [float(v) for v in re.findall(r"([\d.]+) steps/s", text)]
    median = re.search(r"median time to solve: ([\d.]+) s", text)
    return (float(median.group(1)), sum(rates) / len(rates))


def best_over_runs(samples, higher_is_better=False):
    """Best value per metric across rounds: on a machine with background load, the fastest run
    is the closest to the unloaded speed. Matmul tuples alternate microseconds and GFLOP/s."""
    first = samples[0]
    if isinstance(first, dict):
        return {key: best_over_runs([s[key] for s in samples]) for key in first}
    if isinstance(first, tuple):
        return tuple((max if i % 2 else min)(s[i] for s in samples) for i in range(len(first)))
    return (max if higher_is_better else min)(samples)

def style(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def bars(ax, categories, series):
    width = 0.8 / len(series)
    for i, (label, color, values) in enumerate(series):
        xs = [x + (i - (len(series) - 1) / 2) * width for x in range(len(categories))]
        ax.bar(xs, values, width * 0.92, label=label, color=color)
    ax.set_xticks(range(len(categories)), categories, rotation=0, fontsize=8)
    legend = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=8)
    for text in legend.get_texts():
        text.set_color(INK)


def matmul_chart(rust, torch, stamp):
    shapes = list(rust)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), facecolor=BG)
    for ax, (col, title) in zip(axes, [(1, "1 thread"), (3, "all threads")]):
        style(ax, f"Matmul throughput, {title} (GFLOP/s, higher is better)")
        bars(ax, shapes, [("Rust (this project)", COLORS["rust"], [rust[s][col] for s in shapes]),
                          (f"PyTorch {TORCH_VERSION} CPU", COLORS["torch"], [torch[s][col] for s in shapes])])
    fig.text(0.01, 0.01, stamp, color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "benchmark-matmul.png", dpi=130, facecolor=BG)


def training_chart(rust, torch8, torch1, stamp):
    workloads = ["tiny", "small", "medium"]
    names = ["tiny\nbatch 8, 4-32-32-2", "small\nbatch 32, 16-64-64-4", "medium\nbatch 256, 128-256-256-10"]
    fig, ax = plt.subplots(figsize=(11, 5.4), facecolor=BG)
    style(ax, "Training step latency: forward + MSE + backward + AdamW (microseconds, lower is better)")
    bars(ax, names, [("Dynamic tape (Rust)", COLORS["tape"], [rust[w]["dynamic tape"] for w in workloads]),
                     ("Planned IR (Rust)", COLORS["rust"], [rust[w]["planned IR"] for w in workloads]),
                     (f"PyTorch eager, {TORCH_THREADS} threads", COLORS["torch"], [torch8[w]["pytorch eager"] for w in workloads]),
                     ("PyTorch eager, 1 thread", "#f0b429", [torch1[w]["pytorch eager"] for w in workloads])])
    ax.set_yscale("log")
    fig.text(0.01, 0.01, stamp, color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "benchmark-training.png", dpi=130, facecolor=BG)


def gpu_chart(rust, torch, stamp):
    sizes = list(rust["matmul"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), facecolor=BG)
    style(axes[0], "Matmul throughput on the GPU (GFLOP/s, higher is better)")
    bars(axes[0], [f"{n}^3" for n in sizes], [("Rust (wgpu)", COLORS["rust"], [rust["matmul"][n][1] for n in sizes]),
                                                ("PyTorch (CUDA)", COLORS["torch"], [torch["matmul"][n][1] for n in sizes])])
    workloads = ["medium", "large", "xlarge"]
    names = ["medium\nbatch 256", "large\nbatch 2048", "xlarge\nbatch 4096"]
    style(axes[1], "Training step on the GPU (microseconds, lower is better)")
    bars(axes[1], names, [("Rust GPU (resident)", COLORS["rust"], [rust["train"][w]["rust gpu"] for w in workloads]),
                          (f"Rust CPU ({os.cpu_count()} threads)", COLORS["tape"], [rust["train"][w]["rust cpu"] for w in workloads]),
                          ("PyTorch CUDA", COLORS["torch"], [torch["train"][w]["pytorch cuda"] for w in workloads])])
    axes[1].set_yscale("log")
    fig.text(0.01, 0.01, stamp, color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "benchmark-gpu.png", dpi=130, facecolor=BG)


def ppo_chart(rust, torch, stamp):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=BG)
    for ax, index, title in ((axes[0], 1, "Environment steps per second (higher is better)"),
                             (axes[1], 0, "Time to solve CartPole, seconds (lower is better)")):
        style(ax, title)
        bars(ax, ["PPO, 8 envs"], [("Rust (this project)", COLORS["rust"], [rust[index]]),
                                   ("PyTorch, 1 thread", COLORS["torch"], [torch[index]])])
    fig.text(0.01, 0.01, stamp, color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "benchmark-ppo.png", dpi=130, facecolor=BG)


RAW = option("--raw", ROOT / "docs" / "benchmark-raw.json")
PARSERS = {
    "rust_matmul": parse_matmul,
    "torch_matmul": parse_matmul,
    "rust_training": lambda t: parse_training(t, "dynamic tape|reference IR|planned IR"),
    "torch8": lambda t: parse_training(t, "pytorch eager"),
    "torch1": lambda t: parse_training(t, "pytorch eager"),
    "rust_gpu": parse_gpu,
    "torch_gpu": parse_gpu,
    "rust_ppo": parse_ppo,
    "torch_ppo": parse_ppo,
}


def optional(command):
    try:
        return command()
    except RuntimeError:
        return ""


def measure(rounds):
    raw = {key: [] for key in PARSERS}
    for _ in range(rounds):
        raw["rust_matmul"].append(cargo("--bin", "matmul"))
        raw["torch_matmul"].append(run("python", "benchmarks/pytorch_matmul.py"))
        raw["rust_training"].append(cargo())
        raw["torch8"].append(run("python", "benchmarks/pytorch_step.py"))
        raw["torch1"].append(run("python", "benchmarks/pytorch_step.py", "1"))
        raw["rust_gpu"].append(optional(lambda: cargo("--bin", "gpu")))
        raw["torch_gpu"].append(optional(lambda: run("python", "benchmarks/pytorch_gpu.py")))
        raw["rust_ppo"].append(cargo("--bin", "ppo"))
        raw["torch_ppo"].append(run("python", "benchmarks/pytorch_ppo.py"))
        RAW.write_text(json.dumps(raw, indent=1), encoding="utf-8")
    return raw


def main():
    global TORCH_VERSION, TORCH_THREADS
    rounds_argument = next((a for a in sys.argv[1:] if a.isdigit()), None)
    OUT.mkdir(parents=True, exist_ok=True)
    if "--reparse" in sys.argv:
        raw = json.loads(RAW.read_text(encoding="utf-8"))
    else:
        raw = measure(int(rounds_argument) if rounds_argument else 3)
    rounds = len(raw["rust_matmul"])
    best = {key: best_over_runs([PARSERS[key](text) for text in texts]) for key, texts in raw.items() if all(texts)}
    version = re.search(r"torch ([\d.]+).*threads: (\d+)", raw["torch8"][0])
    TORCH_VERSION, TORCH_THREADS = version.groups() if version else ("", "")
    stamp = f"{time.strftime('%Y-%m-%d')} · best of {rounds} interleaved rounds · float32"
    matmul_chart(best["rust_matmul"], best["torch_matmul"], stamp)
    training_chart(best["rust_training"], best["torch8"], best["torch1"], stamp)
    if "rust_gpu" in best and "torch_gpu" in best:
        adapter = re.search(r"adapter: (.+)", raw["rust_gpu"][0])
        gpu_chart(best["rust_gpu"], best["torch_gpu"], stamp + (" · " + adapter.group(1) if adapter else ""))
    else:
        print("GPU comparison skipped: no wgpu adapter or no CUDA PyTorch")
    ppo_chart(best["rust_ppo"], best["torch_ppo"], stamp)
    for name, value in best.items():
        print(name, value)


if __name__ == "__main__":
    main()
