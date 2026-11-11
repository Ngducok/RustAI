"""Trains the demo network in Rust and in PyTorch from identical weights and data, then renders
docs/images/demo.png and writes docs/demo-summary.json.

Usage: python tools/demo_report.py [--steps N] [--no-gpu] [--out DIR]   (default DIR: docs)
Needs numpy, matplotlib, safetensors and torch. The Rust side is `cargo run --example demo`.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent.parent
DOCS = Path(sys.argv[sys.argv.index("--out") + 1]).resolve() if "--out" in sys.argv else ROOT / "docs"
WORK = ROOT / "target" / "demo"
BG, PANEL, INK, MUTED, GRID = "#0d1117", "#161b22", "#e6edf3", "#8b949e", "#30363d"
RUST, TORCH, GPU = "#3fb950", "#d29922", "#58a6ff"
CLASS_COLORS = ["#f85149", "#3fb950", "#58a6ff"]


def run(*args):
    done = subprocess.run(args, cwd=ROOT, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if done.returncode != 0:
        sys.exit(done.stdout)
    return done.stdout


def median(values):
    return statistics.median(values)


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=INK, fontsize=12, loc="left", fontweight="bold")
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def legend(ax, **kw):
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=9, **kw)
    for text in leg.get_texts():
        text.set_color(INK)


def regions(ax, grid, predictions, data, labels, reference=None):
    size = grid["size"]
    extent = grid["extent"]
    image = np.array(predictions).reshape(size, size)
    ax.imshow(image, extent=[-extent, extent, -extent, extent], cmap=ListedColormap(CLASS_COLORS), alpha=0.28, interpolation="nearest")
    if reference is not None:
        differing = (image != np.array(reference).reshape(size, size)).astype(float)
        if differing.any():
            ax.imshow(np.ma.masked_equal(differing, 0), extent=[-extent, extent, -extent, extent], cmap=ListedColormap(["white"]), alpha=0.9, interpolation="nearest")
    points = np.array(data["x"]).reshape(-1, 2)
    ax.scatter(points[:, 0], points[:, 1], c=[CLASS_COLORS[int(c)] for c in labels], s=7, linewidths=0)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")


def render(rust_doc, torch_doc, out):
    config, grid, data = rust_doc["config"], rust_doc["grid"], rust_doc["data"]
    rust, torch_run, gpu = rust_doc["rust"], torch_doc, rust_doc.get("gpu")
    steps = np.arange(config["steps"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.6), facecolor=BG)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.845, bottom=0.07, wspace=0.22, hspace=0.34)

    ax = axes[0, 0]
    style(ax, "Loss", "step", "cross-entropy")
    ax.semilogy(steps, torch_run["loss"], color=TORCH, lw=3.2, alpha=0.55, label="PyTorch")
    ax.semilogy(steps, rust["loss"], color=RUST, lw=1.3, label="Rust CPU")
    if gpu:
        ax.semilogy(steps, gpu["loss"], color=GPU, lw=1.0, ls="--", label="Rust GPU")
    legend(ax)

    ax = axes[0, 1]
    style(ax, "Gradient norm", "step", "global L2 norm of all gradients")
    ax.semilogy(steps, torch_run["grad_norm"], color=TORCH, lw=3.2, alpha=0.55, label="PyTorch")
    ax.semilogy(steps, rust["grad_norm"], color=RUST, lw=1.3, label="Rust CPU")
    legend(ax)

    ax = axes[0, 2]
    style(ax, "Rust vs PyTorch, per step", "step", "|Rust loss - PyTorch loss| / PyTorch loss")
    relative = np.abs(np.array(rust["loss"]) - np.array(torch_run["loss"])) / np.array(torch_run["loss"])
    ax.semilogy(steps, np.maximum(relative, 1e-9), color=RUST, lw=0.9, label="CPU")
    if gpu:
        gpu_relative = np.abs(np.array(gpu["loss"]) - np.array(torch_run["loss"])) / np.array(torch_run["loss"])
        ax.semilogy(steps, np.maximum(gpu_relative, 1e-9), color=GPU, lw=0.9, label="GPU")
    ax.axhline(1e-3, color=MUTED, lw=0.8, ls=":")
    legend(ax, loc="lower right")

    ax = axes[1, 0]
    style(ax, "Time per training step", None, "microseconds (log scale)")
    series = [("PyTorch", torch_run["micros"], TORCH), ("Rust CPU", rust["micros"], RUST)]
    if gpu:
        series.append(("Rust GPU", gpu["micros"], GPU))
    parts = ax.violinplot([s[1] for s in series], showmedians=True, showextrema=False)
    for body, (_, _, color) in zip(parts["bodies"], series):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    parts["cmedians"].set_color(INK)
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(series) + 1))
    ax.set_xticklabels([f"{name}\nmedian {median(v):,.0f} us\ntotal {sum(v) / 1e6:.2f} s" for name, v, _ in series], color=INK, fontsize=9)

    ax = axes[1, 1]
    style(ax, "Rust: decision regions", "x", "y")
    regions(ax, grid, rust["grid_prediction"], data, data["labels"])

    ax = axes[1, 2]
    agreement = float(np.mean(np.array(rust["grid_prediction"]) == np.array(torch_run["grid_prediction"])))
    style(ax, "PyTorch: decision regions", "x", "y")
    regions(ax, grid, torch_run["grid_prediction"], data, data["labels"], reference=rust["grid_prediction"])
    ax.text(0.03, 0.04, f"{100 * agreement:.1f}% of cells agree with Rust", transform=ax.transAxes, color=INK, fontsize=8.5,
            bbox=dict(facecolor=BG, edgecolor=GRID, alpha=0.85, pad=3))

    worst = max(d["max_relative_to_largest"] for d in torch_run["step0_gradient_difference"])
    subtitle = (
        f"MLP {config['sizes']}, {config['batch']} points of 3 interleaved spirals, full-batch AdamW lr={config['learning_rate']:g} "
        f"weight decay={config['weight_decay']:g}, {config['steps']} steps, float32, identical initial weights and data.\n"
        f"Step-0 gradients differ by at most {worst:.1e} of their largest entry.   "
        f"Test accuracy: Rust {100 * rust['test_accuracy']:.0f}%, PyTorch {100 * torch_run['test_accuracy']:.0f}%."
    )
    fig.text(0.05, 0.965, "Training the same network in Rust and in PyTorch", color=INK, fontsize=19, fontweight="bold")
    fig.text(0.05, 0.905, subtitle, color=MUTED, fontsize=9.5, wrap=True)
    fig.text(0.05, 0.872, f"PyTorch {torch_run['torch']} on {torch_run['threads']} threads; Rust uses all {config['threads']} hardware threads"
             + (f"; GPU: {gpu['adapter']} via {gpu['backend']}" if gpu else ""), color=MUTED, fontsize=9.5)
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)
    return agreement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    cargo = ["cargo", "run", "--release", "-q", "-p", "rustai", "--features", "gpu", "--example", "demo", "--", "--out", str(WORK), "--steps", str(args.steps)]
    print(run(*(cargo + ([] if args.no_gpu else ["--gpu"]))).strip())
    print(run(sys.executable, "benchmarks/pytorch_demo.py", str(WORK)).strip())

    rust_doc = json.loads((WORK / "rust.json").read_text())
    torch_doc = json.loads((WORK / "torch.json").read_text())
    (DOCS / "images").mkdir(parents=True, exist_ok=True)
    image = DOCS / "images" / "demo.png"
    agreement = render(rust_doc, torch_doc, image)

    rust, gpu = rust_doc["rust"], rust_doc["gpu"]
    relative = np.abs(np.array(rust["loss"]) - np.array(torch_doc["loss"])) / np.array(torch_doc["loss"])
    summary = {
        "steps": rust_doc["config"]["steps"],
        "sizes": rust_doc["config"]["sizes"],
        "batch": rust_doc["config"]["batch"],
        "torch": torch_doc["torch"],
        "loss_first": {"rust": rust["loss"][0], "torch": torch_doc["loss"][0]},
        "loss_last": {"rust": rust["loss"][-1], "torch": torch_doc["loss"][-1]},
        "max_relative_loss_difference": float(relative.max()),
        "step0_gradient_difference": torch_doc["step0_gradient_difference"],
        "grid_agreement": agreement,
        "test_accuracy": {"rust": rust["test_accuracy"], "torch": torch_doc["test_accuracy"], "gpu": gpu["test_accuracy"] if gpu else None},
        "median_step_micros": {"rust": median(rust["micros"]), "torch": median(torch_doc["micros"]), "gpu": median(gpu["micros"]) if gpu else None},
        "total_seconds": {"rust": rust["total_seconds"], "torch": torch_doc["total_seconds"], "gpu": sum(gpu["micros"]) / 1e6 if gpu else None},
    }
    (DOCS / "demo-summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("max_relative_loss_difference", "grid_agreement", "median_step_micros", "total_seconds")}, indent=1))
    print(f"-> {image}")


if __name__ == "__main__":
    main()

