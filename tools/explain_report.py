"""Draws the figures of docs/PRINCIPLES.md from data the library itself produces.

Usage: python tools/explain_report.py [--out DIR]   (default DIR: docs)
Runs `cargo run --example inspect` (writes target/inspect.json, prints the numbers quoted in the
document to docs/inspect-output.txt) and renders docs/images/principle-{autodiff,memory,gpu}.png.
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
DOCS = Path(sys.argv[sys.argv.index("--out") + 1]).resolve() if "--out" in sys.argv else ROOT / "docs"
OUT = DOCS / "images"
OUT.mkdir(parents=True, exist_ok=True)
BG, PANEL, INK, MUTED, GRID = "#0d1117", "#161b22", "#e6edf3", "#8b949e", "#30363d"
PHASE = {"input": "#8b949e", "forward": "#3fb950", "backward": "#d29922"}
KERNEL = {"matmul": "#3fb950", "reduce": "#58a6ff", "rows": "#bc8cff", "unary": "#d29922", "binary": "#f85149", "ternary": "#db61a2", "adamw": "#39c5cf"}


def inspect():
    done = subprocess.run(
        ["cargo", "run", "--release", "-q", "-p", "rustai", "--features", "gpu", "--example", "inspect", "--", "--json", "target/inspect.json"],
        cwd=ROOT, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if done.returncode != 0:
        sys.exit(done.stdout)
    (DOCS / "inspect-output.txt").write_text(done.stdout.replace("\r\n", "\n"), encoding="utf-8")
    return json.loads((ROOT / "target" / "inspect.json").read_text())


def style(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=INK, fontsize=11.5, loc="left", fontweight="bold")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def heading(fig, title, subtitle):
    fig.text(0.04, 0.955, title, color=INK, fontsize=18, fontweight="bold")
    fig.text(0.04, 0.915, subtitle, color=MUTED, fontsize=10)


def short(op):
    return op.replace("Input(", "").rstrip(")") if op.startswith("Input(") else op


def autodiff_figure(walk):
    nodes = walk["nodes"]
    depth = {}
    for n in nodes:
        depth[n["id"]] = 1 + max((depth[i] for i in n["inputs"]), default=-1)
    lanes = {"input": 2.2, "forward": 0.9, "backward": -1.3}
    stacked, positions = {}, {}
    for n in nodes:
        key = (n["phase"], depth[n["id"]])
        offset = stacked.get(key, 0)
        stacked[key] = offset + 1
        positions[n["id"]] = (depth[n["id"]] * 1.45, lanes[n["phase"]] - offset * 0.62)

    fig = plt.figure(figsize=(16, 7.6), facecolor=BG)
    heading(fig, "Reverse-mode autodiff builds new graph nodes",
            "loss = mean(relu(x*w + b)^2). The gradient pass is appended to the same graph, so the compiler optimises and plans it like any other node.")
    ax = fig.add_axes([0.03, 0.06, 0.62, 0.82])
    ax.set_facecolor(BG)
    ax.axis("off")
    for n in nodes:
        for source in n["inputs"]:
            (x0, y0), (x1, y1) = positions[source], positions[n["id"]]
            ax.annotate("", xy=(x1 - 0.5, y1), xytext=(x0 + 0.5, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0.12"))
    for n in nodes:
        x, y = positions[n["id"]]
        ax.add_patch(FancyBboxPatch((x - 0.5, y - 0.24), 1.0, 0.48, boxstyle="round,pad=0.02,rounding_size=0.08", fc=PANEL, ec=PHASE[n["phase"]], lw=1.6))
        ax.text(x, y + 0.07, f"%{n['id']} {short(n['op'])}", ha="center", va="center", color=INK, fontsize=8.5, fontweight="bold")
        ax.text(x, y - 0.12, str(n["shape"]), ha="center", va="center", color=MUTED, fontsize=7.5)
    ax.set_xlim(-0.8, max(p[0] for p in positions.values()) + 0.8)
    ax.set_ylim(min(p[1] for p in positions.values()) - 0.5, 2.8)
    for label, phase, y in [("inputs", "input", 2.75), ("forward pass", "forward", 1.65), ("backward pass, added by gradients()", "backward", -0.65)]:
        ax.text(0.95 if phase == "forward" else -0.75, y, label, color=PHASE[phase], fontsize=10, fontweight="bold")

    table = fig.add_axes([0.68, 0.06, 0.30, 0.82])
    style(table, "Autodiff vs finite differences")
    table.axis("off")
    table.text(0, 0.97, "parameter     autodiff      (f(x+e)-f(x-e))/2e", color=MUTED, fontsize=9, family="monospace", transform=table.transAxes)
    for i, row in enumerate(walk["check"]):
        table.text(0, 0.92 - i * 0.045, f"{row['parameter']:<8}{row['autodiff']:>12.6f}{row['finite_difference']:>20.6f}", color=INK, fontsize=9.5, family="monospace", transform=table.transAxes)
    table.text(0, 0.92 - len(walk["check"]) * 0.045 - 0.04, f"largest difference {walk['largest_difference']:.1e}  (e = 0.01, float32)", color=PHASE["forward"], fontsize=9.5, transform=table.transAxes)
    notes = [
        "The chain rule, one node at a time:",
        "  ReluGrad(%4, g) = g where the input of ReLU was > 0",
        "  MatMul gives MatMulNT / MatMulTN for its two inputs",
        "  Add of a bias gives SumToShape (the broadcast reversed)",
        "  Mul(h, h) emits two products that are then added;",
        "  common-subexpression elimination later merges them",
    ]
    for i, line in enumerate(notes):
        table.text(0, 0.40 - i * 0.05, line, color=INK if i == 0 else MUTED, fontsize=9, transform=table.transAxes, fontweight="bold" if i == 0 else "normal")
    fig.savefig(OUT / "principle-autodiff.png", dpi=130, facecolor=BG)
    plt.close(fig)


def mib(elements):
    return elements * 4 / 1048576


def size_label(elements):
    size = elements * 4
    return f"{size / 1024:.0f} KiB" if size >= 1024 else f"{size} B"


def memory_figure(cases):
    fig = plt.figure(figsize=(16, 9.4), facecolor=BG)
    heading(fig, "The memory planner reuses buffers once a value is dead",
            "A value lives from the step that computes it to its last reader. Buffers are assigned by a linear scan over the schedule; graph outputs are never recycled.")
    for column, case in enumerate(cases):
        ax = fig.add_axes([0.05 + column * 0.318, 0.55, 0.28, 0.31])
        style(ax, case["name"])
        live = [mib(v) for v in case["live_elements"]]
        steps = range(len(live))
        ax.fill_between(steps, live, step="mid", color="#58a6ff", alpha=0.45, label="live values")
        ax.axhline(mib(case["planned_elements"]), color="#3fb950", lw=2, label=f"planned buffers {mib(case['planned_elements']):.2f} MiB")
        ax.axhline(mib(case["unplanned_elements"]), color="#f85149", lw=1.5, ls="--", label=f"one buffer per value {mib(case['unplanned_elements']):.2f} MiB")
        ax.set_xlabel("schedule step (node index)", color=MUTED, fontsize=9)
        ax.set_ylabel("MiB", color=MUTED, fontsize=9)
        ax.set_ylim(0, mib(case["unplanned_elements"]) * 1.35)
        ax.grid(True, color=GRID, lw=0.5)
        leg = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=8, loc="upper left")
        for t in leg.get_texts():
            t.set_color(INK)
        ratio = case["unplanned_elements"] / case["planned_elements"]
        ax.text(0.98, 0.04, f"{ratio:.1f}x smaller   peak live {mib(case['peak_live_elements']):.2f} MiB", transform=ax.transAxes, ha="right", color=INK, fontsize=9)

    for column, index in enumerate([0, 2]):
        case = cases[index]
        span = [0.05, 0.05 + 0.318 * 1]
        ax = fig.add_axes([0.115 + column * 0.455, 0.08, 0.37 if index == 0 else 0.40, 0.36])
        style(ax, f"Buffer assignment: {case['name']}")
        count = len(case["buffer_sizes"])
        for n in case["nodes"]:
            if n["buffer"] is None:
                continue
            end = n["last_use"] if n["last_use"] is not None else len(case["nodes"]) - 1
            ax.add_patch(Rectangle((n["id"] - 0.4, n["buffer"] - 0.4), max(end - n["id"], 0) + 0.8, 0.8, fc=PHASE[n["phase"]], ec=BG, lw=0.8, alpha=0.9))
            if end - n["id"] >= 1:
                ax.text(n["id"] - 0.3, n["buffer"], str(n["id"]), color=BG, fontsize=6.5, va="center", fontweight="bold")
        ax.set_xlim(-1, len(case["nodes"]))
        ax.set_ylim(-0.7, count - 0.3)
        ax.set_yticks(range(count))
        ax.set_yticklabels([f"buffer {b}  ({size_label(s)})" for b, s in enumerate(case["buffer_sizes"])], fontsize=8, color=MUTED)
        ax.invert_yaxis()
        ax.set_xlabel("schedule step; bar = lifetime of the value in that buffer", color=MUTED, fontsize=9)
        ax.grid(True, axis="x", color=GRID, lw=0.5)
    fig.text(0.05, 0.015, "green: forward value, amber: backward value. Bars end at the last reader; an output bar runs to the end of the schedule.", color=MUTED, fontsize=9)
    fig.savefig(OUT / "principle-memory.png", dpi=130, facecolor=BG)
    plt.close(fig)


def gpu_figure(case, schedule, sizes):
    by_node = {n["id"]: n for n in case["nodes"]}
    params = 2 * (len(sizes) - 1)
    fig = plt.figure(figsize=(16, 7.2), facecolor=BG)
    heading(fig, "How the GPU runs the graph",
            "The optimised graph is compiled once into a list of kernel launches with their bind groups prepared. Every step then replays that list in one compute pass.")
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.82])
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)

    def box(x, y, w, h, label, color, sub=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.8", fc=PANEL, ec=color, lw=1.8))
        ax.text(x + w / 2, y + h - 2.0 if sub else y + h / 2, label, ha="center", va="center", color=INK, fontsize=9.5, fontweight="bold")
        if sub:
            ax.text(x + w / 2, y + (h - 3.6) / 2, sub, ha="center", va="center", color=MUTED, fontsize=8)

    box(1, 44, 22, 11, "host (once)", "#58a6ff", "upload x, labels, weights; build\npipelines and bind groups")
    box(1, 27, 22, 11, "host (every step)", "#58a6ff", "write the AdamW uniforms (step,\nbias corrections), submit; read\nthe outputs only when asked")
    box(1, 6, 22, 15, "device buffers", "#3fb950", f"{len(case['buffer_sizes'])} planned buffers from the\nmemory plan + inputs + AdamW\nmoments; nothing is allocated\nafter compile")
    launches = len(schedule)
    ax.text(27, 56.2, f"one compute pass, {launches} graph kernels + {params} AdamW kernels", color=INK, fontsize=11, fontweight="bold")
    columns = 4
    width = 17.2
    for i, k in enumerate(schedule):
        row, col = divmod(i, columns)
        x, y = 27 + col * (width + 1.0), 46 - row * 9.5
        node = by_node[k["node"]]
        color = KERNEL[k["kernel"]]
        ax.add_patch(FancyBboxPatch((x, y), width, 8.4, boxstyle="round,pad=0.1,rounding_size=0.7", fc=PANEL, ec=color, lw=1.8))
        ax.text(x + 0.9, y + 6.2, f"{i + 1}. {k['kernel']}", color=color, fontsize=9.5, fontweight="bold")
        ax.text(x + 0.9, y + 3.6, f"node %{k['node']}  {short(node['op'])}", color=INK, fontsize=8.2)
        ax.text(x + 0.9, y + 1.2, f"{node['shape']}  groups {k['groups'][0]}x{k['groups'][1]}x{k['groups'][2]}", color=MUTED, fontsize=7.8)
    rows_used = (launches + columns - 1) // columns
    y = 46 - rows_used * 9.5
    ax.add_patch(FancyBboxPatch((27, y), width * 4 + 3, 7.6, boxstyle="round,pad=0.1,rounding_size=0.7", fc=PANEL, ec=KERNEL["adamw"], lw=1.8))
    ax.text(28, y + 5.6, f"+ {params} x adamw   (one dispatch per parameter tensor, 256 threads per workgroup)", color=KERNEL["adamw"], fontsize=9.5, fontweight="bold")
    ax.text(28, y + 3.2, "m = b1*m + (1-b1)*g,  v = b2*v + (1-b2)*g^2,  p -= lr * (m/c1 / (sqrt(v/c2) + eps) + wd*p)   reads the gradient straight from its planned buffer", color=INK, fontsize=8)
    ax.text(28, y + 1.0, "parameters never leave the device between steps", color=MUTED, fontsize=8)
    ax.annotate("", xy=(23.5, 13), xytext=(23.5, 30), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.2))
    ax.text(27, 4.6, "Order is the topological order of the graph. WebGPU inserts the barriers between dispatches of one pass, so a kernel sees the writes of the kernels before it.", color=MUTED, fontsize=8.5)
    ax.text(27, 1.6, "Per step the CPU sends one command buffer and about 1.5 KB of uniforms; outputs (loss, gradients) are copied back only when download() is called.", color=MUTED, fontsize=8.5)
    fig.savefig(OUT / "principle-gpu.png", dpi=130, facecolor=BG)
    plt.close(fig)


def main():
    data = inspect()
    autodiff_figure(data["walkthrough"])
    memory_figure(data["cases"])
    gpu_figure(data["cases"][0], data["gpu_schedule"], data["cases"][0]["sizes"])
    print("wrote", ", ".join(f"principle-{n}.png" for n in ("autodiff", "memory", "gpu")))


if __name__ == "__main__":
    main()
