"""Verifies the library one stage at a time and renders the result as docs/images/stages.png.

Each stage lists the checks that establish it; a stage passes when none of them fail and at
least one test ran. Usage: python tools/check_stages.py [--quick] (quick skips slow learning tests)
"""
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images" / "stages.png"
FONTS = Path("C:/Windows/Fonts")
QUICK = "--quick" in sys.argv

BG, PANEL, INK, MUTED, GREEN, RED, AMBER, RULE = "#0d1117", "#161b22", "#e6edf3", "#8b949e", "#3fb950", "#f85149", "#d29922", "#30363d"


@dataclass
class Check:
    label: str
    command: list
    kind: str = "cargo"  # cargo | python | run
    env: dict = field(default_factory=dict)
    release_only: bool = False


@dataclass
class Result:
    label: str
    passed: int = 0
    failed: int = 0
    skipped: bool = False
    seconds: float = 0.0
    note: str = ""


STAGES = [
    ("Hygiene", "formatting and lints", [
        Check("rustfmt", ["cargo", "fmt", "--all", "--", "--check"], "run"),
        Check("clippy", ["cargo", "clippy", "--workspace", "--all-targets", "--all-features"], "lint"),
    ]),
    ("1  Tensor (CPU)", "kernels, broadcasting, GEMM", [
        Check("unit and kernel tests", ["cargo", "test", "-p", "rt-tensor"]),
    ]),
    ("2  Autograd and training", "gradients, optimizers, parity with PyTorch", [
        Check("autograd, optim, nn", ["cargo", "test", "-p", "rt-autograd", "-p", "rt-optim", "-p", "rt-nn"]),
        Check("PyTorch parity", ["cargo", "test", "-p", "rt-parity", "--test", "pytorch"]),
    ]),
    ("3  IR and runtime", "graph, autodiff, passes, planner, executor", [
        Check("ir, compiler, memory, runtime", ["cargo", "test", "-p", "rt-ir", "-p", "rt-compiler", "-p", "rt-memory", "-p", "rt-runtime"]),
        Check("IR parity with PyTorch", ["cargo", "test", "-p", "rt-parity", "--test", "ir"]),
    ]),
    ("4  GPU backend", "WGSL kernels checked against the CPU", [
        Check("gpu", ["cargo", "test", "-p", "rt-gpu", "--", "--nocapture"]),
        Check("example: gpu_training", ["cargo", "run", "--release", "-q", "-p", "rustai", "--features", "gpu", "--example", "gpu_training"], "run"),
    ]),
    ("5  Reinforcement learning", "env, GAE, PPO, masks, learning", [
        Check("rl", ["cargo", "test", "-p", "rt-rl"]),
        Check("RL parity", ["cargo", "test", "-p", "rt-parity", "--test", "rl"]),
        *([] if QUICK else [Check("PPO solves CartPole (release)", ["cargo", "test", "--release", "-p", "rt-rl", "--", "--ignored"])]),
    ]),
    ("6  Library packaging", "checkpoints, facade, examples, Python", [
        Check("checkpoint", ["cargo", "test", "-p", "rt-checkpoint"]),
        Check("facade and doctests", ["cargo", "test", "-p", "rustai"]),
        Check("safetensors interop", ["cargo", "test", "-p", "rt-parity", "--test", "checkpoint"]),
        Check("example: xor", ["cargo", "run", "--release", "-q", "-p", "rustai", "--example", "xor"], "run"),
        Check("example: checkpoint_resume", ["cargo", "run", "--release", "-q", "-p", "rustai", "--example", "checkpoint_resume"], "run"),
        Check("example: inspect", ["cargo", "run", "--release", "-q", "-p", "rustai", "--features", "gpu", "--example", "inspect"], "run"),
        Check("example: demo", ["cargo", "run", "--release", "-q", "-p", "rustai", "--features", "gpu", "--example", "demo", "--", "--out", "target/demo", "--steps", "200"], "run"),
        Check("build Python module",[sys.executable, "tools/build_python.py"], "run"),
        Check("Python tests", [sys.executable, "-m", "unittest", "discover", "-s", "python/tests"], "python", {"PYTHONPATH": "python"}),
    ]),
]


def execute(check):
    env = {**os.environ, **check.env}
    started = time.time()
    done = subprocess.run(check.command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
    text, result = done.stdout, Result(check.label, seconds=time.time() - started)
    if check.kind == "cargo":
        for m in re.finditer(r"test result: \w+\. (\d+) passed; (\d+) failed", text):
            result.passed += int(m.group(1))
            result.failed += int(m.group(2))
        result.skipped = "skipping:" in text
        adapter = re.search(r"GPU adapter: (.+)", text)
        result.note = adapter.group(1).strip() if adapter else ""
        if done.returncode != 0 and result.failed == 0:
            result.failed = 1
    elif check.kind == "python":
        ran = re.search(r"Ran (\d+) tests", text)
        result.passed = int(ran.group(1)) if ran and done.returncode == 0 else 0
        result.failed = 0 if done.returncode == 0 else 1
    elif check.kind == "lint":
        warnings = len(re.findall(r"^warning: (?!`)", text, re.M))
        result.passed, result.failed = int(done.returncode == 0 and warnings == 0), int(done.returncode != 0 or warnings > 0)
        result.note = "no warnings" if result.failed == 0 else f"{warnings} warnings"
    else:
        result.passed, result.failed = int(done.returncode == 0), int(done.returncode != 0)
    if result.failed:
        print(f"--- {check.label} failed ---\n{text[-2500:]}")
    return result


def font(name, size):
    return ImageFont.truetype(str(FONTS / name), size)


def render(rows, total_seconds):
    width, row, pad = 1280, 92, 36
    height = 150 + sum(row + 8 + 30 * len(checks) for _, _, checks in rows)
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    title, bold, body, small, mono = font("segoeuib.ttf", 28), font("segoeuib.ttf", 21), font("segoeui.ttf", 17), font("segoeui.ttf", 15), font("consola.ttf", 16)
    passed = sum(r.passed for _, _, rs in rows for r in rs)
    failed = sum(r.failed for _, _, rs in rows for r in rs)
    ok = failed == 0 and passed > 0
    d.text((pad, 26), "Library verification, stage by stage", font=title, fill=INK)
    d.text((pad, 68), f"{time.strftime('%Y-%m-%d %H:%M')} · {total_seconds:.0f}s total", font=small, fill=MUTED)
    badge = f"{passed} checks passed · {failed} failed"
    w = d.textlength(badge, font=title)
    d.rounded_rectangle([width - pad - w - 28, 22, width - pad, 66], 10, fill=PANEL, outline=GREEN if ok else RED, width=2)
    d.text((width - pad - w - 14, 26), badge, font=title, fill=GREEN if ok else RED)
    d.line([pad, 108, width - pad, 108], fill=RULE)
    y = 126
    for name, subtitle, results in rows:
        stage_ok = all(r.failed == 0 for r in results) and any(r.passed for r in results)
        d.ellipse([pad, y + 8, pad + 18, y + 26], fill=GREEN if stage_ok else RED)
        d.text((pad + 34, y), name, font=bold, fill=INK)
        d.text((pad + 34, y + 30), subtitle, font=small, fill=MUTED)
        count = sum(r.passed for r in results)
        label = f"{count} passed"
        d.text((width - pad - d.textlength(label, font=bold), y), label, font=bold, fill=GREEN if stage_ok else RED)
        d.text((width - pad - 120, y + 30), f"{sum(r.seconds for r in results):.1f}s", font=small, fill=MUTED)
        cy = y + 62
        for r in results:
            color = RED if r.failed else AMBER if r.skipped else GREEN
            d.ellipse([pad + 34, cy + 6, pad + 44, cy + 16], fill=color)
            text = r.label + (f"  ·  {r.note}" if r.note else "") + ("  ·  skipped (no adapter)" if r.skipped else "")
            d.text((pad + 56, cy), text, font=mono, fill=INK)
            d.text((width - pad - 220, cy), f"{r.passed} passed" + (f", {r.failed} FAILED" if r.failed else ""), font=small, fill=MUTED)
            cy += 30
        y = cy + 14
        d.line([pad, y - 6, width - pad, y - 6], fill=RULE)
    return img.crop((0, 0, width, y + 10))


def main():
    started = time.time()
    rows = []
    for name, subtitle, checks in STAGES:
        print(f"== {name}")
        results = []
        for check in checks:
            result = execute(check)
            print(f"   {check.label:<34} passed={result.passed} failed={result.failed} {result.seconds:5.1f}s {result.note}")
            results.append(result)
        rows.append((name, subtitle, results))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    render(rows, time.time() - started).save(OUT)
    summary = [{"stage": name, "checks": [r.__dict__ for r in results]} for name, _, results in rows]
    (ROOT / "docs" / "stages.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    failed = sum(r.failed for _, _, rs in rows for r in rs)
    print(f"{'ALL STAGES PASS' if not failed else 'FAILURES: ' + str(failed)} -> {OUT}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
