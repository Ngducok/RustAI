"""Checks a fresh clone end to end: environment, tests, training against PyTorch, benchmarks.

Usage: python tools/reproduce.py [--quick] [--skip-bench] [--skip-python]
  --quick        skip the slow learning tests and run one benchmark round instead of two
  --skip-bench   do not run the benchmarks
  --skip-python  do not build or test the Python extension

Everything it generates goes to target/reproduced/, so a checkout stays clean. It exits with a
non-zero status if any step fails; pass criteria are stated next to each step and in
docs/REPRODUCE.md. Performance numbers are reported, never asserted: they depend on the machine.
"""
import importlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "target" / "reproduced"
QUICK, SKIP_BENCH, SKIP_PYTHON = "--quick" in sys.argv, "--skip-bench" in sys.argv, "--skip-python" in sys.argv
REQUIRED_MODULES = ["numpy", "torch", "safetensors", "matplotlib", "PIL"]

results = []


def execute(command, **kwargs):
    done = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
    return done.returncode, done.stdout


def step(name):
    """Runs the decorated function, which returns (ok, detail); a raised error counts as a failure."""
    def decorate(function):
        print(f"\n== {name}", flush=True)
        started = time.time()
        try:
            ok, detail = function()
        except Exception as error:  # noqa: BLE001 - report and continue with the next step
            ok, detail = False, f"{type(error).__name__}: {error}"
        seconds = time.time() - started
        print(f"   {'PASS' if ok else 'FAIL'}  {detail}  ({seconds:.0f}s)", flush=True)
        results.append({"step": name, "ok": ok, "detail": detail, "seconds": round(seconds, 1)})
        return function
    return decorate


def cargo_totals(text):
    found = re.findall(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
    return sum(int(p) for p, _, _ in found), sum(int(f) for _, f, _ in found), sum(int(i) for _, _, i in found)


@step("environment")
def _():
    lines = []
    for command in (["rustc", "--version"], ["cargo", "--version"]):
        code, text = execute(command)
        if code != 0:
            return False, f"`{' '.join(command)}` not found: install Rust from https://rustup.rs"
        lines.append(text.strip())
    missing = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if missing:
        return False, f"missing Python modules {missing}: run `pip install -r requirements.txt`"
    import torch
    lines.append(f"python {sys.version.split()[0]}, torch {torch.__version__}")
    return True, "; ".join(lines)


if not results[-1]["ok"]:
    print("\nFix the environment first.")
    sys.exit(1)


@step("Rust tests (cargo test)")
def _():
    code, text = execute(["cargo", "test", "--workspace", "--exclude", "rt-python"])
    passed, failed, _ = cargo_totals(text)
    if code != 0 or failed or not passed:
        print(text[-3000:])
    skipped = "skipping:" in text
    return code == 0 and failed == 0 and passed > 0, f"{passed} passed, {failed} failed" + (" (GPU tests skipped: no adapter)" if skipped else "")


if not QUICK:
    @step("learning tests: PPO solves CartPole and the masked corridor (release)")
    def _():
        code, text = execute(["cargo", "test", "--release", "--workspace", "--exclude", "rt-python", "--", "--ignored"])
        passed, failed, _ = cargo_totals(text)
        if code != 0 or failed:
            print(text[-3000:])
        return code == 0 and failed == 0 and passed > 0, f"{passed} passed, {failed} failed"


if not SKIP_PYTHON:
    @step("Python extension: build, import, tests against PyTorch and safetensors")
    def _():
        code, text = execute([sys.executable, "tools/build_python.py"])
        if code != 0:
            print(text[-3000:])
            return False, "the extension did not build"
        import os
        code, text = execute([sys.executable, "-m", "unittest", "discover", "-s", "python/tests"], env={**os.environ, "PYTHONPATH": "python"})
        ran = re.search(r"Ran (\d+) tests", text)
        if code != 0:
            print(text[-3000:])
        return code == 0, f"{ran.group(1) if ran else 0} tests"


@step("explain: autodiff check, memory plan and GPU schedule of a real training step")
def _():
    code, text = execute([sys.executable, "tools/explain_report.py", "--out", str(OUT)])
    if code != 0:
        print(text[-3000:])
        return False, "the inspect example failed"
    data = json.loads((ROOT / "target" / "inspect.json").read_text())
    worst = max(c["relative_error"] for c in data["gradient_check"])
    walk = data["walkthrough"]["largest_difference"]
    ok = worst < 1e-2 and walk < 1e-4
    return ok, f"autodiff vs finite differences: worst {worst:.1e} (MLP), {walk:.1e} (worked example)"


@step("demo: train in Rust and PyTorch from identical weights and data")
def _():
    code, text = execute([sys.executable, "tools/demo_report.py", "--out", str(OUT)])
    if code != 0:
        print(text[-3000:])
        return False, "the demo failed"
    s = json.loads((OUT / "demo-summary.json").read_text())
    worst_grad = max(d["max_relative_to_largest"] for d in s["step0_gradient_difference"])
    checks = {
        "final loss below 1e-3 in both": max(s["loss_last"].values()) < 1e-3,
        "loss curves within 5e-3 relative": s["max_relative_loss_difference"] < 5e-3,
        "step-0 gradients within 1e-4 of PyTorch's": worst_grad < 1e-4,
        "decision regions agree on 98% of cells": s["grid_agreement"] >= 0.98,
        "Rust test accuracy at least 95%": s["test_accuracy"]["rust"] >= 0.95,
    }
    for name, ok in checks.items():
        print(f"      {'ok  ' if ok else 'FAIL'} {name}")
    speed = s["median_step_micros"]["torch"] / s["median_step_micros"]["rust"]
    return all(checks.values()), (
        f"loss {s['loss_last']['rust']:.1e} vs {s['loss_last']['torch']:.1e}, step-0 gradient diff {worst_grad:.1e}, "
        f"step {s['median_step_micros']['rust']:.0f} us vs PyTorch {s['median_step_micros']['torch']:.0f} us ({speed:.1f}x)"
    )


if not SKIP_BENCH:
    @step("benchmarks: Rust against PyTorch (matmul, training step, GPU, PPO)")
    def _():
        rounds = "1" if QUICK else "2"
        code, text = execute([sys.executable, "tools/benchmark_report.py", rounds, "--out", str(OUT / "images"), "--raw", str(OUT / "benchmark-raw.json")])
        if code != 0:
            print(text[-3000:])
            return False, "the benchmark run failed"
        sys.path.insert(0, str(ROOT / "tools"))
        sys.argv = sys.argv[:1]
        import benchmark_report as br
        raw = json.loads((OUT / "benchmark-raw.json").read_text())
        best = {key: br.best_over_runs([br.PARSERS[key](t) for t in texts]) for key, texts in raw.items() if all(texts)}
        medium = best["rust_training"]["medium"]["planned IR"], best["torch8"]["medium"]["pytorch eager"]
        tiny = best["rust_training"]["tiny"]["planned IR"], best["torch8"]["tiny"]["pytorch eager"]
        matmul_one = best["rust_matmul"]["512x512x512"][1] / best["torch_matmul"]["512x512x512"][1]
        ppo = best["rust_ppo"][1] / best["torch_ppo"][1]
        parts = [
            f"train step tiny {tiny[1] / tiny[0]:.1f}x, medium {medium[1] / medium[0]:.1f}x faster than PyTorch eager",
            f"1-thread 512^3 matmul at {100 * matmul_one:.0f}% of PyTorch",
            f"PPO {ppo:.1f}x env steps/s",
        ]
        if "rust_gpu" not in best:
            parts.append("GPU comparison skipped (no adapter or no CUDA PyTorch)")
        produced = sorted(p.name for p in (OUT / "images").glob("benchmark-*.png"))
        return len(produced) >= 3, "; ".join(parts)


print("\n== summary")
for r in results:
    print(f"   {'PASS' if r['ok'] else 'FAIL'}  {r['step']}: {r['detail']}")
failed = [r for r in results if not r["ok"]]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "report.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print(f"\n{'ALL STEPS PASSED' if not failed else str(len(failed)) + ' STEP(S) FAILED'}; figures and data are in {OUT}")
sys.exit(1 if failed else 0)
