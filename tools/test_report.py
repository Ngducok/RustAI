"""Runs the test suite and renders the results as PNG images for the README.

Usage: python tools/test_report.py [--release]
Writes docs/images/test-summary.png and test-detail.png (with a -release suffix for --release).
"""
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
FONTS = Path("C:/Windows/Fonts")

BG, PANEL, INK, MUTED = "#0d1117", "#161b22", "#e6edf3", "#8b949e"
GREEN, RED, AMBER, BLUE, RULE = "#3fb950", "#f85149", "#d29922", "#58a6ff", "#30363d"


@dataclass
class Package:
    name: str
    passed: int = 0
    failed: int = 0
    ignored: int = 0
    seconds: float = 0.0
    tests: list = field(default_factory=list)


def font(name, size):
    return ImageFont.truetype(str(FONTS / name), size)


def run(*args):
    done = subprocess.run(
        args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
    )
    return done.returncode, done.stdout


def workspace_packages():
    _, out = run("cargo", "metadata", "--no-deps", "--format-version", "1")
    meta = json.loads(out[out.index("{"):])
    members = set(meta["workspace_members"])
    return [(p["name"], Path(p["manifest_path"]).parent.name) for p in meta["packages"] if p["id"] in members]


def parse(package, label, output):
    target = ""
    for line in output.splitlines():
        running = re.match(r"\s*Running (?:unittests )?(\S+)", line)
        if running:
            target = Path(running.group(1)).stem
            target = "unit" if target == "lib" or target == "main" else target
        test = re.match(r"test (\S+) \.\.\. (ok|FAILED|ignored)", line)
        if test:
            package.tests.append((target, test.group(1), test.group(2)))
        result = re.match(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored.*finished in ([\d.]+)s", line)
        if result:
            package.passed += int(result.group(1))
            package.failed += int(result.group(2))
            package.ignored += int(result.group(3))
            package.seconds += float(result.group(4))


def collect(release):
    packages = []
    for name, label in workspace_packages():
        package = Package(label)
        args = ["cargo", "test", "-p", name, "--no-fail-fast"] + (["--release"] if release else [])
        _, output = run(*args)
        parse(package, label, output)
        packages.append(package)
    return packages


def static_checks():
    clippy_code, clippy_out = run("cargo", "clippy", "--workspace", "--all-targets")
    warnings = len(re.findall(r"^warning: ", clippy_out, re.M)) - len(re.findall(r"^warning: `", clippy_out, re.M))
    fmt_code, _ = run("cargo", "fmt", "--all", "--", "--check")
    return [("clippy", clippy_code == 0 and warnings == 0, "no warnings" if warnings == 0 else f"{warnings} warnings"),
            ("rustfmt", fmt_code == 0, "formatted" if fmt_code == 0 else "needs formatting")]


def dot(draw, x, y, ok, r=6):
    color = MUTED if ok is None else GREEN if ok else RED
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def package_state(package):
    if package.failed:
        return False
    return True if package.passed else None


def summary_image(packages, checks, release, stamp):
    row, pad, width = 46, 36, 1200
    height = 150 + row * len(packages) + 40 + 60 + row * len(checks)
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    title, mono, body, small = font("segoeuib.ttf", 26), font("consola.ttf", 19), font("segoeui.ttf", 18), font("segoeui.ttf", 15)
    total_passed = sum(p.passed for p in packages)
    total_failed = sum(p.failed for p in packages)
    ok = total_failed == 0 and total_passed > 0

    d.text((pad, 28), "cargo test --workspace" + (" --release" if release else ""), font=title, fill=INK)
    d.text((pad, 66), stamp, font=small, fill=MUTED)
    badge = f"{total_passed} passed · {total_failed} failed"
    bw = d.textlength(badge, font=title)
    d.rounded_rectangle([width - pad - bw - 28, 24, width - pad, 66], 10, fill=PANEL, outline=GREEN if ok else RED, width=2)
    d.text((width - pad - bw - 14, 28), badge, font=title, fill=GREEN if ok else RED)

    d.line([pad, 110, width - pad, 110], fill=RULE, width=1)
    top = 126
    peak = max(p.passed for p in packages) or 1
    bar_x, bar_w = 360, 420
    for i, p in enumerate(packages):
        y = top + i * row
        dot(d, pad + 8, y + 17, package_state(p))
        d.text((pad + 28, y + 6), p.name, font=mono, fill=INK)
        d.rounded_rectangle([bar_x, y + 10, bar_x + bar_w, y + 24], 5, fill=PANEL)
        if p.passed:
            d.rounded_rectangle([bar_x, y + 10, bar_x + max(10, bar_w * p.passed // peak), y + 24], 5, fill=GREEN)
        if p.failed:
            d.text((bar_x + bar_w + 20, y + 4), f"{p.failed} failed", font=body, fill=RED)
        else:
            label = f"{p.passed} passed" if p.passed else "no tests"
            d.text((bar_x + bar_w + 20, y + 4), label, font=body, fill=INK if p.passed else MUTED)
        d.text((width - pad - 80, y + 4), f"{p.seconds:.2f}s", font=body, fill=MUTED)
    y = top + row * len(packages) + 18
    d.line([pad, y, width - pad, y], fill=RULE, width=1)
    d.text((pad, y + 14), "Static checks", font=font("segoeuib.ttf", 18), fill=MUTED)
    for i, (name, passed, note) in enumerate(checks):
        cy = y + 50 + i * row
        dot(d, pad + 8, cy + 17, passed)
        d.text((pad + 28, cy + 6), name, font=mono, fill=INK)
        d.text((bar_x, cy + 4), note, font=body, fill=INK)
    return img


def detail_image(packages, release, stamp):
    line, pad, width = 24, 36, 1300
    rows = sum(len(p.tests) + 2 for p in packages)
    img = Image.new("RGB", (width, 110 + rows * line + 30), BG)
    d = ImageDraw.Draw(img)
    title, mono, bold = font("segoeuib.ttf", 24), font("consola.ttf", 16), font("consolab.ttf", 17)
    d.text((pad, 24), "Test results" + (" (release)" if release else ""), font=title, fill=INK)
    d.text((pad, 60), stamp, font=font("segoeui.ttf", 15), fill=MUTED)
    y = 100
    for p in packages:
        if not p.tests:
            continue
        d.text((pad, y), f"{p.name}   {p.passed} passed", font=bold, fill=BLUE)
        y += line + 2
        for target, name, status in p.tests:
            color = {"ok": GREEN, "FAILED": RED}.get(status, AMBER)
            dot(d, pad + 12, y + 11, None if status == "ignored" else status == "ok", r=5)
            label = name if target == "unit" else f"{target} › {name}"
            d.text((pad + 30, y), label, font=mono, fill=INK if status == "ok" else color)
            y += line
        y += line // 2
    return img.crop((0, 0, width, y + 20))


def main():
    release = "--release" in sys.argv
    started = time.time()
    packages = collect(release)
    checks = static_checks()
    stamp = time.strftime("%Y-%m-%d %H:%M") + f" · {time.time() - started:.0f}s total"
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "-release" if release else ""
    summary_image(packages, checks, release, stamp).save(OUT / f"test-summary{suffix}.png")
    detail_image(packages, release, stamp).save(OUT / f"test-detail{suffix}.png")
    failed = sum(p.failed for p in packages)
    print(f"{sum(p.passed for p in packages)} passed, {failed} failed -> {OUT}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
