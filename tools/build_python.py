"""Builds the Python extension and places it at python/rustai.pyd (python/rustai.so elsewhere).

Usage: python tools/build_python.py [--debug]
Then: PYTHONPATH=python python -m unittest discover -s python/tests
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    profile = "debug" if "--debug" in sys.argv else "release"
    command = ["cargo", "build", "-p", "rt-python"] + (["--release"] if profile == "release" else [])
    subprocess.run(command, cwd=ROOT, check=True)
    if sys.platform == "win32":
        built, target = ROOT / "target" / profile / "rustai_py.dll", ROOT / "python" / "rustai.pyd"
    elif sys.platform == "darwin":
        built, target = ROOT / "target" / profile / "librustai_py.dylib", ROOT / "python" / "rustai.so"
    else:
        built, target = ROOT / "target" / profile / "librustai_py.so", ROOT / "python" / "rustai.so"
    shutil.copy2(built, target)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
