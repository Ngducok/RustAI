# Reproducing the results

Everything the README claims can be regenerated from a fresh clone with one command. This page
says what you need, what the command does, what counts as success, and what is and is not expected
to match.

## Requirements

| Needed for | Requirement |
|---|---|
| Everything | a stable Rust toolchain (checked with 1.99.1); on Windows the Visual Studio C++ build tools |
| The Python tooling (fixtures, comparison with PyTorch, charts, extension tests) | Python with `pip install -r requirements.txt` (checked with Python 3.10, PyTorch 2.6) |
| GPU tests, the GPU part of the demo and benchmarks | a GPU that `wgpu` can use (Vulkan, Direct3D 12 or Metal). Without one, GPU tests skip and the GPU columns are left out |
| The GPU benchmark baseline | PyTorch built with CUDA. Without it that comparison is skipped |

The Rust library has no Python dependency; Python is used only to produce reference values and
figures.

## One command

```bash
git clone <this repository> && cd rustai
pip install -r requirements.txt
python tools/reproduce.py            # a few minutes more than --quick (adds the learning tests and a second benchmark round)
python tools/reproduce.py --quick    # skips the learning tests, one benchmark round; about 6 minutes on a laptop including the first build
```

Options: `--skip-bench`, `--skip-python` (no extension build). Everything generated goes to
`target/reproduced/`, so the checkout stays clean; the exit status is non-zero if any step fails.

| Step | What runs | Pass criterion |
|---|---|---|
| environment | `rustc`, `cargo`, Python modules | all present |
| Rust tests | `cargo test --workspace --exclude rt-python` | no failures, at least one test ran |
| learning tests | PPO solves CartPole and a masked corridor, in release mode (skipped by `--quick`) | no failures |
| Python extension | build, import, 7 tests against PyTorch and `safetensors` | all pass |
| explain | `examples/inspect.rs`: autodiff on a worked example and on the demo network against finite differences, the memory plan, the GPU kernel schedule; draws the figures of [PRINCIPLES.md](PRINCIPLES.md) | finite-difference error below 1e-4 on the worked example and 1e-2 on the network |
| demo | trains the same network in Rust and PyTorch from the same weights and data ([demo.rs](../crates/rustai/examples/demo.rs), [pytorch_demo.py](../benchmarks/pytorch_demo.py)) | final loss below 1e-3 in both; loss curves within 5e-3 relative; step-0 gradients within 1e-4 of PyTorch's; decision regions agree on at least 98% of grid cells; Rust test accuracy at least 95% |
| benchmarks | matmul, training step, GPU, PPO, Rust and PyTorch interleaved ([benchmark_report.py](../tools/benchmark_report.py)) | the run completes and charts are produced |

## What to expect

**Deterministic** (same inputs, same result): the data, the initial weights, the Rust loss and
gradient curves. The CPU GEMM accumulates in a fixed order, so the Rust CPU numbers do not depend on
the number of threads. They can differ in the last bits between a CPU with AVX2/FMA and one without,
because the two kernels round differently.

**Matching PyTorch** within the criteria above, not bit for bit: the two frameworks sum in different
orders. The agreement is the point of the demo, and the third panel of `demo.png` shows how large
the difference actually is.

**Machine-dependent**: every time and throughput. The script prints speed ratios but never
asserts them. The ratios quoted in the README are from one laptop (Intel Core i5-13420H, RTX 4050
Laptop GPU, PyTorch 2.6 eager); on other hardware expect the same direction for small and medium
networks but different magnitudes. The benchmarks interleave Rust and PyTorch runs and report the
best of several rounds, because background load only ever slows a run down; close other heavy
programs before measuring.

## Running the pieces by hand

```bash
cargo test --workspace --exclude rt-python                          # Rust tests
cargo test --release --workspace --exclude rt-python -- --ignored   # learning tests
python tools/build_python.py && PYTHONPATH=python python -m unittest discover -s python/tests

cargo run --release -p rustai --features gpu --example inspect      # prints the autodiff walkthrough and the memory plan
python tools/explain_report.py                                      # figures of docs/PRINCIPLES.md

python tools/demo_report.py                                         # Rust vs PyTorch training, docs/images/demo.png
cargo run --release -p rustai --features gpu --example demo -- --out target/demo --gpu   # the Rust half only

python tools/benchmark_report.py 2                                  # two interleaved rounds, docs/images/benchmark-*.png
python tools/check_stages.py                                        # every stage in turn, docs/images/stages.png
python tests/parity/generate.py                                     # regenerate the PyTorch/NumPy fixtures
```

`tools/demo_report.py`, `explain_report.py` and `benchmark_report.py` accept `--out DIR` to write
somewhere other than `docs/`.

## If something fails

* **`link.exe` not found (Windows).** Install the "Desktop development with C++" workload of the
  Visual Studio Build Tools.
* **Python module errors.** `pip install -r requirements.txt`. The Rust tests do not need Python,
  except `rt-parity`'s interoperability test, which reads files with the Python `safetensors` package.
* **The Python extension does not import.** It is built for the interpreter that built it
  (`tools/build_python.py` uses the one that runs it); rebuild after changing Python versions.
* **GPU steps say "skipped".** No adapter was found. On Linux install a Vulkan driver
  (for example Mesa's `mesa-vulkan-drivers`).
* **Demo criteria fail on the loss difference.** Check that the Rust and PyTorch runs used the same
  `target/demo/initial.safetensors`; `demo_report.py` guarantees it, but a hand-run does not.
* **Benchmark ratios look odd.** Background load. Close other programs and add rounds
  (`python tools/benchmark_report.py 4`).

## Last verified

A copy of this repository without its `target/` directory, made after the final change, was built
and run from scratch with `python tools/reproduce.py --quick` on the machine described above. The
console output of that run is in [reproduce-output.txt](reproduce-output.txt). The Rust loss, gradient-norm
and decision-region results of the demo were bit-identical to those of the original checkout. The full mode adds
only the learning tests and a second benchmark round; those were run separately
(`tools/check_stages.py`, `tools/benchmark_report.py`), not as part of that clean-copy run. The
command has not been run
on another operating system or on GitHub Actions; the workflow in `.github/workflows/ci.yml` is
written to run the same command but has not executed yet.
