# Architecture

The library turns a training step into a graph, optimises it, plans its memory once, and then
runs it repeatedly out of preallocated buffers, on the CPU or the GPU.

```
 Tensor + kernels ──► IR (typed graph) ──► autodiff ──► passes ──► memory plan ──► executor
   (rt-tensor)         (rt-ir)             (rt-ir)    (rt-compiler)  (rt-memory)   (rt-runtime / rt-gpu)
```

[PRINCIPLES.md](PRINCIPLES.md) explains why the autodiff, the memory planner and the GPU executor
work, with figures drawn from the library's own output; [REPRODUCE.md](REPRODUCE.md) explains how to
regenerate every result.

## Crates

| Directory | Package | Role |
|---|---|---|
| `crates/tensor` | `rt-tensor` | f32 tensor, broadcasting, allocation-free kernels, blocked GEMM |
| `crates/autograd` | `rt-autograd` | dynamic reverse-mode engine (kept as an independent second implementation) |
| `crates/ir` | `rt-ir` | typed graph, shape inference, autodiff that emits graph nodes |
| `crates/compiler` | `rt-compiler` | broadcast folding, CSE, `Linear` and masked-product fusion, dead-code elimination |
| `crates/memory` | `rt-memory` | liveness analysis and buffer planning |
| `crates/runtime` | `rt-runtime` | reference executor and the planned, allocation-free executor |
| `crates/gpu` | `rt-gpu` | WGSL compute kernels through `wgpu`, on-device AdamW |
| `crates/nn`, `crates/optim` | `rt-nn`, `rt-optim` | MLP builders, bucketed inference, SGD and AdamW |
| `crates/rl` | `rt-rl` | environments, GAE, PPO |
| `crates/checkpoint` | `rt-checkpoint` | `safetensors` reader and writer |
| `crates/rustai` | `rustai` | the facade crate: `Trainer`, re-exports, prelude |
| `crates/python` | `rt-python` | Python bindings (PyO3) |

## Design decisions

**One implementation per operation.** Every heavy operation is a function over slices
(`zip_into`, `matmul_into`, `reduce_into`, ...). `Tensor` methods allocate an output and call the
same function, so the dynamic engine, the reference executor and the planned executor all share
kernels.

**Optimisations preserve results bit for bit.** Broadcast folding, CSE and fusion are exact: fused
epilogues run on a finished dot product, in the same order as the separate operations would.
Tests compare optimised graphs with unoptimised ones on random graphs and on training steps and
require identical bits.

**Deterministic GEMM.** Every output element accumulates in ascending-k order regardless of
blocking, thread count or operand layout, so a transposed operand gives bit-identical results to an
explicit transpose, and results do not depend on parallelism. On CPUs with AVX2 and FMA a faster
kernel is chosen at run time; it rounds once per product, so its last bit can differ from the
portable kernel.

**Memory is planned, not allocated.** Liveness analysis and a greedy linear scan assign every
intermediate to one of a few buffers. A training step after warm-up makes no allocations.

**Reinforcement learning reuses the pipeline.** The PPO loss is an IR graph with `LogSoftmax`,
`Exp`, `Clamp` and `Minimum`, differentiated and compiled like any other. Illegal actions are masked
by a large negative logit, so their probability is exactly zero.

## How each stage is checked

`python tools/check_stages.py` runs the checks of every stage in turn and writes
`docs/images/stages.png`. The sources of truth are independent of this code: PyTorch for gradients,
optimizers and PPO loss; NumPy for GAE and CartPole dynamics; the Python `safetensors` library for
checkpoints; and the CPU executor for the GPU kernels.
