//! GPU throughput: matmul and full training steps with parameters resident on the device.
//! Run with `cargo run --release -p rt-bench --bin gpu`; compare with `benchmarks/pytorch_gpu.py`.

use std::time::Instant;

use rt_gpu::{GpuAdamW, GpuContext, GpuExecutor};
use rt_ir::Graph;
use rt_nn::{mlp_train_step, Loss, Mlp};
use rt_optim::{AdamW, Optimizer};
use rt_runtime::Executor;
use rt_tensor::{Rng, Tensor};

fn median_micros(rounds: usize, per_round: usize, mut round: impl FnMut()) -> f64 {
    (0..per_round).for_each(|_| round());
    let mut samples: Vec<f64> = (0..rounds)
        .map(|_| {
            let start = Instant::now();
            round();
            start.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn matmul_bench(ctx: &std::sync::Arc<GpuContext>) {
    println!("matmul (inputs resident on the device)");
    for n in [512usize, 1024, 2048, 4096] {
        let mut g = Graph::new();
        let a = g.input("a", &[n, n]).unwrap();
        let b = g.input("b", &[n, n]).unwrap();
        let c = g.matmul(a, b).unwrap();
        g.set_outputs(vec![c]).unwrap();
        let mut gpu = GpuExecutor::compile(ctx.clone(), &g).unwrap();
        let mut rng = Rng::new(1);
        let (x, y) = (
            Tensor::uniform(&[n, n], -1.0, 1.0, &mut rng),
            Tensor::uniform(&[n, n], -1.0, 1.0, &mut rng),
        );
        gpu.set_inputs(&[("a", &x), ("b", &y)]).unwrap();
        let micros = median_micros(15, 5, || {
            gpu.execute();
            gpu.synchronize();
        });
        println!(
            "  {n}x{n}x{n}: {micros:10.1} us  {:8.1} GFLOP/s",
            2.0 * (n * n * n) as f64 / micros / 1e3
        );
    }
}

struct Workload {
    name: &'static str,
    sizes: &'static [usize],
    batch: usize,
}

const WORKLOADS: [Workload; 3] = [
    Workload {
        name: "medium (batch 256,  128-256-256-10)",
        sizes: &[128, 256, 256, 10],
        batch: 256,
    },
    Workload {
        name: "large  (batch 2048, 512-1024-1024-10)",
        sizes: &[512, 1024, 1024, 10],
        batch: 2048,
    },
    Workload {
        name: "xlarge (batch 4096, 1024-2048-2048-10)",
        sizes: &[1024, 2048, 2048, 10],
        batch: 4096,
    },
];

fn training_bench(ctx: &std::sync::Arc<GpuContext>) {
    println!("training step: forward + MSE + backward + AdamW");
    for w in &WORKLOADS {
        let step = mlp_train_step(w.sizes, w.batch, Loss::Mse).unwrap();
        let mut rng = Rng::new(2);
        let model = Mlp::new(w.sizes, &mut rng);
        let out = *w.sizes.last().unwrap();
        let x = Tensor::uniform(&[w.batch, w.sizes[0]], -1.0, 1.0, &mut rng);
        let target = Tensor::uniform(&[w.batch, out], -1.0, 1.0, &mut rng);

        let mut gpu = GpuExecutor::compile(ctx.clone(), &step.graph).unwrap();
        let mut all: Vec<(&str, &Tensor)> = vec![("x", &x), ("target", &target)];
        all.extend(
            step.param_names
                .iter()
                .map(String::as_str)
                .zip(model.params()),
        );
        gpu.set_inputs(&all).unwrap();
        let updates: Vec<(&str, usize)> = step
            .param_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i + 1))
            .collect();
        gpu.enable_adamw(
            &updates,
            GpuAdamW {
                learning_rate: 1e-3,
                ..GpuAdamW::default()
            },
        )
        .unwrap();
        let steps_per_round = 10;
        let gpu_micros = median_micros(15, 3, || {
            for _ in 0..steps_per_round {
                gpu.execute();
            }
            gpu.synchronize();
        }) / steps_per_round as f64;

        let mut cpu_model = Mlp::new(w.sizes, &mut Rng::new(2));
        let mut cpu = Executor::compile(&step.graph).unwrap();
        let mut optimizer = AdamW::new(1e-3);
        let mut grads: Vec<Tensor> = cpu_model
            .params_mut()
            .iter()
            .map(|p| Tensor::zeros(p.shape()))
            .collect();
        let cpu_micros = median_micros(15, 3, || {
            let mut params = cpu_model.params_mut();
            let mut feeds = vec![("x", &x), ("target", &target)];
            feeds.extend(
                step.param_names
                    .iter()
                    .map(String::as_str)
                    .zip(params.iter().map(|p| &**p)),
            );
            cpu.run(&feeds).unwrap();
            grads
                .iter_mut()
                .enumerate()
                .for_each(|(i, g)| cpu.read_output(i + 1, g));
            optimizer.step(&mut params, &grads).unwrap();
        });
        println!("  {}", w.name);
        println!("    rust gpu (resident) {gpu_micros:10.1} us/step");
        println!("    rust cpu (12 threads) {cpu_micros:8.1} us/step");
    }
}

fn main() {
    let ctx = GpuContext::new().expect("a GPU is required for this benchmark");
    println!("adapter: {} ({})", ctx.adapter_name(), ctx.backend());
    matmul_bench(&ctx);
    training_bench(&ctx);
}
