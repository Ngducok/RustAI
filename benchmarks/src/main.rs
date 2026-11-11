//! Per-step training latency of the three execution paths on identical MLP + MSE + AdamW work.
//! Run with `cargo run --release -p rt-bench`; compare with `benchmarks/pytorch_step.py`.

use std::time::Instant;

use rt_autograd::Graph;
use rt_nn::{mlp_train_step, mse, Loss, Mlp};
use rt_optim::{AdamW, Optimizer};
use rt_runtime::{run_reference, Executor};
use rt_tensor::{Rng, Tensor};

struct Workload {
    name: &'static str,
    sizes: &'static [usize],
    batch: usize,
    steps: usize,
}

const WORKLOADS: [Workload; 3] = [
    Workload {
        name: "tiny   (batch 8,   4-32-32-2)",
        sizes: &[4, 32, 32, 2],
        batch: 8,
        steps: 3000,
    },
    Workload {
        name: "small  (batch 32,  16-64-64-4)",
        sizes: &[16, 64, 64, 4],
        batch: 32,
        steps: 1500,
    },
    Workload {
        name: "medium (batch 256, 128-256-256-10)",
        sizes: &[128, 256, 256, 10],
        batch: 256,
        steps: 200,
    },
];

fn report(label: &str, mut micros: Vec<f64>) {
    micros.sort_by(f64::total_cmp);
    let (median, p95) = (micros[micros.len() / 2], micros[micros.len() * 95 / 100]);
    println!("  {label:<14} median {median:>9.1} us   p95 {p95:>9.1} us");
}

fn timed(steps: usize, mut step: impl FnMut()) -> Vec<f64> {
    (0..steps / 10).for_each(|_| step());
    (0..steps)
        .map(|_| {
            let start = Instant::now();
            step();
            start.elapsed().as_secs_f64() * 1e6
        })
        .collect()
}

fn main() {
    println!(
        "threads available: {}",
        std::thread::available_parallelism().map_or(1, |n| n.get())
    );
    for w in &WORKLOADS {
        println!("{}", w.name);
        let out_dim = *w.sizes.last().unwrap();
        let mut rng = Rng::new(1);
        let x = Tensor::uniform(&[w.batch, w.sizes[0]], -1.0, 1.0, &mut rng);
        let target = Tensor::uniform(&[w.batch, out_dim], -1.0, 1.0, &mut rng);
        let fresh_model = || Mlp::new(w.sizes, &mut Rng::new(2));

        let mut model = fresh_model();
        let mut opt = AdamW::new(1e-3);
        report(
            "dynamic tape",
            timed(w.steps, || {
                let mut g = Graph::new();
                let bound = model.bind(&mut g);
                let input = g.constant(x.clone());
                let out = bound.forward(&mut g, input).unwrap();
                let t = g.constant(target.clone());
                let loss = mse(&mut g, out, t).unwrap();
                let grads = g.backward(loss).unwrap().take_all(&bound.params()).unwrap();
                opt.step(&mut model.params_mut(), &grads).unwrap();
            }),
        );

        let step = mlp_train_step(w.sizes, w.batch, Loss::Mse).unwrap();
        let mut model = fresh_model();
        let mut opt = AdamW::new(1e-3);
        report(
            "reference IR",
            timed(w.steps, || {
                let mut params = model.params_mut();
                let mut feeds = vec![("x", &x), ("target", &target)];
                feeds.extend(
                    step.param_names
                        .iter()
                        .map(String::as_str)
                        .zip(params.iter().map(|p| &**p)),
                );
                let outputs = run_reference(&step.graph, &feeds).unwrap();
                opt.step(&mut params, &outputs[1..]).unwrap();
            }),
        );

        let mut model = fresh_model();
        let mut opt = AdamW::new(1e-3);
        let mut executor = Executor::compile(&step.graph).unwrap();
        let mut grads: Vec<Tensor> = model
            .params_mut()
            .iter()
            .map(|p| Tensor::zeros(p.shape()))
            .collect();
        report(
            "planned IR",
            timed(w.steps, || {
                let mut params = model.params_mut();
                let mut feeds = vec![("x", &x), ("target", &target)];
                feeds.extend(
                    step.param_names
                        .iter()
                        .map(String::as_str)
                        .zip(params.iter().map(|p| &**p)),
                );
                executor.run(&feeds).unwrap();
                grads
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, g)| executor.read_output(i + 1, g));
                opt.step(&mut params, &grads).unwrap();
            }),
        );
        println!(
            "  intermediate buffers: {} elements planned vs {} unplanned",
            executor.planned_elements(),
            executor.unplanned_elements()
        );
    }
}
