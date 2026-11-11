//! Throughput of the matmul kernel on training-sized shapes, single-threaded and with all threads.
//! Run with `cargo run --release -p rt-bench --bin matmul`; compare with `benchmarks/pytorch_matmul.py`.

use std::time::Instant;

use rt_tensor::kernels::matmul_into;
use rt_tensor::{Rng, Tensor};

const SHAPES: [(usize, usize, usize); 8] = [
    (32, 64, 64),
    (64, 64, 64),
    (128, 128, 128),
    (256, 128, 256),
    (256, 256, 256),
    (512, 512, 512),
    (256, 256, 10),
    (1024, 1024, 1024),
];

fn median_micros(m: usize, k: usize, n: usize) -> f64 {
    let mut rng = Rng::new(7);
    let a = Tensor::uniform(&[m, k], -1.0, 1.0, &mut rng);
    let b = Tensor::uniform(&[k, n], -1.0, 1.0, &mut rng);
    let mut out = vec![0.0; m * n];
    let iterations = (4e8 / (2.0 * (m * k * n) as f64)).clamp(5.0, 3000.0) as usize;
    let mut run = || matmul_into(a.data(), b.data(), m, k, n, &mut out).unwrap();
    (0..iterations / 10 + 1).for_each(|_| run());
    let mut micros: Vec<f64> = (0..iterations)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    micros.sort_by(f64::total_cmp);
    micros[micros.len() / 2]
}

fn main() {
    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    println!(
        "{:<18} {:>10} {:>10} {:>10} {:>10}",
        "shape (m,k,n)", "1T us", "1T GFLOP/s", "all us", "all GFLOP/s"
    );
    for (m, k, n) in SHAPES {
        let one = single.install(|| median_micros(m, k, n));
        let all = median_micros(m, k, n);
        let gflops = |micros: f64| 2.0 * (m * k * n) as f64 / micros / 1e3;
        println!(
            "{:<18} {one:>10.1} {:>10.2} {all:>10.1} {:>10.2}",
            format!("{m}x{k}x{n}"),
            gflops(one),
            gflops(all)
        );
    }
}
