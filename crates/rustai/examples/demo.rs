//! Trains an MLP on three interleaved spirals and records everything `tools/demo_report.py`
//! needs to compare the run with PyTorch: loss, gradient norm and time of every step, the
//! gradients of step 0, and the decision regions of the trained network.
//!
//! `cargo run --release -p rustai --features gpu --example demo -- --out target/demo [--steps 1500] [--gpu]`
//!
//! It writes `initial.safetensors` (the starting weights, which `benchmarks/pytorch_demo.py` loads
//! so both frameworks start from the same point) and `rust.json`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rustai::prelude::*;
use serde_json::{json, Value};

const CLASSES: usize = 3;
const PER_CLASS: usize = 170;
const GRID: usize = 100;
const EXTENT: f32 = 1.2;

struct Options {
    out: PathBuf,
    steps: usize,
    gpu: bool,
}

fn options() -> Options {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
    };
    Options {
        out: PathBuf::from(value("--out").map_or("target/demo", String::as_str)),
        steps: value("--steps")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1500),
        gpu: args.iter().any(|a| a == "--gpu"),
    }
}

fn spirals(rng: &mut Rng) -> rustai::Result<(Tensor, Tensor)> {
    let mut points = Vec::new();
    let mut labels = Vec::new();
    for class in 0..CLASSES {
        for _ in 0..PER_CLASS {
            let t = rng.next_f32();
            let radius = 0.1 + 0.9 * t;
            let angle = class as f32 * std::f32::consts::TAU / CLASSES as f32
                + 3.0 * t
                + rng.uniform(-0.12, 0.12);
            points.extend([radius * angle.cos(), radius * angle.sin()]);
            labels.push(class as f32);
        }
    }
    let n = labels.len();
    Ok((
        Tensor::from_vec(points, &[n, 2])?,
        Tensor::from_vec(labels, &[n])?,
    ))
}

fn accuracy(predictions: &[usize], labels: &Tensor) -> f64 {
    let hits = predictions
        .iter()
        .zip(labels.data())
        .filter(|(&p, &l)| p == l as usize)
        .count();
    hits as f64 / predictions.len() as f64
}

fn predict_all(trainer: &mut Trainer, x: &Tensor) -> rustai::Result<Vec<usize>> {
    let batch = trainer.config().batch;
    let mut classes = Vec::with_capacity(x.shape()[0]);
    for chunk in x.data().chunks(batch * 2) {
        let rows = chunk.len() / 2;
        let part = Tensor::from_vec(chunk.to_vec(), &[rows, 2])?;
        classes.extend(trainer.predict(&part)?.argmax_rows()?);
    }
    Ok(classes)
}

fn grid() -> rustai::Result<Tensor> {
    let mut points = Vec::with_capacity(GRID * GRID * 2);
    for row in 0..GRID {
        for col in 0..GRID {
            let at = |i: usize| -EXTENT + 2.0 * EXTENT * i as f32 / (GRID - 1) as f32;
            points.extend([at(col), at(GRID - 1 - row)]);
        }
    }
    Ok(Tensor::from_vec(points, &[GRID * GRID, 2])?)
}

fn norm(tensors: &[Tensor]) -> f64 {
    tensors
        .iter()
        .flat_map(|t| t.data())
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn floats(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

#[cfg(feature = "gpu")]
fn train_on_gpu(
    config: &TrainerConfig,
    initial: &Path,
    x: &Tensor,
    labels: &Tensor,
    grid_points: &Tensor,
    test: &(Tensor, Tensor),
    steps: usize,
) -> rustai::Result<Value> {
    use rustai::gpu::{GpuAdamW, GpuContext, GpuError, GpuExecutor};
    use rustai::nn::mlp_train_step;

    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => return Ok(Value::Null),
        Err(e) => return Err(Error::Invalid(e.to_string())),
    };
    let (adapter, backend) = (ctx.adapter_name().to_string(), ctx.backend().to_string());
    let device_error = |e: GpuError| Error::Invalid(e.to_string());

    let mut start = Trainer::new(config.clone())?;
    start.load(initial)?;
    let step = mlp_train_step(&config.sizes, config.batch, config.loss)?;
    let mut gpu = GpuExecutor::compile(ctx, &step.graph).map_err(device_error)?;
    let mut feeds: Vec<(&str, &Tensor)> = vec![("x", x), ("target", labels)];
    feeds.extend(
        step.param_names
            .iter()
            .map(String::as_str)
            .zip(start.model().params()),
    );
    gpu.set_inputs(&feeds).map_err(device_error)?;
    let updates: Vec<(&str, usize)> = step
        .param_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i + 1))
        .collect();
    gpu.enable_adamw(
        &updates,
        GpuAdamW {
            learning_rate: config.learning_rate,
            weight_decay: config.weight_decay,
            ..GpuAdamW::default()
        },
    )
    .map_err(device_error)?;

    let (mut losses, mut micros) = (Vec::with_capacity(steps), Vec::with_capacity(steps));
    for _ in 0..steps {
        let started = Instant::now();
        gpu.execute();
        gpu.download().map_err(device_error)?;
        micros.push(started.elapsed().as_secs_f64() * 1e6);
        losses.push(f64::from(gpu.output(0)[0]));
    }

    let mut trained = Trainer::new(config.clone())?;
    for (name, param) in step
        .param_names
        .iter()
        .zip(trained.model_mut().params_mut())
    {
        *param = gpu.read_input(name).map_err(device_error)?;
    }
    let on_data = predict_all(&mut trained, x)?;
    let on_grid = predict_all(&mut trained, grid_points)?;
    Ok(json!({
        "adapter": adapter,
        "backend": backend,
        "loss": losses,
        "micros": micros,
        "accuracy": accuracy(&on_data, labels),
        "test_accuracy": accuracy(&predict_all(&mut trained, &test.0)?, &test.1),
        "grid": on_grid,
    }))
}

#[cfg(not(feature = "gpu"))]
fn train_on_gpu(
    _: &TrainerConfig,
    _: &Path,
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: &(Tensor, Tensor),
    _: usize,
) -> rustai::Result<Value> {
    println!("built without the `gpu` feature; skipping the GPU run");
    Ok(Value::Null)
}

fn main() -> rustai::Result<()> {
    let options = options();
    std::fs::create_dir_all(&options.out).map_err(|e| Error::Invalid(e.to_string()))?;
    let (x, labels) = spirals(&mut Rng::new(2024))?;
    let test = spirals(&mut Rng::new(99))?;
    let config = TrainerConfig {
        sizes: vec![2, 64, 64, CLASSES],
        batch: x.shape()[0],
        loss: Loss::CrossEntropy,
        learning_rate: 5e-3,
        weight_decay: 0.01,
        seed: 1,
    };
    let mut trainer = Trainer::new(config.clone())?;
    let initial = options.out.join("initial.safetensors");
    trainer.save(&initial)?;
    let grid_points = grid()?;

    let mut losses = Vec::new();
    let mut norms = Vec::new();
    let mut micros = Vec::new();
    let mut accuracies = Vec::new();
    let mut first_grads = Value::Null;
    for step in 0..options.steps {
        let started = Instant::now();
        let loss = trainer.step(&x, &labels)?;
        micros.push(started.elapsed().as_secs_f64() * 1e6);
        losses.push(f64::from(loss));
        norms.push(norm(trainer.gradients()));
        if step == 0 {
            let names = trainer.model().param_names();
            first_grads = Value::Array(
                names
                    .iter()
                    .zip(trainer.gradients())
                    .map(|(name, g)| json!({"name": name, "shape": g.shape(), "data": floats(g.data())}))
                    .collect(),
            );
        }
        if step % 25 == 0 || step + 1 == options.steps {
            let hits = accuracy(&predict_all(&mut trainer, &x)?, &labels);
            accuracies.push(json!([step, hits]));
        }
    }
    let on_data = predict_all(&mut trainer, &x)?;
    let on_grid = predict_all(&mut trainer, &grid_points)?;
    let total_seconds: f64 = micros.iter().sum::<f64>() / 1e6;

    let gpu = if options.gpu {
        train_on_gpu(
            &config,
            &initial,
            &x,
            &labels,
            &grid_points,
            &test,
            options.steps,
        )?
    } else {
        Value::Null
    };

    let document = json!({
        "config": {
            "sizes": config.sizes,
            "batch": config.batch,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "steps": options.steps,
            "threads": std::thread::available_parallelism().map_or(1, usize::from),
        },
        "data": {
            "x": floats(x.data()),
            "labels": floats(labels.data()),
            "test_x": floats(test.0.data()),
            "test_labels": floats(test.1.data()),
        },
        "grid": {"size": GRID, "extent": EXTENT},
        "rust": {
            "loss": losses,
            "grad_norm": norms,
            "micros": micros,
            "total_seconds": total_seconds,
            "accuracy_curve": accuracies,
            "final_accuracy": accuracy(&on_data, &labels),
            "test_accuracy": accuracy(&predict_all(&mut trainer, &test.0)?, &test.1),
            "grid_prediction": on_grid,
        },
        "first_grads": first_grads,
        "gpu": gpu,
    });
    let path = options.out.join("rust.json");
    let text = serde_json::to_string(&document).map_err(|e| Error::Invalid(e.to_string()))?;
    std::fs::write(&path, text).map_err(|e| Error::Invalid(e.to_string()))?;

    println!(
        "{} steps in {total_seconds:.3} s; loss {:.4} -> {:.4}; train accuracy {:.1}%, test accuracy {:.1}%",
        options.steps,
        losses[0],
        losses[losses.len() - 1],
        100.0 * accuracy(&on_data, &labels),
        100.0 * accuracy(&predict_all(&mut trainer, &test.0)?, &test.1)
    );
    println!("wrote {}", path.display());
    Ok(())
}
