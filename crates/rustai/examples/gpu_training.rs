//! Training on the GPU with parameters and AdamW resident on the device.
//! Run with `cargo run --release -p rustai --features gpu --example gpu_training`.

use rustai::gpu::{GpuAdamW, GpuContext, GpuError, GpuExecutor};
use rustai::nn::{mlp_train_step, Loss, Mlp};
use rustai::prelude::*;

fn main() -> rustai::Result<()> {
    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(GpuError::NoAdapter) => {
            println!("no GPU adapter on this machine; nothing to do");
            return Ok(());
        }
        Err(e) => return Err(Error::Invalid(e.to_string())),
    };
    println!("training on {} ({})", ctx.adapter_name(), ctx.backend());

    let sizes = [2, 16, 2];
    let step = mlp_train_step(&sizes, 4, Loss::CrossEntropy)?;
    let model = Mlp::new(&sizes, &mut Rng::new(42));
    let x = Tensor::from_vec(vec![0., 0., 0., 1., 1., 0., 1., 1.], &[4, 2])?;
    let labels = Tensor::from_vec(vec![0., 1., 1., 0.], &[4])?;

    let gpu_err = |e: rustai::gpu::GpuError| Error::Invalid(e.to_string());
    let mut gpu = GpuExecutor::compile(ctx, &step.graph).map_err(gpu_err)?;
    let mut inputs: Vec<(&str, &Tensor)> = vec![("x", &x), ("target", &labels)];
    inputs.extend(
        step.param_names
            .iter()
            .map(String::as_str)
            .zip(model.params()),
    );
    gpu.set_inputs(&inputs).map_err(gpu_err)?;
    let gradients: Vec<(&str, usize)> = step
        .param_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i + 1))
        .collect();
    gpu.enable_adamw(
        &gradients,
        GpuAdamW {
            learning_rate: 0.05,
            ..GpuAdamW::default()
        },
    )
    .map_err(gpu_err)?;

    for i in 0..=300 {
        gpu.execute();
        if i % 100 == 0 {
            gpu.download().map_err(gpu_err)?;
            println!("step {i:3}  loss {:.4}", gpu.output(0)[0]);
        }
    }
    Ok(())
}
