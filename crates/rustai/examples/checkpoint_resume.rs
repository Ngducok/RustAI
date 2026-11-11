//! Saving and resuming training. The checkpoint is a standard `safetensors` file, so it can also be
//! opened from Python with `safetensors.numpy.load_file`.
//! Run with `cargo run --release -p rustai --example checkpoint_resume`.

use rustai::prelude::*;

fn main() -> Result<()> {
    let config = TrainerConfig {
        sizes: vec![1, 8, 1],
        batch: 16,
        learning_rate: 0.02,
        ..TrainerConfig::default()
    };
    let x = Tensor::from_vec((0..16).map(|i| i as f32 / 8.0 - 1.0).collect(), &[16, 1])?;
    let y = x.map(|v| v * v);

    let mut trainer = Trainer::new(config.clone())?;
    for _ in 0..200 {
        trainer.step(&x, &y)?;
    }
    let path = std::env::temp_dir().join("rustai-example.safetensors");
    trainer.save(&path)?;
    println!(
        "saved after {} steps to {}",
        trainer.steps(),
        path.display()
    );

    let mut resumed = Trainer::new(config)?;
    resumed.load(&path)?;
    let loss = resumed.step(&x, &y)?;
    println!("resumed at step {}, loss {loss:.5}", resumed.steps() - 1);
    std::fs::remove_file(&path).ok();
    Ok(())
}
