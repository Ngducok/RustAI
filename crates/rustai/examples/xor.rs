//! Supervised learning: train a small network on XOR with the high-level `Trainer`.
//! Run with `cargo run --release -p rustai --example xor`.

use rustai::prelude::*;

fn main() -> Result<()> {
    let mut trainer = Trainer::new(TrainerConfig {
        sizes: vec![2, 16, 2],
        batch: 4,
        loss: Loss::CrossEntropy,
        learning_rate: 0.05,
        ..TrainerConfig::default()
    })?;
    let x = Tensor::from_vec(vec![0., 0., 0., 1., 1., 0., 1., 1.], &[4, 2])?;
    let labels = Tensor::from_vec(vec![0., 1., 1., 0.], &[4])?;

    for step in 0..=300 {
        let loss = trainer.step(&x, &labels)?;
        if step % 100 == 0 {
            println!("step {step:3}  loss {loss:.4}");
        }
    }
    println!("predictions: {:?}", trainer.predict(&x)?.argmax_rows()?);
    Ok(())
}
