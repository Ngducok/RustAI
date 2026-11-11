//! A Rust-native AI training engine.
//!
//! The library is layered. [`tensor`] holds the CPU tensor and its kernels; [`autograd`] is a
//! dynamic reverse-mode engine; [`ir`], [`compiler`], [`memory`] and [`runtime`] turn a training
//! step into an optimised graph that runs out of preallocated buffers; [`nn`] and [`optim`] build
//! models on top; [`rl`] adds PPO; [`checkpoint`] reads and writes `safetensors` files.
//!
//! Most programs only need [`Trainer`] for supervised learning or [`rl::Ppo`] for reinforcement
//! learning:
//!
//! ```
//! use rustai::prelude::*;
//!
//! let mut trainer = Trainer::new(TrainerConfig {
//!     sizes: vec![2, 8, 2],
//!     batch: 4,
//!     loss: Loss::CrossEntropy,
//!     ..TrainerConfig::default()
//! })?;
//! let x = Tensor::from_vec(vec![0., 0., 0., 1., 1., 0., 1., 1.], &[4, 2])?;
//! let labels = Tensor::from_vec(vec![0., 1., 1., 0.], &[4])?;
//! let first = trainer.step(&x, &labels)?;
//! for _ in 0..200 {
//!     trainer.step(&x, &labels)?;
//! }
//! assert!(trainer.step(&x, &labels)? < first);
//! # Ok::<(), rustai::Error>(())
//! ```

mod error;
mod trainer;

pub use rt_autograd as autograd;
pub use rt_checkpoint as checkpoint;
pub use rt_compiler as compiler;
#[cfg(feature = "gpu")]
pub use rt_gpu as gpu;
pub use rt_ir as ir;
pub use rt_memory as memory;
pub use rt_nn as nn;
pub use rt_optim as optim;
pub use rt_rl as rl;
pub use rt_runtime as runtime;
pub use rt_tensor as tensor;

pub use error::{Error, Result};
pub use trainer::{Trainer, TrainerConfig};

/// The names most programs need.
pub mod prelude {
    pub use crate::{Error, Result, Trainer, TrainerConfig};
    pub use rt_nn::{Loss, Mlp};
    pub use rt_optim::{AdamW, Optimizer, Sgd};
    pub use rt_rl::{CartPole, Env, MaskedCorridor, Ppo, PpoConfig};
    pub use rt_tensor::{Rng, Tensor};
}
