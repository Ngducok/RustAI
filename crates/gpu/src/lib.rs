//! A GPU backend that runs optimised IR graphs as WGSL compute shaders through `wgpu`
//! (Vulkan, Direct3D 12 or Metal).
//!
//! It supports every op of the IR, including the reinforcement-learning ops, and reuses the CPU
//! compiler passes and the memory planner, so a graph compiled for the CPU compiles unchanged for
//! the GPU.

mod context;
mod error;
mod executor;
mod kernels;

pub use context::GpuContext;
pub use error::{GpuError, Result};
pub use executor::{GpuAdamW, GpuExecutor};
pub use kernels::{schedule, KernelLaunch};
