mod error;
mod gemm;
pub mod kernels;
mod rng;
mod shape;
mod tensor;

pub use error::{Result, TensorError};
pub use rng::Rng;
pub use shape::{broadcast_shape, broadcast_strides, contiguous_strides, Strides, MAX_RANK};
pub use tensor::Tensor;
