use std::fmt;

use rt_ir::IrError;
use rt_runtime::RuntimeError;
use rt_tensor::TensorError;

#[derive(Debug)]
pub enum GpuError {
    /// No suitable adapter (graphics card) was found.
    NoAdapter,
    Device(String),
    Graph(IrError),
    Input(RuntimeError),
    Tensor(TensorError),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no GPU adapter is available"),
            Self::Device(m) => write!(f, "GPU error: {m}"),
            Self::Graph(e) => write!(f, "{e}"),
            Self::Input(e) => write!(f, "{e}"),
            Self::Tensor(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<IrError> for GpuError {
    fn from(e: IrError) -> Self {
        Self::Graph(e)
    }
}

impl From<RuntimeError> for GpuError {
    fn from(e: RuntimeError) -> Self {
        Self::Input(e)
    }
}

impl From<TensorError> for GpuError {
    fn from(e: TensorError) -> Self {
        Self::Tensor(e)
    }
}

pub type Result<T> = std::result::Result<T, GpuError>;
