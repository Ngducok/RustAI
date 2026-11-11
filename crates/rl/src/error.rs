use std::fmt;

use rt_ir::IrError;
use rt_runtime::RuntimeError;
use rt_tensor::TensorError;

#[derive(Debug, Clone, PartialEq)]
pub enum RlError {
    Invalid(String),
    Graph(IrError),
    Runtime(RuntimeError),
    Tensor(TensorError),
}

impl fmt::Display for RlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(m) => write!(f, "invalid argument: {m}"),
            Self::Graph(e) => write!(f, "{e}"),
            Self::Runtime(e) => write!(f, "{e}"),
            Self::Tensor(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RlError {}

impl From<IrError> for RlError {
    fn from(e: IrError) -> Self {
        Self::Graph(e)
    }
}

impl From<RuntimeError> for RlError {
    fn from(e: RuntimeError) -> Self {
        Self::Runtime(e)
    }
}

impl From<TensorError> for RlError {
    fn from(e: TensorError) -> Self {
        Self::Tensor(e)
    }
}

pub type Result<T> = std::result::Result<T, RlError>;
