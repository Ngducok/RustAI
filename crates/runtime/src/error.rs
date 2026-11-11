use std::fmt;

use rt_ir::IrError;
use rt_tensor::TensorError;

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeError {
    MissingInput(String),
    InputShape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    Kernel(TensorError),
    Graph(IrError),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingInput(name) => write!(f, "input {name} was not provided"),
            Self::InputShape {
                name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "input {name} has shape {actual:?}, expected {expected:?}"
                )
            }
            Self::Kernel(e) => write!(f, "kernel failed: {e}"),
            Self::Graph(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<TensorError> for RuntimeError {
    fn from(e: TensorError) -> Self {
        Self::Kernel(e)
    }
}

impl From<IrError> for RuntimeError {
    fn from(e: IrError) -> Self {
        Self::Graph(e)
    }
}

pub type Result<T> = std::result::Result<T, RuntimeError>;
