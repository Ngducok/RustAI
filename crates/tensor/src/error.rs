use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    ShapeMismatch(String),
    InvalidArgument(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch(m) => write!(f, "shape mismatch: {m}"),
            Self::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
        }
    }
}

impl std::error::Error for TensorError {}

pub type Result<T> = std::result::Result<T, TensorError>;
