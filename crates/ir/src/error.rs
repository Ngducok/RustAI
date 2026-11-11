use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum IrError {
    Shape(String),
    Invalid(String),
}

impl fmt::Display for IrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(m) => write!(f, "shape error: {m}"),
            Self::Invalid(m) => write!(f, "invalid graph: {m}"),
        }
    }
}

impl std::error::Error for IrError {}

pub type Result<T> = std::result::Result<T, IrError>;
