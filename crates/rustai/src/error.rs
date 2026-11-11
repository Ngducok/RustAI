use std::fmt;

/// Any error the library can return.
#[derive(Debug)]
pub enum Error {
    Invalid(String),
    Tensor(rt_tensor::TensorError),
    Graph(rt_ir::IrError),
    Runtime(rt_runtime::RuntimeError),
    Checkpoint(rt_checkpoint::CheckpointError),
    Rl(rt_rl::RlError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(m) => write!(f, "invalid argument: {m}"),
            Self::Tensor(e) => write!(f, "{e}"),
            Self::Graph(e) => write!(f, "{e}"),
            Self::Runtime(e) => write!(f, "{e}"),
            Self::Checkpoint(e) => write!(f, "{e}"),
            Self::Rl(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {}

macro_rules! convert {
    ($($source:ty => $variant:ident),* $(,)?) => {
        $(impl From<$source> for Error {
            fn from(e: $source) -> Self {
                Self::$variant(e)
            }
        })*
    };
}

convert! {
    rt_tensor::TensorError => Tensor,
    rt_ir::IrError => Graph,
    rt_runtime::RuntimeError => Runtime,
    rt_checkpoint::CheckpointError => Checkpoint,
    rt_rl::RlError => Rl,
}

pub type Result<T> = std::result::Result<T, Error>;
