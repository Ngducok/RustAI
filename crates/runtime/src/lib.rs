mod dispatch;
mod error;
mod planned;
mod reference;

pub use error::{Result, RuntimeError};
pub use planned::Executor;
pub use reference::run_reference;

pub type Feeds<'a> = &'a [(&'a str, &'a rt_tensor::Tensor)];
