mod gradcheck;
mod graph;

pub use gradcheck::{check_gradients, GradMismatch};
pub use graph::{Grads, Graph, Var};
