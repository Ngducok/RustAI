mod broadcast;
mod cse;
mod dce;
mod fusion;

pub use broadcast::fold_broadcasts;
pub use cse::eliminate_common_subexpressions;
pub use dce::eliminate_dead_code;
pub use fusion::{fuse_linear, fuse_relu_grad};

use rt_ir::{Graph, Result};

/// The default pipeline. Every pass preserves results bit for bit.
pub fn optimize(graph: &Graph) -> Result<Graph> {
    let folded = fold_broadcasts(graph)?;
    let merged = eliminate_common_subexpressions(&folded)?;
    let linear = fuse_linear(&merged)?;
    let fused = fuse_relu_grad(&linear)?;
    eliminate_dead_code(&fused)
}
