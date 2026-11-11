use rt_autograd::{Graph, Var};
use rt_tensor::Result;

pub fn mse(g: &mut Graph, prediction: Var, target: Var) -> Result<Var> {
    let diff = g.sub(prediction, target)?;
    let squared = g.mul(diff, diff)?;
    Ok(g.mean(squared))
}
