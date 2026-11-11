use std::fmt;

use rt_tensor::{Result, Tensor};

use crate::graph::{Graph, Var};

#[derive(Debug)]
pub struct GradMismatch {
    pub input: usize,
    pub index: usize,
    pub analytic: f32,
    pub numeric: f32,
}

impl fmt::Display for GradMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "input {} element {}: analytic {} vs numeric {}",
            self.input, self.index, self.analytic, self.numeric
        )
    }
}

impl std::error::Error for GradMismatch {}

/// Compares reverse-mode gradients of a scalar function against central differences.
pub fn check_gradients(
    f: impl Fn(&mut Graph, &[Var]) -> Result<Var>,
    inputs: &[Tensor],
    eps: f32,
    tol: f32,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let eval = |inputs: &[Tensor]| -> Result<f32> {
        let mut g = Graph::new();
        let vars: Vec<Var> = inputs.iter().cloned().map(|t| g.leaf(t)).collect();
        let out = f(&mut g, &vars)?;
        Ok(g.value(out).data()[0])
    };

    let mut g = Graph::new();
    let vars: Vec<Var> = inputs.iter().cloned().map(|t| g.leaf(t)).collect();
    let loss = f(&mut g, &vars)?;
    let grads = g.backward(loss)?;

    for (input, var) in vars.iter().enumerate() {
        let analytic = grads.get(*var).ok_or("input has no gradient")?;
        for index in 0..inputs[input].numel() {
            let mut shifted = inputs.to_vec();
            shifted[input].data_mut()[index] += eps;
            let plus = eval(&shifted)?;
            shifted[input].data_mut()[index] -= 2.0 * eps;
            let minus = eval(&shifted)?;
            let numeric = (plus - minus) / (2.0 * eps);
            let analytic = analytic.data()[index];
            if (analytic - numeric).abs() > tol * (1.0 + numeric.abs()) {
                return Err(GradMismatch {
                    input,
                    index,
                    analytic,
                    numeric,
                }
                .into());
            }
        }
    }
    Ok(())
}
