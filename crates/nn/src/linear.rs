use rt_autograd::{Graph, Var};
use rt_tensor::{Result, Rng, Tensor};

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

pub struct BoundLinear {
    weight: Var,
    bias: Var,
}

impl Linear {
    pub fn new(inputs: usize, outputs: usize, rng: &mut Rng) -> Self {
        let limit = (6.0 / (inputs + outputs) as f32).sqrt();
        Self {
            weight: Tensor::uniform(&[inputs, outputs], -limit, limit, rng),
            bias: Tensor::zeros(&[outputs]),
        }
    }

    pub fn bind(&self, g: &mut Graph) -> BoundLinear {
        BoundLinear {
            weight: g.leaf(self.weight.clone()),
            bias: g.leaf(self.bias.clone()),
        }
    }
}

impl BoundLinear {
    pub fn forward(&self, g: &mut Graph, x: Var) -> Result<Var> {
        let y = g.matmul(x, self.weight)?;
        g.add(y, self.bias)
    }

    pub fn params(&self) -> [Var; 2] {
        [self.weight, self.bias]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xavier_initialisation_fills_its_range() {
        let layer = Linear::new(10, 30, &mut Rng::new(1));
        let limit = (6.0f32 / 40.0).sqrt();
        let max = layer
            .weight
            .data()
            .iter()
            .fold(0.0f32, |m, w| m.max(w.abs()));
        assert!(
            max <= limit && max > 0.9 * limit,
            "max |w| = {max}, limit = {limit}"
        );
        assert_eq!(layer.weight.shape(), &[10, 30]);
        assert!(layer.bias.data().iter().all(|&b| b == 0.0));
    }
}
