use rt_autograd::{Graph, Var};
use rt_tensor::{Result, Rng, Tensor};

use crate::linear::{BoundLinear, Linear};

pub struct Mlp {
    layers: Vec<Linear>,
}

pub struct BoundMlp {
    layers: Vec<BoundLinear>,
}

impl Mlp {
    /// `sizes` lists the input width, every hidden width and the output width.
    pub fn new(sizes: &[usize], rng: &mut Rng) -> Self {
        Self {
            layers: sizes
                .windows(2)
                .map(|w| Linear::new(w[0], w[1], rng))
                .collect(),
        }
    }

    pub fn from_layers(layers: Vec<Linear>) -> Self {
        Self { layers }
    }

    /// Names of `params()` in order: `w0`, `b0`, `w1`, `b1`, ...
    pub fn param_names(&self) -> Vec<String> {
        (0..self.layers.len())
            .flat_map(|i| [format!("w{i}"), format!("b{i}")])
            .collect()
    }

    pub fn bind(&self, g: &mut Graph) -> BoundMlp {
        BoundMlp {
            layers: self.layers.iter().map(|l| l.bind(g)).collect(),
        }
    }

    /// Parameters ordered as `[w0, b0, w1, b1, ...]`, matching `BoundMlp::params`.
    pub fn params(&self) -> Vec<&Tensor> {
        self.layers
            .iter()
            .flat_map(|l| [&l.weight, &l.bias])
            .collect()
    }

    /// Parameters ordered as `[w0, b0, w1, b1, ...]`, matching `BoundMlp::params`.
    pub fn params_mut(&mut self) -> Vec<&mut Tensor> {
        self.layers
            .iter_mut()
            .flat_map(|l| [&mut l.weight, &mut l.bias])
            .collect()
    }
}

impl BoundMlp {
    pub fn forward(&self, g: &mut Graph, x: Var) -> Result<Var> {
        let last = self.layers.len().saturating_sub(1);
        let mut h = x;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(g, h)?;
            if i < last {
                h = g.relu(h);
            }
        }
        Ok(h)
    }

    pub fn params(&self) -> Vec<Var> {
        self.layers.iter().flat_map(|l| l.params()).collect()
    }
}
