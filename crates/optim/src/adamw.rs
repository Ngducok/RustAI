use rt_tensor::{Result, Tensor};

use crate::{check_pairs, zeros_like, Optimizer};

pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    steps: i32,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            steps: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl AdamW {
    /// Number of steps taken so far.
    pub fn steps(&self) -> i32 {
        self.steps
    }

    /// First and second moment estimates, one tensor per parameter (empty before the first step).
    pub fn moments(&self) -> (&[Tensor], &[Tensor]) {
        (&self.m, &self.v)
    }

    /// Restores state saved from `steps` and `moments`, so training can resume exactly.
    pub fn restore(&mut self, steps: i32, m: Vec<Tensor>, v: Vec<Tensor>) -> Result<()> {
        let aligned = m.len() == v.len() && m.iter().zip(&v).all(|(a, b)| a.shape() == b.shape());
        if !aligned || steps < 0 || (steps > 0) != !m.is_empty() {
            return Err(rt_tensor::TensorError::InvalidArgument(
                "inconsistent optimizer state".into(),
            ));
        }
        (self.steps, self.m, self.v) = (steps, m, v);
        Ok(())
    }
}

impl Optimizer for AdamW {
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn step(&mut self, params: &mut [&mut Tensor], grads: &[Tensor]) -> Result<()> {
        check_pairs(params, grads)?;
        if self.m.is_empty() {
            self.m = zeros_like(params);
            self.v = zeros_like(params);
        }
        self.steps += 1;
        let correction1 = 1.0 - self.beta1.powi(self.steps);
        let correction2 = 1.0 - self.beta2.powi(self.steps);
        for (((p, g), m), v) in params
            .iter_mut()
            .zip(grads)
            .zip(&mut self.m)
            .zip(&mut self.v)
        {
            let state = m.data_mut().iter_mut().zip(v.data_mut());
            for ((p, g), (m, v)) in p.data_mut().iter_mut().zip(g.data()).zip(state) {
                *m = self.beta1 * *m + (1.0 - self.beta1) * g;
                *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
                let update = (*m / correction1) / ((*v / correction2).sqrt() + self.eps);
                *p -= self.lr * (update + self.weight_decay * *p);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_step_moves_by_learning_rate() {
        let mut x = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let grad = Tensor::from_vec(vec![0.5], &[1]).unwrap();
        AdamW::new(0.1)
            .with_weight_decay(0.0)
            .step(&mut [&mut x], &[grad])
            .unwrap();
        assert!((x.data()[0] - 0.9).abs() < 1e-5);
    }

    #[test]
    fn epsilon_is_added_outside_the_square_root() {
        let mut x = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let grad = Tensor::from_vec(vec![1e-8], &[1]).unwrap();
        AdamW::new(0.1)
            .with_weight_decay(0.0)
            .step(&mut [&mut x], &[grad])
            .unwrap();
        assert!((x.data()[0] - 0.95).abs() < 1e-4, "got {}", x.data()[0]);
    }

    #[test]
    fn weight_decay_is_decoupled_from_gradient() {
        let mut x = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        AdamW::new(0.1)
            .with_weight_decay(0.1)
            .step(&mut [&mut x], &[Tensor::zeros(&[1])])
            .unwrap();
        assert!((x.data()[0] - 0.99).abs() < 1e-6);
    }

    #[test]
    fn descends_a_quadratic() {
        let mut x = Tensor::from_vec(vec![5.0, -3.0], &[2]).unwrap();
        let mut opt = AdamW::new(0.1).with_weight_decay(0.0);
        for _ in 0..500 {
            let grad = x.scale(2.0);
            opt.step(&mut [&mut x], &[grad]).unwrap();
        }
        assert!(x.data().iter().all(|v| v.abs() < 1e-2));
    }
}
