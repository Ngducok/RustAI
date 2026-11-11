use rt_tensor::{Result, Tensor};

use crate::{check_pairs, zeros_like, Optimizer};

pub struct Sgd {
    lr: f32,
    momentum: f32,
    velocity: Vec<Tensor>,
}

impl Sgd {
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for Sgd {
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn step(&mut self, params: &mut [&mut Tensor], grads: &[Tensor]) -> Result<()> {
        check_pairs(params, grads)?;
        if self.velocity.is_empty() {
            self.velocity = zeros_like(params);
        }
        for ((p, g), v) in params.iter_mut().zip(grads).zip(&mut self.velocity) {
            for ((p, g), v) in p.data_mut().iter_mut().zip(g.data()).zip(v.data_mut()) {
                *v = self.momentum * *v + g;
                *p -= self.lr * *v;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descends_a_quadratic() {
        let mut x = Tensor::from_vec(vec![5.0, -3.0], &[2]).unwrap();
        let mut opt = Sgd::new(0.1, 0.0);
        for _ in 0..200 {
            let grad = x.scale(2.0);
            opt.step(&mut [&mut x], &[grad]).unwrap();
        }
        assert!(x.data().iter().all(|v| v.abs() < 1e-3));
    }

    #[test]
    fn momentum_matches_hand_computation() {
        let mut x = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let grad = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let mut opt = Sgd::new(0.1, 0.9);
        opt.step(&mut [&mut x], std::slice::from_ref(&grad))
            .unwrap();
        assert!((x.data()[0] - 0.9).abs() < 1e-6);
        opt.step(&mut [&mut x], std::slice::from_ref(&grad))
            .unwrap();
        assert!((x.data()[0] - 0.71).abs() < 1e-6);
    }

    #[test]
    fn rejects_mismatched_gradients() {
        let mut x = Tensor::zeros(&[2]);
        assert!(Sgd::new(0.1, 0.0)
            .step(&mut [&mut x], &[Tensor::zeros(&[3])])
            .is_err());
    }
}
