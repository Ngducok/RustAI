mod adamw;
mod sgd;

pub use adamw::AdamW;
pub use sgd::Sgd;

use rt_tensor::{Result, Tensor, TensorError};

pub trait Optimizer {
    fn step(&mut self, params: &mut [&mut Tensor], grads: &[Tensor]) -> Result<()>;
    fn set_learning_rate(&mut self, lr: f32);
}

/// Scales `grads` in place so their global L2 norm is at most `max_norm`; returns the norm
/// before clipping.
pub fn clip_grad_norm(grads: &mut [Tensor], max_norm: f32) -> f32 {
    let squares: f64 = grads
        .iter()
        .flat_map(|g| g.data())
        .map(|&v| f64::from(v) * f64::from(v))
        .sum();
    let norm = squares.sqrt() as f32;
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for g in grads.iter_mut() {
            g.data_mut().iter_mut().for_each(|v| *v *= scale);
        }
    }
    norm
}

fn check_pairs(params: &[&mut Tensor], grads: &[Tensor]) -> Result<()> {
    let aligned = params.len() == grads.len()
        && params
            .iter()
            .zip(grads)
            .all(|(p, g)| p.shape() == g.shape());
    if aligned {
        Ok(())
    } else {
        Err(TensorError::ShapeMismatch(
            "parameters and gradients do not line up".into(),
        ))
    }
}

fn zeros_like(params: &[&mut Tensor]) -> Vec<Tensor> {
    params.iter().map(|p| Tensor::zeros(p.shape())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipping_rescales_only_when_the_norm_is_too_large() {
        let mut grads = vec![
            Tensor::from_vec(vec![3.0, 0.0], &[2]).unwrap(),
            Tensor::from_vec(vec![0.0, 4.0], &[2]).unwrap(),
        ];
        assert!((clip_grad_norm(&mut grads, 10.0) - 5.0).abs() < 1e-6);
        assert_eq!(grads[0].data(), &[3.0, 0.0]);
        assert!((clip_grad_norm(&mut grads, 1.0) - 5.0).abs() < 1e-6);
        let clipped: f32 = grads.iter().flat_map(|g| g.data()).map(|v| v * v).sum();
        assert!((clipped.sqrt() - 1.0).abs() < 1e-6);
        assert!((grads[0].data()[0] - 0.6).abs() < 1e-6);
    }
}
