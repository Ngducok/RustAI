use rt_nn::{mlp_forward, Mlp};
use rt_runtime::Executor;
use rt_tensor::{Rng, Tensor};

use crate::error::Result;

/// Added to the logits of illegal actions; large enough that their probability is exactly zero.
pub const ILLEGAL_LOGIT: f32 = -1e9;

/// A policy network and a value network with the same input.
pub struct ActorCritic {
    pub actor: Mlp,
    pub critic: Mlp,
    pub actor_sizes: Vec<usize>,
    pub critic_sizes: Vec<usize>,
}

impl ActorCritic {
    pub fn new(observations: usize, actions: usize, hidden: &[usize], rng: &mut Rng) -> Self {
        let sizes = |out: usize| {
            let mut s = vec![observations];
            s.extend_from_slice(hidden);
            s.push(out);
            s
        };
        let (actor_sizes, critic_sizes) = (sizes(actions), sizes(1));
        Self {
            actor: Mlp::new(&actor_sizes, rng),
            critic: Mlp::new(&critic_sizes, rng),
            actor_sizes,
            critic_sizes,
        }
    }

    /// Actor parameters followed by critic parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut all = self.actor.params_mut();
        all.extend(self.critic.params_mut());
        all
    }
}

/// Runs both networks on a fixed batch of observations through compiled graphs.
pub struct PolicyRunner {
    actor: Executor,
    critic: Executor,
    actor_names: Vec<String>,
    critic_names: Vec<String>,
    input: Tensor,
}

impl PolicyRunner {
    pub fn new(model: &ActorCritic, batch: usize) -> Result<Self> {
        let actor = mlp_forward("actor", &model.actor_sizes, batch)?;
        let critic = mlp_forward("critic", &model.critic_sizes, batch)?;
        Ok(Self {
            actor: Executor::compile(&actor.graph)?,
            critic: Executor::compile(&critic.graph)?,
            actor_names: actor.param_names,
            critic_names: critic.param_names,
            input: Tensor::zeros(&[batch, model.actor_sizes[0]]),
        })
    }

    /// Unmasked logits, `batch x actions`.
    pub fn logits(&mut self, model: &ActorCritic, observations: &[f32]) -> Result<&[f32]> {
        self.input.data_mut().copy_from_slice(observations);
        let mut feeds = vec![("x", &self.input)];
        feeds.extend(
            self.actor_names
                .iter()
                .map(String::as_str)
                .zip(model.actor.params()),
        );
        self.actor.run(&feeds)?;
        Ok(self.actor.output(0))
    }

    /// State values, one per row.
    pub fn values(&mut self, model: &ActorCritic, observations: &[f32]) -> Result<&[f32]> {
        self.input.data_mut().copy_from_slice(observations);
        let mut feeds = vec![("x", &self.input)];
        feeds.extend(
            self.critic_names
                .iter()
                .map(String::as_str)
                .zip(model.critic.params()),
        );
        self.critic.run(&feeds)?;
        Ok(self.critic.output(0))
    }
}

/// Log-probabilities of one row of logits after masking illegal actions, computed the same way
/// as the training graph so that old and new log-probabilities agree at the start of an update.
pub fn masked_log_probs(logits: &[f32], legal: &[bool], out: &mut [f32]) {
    let biased = |i: usize| logits[i] + if legal[i] { 0.0 } else { ILLEGAL_LOGIT };
    let max = (0..logits.len())
        .map(biased)
        .fold(f32::NEG_INFINITY, f32::max);
    let log_sum = (0..logits.len())
        .map(|i| (biased(i) - max).exp())
        .sum::<f32>()
        .ln();
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (biased(i) - max) - log_sum;
    }
}

/// Draws an index with probability `exp(log_probs[i])`.
pub fn sample_categorical(log_probs: &[f32], rng: &mut Rng) -> usize {
    let target = rng.next_f32();
    let mut cumulative = 0.0;
    let mut last_possible = 0;
    for (i, &lp) in log_probs.iter().enumerate() {
        let p = lp.exp();
        if p > 0.0 {
            last_possible = i;
        }
        cumulative += p;
        if target < cumulative {
            return i;
        }
    }
    last_possible
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masked_actions_have_zero_probability_and_the_rest_normalise() {
        let mut out = [0.0; 4];
        masked_log_probs(&[1.0, 5.0, -2.0, 0.5], &[true, false, true, true], &mut out);
        assert_eq!(out[1].exp(), 0.0);
        let total: f32 = out.iter().map(|lp| lp.exp()).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sampling_never_picks_an_illegal_action_and_follows_the_distribution() {
        let mut out = [0.0; 3];
        masked_log_probs(&[0.0, 0.0, 1.0], &[true, false, true], &mut out);
        let mut rng = Rng::new(3);
        let mut counts = [0usize; 3];
        for _ in 0..20_000 {
            counts[sample_categorical(&out, &mut rng)] += 1;
        }
        assert_eq!(counts[1], 0);
        let expected = out[2].exp();
        let observed = counts[2] as f32 / 20_000.0;
        assert!(
            (observed - expected).abs() < 0.02,
            "{observed} vs {expected}"
        );
    }

    #[test]
    fn runner_agrees_with_the_dynamic_forward_pass() {
        let mut rng = Rng::new(9);
        let model = ActorCritic::new(4, 3, &[8], &mut rng);
        let mut runner = PolicyRunner::new(&model, 5).unwrap();
        let observations = Tensor::uniform(&[5, 4], -1.0, 1.0, &mut rng);

        let logits = runner.logits(&model, observations.data()).unwrap().to_vec();
        let mut g = rt_autograd::Graph::new();
        let bound = model.actor.bind(&mut g);
        let input = g.constant(observations.clone());
        let out = bound.forward(&mut g, input).unwrap();
        assert_eq!(logits, g.value(out).data());
        assert_eq!(runner.values(&model, observations.data()).unwrap().len(), 5);
    }
}
