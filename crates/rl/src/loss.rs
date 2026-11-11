use rt_ir::{gradients, Graph};
use rt_nn::mlp_nodes;

use crate::error::{Result, RlError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossConfig {
    pub clip: f32,
    pub value_coef: f32,
    pub entropy_coef: f32,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            clip: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
        }
    }
}

/// IR for one PPO minibatch: the clipped policy objective, a value regression loss and an
/// entropy bonus, with their gradients.
///
/// Inputs: `obs`, `action_onehot`, `mask_bias`, `old_logp`, `advantage`, `returns`, then the
/// actor (`actor.w0`, `actor.b0`, ...) and critic (`critic.w0`, ...) parameters.
/// Outputs: `loss`, `policy_loss`, `value_loss`, `entropy`, `kl`, then one gradient per parameter
/// in the order of `actor_params` followed by `critic_params`.
pub struct PpoGraph {
    pub graph: Graph,
    pub actor_params: Vec<String>,
    pub critic_params: Vec<String>,
}

pub fn ppo_graph(
    actor_sizes: &[usize],
    critic_sizes: &[usize],
    batch: usize,
    config: &LossConfig,
) -> Result<PpoGraph> {
    let (observations, actions) = (actor_sizes[0], actor_sizes[actor_sizes.len() - 1]);
    if critic_sizes.first() != Some(&observations) || critic_sizes.last() != Some(&1) {
        return Err(RlError::Invalid(
            "the critic must map observations to one value".into(),
        ));
    }
    let scale = 1.0 / batch as f32;
    let mut g = Graph::new();
    let obs = g.input("obs", &[batch, observations])?;
    let onehot = g.input("action_onehot", &[batch, actions])?;
    let mask_bias = g.input("mask_bias", &[batch, actions])?;
    let old_logp = g.input("old_logp", &[batch, 1])?;
    let advantage = g.input("advantage", &[batch, 1])?;
    let returns = g.input("returns", &[batch, 1])?;

    let actor = mlp_nodes(&mut g, "actor", actor_sizes, obs)?;
    let critic = mlp_nodes(&mut g, "critic", critic_sizes, obs)?;

    let logits = g.add(actor.output, mask_bias)?;
    let log_probs = g.log_softmax(logits)?;
    let picked = g.mul(log_probs, onehot)?;
    let logp = g.sum_to_shape(picked, &[batch, 1])?;
    let log_ratio = g.sub(logp, old_logp)?;
    let ratio = g.exp(log_ratio)?;
    let unclipped = g.mul(ratio, advantage)?;
    let clamped = g.clamp(ratio, 1.0 - config.clip, 1.0 + config.clip)?;
    let clipped = g.mul(clamped, advantage)?;
    let surrogate = g.minimum(unclipped, clipped)?;
    let surrogate_sum = g.sum_all(surrogate)?;
    let policy_loss = g.scale(surrogate_sum, -scale)?;

    let error = g.sub(critic.output, returns)?;
    let squared = g.mul(error, error)?;
    let value_loss = g.mean(squared)?;

    let probs = g.exp(log_probs)?;
    let weighted = g.mul(probs, log_probs)?;
    let row_entropy = g.sum_to_shape(weighted, &[batch, 1])?;
    let entropy_sum = g.sum_all(row_entropy)?;
    let entropy = g.scale(entropy_sum, -scale)?;

    let drift = g.sub(old_logp, logp)?;
    let kl = g.mean(drift)?;

    let value_term = g.scale(value_loss, config.value_coef)?;
    let entropy_term = g.scale(entropy, -config.entropy_coef)?;
    let partial = g.add(policy_loss, value_term)?;
    let loss = g.add(partial, entropy_term)?;

    let params: Vec<_> = actor.params.iter().chain(&critic.params).copied().collect();
    let grads = gradients(&mut g, loss, &params)?;
    g.set_outputs(
        [loss, policy_loss, value_loss, entropy, kl]
            .into_iter()
            .chain(grads)
            .collect(),
    )?;
    Ok(PpoGraph {
        graph: g,
        actor_params: actor.param_names,
        critic_params: critic.param_names,
    })
}
