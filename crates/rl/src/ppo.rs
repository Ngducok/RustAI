use rt_optim::{clip_grad_norm, AdamW, Optimizer};
use rt_runtime::Executor;
use rt_tensor::{Rng, Tensor};

use crate::env::{Env, VecEnv};
use crate::error::{Result, RlError};
use crate::gae::compute_gae;
use crate::loss::{ppo_graph, LossConfig, PpoGraph};
use crate::policy::{
    masked_log_probs, sample_categorical, ActorCritic, PolicyRunner, ILLEGAL_LOGIT,
};

#[derive(Debug, Clone)]
pub struct PpoConfig {
    pub n_steps: usize,
    pub epochs: usize,
    pub minibatches: usize,
    pub gamma: f32,
    pub lambda: f32,
    pub loss: LossConfig,
    pub learning_rate: f32,
    pub anneal_learning_rate: bool,
    pub max_grad_norm: f32,
    pub hidden: Vec<usize>,
    pub normalize_advantages: bool,
    pub seed: u64,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            n_steps: 128,
            epochs: 4,
            minibatches: 4,
            gamma: 0.99,
            lambda: 0.95,
            loss: LossConfig::default(),
            learning_rate: 2.5e-4,
            anneal_learning_rate: true,
            max_grad_norm: 0.5,
            hidden: vec![64, 64],
            normalize_advantages: true,
            seed: 1,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct UpdateStats {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub kl: f32,
    pub mean_episode_reward: Option<f32>,
    pub episodes: usize,
    /// Total environment steps taken so far, summed over all environments.
    pub env_steps: usize,
}

struct Rollout {
    observations: Vec<f32>,
    mask_bias: Vec<f32>,
    actions: Vec<usize>,
    log_probs: Vec<f32>,
    values: Vec<f32>,
    rewards: Vec<f32>,
    dones: Vec<bool>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
}

struct Minibatch {
    observations: Tensor,
    action_onehot: Tensor,
    mask_bias: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    returns: Tensor,
}

pub struct Ppo {
    config: PpoConfig,
    envs: VecEnv,
    model: ActorCritic,
    runner: PolicyRunner,
    graph: PpoGraph,
    train: Executor,
    optimizer: AdamW,
    rng: Rng,
    rollout: Rollout,
    batch: Minibatch,
    grads: Vec<Tensor>,
    env_steps: usize,
}

impl Ppo {
    pub fn new(envs: Vec<Box<dyn Env>>, config: PpoConfig) -> Result<Self> {
        let envs = VecEnv::new(envs)?;
        let (n, d, a) = (envs.len(), envs.observation_size(), envs.action_count());
        let samples = n * config.n_steps;
        if config.minibatches == 0 || !samples.is_multiple_of(config.minibatches) {
            return Err(RlError::Invalid(format!(
                "{samples} samples cannot be split into {} equal minibatches",
                config.minibatches
            )));
        }
        let minibatch = samples / config.minibatches;
        let mut rng = Rng::new(config.seed);
        let mut model = ActorCritic::new(d, a, &config.hidden, &mut rng);
        let runner = PolicyRunner::new(&model, n)?;
        let graph = ppo_graph(
            &model.actor_sizes,
            &model.critic_sizes,
            minibatch,
            &config.loss,
        )?;
        let train = Executor::compile(&graph.graph)?;
        let grads = model
            .parameters_mut()
            .iter()
            .map(|p| Tensor::zeros(p.shape()))
            .collect();
        let steps = config.n_steps;
        let rollout = Rollout {
            observations: vec![0.0; steps * n * d],
            mask_bias: vec![0.0; steps * n * a],
            actions: vec![0; steps * n],
            log_probs: vec![0.0; steps * n],
            values: vec![0.0; (steps + 1) * n],
            rewards: vec![0.0; steps * n],
            dones: vec![false; steps * n],
            advantages: Vec::new(),
            returns: Vec::new(),
        };
        let batch = Minibatch {
            observations: Tensor::zeros(&[minibatch, d]),
            action_onehot: Tensor::zeros(&[minibatch, a]),
            mask_bias: Tensor::zeros(&[minibatch, a]),
            old_log_probs: Tensor::zeros(&[minibatch, 1]),
            advantages: Tensor::zeros(&[minibatch, 1]),
            returns: Tensor::zeros(&[minibatch, 1]),
        };
        Ok(Self {
            optimizer: AdamW::new(config.learning_rate).with_weight_decay(0.0),
            config,
            envs,
            model,
            runner,
            graph,
            train,
            rng,
            rollout,
            batch,
            grads,
            env_steps: 0,
        })
    }

    pub fn model(&self) -> &ActorCritic {
        &self.model
    }

    pub fn env_steps(&self) -> usize {
        self.env_steps
    }

    /// Collects one rollout and optimises on it.
    pub fn update(&mut self) -> Result<UpdateStats> {
        self.collect()?;
        let mut stats = self.optimise()?;
        self.env_steps += self.config.n_steps * self.envs.len();
        let finished = self.envs.take_finished();
        stats.episodes = finished.len();
        if !finished.is_empty() {
            stats.mean_episode_reward =
                Some(finished.iter().map(|e| e.reward).sum::<f32>() / finished.len() as f32);
        }
        stats.env_steps = self.env_steps;
        Ok(stats)
    }

    /// Trains for at least `total_steps` environment steps, annealing the learning rate if configured.
    pub fn train(
        &mut self,
        total_steps: usize,
        mut on_update: impl FnMut(&UpdateStats),
    ) -> Result<Vec<UpdateStats>> {
        let mut history = Vec::new();
        while self.env_steps < total_steps {
            if self.config.anneal_learning_rate {
                let remaining = 1.0 - self.env_steps as f32 / total_steps as f32;
                self.optimizer
                    .set_learning_rate(self.config.learning_rate * remaining);
            }
            let stats = self.update()?;
            on_update(&stats);
            history.push(stats);
        }
        Ok(history)
    }

    /// Mean return of the greedy policy over `episodes` episodes of `env`.
    pub fn evaluate(&self, env: &mut dyn Env, episodes: usize) -> Result<f32> {
        let a = env.action_count();
        let mut runner = PolicyRunner::new(&self.model, 1)?;
        let (mut total, mut legal) = (0.0, vec![true; a]);
        for _ in 0..episodes {
            let mut observation = env.reset();
            loop {
                env.legal_actions(&mut legal);
                let logits = runner.logits(&self.model, &observation)?;
                let action = (0..a)
                    .max_by(|&x, &y| {
                        let score =
                            |i: usize| logits[i] + if legal[i] { 0.0 } else { ILLEGAL_LOGIT };
                        score(x).total_cmp(&score(y))
                    })
                    .unwrap_or(0);
                let step = env.step(action);
                total += step.reward;
                if step.done {
                    break;
                }
                observation = step.observation;
            }
        }
        Ok(total / episodes.max(1) as f32)
    }

    fn collect(&mut self) -> Result<()> {
        let (steps, n) = (self.config.n_steps, self.envs.len());
        let (d, a) = (self.envs.observation_size(), self.envs.action_count());
        let (mut actions, mut rewards) = (vec![0usize; n], vec![0.0f32; n]);
        let (mut dones, mut scratch) = (vec![false; n], vec![0.0f32; a]);
        for t in 0..steps {
            let observations = self.envs.observations();
            self.rollout.observations[t * n * d..(t + 1) * n * d].copy_from_slice(observations);
            let legal = self.envs.legal_actions();
            for (slot, &ok) in self.rollout.mask_bias[t * n * a..(t + 1) * n * a]
                .iter_mut()
                .zip(legal)
            {
                *slot = if ok { 0.0 } else { ILLEGAL_LOGIT };
            }
            let logits = self.runner.logits(&self.model, observations)?.to_vec();
            for env in 0..n {
                let row = env * a..(env + 1) * a;
                masked_log_probs(&logits[row.clone()], &legal[row], &mut scratch);
                actions[env] = sample_categorical(&scratch, &mut self.rng);
                self.rollout.log_probs[t * n + env] = scratch[actions[env]];
            }
            let values = self.runner.values(&self.model, observations)?;
            self.rollout.values[t * n..(t + 1) * n].copy_from_slice(values);
            self.rollout.actions[t * n..(t + 1) * n].copy_from_slice(&actions);

            self.envs.step(&actions, &mut rewards, &mut dones);
            self.rollout.rewards[t * n..(t + 1) * n].copy_from_slice(&rewards);
            self.rollout.dones[t * n..(t + 1) * n].copy_from_slice(&dones);
        }
        let bootstrap = self.runner.values(&self.model, self.envs.observations())?;
        self.rollout.values[steps * n..].copy_from_slice(bootstrap);
        let (advantages, returns) = compute_gae(
            &self.rollout.rewards,
            &self.rollout.values,
            &self.rollout.dones,
            (steps, n),
            (self.config.gamma, self.config.lambda),
        )?;
        self.rollout.advantages = advantages;
        self.rollout.returns = returns;
        Ok(())
    }

    fn fill_minibatch(&mut self, indices: &[usize]) {
        let (d, a) = (self.envs.observation_size(), self.envs.action_count());
        let r = &self.rollout;
        let b = &mut self.batch;
        for (row, &i) in indices.iter().enumerate() {
            b.observations.data_mut()[row * d..(row + 1) * d]
                .copy_from_slice(&r.observations[i * d..(i + 1) * d]);
            let onehot = &mut b.action_onehot.data_mut()[row * a..(row + 1) * a];
            onehot.fill(0.0);
            onehot[r.actions[i]] = 1.0;
            b.mask_bias.data_mut()[row * a..(row + 1) * a]
                .copy_from_slice(&r.mask_bias[i * a..(i + 1) * a]);
            b.old_log_probs.data_mut()[row] = r.log_probs[i];
            b.advantages.data_mut()[row] = r.advantages[i];
            b.returns.data_mut()[row] = r.returns[i];
        }
        if self.config.normalize_advantages {
            let advantages = b.advantages.data_mut();
            let mean = advantages.iter().sum::<f32>() / advantages.len() as f32;
            let variance = advantages
                .iter()
                .map(|v| (v - mean) * (v - mean))
                .sum::<f32>()
                / advantages.len() as f32;
            let scale = 1.0 / (variance.sqrt() + 1e-8);
            advantages.iter_mut().for_each(|v| *v = (*v - mean) * scale);
        }
    }

    fn optimise(&mut self) -> Result<UpdateStats> {
        let samples = self.config.n_steps * self.envs.len();
        let size = samples / self.config.minibatches;
        let mut order: Vec<usize> = (0..samples).collect();
        let (mut totals, mut count) = ([0.0f32; 4], 0.0f32);
        for _ in 0..self.config.epochs {
            for i in (1..samples).rev() {
                order.swap(i, self.rng.below(i + 1));
            }
            for indices in order.chunks(size) {
                self.fill_minibatch(indices);
                let b = &self.batch;
                let mut feeds = vec![
                    ("obs", &b.observations),
                    ("action_onehot", &b.action_onehot),
                    ("mask_bias", &b.mask_bias),
                    ("old_logp", &b.old_log_probs),
                    ("advantage", &b.advantages),
                    ("returns", &b.returns),
                ];
                feeds.extend(
                    self.graph
                        .actor_params
                        .iter()
                        .map(String::as_str)
                        .zip(self.model.actor.params()),
                );
                feeds.extend(
                    self.graph
                        .critic_params
                        .iter()
                        .map(String::as_str)
                        .zip(self.model.critic.params()),
                );
                self.train.run(&feeds)?;
                for (slot, output) in totals.iter_mut().zip(1..=4) {
                    *slot += self.train.output(output)[0];
                }
                count += 1.0;
                for (k, grad) in self.grads.iter_mut().enumerate() {
                    self.train.read_output(5 + k, grad);
                }
                clip_grad_norm(&mut self.grads, self.config.max_grad_norm);
                self.optimizer
                    .step(&mut self.model.parameters_mut(), &self.grads)?;
            }
        }
        Ok(UpdateStats {
            policy_loss: totals[0] / count,
            value_loss: totals[1] / count,
            entropy: totals[2] / count,
            kl: totals[3] / count,
            ..UpdateStats::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::CartPole;

    fn trainer() -> Ppo {
        let envs: Vec<Box<dyn Env>> = (0..3)
            .map(|i| Box::new(CartPole::new(50 + i)) as Box<dyn Env>)
            .collect();
        let config = PpoConfig {
            n_steps: 8,
            minibatches: 2,
            hidden: vec![8],
            ..PpoConfig::default()
        };
        Ppo::new(envs, config).unwrap()
    }

    #[test]
    fn a_rollout_bootstraps_from_the_value_of_the_final_observation() {
        let mut ppo = trainer();
        ppo.collect().unwrap();
        let (steps, n) = (ppo.config.n_steps, ppo.envs.len());
        let mut runner = PolicyRunner::new(&ppo.model, n).unwrap();
        let expected = runner
            .values(&ppo.model, ppo.envs.observations())
            .unwrap()
            .to_vec();
        assert!(
            expected.iter().any(|v| *v != 0.0),
            "the test needs a non-zero critic"
        );
        assert_eq!(&ppo.rollout.values[steps * n..], expected.as_slice());
    }

    #[test]
    fn stored_advantages_are_the_gae_of_the_stored_rollout() {
        let mut ppo = trainer();
        ppo.collect().unwrap();
        let r = &ppo.rollout;
        let shape = (ppo.config.n_steps, ppo.envs.len());
        let (advantages, returns) = compute_gae(
            &r.rewards,
            &r.values,
            &r.dones,
            shape,
            (ppo.config.gamma, ppo.config.lambda),
        )
        .unwrap();
        assert_eq!(r.advantages, advantages);
        assert_eq!(r.returns, returns);
        assert!(r.log_probs.iter().all(|lp| *lp <= 0.0 && lp.is_finite()));
    }
}
