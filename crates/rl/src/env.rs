use rayon::prelude::*;
use rt_tensor::Rng;

use crate::error::{Result, RlError};

pub struct Step {
    pub observation: Vec<f32>,
    pub reward: f32,
    /// The episode ended (terminated or truncated); the next call must be `reset`.
    pub done: bool,
}

pub trait Env: Send {
    fn observation_size(&self) -> usize;
    fn action_count(&self) -> usize;
    fn reset(&mut self) -> Vec<f32>;
    fn step(&mut self, action: usize) -> Step;
    /// Marks which actions are legal in the current state (all of them by default).
    fn legal_actions(&self, legal: &mut [bool]) {
        legal.fill(true);
    }
}

/// Classic cart-pole balancing (the dynamics and thresholds of Gym's `CartPole-v1`).
pub struct CartPole {
    state: [f64; 4],
    steps: usize,
    max_steps: usize,
    rng: Rng,
}

const GRAVITY: f64 = 9.8;
const CART_MASS: f64 = 1.0;
const POLE_MASS: f64 = 0.1;
const POLE_HALF_LENGTH: f64 = 0.5;
const FORCE: f64 = 10.0;
const TAU: f64 = 0.02;
const ANGLE_LIMIT: f64 = 12.0 * 2.0 * std::f64::consts::PI / 360.0;
const POSITION_LIMIT: f64 = 2.4;

impl CartPole {
    pub fn new(seed: u64) -> Self {
        Self::with_state([0.0; 4], seed)
    }

    pub fn with_state(state: [f64; 4], seed: u64) -> Self {
        Self {
            state,
            steps: 0,
            max_steps: 500,
            rng: Rng::new(seed),
        }
    }

    pub fn state(&self) -> [f64; 4] {
        self.state
    }

    fn observation(&self) -> Vec<f32> {
        self.state.iter().map(|&v| v as f32).collect()
    }

    fn advance(&mut self, action: usize) -> bool {
        let [x, x_dot, theta, theta_dot] = self.state;
        let force = if action == 1 { FORCE } else { -FORCE };
        let total_mass = CART_MASS + POLE_MASS;
        let pole_moment = POLE_MASS * POLE_HALF_LENGTH;
        let (sin, cos) = theta.sin_cos();
        let temp = (force + pole_moment * theta_dot * theta_dot * sin) / total_mass;
        let theta_acc = (GRAVITY * sin - cos * temp)
            / (POLE_HALF_LENGTH * (4.0 / 3.0 - POLE_MASS * cos * cos / total_mass));
        let x_acc = temp - pole_moment * theta_acc * cos / total_mass;
        self.state = [
            x + TAU * x_dot,
            x_dot + TAU * x_acc,
            theta + TAU * theta_dot,
            theta_dot + TAU * theta_acc,
        ];
        let [x, _, theta, _] = self.state;
        x < -POSITION_LIMIT || x > POSITION_LIMIT || theta < -ANGLE_LIMIT || theta > ANGLE_LIMIT
    }
}

impl Env for CartPole {
    fn observation_size(&self) -> usize {
        4
    }

    fn action_count(&self) -> usize {
        2
    }

    fn reset(&mut self) -> Vec<f32> {
        for value in &mut self.state {
            *value = f64::from(self.rng.uniform(-0.05, 0.05));
        }
        self.steps = 0;
        self.observation()
    }

    fn step(&mut self, action: usize) -> Step {
        let terminated = self.advance(action);
        self.steps += 1;
        Step {
            observation: self.observation(),
            reward: 1.0,
            done: terminated || self.steps >= self.max_steps,
        }
    }
}

/// A one-dimensional corridor where the agent must reach the far end. Moves that would leave the
/// corridor are illegal, which makes it a small testbed for action masking.
pub struct MaskedCorridor {
    length: usize,
    position: usize,
    steps: usize,
}

impl MaskedCorridor {
    pub const LEFT: usize = 0;
    pub const RIGHT: usize = 1;
    pub const JUMP: usize = 2;

    pub fn new(length: usize) -> Self {
        Self {
            length,
            position: 0,
            steps: 0,
        }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    fn is_legal(&self, action: usize) -> bool {
        match action {
            Self::LEFT => self.position > 0,
            Self::RIGHT => self.position + 1 < self.length,
            Self::JUMP => self.position + 2 < self.length,
            _ => false,
        }
    }

    fn observation(&self) -> Vec<f32> {
        let mut observation = vec![0.0; self.length];
        observation[self.position] = 1.0;
        observation
    }
}

impl Env for MaskedCorridor {
    fn observation_size(&self) -> usize {
        self.length
    }

    fn action_count(&self) -> usize {
        3
    }

    fn reset(&mut self) -> Vec<f32> {
        self.position = 0;
        self.steps = 0;
        self.observation()
    }

    fn step(&mut self, action: usize) -> Step {
        self.steps += 1;
        let mut reward = -0.01;
        if self.is_legal(action) {
            match action {
                Self::LEFT => self.position -= 1,
                Self::RIGHT => self.position += 1,
                _ => self.position += 2,
            }
        } else {
            reward = -1.0;
        }
        let reached = self.position + 1 == self.length;
        if reached {
            reward = 1.0;
        }
        Step {
            observation: self.observation(),
            reward,
            done: reached || self.steps >= 4 * self.length,
        }
    }

    fn legal_actions(&self, legal: &mut [bool]) {
        for (action, slot) in legal.iter_mut().enumerate() {
            *slot = self.is_legal(action);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Episode {
    pub reward: f32,
    pub length: usize,
}

/// Several environments stepped in parallel with automatic resets.
pub struct VecEnv {
    envs: Vec<Box<dyn Env>>,
    observation_size: usize,
    action_count: usize,
    observations: Vec<f32>,
    legal: Vec<bool>,
    returns: Vec<f32>,
    lengths: Vec<usize>,
    finished: Vec<Episode>,
}

impl VecEnv {
    pub fn new(mut envs: Vec<Box<dyn Env>>) -> Result<Self> {
        let first = envs
            .first()
            .ok_or_else(|| RlError::Invalid("at least one environment is required".into()))?;
        let (observation_size, action_count) = (first.observation_size(), first.action_count());
        if envs
            .iter()
            .any(|e| e.observation_size() != observation_size || e.action_count() != action_count)
        {
            return Err(RlError::Invalid("environments differ in shape".into()));
        }
        let count = envs.len();
        let mut observations = Vec::with_capacity(count * observation_size);
        let mut legal = vec![true; count * action_count];
        for (env, legal) in envs.iter_mut().zip(legal.chunks_mut(action_count)) {
            observations.extend(env.reset());
            env.legal_actions(legal);
        }
        Ok(Self {
            envs,
            observation_size,
            action_count,
            observations,
            legal,
            returns: vec![0.0; count],
            lengths: vec![0; count],
            finished: Vec::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.envs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    pub fn observation_size(&self) -> usize {
        self.observation_size
    }

    pub fn action_count(&self) -> usize {
        self.action_count
    }

    /// Current observations, `len() x observation_size`, row-major.
    pub fn observations(&self) -> &[f32] {
        &self.observations
    }

    /// Legal-action flags for the current states, `len() x action_count`, row-major.
    pub fn legal_actions(&self) -> &[bool] {
        &self.legal
    }

    /// Steps every environment; finished episodes are reset and the new first observation is what
    /// `observations()` returns afterwards.
    pub fn step(&mut self, actions: &[usize], rewards: &mut [f32], dones: &mut [bool]) {
        let (d, a) = (self.observation_size, self.action_count);
        let outcomes: Vec<(f32, bool)> = self
            .envs
            .par_iter_mut()
            .zip(self.observations.par_chunks_mut(d))
            .zip(self.legal.par_chunks_mut(a))
            .zip(actions.par_iter())
            .map(|(((env, observation), legal), &action)| {
                let step = env.step(action);
                let next = if step.done {
                    env.reset()
                } else {
                    step.observation
                };
                observation.copy_from_slice(&next);
                env.legal_actions(legal);
                (step.reward, step.done)
            })
            .collect();
        for (i, (reward, done)) in outcomes.into_iter().enumerate() {
            rewards[i] = reward;
            dones[i] = done;
            self.returns[i] += reward;
            self.lengths[i] += 1;
            if done {
                self.finished.push(Episode {
                    reward: self.returns[i],
                    length: self.lengths[i],
                });
                self.returns[i] = 0.0;
                self.lengths[i] = 0;
            }
        }
    }

    /// Episodes that finished since the last call.
    pub fn take_finished(&mut self) -> Vec<Episode> {
        std::mem::take(&mut self.finished)
    }
}
