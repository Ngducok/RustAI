//! Reinforcement learning on top of the IR pipeline: environments, GAE and PPO.

mod env;
mod error;
mod gae;
mod loss;
mod policy;
mod ppo;

pub use env::{CartPole, Env, Episode, MaskedCorridor, Step, VecEnv};
pub use error::{Result, RlError};
pub use gae::compute_gae;
pub use loss::{ppo_graph, LossConfig, PpoGraph};
pub use policy::{masked_log_probs, sample_categorical, ActorCritic, PolicyRunner, ILLEGAL_LOGIT};
pub use ppo::{Ppo, PpoConfig, UpdateStats};
