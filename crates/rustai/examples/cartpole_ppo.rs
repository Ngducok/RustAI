//! Reinforcement learning: PPO on CartPole, then a greedy evaluation of the learned policy.
//! Run with `cargo run --release -p rustai --example cartpole_ppo`.

use rustai::prelude::*;

fn main() -> Result<()> {
    let envs: Vec<Box<dyn Env>> = (0..8)
        .map(|i| Box::new(CartPole::new(100 + i)) as Box<dyn Env>)
        .collect();
    let mut ppo = Ppo::new(
        envs,
        PpoConfig {
            learning_rate: 1e-3,
            ..PpoConfig::default()
        },
    )?;

    ppo.train(60_000, |stats| {
        if let Some(reward) = stats.mean_episode_reward {
            if stats.env_steps % 8192 == 0 {
                println!("{:6} steps  mean return {reward:6.1}", stats.env_steps);
            }
        }
    })?;
    let score = ppo.evaluate(&mut CartPole::new(7), 10)?;
    println!("greedy policy return over 10 episodes: {score:.1} (maximum 500)");
    Ok(())
}
