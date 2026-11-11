//! End-to-end PPO on CartPole: wall-clock time and environment steps per second until solved.
//! Run with `cargo run --release -p rt-bench --bin ppo`; compare with `benchmarks/pytorch_ppo.py`.

use std::time::Instant;

use rt_rl::{CartPole, Env, Ppo, PpoConfig};

const TARGET: f32 = 195.0;
const WINDOW: usize = 3;
const MAX_STEPS: usize = 400_000;
const SEEDS: [u64; 3] = [1, 2, 3];

fn run(seed: u64) -> (f64, usize) {
    let envs: Vec<Box<dyn Env>> = (0..8)
        .map(|i| Box::new(CartPole::new(seed * 100 + i)) as Box<dyn Env>)
        .collect();
    let config = PpoConfig {
        learning_rate: 1e-3,
        seed,
        ..PpoConfig::default()
    };
    let mut ppo = Ppo::new(envs, config).unwrap();
    let mut recent: Vec<(f32, usize)> = Vec::new();
    let start = Instant::now();
    while ppo.env_steps() < MAX_STEPS {
        let stats = ppo.update().unwrap();
        recent.push((
            stats.mean_episode_reward.unwrap_or(0.0) * stats.episodes as f32,
            stats.episodes,
        ));
        let tail = &recent[recent.len().saturating_sub(WINDOW)..];
        let episodes: usize = tail.iter().map(|t| t.1).sum();
        let reward: f32 = tail.iter().map(|t| t.0).sum();
        if recent.len() >= WINDOW && episodes > 0 && reward / episodes as f32 >= TARGET {
            break;
        }
    }
    (start.elapsed().as_secs_f64(), ppo.env_steps())
}

fn main() {
    println!("PPO on CartPole, 8 envs, solved when the mean return of the last {WINDOW} updates >= {TARGET}");
    let mut times = Vec::new();
    for seed in SEEDS {
        let (seconds, steps) = run(seed);
        println!(
            "seed {seed}: {seconds:6.2} s, {steps:7} env steps, {:8.0} steps/s",
            steps as f64 / seconds
        );
        times.push(seconds);
    }
    times.sort_by(f64::total_cmp);
    println!("median time to solve: {:.2} s", times[times.len() / 2]);
}
