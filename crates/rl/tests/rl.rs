use rt_rl::{CartPole, Env, MaskedCorridor, Ppo, PpoConfig, VecEnv};

fn cartpoles(count: usize, seed: u64) -> Vec<Box<dyn Env>> {
    (0..count)
        .map(|i| Box::new(CartPole::new(seed + i as u64)) as Box<dyn Env>)
        .collect()
}

#[test]
fn vec_env_resets_finished_episodes_and_tracks_their_returns() {
    let mut envs = VecEnv::new(cartpoles(3, 10)).unwrap();
    let (mut rewards, mut dones) = (vec![0.0; 3], vec![false; 3]);
    let mut seen = Vec::new();
    for _ in 0..60 {
        envs.step(&[1, 1, 1], &mut rewards, &mut dones);
        seen.extend(envs.take_finished());
    }
    assert!(!seen.is_empty(), "pushing one way must topple the pole");
    for episode in &seen {
        assert_eq!(episode.reward, episode.length as f32);
        assert!(episode.length < 60);
    }
    assert!(envs.observations().iter().all(|v| v.is_finite()));
    assert!(envs.take_finished().is_empty());
}

#[test]
fn vec_env_rejects_empty_and_mismatched_environments() {
    assert!(VecEnv::new(Vec::new()).is_err());
    let mixed: Vec<Box<dyn Env>> =
        vec![Box::new(CartPole::new(1)), Box::new(MaskedCorridor::new(5))];
    assert!(VecEnv::new(mixed).is_err());
}

#[test]
fn corridor_masks_moves_that_leave_the_corridor() {
    let mut env = MaskedCorridor::new(5);
    env.reset();
    let mut legal = [true; 3];
    env.legal_actions(&mut legal);
    assert_eq!(legal, [false, true, true]);

    for _ in 0..2 {
        env.step(MaskedCorridor::JUMP);
    }
    assert_eq!(env.position(), 4);
    env.legal_actions(&mut legal);
    assert_eq!(legal, [true, false, false]);

    env.reset();
    env.step(MaskedCorridor::JUMP);
    env.step(MaskedCorridor::RIGHT);
    assert_eq!(env.position(), 3);
    env.legal_actions(&mut legal);
    assert_eq!(
        legal,
        [true, true, false],
        "a jump from the cell before the goal would overshoot"
    );
}

#[test]
fn corridor_rewards_the_goal_and_punishes_illegal_moves() {
    let mut env = MaskedCorridor::new(4);
    env.reset();
    let illegal = env.step(MaskedCorridor::LEFT);
    assert_eq!(illegal.reward, -1.0);
    assert_eq!(env.position(), 0);
    env.step(MaskedCorridor::JUMP);
    let goal = env.step(MaskedCorridor::RIGHT);
    assert!(goal.done);
    assert_eq!(goal.reward, 1.0);
}

fn small_config() -> PpoConfig {
    PpoConfig {
        n_steps: 16,
        minibatches: 2,
        epochs: 2,
        hidden: vec![16],
        ..PpoConfig::default()
    }
}

#[test]
fn ppo_rejects_uneven_minibatches() {
    let config = PpoConfig {
        n_steps: 5,
        minibatches: 4,
        ..PpoConfig::default()
    };
    assert!(Ppo::new(cartpoles(3, 1), config).is_err());
}

#[test]
fn ppo_updates_produce_finite_statistics_and_count_steps() {
    let mut ppo = Ppo::new(cartpoles(4, 20), small_config()).unwrap();
    let first = ppo.update().unwrap();
    let second = ppo.update().unwrap();
    for stats in [&first, &second] {
        assert!(stats.policy_loss.is_finite() && stats.value_loss.is_finite());
        assert!(stats.kl.is_finite());
        assert!(
            stats.entropy > 0.3 && stats.entropy < 0.75,
            "entropy {}",
            stats.entropy
        );
    }
    assert_eq!(first.env_steps, 64);
    assert_eq!(second.env_steps, 128);
}

#[test]
fn training_is_deterministic_for_a_seed() {
    let run = |seed| {
        let mut ppo = Ppo::new(
            cartpoles(4, 30),
            PpoConfig {
                seed,
                ..small_config()
            },
        )
        .unwrap();
        (0..3).map(|_| ppo.update().unwrap()).collect::<Vec<_>>()
    };
    assert_eq!(run(5), run(5));
    assert_ne!(run(5), run(6));
}

#[test]
fn a_trained_policy_survives_longer_than_an_untrained_one() {
    let mut ppo = Ppo::new(
        cartpoles(8, 40),
        PpoConfig {
            learning_rate: 1e-3,
            ..PpoConfig::default()
        },
    )
    .unwrap();
    let before = ppo.evaluate(&mut CartPole::new(999), 10).unwrap();
    ppo.train(30_000, |_| {}).unwrap();
    let after = ppo.evaluate(&mut CartPole::new(999), 10).unwrap();
    assert!(
        after > before + 20.0,
        "return went from {before} to {after}"
    );
}

#[test]
#[ignore = "learning test, run with `cargo test --release -- --ignored`"]
fn ppo_solves_cartpole() {
    let config = PpoConfig {
        learning_rate: 1e-3,
        seed: 7,
        ..PpoConfig::default()
    };
    let mut ppo = Ppo::new(cartpoles(8, 100), config).unwrap();
    ppo.train(150_000, |_| {}).unwrap();
    let score = ppo.evaluate(&mut CartPole::new(4242), 20).unwrap();
    assert!(score >= 195.0, "greedy return {score}");
}

#[test]
#[ignore = "learning test, run with `cargo test --release -- --ignored`"]
fn ppo_learns_the_masked_corridor_without_illegal_moves() {
    let envs: Vec<Box<dyn Env>> = (0..8)
        .map(|_| Box::new(MaskedCorridor::new(8)) as Box<dyn Env>)
        .collect();
    let config = PpoConfig {
        learning_rate: 1e-3,
        hidden: vec![32],
        seed: 3,
        ..PpoConfig::default()
    };
    let mut ppo = Ppo::new(envs, config).unwrap();
    ppo.train(40_000, |_| {}).unwrap();
    let score = ppo.evaluate(&mut MaskedCorridor::new(8), 10).unwrap();
    assert!(score >= 0.9, "greedy return {score} (the optimum is 0.97)");
}
