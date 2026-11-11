use crate::error::{Result, RlError};

/// Generalised advantage estimation over a `steps x envs` rollout.
///
/// `values` holds `steps + 1` rows (the last is the bootstrap value) and `dones[t]` marks that the
/// episode ended at step `t`. Returns `(advantages, returns)` with `returns = advantages + values`.
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    (steps, envs): (usize, usize),
    (gamma, lambda): (f32, f32),
) -> Result<(Vec<f32>, Vec<f32>)> {
    if rewards.len() != steps * envs
        || dones.len() != steps * envs
        || values.len() != (steps + 1) * envs
    {
        return Err(RlError::Invalid(
            "rollout arrays do not match steps x envs".into(),
        ));
    }
    let mut advantages = vec![0.0; steps * envs];
    let mut last = vec![0.0f32; envs];
    for t in (0..steps).rev() {
        for (env, carry) in last.iter_mut().enumerate() {
            let i = t * envs + env;
            let non_terminal = if dones[i] { 0.0 } else { 1.0 };
            let delta = rewards[i] + gamma * values[i + envs] * non_terminal - values[i];
            *carry = delta + gamma * lambda * non_terminal * *carry;
            advantages[i] = *carry;
        }
    }
    let returns = advantages.iter().zip(values).map(|(a, v)| a + v).collect();
    Ok((advantages, returns))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_step_matches_the_td_error() {
        let (adv, ret) = compute_gae(&[1.0], &[0.5, 2.0], &[false], (1, 1), (0.9, 0.95)).unwrap();
        assert!((adv[0] - (1.0 + 0.9 * 2.0 - 0.5)).abs() < 1e-6);
        assert!((ret[0] - (adv[0] + 0.5)).abs() < 1e-6);
    }

    #[test]
    fn episode_ends_cut_off_the_bootstrap_and_the_trace() {
        let rewards = [1.0, 1.0, 1.0];
        let values = [0.0, 0.0, 0.0, 100.0];
        let (adv, _) =
            compute_gae(&rewards, &values, &[false, true, false], (3, 1), (1.0, 1.0)).unwrap();
        assert!((adv[2] - (1.0 + 100.0)).abs() < 1e-4);
        assert!(
            (adv[1] - 1.0).abs() < 1e-6,
            "a terminal step ignores the next value"
        );
        assert!(
            (adv[0] - 2.0).abs() < 1e-6,
            "the trace stops at the episode boundary"
        );
    }

    #[test]
    fn lambda_zero_is_one_step_td_and_lambda_one_is_monte_carlo() {
        let rewards = [1.0, 2.0, 3.0];
        let values = [0.5, 0.25, 0.125, 0.0];
        let dones = [false; 3];
        let (td, _) = compute_gae(&rewards, &values, &dones, (3, 1), (0.5, 0.0)).unwrap();
        assert!((td[0] - (1.0 + 0.5 * 0.25 - 0.5)).abs() < 1e-6);
        let (mc, ret) = compute_gae(&rewards, &values, &dones, (3, 1), (0.5, 1.0)).unwrap();
        let discounted = 1.0 + 0.5 * 2.0 + 0.25 * 3.0;
        assert!(
            (ret[0] - discounted).abs() < 1e-5,
            "{} vs {discounted}",
            ret[0]
        );
        assert!((mc[0] - (discounted - 0.5)).abs() < 1e-5);
    }

    #[test]
    fn rejects_mismatched_arrays() {
        assert!(compute_gae(&[1.0; 3], &[0.0; 4], &[false; 3], (2, 2), (0.9, 0.9)).is_err());
    }
}
