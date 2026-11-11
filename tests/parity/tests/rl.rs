use rt_parity::*;
use rt_rl::{compute_gae, ppo_graph, CartPole, Env, LossConfig};
use rt_runtime::{run_reference, Executor};
use rt_tensor::Tensor;

fn floats(v: &serde_json::Value) -> Vec<f32> {
    tensor(v).data().to_vec()
}

#[test]
fn gae_matches_numpy() {
    let fixtures = fixtures();
    for case in fixtures["gae"].as_array().unwrap() {
        let steps = case["steps"].as_u64().unwrap() as usize;
        let envs = case["envs"].as_u64().unwrap() as usize;
        let dones: Vec<bool> = floats(&case["dones"]).iter().map(|&d| d > 0.5).collect();
        let (advantages, returns) = compute_gae(
            &floats(&case["rewards"]),
            &floats(&case["values"]),
            &dones,
            (steps, envs),
            (
                case["gamma"].as_f64().unwrap() as f32,
                case["lambda"].as_f64().unwrap() as f32,
            ),
        )
        .unwrap();
        let shape = [steps, envs];
        let label = format!("gae {steps}x{envs}");
        assert_close(
            &label,
            "advantages",
            &Tensor::from_vec(advantages, &shape).unwrap(),
            &tensor(&case["advantages"]),
            SINGLE_STEP,
        );
        assert_close(
            &label,
            "returns",
            &Tensor::from_vec(returns, &shape).unwrap(),
            &tensor(&case["returns"]),
            SINGLE_STEP,
        );
    }
}

#[test]
fn cartpole_dynamics_match_the_reference_trajectory() {
    let fixtures = fixtures();
    let case = &fixtures["cartpole"];
    let initial: Vec<f64> = case["initial"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let mut env = CartPole::with_state([initial[0], initial[1], initial[2], initial[3]], 0);
    let terminated_at = case["terminated_at"].as_u64().unwrap() as usize;
    let actions = case["actions"].as_array().unwrap();
    for (step, (action, expected)) in actions
        .iter()
        .zip(case["trajectory"].as_array().unwrap())
        .enumerate()
    {
        let outcome = env.step(action.as_u64().unwrap() as usize);
        for (i, (got, want)) in env
            .state()
            .iter()
            .zip(expected.as_array().unwrap())
            .enumerate()
        {
            let want = want.as_f64().unwrap();
            assert!(
                (got - want).abs() < 1e-9,
                "step {step} state[{i}]: {got} vs {want}"
            );
        }
        assert_eq!(outcome.done, step == terminated_at, "step {step}");
        assert_eq!(outcome.reward, 1.0);
    }
}

#[test]
fn ppo_loss_and_gradients_match_pytorch() {
    let fixtures = fixtures();
    for case in fixtures["ppo"].as_array().unwrap() {
        let label = name(case);
        let sizes = |key: &str| -> Vec<usize> {
            case[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        };
        let config = LossConfig {
            clip: case["clip"].as_f64().unwrap() as f32,
            value_coef: case["vf"].as_f64().unwrap() as f32,
            entropy_coef: case["ent"].as_f64().unwrap() as f32,
        };
        let built = ppo_graph(
            &sizes("actor_sizes"),
            &sizes("critic_sizes"),
            case["batch"].as_u64().unwrap() as usize,
            &config,
        )
        .unwrap();

        let data = [
            ("obs", tensor(&case["obs"])),
            ("action_onehot", tensor(&case["action_onehot"])),
            ("mask_bias", tensor(&case["mask_bias"])),
            ("old_logp", tensor(&case["old_logp"])),
            ("advantage", tensor(&case["advantage"])),
            ("returns", tensor(&case["returns"])),
        ];
        let params = tensors(&case["params"]);
        let names: Vec<&str> = built
            .actor_params
            .iter()
            .chain(&built.critic_params)
            .map(String::as_str)
            .collect();
        let mut feeds: Vec<(&str, &Tensor)> = data.iter().map(|(n, t)| (*n, t)).collect();
        feeds.extend(names.iter().copied().zip(&params));

        let reference = run_reference(&built.graph, &feeds).unwrap();
        let mut executor = Executor::compile(&built.graph).unwrap();
        executor.run(&feeds).unwrap();
        let planned: Vec<Tensor> = (0..reference.len())
            .map(|i| executor.output_tensor(i))
            .collect();

        let scalars = ["loss", "policy_loss", "value_loss", "entropy", "kl"];
        for (path, outputs) in [("reference", &reference), ("planned", &planned)] {
            for (i, key) in scalars.iter().enumerate() {
                assert_scalar_close(
                    label,
                    &format!("{path} {key}"),
                    outputs[i].data()[0],
                    case[key].as_f64().unwrap(),
                    SINGLE_STEP,
                );
            }
            for (i, expected) in tensors(&case["grads"]).iter().enumerate() {
                assert_close(
                    label,
                    &format!("{path} grad {i}"),
                    &outputs[5 + i],
                    expected,
                    SINGLE_STEP,
                );
            }
        }
    }
}
