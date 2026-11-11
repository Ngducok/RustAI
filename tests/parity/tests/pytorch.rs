use rt_autograd::{Graph, Var};
use rt_nn::{mse, Linear, Mlp};
use rt_parity::*;
use rt_tensor::{Result, Tensor};
/// Ops that only the IR pipeline implements.
const IR_ONLY_OPS: [&str; 4] = ["exp", "log_softmax", "clamp", "minimum"];

fn square_sum(g: &mut Graph, v: Var) -> Result<Var> {
    let squared = g.mul(v, v)?;
    Ok(g.sum_all(squared))
}

fn mlp_output(g: &mut Graph, p: &[Var]) -> Result<Var> {
    let h = g.matmul(p[0], p[1])?;
    let h = g.add(h, p[2])?;
    let h = g.relu(h);
    let out = g.matmul(h, p[3])?;
    g.add(out, p[4])
}

fn build(op: &str, g: &mut Graph, v: &[Var], targets: &[usize]) -> Result<Var> {
    match op {
        "add_bcast" => {
            let y = g.add(v[0], v[1])?;
            square_sum(g, y)
        }
        "sub_mul_col" => {
            let d = g.sub(v[0], v[1])?;
            let y = g.mul(d, v[1])?;
            square_sum(g, y)
        }
        "scalar_bcast" => {
            let y = g.mul(v[0], v[1])?;
            square_sum(g, y)
        }
        "matmul" => {
            let y = g.matmul(v[0], v[1])?;
            square_sum(g, y)
        }
        "scale" => {
            let a = g.scale(v[0], -2.5);
            let y = g.mul(a, v[1])?;
            Ok(g.sum_all(y))
        }
        "relu_mean" => {
            let y = g.relu(v[0]);
            let squared = g.mul(y, y)?;
            Ok(g.mean(squared))
        }
        "cross_entropy" => g.cross_entropy(v[0], targets),
        "mlp_ce" => {
            let out = mlp_output(g, v)?;
            g.cross_entropy(out, targets)
        }
        "mlp_mse" => {
            let out = mlp_output(g, v)?;
            mse(g, out, v[5])
        }
        other => panic!("unknown op {other}"),
    }
}

#[test]
fn autograd_matches_pytorch() {
    let fixtures = fixtures();
    for case in cases(&fixtures, "autograd") {
        if IR_ONLY_OPS.contains(&case["op"].as_str().unwrap()) {
            continue;
        }
        let label = name(case);
        let mut g = Graph::new();
        let vars: Vec<Var> = tensors(&case["inputs"])
            .into_iter()
            .map(|t| g.leaf(t))
            .collect();
        let loss = build(
            case["op"].as_str().unwrap(),
            &mut g,
            &vars,
            &indices(&case["targets"]),
        )
        .unwrap();
        assert_scalar_close(
            label,
            "loss",
            g.value(loss).data()[0],
            case["loss"].as_f64().unwrap(),
            SINGLE_STEP,
        );

        let grads = g.backward(loss).unwrap();
        for (i, (var, expected)) in vars.iter().zip(tensors(&case["grads"])).enumerate() {
            assert_close(
                label,
                &format!("grad {i}"),
                grads.get(*var).unwrap(),
                &expected,
                SINGLE_STEP,
            );
        }
    }
}

#[test]
fn optimizers_match_pytorch() {
    let fixtures = fixtures();
    for case in cases(&fixtures, "optimizers") {
        let label = name(case);
        let mut opt = optimizer(&case["optimizer"]);
        let mut params = tensors(&case["init"]);
        let steps = case["grads"].as_array().unwrap();
        let trajectory = case["trajectory"].as_array().unwrap();
        for (step, (grads, expected)) in steps.iter().zip(trajectory).enumerate() {
            let mut refs: Vec<&mut Tensor> = params.iter_mut().collect();
            opt.step(&mut refs, &tensors(grads)).unwrap();
            for (i, (p, e)) in params.iter().zip(tensors(expected)).enumerate() {
                assert_close(label, &format!("step {step} param {i}"), p, &e, SINGLE_STEP);
            }
        }
    }
}

#[test]
fn training_matches_pytorch() {
    let fixtures = fixtures();
    for case in cases(&fixtures, "training") {
        let label = name(case);
        let p = tensors(&case["params"]);
        let mut model = Mlp::from_layers(vec![
            Linear {
                weight: p[0].clone(),
                bias: p[1].clone(),
            },
            Linear {
                weight: p[2].clone(),
                bias: p[3].clone(),
            },
        ]);
        let mut opt = optimizer(&case["optimizer"]);
        let (x, y, targets) = (
            tensor(&case["x"]),
            tensor(&case["y"]),
            indices(&case["targets"]),
        );
        let mse_loss = case["loss_kind"] == "mse";

        for (step, expected) in case["losses"].as_array().unwrap().iter().enumerate() {
            let mut g = Graph::new();
            let bound = model.bind(&mut g);
            let input = g.constant(x.clone());
            let out = bound.forward(&mut g, input).unwrap();
            let loss = if mse_loss {
                let target = g.constant(y.clone());
                mse(&mut g, out, target).unwrap()
            } else {
                g.cross_entropy(out, &targets).unwrap()
            };
            assert_scalar_close(
                label,
                &format!("loss at step {step}"),
                g.value(loss).data()[0],
                expected.as_f64().unwrap(),
                MULTI_STEP,
            );

            let grads = g.backward(loss).unwrap().take_all(&bound.params()).unwrap();
            opt.step(&mut model.params_mut(), &grads).unwrap();
        }

        for (i, (actual, expected)) in model
            .params_mut()
            .into_iter()
            .zip(tensors(&case["final"]))
            .enumerate()
        {
            assert_close(
                label,
                &format!("final param {i}"),
                actual,
                &expected,
                MULTI_STEP,
            );
        }
    }
}
