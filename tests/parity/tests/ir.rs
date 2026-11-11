use rt_ir::{gradients, Graph, NodeId, Result};
use rt_nn::{mlp_train_step, Loss};
use rt_parity::*;
use rt_runtime::{run_reference, Executor};
use rt_tensor::Tensor;
use serde_json::Value;

fn square_sum(g: &mut Graph, v: NodeId) -> Result<NodeId> {
    let squared = g.mul(v, v)?;
    g.sum_all(squared)
}

fn mlp_output(g: &mut Graph, p: &[NodeId]) -> Result<NodeId> {
    let h = g.matmul(p[0], p[1])?;
    let h = g.add(h, p[2])?;
    let h = g.relu(h)?;
    let out = g.matmul(h, p[3])?;
    g.add(out, p[4])
}

fn build(op: &str, g: &mut Graph, v: &[NodeId], labels: Option<NodeId>) -> Result<NodeId> {
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
            let a = g.scale(v[0], -2.5)?;
            let y = g.mul(a, v[1])?;
            g.sum_all(y)
        }
        "relu_mean" => {
            let y = g.relu(v[0])?;
            let squared = g.mul(y, y)?;
            g.mean(squared)
        }
        "exp" => {
            let y = g.exp(v[0])?;
            square_sum(g, y)
        }
        "log_softmax" => {
            let y = g.log_softmax(v[0])?;
            square_sum(g, y)
        }
        "clamp" => {
            let y = g.clamp(v[0], -0.3, 0.4)?;
            let squared = g.mul(y, y)?;
            g.mean(squared)
        }
        "minimum" => {
            let y = g.minimum(v[0], v[1])?;
            square_sum(g, y)
        }
        "cross_entropy" => g.cross_entropy(v[0], labels.unwrap()),
        "mlp_ce" => {
            let out = mlp_output(g, v)?;
            g.cross_entropy(out, labels.unwrap())
        }
        "mlp_mse" => {
            let out = mlp_output(g, v)?;
            let diff = g.sub(out, v[5])?;
            let squared = g.mul(diff, diff)?;
            g.mean(squared)
        }
        other => panic!("unknown op {other}"),
    }
}

struct Case {
    graph: Graph,
    feeds: Vec<(String, Tensor)>,
}

impl Case {
    fn feeds(&self) -> Vec<(&str, &Tensor)> {
        self.feeds.iter().map(|(n, t)| (n.as_str(), t)).collect()
    }
}

fn build_case(case: &Value) -> Case {
    let inputs = tensors(&case["inputs"]);
    let targets = indices(&case["targets"]);
    let mut g = Graph::new();
    let vars: Vec<NodeId> = inputs
        .iter()
        .enumerate()
        .map(|(i, t)| g.input(&format!("i{i}"), t.shape()).unwrap())
        .collect();
    let mut feeds: Vec<(String, Tensor)> = inputs
        .into_iter()
        .enumerate()
        .map(|(i, t)| (format!("i{i}"), t))
        .collect();

    let labels = (!targets.is_empty()).then(|| {
        let values = targets.iter().map(|&t| t as f32).collect();
        feeds.push((
            "labels".into(),
            Tensor::from_vec(values, &[targets.len()]).unwrap(),
        ));
        g.input("labels", &[targets.len()]).unwrap()
    });
    let loss = build(case["op"].as_str().unwrap(), &mut g, &vars, labels).unwrap();
    let grads = gradients(&mut g, loss, &vars).unwrap();
    g.set_outputs(std::iter::once(loss).chain(grads).collect())
        .unwrap();
    Case { graph: g, feeds }
}

#[test]
fn reference_executor_matches_pytorch() {
    let fixtures = fixtures();
    for case in cases(&fixtures, "autograd") {
        let label = name(case);
        let built = build_case(case);
        let outputs = run_reference(&built.graph, &built.feeds()).unwrap();
        assert_scalar_close(
            label,
            "loss",
            outputs[0].data()[0],
            case["loss"].as_f64().unwrap(),
            SINGLE_STEP,
        );
        for (i, (actual, expected)) in outputs[1..].iter().zip(tensors(&case["grads"])).enumerate()
        {
            assert_close(label, &format!("grad {i}"), actual, &expected, SINGLE_STEP);
        }
    }
}

#[test]
fn planned_executor_matches_pytorch() {
    let fixtures = fixtures();
    for case in cases(&fixtures, "autograd") {
        let label = name(case);
        let built = build_case(case);
        let mut executor = Executor::compile(&built.graph).unwrap();
        for round in 0..2 {
            executor.run(&built.feeds()).unwrap();
            let loss = executor.output_tensor(0).data()[0];
            assert_scalar_close(
                label,
                &format!("loss round {round}"),
                loss,
                case["loss"].as_f64().unwrap(),
                SINGLE_STEP,
            );
            for (i, expected) in tensors(&case["grads"]).iter().enumerate() {
                assert_close(
                    label,
                    &format!("grad {i} round {round}"),
                    &executor.output_tensor(i + 1),
                    expected,
                    SINGLE_STEP,
                );
            }
        }
    }
}

#[test]
fn ir_training_matches_pytorch() {
    let fixtures = fixtures();
    for case in cases(&fixtures, "training") {
        let label = name(case);
        let mut params = tensors(&case["params"]);
        let (x, y, targets) = (
            tensor(&case["x"]),
            tensor(&case["y"]),
            indices(&case["targets"]),
        );
        let (loss_kind, target) = if case["loss_kind"] == "mse" {
            (Loss::Mse, y)
        } else {
            (
                Loss::CrossEntropy,
                Tensor::from_vec(
                    targets.iter().map(|&t| t as f32).collect(),
                    &[targets.len()],
                )
                .unwrap(),
            )
        };
        let sizes = [x.shape()[1], params[0].shape()[1], params[2].shape()[1]];
        let step = mlp_train_step(&sizes, x.shape()[0], loss_kind).unwrap();
        let mut executor = Executor::compile(&step.graph).unwrap();
        let mut opt = optimizer(&case["optimizer"]);
        let mut grads: Vec<Tensor> = params.iter().map(|p| Tensor::zeros(p.shape())).collect();

        for (n, expected) in case["losses"].as_array().unwrap().iter().enumerate() {
            let mut feeds = vec![("x", &x), ("target", &target)];
            feeds.extend(
                step.param_names
                    .iter()
                    .map(String::as_str)
                    .zip(params.iter()),
            );
            executor.run(&feeds).unwrap();
            assert_scalar_close(
                label,
                &format!("loss at step {n}"),
                executor.output(0)[0],
                expected.as_f64().unwrap(),
                MULTI_STEP,
            );

            grads
                .iter_mut()
                .enumerate()
                .for_each(|(i, g)| executor.read_output(i + 1, g));
            let mut refs: Vec<&mut Tensor> = params.iter_mut().collect();
            opt.step(&mut refs, &grads).unwrap();
        }
        for (i, (actual, expected)) in params.iter().zip(tensors(&case["final"])).enumerate() {
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
