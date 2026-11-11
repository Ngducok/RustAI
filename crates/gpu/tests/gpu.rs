//! GPU results are compared with the CPU executor. Every test is skipped when no adapter exists.

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use rt_gpu::{GpuAdamW, GpuContext, GpuError, GpuExecutor};
use rt_ir::{gradients, Graph, NodeId};
use rt_nn::{mlp_train_step, Loss, Mlp};
use rt_optim::{AdamW, Optimizer};
use rt_rl::{ppo_graph, LossConfig};
use rt_runtime::Executor;
use rt_tensor::{Rng, Tensor};

static SHARED: OnceLock<Option<Arc<GpuContext>>> = OnceLock::new();
static SERIAL: Mutex<()> = Mutex::new(());

/// One device shared by every test, used one test at a time: opening many devices concurrently
/// is not something drivers handle well. Returns `None` (the test is skipped) without an adapter.
fn context() -> Option<(Arc<GpuContext>, MutexGuard<'static, ()>)> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    SHARED
        .get_or_init(|| match GpuContext::new() {
            Ok(ctx) => Some(ctx),
            Err(GpuError::NoAdapter) => {
                eprintln!("skipping: no GPU adapter");
                None
            }
            Err(e) => panic!("{e}"),
        })
        .clone()
        .map(|ctx| (ctx, guard))
}
fn assert_close(label: &str, gpu: &[f32], cpu: &[f32], atol: f32, rtol: f32) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: length");
    for (i, (&g, &c)) in gpu.iter().zip(cpu).enumerate() {
        if !g.is_finite() || !c.is_finite() {
            assert_eq!(
                g.is_finite(),
                c.is_finite(),
                "{label}[{i}]: GPU {g} vs CPU {c}"
            );
        } else {
            assert!(
                (g - c).abs() <= atol + rtol * c.abs(),
                "{label}[{i}]: GPU {g} vs CPU {c}"
            );
        }
    }
}

fn feeds(named: &[(String, Tensor)]) -> Vec<(&str, &Tensor)> {
    named.iter().map(|(n, t)| (n.as_str(), t)).collect()
}

/// Runs `graph` on both backends and compares every output.
fn compare(
    ctx: &Arc<GpuContext>,
    label: &str,
    graph: &Graph,
    inputs: &[(String, Tensor)],
    atol: f32,
    rtol: f32,
) {
    let feeds = feeds(inputs);
    let mut cpu = Executor::compile(graph).unwrap();
    cpu.run(&feeds).unwrap();
    let mut gpu = GpuExecutor::compile(ctx.clone(), graph).unwrap();
    gpu.run(&feeds).unwrap();
    for i in 0..graph.outputs().len() {
        assert_close(
            &format!("{label} output {i}"),
            gpu.output(i),
            cpu.output(i),
            atol,
            rtol,
        );
    }
}

fn uniform(name: &str, shape: &[usize], rng: &mut Rng) -> (String, Tensor) {
    (name.to_string(), Tensor::uniform(shape, -1.0, 1.0, rng))
}

#[test]
fn reports_the_adapter() {
    if let Some((ctx, _gpu)) = context() {
        println!("GPU adapter: {} ({})", ctx.adapter_name(), ctx.backend());
        assert!(!ctx.adapter_name().is_empty());
    }
}

#[test]
fn matmul_variants_match_the_cpu_across_tile_boundaries() {
    let Some((ctx, _gpu)) = context() else { return };
    let mut rng = Rng::new(1);
    for (m, k, n) in [
        (1, 1, 1),
        (3, 5, 4),
        (17, 16, 15),
        (64, 64, 64),
        (65, 17, 63),
        (70, 130, 90),
        (130, 300, 20),
    ] {
        let mut g = Graph::new();
        let a = g.input("a", &[m, k]).unwrap();
        let b = g.input("b", &[k, n]).unwrap();
        let bt = g.input("bt", &[n, k]).unwrap();
        let at = g.input("at", &[k, m]).unwrap();
        let plain = g.matmul(a, b).unwrap();
        let nt = g.matmul_nt(a, bt).unwrap();
        let tn = g.matmul_tn(at, b).unwrap();
        g.set_outputs(vec![plain, nt, tn]).unwrap();
        let inputs = [
            uniform("a", &[m, k], &mut rng),
            uniform("b", &[k, n], &mut rng),
            uniform("bt", &[n, k], &mut rng),
            uniform("at", &[k, m], &mut rng),
        ];
        compare(&ctx, &format!("{m}x{k}x{n}"), &g, &inputs, 1e-4, 1e-4);
    }
}

#[test]
fn fused_epilogues_match_the_cpu() {
    let Some((ctx, _gpu)) = context() else { return };
    let mut rng = Rng::new(2);
    let (m, k, n) = (37, 70, 29);
    let mut g = Graph::new();
    let (x, w, bias) = (
        g.input("x", &[m, k]).unwrap(),
        g.input("w", &[k, n]).unwrap(),
        g.input("bias", &[n]).unwrap(),
    );
    let (grad, wt, mask) = (
        g.input("grad", &[m, n]).unwrap(),
        g.input("wt", &[k, n]).unwrap(),
        g.input("mask", &[m, k]).unwrap(),
    );
    let plain = g.linear(x, w, bias, false).unwrap();
    let relu = g.linear(x, w, bias, true).unwrap();
    let masked = g.matmul_nt_masked(grad, wt, mask).unwrap();
    g.set_outputs(vec![plain, relu, masked]).unwrap();
    let inputs = [
        uniform("x", &[m, k], &mut rng),
        uniform("w", &[k, n], &mut rng),
        uniform("bias", &[n], &mut rng),
        uniform("grad", &[m, n], &mut rng),
        uniform("wt", &[k, n], &mut rng),
        uniform("mask", &[m, k], &mut rng),
    ];
    compare(&ctx, "epilogues", &g, &inputs, 1e-4, 1e-4);
}

#[test]
fn reductions_and_broadcasts_match_the_cpu() {
    let Some((ctx, _gpu)) = context() else { return };
    let mut rng = Rng::new(3);
    let mut g = Graph::new();
    let a = g.input("a", &[7, 5]).unwrap();
    let cube = g.input("cube", &[3, 4, 5]).unwrap();
    let row = g.input("row", &[5]).unwrap();
    let outputs = vec![
        g.sum_to_shape(a, &[5]).unwrap(),
        g.sum_to_shape(a, &[7, 1]).unwrap(),
        g.sum_to_shape(a, &[1, 1]).unwrap(),
        g.sum_all(a).unwrap(),
        g.sum_to_shape(cube, &[4, 1]).unwrap(),
        g.sum_to_shape(cube, &[3, 1, 5]).unwrap(),
        g.broadcast_to(row, &[7, 5]).unwrap(),
        g.transpose(a).unwrap(),
    ];
    g.set_outputs(outputs).unwrap();
    let inputs = [
        uniform("a", &[7, 5], &mut rng),
        uniform("cube", &[3, 4, 5], &mut rng),
        uniform("row", &[5], &mut rng),
    ];
    compare(&ctx, "reductions", &g, &inputs, 1e-5, 1e-5);
}

#[test]
fn elementwise_ops_with_broadcasting_match_the_cpu() {
    let Some((ctx, _gpu)) = context() else { return };
    let mut rng = Rng::new(4);
    let mut g = Graph::new();
    let a = g.input("a", &[6, 5]).unwrap();
    let b = g.input("b", &[6, 5]).unwrap();
    let col = g.input("col", &[6, 1]).unwrap();
    let row = g.input("row", &[5]).unwrap();
    let scalar = g.input("scalar", &[]).unwrap();
    let outputs = vec![
        g.add(a, row).unwrap(),
        g.sub(a, col).unwrap(),
        g.mul(a, scalar).unwrap(),
        g.mul(col, row).unwrap(),
        g.minimum(a, b).unwrap(),
        g.relu(a).unwrap(),
        g.scale(a, -2.5).unwrap(),
        g.exp(a).unwrap(),
        g.clamp(a, -0.3, 0.4).unwrap(),
        g.log_softmax(a).unwrap(),
        g.relu_grad(a, b).unwrap(),
        g.clamp_grad(a, b, -0.3, 0.4).unwrap(),
    ];
    g.set_outputs(outputs).unwrap();
    let inputs = [
        uniform("a", &[6, 5], &mut rng),
        uniform("b", &[6, 5], &mut rng),
        uniform("col", &[6, 1], &mut rng),
        uniform("row", &[5], &mut rng),
        ("scalar".to_string(), Tensor::full(&[], 0.7)),
    ];
    compare(&ctx, "elementwise", &g, &inputs, 1e-5, 1e-5);
}

#[test]
fn minimum_gradients_split_ties_like_the_cpu() {
    let Some((ctx, _gpu)) = context() else { return };
    let mut g = Graph::new();
    let a = g.input("a", &[5]).unwrap();
    let b = g.input("b", &[5]).unwrap();
    let grad = g.input("grad", &[5]).unwrap();
    let first = g.minimum_grad(a, b, grad, true).unwrap();
    let second = g.minimum_grad(a, b, grad, false).unwrap();
    g.set_outputs(vec![first, second]).unwrap();
    let t = |v: &[f32]| Tensor::from_vec(v.to_vec(), &[5]).unwrap();
    let inputs = [
        ("a".to_string(), t(&[1.0, 2.0, 3.0, 4.0, 5.0])),
        ("b".to_string(), t(&[2.0, 2.0, 1.0, 4.0, 9.0])),
        ("grad".to_string(), t(&[1.0, 1.0, 1.0, 2.0, 3.0])),
    ];
    compare(&ctx, "ties", &g, &inputs, 0.0, 0.0);
}

#[test]
fn cross_entropy_and_its_gradient_match_the_cpu() {
    let Some((ctx, _gpu)) = context() else { return };
    let mut rng = Rng::new(5);
    for (rows, classes) in [(1, 2), (5, 3), (200, 10), (130, 1)] {
        let mut g = Graph::new();
        let logits = g.input("logits", &[rows, classes]).unwrap();
        let labels = g.input("labels", &[rows]).unwrap();
        let loss = g.cross_entropy(logits, labels).unwrap();
        let grads = gradients(&mut g, loss, &[logits]).unwrap();
        g.set_outputs(vec![loss, grads[0]]).unwrap();
        let labels_data = (0..rows).map(|i| ((i * 7 + 1) % classes) as f32).collect();
        let inputs = [
            (
                "logits".to_string(),
                Tensor::uniform(&[rows, classes], -5.0, 5.0, &mut rng),
            ),
            (
                "labels".to_string(),
                Tensor::from_vec(labels_data, &[rows]).unwrap(),
            ),
        ];
        compare(
            &ctx,
            &format!("ce {rows}x{classes}"),
            &g,
            &inputs,
            1e-5,
            1e-4,
        );
    }
}

fn random_graph(rng: &mut Rng) -> Graph {
    const SIDE: usize = 5;
    let mut g = Graph::new();
    let mut values: Vec<NodeId> = ["a", "b", "c"]
        .iter()
        .map(|n| g.input(n, &[SIDE, SIDE]).unwrap())
        .collect();
    let vector = g.input("v", &[SIDE]).unwrap();
    for _ in 0..6 + rng.below(20) {
        let (x, y) = (
            values[rng.below(values.len())],
            values[rng.below(values.len())],
        );
        let node = match rng.below(14) {
            0 => g.add(x, y),
            1 => g.sub(x, y),
            2 => g.mul(x, y),
            3 => g.matmul(x, y),
            4 => g.relu(x),
            5 => g.scale(x, 0.5),
            6 => g.transpose(x),
            7 => g.add(x, vector),
            8 => g.sum_all(x).and_then(|s| g.broadcast_to(s, &[SIDE, SIDE])),
            9 => g.matmul_nt(x, y),
            10 => g.matmul_tn(x, y),
            11 => g.clamp(x, -2.0, 2.0),
            12 => g.minimum(x, y),
            _ => g.log_softmax(x),
        }
        .unwrap();
        values.push(node);
    }
    let computed = &values[3..];
    g.set_outputs(vec![
        *computed.last().unwrap(),
        computed[rng.below(computed.len())],
    ])
    .unwrap();
    g
}

#[test]
fn random_graphs_match_the_cpu() {
    let Some((ctx, _gpu)) = context() else { return };
    let mut rng = Rng::new(2024);
    for case in 0..60 {
        let graph = random_graph(&mut rng);
        let mut inputs: Vec<(String, Tensor)> = Vec::new();
        for name in ["a", "b", "c"] {
            inputs.push(uniform(name, &[5, 5], &mut rng));
        }
        inputs.push(uniform("v", &[5], &mut rng));
        compare(
            &ctx,
            &format!("random graph {case}"),
            &graph,
            &inputs,
            1e-3,
            1e-3,
        );
    }
}

#[test]
fn training_steps_match_the_cpu_for_both_losses() {
    let Some((ctx, _gpu)) = context() else { return };
    for (loss, sizes, batch) in [
        (Loss::Mse, vec![7, 33, 20, 5], 29),
        (Loss::CrossEntropy, vec![16, 64, 64, 4], 70),
    ] {
        let mut rng = Rng::new(6);
        let mut model = Mlp::new(&sizes, &mut rng);
        let step = mlp_train_step(&sizes, batch, loss).unwrap();
        let classes = *sizes.last().unwrap();
        let target = match loss {
            Loss::Mse => Tensor::uniform(&[batch, classes], -1.0, 1.0, &mut rng),
            Loss::CrossEntropy => {
                Tensor::from_vec((0..batch).map(|i| (i % classes) as f32).collect(), &[batch])
                    .unwrap()
            }
        };
        let mut inputs = vec![
            uniform("x", &[batch, sizes[0]], &mut rng),
            ("target".to_string(), target),
        ];
        inputs.extend(
            step.param_names
                .iter()
                .cloned()
                .zip(model.params_mut().into_iter().map(|p| p.clone())),
        );
        compare(
            &ctx,
            &format!("{loss:?} step"),
            &step.graph,
            &inputs,
            1e-4,
            1e-3,
        );
    }
}

#[test]
fn ppo_loss_and_gradients_match_the_cpu() {
    let Some((ctx, _gpu)) = context() else { return };
    let mut rng = Rng::new(7);
    let (batch, obs, actions) = (48, 6, 4);
    let built = ppo_graph(
        &[obs, 16, actions],
        &[obs, 16, 1],
        batch,
        &LossConfig::default(),
    )
    .unwrap();
    let onehot: Vec<f32> = (0..batch * actions)
        .map(|i| f32::from(i % actions == (i / actions) % actions))
        .collect();
    let bias: Vec<f32> = (0..batch * actions)
        .map(|i| if i % 7 == 3 { -1e9 } else { 0.0 })
        .collect();
    let mut inputs = vec![
        uniform("obs", &[batch, obs], &mut rng),
        (
            "action_onehot".to_string(),
            Tensor::from_vec(onehot, &[batch, actions]).unwrap(),
        ),
        (
            "mask_bias".to_string(),
            Tensor::from_vec(bias, &[batch, actions]).unwrap(),
        ),
        (
            "old_logp".to_string(),
            Tensor::uniform(&[batch, 1], -2.0, -0.5, &mut rng),
        ),
        uniform("advantage", &[batch, 1], &mut rng),
        uniform("returns", &[batch, 1], &mut rng),
    ];
    for (name, shape) in built
        .graph
        .input_names()
        .map(|(n, id)| (n.to_string(), built.graph.shape(id).to_vec()))
        .collect::<Vec<_>>()
    {
        if name.starts_with("actor.") || name.starts_with("critic.") {
            inputs.push(uniform(&name, &shape, &mut rng));
        }
    }
    compare(&ctx, "ppo", &built.graph, &inputs, 1e-3, 1e-3);
}

#[test]
fn a_network_trains_on_the_gpu() {
    let Some((ctx, _gpu)) = context() else { return };
    let mut model = Mlp::new(&[2, 16, 2], &mut Rng::new(42));
    let step = mlp_train_step(&[2, 16, 2], 4, Loss::CrossEntropy).unwrap();
    let mut gpu = GpuExecutor::compile(ctx, &step.graph).unwrap();
    let mut optimizer = AdamW::new(0.05).with_weight_decay(0.0);
    let x = Tensor::from_vec(vec![0., 0., 0., 1., 1., 0., 1., 1.], &[4, 2]).unwrap();
    let target = Tensor::from_vec(vec![0., 1., 1., 0.], &[4]).unwrap();
    let mut grads: Vec<Tensor> = model
        .params_mut()
        .iter()
        .map(|p| Tensor::zeros(p.shape()))
        .collect();
    let (mut first, mut last) = (0.0, 0.0);
    for i in 0..300 {
        let mut named = vec![("x", &x), ("target", &target)];
        named.extend(
            step.param_names
                .iter()
                .map(String::as_str)
                .zip(model.params()),
        );
        gpu.run(&named).unwrap();
        last = gpu.output(0)[0];
        if i == 0 {
            first = last;
        }
        grads
            .iter_mut()
            .enumerate()
            .for_each(|(k, g)| gpu.read_output(k + 1, g));
        optimizer.step(&mut model.params_mut(), &grads).unwrap();
    }
    assert!(last < first * 0.1, "loss went from {first} to {last}");
    let mut g = rt_autograd::Graph::new();
    let bound = model.bind(&mut g);
    let input = g.constant(x);
    let logits = bound.forward(&mut g, input).unwrap();
    assert_eq!(g.value(logits).argmax_rows().unwrap(), [0, 1, 1, 0]);
}

#[test]
fn rejects_missing_and_misshapen_inputs_and_survives_reuse() {
    let Some((ctx, _gpu)) = context() else { return };
    let mut g = Graph::new();
    let a = g.input("a", &[2, 2]).unwrap();
    let out = g.relu(a).unwrap();
    g.set_outputs(vec![out]).unwrap();
    let mut gpu = GpuExecutor::compile(ctx.clone(), &g).unwrap();
    assert!(gpu.run(&[]).is_err());
    assert!(gpu.run(&[("a", &Tensor::zeros(&[3]))]).is_err());
    for scale in [1.0, -1.0, 2.0] {
        let x = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], &[2, 2])
            .unwrap()
            .scale(scale);
        gpu.run(&[("a", &x)]).unwrap();
        assert_eq!(gpu.output(0), x.relu().data());
    }

    let mut bare = Graph::new();
    let input = bare.input("x", &[2]).unwrap();
    bare.set_outputs(vec![input]).unwrap();
    assert!(GpuExecutor::compile(ctx.clone(), &bare).is_err());
}

#[test]
fn on_device_adamw_tracks_the_cpu_optimizer() {
    let Some((ctx, _gpu)) = context() else { return };
    let sizes = [4, 12, 3];
    let batch = 10;
    let step = mlp_train_step(&sizes, batch, Loss::Mse).unwrap();
    let mut rng = Rng::new(8);
    let x = Tensor::uniform(&[batch, 4], -1.0, 1.0, &mut rng);
    let target = Tensor::uniform(&[batch, 3], -1.0, 1.0, &mut rng);
    let mut cpu_model = Mlp::new(&sizes, &mut Rng::new(9));
    let initial: Vec<Tensor> = cpu_model.params().into_iter().cloned().collect();
    let config = GpuAdamW {
        learning_rate: 0.01,
        weight_decay: 0.01,
        ..GpuAdamW::default()
    };

    let mut gpu = GpuExecutor::compile(ctx, &step.graph).unwrap();
    let mut all: Vec<(&str, &Tensor)> = vec![("x", &x), ("target", &target)];
    all.extend(step.param_names.iter().map(String::as_str).zip(&initial));
    gpu.set_inputs(&all).unwrap();
    let updates: Vec<(&str, usize)> = step
        .param_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i + 1))
        .collect();
    gpu.enable_adamw(&updates, config).unwrap();

    let mut cpu = Executor::compile(&step.graph).unwrap();
    let mut optimizer = AdamW::new(0.01).with_weight_decay(0.01);
    let mut grads: Vec<Tensor> = initial.iter().map(|p| Tensor::zeros(p.shape())).collect();
    let (mut cpu_loss, mut gpu_loss) = (0.0, 0.0);
    for step_number in 1..=100 {
        let mut feeds = vec![("x", &x), ("target", &target)];
        feeds.extend(
            step.param_names
                .iter()
                .map(String::as_str)
                .zip(cpu_model.params()),
        );
        cpu.run(&feeds).unwrap();
        cpu_loss = cpu.output(0)[0];
        grads
            .iter_mut()
            .enumerate()
            .for_each(|(k, g)| cpu.read_output(k + 1, g));
        optimizer.step(&mut cpu_model.params_mut(), &grads).unwrap();

        gpu.execute();
        if step_number == 5 || step_number == 100 {
            gpu.download().unwrap();
            gpu_loss = gpu.output(0)[0];
            let tolerance = if step_number == 5 { 1e-5 } else { 5e-3 };
            for (name, expected) in step.param_names.iter().zip(cpu_model.params()) {
                let got = gpu.read_input(name).unwrap();
                assert_close(
                    &format!("step {step_number} {name}"),
                    got.data(),
                    expected.data(),
                    tolerance,
                    tolerance,
                );
            }
        }
    }
    gpu.synchronize();
    assert_close("final loss", &[gpu_loss], &[cpu_loss], 1e-3, 1e-2);
}

#[test]
fn partial_uploads_keep_the_other_inputs_on_the_device() {
    let Some((ctx, _gpu)) = context() else { return };
    let mut g = Graph::new();
    let a = g.input("a", &[3]).unwrap();
    let b = g.input("b", &[3]).unwrap();
    let sum = g.add(a, b).unwrap();
    g.set_outputs(vec![sum]).unwrap();
    let mut gpu = GpuExecutor::compile(ctx, &g).unwrap();
    let t = |v: &[f32]| Tensor::from_vec(v.to_vec(), &[3]).unwrap();
    gpu.set_inputs(&[("a", &t(&[1., 2., 3.])), ("b", &t(&[10., 20., 30.]))])
        .unwrap();
    gpu.execute();
    gpu.download().unwrap();
    assert_eq!(gpu.output(0), [11., 22., 33.]);
    gpu.set_inputs(&[("b", &t(&[0., 0., 0.]))]).unwrap();
    gpu.execute();
    gpu.download().unwrap();
    assert_eq!(gpu.output(0), [1., 2., 3.]);
    assert!(gpu.set_inputs(&[("nope", &t(&[0., 0., 0.]))]).is_err());
}
