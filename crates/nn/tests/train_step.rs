use rt_autograd::Graph;
use rt_ir::Op;
use rt_nn::{mlp_train_step, mse, Loss, Mlp};
use rt_optim::{AdamW, Optimizer};
use rt_runtime::{run_reference, Executor};
use rt_tensor::{Rng, Tensor};

fn assert_close(what: &str, actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.shape(), expected.shape(), "{what} shape");
    for (i, (&a, &e)) in actual.data().iter().zip(expected.data()).enumerate() {
        assert!(
            (a - e).abs() <= 1e-5 + 1e-4 * e.abs(),
            "{what}[{i}]: IR gives {a}, autograd gives {e}"
        );
    }
}

fn labels(batch: usize, classes: usize) -> Vec<usize> {
    (0..batch).map(|i| (i * 7 + 3) % classes).collect()
}

/// (loss, gradients) from the dynamic-tape engine, used as an independent reference.
fn autograd_step(
    model: &Mlp,
    x: &Tensor,
    target: &Tensor,
    loss: Loss,
    classes: &[usize],
) -> (f32, Vec<Tensor>) {
    let mut g = Graph::new();
    let bound = model.bind(&mut g);
    let input = g.constant(x.clone());
    let out = bound.forward(&mut g, input).unwrap();
    let loss_var = match loss {
        Loss::Mse => {
            let t = g.constant(target.clone());
            mse(&mut g, out, t).unwrap()
        }
        Loss::CrossEntropy => g.cross_entropy(out, classes).unwrap(),
    };
    let value = g.value(loss_var).data()[0];
    (
        value,
        g.backward(loss_var)
            .unwrap()
            .take_all(&bound.params())
            .unwrap(),
    )
}

fn feeds<'a>(
    names: &'a [String],
    params: &'a [&'a mut Tensor],
    x: &'a Tensor,
    target: &'a Tensor,
) -> Vec<(&'a str, &'a Tensor)> {
    let mut feeds = vec![("x", x), ("target", target)];
    feeds.extend(
        names
            .iter()
            .map(String::as_str)
            .zip(params.iter().map(|p| &**p)),
    );
    feeds
}

#[test]
fn ir_train_step_matches_the_dynamic_engine() {
    for (loss, sizes) in [
        (Loss::Mse, [5, 12, 7, 3]),
        (Loss::CrossEntropy, [5, 12, 7, 3]),
    ] {
        let mut rng = Rng::new(21);
        let batch = 9;
        let mut model = Mlp::new(&sizes, &mut rng);
        let x = Tensor::uniform(&[batch, sizes[0]], -1.0, 1.0, &mut rng);
        let classes = labels(batch, sizes[3]);
        let target = match loss {
            Loss::Mse => Tensor::uniform(&[batch, sizes[3]], -1.0, 1.0, &mut rng),
            Loss::CrossEntropy => {
                Tensor::from_vec(classes.iter().map(|&c| c as f32).collect(), &[batch]).unwrap()
            }
        };
        let (expected_loss, expected_grads) = autograd_step(&model, &x, &target, loss, &classes);

        let step = mlp_train_step(&sizes, batch, loss).unwrap();
        let params = model.params_mut();
        let feeds = feeds(&step.param_names, &params, &x, &target);
        let reference = run_reference(&step.graph, &feeds).unwrap();
        let mut executor = Executor::compile(&step.graph).unwrap();
        executor.run(&feeds).unwrap();

        let planned = (0..executor.graph().outputs().len())
            .map(|i| executor.output_tensor(i))
            .collect();
        for (label, outputs) in [("reference", reference), ("planned", planned)] {
            assert!(
                (outputs[0].data()[0] - expected_loss).abs() < 1e-5,
                "{loss:?} {label} loss"
            );
            for (i, (actual, expected)) in outputs[1..].iter().zip(&expected_grads).enumerate() {
                assert_close(&format!("{loss:?} {label} grad {i}"), actual, expected);
            }
        }
    }
}

#[test]
fn planned_executor_trains_xor() {
    let mut model = Mlp::new(&[2, 16, 2], &mut Rng::new(42));
    let step = mlp_train_step(&[2, 16, 2], 4, Loss::CrossEntropy).unwrap();
    let mut executor = Executor::compile(&step.graph).unwrap();
    let mut opt = AdamW::new(0.05).with_weight_decay(0.0);
    let x = Tensor::from_vec(vec![0., 0., 0., 1., 1., 0., 1., 1.], &[4, 2]).unwrap();
    let target = Tensor::from_vec(vec![0., 1., 1., 0.], &[4]).unwrap();
    let mut grads: Vec<Tensor> = model
        .params_mut()
        .iter()
        .map(|p| Tensor::zeros(p.shape()))
        .collect();

    for _ in 0..300 {
        let mut params = model.params_mut();
        executor
            .run(&feeds(&step.param_names, &params, &x, &target))
            .unwrap();
        grads
            .iter_mut()
            .enumerate()
            .for_each(|(i, g)| executor.read_output(i + 1, g));
        opt.step(&mut params, &grads).unwrap();
    }

    let mut g = Graph::new();
    let bound = model.bind(&mut g);
    let input = g.constant(x);
    let logits = bound.forward(&mut g, input).unwrap();
    assert_eq!(g.value(logits).argmax_rows().unwrap(), [0, 1, 1, 0]);
}

#[test]
fn rejects_degenerate_networks() {
    assert!(mlp_train_step(&[4], 2, Loss::Mse).is_err());
}

#[test]
fn optimised_training_steps_are_bit_identical_to_the_unoptimised_graph() {
    let cases = [
        (Loss::Mse, vec![3, 5, 2], 7),
        (Loss::CrossEntropy, vec![6, 1, 4], 5),
        (Loss::Mse, vec![130, 70, 33, 9], 37),
        (Loss::CrossEntropy, vec![4, 9, 9, 9, 3], 1),
        (Loss::Mse, vec![256, 128, 64, 10], 200),
    ];
    for (loss, sizes, batch) in cases {
        let mut rng = Rng::new(61);
        let mut model = Mlp::new(&sizes, &mut rng);
        let classes = *sizes.last().unwrap();
        let x = Tensor::uniform(&[batch, sizes[0]], -1.0, 1.0, &mut rng);
        let target = match loss {
            Loss::Mse => Tensor::uniform(&[batch, classes], -1.0, 1.0, &mut rng),
            Loss::CrossEntropy => {
                let labels = (0..batch).map(|i| (i * 5 + 1) % classes).map(|c| c as f32);
                Tensor::from_vec(labels.collect(), &[batch]).unwrap()
            }
        };
        let step = mlp_train_step(&sizes, batch, loss).unwrap();
        let params = model.params_mut();
        let feeds = feeds(&step.param_names, &params, &x, &target);

        let expected = run_reference(&step.graph, &feeds).unwrap();
        let mut executor = Executor::compile(&step.graph).unwrap();
        executor.run(&feeds).unwrap();

        let fused = executor
            .graph()
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Linear { .. }))
            .count();
        assert_eq!(fused, sizes.len() - 1, "{loss:?} {sizes:?}");
        let masked = executor
            .graph()
            .nodes()
            .iter()
            .filter(|n| n.op == Op::MatMulNTMasked)
            .count();
        assert_eq!(masked, sizes.len() - 2, "{loss:?} {sizes:?}");
        for (i, want) in expected.iter().enumerate() {
            let got: Vec<u32> = executor.output(i).iter().map(|v| v.to_bits()).collect();
            let want: Vec<u32> = want.data().iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want, "{loss:?} {sizes:?} output {i}");
        }
    }
}
