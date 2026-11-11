use rt_autograd::Graph;
use rt_nn::{mse, Mlp};
use rt_optim::{AdamW, Optimizer};
use rt_tensor::{Result, Rng, Tensor};

fn xor_inputs() -> Tensor {
    Tensor::from_vec(vec![0., 0., 0., 1., 1., 0., 1., 1.], &[4, 2]).unwrap()
}

fn train_step(
    model: &mut Mlp,
    opt: &mut AdamW,
    loss_fn: impl Fn(&mut Graph, &rt_nn::BoundMlp) -> Result<rt_autograd::Var>,
) -> f32 {
    let mut g = Graph::new();
    let bound = model.bind(&mut g);
    let loss = loss_fn(&mut g, &bound).unwrap();
    let value = g.value(loss).data()[0];
    let grads = g.backward(loss).unwrap().take_all(&bound.params()).unwrap();
    opt.step(&mut model.params_mut(), &grads).unwrap();
    value
}

#[test]
fn learns_xor_with_cross_entropy() {
    let mut model = Mlp::new(&[2, 16, 2], &mut Rng::new(42));
    let mut opt = AdamW::new(0.05).with_weight_decay(0.0);
    let targets = [0, 1, 1, 0];

    for _ in 0..300 {
        train_step(&mut model, &mut opt, |g, m| {
            let x = g.constant(xor_inputs());
            let logits = m.forward(g, x)?;
            g.cross_entropy(logits, &targets)
        });
    }

    let mut g = Graph::new();
    let bound = model.bind(&mut g);
    let x = g.constant(xor_inputs());
    let logits = bound.forward(&mut g, x).unwrap();
    assert_eq!(g.value(logits).argmax_rows().unwrap(), targets);
}

#[test]
fn regression_loss_decreases() {
    let mut rng = Rng::new(7);
    let x = Tensor::uniform(&[32, 3], -1.0, 1.0, &mut rng);
    let weights = Tensor::from_vec(vec![1.0, -2.0, 0.5], &[3, 1]).unwrap();
    let y = x.matmul(&weights).unwrap();

    let mut model = Mlp::new(&[3, 8, 1], &mut rng);
    let mut opt = AdamW::new(0.02).with_weight_decay(0.0);
    let mut step = || {
        train_step(&mut model, &mut opt, |g, m| {
            let (xv, yv) = (g.constant(x.clone()), g.constant(y.clone()));
            let pred = m.forward(g, xv)?;
            mse(g, pred, yv)
        })
    };

    let first = step();
    let last = (0..400).map(|_| step()).last().unwrap();
    assert!(last < first * 0.05, "loss went from {first} to {last}");
}
