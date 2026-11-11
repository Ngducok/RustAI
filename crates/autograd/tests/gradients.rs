use rt_autograd::{check_gradients, Graph, Var};
use rt_tensor::{Result, Rng, Tensor};

const EPS: f32 = 1e-2;
const TOL: f32 = 2e-2;
const RELU_KINK_MARGIN: f32 = 0.02;

fn random(shape: &[usize], seed: u64) -> Tensor {
    Tensor::uniform(shape, -1.0, 1.0, &mut Rng::new(seed))
}

fn squared_sum(g: &mut Graph, v: Var) -> Result<Var> {
    let sq = g.mul(v, v)?;
    Ok(g.sum_all(sq))
}

#[test]
fn add_with_broadcast() {
    let inputs = [random(&[3, 4], 1), random(&[4], 2)];
    check_gradients(
        |g, v| {
            let y = g.add(v[0], v[1])?;
            squared_sum(g, y)
        },
        &inputs,
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn sub_and_mul_with_column_broadcast() {
    let inputs = [random(&[3, 4], 3), random(&[3, 1], 4)];
    check_gradients(
        |g, v| {
            let d = g.sub(v[0], v[1])?;
            let y = g.mul(d, v[1])?;
            squared_sum(g, y)
        },
        &inputs,
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn matmul() {
    let inputs = [random(&[3, 4], 5), random(&[4, 2], 6)];
    check_gradients(
        |g, v| {
            let y = g.matmul(v[0], v[1])?;
            squared_sum(g, y)
        },
        &inputs,
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn relu_and_mean() {
    let x = Tensor::from_vec(vec![-0.7, 0.4, 0.9, -0.3, 0.5, -0.8], &[2, 3]).unwrap();
    check_gradients(
        |g, v| {
            let y = g.relu(v[0]);
            let sq = g.mul(y, y)?;
            Ok(g.mean(sq))
        },
        &[x],
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn cross_entropy() {
    check_gradients(
        |g, v| g.cross_entropy(v[0], &[2, 0, 1, 1]),
        &[random(&[4, 3], 7)],
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn matmul_gradients_match_hand_computation() {
    let mut g = Graph::new();
    let x = g.leaf(Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap());
    let w = g.leaf(Tensor::from_vec(vec![3.0, 4.0], &[2, 1]).unwrap());
    let y = g.matmul(x, w).unwrap();
    let loss = g.sum_all(y);
    let grads = g.backward(loss).unwrap();
    assert_eq!(g.value(loss).data(), &[11.0]);
    assert_eq!(grads.get(x).unwrap().data(), &[3.0, 4.0]);
    assert_eq!(grads.get(w).unwrap().data(), &[1.0, 2.0]);
}

#[test]
fn reused_variable_accumulates_gradient() {
    let mut g = Graph::new();
    let x = g.leaf(Tensor::from_vec(vec![3.0], &[1]).unwrap());
    let sq = g.mul(x, x).unwrap();
    let loss = g.sum_all(sq);
    assert_eq!(g.backward(loss).unwrap().get(x).unwrap().data(), &[6.0]);
}

#[test]
fn mlp_style_gradients_hold_across_seeds() {
    let mut checked = 0;
    for seed in 1..=100 {
        let inputs = [
            random(&[5, 3], seed),
            random(&[3, 6], seed + 100),
            random(&[6], seed + 200),
            random(&[6, 4], seed + 300),
        ];
        let pre = inputs[0]
            .matmul(&inputs[1])
            .unwrap()
            .add(&inputs[2])
            .unwrap();
        if pre.data().iter().any(|v| v.abs() < RELU_KINK_MARGIN) {
            continue;
        }
        checked += 1;
        check_gradients(
            |g, v| {
                let h = g.matmul(v[0], v[1])?;
                let h = g.add(h, v[2])?;
                let h = g.relu(h);
                let logits = g.matmul(h, v[3])?;
                g.cross_entropy(logits, &[0, 1, 2, 3, 1])
            },
            &inputs,
            1e-3,
            1e-2,
        )
        .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
    }
    assert!(checked >= 20, "only {checked} seeds were usable");
}

#[test]
fn gradcheck_detects_a_wrong_gradient() {
    let result = check_gradients(
        |g, v| {
            let frozen = g.constant(g.value(v[0]).clone());
            let y = g.mul(v[0], frozen)?;
            Ok(g.sum_all(y))
        },
        &[random(&[3], 11)],
        EPS,
        TOL,
    );
    assert!(result.is_err());
}

#[test]
fn cross_entropy_is_finite_for_extreme_logits() {
    for target in 0..3 {
        let mut g = Graph::new();
        let logits = g.leaf(Tensor::from_vec(vec![1000.0, -1000.0, 0.0], &[1, 3]).unwrap());
        let loss = g.cross_entropy(logits, &[target]).unwrap();
        let grads = g.backward(loss).unwrap();
        assert!(g.value(loss).data()[0].is_finite());
        assert!(grads
            .get(logits)
            .unwrap()
            .data()
            .iter()
            .all(|v| v.is_finite()));
    }
}

#[test]
fn cross_entropy_rejects_bad_targets() {
    let mut g = Graph::new();
    let logits = g.leaf(random(&[2, 3], 12));
    assert!(g.cross_entropy(logits, &[0]).is_err());
    assert!(g.cross_entropy(logits, &[0, 3]).is_err());
}

#[test]
fn constants_receive_no_gradient() {
    let mut g = Graph::new();
    let w = g.leaf(random(&[2, 2], 8));
    let x = g.constant(random(&[2, 2], 9));
    let y = g.matmul(x, w).unwrap();
    let loss = g.sum_all(y);
    let grads = g.backward(loss).unwrap();
    assert!(grads.get(w).is_some());
    assert!(grads.get(x).is_none());
}

#[test]
fn rejects_non_scalar_loss() {
    let mut g = Graph::new();
    let x = g.leaf(random(&[2, 2], 10));
    assert!(g.backward(x).is_err());
}
