use rt_ir::{gradients, Graph, NodeId};
use rt_runtime::{run_reference, Executor, RuntimeError};
use rt_tensor::{Rng, Tensor};

const SIDE: usize = 4;

fn random_graph(rng: &mut Rng, pool: usize) -> Graph {
    let mut g = Graph::new();
    let mut values: Vec<NodeId> = ["a", "b", "c"]
        .iter()
        .map(|n| g.input(n, &[SIDE, SIDE]).unwrap())
        .collect();
    let vector = g.input("v", &[SIDE]).unwrap();
    for _ in 0..6 + rng.below(25) {
        let (x, y) = (
            values[rng.below(values.len().min(pool))],
            values[rng.below(values.len().min(pool))],
        );
        let node = match rng.below(12) {
            0 => g.add(x, y),
            1 => g.sub(x, y),
            2 => g.mul(x, y),
            3 => g.matmul(x, y),
            4 => g.relu(x),
            5 => g.scale(x, 0.5),
            6 => g.transpose(x),
            7 => g.add(x, vector),
            8 => g.sum_all(x).and_then(|s| g.broadcast_to(s, &[SIDE, SIDE])),
            9 => g.full(0.25, &[SIDE, SIDE]),
            10 => g.matmul_nt(x, y),
            _ => g.matmul_tn(x, y),
        }
        .unwrap();
        values.push(node);
    }
    let computed = &values[3..];
    let mut outputs = vec![*computed.last().unwrap()];
    outputs.extend((0..rng.below(3)).map(|_| computed[rng.below(computed.len())]));
    g.set_outputs(outputs).unwrap();
    g
}

fn random_feeds(rng: &mut Rng) -> Vec<(&'static str, Tensor)> {
    let matrix = |rng: &mut Rng| Tensor::uniform(&[SIDE, SIDE], -1.0, 1.0, rng);
    vec![
        ("a", matrix(rng)),
        ("b", matrix(rng)),
        ("c", matrix(rng)),
        ("v", Tensor::uniform(&[SIDE], -1.0, 1.0, rng)),
    ]
}

fn as_feeds<'a>(feeds: &'a [(&'static str, Tensor)]) -> Vec<(&'a str, &'a Tensor)> {
    feeds.iter().map(|(n, t)| (*n, t)).collect()
}

fn bits(data: &[f32]) -> Vec<u32> {
    data.iter().map(|x| x.to_bits()).collect()
}

fn assert_optimised_graphs_match_the_reference(seed: u64, pool: usize) {
    let mut rng = Rng::new(seed);
    for case in 0..200 {
        let graph = random_graph(&mut rng, pool);
        let mut executor = Executor::compile(&graph).unwrap();
        executor
            .plan()
            .validate(executor.graph())
            .unwrap_or_else(|e| panic!("case {case}: {e}"));

        for round in 0..3 {
            let feeds = random_feeds(&mut rng);
            let expected = run_reference(&graph, &as_feeds(&feeds)).unwrap();
            executor.run(&as_feeds(&feeds)).unwrap();
            for (i, want) in expected.iter().enumerate() {
                assert_eq!(
                    bits(executor.output(i)),
                    bits(want.data()),
                    "case {case} round {round} output {i}"
                );
            }
        }
    }
}

#[test]
fn planned_executor_is_bit_identical_to_the_reference_on_random_graphs() {
    assert_optimised_graphs_match_the_reference(2024, usize::MAX);
}

#[test]
fn graphs_full_of_repeated_subexpressions_stay_bit_identical_after_optimisation() {
    assert_optimised_graphs_match_the_reference(555, 5);
}

#[test]
fn optimisation_shrinks_graphs_with_repeated_subexpressions() {
    let mut rng = Rng::new(556);
    let (mut before, mut after) = (0, 0);
    for _ in 0..100 {
        let graph = random_graph(&mut rng, 5);
        before += graph.len();
        after += Executor::compile(&graph).unwrap().graph().len();
    }
    assert!(after * 10 < before * 9, "{before} nodes became {after}");
}
#[test]
fn planning_reuses_memory_on_deep_graphs() {
    let mut g = Graph::new();
    let mut v = g.input("a", &[SIDE, SIDE]).unwrap();
    for _ in 0..20 {
        v = g.relu(v).unwrap();
        v = g.scale(v, 0.9).unwrap();
    }
    g.set_outputs(vec![v]).unwrap();
    let executor = Executor::compile(&g).unwrap();
    assert_eq!(executor.plan().buffer_sizes.len(), 2);
}

#[test]
fn reports_missing_and_misshapen_inputs() {
    let mut g = Graph::new();
    let a = g.input("a", &[2, 2]).unwrap();
    let out = g.relu(a).unwrap();
    g.set_outputs(vec![out]).unwrap();
    let mut executor = Executor::compile(&g).unwrap();

    assert!(matches!(
        executor.run(&[]),
        Err(RuntimeError::MissingInput(_))
    ));
    let wrong = Tensor::zeros(&[3]);
    assert!(matches!(
        executor.run(&[("a", &wrong)]),
        Err(RuntimeError::InputShape { .. })
    ));
    assert!(matches!(
        run_reference(&g, &[]),
        Err(RuntimeError::MissingInput(_))
    ));
}

#[test]
fn executor_recovers_after_a_kernel_error() {
    let mut g = Graph::new();
    let logits = g.input("logits", &[2, 3]).unwrap();
    let labels = g.input("labels", &[2]).unwrap();
    let loss = g.cross_entropy(logits, labels).unwrap();
    g.set_outputs(vec![loss]).unwrap();
    let mut executor = Executor::compile(&g).unwrap();

    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0], &[2, 3]).unwrap();
    let bad = Tensor::from_vec(vec![0.0, 7.0], &[2]).unwrap();
    let good = Tensor::from_vec(vec![2.0, 0.0], &[2]).unwrap();
    assert!(matches!(
        executor.run(&[("logits", &logits), ("labels", &bad)]),
        Err(RuntimeError::Kernel(_))
    ));
    executor
        .run(&[("logits", &logits), ("labels", &good)])
        .unwrap();
    let expected = run_reference(&g, &[("logits", &logits), ("labels", &good)]).unwrap();
    assert_eq!(executor.output(0), expected[0].data());
}

#[test]
fn rejects_outputs_that_are_bare_inputs() {
    let mut g = Graph::new();
    let a = g.input("a", &[2]).unwrap();
    g.set_outputs(vec![a]).unwrap();
    assert!(Executor::compile(&g).is_err());
}

#[test]
fn dead_code_never_runs() {
    let mut g = Graph::new();
    let logits = g.input("logits", &[2, 3]).unwrap();
    let labels = g.input("labels", &[2]).unwrap();
    let _dead_loss = g.cross_entropy(logits, labels).unwrap();
    let live = g.relu(logits).unwrap();
    g.set_outputs(vec![live]).unwrap();
    let mut executor = Executor::compile(&g).unwrap();
    assert_eq!(executor.graph().len(), 2);
    let invalid_labels = Tensor::from_vec(vec![9.0, 9.0], &[2]).unwrap();
    let logits = Tensor::from_vec(vec![-1.0, 0.0, 2.0, 1.0, -2.0, 3.0], &[2, 3]).unwrap();
    executor.run(&[("logits", &logits)]).unwrap();
    assert_eq!(executor.output(0), [0., 0., 2., 1., 0., 3.]);
    assert!(run_reference(&g, &[("logits", &logits), ("labels", &invalid_labels)]).is_err());
}

fn transposed_products_graph(explicit: bool) -> Graph {
    let (m, k, n) = (5, 7, 4);
    let mut g = Graph::new();
    let a = g.input("a", &[m, k]).unwrap();
    let b = g.input("b", &[n, k]).unwrap();
    let c = g.input("c", &[k, m]).unwrap();
    let d = g.input("d", &[k, n]).unwrap();
    let (first, second) = if explicit {
        let (b_t, c_t) = (g.transpose(b).unwrap(), g.transpose(c).unwrap());
        (g.matmul(a, b_t).unwrap(), g.matmul(c_t, d).unwrap())
    } else {
        (g.matmul_nt(a, b).unwrap(), g.matmul_tn(c, d).unwrap())
    };
    let squares = [first, second].map(|v| {
        let squared = g.mul(v, v).unwrap();
        g.sum_all(squared).unwrap()
    });
    let loss = g.add(squares[0], squares[1]).unwrap();
    let grads = gradients(&mut g, loss, &[a, b, c, d]).unwrap();
    g.set_outputs(std::iter::once(loss).chain(grads).collect())
        .unwrap();
    g
}

#[test]
fn transposed_product_ops_and_their_gradients_match_explicit_transposes_bit_for_bit() {
    let mut rng = Rng::new(77);
    let shapes = [("a", [5, 7]), ("b", [4, 7]), ("c", [7, 5]), ("d", [7, 4])];
    let inputs: Vec<(&str, Tensor)> = shapes
        .iter()
        .map(|(name, shape)| (*name, Tensor::uniform(shape, -1.0, 1.0, &mut rng)))
        .collect();
    let (fused, explicit) = (
        transposed_products_graph(false),
        transposed_products_graph(true),
    );
    let feeds = as_feeds(&inputs);
    let (expected, got) = (
        run_reference(&explicit, &feeds).unwrap(),
        run_reference(&fused, &feeds).unwrap(),
    );
    let mut executor = Executor::compile(&fused).unwrap();
    executor.run(&feeds).unwrap();
    for (i, want) in expected.iter().enumerate() {
        assert_eq!(
            bits(got[i].data()),
            bits(want.data()),
            "reference output {i}"
        );
        assert_eq!(
            bits(executor.output(i)),
            bits(want.data()),
            "planned output {i}"
        );
    }
}
