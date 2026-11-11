use rt_ir::{gradients, Graph, IrError, Op};

#[test]
fn infers_shapes_through_an_mlp() {
    let mut g = Graph::new();
    let x = g.input("x", &[8, 4]).unwrap();
    let w = g.input("w", &[4, 6]).unwrap();
    let b = g.input("b", &[6]).unwrap();
    let h = g.matmul(x, w).unwrap();
    let h = g.add(h, b).unwrap();
    assert_eq!(g.shape(h), &[8, 6]);
    let t = g.transpose(h).unwrap();
    assert_eq!(g.shape(t), &[6, 8]);
    let s = g.sum_all(t).unwrap();
    assert!(g.shape(s).is_empty());
    let col = g.input("col", &[8, 1]).unwrap();
    let scaled = g.mul(h, col).unwrap();
    assert_eq!(g.shape(scaled), &[8, 6]);
}

#[test]
fn rejects_invalid_shapes() {
    let mut g = Graph::new();
    let a = g.input("a", &[2, 3]).unwrap();
    let b = g.input("b", &[4, 5]).unwrap();
    let v = g.input("v", &[3]).unwrap();
    assert!(matches!(g.matmul(a, b), Err(IrError::Shape(_))));
    assert!(matches!(g.add(a, b), Err(IrError::Shape(_))));
    assert!(matches!(g.transpose(v), Err(IrError::Shape(_))));
    assert!(matches!(g.sum_to_shape(a, &[4]), Err(IrError::Shape(_))));
    assert!(matches!(g.broadcast_to(a, &[3]), Err(IrError::Shape(_))));
    assert!(matches!(g.input("deep", &[1; 9]), Err(IrError::Shape(_))));
}

#[test]
fn rejects_malformed_graphs() {
    let mut g = Graph::new();
    let a = g.input("a", &[2, 2]).unwrap();
    assert!(matches!(g.input("a", &[2, 2]), Err(IrError::Invalid(_))));
    assert!(g.add_node(Op::Add, &[a], None).is_err());
    assert!(g.add_node(Op::Relu, &[rt_ir::NodeId(99)], None).is_err());
    assert!(g.add_node(Op::Input("z".into()), &[], None).is_err());
    assert!(g.add_node(Op::Relu, &[a], Some(&[3, 3])).is_err());
    assert!(g.set_outputs(vec![rt_ir::NodeId(5)]).is_err());
}

#[test]
fn cross_entropy_checks_logits_and_labels() {
    let mut g = Graph::new();
    let logits = g.input("logits", &[4, 3]).unwrap();
    let labels = g.input("labels", &[4]).unwrap();
    let wrong = g.input("wrong", &[5]).unwrap();
    let loss = g.cross_entropy(logits, labels).unwrap();
    assert!(g.shape(loss).is_empty());
    assert!(g.cross_entropy(logits, wrong).is_err());
}

#[test]
fn gradients_have_the_shape_of_their_targets() {
    let mut g = Graph::new();
    let x = g.input("x", &[8, 4]).unwrap();
    let w = g.input("w", &[4, 6]).unwrap();
    let b = g.input("b", &[6]).unwrap();
    let labels = g.input("labels", &[8]).unwrap();
    let h = g.matmul(x, w).unwrap();
    let h = g.add(h, b).unwrap();
    let h = g.relu(h).unwrap();
    let loss = g.cross_entropy(h, labels).unwrap();

    let grads = gradients(&mut g, loss, &[w, b, x]).unwrap();
    for (grad, target) in grads.iter().zip([w, b, x]) {
        assert_eq!(g.shape(*grad), g.shape(target));
    }
}

#[test]
fn gradients_reject_unusable_requests() {
    let mut g = Graph::new();
    let a = g.input("a", &[2, 2]).unwrap();
    let unrelated = g.input("unrelated", &[2, 2]).unwrap();
    let vector = g.relu(a).unwrap();
    let loss = g.sum_all(vector).unwrap();
    assert!(gradients(&mut g, vector, &[a]).is_err(), "non-scalar loss");
    assert!(
        gradients(&mut g, loss, &[unrelated]).is_err(),
        "unrelated input"
    );
    assert!(gradients(&mut g, loss, &[a]).is_ok());
}

#[test]
fn second_order_gradients_are_rejected() {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 2]).unwrap();
    let relu = g.relu(x).unwrap();
    let loss = g.sum_all(relu).unwrap();
    let grad = gradients(&mut g, loss, &[x]).unwrap()[0];
    let again = g.sum_all(grad).unwrap();
    assert!(gradients(&mut g, again, &[x]).is_err());
}

#[test]
fn transposed_products_infer_shapes_and_reject_bad_operands() {
    let mut g = Graph::new();
    let a = g.input("a", &[5, 7]).unwrap();
    let b = g.input("b", &[4, 7]).unwrap();
    let c = g.input("c", &[7, 5]).unwrap();
    let d = g.input("d", &[7, 4]).unwrap();
    let nt = g.matmul_nt(a, b).unwrap();
    let tn = g.matmul_tn(c, d).unwrap();
    assert_eq!(g.shape(nt), &[5, 4]);
    assert_eq!(g.shape(tn), &[5, 4]);
    assert!(g.matmul_nt(a, d).is_err());
    assert!(g.matmul_tn(a, d).is_err());
    let vector = g.input("v", &[7]).unwrap();
    assert!(g.matmul_nt(a, vector).is_err());
}

#[test]
fn matmul_gradients_use_transposed_products_instead_of_transpose_nodes() {
    let mut g = Graph::new();
    let a = g.input("a", &[5, 7]).unwrap();
    let b = g.input("b", &[7, 3]).unwrap();
    let y = g.matmul(a, b).unwrap();
    let loss = g.sum_all(y).unwrap();
    gradients(&mut g, loss, &[a, b]).unwrap();
    assert!(g.nodes().iter().all(|n| n.op != Op::Transpose));
    assert!(g.nodes().iter().any(|n| n.op == Op::MatMulNT));
    assert!(g.nodes().iter().any(|n| n.op == Op::MatMulTN));
}

#[test]
fn fused_ops_infer_shapes_and_reject_bad_operands() {
    let mut g = Graph::new();
    let x = g.input("x", &[4, 3]).unwrap();
    let w = g.input("w", &[3, 5]).unwrap();
    let b = g.input("b", &[5]).unwrap();
    let linear = g.linear(x, w, b, true).unwrap();
    assert_eq!(g.shape(linear), &[4, 5]);
    for (name, shape) in [("short_bias", vec![4]), ("matrix_bias", vec![1, 5])] {
        let bad = g.input(name, &shape).unwrap();
        assert!(g.linear(x, w, bad, false).is_err(), "{name}");
    }
    let bad_w = g.input("bad_w", &[4, 5]).unwrap();
    assert!(g.linear(x, bad_w, b, false).is_err());

    let grad = g.input("grad", &[4, 5]).unwrap();
    let wt = g.input("wt", &[3, 5]).unwrap();
    let mask = g.input("mask", &[4, 3]).unwrap();
    let masked = g.matmul_nt_masked(grad, wt, mask).unwrap();
    assert_eq!(g.shape(masked), &[4, 3]);
    for (name, shape) in [("transposed_mask", vec![3, 4]), ("narrow_mask", vec![4, 2])] {
        let bad = g.input(name, &shape).unwrap();
        assert!(g.matmul_nt_masked(grad, wt, bad).is_err(), "{name}");
    }
}

#[test]
fn fused_ops_are_not_differentiable() {
    let mut g = Graph::new();
    let x = g.input("x", &[4, 3]).unwrap();
    let w = g.input("w", &[3, 5]).unwrap();
    let b = g.input("b", &[5]).unwrap();
    let linear = g.linear(x, w, b, true).unwrap();
    let loss = g.sum_all(linear).unwrap();
    assert!(gradients(&mut g, loss, &[w]).is_err());
}
