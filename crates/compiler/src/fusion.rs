use std::collections::{HashMap, HashSet};

use rt_ir::{Graph, NodeId, Op, Result};

struct Plan {
    matmul: usize,
    bias: usize,
    relu_users: Vec<usize>,
}

fn users(graph: &Graph) -> Vec<Vec<usize>> {
    let mut users = vec![Vec::new(); graph.len()];
    for (index, node) in graph.nodes().iter().enumerate() {
        for input in &node.inputs {
            users[input.0].push(index);
        }
    }
    users
}

fn matmul_and_bias(graph: &Graph, users: &[Vec<usize>], add: usize) -> Option<(usize, usize)> {
    let node = &graph.nodes()[add];
    [(0, 1), (1, 0)].into_iter().find_map(|(m, b)| {
        let (matmul, bias) = (node.inputs[m].0, node.inputs[b].0);
        let (product, bias_node) = (&graph.nodes()[matmul], &graph.nodes()[bias]);
        let fits = product.op == Op::MatMul
            && users[matmul].len() == 1
            && node.shape == product.shape
            && bias_node.shape == [product.shape[1]];
        fits.then_some((matmul, bias))
    })
}

/// A ReLU may be folded in when the pre-activation is read only by ReLUs and by the masks of
/// their gradients; for `y = max(x, 0)` the mask `y > 0` is exactly the mask `x > 0`.
fn relu_users(graph: &Graph, users: &[Vec<usize>], add: usize) -> Vec<usize> {
    let readers = &users[add];
    let relus: Vec<usize> = readers
        .iter()
        .copied()
        .filter(|&u| graph.nodes()[u].op == Op::Relu)
        .collect();
    let only_relu_readers = readers.iter().all(|&u| {
        let node = &graph.nodes()[u];
        node.op == Op::Relu
            || (node.op == Op::ReluGrad
                && node.inputs[0] == NodeId(add)
                && node.inputs[1] != NodeId(add))
    });
    let is_output = graph.outputs().contains(&NodeId(add));
    if only_relu_readers && !is_output {
        relus
    } else {
        Vec::new()
    }
}

fn linear_plans(graph: &Graph) -> HashMap<usize, Plan> {
    let users = users(graph);
    let mut plans = HashMap::new();
    for (index, node) in graph.nodes().iter().enumerate() {
        if node.op != Op::Add {
            continue;
        }
        let Some((matmul, bias)) = matmul_and_bias(graph, &users, index) else {
            continue;
        };
        if graph.outputs().contains(&NodeId(matmul)) {
            continue;
        }
        let relu_users = relu_users(graph, &users, index);
        plans.insert(
            index,
            Plan {
                matmul,
                bias,
                relu_users,
            },
        );
    }
    plans
}

/// Fuses `MatMul` + bias `Add` (+ `Relu`) into one `Linear` node. Results are bit-identical
/// because the epilogue runs on the finished dot product, exactly as the separate ops would.
pub fn fuse_linear(graph: &Graph) -> Result<Graph> {
    let plans = linear_plans(graph);
    let skipped: HashSet<usize> = plans.values().map(|p| p.matmul).collect();
    let relu_owner: HashMap<usize, usize> = plans
        .iter()
        .flat_map(|(&add, plan)| plan.relu_users.iter().map(move |&relu| (relu, add)))
        .collect();

    let mut fused = Graph::new();
    let mut renumbered = vec![NodeId(usize::MAX); graph.len()];
    for (index, node) in graph.nodes().iter().enumerate() {
        if skipped.contains(&index) {
            continue;
        }
        if let Some(&add) = relu_owner.get(&index) {
            renumbered[index] = renumbered[add];
            continue;
        }
        renumbered[index] = match plans.get(&index) {
            Some(plan) => {
                let product = &graph.nodes()[plan.matmul];
                let inputs = [
                    renumbered[product.inputs[0].0],
                    renumbered[product.inputs[1].0],
                    renumbered[plan.bias],
                ];
                let op = Op::Linear {
                    relu: !plan.relu_users.is_empty(),
                };
                fused.add_node(op, &inputs, Some(&node.shape))?
            }
            None => {
                let inputs: Vec<NodeId> = node.inputs.iter().map(|i| renumbered[i.0]).collect();
                fused.add_node(node.op.clone(), &inputs, Some(&node.shape))?
            }
        };
    }
    fused.set_outputs(graph.outputs().iter().map(|o| renumbered[o.0]).collect())?;
    Ok(fused)
}

/// Fuses `ReluGrad(x, MatMulNT(a, b))` into `MatMulNTMasked(a, b, x)` when the product has no
/// other reader. The mask is applied to the finished dot product, so results are bit-identical.
pub fn fuse_relu_grad(graph: &Graph) -> Result<Graph> {
    let users = users(graph);
    let is_fusable_product = |id: NodeId| {
        graph.nodes()[id.0].op == Op::MatMulNT
            && users[id.0].len() == 1
            && !graph.outputs().contains(&id)
    };
    let fusable: HashMap<usize, usize> = graph
        .nodes()
        .iter()
        .enumerate()
        .filter(|(_, node)| node.op == Op::ReluGrad && is_fusable_product(node.inputs[1]))
        .map(|(index, node)| (index, node.inputs[1].0))
        .collect();
    let skipped: HashSet<usize> = fusable.values().copied().collect();

    let mut fused = Graph::new();
    let mut renumbered = vec![NodeId(usize::MAX); graph.len()];
    for (index, node) in graph.nodes().iter().enumerate() {
        if skipped.contains(&index) {
            continue;
        }
        renumbered[index] = match fusable.get(&index) {
            Some(&product) => {
                let operands = &graph.nodes()[product].inputs;
                let inputs = [
                    renumbered[operands[0].0],
                    renumbered[operands[1].0],
                    renumbered[node.inputs[0].0],
                ];
                fused.add_node(Op::MatMulNTMasked, &inputs, Some(&node.shape))?
            }
            None => {
                let inputs: Vec<NodeId> = node.inputs.iter().map(|i| renumbered[i.0]).collect();
                fused.add_node(node.op.clone(), &inputs, Some(&node.shape))?
            }
        };
    }
    fused.set_outputs(graph.outputs().iter().map(|o| renumbered[o.0]).collect())?;
    Ok(fused)
}
#[cfg(test)]
mod tests {
    use super::*;
    use rt_ir::gradients;

    fn layer(relu: bool, bias_first: bool) -> Graph {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 3]).unwrap();
        let w = g.input("w", &[3, 5]).unwrap();
        let b = g.input("b", &[5]).unwrap();
        let product = g.matmul(x, w).unwrap();
        let shifted = if bias_first {
            g.add(b, product).unwrap()
        } else {
            g.add(product, b).unwrap()
        };
        let out = if relu {
            g.relu(shifted).unwrap()
        } else {
            shifted
        };
        g.set_outputs(vec![out]).unwrap();
        g
    }

    fn ops(graph: &Graph) -> Vec<Op> {
        graph.nodes().iter().map(|n| n.op.clone()).collect()
    }

    #[test]
    fn fuses_matmul_bias_and_relu_in_either_operand_order() {
        for bias_first in [false, true] {
            let fused = fuse_linear(&layer(true, bias_first)).unwrap();
            assert_eq!(fused.len(), 4, "bias_first={bias_first}");
            assert_eq!(ops(&fused)[3], Op::Linear { relu: true });
            assert_eq!(fused.shape(fused.outputs()[0]), &[4, 5]);
        }
    }

    #[test]
    fn fuses_without_relu_when_there_is_none() {
        let fused = fuse_linear(&layer(false, false)).unwrap();
        assert_eq!(ops(&fused)[3], Op::Linear { relu: false });
    }

    #[test]
    fn leaves_the_product_alone_when_it_has_another_reader() {
        let mut g = layer(false, false);
        let product = NodeId(3);
        let extra = g.relu(product).unwrap();
        g.set_outputs(vec![NodeId(4), extra]).unwrap();
        let fused = fuse_linear(&g).unwrap();
        assert_eq!(fused, g);
    }

    #[test]
    fn leaves_non_vector_bias_alone() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 3]).unwrap();
        let w = g.input("w", &[3, 5]).unwrap();
        let b = g.input("b", &[1, 5]).unwrap();
        let product = g.matmul(x, w).unwrap();
        let out = g.add(product, b).unwrap();
        g.set_outputs(vec![out]).unwrap();
        assert_eq!(fuse_linear(&g).unwrap(), g);
    }

    #[test]
    fn keeps_the_relu_separate_when_the_pre_activation_is_read_elsewhere() {
        let mut g = layer(true, false);
        let pre_activation = NodeId(4);
        let copy = g.scale(pre_activation, 2.0).unwrap();
        g.set_outputs(vec![NodeId(5), copy]).unwrap();
        let fused = fuse_linear(&g).unwrap();
        assert!(ops(&fused).contains(&Op::Linear { relu: false }));
        assert!(ops(&fused).contains(&Op::Relu));
    }

    #[test]
    fn backward_relu_masks_read_the_fused_output() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 3]).unwrap();
        let w = g.input("w", &[3, 5]).unwrap();
        let b = g.input("b", &[5]).unwrap();
        let product = g.matmul(x, w).unwrap();
        let shifted = g.add(product, b).unwrap();
        let hidden = g.relu(shifted).unwrap();
        let loss = g.sum_all(hidden).unwrap();
        let grads = gradients(&mut g, loss, &[w, b]).unwrap();
        g.set_outputs(std::iter::once(loss).chain(grads).collect())
            .unwrap();

        let fused = fuse_linear(&g).unwrap();
        let linear = fused
            .nodes()
            .iter()
            .position(|n| n.op == Op::Linear { relu: true })
            .unwrap();
        let mask_reader = fused.nodes().iter().find(|n| n.op == Op::ReluGrad).unwrap();
        assert_eq!(mask_reader.inputs[0], NodeId(linear));
        assert!(fused.nodes().iter().all(|n| n.op != Op::Add));
    }

    #[test]
    fn fuses_a_relu_gradient_into_the_product_that_feeds_it() {
        let mut g = Graph::new();
        let grad = g.input("grad", &[4, 5]).unwrap();
        let w = g.input("w", &[3, 5]).unwrap();
        let x = g.input("x", &[4, 3]).unwrap();
        let product = g.matmul_nt(grad, w).unwrap();
        let masked = g.relu_grad(x, product).unwrap();
        g.set_outputs(vec![masked]).unwrap();

        let fused = fuse_relu_grad(&g).unwrap();
        assert_eq!(ops(&fused)[3], Op::MatMulNTMasked);
        assert_eq!(fused.len(), 4);
        assert_eq!(
            fused.node(NodeId(3)).inputs,
            vec![NodeId(0), NodeId(1), NodeId(2)]
        );
        assert_eq!(fuse_relu_grad(&fused).unwrap(), fused);
    }

    fn masked_product_graph() -> (Graph, NodeId, NodeId) {
        let mut g = Graph::new();
        let grad = g.input("grad", &[4, 5]).unwrap();
        let w = g.input("w", &[3, 5]).unwrap();
        let x = g.input("x", &[4, 3]).unwrap();
        let product = g.matmul_nt(grad, w).unwrap();
        let masked = g.relu_grad(x, product).unwrap();
        (g, product, masked)
    }

    #[test]
    fn leaves_a_product_that_is_also_a_graph_output_unfused() {
        let (mut g, product, masked) = masked_product_graph();
        g.set_outputs(vec![masked, product]).unwrap();
        assert_eq!(fuse_relu_grad(&g).unwrap(), g);
    }

    #[test]
    fn leaves_a_product_with_another_reader_unfused() {
        let (mut g, product, masked) = masked_product_graph();
        let doubled = g.scale(product, 2.0).unwrap();
        g.set_outputs(vec![masked, doubled]).unwrap();
        assert_eq!(fuse_relu_grad(&g).unwrap(), g);
    }
    #[test]
    fn leaves_a_bias_that_is_not_one_element_per_column_alone() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 3]).unwrap();
        let w = g.input("w", &[3, 5]).unwrap();
        let b = g.input("b", &[1]).unwrap();
        let product = g.matmul(x, w).unwrap();
        let out = g.add(product, b).unwrap();
        g.set_outputs(vec![out]).unwrap();
        assert_eq!(fuse_linear(&g).unwrap(), g);
    }

    #[test]
    fn keeps_the_relu_separate_when_the_pre_activation_is_an_output() {
        let mut g = layer(true, false);
        g.set_outputs(vec![NodeId(4), NodeId(5)]).unwrap();
        let fused = fuse_linear(&g).unwrap();
        assert!(ops(&fused).contains(&Op::Linear { relu: false }));
        assert!(ops(&fused).contains(&Op::Relu));
    }

    #[test]
    fn keeps_the_relu_separate_when_the_pre_activation_is_a_gradient_operand() {
        let mut g = layer(true, false);
        let reversed = g.relu_grad(NodeId(4), NodeId(4)).unwrap();
        g.set_outputs(vec![NodeId(5), reversed]).unwrap();
        let fused = fuse_linear(&g).unwrap();
        assert!(ops(&fused).contains(&Op::Linear { relu: false }));
        assert!(ops(&fused).contains(&Op::Relu));
    }
    #[test]
    fn is_idempotent() {
        let once = fuse_linear(&layer(true, false)).unwrap();
        assert_eq!(fuse_linear(&once).unwrap(), once);
    }
}
