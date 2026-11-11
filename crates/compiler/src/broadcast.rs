use rt_ir::{Graph, NodeId, Op, Result};
use rt_tensor::broadcast_shape;

fn is_elementwise_binary(op: &Op) -> bool {
    matches!(op, Op::Add | Op::Sub | Op::Mul)
}

/// Element-wise binary ops broadcast natively, so an explicit `BroadcastTo` feeding one is
/// redundant whenever dropping it leaves the result shape unchanged.
pub fn fold_broadcasts(graph: &Graph) -> Result<Graph> {
    let mut folded = Graph::new();
    let mut renumbered: Vec<NodeId> = Vec::with_capacity(graph.len());
    for node in graph.nodes() {
        let mut inputs: Vec<NodeId> = node.inputs.iter().map(|i| renumbered[i.0]).collect();
        if is_elementwise_binary(&node.op) {
            for slot in 0..2 {
                let original = graph.node(node.inputs[slot]);
                if original.op != Op::BroadcastTo {
                    continue;
                }
                let source = renumbered[original.inputs[0].0];
                let other = folded.shape(inputs[1 - slot]);
                let unchanged = broadcast_shape(folded.shape(source), other)
                    .is_ok_and(|shape| shape == node.shape);
                if unchanged {
                    inputs[slot] = source;
                }
            }
        }
        renumbered.push(folded.add_node(node.op.clone(), &inputs, Some(&node.shape))?);
    }
    folded.set_outputs(graph.outputs().iter().map(|o| renumbered[o.0]).collect())?;
    Ok(folded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eliminate_dead_code;

    #[test]
    fn folds_a_broadcast_that_the_consumer_would_do_anyway() {
        let mut g = Graph::new();
        let scalar = g.input("s", &[]).unwrap();
        let matrix = g.input("m", &[2, 3]).unwrap();
        let expanded = g.broadcast_to(scalar, &[2, 3]).unwrap();
        let product = g.mul(expanded, matrix).unwrap();
        g.set_outputs(vec![product]).unwrap();

        let folded = eliminate_dead_code(&fold_broadcasts(&g).unwrap()).unwrap();
        assert_eq!(folded.len(), 3);
        assert!(folded.nodes().iter().all(|n| n.op != Op::BroadcastTo));
        assert_eq!(folded.shape(folded.outputs()[0]), &[2, 3]);
    }

    #[test]
    fn keeps_a_broadcast_that_defines_the_result_shape() {
        let mut g = Graph::new();
        let row = g.input("row", &[3]).unwrap();
        let narrow = g.input("narrow", &[1, 3]).unwrap();
        let expanded = g.broadcast_to(row, &[2, 3]).unwrap();
        let sum = g.add(expanded, narrow).unwrap();
        g.set_outputs(vec![sum]).unwrap();

        let folded = eliminate_dead_code(&fold_broadcasts(&g).unwrap()).unwrap();
        assert!(folded.nodes().iter().any(|n| n.op == Op::BroadcastTo));
    }

    #[test]
    fn folds_both_operands_when_the_result_shape_allows() {
        let mut g = Graph::new();
        let a = g.input("a", &[2, 1]).unwrap();
        let b = g.input("b", &[1, 3]).unwrap();
        let wide = g.broadcast_to(a, &[2, 3]).unwrap();
        let tall = g.broadcast_to(b, &[2, 3]).unwrap();
        let sum = g.add(wide, tall).unwrap();
        g.set_outputs(vec![sum]).unwrap();

        let folded = eliminate_dead_code(&fold_broadcasts(&g).unwrap()).unwrap();
        assert_eq!(folded.len(), 3);
        assert_eq!(folded.shape(folded.outputs()[0]), &[2, 3]);
    }
}
