use std::collections::HashMap;
use std::mem::{discriminant, Discriminant};

use rt_ir::{Graph, Node, NodeId, Op, Result};

#[derive(PartialEq, Eq, Hash)]
struct Signature {
    op: Discriminant<Op>,
    payload: (u32, u32),
    inputs: Vec<NodeId>,
    shape: Vec<usize>,
}

fn payload(op: &Op) -> (u32, u32) {
    match op {
        Op::Scale(value) | Op::Full(value) => (value.to_bits(), 0),
        Op::Linear { relu } => (u32::from(*relu), 0),
        Op::Clamp { min, max } | Op::ClampGrad { min, max } => (min.to_bits(), max.to_bits()),
        Op::MinimumGrad { first } => (u32::from(*first), 0),
        _ => (0, 0),
    }
}

/// Inputs are never merged; commutative ops ignore operand order.
fn signature(node: &Node, inputs: &[NodeId]) -> Option<Signature> {
    if matches!(node.op, Op::Input(_)) {
        return None;
    }
    let mut inputs = inputs.to_vec();
    if matches!(node.op, Op::Add | Op::Mul) {
        inputs.sort();
    }
    Some(Signature {
        op: discriminant(&node.op),
        payload: payload(&node.op),
        inputs,
        shape: node.shape.clone(),
    })
}

/// Merges nodes that apply the same op to the same operands.
pub fn eliminate_common_subexpressions(graph: &Graph) -> Result<Graph> {
    let mut merged = Graph::new();
    let mut renumbered: Vec<NodeId> = Vec::with_capacity(graph.len());
    let mut seen: HashMap<Signature, NodeId> = HashMap::new();
    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| renumbered[i.0]).collect();
        let id = match signature(node, &inputs) {
            Some(sig) => match seen.get(&sig) {
                Some(&existing) => existing,
                None => {
                    let id = merged.add_node(node.op.clone(), &inputs, Some(&node.shape))?;
                    seen.insert(sig, id);
                    id
                }
            },
            None => merged.add_node(node.op.clone(), &inputs, Some(&node.shape))?,
        };
        renumbered.push(id);
    }
    merged.set_outputs(graph.outputs().iter().map(|o| renumbered[o.0]).collect())?;
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_with_duplicates() -> (Graph, NodeId, NodeId) {
        let mut g = Graph::new();
        let a = g.input("a", &[2, 3]).unwrap();
        let b = g.input("b", &[2, 3]).unwrap();
        let first = g.add(a, b).unwrap();
        let second = g.add(b, a).unwrap();
        (g, first, second)
    }

    #[test]
    fn merges_identical_and_commuted_operations() {
        let (mut g, first, second) = graph_with_duplicates();
        let sum = g.add(first, second).unwrap();
        g.set_outputs(vec![sum]).unwrap();
        let merged = eliminate_common_subexpressions(&g).unwrap();
        assert_eq!(merged.len(), 4);
        let last = merged.node(merged.outputs()[0]);
        assert_eq!(last.inputs[0], last.inputs[1]);
    }

    #[test]
    fn keeps_non_commutative_operand_order() {
        let mut g = Graph::new();
        let a = g.input("a", &[2]).unwrap();
        let b = g.input("b", &[2]).unwrap();
        let (x, y) = (g.sub(a, b).unwrap(), g.sub(b, a).unwrap());
        g.set_outputs(vec![x, y]).unwrap();
        assert_eq!(eliminate_common_subexpressions(&g).unwrap().len(), 4);
    }

    #[test]
    fn distinguishes_constants_scales_and_inputs() {
        let mut g = Graph::new();
        let a = g.input("a", &[2]).unwrap();
        let other = g.input("other", &[2]).unwrap();
        let outputs = vec![
            g.full(1.0, &[2]).unwrap(),
            g.full(2.0, &[2]).unwrap(),
            g.full(1.0, &[2]).unwrap(),
            g.scale(a, 2.0).unwrap(),
            g.scale(a, 3.0).unwrap(),
            g.scale(a, 2.0).unwrap(),
            other,
        ];
        g.set_outputs(outputs).unwrap();
        let merged = eliminate_common_subexpressions(&g).unwrap();
        assert_eq!(merged.len(), 6);
        let outs = merged.outputs();
        assert_eq!(outs[0], outs[2]);
        assert_eq!(outs[3], outs[5]);
        assert_ne!(outs[0], outs[1]);
        assert_ne!(outs[3], outs[4]);
    }

    #[test]
    fn broadcast_targets_are_part_of_the_signature() {
        let mut g = Graph::new();
        let a = g.input("a", &[3]).unwrap();
        let wide = g.broadcast_to(a, &[2, 3]).unwrap();
        let tall = g.broadcast_to(a, &[4, 3]).unwrap();
        let again = g.broadcast_to(a, &[2, 3]).unwrap();
        g.set_outputs(vec![wide, tall, again]).unwrap();
        let merged = eliminate_common_subexpressions(&g).unwrap();
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn linear_nodes_that_differ_only_in_relu_stay_separate() {
        let mut g = Graph::new();
        let x = g.input("x", &[2, 3]).unwrap();
        let w = g.input("w", &[3, 4]).unwrap();
        let b = g.input("b", &[4]).unwrap();
        let outputs = vec![
            g.linear(x, w, b, false).unwrap(),
            g.linear(x, w, b, true).unwrap(),
            g.linear(x, w, b, false).unwrap(),
        ];
        g.set_outputs(outputs).unwrap();
        let merged = eliminate_common_subexpressions(&g).unwrap();
        assert_eq!(merged.len(), 5);
        let outs = merged.outputs();
        assert_eq!(outs[0], outs[2]);
        assert_ne!(outs[0], outs[1]);
    }
    #[test]
    fn is_idempotent() {
        let (mut g, first, second) = graph_with_duplicates();
        let product = g.mul(first, second).unwrap();
        g.set_outputs(vec![product]).unwrap();
        let once = eliminate_common_subexpressions(&g).unwrap();
        assert_eq!(eliminate_common_subexpressions(&once).unwrap(), once);
    }
}
