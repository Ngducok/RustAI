use rt_ir::{Graph, NodeId, Result};

/// Returns a copy of `graph` holding only nodes that the outputs depend on.
pub fn eliminate_dead_code(graph: &Graph) -> Result<Graph> {
    let mut live = vec![false; graph.len()];
    graph.outputs().iter().for_each(|o| live[o.0] = true);
    for index in (0..graph.len()).rev() {
        if live[index] {
            graph.nodes()[index]
                .inputs
                .iter()
                .for_each(|i| live[i.0] = true);
        }
    }

    let mut renumbered: Vec<Option<NodeId>> = vec![None; graph.len()];
    let mut pruned = Graph::new();
    for (index, node) in graph.nodes().iter().enumerate().filter(|(i, _)| live[*i]) {
        let inputs: Vec<NodeId> = node.inputs.iter().filter_map(|i| renumbered[i.0]).collect();
        renumbered[index] = Some(pruned.add_node(node.op.clone(), &inputs, Some(&node.shape))?);
    }
    pruned.set_outputs(
        graph
            .outputs()
            .iter()
            .filter_map(|o| renumbered[o.0])
            .collect(),
    )?;
    Ok(pruned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_unreachable_nodes_and_keeps_order() {
        let mut g = Graph::new();
        let a = g.input("a", &[2, 2]).unwrap();
        let dead_input = g.input("unused", &[3]).unwrap();
        let dead = g.relu(a).unwrap();
        let _ = g.sum_all(dead).unwrap();
        let _ = dead_input;
        let kept = g.scale(a, 2.0).unwrap();
        g.set_outputs(vec![kept]).unwrap();

        let pruned = eliminate_dead_code(&g).unwrap();
        assert_eq!(pruned.len(), 2);
        assert_eq!(
            pruned.input_names().map(|(n, _)| n).collect::<Vec<_>>(),
            ["a"]
        );
        assert_eq!(pruned.outputs(), &[NodeId(1)]);
    }

    #[test]
    fn keeps_everything_that_is_needed_and_is_idempotent() {
        let mut g = Graph::new();
        let a = g.input("a", &[2, 2]).unwrap();
        let b = g.relu(a).unwrap();
        let c = g.add(a, b).unwrap();
        g.set_outputs(vec![c, b]).unwrap();
        let once = eliminate_dead_code(&g).unwrap();
        assert_eq!(once, g);
        assert_eq!(eliminate_dead_code(&once).unwrap(), once);
    }
}
