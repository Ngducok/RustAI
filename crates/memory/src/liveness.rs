use rt_ir::Graph;

/// A node's value exists from step `i` (its own index) until step `last_use[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Liveness {
    pub last_use: Vec<usize>,
}

/// Graph outputs stay live to the end of the schedule.
pub fn liveness(graph: &Graph) -> Liveness {
    let mut last_use: Vec<usize> = (0..graph.len()).collect();
    for (index, node) in graph.nodes().iter().enumerate() {
        for input in &node.inputs {
            last_use[input.0] = index;
        }
    }
    for output in graph.outputs() {
        last_use[output.0] = usize::MAX;
    }
    Liveness { last_use }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_use_is_the_latest_consumer() {
        let mut g = Graph::new();
        let a = g.input("a", &[2]).unwrap();
        let b = g.relu(a).unwrap();
        let c = g.scale(b, 2.0).unwrap();
        let d = g.add(b, c).unwrap();
        g.set_outputs(vec![d]).unwrap();
        let live = liveness(&g);
        assert_eq!(live.last_use, vec![1, 3, 3, usize::MAX]);
    }

    #[test]
    fn unused_nodes_die_immediately() {
        let mut g = Graph::new();
        let a = g.input("a", &[2]).unwrap();
        let _unused = g.relu(a).unwrap();
        assert_eq!(liveness(&g).last_use, vec![1, 1]);
    }
}
