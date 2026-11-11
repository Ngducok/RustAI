use rt_ir::{Graph, Op};

use crate::liveness::liveness;

/// Maps every computed node to a reusable buffer; inputs are read straight from the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub buffer_of: Vec<Option<usize>>,
    pub buffer_sizes: Vec<usize>,
}

fn numel(graph: &Graph, index: usize) -> usize {
    graph.nodes()[index].shape.iter().product()
}

fn is_computed(graph: &Graph, index: usize) -> bool {
    !matches!(graph.nodes()[index].op, Op::Input(_))
}

impl Plan {
    pub fn planned_elements(&self) -> usize {
        self.buffer_sizes.iter().sum()
    }

    /// Elements needed if every computed node kept its own buffer.
    pub fn unplanned_elements(graph: &Graph) -> usize {
        (0..graph.len())
            .filter(|&i| is_computed(graph, i))
            .map(|i| numel(graph, i))
            .sum()
    }

    /// Independent check: a buffer never holds two values at the same time.
    pub fn validate(&self, graph: &Graph) -> Result<(), String> {
        let last_use = liveness(graph).last_use;
        for index in 0..graph.len() {
            match (is_computed(graph, index), self.buffer_of[index]) {
                (true, None) => return Err(format!("node {index} has no buffer")),
                (false, Some(_)) => {
                    return Err(format!("input node {index} must not have a buffer"))
                }
                (true, Some(b)) if self.buffer_sizes[b] < numel(graph, index) => {
                    return Err(format!("buffer {b} is too small for node {index}"))
                }
                _ => {}
            }
        }
        for (a, &last) in last_use.iter().enumerate() {
            for b in a + 1..graph.len() {
                let shared = self.buffer_of[a].is_some() && self.buffer_of[a] == self.buffer_of[b];
                if shared && b <= last {
                    return Err(format!(
                        "nodes {a} and {b} overlap in buffer {:?}",
                        self.buffer_of[a]
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Greedy linear scan over the schedule: reuse the tightest free buffer, else grow the largest.
pub fn plan(graph: &Graph) -> Plan {
    let last_use = liveness(graph).last_use;
    let mut expiring: Vec<Vec<usize>> = vec![Vec::new(); graph.len()];
    for (index, &last) in last_use.iter().enumerate() {
        if last < graph.len() {
            expiring[last].push(index);
        }
    }

    let mut buffer_of = vec![None; graph.len()];
    let mut buffer_sizes: Vec<usize> = Vec::new();
    let mut free: Vec<usize> = Vec::new();
    for index in (0..graph.len()).filter(|&i| is_computed(graph, i)) {
        let need = numel(graph, index);
        let tightest = free
            .iter()
            .enumerate()
            .filter(|(_, &b)| buffer_sizes[b] >= need)
            .min_by_key(|(_, &b)| buffer_sizes[b]);
        let largest = free
            .iter()
            .enumerate()
            .max_by_key(|(_, &b)| buffer_sizes[b]);
        let buffer = match tightest.or(largest) {
            Some((slot, _)) => free.swap_remove(slot),
            None => {
                buffer_sizes.push(0);
                buffer_sizes.len() - 1
            }
        };
        buffer_sizes[buffer] = buffer_sizes[buffer].max(need);
        buffer_of[index] = Some(buffer);

        for &dead in &expiring[index] {
            if let Some(b) = buffer_of[dead] {
                free.push(b);
            }
        }
    }
    Plan {
        buffer_of,
        buffer_sizes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relu_chain(length: usize) -> Graph {
        let mut g = Graph::new();
        let mut v = g.input("x", &[8]).unwrap();
        for _ in 0..length {
            v = g.relu(v).unwrap();
        }
        g.set_outputs(vec![v]).unwrap();
        g
    }

    #[test]
    fn a_chain_ping_pongs_between_two_buffers() {
        let g = relu_chain(10);
        let p = plan(&g);
        assert_eq!(p.buffer_sizes, vec![8, 8]);
        assert_eq!(p.planned_elements(), 16);
        assert_eq!(Plan::unplanned_elements(&g), 80);
        p.validate(&g).unwrap();
    }

    #[test]
    fn outputs_are_never_recycled() {
        let mut g = Graph::new();
        let x = g.input("x", &[4]).unwrap();
        let a = g.relu(x).unwrap();
        let b = g.scale(a, 2.0).unwrap();
        let c = g.scale(b, 3.0).unwrap();
        g.set_outputs(vec![a, c]).unwrap();
        let p = plan(&g);
        p.validate(&g).unwrap();
        assert_ne!(p.buffer_of[a.0], p.buffer_of[b.0]);
        assert_ne!(p.buffer_of[a.0], p.buffer_of[c.0]);
    }

    #[test]
    fn grows_a_free_buffer_instead_of_allocating_a_new_one() {
        let mut g = Graph::new();
        let x = g.input("x", &[2]).unwrap();
        let small = g.relu(x).unwrap();
        let small2 = g.scale(small, 2.0).unwrap();
        let big = g.broadcast_to(small2, &[4, 2]).unwrap();
        let big2 = g.scale(big, 2.0).unwrap();
        g.set_outputs(vec![big2]).unwrap();
        let p = plan(&g);
        p.validate(&g).unwrap();
        assert_eq!(p.buffer_sizes, vec![8, 8]);
        assert_eq!(Plan::unplanned_elements(&g), 2 + 2 + 8 + 8);
    }

    #[test]
    fn validate_catches_overlapping_assignments() {
        let g = relu_chain(3);
        let mut p = plan(&g);
        p.buffer_of[2] = p.buffer_of[1];
        assert!(p.validate(&g).is_err());
    }

    #[test]
    fn validate_catches_missing_and_undersized_buffers() {
        let g = relu_chain(3);
        let mut p = plan(&g);
        p.buffer_of[1] = None;
        assert!(p.validate(&g).is_err());
        let mut p = plan(&g);
        p.buffer_sizes[0] = 1;
        assert!(p.validate(&g).is_err());
    }
}
