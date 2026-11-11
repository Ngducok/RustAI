use rt_ir::{Graph, Op};
use rt_tensor::Tensor;

use crate::dispatch::{check_feeds, execute_node, find_feed, MAX_INPUTS};
use crate::error::Result;
use crate::Feeds;

/// Straightforward executor that allocates a fresh tensor for every node.
pub fn run_reference(graph: &Graph, feeds: Feeds) -> Result<Vec<Tensor>> {
    check_feeds(graph, feeds)?;
    let mut values: Vec<Tensor> = Vec::with_capacity(graph.len());
    for (index, node) in graph.nodes().iter().enumerate() {
        let value = match &node.op {
            Op::Input(name) => find_feed(feeds, name)?.clone(),
            _ => {
                let mut ins: [&[f32]; MAX_INPUTS] = [&[]; MAX_INPUTS];
                for (slot, input) in ins.iter_mut().zip(&node.inputs) {
                    *slot = values[input.0].data();
                }
                let mut out = Tensor::zeros(&node.shape);
                execute_node(graph, rt_ir::NodeId(index), &ins, out.data_mut())?;
                out
            }
        };
        values.push(value);
    }
    Ok(graph
        .outputs()
        .iter()
        .map(|o| values[o.0].clone())
        .collect())
}
