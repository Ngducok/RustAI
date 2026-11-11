use rt_compiler::optimize;
use rt_ir::{Graph, IrError, NodeId, Op};
use rt_memory::{plan, Plan};
use rt_tensor::Tensor;

use crate::dispatch::{check_feeds, execute_node, find_feed, MAX_INPUTS};
use crate::error::Result;
use crate::Feeds;

/// Executes a compiled graph out of buffers allocated once and reused across runs.
pub struct Executor {
    graph: Graph,
    plan: Plan,
    buffers: Vec<Vec<f32>>,
}

impl Executor {
    /// Dead-code elimination, then memory planning.
    pub fn compile(graph: &Graph) -> Result<Self> {
        let graph = optimize(graph)?;
        let plan = plan(&graph);
        Self::with_plan(graph, plan)
    }

    pub fn with_plan(graph: Graph, plan: Plan) -> Result<Self> {
        if graph
            .outputs()
            .iter()
            .any(|o| matches!(graph.node(*o).op, Op::Input(_)))
        {
            return Err(IrError::Invalid("an output cannot be a bare input".into()).into());
        }
        let buffers = plan
            .buffer_sizes
            .iter()
            .map(|&size| vec![0.0; size])
            .collect();
        Ok(Self {
            graph,
            plan,
            buffers,
        })
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    pub fn planned_elements(&self) -> usize {
        self.plan.planned_elements()
    }

    pub fn unplanned_elements(&self) -> usize {
        Plan::unplanned_elements(&self.graph)
    }

    pub fn run(&mut self, feeds: Feeds) -> Result<()> {
        check_feeds(&self.graph, feeds)?;
        for index in 0..self.graph.len() {
            if !matches!(self.graph.nodes()[index].op, Op::Input(_)) {
                self.run_node(index, feeds)?;
            }
        }
        Ok(())
    }

    pub fn output(&self, index: usize) -> &[f32] {
        let id = self.graph.outputs()[index];
        &self.buffers[self.buffer_of(id)][..self.numel(id)]
    }

    pub fn read_output(&self, index: usize, dst: &mut Tensor) {
        dst.data_mut().copy_from_slice(self.output(index));
    }

    pub fn output_tensor(&self, index: usize) -> Tensor {
        let id = self.graph.outputs()[index];
        Tensor::from_vec(self.output(index).to_vec(), self.graph.shape(id))
            .expect("output buffer fits its shape")
    }

    fn buffer_of(&self, id: NodeId) -> usize {
        self.plan.buffer_of[id.0].expect("computed nodes have buffers")
    }

    fn numel(&self, id: NodeId) -> usize {
        self.graph.shape(id).iter().product()
    }

    fn run_node(&mut self, index: usize, feeds: Feeds) -> Result<()> {
        let id = NodeId(index);
        let target = self.buffer_of(id);
        let mut out = std::mem::take(&mut self.buffers[target]);
        let numel = self.numel(id);
        let result = (|| {
            let mut ins: [&[f32]; MAX_INPUTS] = [&[]; MAX_INPUTS];
            for (slot, &input) in ins.iter_mut().zip(&self.graph.node(id).inputs) {
                *slot = match &self.graph.node(input).op {
                    Op::Input(name) => find_feed(feeds, name)?.data(),
                    _ => &self.buffers[self.buffer_of(input)][..self.numel(input)],
                };
            }
            execute_node(&self.graph, id, &ins, &mut out[..numel])
        })();
        self.buffers[target] = out;
        result
    }
}
