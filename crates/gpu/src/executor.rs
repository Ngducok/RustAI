use std::sync::Arc;

use rt_compiler::optimize;
use rt_ir::{Graph, IrError, NodeId, Op};
use rt_memory::plan;
use rt_runtime::{Feeds, RuntimeError};
use rt_tensor::Tensor;

use crate::context::GpuContext;
use crate::error::{GpuError, Result};
use crate::kernels::{launch, Kernel, WORDS};

struct InputSlot {
    name: String,
    shape: Vec<usize>,
    buffer: wgpu::Buffer,
}

pub(crate) struct Dispatch {
    pub(crate) pipeline: Arc<wgpu::ComputePipeline>,
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) groups: [u32; 3],
}

struct OutputSlot {
    source: usize,
    elements: usize,
    staging: wgpu::Buffer,
}

/// Hyperparameters for the on-device AdamW optimizer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuAdamW {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
}

impl Default for GpuAdamW {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }
}

struct ParamUpdate {
    dispatch: Dispatch,
    uniform: wgpu::Buffer,
    elements: usize,
}

struct OnDeviceAdamW {
    config: GpuAdamW,
    steps: i32,
    updates: Vec<ParamUpdate>,
}

/// Runs an optimised graph on the GPU. Buffers are allocated once from the memory plan.
///
/// `run` uploads every input, executes the graph and reads the outputs back. For training,
/// `set_inputs`, `execute` and `download` can be used separately so that parameters stay on the
/// device, optionally updated there by `enable_adamw`.
pub struct GpuExecutor {
    ctx: Arc<GpuContext>,
    graph: Graph,
    buffers: Vec<wgpu::Buffer>,
    inputs: Vec<InputSlot>,
    dispatches: Vec<Dispatch>,
    outputs: Vec<OutputSlot>,
    host: Vec<Vec<f32>>,
    optimizer: Option<OnDeviceAdamW>,
}

fn dispatch_all<'a>(
    pass: &mut wgpu::ComputePass<'a>,
    dispatches: impl Iterator<Item = &'a Dispatch>,
) {
    for d in dispatches {
        pass.set_pipeline(&d.pipeline);
        pass.set_bind_group(0, &d.bind_group, &[]);
        pass.dispatch_workgroups(d.groups[0], d.groups[1], d.groups[2]);
    }
}

impl GpuExecutor {
    pub fn compile(ctx: Arc<GpuContext>, graph: &Graph) -> Result<Self> {
        let graph = optimize(graph)?;
        if graph
            .outputs()
            .iter()
            .any(|o| matches!(graph.node(*o).op, Op::Input(_)))
        {
            return Err(IrError::Invalid("an output cannot be a bare input".into()).into());
        }
        let layout = plan(&graph);
        let buffers: Vec<wgpu::Buffer> = layout
            .buffer_sizes
            .iter()
            .map(|&n| ctx.storage_buffer(n))
            .collect();
        let placeholder = ctx.storage_buffer(4);

        let mut inputs: Vec<InputSlot> = Vec::new();
        let mut input_of = vec![usize::MAX; graph.len()];
        for (name, id) in graph.input_names() {
            input_of[id.0] = inputs.len();
            inputs.push(InputSlot {
                name: name.to_string(),
                shape: graph.shape(id).to_vec(),
                buffer: ctx.storage_buffer(graph.shape(id).iter().product()),
            });
        }
        let buffer_for = |id: NodeId| -> &wgpu::Buffer {
            match layout.buffer_of[id.0] {
                Some(b) => &buffers[b],
                None => &inputs[input_of[id.0]].buffer,
            }
        };

        let mut dispatches = Vec::new();
        for (index, node) in graph.nodes().iter().enumerate() {
            let Some(plan) = launch(&graph, node)? else {
                continue;
            };
            let params = ctx.uniform_buffer(&plan.words);
            let mut bound: Vec<&wgpu::Buffer> = vec![&params];
            bound.extend(
                plan.bindings
                    .iter()
                    .map(|b| b.map_or(&placeholder, buffer_for)),
            );
            bound.push(buffer_for(NodeId(index)));
            dispatches.push(ctx.dispatch(plan.kernel, &bound, plan.groups));
        }

        let outputs = graph
            .outputs()
            .iter()
            .map(|&o| {
                let elements = graph.shape(o).iter().product::<usize>();
                OutputSlot {
                    source: layout.buffer_of[o.0].expect("outputs are computed nodes"),
                    elements,
                    staging: ctx.staging_buffer(elements),
                }
            })
            .collect::<Vec<_>>();
        let host = outputs.iter().map(|o| vec![0.0; o.elements]).collect();
        Ok(Self {
            ctx,
            graph,
            buffers,
            inputs,
            dispatches,
            outputs,
            host,
            optimizer: None,
        })
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Uploads the inputs named in `feeds`; inputs not mentioned keep their device values.
    pub fn set_inputs(&mut self, feeds: Feeds) -> Result<()> {
        for (name, tensor) in feeds {
            let slot = self
                .inputs
                .iter()
                .find(|s| s.name == *name)
                .ok_or_else(|| RuntimeError::MissingInput((*name).to_string()))?;
            if tensor.shape() != slot.shape {
                return Err(RuntimeError::InputShape {
                    name: slot.name.clone(),
                    expected: slot.shape.clone(),
                    actual: tensor.shape().to_vec(),
                }
                .into());
            }
            self.ctx
                .queue
                .write_buffer(&slot.buffer, 0, bytemuck::cast_slice(tensor.data()));
        }
        Ok(())
    }

    /// Enqueues the whole graph (and the optimizer step, if enabled) without waiting for it.
    pub fn execute(&mut self) {
        if let Some(opt) = &mut self.optimizer {
            opt.steps += 1;
            let (c, t) = (opt.config, opt.steps);
            let corrections = (1.0 - c.beta1.powi(t), 1.0 - c.beta2.powi(t));
            for update in &opt.updates {
                let mut words = [0u32; WORDS];
                let values = [
                    c.learning_rate,
                    c.beta1,
                    c.beta2,
                    c.epsilon,
                    c.weight_decay,
                    corrections.0,
                    corrections.1,
                ];
                words[0] = update.elements as u32;
                for (i, v) in values.iter().enumerate() {
                    words[1 + i] = v.to_bits();
                }
                self.ctx
                    .queue
                    .write_buffer(&update.uniform, 0, bytemuck::cast_slice(&words));
            }
        }
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            dispatch_all(&mut pass, self.dispatches.iter());
            if let Some(opt) = &self.optimizer {
                dispatch_all(&mut pass, opt.updates.iter().map(|u| &u.dispatch));
            }
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Blocks until everything enqueued so far has finished.
    pub fn synchronize(&self) {
        self.ctx.device.poll(wgpu::Maintain::Wait);
    }

    /// Copies the graph outputs to host memory.
    pub fn download(&mut self) -> Result<()> {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for out in self.outputs.iter().filter(|o| o.elements > 0) {
            encoder.copy_buffer_to_buffer(
                &self.buffers[out.source],
                0,
                &out.staging,
                0,
                (out.elements * 4) as u64,
            );
        }
        self.ctx.queue.submit(Some(encoder.finish()));
        let (sender, receiver) = std::sync::mpsc::channel();
        for out in self.outputs.iter().filter(|o| o.elements > 0) {
            let sender = sender.clone();
            out.staging.slice(..(out.elements * 4) as u64).map_async(
                wgpu::MapMode::Read,
                move |r| {
                    sender.send(r).ok();
                },
            );
        }
        drop(sender);
        self.ctx.device.poll(wgpu::Maintain::Wait);
        for result in receiver {
            result.map_err(|e| GpuError::Device(e.to_string()))?;
        }
        for (out, host) in self.outputs.iter().zip(&mut self.host) {
            if out.elements > 0 {
                {
                    let view = out
                        .staging
                        .slice(..(out.elements * 4) as u64)
                        .get_mapped_range();
                    host.copy_from_slice(bytemuck::cast_slice(&view));
                }
                out.staging.unmap();
            }
        }
        Ok(())
    }

    /// Uploads all inputs, runs the graph and downloads the outputs.
    pub fn run(&mut self, feeds: Feeds) -> Result<()> {
        if let Some(missing) = self
            .inputs
            .iter()
            .find(|s| !feeds.iter().any(|(n, _)| *n == s.name))
        {
            return Err(RuntimeError::MissingInput(missing.name.clone()).into());
        }
        let used: Vec<_> = feeds
            .iter()
            .filter(|(name, _)| self.inputs.iter().any(|s| s.name == *name))
            .copied()
            .collect();
        self.set_inputs(&used)?;
        self.execute();
        self.download()
    }

    /// Updates the named inputs in place on the device after every `execute`, using the graph
    /// outputs at the given indices as their gradients.
    pub fn enable_adamw(&mut self, params: &[(&str, usize)], config: GpuAdamW) -> Result<()> {
        let mut updates = Vec::new();
        for (name, output) in params {
            let slot = self
                .inputs
                .iter()
                .find(|s| s.name == *name)
                .ok_or_else(|| RuntimeError::MissingInput((*name).to_string()))?;
            let out = self
                .outputs
                .get(*output)
                .ok_or_else(|| GpuError::Device(format!("no output {output}")))?;
            let elements: usize = slot.shape.iter().product();
            if out.elements != elements {
                return Err(GpuError::Device(format!(
                    "gradient {output} does not match parameter {name}"
                )));
            }
            let (first, second) = (
                self.ctx.storage_buffer(elements),
                self.ctx.storage_buffer(elements),
            );
            let uniform = self.ctx.uniform_buffer(&[0; WORDS]);
            let bound = [
                &uniform,
                &slot.buffer,
                &self.buffers[out.source],
                &first,
                &second,
            ];
            let dispatch = self.ctx.dispatch(
                Kernel::AdamW,
                &bound,
                [elements.div_ceil(256).max(1) as u32, 1, 1],
            );
            updates.push(ParamUpdate {
                dispatch,
                uniform,
                elements,
            });
        }
        self.optimizer = Some(OnDeviceAdamW {
            config,
            steps: 0,
            updates,
        });
        Ok(())
    }

    /// Reads an input (for example a parameter updated on the device) back to the host.
    pub fn read_input(&self, name: &str) -> Result<Tensor> {
        let slot = self
            .inputs
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| RuntimeError::MissingInput(name.to_string()))?;
        let elements: usize = slot.shape.iter().product();
        let staging = self.ctx.staging_buffer(elements);
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&slot.buffer, 0, &staging, 0, (elements * 4) as u64);
        self.ctx.queue.submit(Some(encoder.finish()));
        let (sender, receiver) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            sender.send(r).ok();
        });
        self.ctx.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .expect("map callback runs")
            .map_err(|e| GpuError::Device(e.to_string()))?;
        let data = {
            let view = staging.slice(..).get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&view)[..elements].to_vec()
        };
        Ok(Tensor::from_vec(data, &slot.shape)?)
    }

    pub fn output(&self, index: usize) -> &[f32] {
        &self.host[index]
    }

    pub fn output_tensor(&self, index: usize) -> Tensor {
        let shape = self.graph.shape(self.graph.outputs()[index]);
        Tensor::from_vec(self.host[index].clone(), shape).expect("output fits its shape")
    }

    pub fn read_output(&self, index: usize, dst: &mut Tensor) {
        dst.data_mut().copy_from_slice(&self.host[index]);
    }
}
