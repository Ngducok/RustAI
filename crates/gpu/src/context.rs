use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{GpuError, Result};
use crate::kernels::Kernel;

/// A device, its queue and a cache of compiled shader pipelines.
pub struct GpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    name: String,
    backend: String,
    pipelines: Mutex<HashMap<Kernel, Arc<wgpu::ComputePipeline>>>,
}

impl GpuContext {
    /// Opens the highest-performance adapter available.
    pub fn new() -> Result<Arc<Self>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or(GpuError::NoAdapter)?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rt-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .map_err(|e| GpuError::Device(e.to_string()))?;
        Ok(Arc::new(Self {
            device,
            queue,
            name: info.name,
            backend: format!("{:?}", info.backend),
            pipelines: Mutex::new(HashMap::new()),
        }))
    }

    pub fn adapter_name(&self) -> &str {
        &self.name
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub(crate) fn storage_buffer(&self, elements: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (elements.max(4) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub(crate) fn pipeline(&self, kernel: Kernel) -> Arc<wgpu::ComputePipeline> {
        let mut cache = self
            .pipelines
            .lock()
            .expect("pipeline cache is never poisoned");
        cache
            .entry(kernel)
            .or_insert_with(|| {
                let module = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(kernel.name()),
                        source: wgpu::ShaderSource::Wgsl(kernel.source().into()),
                    });
                Arc::new(
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(kernel.name()),
                            layout: None,
                            module: &module,
                            entry_point: "main",
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            cache: None,
                        }),
                )
            })
            .clone()
    }

    pub(crate) fn uniform_buffer(&self, words: &[u32; crate::kernels::WORDS]) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (crate::kernels::WORDS * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(words));
        buffer
    }

    pub(crate) fn staging_buffer(&self, elements: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (elements.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Binds `buffers` (parameter block first) to a kernel's slots in order.
    pub(crate) fn dispatch(
        &self,
        kernel: Kernel,
        buffers: &[&wgpu::Buffer],
        groups: [u32; 3],
    ) -> crate::executor::Dispatch {
        let pipeline = self.pipeline(kernel);
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(kernel.name()),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        crate::executor::Dispatch {
            pipeline,
            bind_group,
            groups,
        }
    }
}
