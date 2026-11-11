use std::collections::BTreeMap;
use std::path::Path;

use rt_checkpoint::{load, save, Checkpoint};
use rt_nn::{mlp_train_step, BucketedMlp, Loss, Mlp};
use rt_optim::{AdamW, Optimizer};
use rt_runtime::Executor;
use rt_tensor::{Rng, Tensor};

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Layer widths: the input size, every hidden size and the output size.
    pub sizes: Vec<usize>,
    /// Number of rows in every training batch.
    pub batch: usize,
    pub loss: Loss,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub seed: u64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            sizes: vec![2, 16, 2],
            batch: 32,
            loss: Loss::Mse,
            learning_rate: 1e-2,
            weight_decay: 0.0,
            seed: 1,
        }
    }
}

/// Supervised training of an MLP: each step runs a compiled forward + backward graph out of
/// preallocated buffers and applies AdamW.
pub struct Trainer {
    config: TrainerConfig,
    model: Mlp,
    executor: Executor,
    param_names: Vec<String>,
    optimizer: AdamW,
    grads: Vec<Tensor>,
    predictor: BucketedMlp,
    steps: usize,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> Result<Self> {
        let step = mlp_train_step(&config.sizes, config.batch, config.loss)?;
        let mut model = Mlp::new(&config.sizes, &mut Rng::new(config.seed));
        let grads = model
            .params_mut()
            .iter()
            .map(|p| Tensor::zeros(p.shape()))
            .collect();
        let mut buckets: Vec<usize> = (0..)
            .map(|p| 1usize << p)
            .take_while(|&b| b < config.batch)
            .collect();
        buckets.push(config.batch);
        Ok(Self {
            executor: Executor::compile(&step.graph)?,
            param_names: step.param_names,
            optimizer: AdamW::new(config.learning_rate).with_weight_decay(config.weight_decay),
            predictor: BucketedMlp::new("", &config.sizes, &buckets)?,
            model,
            grads,
            config,
            steps: 0,
        })
    }

    pub fn model(&self) -> &Mlp {
        &self.model
    }

    /// Mutable access to the model, for example to load weights trained elsewhere.
    pub fn model_mut(&mut self) -> &mut Mlp {
        &mut self.model
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    /// The gradients of the most recent step, in the order of `model().params()`.
    pub fn gradients(&self) -> &[Tensor] {
        &self.grads
    }

    /// One optimisation step on a batch; returns the loss before the update. `target` holds the
    /// regression targets (`batch x outputs`) or, for cross-entropy, one class index per row.
    pub fn step(&mut self, x: &Tensor, target: &Tensor) -> Result<f32> {
        let mut feeds = vec![("x", x), ("target", target)];
        feeds.extend(
            self.param_names
                .iter()
                .map(String::as_str)
                .zip(self.model.params()),
        );
        self.executor.run(&feeds)?;
        let loss = self.executor.output(0)[0];
        for (i, grad) in self.grads.iter_mut().enumerate() {
            self.executor.read_output(i + 1, grad);
        }
        self.optimizer
            .step(&mut self.model.params_mut(), &self.grads)?;
        self.steps += 1;
        Ok(loss)
    }

    /// Network outputs for any number of rows up to `batch`.
    pub fn predict(&mut self, x: &Tensor) -> Result<Tensor> {
        Ok(self.predictor.forward(&self.model.params(), x)?)
    }

    /// Writes parameters, optimizer state and the step count as a `safetensors` file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let names = self.model.param_names();
        let mut entries: Vec<(String, &Tensor)> =
            names.iter().cloned().zip(self.model.params()).collect();
        let (m, v) = self.optimizer.moments();
        entries.extend(
            m.iter()
                .enumerate()
                .map(|(i, t)| (format!("adam.m.{i}"), t)),
        );
        entries.extend(
            v.iter()
                .enumerate()
                .map(|(i, t)| (format!("adam.v.{i}"), t)),
        );
        let metadata = BTreeMap::from([
            ("steps".to_string(), self.steps.to_string()),
            ("adam_steps".to_string(), self.optimizer.steps().to_string()),
        ]);
        let borrowed: Vec<(&str, &Tensor)> =
            entries.iter().map(|(n, t)| (n.as_str(), *t)).collect();
        Ok(save(path, &borrowed, &metadata)?)
    }

    /// Restores a file written by `save`; training then continues exactly where it stopped.
    pub fn load(&mut self, path: &Path) -> Result<()> {
        let checkpoint: Checkpoint = load(path)?;
        let names = self.model.param_names();
        let mut params = self.model.params_mut();
        for (name, param) in names.iter().zip(params.iter_mut()) {
            **param = checkpoint.expect(name, param.shape())?.clone();
        }
        let adam_steps: i32 = parse(&checkpoint, "adam_steps")?;
        let moments = |prefix: &str| -> Result<Vec<Tensor>> {
            if adam_steps == 0 {
                return Ok(Vec::new());
            }
            (0..names.len())
                .map(|i| {
                    Ok(checkpoint
                        .expect(&format!("{prefix}.{i}"), self.grads[i].shape())?
                        .clone())
                })
                .collect()
        };
        self.optimizer
            .restore(adam_steps, moments("adam.m")?, moments("adam.v")?)?;
        self.steps = parse(&checkpoint, "steps")?;
        Ok(())
    }

    pub fn config(&self) -> &TrainerConfig {
        &self.config
    }
}

fn parse<T: std::str::FromStr>(checkpoint: &Checkpoint, key: &str) -> Result<T> {
    checkpoint
        .metadata
        .get(key)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| Error::Invalid(format!("checkpoint is missing the `{key}` entry")))
}
