use rt_runtime::{Executor, Result, RuntimeError};
use rt_tensor::Tensor;

use crate::train_step::mlp_forward;

/// Evaluates an MLP on batches of varying size by rounding each batch up to a fixed bucket, so
/// only a handful of graphs are ever compiled. Rows are independent, and the matmul accumulates
/// each element in a fixed order, so padding never changes the rows that matter.
pub struct BucketedMlp {
    prefix: String,
    sizes: Vec<usize>,
    buckets: Vec<usize>,
    executors: Vec<Option<Executor>>,
    param_names: Vec<String>,
}

impl BucketedMlp {
    pub fn new(prefix: &str, sizes: &[usize], buckets: &[usize]) -> Result<Self> {
        let mut buckets = buckets.to_vec();
        buckets.sort_unstable();
        buckets.dedup();
        if buckets.first().is_none_or(|&b| b == 0) {
            return Err(RuntimeError::Graph(rt_ir::IrError::Invalid(
                "buckets must be non-empty and positive".into(),
            )));
        }
        let param_names = mlp_forward(prefix, sizes, buckets[0])?.param_names;
        Ok(Self {
            prefix: prefix.to_string(),
            sizes: sizes.to_vec(),
            executors: buckets.iter().map(|_| None).collect(),
            buckets,
            param_names,
        })
    }

    pub fn largest_bucket(&self) -> usize {
        self.buckets[self.buckets.len() - 1]
    }

    /// How many bucket graphs have been compiled so far.
    pub fn compiled(&self) -> usize {
        self.executors.iter().flatten().count()
    }

    /// Runs the network on `x` (`rows x inputs`) and returns `rows x outputs`.
    pub fn forward(&mut self, params: &[&Tensor], x: &Tensor) -> Result<Tensor> {
        let rows = x.shape()[0];
        let inputs = self.sizes[0];
        let slot = self
            .buckets
            .iter()
            .position(|&b| b >= rows)
            .ok_or_else(|| {
                RuntimeError::Graph(rt_ir::IrError::Invalid(format!(
                    "batch of {rows} exceeds the largest bucket {}",
                    self.largest_bucket()
                )))
            })?;
        let bucket = self.buckets[slot];
        if self.executors[slot].is_none() {
            let graph = mlp_forward(&self.prefix, &self.sizes, bucket)?.graph;
            self.executors[slot] = Some(Executor::compile(&graph)?);
        }
        let mut padded = Tensor::zeros(&[bucket, inputs]);
        padded.data_mut()[..rows * inputs].copy_from_slice(x.data());

        let executor = self.executors[slot].as_mut().expect("compiled above");
        let mut feeds = vec![("x", &padded)];
        feeds.extend(
            self.param_names
                .iter()
                .map(String::as_str)
                .zip(params.iter().copied()),
        );
        executor.run(&feeds)?;
        let outputs = *self.sizes.last().expect("an MLP has an output layer");
        Tensor::from_vec(
            executor.output(0)[..rows * outputs].to_vec(),
            &[rows, outputs],
        )
        .map_err(RuntimeError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mlp;
    use rt_tensor::Rng;

    #[test]
    fn padding_to_a_bucket_never_changes_the_rows_that_matter() {
        let mut rng = Rng::new(5);
        let model = Mlp::new(&[7, 20, 3], &mut rng);
        let mut bucketed = BucketedMlp::new("", &[7, 20, 3], &[4, 16, 64]).unwrap();
        for rows in [1, 3, 4, 5, 16, 17, 63, 64] {
            let x = Tensor::uniform(&[rows, 7], -1.0, 1.0, &mut rng);
            let exact = mlp_forward("", &[7, 20, 3], rows).unwrap();
            let mut direct = Executor::compile(&exact.graph).unwrap();
            let mut feeds = vec![("x", &x)];
            feeds.extend(
                exact
                    .param_names
                    .iter()
                    .map(String::as_str)
                    .zip(model.params()),
            );
            direct.run(&feeds).unwrap();

            let got = bucketed.forward(&model.params(), &x).unwrap();
            assert_eq!(got.shape(), &[rows, 3]);
            assert_eq!(got.data(), direct.output(0), "{rows} rows");
        }
    }

    #[test]
    fn compiles_one_graph_per_bucket_actually_used() {
        let mut rng = Rng::new(6);
        let model = Mlp::new(&[2, 4, 1], &mut rng);
        let mut bucketed = BucketedMlp::new("", &[2, 4, 1], &[8, 2, 8, 32]).unwrap();
        for rows in [1, 2, 1, 7, 8] {
            bucketed
                .forward(&model.params(), &Tensor::zeros(&[rows, 2]))
                .unwrap();
        }
        assert_eq!(bucketed.compiled(), 2);
        assert_eq!(bucketed.largest_bucket(), 32);
    }

    #[test]
    fn rejects_oversized_batches_and_empty_bucket_lists() {
        let mut rng = Rng::new(7);
        let model = Mlp::new(&[2, 1], &mut rng);
        let mut bucketed = BucketedMlp::new("", &[2, 1], &[4]).unwrap();
        assert!(bucketed
            .forward(&model.params(), &Tensor::zeros(&[5, 2]))
            .is_err());
        assert!(BucketedMlp::new("", &[2, 1], &[]).is_err());
        assert!(BucketedMlp::new("", &[2, 1], &[0, 4]).is_err());
    }
}
