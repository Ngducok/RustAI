//! Python bindings. Arrays cross the boundary through the buffer protocol, so any C-contiguous
//! `float32` array (NumPy, PyTorch CPU tensors via `.numpy()`) can be passed without extra copies
//! on the Python side; results come back as NumPy arrays.

// Newer clippy flags the error conversions that PyO3's macros generate; they are not ours.
#![allow(clippy::useless_conversion)]

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rustai::prelude::*;

fn failure(error: rustai::Error) -> PyErr {
    match error {
        rustai::Error::Invalid(_) | rustai::Error::Tensor(_) | rustai::Error::Runtime(_) => {
            PyValueError::new_err(error.to_string())
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

fn to_tensor(array: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    let buffer: PyBuffer<f32> = PyBuffer::get_bound(array)?;
    let data = buffer.to_vec(array.py())?;
    Tensor::from_vec(data, buffer.shape()).map_err(|e| failure(e.into()))
}

fn to_numpy<'py>(py: Python<'py>, tensor: &Tensor) -> PyResult<Bound<'py, PyAny>> {
    let bytes: Vec<u8> = tensor.data().iter().flat_map(|v| v.to_le_bytes()).collect();
    let flat = py
        .import_bound("numpy")?
        .call_method1("frombuffer", (PyBytes::new_bound(py, &bytes), "float32"))?;
    flat.call_method1("reshape", (tensor.shape().to_vec(),))?
        .call_method0("copy")
}

/// Supervised training of a multilayer perceptron with AdamW.
#[pyclass]
struct MlpTrainer {
    inner: Trainer,
}

#[pymethods]
impl MlpTrainer {
    #[new]
    #[pyo3(signature = (sizes, batch, loss="mse", learning_rate=1e-2, weight_decay=0.0, seed=1))]
    fn new(
        sizes: Vec<usize>,
        batch: usize,
        loss: &str,
        learning_rate: f32,
        weight_decay: f32,
        seed: u64,
    ) -> PyResult<Self> {
        let loss = match loss {
            "mse" => Loss::Mse,
            "cross_entropy" => Loss::CrossEntropy,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown loss {other:?}; use \"mse\" or \"cross_entropy\""
                )))
            }
        };
        let config = TrainerConfig {
            sizes,
            batch,
            loss,
            learning_rate,
            weight_decay,
            seed,
        };
        Ok(Self {
            inner: Trainer::new(config).map_err(failure)?,
        })
    }

    /// One optimisation step; returns the loss before the update.
    fn step(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
    ) -> PyResult<f32> {
        let (x, target) = (to_tensor(x)?, to_tensor(target)?);
        py.allow_threads(|| self.inner.step(&x, &target))
            .map_err(failure)
    }

    /// Network outputs for up to `batch` rows.
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let x = to_tensor(x)?;
        let out = py
            .allow_threads(|| self.inner.predict(&x))
            .map_err(failure)?;
        to_numpy(py, &out)
    }

    /// Parameters as a dict of NumPy arrays: `w0`, `b0`, `w1`, `b1`, ... (weights are `in x out`).
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        let model = self.inner.model();
        for (name, tensor) in model.param_names().iter().zip(model.params()) {
            dict.set_item(name, to_numpy(py, tensor)?)?;
        }
        Ok(dict)
    }

    /// Replaces parameters from a dict shaped like the result of `parameters()`.
    fn set_parameters(&mut self, parameters: &Bound<'_, PyDict>) -> PyResult<()> {
        let names = self.inner.model().param_names();
        let mut replacements = Vec::new();
        for name in &names {
            let value = parameters
                .get_item(name)?
                .ok_or_else(|| PyValueError::new_err(format!("missing parameter {name}")))?;
            replacements.push(to_tensor(&value)?);
        }
        for ((name, slot), new) in names
            .iter()
            .zip(self.inner.model_mut().params_mut())
            .zip(replacements)
        {
            if slot.shape() != new.shape() {
                return Err(PyValueError::new_err(format!(
                    "parameter {name} has shape {:?}, expected {:?}",
                    new.shape(),
                    slot.shape()
                )));
            }
            *slot = new;
        }
        Ok(())
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(std::path::Path::new(path)).map_err(failure)
    }

    fn load(&mut self, path: &str) -> PyResult<()> {
        self.inner.load(std::path::Path::new(path)).map_err(failure)
    }

    #[getter]
    fn steps(&self) -> usize {
        self.inner.steps()
    }
}

/// Trains PPO on CartPole and returns the mean return of the greedy policy over 10 episodes.
#[pyfunction]
#[pyo3(signature = (total_steps, seed=1, learning_rate=1e-3))]
fn train_ppo_cartpole(
    py: Python<'_>,
    total_steps: usize,
    seed: u64,
    learning_rate: f32,
) -> PyResult<f32> {
    py.allow_threads(|| -> rustai::Result<f32> {
        let envs: Vec<Box<dyn Env>> = (0..8)
            .map(|i| Box::new(CartPole::new(seed * 100 + i)) as Box<dyn Env>)
            .collect();
        let mut ppo = Ppo::new(
            envs,
            PpoConfig {
                learning_rate,
                seed,
                ..PpoConfig::default()
            },
        )?;
        ppo.train(total_steps, |_| {})?;
        Ok(ppo.evaluate(&mut CartPole::new(seed + 9999), 10)?)
    })
    .map_err(failure)
}

#[pymodule]
#[pyo3(name = "rustai")]
fn rustai_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MlpTrainer>()?;
    m.add_function(wrap_pyfunction!(train_ppo_cartpole, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
