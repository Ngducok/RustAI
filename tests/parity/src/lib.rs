//! Shared helpers for parity tests against PyTorch reference values; see `generate.py`.

use std::path::Path;

use rt_optim::{AdamW, Optimizer, Sgd};
use rt_tensor::Tensor;
use serde_json::Value;

#[derive(Clone, Copy)]
pub struct Tol {
    pub atol: f32,
    pub rtol: f32,
}

pub const SINGLE_STEP: Tol = Tol {
    atol: 1e-5,
    rtol: 1e-4,
};
pub const MULTI_STEP: Tol = Tol {
    atol: 1e-3,
    rtol: 1e-3,
};

pub fn fixtures() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/pytorch.json");
    let text = std::fs::read_to_string(path).expect("run tests/parity/generate.py first");
    serde_json::from_str(&text).unwrap()
}

pub fn cases<'a>(fixtures: &'a Value, group: &str) -> &'a Vec<Value> {
    let list = fixtures[group].as_array().unwrap();
    assert!(!list.is_empty(), "no {group} fixtures");
    list
}

pub fn name(case: &Value) -> &str {
    case["name"].as_str().unwrap()
}

pub fn tensor(v: &Value) -> Tensor {
    let shape: Vec<usize> = v["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_u64().unwrap() as usize)
        .collect();
    let data = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    Tensor::from_vec(data, &shape).unwrap()
}

pub fn tensors(v: &Value) -> Vec<Tensor> {
    v.as_array().unwrap().iter().map(tensor).collect()
}

pub fn indices(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|i| i.as_u64().unwrap() as usize)
        .collect()
}

pub fn assert_close(case: &str, what: &str, actual: &Tensor, expected: &Tensor, tol: Tol) {
    assert_eq!(actual.shape(), expected.shape(), "{case}: {what} shape");
    for (i, (&a, &e)) in actual.data().iter().zip(expected.data()).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= tol.atol + tol.rtol * e.abs(),
            "{case}: {what}[{i}] is {a}, PyTorch gives {e}"
        );
    }
}

pub fn assert_scalar_close(case: &str, what: &str, actual: f32, expected: f64, tol: Tol) {
    let expected = Tensor::full(&[], expected as f32);
    assert_close(case, what, &Tensor::full(&[], actual), &expected, tol);
}

pub fn optimizer(spec: &Value) -> Box<dyn Optimizer> {
    let lr = spec["lr"].as_f64().unwrap() as f32;
    match spec["kind"].as_str().unwrap() {
        "adamw" => {
            let decay = spec["weight_decay"].as_f64().unwrap() as f32;
            Box::new(AdamW::new(lr).with_weight_decay(decay))
        }
        "sgd" => Box::new(Sgd::new(lr, spec["momentum"].as_f64().unwrap() as f32)),
        other => panic!("unknown optimizer {other}"),
    }
}
