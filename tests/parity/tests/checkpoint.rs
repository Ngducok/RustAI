//! Interoperability with the Python `safetensors` library. Skipped when Python or the library is missing.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use rt_checkpoint::{load, save};
use rt_tensor::Tensor;
use serde_json::Value;

fn python(args: &[&str]) -> Option<String> {
    let output = Command::new("python").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

fn safetensors_available() -> bool {
    let available = python(&["-c", "import safetensors, numpy"]).is_some();
    if !available {
        eprintln!("skipping: python with safetensors and numpy is not available");
    }
    available
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("rt-parity-{}-{name}", std::process::id()))
}

#[test]
fn python_reads_files_written_by_rust() {
    if !safetensors_available() {
        return;
    }
    let weights =
        Tensor::from_vec((0..12).map(|i| i as f32 * 0.25 - 1.0).collect(), &[3, 4]).unwrap();
    let bias = Tensor::from_vec(vec![1.5, -2.25, 3.0], &[3]).unwrap();
    let scalar = Tensor::full(&[], 7.0);
    let path = scratch("rust.safetensors");
    let meta = BTreeMap::from([("origin".to_string(), "rust".to_string())]);
    save(
        &path,
        &[
            ("layer.w", &weights),
            ("layer.b", &bias),
            ("scalar", &scalar),
        ],
        &meta,
    )
    .unwrap();

    let script = "import json, sys\nfrom safetensors import safe_open\nout = {}\n\
        with safe_open(sys.argv[1], framework='numpy') as f:\n    meta = f.metadata()\n    \
        for k in f.keys():\n        t = f.get_tensor(k)\n        out[k] = {'shape': list(t.shape), 'dtype': str(t.dtype), 'data': t.ravel().tolist()}\n\
        print(json.dumps({'meta': meta, 'tensors': out}))";
    let printed =
        python(&["-c", script, path.to_str().unwrap()]).expect("python failed to read the file");
    std::fs::remove_file(&path).unwrap();
    let parsed: Value = serde_json::from_str(printed.trim()).unwrap();

    assert_eq!(parsed["meta"]["origin"], "rust");
    for (name, tensor) in [
        ("layer.w", &weights),
        ("layer.b", &bias),
        ("scalar", &scalar),
    ] {
        let entry = &parsed["tensors"][name];
        assert_eq!(entry["dtype"], "float32", "{name}");
        let shape: Vec<u64> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(
            shape,
            tensor.shape().iter().map(|&d| d as u64).collect::<Vec<_>>(),
            "{name}"
        );
        let data: Vec<f32> = entry["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(data, tensor.data(), "{name}");
    }
}

#[test]
fn rust_reads_files_written_by_python() {
    if !safetensors_available() {
        return;
    }
    let path = scratch("python.safetensors");
    let script = "import sys\nimport numpy as np\nfrom safetensors.numpy import save_file\n\
        save_file({'a': (np.arange(12, dtype=np.float32).reshape(3, 4) * 0.5), \
        'b': np.array([1.5, -2.25, 3.0], dtype=np.float32), 's': np.array(7.0, dtype=np.float32)}, \
        sys.argv[1], metadata={'origin': 'python'})";
    python(&["-c", script, path.to_str().unwrap()]).expect("python failed to write the file");
    let loaded = load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(loaded.metadata["origin"], "python");
    let expected_a: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    assert_eq!(loaded.expect("a", &[3, 4]).unwrap().data(), expected_a);
    assert_eq!(loaded.expect("b", &[3]).unwrap().data(), &[1.5, -2.25, 3.0]);
    assert_eq!(loaded.expect("s", &[]).unwrap().data(), &[7.0]);
}
