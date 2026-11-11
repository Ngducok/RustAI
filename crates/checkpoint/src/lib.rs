//! Reading and writing tensors in the `safetensors` format, so checkpoints open directly in
//! Python (`safetensors.torch.load_file`, `safetensors.numpy.load_file`).
//!
//! Layout: an 8-byte little-endian header length, a JSON header mapping each tensor name to its
//! dtype, shape and byte range, then the raw little-endian data. Only `F32` tensors are supported.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use rt_tensor::Tensor;
use serde_json::{json, Value};

#[derive(Debug)]
pub enum CheckpointError {
    Io(std::io::Error),
    Format(String),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "i/o error: {e}"),
            Self::Format(m) => write!(f, "invalid checkpoint: {m}"),
        }
    }
}

impl std::error::Error for CheckpointError {}

impl From<std::io::Error> for CheckpointError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, CheckpointError>;

fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(CheckpointError::Format(message.into()))
}

/// Named tensors plus free-form string metadata.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Checkpoint {
    pub tensors: Vec<(String, Tensor)>,
    pub metadata: BTreeMap<String, String>,
}

impl Checkpoint {
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    /// Like `get`, but an absent tensor or one of the wrong shape is an error.
    pub fn expect(&self, name: &str, shape: &[usize]) -> Result<&Tensor> {
        match self.get(name) {
            None => invalid(format!("missing tensor {name}")),
            Some(t) if t.shape() != shape => invalid(format!(
                "tensor {name} has shape {:?}, expected {shape:?}",
                t.shape()
            )),
            Some(t) => Ok(t),
        }
    }
}

pub fn to_bytes(
    tensors: &[(&str, &Tensor)],
    metadata: &BTreeMap<String, String>,
) -> Result<Vec<u8>> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        header.insert("__metadata__".into(), json!(metadata));
    }
    let mut offset = 0usize;
    for (name, tensor) in tensors {
        if *name == "__metadata__" || header.contains_key(*name) {
            return invalid(format!("duplicate or reserved tensor name {name}"));
        }
        let end = offset + tensor.numel() * 4;
        header.insert(
            (*name).into(),
            json!({"dtype": "F32", "shape": tensor.shape(), "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    let mut text = Value::Object(header).to_string().into_bytes();
    text.resize(text.len().next_multiple_of(8), b' ');

    let mut bytes = Vec::with_capacity(8 + text.len() + offset);
    bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&text);
    for (_, tensor) in tensors {
        bytes.extend(tensor.data().iter().flat_map(|v| v.to_le_bytes()));
    }
    Ok(bytes)
}

pub fn from_bytes(bytes: &[u8]) -> Result<Checkpoint> {
    let length = bytes
        .get(..8)
        .ok_or_else(|| CheckpointError::Format("file too short".into()))?;
    let header_len = u64::from_le_bytes(length.try_into().expect("8 bytes")) as usize;
    let header_end = 8usize.checked_add(header_len).filter(|&e| e <= bytes.len());
    let Some(header_end) = header_end else {
        return invalid("header length exceeds the file");
    };
    let header: Value = serde_json::from_slice(&bytes[8..header_end])
        .map_err(|e| CheckpointError::Format(format!("header is not valid JSON: {e}")))?;
    let Value::Object(entries) = header else {
        return invalid("header must be a JSON object");
    };
    let data = &bytes[header_end..];

    let mut checkpoint = Checkpoint::default();
    let mut ranges = Vec::new();
    for (name, entry) in &entries {
        if name == "__metadata__" {
            let Value::Object(map) = entry else {
                return invalid("metadata must be an object");
            };
            for (key, value) in map {
                let Value::String(text) = value else {
                    return invalid("metadata values must be strings");
                };
                checkpoint.metadata.insert(key.clone(), text.clone());
            }
            continue;
        }
        if entry["dtype"] != "F32" {
            return invalid(format!(
                "tensor {name} has unsupported dtype {}",
                entry["dtype"]
            ));
        }
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .ok_or_else(|| CheckpointError::Format(format!("tensor {name} has no shape")))?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize))
            .collect::<Option<_>>()
            .ok_or_else(|| CheckpointError::Format(format!("tensor {name} has a bad shape")))?;
        let offsets = entry["data_offsets"]
            .as_array()
            .map(|o| (o[0].as_u64(), o.get(1).and_then(Value::as_u64)));
        let Some((Some(start), Some(end))) = offsets else {
            return invalid(format!("tensor {name} has bad data offsets"));
        };
        let (start, end) = (start as usize, end as usize);
        if start > end || end > data.len() || end - start != shape.iter().product::<usize>() * 4 {
            return invalid(format!("tensor {name} has inconsistent data offsets"));
        }
        let values = data[start..end]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        ranges.push((start, end));
        checkpoint.tensors.push((
            name.clone(),
            Tensor::from_vec(values, &shape).map_err(|e| CheckpointError::Format(e.to_string()))?,
        ));
    }
    ranges.sort_unstable();
    if ranges.windows(2).any(|w| w[0].1 > w[1].0) {
        return invalid("tensor data ranges overlap");
    }
    Ok(checkpoint)
}

pub fn save(
    path: &Path,
    tensors: &[(&str, &Tensor)],
    metadata: &BTreeMap<String, String>,
) -> Result<()> {
    std::fs::write(path, to_bytes(tensors, metadata)?)?;
    Ok(())
}

pub fn load(path: &Path) -> Result<Checkpoint> {
    from_bytes(&std::fs::read(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rt_tensor::Rng;

    fn sample() -> (Tensor, Tensor, Tensor) {
        let mut rng = Rng::new(1);
        (
            Tensor::uniform(&[3, 4], -1.0, 1.0, &mut rng),
            Tensor::uniform(&[5], -2.0, 2.0, &mut rng),
            Tensor::full(&[], 7.5),
        )
    }

    #[test]
    fn round_trips_values_shapes_and_metadata_exactly() {
        let (a, b, c) = sample();
        let meta = BTreeMap::from([("step".to_string(), "42".to_string())]);
        let bytes = to_bytes(&[("layer.w", &a), ("layer.b", &b), ("scalar", &c)], &meta).unwrap();
        let loaded = from_bytes(&bytes).unwrap();
        assert_eq!(loaded.metadata, meta);
        for (name, original) in [("layer.w", &a), ("layer.b", &b), ("scalar", &c)] {
            assert_eq!(loaded.get(name).unwrap(), original, "{name}");
        }
    }

    #[test]
    fn writes_the_documented_layout() {
        let (a, ..) = sample();
        let bytes = to_bytes(&[("w", &a)], &BTreeMap::new()).unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!(header_len % 8, 0, "header is padded to 8 bytes");
        assert_eq!(bytes.len(), 8 + header_len + 12 * 4);
        let header: Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        assert_eq!(header["w"]["dtype"], "F32");
        assert_eq!(header["w"]["shape"], json!([3, 4]));
        assert_eq!(header["w"]["data_offsets"], json!([0, 48]));
    }

    #[test]
    fn rejects_corrupt_and_unsupported_files() {
        let (a, b, _) = sample();
        let good = to_bytes(&[("a", &a), ("b", &b)], &BTreeMap::new()).unwrap();
        assert!(from_bytes(&good[..5]).is_err(), "truncated length");
        assert!(
            from_bytes(&good[..good.len() - 4]).is_err(),
            "truncated data"
        );
        let mut huge = good.clone();
        huge[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(from_bytes(&huge).is_err(), "absurd header length");

        let header = br#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0, 0]);
        assert!(from_bytes(&bytes).is_err(), "unsupported dtype");

        let overlapping = br#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#;
        let mut bytes = (overlapping.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(overlapping);
        bytes.extend_from_slice(&[0; 12]);
        assert!(from_bytes(&bytes).is_err(), "overlapping ranges");
    }

    #[test]
    fn rejects_duplicate_names_and_wrong_shapes_on_expect() {
        let (a, ..) = sample();
        assert!(to_bytes(&[("x", &a), ("x", &a)], &BTreeMap::new()).is_err());
        assert!(to_bytes(&[("__metadata__", &a)], &BTreeMap::new()).is_err());
        let loaded = from_bytes(&to_bytes(&[("x", &a)], &BTreeMap::new()).unwrap()).unwrap();
        assert!(loaded.expect("x", &[3, 4]).is_ok());
        assert!(loaded.expect("x", &[4, 3]).is_err());
        assert!(loaded.expect("missing", &[1]).is_err());
    }

    #[test]
    fn files_round_trip_through_the_filesystem() {
        let (a, ..) = sample();
        let path = std::env::temp_dir().join(format!(
            "rt-checkpoint-test-{}.safetensors",
            std::process::id()
        ));
        save(&path, &[("w", &a)], &BTreeMap::new()).unwrap();
        let loaded = load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded.get("w").unwrap(), &a);
        assert!(load(&path).is_err());
    }
}
