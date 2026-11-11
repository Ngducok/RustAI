use crate::error::{Result, TensorError};

pub const MAX_RANK: usize = 8;
pub type Strides = [usize; MAX_RANK];

pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let n = a.len().max(b.len());
    let dim = |s: &[usize], i: usize| {
        if i < n - s.len() {
            1
        } else {
            s[i - (n - s.len())]
        }
    };
    (0..n)
        .map(|i| match (dim(a, i), dim(b, i)) {
            (x, y) if x == y || y == 1 => Ok(x),
            (1, y) => Ok(y),
            _ => Err(TensorError::ShapeMismatch(format!(
                "cannot broadcast {a:?} with {b:?}"
            ))),
        })
        .collect()
}

/// Strides that read `shape` as if it were broadcast to `out_shape` (0 on broadcast dims).
pub fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Result<Strides> {
    if out_shape.len() > MAX_RANK || shape.len() > out_shape.len() {
        return Err(TensorError::ShapeMismatch(format!(
            "cannot view {shape:?} as {out_shape:?} (max rank {MAX_RANK})"
        )));
    }
    let offset = out_shape.len() - shape.len();
    let mut strides = [0; MAX_RANK];
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        let dim = shape[i];
        if dim != 1 && dim != out_shape[i + offset] {
            return Err(TensorError::ShapeMismatch(format!(
                "cannot view {shape:?} as {out_shape:?}"
            )));
        }
        strides[i + offset] = if dim == 1 { 0 } else { stride };
        stride *= dim;
    }
    Ok(strides)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert!(contiguous_strides(&[]).is_empty());
    }

    #[test]
    fn broadcast() {
        assert_eq!(broadcast_shape(&[2, 3], &[3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[4, 1], &[1, 5]).unwrap(), vec![4, 5]);
        assert!(broadcast_shape(&[2, 3], &[4]).is_err());
    }

    #[test]
    fn broadcast_strides_zero_on_expanded_dims() {
        assert_eq!(&broadcast_strides(&[3], &[2, 3]).unwrap()[..2], &[0, 1]);
        assert_eq!(&broadcast_strides(&[2, 1], &[2, 3]).unwrap()[..2], &[1, 0]);
        assert_eq!(&broadcast_strides(&[], &[2, 3]).unwrap()[..2], &[0, 0]);
        assert!(broadcast_strides(&[4], &[2, 3]).is_err());
        assert!(broadcast_strides(&[2, 3], &[3]).is_err());
    }
}
