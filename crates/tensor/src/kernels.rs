//! Allocation-free kernels writing into caller-provided buffers.

use crate::error::{Result, TensorError};
use crate::gemm::{self, Epilogue, Layout, Product};
use crate::shape::{broadcast_strides, Strides, MAX_RANK};

fn check_len(data: &[f32], shape: &[usize]) -> Result<()> {
    if data.len() == shape.iter().product::<usize>() {
        Ok(())
    } else {
        Err(TensorError::ShapeMismatch(format!(
            "buffer of {} elements does not fit shape {shape:?}",
            data.len()
        )))
    }
}

fn walk(shape: &[usize], sa: &Strides, sb: &Strides, mut f: impl FnMut(usize, usize, usize)) {
    let mut index = [0usize; MAX_RANK];
    let (mut oa, mut ob) = (0, 0);
    for i in 0..shape.iter().product::<usize>() {
        f(i, oa, ob);
        for d in (0..shape.len()).rev() {
            index[d] += 1;
            oa += sa[d];
            ob += sb[d];
            if index[d] < shape[d] {
                break;
            }
            oa -= sa[d] * shape[d];
            ob -= sb[d] * shape[d];
            index[d] = 0;
        }
    }
}

pub fn map_into(src: &[f32], out: &mut [f32], f: impl Fn(f32) -> f32) -> Result<()> {
    if src.len() != out.len() {
        return Err(TensorError::ShapeMismatch(
            "map buffers differ in length".into(),
        ));
    }
    out.iter_mut().zip(src).for_each(|(o, &x)| *o = f(x));
    Ok(())
}

pub fn zip_into(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    out: &mut [f32],
    out_shape: &[usize],
    f: impl Fn(f32, f32) -> f32,
) -> Result<()> {
    check_len(a, a_shape)?;
    check_len(b, b_shape)?;
    check_len(out, out_shape)?;
    if a_shape == out_shape && b_shape == out_shape {
        out.iter_mut()
            .zip(a)
            .zip(b)
            .for_each(|((o, &x), &y)| *o = f(x, y));
        return Ok(());
    }
    let sa = broadcast_strides(a_shape, out_shape)?;
    let sb = broadcast_strides(b_shape, out_shape)?;
    walk(out_shape, &sa, &sb, |i, oa, ob| out[i] = f(a[oa], b[ob]));
    Ok(())
}

pub fn broadcast_into(
    src: &[f32],
    src_shape: &[usize],
    out: &mut [f32],
    out_shape: &[usize],
) -> Result<()> {
    check_len(src, src_shape)?;
    check_len(out, out_shape)?;
    let strides = broadcast_strides(src_shape, out_shape)?;
    walk(out_shape, &strides, &[0; MAX_RANK], |i, offset, _| {
        out[i] = src[offset]
    });
    Ok(())
}

/// Sums `src` down to `out_shape`, the adjoint of `broadcast_into`.
pub fn reduce_into(
    src: &[f32],
    src_shape: &[usize],
    out: &mut [f32],
    out_shape: &[usize],
) -> Result<()> {
    check_len(src, src_shape)?;
    check_len(out, out_shape)?;
    let strides = broadcast_strides(out_shape, src_shape)?;
    out.fill(0.0);
    walk(src_shape, &[0; MAX_RANK], &strides, |i, _, offset| {
        out[offset] += src[i]
    });
    Ok(())
}

fn multiply(product: Product, out: &mut [f32]) -> Result<()> {
    let Product {
        a,
        b,
        m,
        k,
        n,
        epilogue,
        ..
    } = product;
    let bias_fits = epilogue.bias.is_none_or(|bias| bias.len() == n);
    let mask_fits = epilogue.mask.is_none_or(|mask| mask.len() == m * n);
    if a.len() != m * k || b.len() != k * n || out.len() != m * n || !bias_fits || !mask_fits {
        return Err(TensorError::ShapeMismatch(format!(
            "matmul buffers do not fit {m}x{k} * {k}x{n}"
        )));
    }
    if m * n == 0 {
        return Ok(());
    }
    if k == 0 {
        gemm::empty_product(n, &epilogue, out);
    } else {
        gemm::matmul(&product, out);
    }
    Ok(())
}

fn plain<'a>(
    a: &'a [f32],
    a_layout: Layout,
    b: &'a [f32],
    b_layout: Layout,
    dims: (usize, usize, usize),
) -> Product<'a> {
    let (m, k, n) = dims;
    let epilogue = Epilogue::default();
    Product {
        a,
        a_layout,
        b,
        b_layout,
        m,
        k,
        n,
        epilogue,
    }
}

/// `out = a * b` with `a` as `m x k` and `b` as `k x n`.
pub fn matmul_into(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    multiply(
        plain(a, Layout::RowMajor, b, Layout::RowMajor, (m, k, n)),
        out,
    )
}

/// `out = a * b^T` with `a` as `m x k` and `b` stored as `n x k`.
pub fn matmul_nt_into(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    multiply(
        plain(a, Layout::RowMajor, b, Layout::Transposed, (m, k, n)),
        out,
    )
}

/// `out = a^T * b` with `a` stored as `k x m` and `b` as `k x n`.
pub fn matmul_tn_into(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    multiply(
        plain(a, Layout::Transposed, b, Layout::RowMajor, (m, k, n)),
        out,
    )
}

/// `out = x * w + bias`, optionally clamped at zero, computed as one pass over the output.
/// `x` is `m x k`, `w` is `k x n`, `bias` has `n` elements; `dims` is `(m, k, n)`.
pub fn linear_into(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    relu: bool,
    dims: (usize, usize, usize),
    out: &mut [f32],
) -> Result<()> {
    let mut product = plain(x, Layout::RowMajor, w, Layout::RowMajor, dims);
    product.epilogue = Epilogue {
        bias: Some(bias),
        relu,
        mask: None,
    };
    multiply(product, out)
}

/// `out = (a * b^T)` zeroed wherever `mask <= 0`: a matmul fused with a ReLU gradient mask.
/// `a` is `m x k`, `b` is stored `n x k`, `mask` is `m x n`; `dims` is `(m, k, n)`.
pub fn matmul_nt_masked_into(
    a: &[f32],
    b: &[f32],
    mask: &[f32],
    dims: (usize, usize, usize),
    out: &mut [f32],
) -> Result<()> {
    let mut product = plain(a, Layout::RowMajor, b, Layout::Transposed, dims);
    product.epilogue = Epilogue {
        bias: None,
        relu: false,
        mask: Some(mask),
    };
    multiply(product, out)
}

pub use crate::gemm::uses_fma as matmul_uses_fma;
pub fn transpose_into(src: &[f32], rows: usize, cols: usize, out: &mut [f32]) -> Result<()> {
    if src.len() != rows * cols || out.len() != rows * cols {
        return Err(TensorError::ShapeMismatch(
            "transpose buffers do not fit".into(),
        ));
    }
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = src[i * cols + j];
        }
    }
    Ok(())
}

/// Row-wise `x - max - ln(sum(exp(x - max)))`, stable for large logits.
pub fn log_softmax_rows_into(src: &[f32], rows: usize, cols: usize, out: &mut [f32]) -> Result<()> {
    if src.len() != rows * cols || out.len() != src.len() {
        return Err(TensorError::ShapeMismatch(
            "log-softmax buffers do not fit".into(),
        ));
    }
    for (row, dst) in src.chunks(cols.max(1)).zip(out.chunks_mut(cols.max(1))) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum = row.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
        for (d, &x) in dst.iter_mut().zip(row) {
            *d = (x - max) - log_sum;
        }
    }
    Ok(())
}

/// Gradient of `min(a, b)` for one operand: the whole gradient where it is the smaller value,
/// half of it on ties, none otherwise.
pub fn minimum_grad_into(
    a: &[f32],
    b: &[f32],
    grad: &[f32],
    first: bool,
    out: &mut [f32],
) -> Result<()> {
    if a.len() != b.len() || a.len() != grad.len() || a.len() != out.len() {
        return Err(TensorError::ShapeMismatch(
            "minimum grad buffers differ in length".into(),
        ));
    }
    for (((o, &x), &y), &g) in out.iter_mut().zip(a).zip(b).zip(grad) {
        let (mine, other) = if first { (x, y) } else { (y, x) };
        *o = if mine < other {
            g
        } else if mine == other {
            0.5 * g
        } else {
            0.0
        };
    }
    Ok(())
}

fn class_of(label: f32, cols: usize) -> Result<usize> {
    if label >= 0.0 && label.fract() == 0.0 && (label as usize) < cols {
        Ok(label as usize)
    } else {
        Err(TensorError::InvalidArgument(format!(
            "label {label} is not a class below {cols}"
        )))
    }
}

fn row_max_and_sum(row: &[f32]) -> (f32, f32) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (max, row.iter().map(|&x| (x - max).exp()).sum())
}

/// Mean negative log-likelihood of integer class `labels` (stored as f32) under softmax rows.
pub fn cross_entropy(logits: &[f32], rows: usize, cols: usize, labels: &[f32]) -> Result<f32> {
    if logits.len() != rows * cols || labels.len() != rows {
        return Err(TensorError::ShapeMismatch(
            "cross-entropy buffers do not fit".into(),
        ));
    }
    let mut total = 0.0;
    for (row, &label) in logits.chunks(cols.max(1)).zip(labels) {
        let (max, sum) = row_max_and_sum(row);
        total -= (row[class_of(label, cols)?] - max) - sum.ln();
    }
    Ok(total / rows.max(1) as f32)
}

pub fn cross_entropy_grad_into(
    logits: &[f32],
    rows: usize,
    cols: usize,
    labels: &[f32],
    upstream: f32,
    out: &mut [f32],
) -> Result<()> {
    if logits.len() != rows * cols || labels.len() != rows || out.len() != logits.len() {
        return Err(TensorError::ShapeMismatch(
            "cross-entropy buffers do not fit".into(),
        ));
    }
    let scale = upstream / rows.max(1) as f32;
    for ((row, grad), &label) in logits
        .chunks(cols.max(1))
        .zip(out.chunks_mut(cols.max(1)))
        .zip(labels)
    {
        let class = class_of(label, cols)?;
        let (max, sum) = row_max_and_sum(row);
        for (g, &x) in grad.iter_mut().zip(row) {
            *g = (x - max).exp() / sum * scale;
        }
        grad[class] -= scale;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rng, Tensor};

    #[test]
    fn cross_entropy_matches_log_softmax() {
        let logits = Tensor::uniform(&[5, 4], -3.0, 3.0, &mut Rng::new(1));
        let labels = [0.0, 3.0, 1.0, 2.0, 2.0];
        let log_probs = logits.log_softmax_rows().unwrap();
        let expected: f32 = labels
            .iter()
            .enumerate()
            .map(|(r, &l)| -log_probs.data()[r * 4 + l as usize])
            .sum::<f32>()
            / 5.0;
        assert!((cross_entropy(logits.data(), 5, 4, &labels).unwrap() - expected).abs() < 1e-5);
    }

    #[test]
    fn cross_entropy_grad_matches_finite_differences() {
        let logits = Tensor::uniform(&[3, 4], -2.0, 2.0, &mut Rng::new(2));
        let labels = [1.0, 0.0, 3.0];
        let mut grad = vec![0.0; 12];
        cross_entropy_grad_into(logits.data(), 3, 4, &labels, 1.0, &mut grad).unwrap();
        for i in 0..12 {
            let eval = |delta: f32| {
                let mut x = logits.data().to_vec();
                x[i] += delta;
                cross_entropy(&x, 3, 4, &labels).unwrap()
            };
            let numeric = (eval(1e-2) - eval(-1e-2)) / 2e-2;
            assert!(
                (grad[i] - numeric).abs() < 2e-3,
                "element {i}: {} vs {numeric}",
                grad[i]
            );
        }
        let mut doubled = vec![0.0; 12];
        cross_entropy_grad_into(logits.data(), 3, 4, &labels, 2.0, &mut doubled).unwrap();
        assert!(grad
            .iter()
            .zip(&doubled)
            .all(|(g, d)| (2.0 * g - d).abs() < 1e-6));
    }

    #[test]
    fn cross_entropy_rejects_invalid_labels() {
        let logits = [0.0; 6];
        for bad in [-1.0, 0.5, 3.0, f32::NAN] {
            assert!(
                cross_entropy(&logits, 2, 3, &[0.0, bad]).is_err(),
                "label {bad}"
            );
        }
        assert!(cross_entropy(&logits, 2, 3, &[0.0]).is_err());
    }

    #[test]
    fn reduce_is_the_adjoint_of_broadcast() {
        let src = [1.0, 2.0, 3.0];
        let mut expanded = [0.0; 6];
        broadcast_into(&src, &[3], &mut expanded, &[2, 3]).unwrap();
        assert_eq!(expanded, [1., 2., 3., 1., 2., 3.]);
        let mut reduced = [0.0; 3];
        reduce_into(&expanded, &[2, 3], &mut reduced, &[3]).unwrap();
        assert_eq!(reduced, [2., 4., 6.]);
        let mut scalar = [0.0; 1];
        reduce_into(&expanded, &[2, 3], &mut scalar, &[]).unwrap();
        assert_eq!(scalar, [12.0]);
    }

    #[test]
    fn linear_into_equals_matmul_then_bias_then_relu_bit_for_bit() {
        let mut rng = Rng::new(41);
        for (m, k, n) in [(1, 1, 1), (5, 7, 9), (13, 257, 17), (40, 64, 33)] {
            let x = Tensor::uniform(&[m, k], -1.0, 1.0, &mut rng);
            let w = Tensor::uniform(&[k, n], -1.0, 1.0, &mut rng);
            let bias = Tensor::uniform(&[n], -1.0, 1.0, &mut rng);
            let separate = x.matmul(&w).unwrap().add(&bias).unwrap();
            for relu in [false, true] {
                let expected = if relu {
                    separate.relu()
                } else {
                    separate.clone()
                };
                let mut out = vec![f32::NAN; m * n];
                linear_into(x.data(), w.data(), bias.data(), relu, (m, k, n), &mut out).unwrap();
                assert_eq!(out, expected.data(), "{m}x{k}x{n} relu={relu}");
            }
        }
    }

    #[test]
    fn masked_product_equals_matmul_nt_then_relu_gradient_bit_for_bit() {
        let mut rng = Rng::new(51);
        for (m, k, n) in [(1, 1, 1), (7, 5, 9), (13, 257, 17), (40, 64, 33)] {
            let a = Tensor::uniform(&[m, k], -1.0, 1.0, &mut rng);
            let b = Tensor::uniform(&[n, k], -1.0, 1.0, &mut rng);
            let mask = Tensor::uniform(&[m, n], -1.0, 1.0, &mut rng);
            let expected = mask
                .zip_with(
                    &a.matmul_nt(&b).unwrap(),
                    |x, g| if x > 0.0 { g } else { 0.0 },
                )
                .unwrap();
            let mut out = vec![f32::NAN; m * n];
            matmul_nt_masked_into(a.data(), b.data(), mask.data(), (m, k, n), &mut out).unwrap();
            assert_eq!(out, expected.data(), "{m}x{k}x{n}");
            assert!(matmul_nt_masked_into(
                a.data(),
                b.data(),
                &mask.data()[1..],
                (m, k, n),
                &mut out
            )
            .is_err());
        }
    }
    #[test]
    fn linear_into_handles_empty_depth_and_rejects_bad_bias() {
        let bias = [-1.0, 2.0, 0.5];
        let mut out = [f32::NAN; 6];
        linear_into(&[], &[], &bias, true, (2, 0, 3), &mut out).unwrap();
        assert_eq!(out, [0.0, 2.0, 0.5, 0.0, 2.0, 0.5]);
        linear_into(&[], &[], &bias, false, (2, 0, 3), &mut out).unwrap();
        assert_eq!(out, [-1.0, 2.0, 0.5, -1.0, 2.0, 0.5]);
        let mut sized = [0.0; 4];
        assert!(linear_into(&[0.0; 4], &[0.0; 4], &bias, false, (2, 2, 2), &mut sized).is_err());
    }
    #[test]
    fn matmul_into_checks_every_buffer_independently() {
        let (a, b) = ([0.0f32; 6], [0.0f32; 12]);
        let mut out = [0.0f32; 8];
        assert!(matmul_into(&a, &b, 2, 3, 4, &mut out).is_ok());
        assert!(matmul_into(&a[..5], &b, 2, 3, 4, &mut out).is_err());
        assert!(matmul_into(&a, &b[..11], 2, 3, 4, &mut out).is_err());
        assert!(matmul_into(&a, &b, 2, 3, 4, &mut out[..7]).is_err());
    }

    #[test]
    fn matmul_into_zeroes_the_output_for_empty_products() {
        let mut out = [f32::NAN; 6];
        matmul_into(&[], &[], 2, 0, 3, &mut out).unwrap();
        assert_eq!(out, [0.0; 6]);
        let mut empty: [f32; 0] = [];
        matmul_into(&[], &[1.0; 4], 0, 1, 4, &mut empty).unwrap();
    }

    #[test]
    fn kernels_reject_mismatched_buffers() {
        let mut out = [0.0; 4];
        assert!(zip_into(
            &[1.0; 3],
            &[2, 2],
            &[1.0; 4],
            &[2, 2],
            &mut out,
            &[2, 2],
            |a, b| a + b
        )
        .is_err());
        assert!(zip_into(
            &[1.0; 4],
            &[2, 2],
            &[1.0; 4],
            &[2, 2],
            &mut out[..3],
            &[2, 2],
            |a, b| a + b
        )
        .is_err());
        assert!(matmul_into(&[0.0; 4], &[0.0; 4], 2, 2, 3, &mut out).is_err());
        assert!(transpose_into(&[0.0; 4], 2, 3, &mut out).is_err());
        assert!(map_into(&[0.0; 3], &mut out, |x| x).is_err());
    }

    #[test]
    fn broadcasting_rejects_rank_above_maximum() {
        let shape = [1usize; MAX_RANK + 1];
        let mut out = [0.0; 1];
        assert!(broadcast_into(&[1.0], &[], &mut out, &shape).is_err());
    }
}
