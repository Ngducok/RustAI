use crate::error::{Result, TensorError};
use crate::kernels;
use crate::rng::Rng;
use crate::shape::broadcast_shape;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        let n: usize = shape.iter().product();
        if n != data.len() {
            return Err(TensorError::ShapeMismatch(format!(
                "shape {shape:?} needs {n} elements, got {}",
                data.len()
            )));
        }
        Ok(Self {
            data,
            shape: shape.to_vec(),
        })
    }

    pub fn full(shape: &[usize], value: f32) -> Self {
        Self {
            data: vec![value; shape.iter().product()],
            shape: shape.to_vec(),
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self::full(shape, 0.0)
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self::full(shape, 1.0)
    }

    pub fn uniform(shape: &[usize], lo: f32, hi: f32, rng: &mut Rng) -> Self {
        let n = shape.iter().product();
        Self {
            data: (0..n).map(|_| rng.uniform(lo, hi)).collect(),
            shape: shape.to_vec(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        Self::from_vec(self.data.clone(), shape)
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        Self {
            data: self.data.iter().map(|&x| f(x)).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn scale(&self, s: f32) -> Self {
        self.map(|x| x * s)
    }

    pub fn relu(&self) -> Self {
        self.map(|x| x.max(0.0))
    }

    pub fn zip_with(&self, other: &Self, f: impl Fn(f32, f32) -> f32) -> Result<Self> {
        let shape = broadcast_shape(&self.shape, &other.shape)?;
        let mut data = vec![0.0; shape.iter().product()];
        kernels::zip_into(
            &self.data,
            &self.shape,
            &other.data,
            &other.shape,
            &mut data,
            &shape,
            f,
        )?;
        Ok(Self { data, shape })
    }
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, |a, b| a * b)
    }

    pub fn div(&self, other: &Self) -> Result<Self> {
        self.zip_with(other, |a, b| a / b)
    }

    pub fn sum_all(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn sum_axis(&self, axis: usize) -> Result<Self> {
        if axis >= self.shape.len() {
            return Err(TensorError::InvalidArgument(format!(
                "axis {axis} out of range for rank {}",
                self.shape.len()
            )));
        }
        let outer: usize = self.shape[..axis].iter().product();
        let len = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut data = vec![0.0f32; outer * inner];
        for o in 0..outer {
            for l in 0..len {
                let src = &self.data[(o * len + l) * inner..][..inner];
                for (dst, s) in data[o * inner..][..inner].iter_mut().zip(src) {
                    *dst += s;
                }
            }
        }
        let mut shape = self.shape.clone();
        shape.remove(axis);
        Ok(Self { data, shape })
    }

    /// Reverses broadcasting: sums this tensor down to `target`.
    pub fn sum_to_shape(&self, target: &[usize]) -> Result<Self> {
        let mut data = vec![0.0; target.iter().product()];
        kernels::reduce_into(&self.data, &self.shape, &mut data, target)?;
        Ok(Self {
            data,
            shape: target.to_vec(),
        })
    }
    pub fn transpose2d(&self) -> Result<Self> {
        let (rows, cols) = self.dims2()?;
        let mut data = vec![0.0; rows * cols];
        kernels::transpose_into(&self.data, rows, cols, &mut data)?;
        Ok(Self {
            data,
            shape: vec![cols, rows],
        })
    }

    pub fn matmul(&self, other: &Self) -> Result<Self> {
        let (m, k) = self.dims2()?;
        let (k2, n) = other.dims2()?;
        if k != k2 {
            return Err(TensorError::ShapeMismatch(format!(
                "matmul {:?} x {:?}",
                self.shape, other.shape
            )));
        }
        let mut data = vec![0.0; m * n];
        kernels::matmul_into(&self.data, &other.data, m, k, n, &mut data)?;
        Ok(Self {
            data,
            shape: vec![m, n],
        })
    }
    /// `self * other^T` for `self` as `m x k` and `other` as `n x k`.
    pub fn matmul_nt(&self, other: &Self) -> Result<Self> {
        let (m, k) = self.dims2()?;
        let (n, k2) = other.dims2()?;
        if k != k2 {
            return Err(TensorError::ShapeMismatch(format!(
                "matmul_nt {:?} x {:?}^T",
                self.shape, other.shape
            )));
        }
        let mut data = vec![0.0; m * n];
        kernels::matmul_nt_into(&self.data, &other.data, m, k, n, &mut data)?;
        Ok(Self {
            data,
            shape: vec![m, n],
        })
    }

    /// `self^T * other` for `self` as `k x m` and `other` as `k x n`.
    pub fn matmul_tn(&self, other: &Self) -> Result<Self> {
        let (k, m) = self.dims2()?;
        let (k2, n) = other.dims2()?;
        if k != k2 {
            return Err(TensorError::ShapeMismatch(format!(
                "matmul_tn {:?}^T x {:?}",
                self.shape, other.shape
            )));
        }
        let mut data = vec![0.0; m * n];
        kernels::matmul_tn_into(&self.data, &other.data, m, k, n, &mut data)?;
        Ok(Self {
            data,
            shape: vec![m, n],
        })
    }
    pub fn log_softmax_rows(&self) -> Result<Self> {
        let (rows, cols) = self.dims2()?;
        let mut data = vec![0.0; self.data.len()];
        kernels::log_softmax_rows_into(&self.data, rows, cols, &mut data)?;
        Ok(Self {
            data,
            shape: self.shape.clone(),
        })
    }

    pub fn argmax_rows(&self) -> Result<Vec<usize>> {
        let (_, cols) = self.dims2()?;
        Ok(self
            .data
            .chunks(cols.max(1))
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or(0, |(i, _)| i)
            })
            .collect())
    }

    fn dims2(&self) -> Result<(usize, usize)> {
        match self.shape[..] {
            [r, c] => Ok((r, c)),
            _ => Err(TensorError::InvalidArgument(format!(
                "expected rank 2, got shape {:?}",
                self.shape
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape).unwrap()
    }

    #[test]
    fn add_broadcast_row() {
        let sum = t(&[1., 2., 3., 4., 5., 6.], &[2, 3])
            .add(&t(&[10., 20., 30.], &[3]))
            .unwrap();
        assert_eq!(sum.data(), &[11., 22., 33., 14., 25., 36.]);
    }

    #[test]
    fn add_broadcast_outer() {
        let sum = t(&[1., 2.], &[2, 1])
            .add(&t(&[10., 20., 30.], &[1, 3]))
            .unwrap();
        assert_eq!(sum.shape(), &[2, 3]);
        assert_eq!(sum.data(), &[11., 21., 31., 12., 22., 32.]);
    }

    #[test]
    fn matmul() {
        let a = t(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        let b = t(&[7., 8., 9., 10., 11., 12.], &[3, 2]);
        assert_eq!(a.matmul(&b).unwrap().data(), &[58., 64., 139., 154.]);
        assert!(a.matmul(&Tensor::zeros(&[2, 2])).is_err());
    }

    #[test]
    fn reductions() {
        let a = t(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        assert_eq!(a.sum_axis(0).unwrap().data(), &[5., 7., 9.]);
        assert_eq!(a.sum_axis(1).unwrap().data(), &[6., 15.]);
        assert_eq!(a.sum_all(), 21.0);
    }

    #[test]
    fn sum_to_shape_inverts_broadcast() {
        let g = Tensor::ones(&[2, 3]);
        assert_eq!(g.sum_to_shape(&[3]).unwrap().data(), &[2., 2., 2.]);
        assert_eq!(g.sum_to_shape(&[2, 1]).unwrap().data(), &[3., 3.]);
        assert_eq!(g.sum_to_shape(&[1, 1]).unwrap().data(), &[6.]);
        assert!(g.sum_to_shape(&[4]).is_err());
    }

    #[test]
    fn transpose() {
        let b = t(&[1., 2., 3., 4., 5., 6.], &[2, 3]).transpose2d().unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.data(), &[1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn log_softmax_rows_normalise() {
        let p = t(&[1., 2., 3., 1000., 1000., 1000.], &[2, 3])
            .log_softmax_rows()
            .unwrap();
        for row in p.data().chunks(3) {
            assert!((row.iter().map(|x| x.exp()).sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn matmul_matches_naive_reference() {
        let mut rng = Rng::new(3);
        for (m, k, n) in [(1, 1, 1), (5, 3, 4), (17, 9, 13), (2, 0, 3), (0, 4, 2)] {
            let a = Tensor::uniform(&[m, k], -1.0, 1.0, &mut rng);
            let b = Tensor::uniform(&[k, n], -1.0, 1.0, &mut rng);
            let c = a.matmul(&b).unwrap();
            assert_eq!(c.shape(), &[m, n]);
            for i in 0..m {
                for j in 0..n {
                    let expected: f32 = (0..k)
                        .map(|p| a.data()[i * k + p] * b.data()[p * n + j])
                        .sum();
                    assert!((c.data()[i * n + j] - expected).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn scalar_broadcasts_against_any_shape() {
        let scaled = Tensor::full(&[], 2.0).mul(&t(&[1., 2., 3.], &[3])).unwrap();
        assert_eq!(scaled.shape(), &[3]);
        assert_eq!(scaled.data(), &[2., 4., 6.]);
    }

    #[test]
    fn rejects_bad_construction() {
        assert!(Tensor::from_vec(vec![1.0; 5], &[2, 3]).is_err());
        assert!(Tensor::zeros(&[3]).matmul(&Tensor::zeros(&[3])).is_err());
        assert!(Tensor::zeros(&[2, 3]).sum_axis(2).is_err());
    }

    #[test]
    fn transpose_twice_is_identity() {
        let a = Tensor::uniform(&[4, 7], -1.0, 1.0, &mut Rng::new(5));
        assert_eq!(a.transpose2d().unwrap().transpose2d().unwrap(), a);
    }

    fn random_shape_pair(rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
        let full: Vec<usize> = (0..1 + rng.below(4)).map(|_| 1 + rng.below(4)).collect();
        let derive = |rng: &mut Rng| {
            let dropped = rng.below(full.len() + 1);
            full[dropped..]
                .iter()
                .map(|&d| if rng.below(2) == 0 { 1 } else { d })
                .collect()
        };
        (derive(rng), derive(rng))
    }

    fn naive_broadcast_add(a: &Tensor, b: &Tensor) -> Tensor {
        let out = crate::broadcast_shape(a.shape(), b.shape()).unwrap();
        let out_strides = crate::contiguous_strides(&out);
        let pick = |t: &Tensor, flat: usize| {
            let offset = out.len() - t.shape().len();
            let strides = crate::contiguous_strides(t.shape());
            let index: usize = (0..t.shape().len())
                .map(|d| {
                    let coord = (flat / out_strides[d + offset]) % out[d + offset];
                    if t.shape()[d] == 1 {
                        0
                    } else {
                        coord * strides[d]
                    }
                })
                .sum();
            t.data()[index]
        };
        let data = (0..out.iter().product())
            .map(|i| pick(a, i) + pick(b, i))
            .collect();
        Tensor::from_vec(data, &out).unwrap()
    }

    #[test]
    fn broadcast_add_matches_naive_indexing_on_random_shapes() {
        let mut rng = Rng::new(11);
        for _ in 0..300 {
            let (sa, sb) = random_shape_pair(&mut rng);
            let a = Tensor::uniform(&sa, -1.0, 1.0, &mut rng);
            let b = Tensor::uniform(&sb, -1.0, 1.0, &mut rng);
            assert_eq!(
                a.add(&b).unwrap(),
                naive_broadcast_add(&a, &b),
                "{sa:?} + {sb:?}"
            );
        }
    }

    #[test]
    fn sum_to_shape_is_the_adjoint_of_broadcasting() {
        let mut rng = Rng::new(12);
        for _ in 0..300 {
            let (small, other) = random_shape_pair(&mut rng);
            let out = crate::broadcast_shape(&small, &other).unwrap();
            let x = Tensor::uniform(&small, -1.0, 1.0, &mut rng);
            let g = Tensor::uniform(&out, -1.0, 1.0, &mut rng);
            let expanded = x.add(&Tensor::zeros(&out)).unwrap();
            let lhs = expanded.mul(&g).unwrap().sum_all();
            let rhs = x.mul(&g.sum_to_shape(&small).unwrap()).unwrap().sum_all();
            assert!(
                (lhs - rhs).abs() < 1e-3,
                "{small:?} -> {out:?}: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn transposed_products_equal_explicit_transposes_bit_for_bit() {
        let mut rng = Rng::new(31);
        for (m, k, n) in [(1, 1, 1), (3, 5, 4), (17, 9, 13), (40, 300, 21)] {
            let a = Tensor::uniform(&[m, k], -1.0, 1.0, &mut rng);
            let b = Tensor::uniform(&[k, n], -1.0, 1.0, &mut rng);
            let expected = a.matmul(&b).unwrap();
            let nt = a.matmul_nt(&b.transpose2d().unwrap()).unwrap();
            let tn = a.transpose2d().unwrap().matmul_tn(&b).unwrap();
            assert_eq!(nt, expected, "nt {m}x{k}x{n}");
            assert_eq!(tn, expected, "tn {m}x{k}x{n}");
        }
    }

    #[test]
    fn transposed_products_reject_mismatched_shapes() {
        let (a, b) = (Tensor::zeros(&[2, 3]), Tensor::zeros(&[4, 5]));
        assert!(a.matmul_nt(&b).is_err());
        assert!(a.matmul_tn(&b).is_err());
        assert!(Tensor::zeros(&[3]).matmul_nt(&a).is_err());
    }

    #[test]
    fn argmax() {
        assert_eq!(
            t(&[1., 3., 2., 9., 0., 1.], &[2, 3]).argmax_rows().unwrap(),
            vec![1, 0]
        );
    }
}
