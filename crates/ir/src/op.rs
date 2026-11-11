/// Operations over f32 tensors. `Input`, `Full`, `SumToShape` and `BroadcastTo`
/// take their output shape from the node; all other shapes are inferred.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Input(String),
    Full(f32),
    Add,
    Sub,
    Mul,
    MatMul,
    /// `a * b^T`, with `b` stored as `n x k`.
    MatMulNT,
    /// `a^T * b`, with `a` stored as `k x m`.
    MatMulTN,
    Transpose,
    Relu,
    Scale(f32),
    SumAll,
    SumToShape,
    BroadcastTo,
    ReluGrad,
    CrossEntropy,
    CrossEntropyGrad,
    /// `x * w + bias`, optionally clamped at zero: the fused form of `MatMul`, `Add`, `Relu`.
    Linear {
        relu: bool,
    },
    /// `(a * b^T)` zeroed where `mask <= 0`: the fused form of `MatMulNT` and `ReluGrad`.
    MatMulNTMasked,
    Exp,
    /// Row-wise log-softmax of a matrix.
    LogSoftmax,
    Clamp {
        min: f32,
        max: f32,
    },
    /// `grad` where `min <= x <= max`, else zero.
    ClampGrad {
        min: f32,
        max: f32,
    },
    /// Element-wise minimum of two tensors of equal shape.
    Minimum,
    /// Gradient of `Minimum` for one operand; ties split the gradient evenly.
    MinimumGrad {
        first: bool,
    },
}

impl Op {
    pub fn arity(&self) -> usize {
        match self {
            Self::Input(_) | Self::Full(_) => 0,
            Self::Transpose
            | Self::Relu
            | Self::Scale(_)
            | Self::SumAll
            | Self::SumToShape
            | Self::BroadcastTo
            | Self::Exp
            | Self::LogSoftmax
            | Self::Clamp { .. } => 1,
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::MatMul
            | Self::MatMulNT
            | Self::MatMulTN
            | Self::ReluGrad
            | Self::CrossEntropy
            | Self::ClampGrad { .. }
            | Self::Minimum => 2,
            Self::CrossEntropyGrad
            | Self::Linear { .. }
            | Self::MatMulNTMasked
            | Self::MinimumGrad { .. } => 3,
        }
    }

    pub fn has_declared_shape(&self) -> bool {
        matches!(
            self,
            Self::Input(_) | Self::Full(_) | Self::SumToShape | Self::BroadcastTo
        )
    }
}
