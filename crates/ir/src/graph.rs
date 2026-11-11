use rt_tensor::{broadcast_shape, MAX_RANK};

use crate::error::{IrError, Result};
use crate::op::Op;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<NodeId>,
    pub shape: Vec<usize>,
}

/// Nodes are stored in topological order: a node only reads earlier nodes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    nodes: Vec<Node>,
    outputs: Vec<NodeId>,
}

fn shape_err<T>(message: String) -> Result<T> {
    Err(IrError::Shape(message))
}

fn can_expand(small: &[usize], big: &[usize]) -> bool {
    let offset = big.len().wrapping_sub(small.len());
    small.len() <= big.len()
        && small
            .iter()
            .enumerate()
            .all(|(i, &d)| d == 1 || d == big[i + offset])
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0]
    }

    pub fn shape(&self, id: NodeId) -> &[usize] {
        &self.nodes[id.0].shape
    }

    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    pub fn set_outputs(&mut self, outputs: Vec<NodeId>) -> Result<()> {
        if let Some(bad) = outputs.iter().find(|o| o.0 >= self.nodes.len()) {
            return Err(IrError::Invalid(format!("output {} is not a node", bad.0)));
        }
        self.outputs = outputs;
        Ok(())
    }

    pub fn input_names(&self) -> impl Iterator<Item = (&str, NodeId)> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| match &n.op {
                Op::Input(name) => Some((name.as_str(), NodeId(i))),
                _ => None,
            })
    }

    /// Appends a node after validating arity and inferring its shape.
    /// `declared` is required for ops that carry their own shape and must
    /// match the inferred shape for all others.
    pub fn add_node(
        &mut self,
        op: Op,
        inputs: &[NodeId],
        declared: Option<&[usize]>,
    ) -> Result<NodeId> {
        if inputs.len() != op.arity() || inputs.iter().any(|i| i.0 >= self.nodes.len()) {
            return Err(IrError::Invalid(format!(
                "{op:?} needs {} existing inputs",
                op.arity()
            )));
        }
        if let Op::Input(name) = &op {
            if self.input_names().any(|(existing, _)| existing == name) {
                return Err(IrError::Invalid(format!("duplicate input {name}")));
            }
        }
        let shape = self.infer_shape(&op, inputs, declared)?;
        self.nodes.push(Node {
            op,
            inputs: inputs.to_vec(),
            shape,
        });
        Ok(NodeId(self.nodes.len() - 1))
    }

    fn infer_shape(
        &self,
        op: &Op,
        inputs: &[NodeId],
        declared: Option<&[usize]>,
    ) -> Result<Vec<usize>> {
        let s = |i: usize| self.shape(inputs[i]);
        let inferred = match op {
            Op::Input(_) | Op::Full(_) | Op::SumToShape | Op::BroadcastTo => {
                let shape = declared
                    .ok_or_else(|| IrError::Invalid(format!("{op:?} needs a declared shape")))?;
                if shape.len() > MAX_RANK {
                    return shape_err(format!("rank of {shape:?} exceeds {MAX_RANK}"));
                }
                match op {
                    Op::SumToShape if !can_expand(shape, s(0)) => {
                        return shape_err(format!("cannot sum {:?} to {shape:?}", s(0)))
                    }
                    Op::BroadcastTo if !can_expand(s(0), shape) => {
                        return shape_err(format!("cannot broadcast {:?} to {shape:?}", s(0)))
                    }
                    _ => {}
                }
                return Ok(shape.to_vec());
            }
            Op::Add | Op::Sub | Op::Mul => {
                broadcast_shape(s(0), s(1)).or_else(|e| shape_err(e.to_string()))?
            }
            Op::MatMul => match (s(0), s(1)) {
                ([m, k], [k2, n]) if k == k2 => vec![*m, *n],
                (a, b) => return shape_err(format!("matmul {a:?} x {b:?}")),
            },
            Op::Linear { .. } => match (s(0), s(1), s(2)) {
                ([m, k], [k2, n], [n2]) if k == k2 && n == n2 => vec![*m, *n],
                (x, w, b) => return shape_err(format!("linear {x:?} * {w:?} + {b:?}")),
            },
            Op::MatMulNTMasked => match (s(0), s(1), s(2)) {
                ([m, k], [n, k2], [m2, n2]) if k == k2 && m == m2 && n == n2 => vec![*m, *n],
                (a, b, mask) => {
                    return shape_err(format!(
                        "matmul_nt_masked {a:?} x {b:?}^T masked by {mask:?}"
                    ))
                }
            },
            Op::MatMulNT => match (s(0), s(1)) {
                ([m, k], [n, k2]) if k == k2 => vec![*m, *n],
                (a, b) => return shape_err(format!("matmul_nt {a:?} x {b:?}^T")),
            },
            Op::MatMulTN => match (s(0), s(1)) {
                ([k, m], [k2, n]) if k == k2 => vec![*m, *n],
                (a, b) => return shape_err(format!("matmul_tn {a:?}^T x {b:?}")),
            },
            Op::Transpose => match s(0) {
                [r, c] => vec![*c, *r],
                other => return shape_err(format!("transpose needs rank 2, got {other:?}")),
            },
            Op::Relu | Op::Scale(_) | Op::Exp | Op::Clamp { .. } => s(0).to_vec(),
            Op::LogSoftmax => match s(0) {
                [_, _] => s(0).to_vec(),
                other => return shape_err(format!("log_softmax needs rank 2, got {other:?}")),
            },
            Op::MinimumGrad { .. } if s(0) == s(1) && s(1) == s(2) => s(0).to_vec(),
            Op::MinimumGrad { .. } => {
                return shape_err(format!("minimum grad {:?} {:?} {:?}", s(0), s(1), s(2)))
            }
            Op::SumAll => vec![],
            Op::ReluGrad | Op::ClampGrad { .. } | Op::Minimum if s(0) == s(1) => s(0).to_vec(),
            Op::ReluGrad | Op::ClampGrad { .. } | Op::Minimum => {
                return shape_err(format!(
                    "{op:?} needs equal shapes, got {:?} and {:?}",
                    s(0),
                    s(1)
                ))
            }
            Op::CrossEntropy | Op::CrossEntropyGrad => match (s(0), s(1)) {
                ([n, c], [labels]) if n == labels => {
                    if matches!(op, Op::CrossEntropyGrad) {
                        if !s(2).is_empty() {
                            return shape_err(format!(
                                "upstream gradient must be scalar, got {:?}",
                                s(2)
                            ));
                        }
                        vec![*n, *c]
                    } else {
                        vec![]
                    }
                }
                (a, b) => {
                    return shape_err(format!("cross-entropy logits {a:?} with labels {b:?}"))
                }
            },
        };
        match declared {
            Some(d) if d != inferred.as_slice() => {
                shape_err(format!("declared {d:?} but inferred {inferred:?}"))
            }
            _ => Ok(inferred),
        }
    }

    pub fn input(&mut self, name: &str, shape: &[usize]) -> Result<NodeId> {
        self.add_node(Op::Input(name.to_string()), &[], Some(shape))
    }

    pub fn full(&mut self, value: f32, shape: &[usize]) -> Result<NodeId> {
        self.add_node(Op::Full(value), &[], Some(shape))
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
        self.add_node(Op::Add, &[a, b], None)
    }

    pub fn sub(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
        self.add_node(Op::Sub, &[a, b], None)
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
        self.add_node(Op::Mul, &[a, b], None)
    }

    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
        self.add_node(Op::MatMul, &[a, b], None)
    }

    pub fn transpose(&mut self, a: NodeId) -> Result<NodeId> {
        self.add_node(Op::Transpose, &[a], None)
    }

    pub fn linear(&mut self, x: NodeId, w: NodeId, bias: NodeId, relu: bool) -> Result<NodeId> {
        self.add_node(Op::Linear { relu }, &[x, w, bias], None)
    }

    pub fn matmul_nt_masked(&mut self, a: NodeId, b: NodeId, mask: NodeId) -> Result<NodeId> {
        self.add_node(Op::MatMulNTMasked, &[a, b, mask], None)
    }

    pub fn exp(&mut self, a: NodeId) -> Result<NodeId> {
        self.add_node(Op::Exp, &[a], None)
    }

    pub fn log_softmax(&mut self, a: NodeId) -> Result<NodeId> {
        self.add_node(Op::LogSoftmax, &[a], None)
    }

    pub fn clamp(&mut self, a: NodeId, min: f32, max: f32) -> Result<NodeId> {
        self.add_node(Op::Clamp { min, max }, &[a], None)
    }

    pub fn clamp_grad(&mut self, x: NodeId, grad: NodeId, min: f32, max: f32) -> Result<NodeId> {
        self.add_node(Op::ClampGrad { min, max }, &[x, grad], None)
    }

    pub fn minimum(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
        self.add_node(Op::Minimum, &[a, b], None)
    }

    pub fn minimum_grad(
        &mut self,
        a: NodeId,
        b: NodeId,
        grad: NodeId,
        first: bool,
    ) -> Result<NodeId> {
        self.add_node(Op::MinimumGrad { first }, &[a, b, grad], None)
    }

    pub fn matmul_nt(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
        self.add_node(Op::MatMulNT, &[a, b], None)
    }

    pub fn matmul_tn(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
        self.add_node(Op::MatMulTN, &[a, b], None)
    }

    pub fn relu(&mut self, a: NodeId) -> Result<NodeId> {
        self.add_node(Op::Relu, &[a], None)
    }

    pub fn scale(&mut self, a: NodeId, s: f32) -> Result<NodeId> {
        self.add_node(Op::Scale(s), &[a], None)
    }

    pub fn sum_all(&mut self, a: NodeId) -> Result<NodeId> {
        self.add_node(Op::SumAll, &[a], None)
    }

    pub fn mean(&mut self, a: NodeId) -> Result<NodeId> {
        let n = self.shape(a).iter().product::<usize>().max(1);
        let sum = self.sum_all(a)?;
        self.scale(sum, 1.0 / n as f32)
    }

    pub fn sum_to_shape(&mut self, a: NodeId, shape: &[usize]) -> Result<NodeId> {
        self.add_node(Op::SumToShape, &[a], Some(shape))
    }

    pub fn broadcast_to(&mut self, a: NodeId, shape: &[usize]) -> Result<NodeId> {
        self.add_node(Op::BroadcastTo, &[a], Some(shape))
    }

    pub fn relu_grad(&mut self, x: NodeId, grad: NodeId) -> Result<NodeId> {
        self.add_node(Op::ReluGrad, &[x, grad], None)
    }

    pub fn cross_entropy(&mut self, logits: NodeId, labels: NodeId) -> Result<NodeId> {
        self.add_node(Op::CrossEntropy, &[logits, labels], None)
    }

    pub fn cross_entropy_grad(
        &mut self,
        logits: NodeId,
        labels: NodeId,
        upstream: NodeId,
    ) -> Result<NodeId> {
        self.add_node(Op::CrossEntropyGrad, &[logits, labels, upstream], None)
    }
}
