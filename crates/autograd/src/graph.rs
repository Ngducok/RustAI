use rt_tensor::{Result, Tensor, TensorError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Var(usize);

enum Op {
    Leaf,
    Add(Var, Var),
    Sub(Var, Var),
    Mul(Var, Var),
    MatMul(Var, Var),
    Relu(Var),
    SumAll(Var),
    Scale(Var, f32),
    CrossEntropy { logits: Var, targets: Vec<usize> },
}

struct Node {
    value: Tensor,
    op: Op,
    requires_grad: bool,
}

#[derive(Default)]
pub struct Graph {
    nodes: Vec<Node>,
}

pub struct Grads(Vec<Option<Tensor>>);

impl Grads {
    pub fn get(&self, var: Var) -> Option<&Tensor> {
        self.0[var.0].as_ref()
    }

    pub fn take_all(&mut self, vars: &[Var]) -> Result<Vec<Tensor>> {
        vars.iter()
            .map(|v| {
                self.0[v.0]
                    .take()
                    .ok_or_else(|| TensorError::InvalidArgument("variable has no gradient".into()))
            })
            .collect()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn leaf(&mut self, value: Tensor) -> Var {
        self.push(value, Op::Leaf, true)
    }

    pub fn constant(&mut self, value: Tensor) -> Var {
        self.push(value, Op::Leaf, false)
    }

    pub fn value(&self, var: Var) -> &Tensor {
        &self.nodes[var.0].value
    }

    pub fn add(&mut self, a: Var, b: Var) -> Result<Var> {
        let value = self.value(a).add(self.value(b))?;
        Ok(self.push_with(value, Op::Add(a, b), &[a, b]))
    }

    pub fn sub(&mut self, a: Var, b: Var) -> Result<Var> {
        let value = self.value(a).sub(self.value(b))?;
        Ok(self.push_with(value, Op::Sub(a, b), &[a, b]))
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Result<Var> {
        let value = self.value(a).mul(self.value(b))?;
        Ok(self.push_with(value, Op::Mul(a, b), &[a, b]))
    }

    pub fn matmul(&mut self, a: Var, b: Var) -> Result<Var> {
        let value = self.value(a).matmul(self.value(b))?;
        Ok(self.push_with(value, Op::MatMul(a, b), &[a, b]))
    }

    pub fn relu(&mut self, a: Var) -> Var {
        let value = self.value(a).relu();
        self.push_with(value, Op::Relu(a), &[a])
    }

    pub fn scale(&mut self, a: Var, s: f32) -> Var {
        let value = self.value(a).scale(s);
        self.push_with(value, Op::Scale(a, s), &[a])
    }

    pub fn sum_all(&mut self, a: Var) -> Var {
        let value = Tensor::full(&[], self.value(a).sum_all());
        self.push_with(value, Op::SumAll(a), &[a])
    }

    pub fn mean(&mut self, a: Var) -> Var {
        let n = self.value(a).numel().max(1);
        let sum = self.sum_all(a);
        self.scale(sum, 1.0 / n as f32)
    }

    pub fn cross_entropy(&mut self, logits: Var, targets: &[usize]) -> Result<Var> {
        let log_probs = self.value(logits).log_softmax_rows()?;
        let classes = log_probs.shape()[1];
        if targets.len() != log_probs.shape()[0] || targets.iter().any(|&t| t >= classes) {
            return Err(TensorError::InvalidArgument(
                "targets must hold one valid class index per row".into(),
            ));
        }
        let total: f32 = targets
            .iter()
            .enumerate()
            .map(|(row, &t)| log_probs.data()[row * classes + t])
            .sum();
        let value = Tensor::full(&[], -total / targets.len().max(1) as f32);
        let op = Op::CrossEntropy {
            logits,
            targets: targets.to_vec(),
        };
        Ok(self.push_with(value, op, &[logits]))
    }

    pub fn backward(&self, loss: Var) -> Result<Grads> {
        if self.value(loss).numel() != 1 {
            return Err(TensorError::InvalidArgument("loss must be a scalar".into()));
        }
        let mut grads: Vec<Option<Tensor>> = vec![None; self.nodes.len()];
        grads[loss.0] = Some(Tensor::ones(self.value(loss).shape()));
        for i in (0..=loss.0).rev() {
            let Some(grad) = grads[i].take() else {
                continue;
            };
            for (parent, local) in self.local_grads(i, &grad)? {
                if !self.nodes[parent.0].requires_grad {
                    continue;
                }
                grads[parent.0] = Some(match grads[parent.0].take() {
                    Some(acc) => acc.add(&local)?,
                    None => local,
                });
            }
            grads[i] = Some(grad);
        }
        Ok(Grads(grads))
    }

    fn push(&mut self, value: Tensor, op: Op, requires_grad: bool) -> Var {
        self.nodes.push(Node {
            value,
            op,
            requires_grad,
        });
        Var(self.nodes.len() - 1)
    }

    fn push_with(&mut self, value: Tensor, op: Op, parents: &[Var]) -> Var {
        let requires_grad = parents.iter().any(|p| self.nodes[p.0].requires_grad);
        self.push(value, op, requires_grad)
    }

    fn local_grads(&self, index: usize, grad: &Tensor) -> Result<Vec<(Var, Tensor)>> {
        let shape = |v: Var| self.value(v).shape();
        Ok(match &self.nodes[index].op {
            Op::Leaf => vec![],
            Op::Add(a, b) => vec![
                (*a, grad.sum_to_shape(shape(*a))?),
                (*b, grad.sum_to_shape(shape(*b))?),
            ],
            Op::Sub(a, b) => vec![
                (*a, grad.sum_to_shape(shape(*a))?),
                (*b, grad.scale(-1.0).sum_to_shape(shape(*b))?),
            ],
            Op::Mul(a, b) => vec![
                (*a, grad.mul(self.value(*b))?.sum_to_shape(shape(*a))?),
                (*b, grad.mul(self.value(*a))?.sum_to_shape(shape(*b))?),
            ],
            Op::MatMul(a, b) => vec![
                (*a, grad.matmul_nt(self.value(*b))?),
                (*b, self.value(*a).matmul_tn(grad)?),
            ],
            Op::Relu(a) => {
                let masked = self
                    .value(*a)
                    .zip_with(grad, |x, g| if x > 0.0 { g } else { 0.0 })?;
                vec![(*a, masked)]
            }
            Op::SumAll(a) => vec![(*a, Tensor::full(shape(*a), grad.data()[0]))],
            Op::Scale(a, s) => vec![(*a, grad.scale(*s))],
            Op::CrossEntropy { logits, targets } => {
                let mut d = self.value(*logits).log_softmax_rows()?.map(f32::exp);
                let classes = d.shape()[1];
                for (row, &t) in targets.iter().enumerate() {
                    d.data_mut()[row * classes + t] -= 1.0;
                }
                vec![(
                    *logits,
                    d.scale(grad.data()[0] / targets.len().max(1) as f32),
                )]
            }
        })
    }
}
