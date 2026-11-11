use crate::error::{IrError, Result};
use crate::graph::{Graph, Node, NodeId};
use crate::op::Op;

/// Appends reverse-mode gradient nodes for `loss` and returns the gradient of each `wrt` node.
pub fn gradients(g: &mut Graph, loss: NodeId, wrt: &[NodeId]) -> Result<Vec<NodeId>> {
    if !g.shape(loss).is_empty() {
        return Err(IrError::Invalid("loss must be a scalar".into()));
    }
    let forward_len = loss.0 + 1;
    let needs = depends_on(g, forward_len, wrt);
    if !needs[loss.0] {
        return Err(IrError::Invalid(
            "loss does not depend on the requested nodes".into(),
        ));
    }

    let mut grads: Vec<Option<NodeId>> = vec![None; forward_len];
    grads[loss.0] = Some(g.full(1.0, &[])?);
    for index in (0..forward_len).rev() {
        let Some(grad) = grads[index] else { continue };
        let node = g.node(NodeId(index)).clone();
        for (input, contribution) in local_gradients(g, &node, NodeId(index), grad)? {
            if needs[input.0] {
                grads[input.0] = Some(match grads[input.0] {
                    Some(sum) => g.add(sum, contribution)?,
                    None => contribution,
                });
            }
        }
    }

    wrt.iter()
        .map(|w| {
            grads[w.0].ok_or_else(|| IrError::Invalid(format!("node {} gets no gradient", w.0)))
        })
        .collect()
}

fn depends_on(g: &Graph, len: usize, wrt: &[NodeId]) -> Vec<bool> {
    let mut needs = vec![false; len];
    wrt.iter()
        .filter(|w| w.0 < len)
        .for_each(|w| needs[w.0] = true);
    for i in 0..len {
        needs[i] |= g.nodes()[i].inputs.iter().any(|input| needs[input.0]);
    }
    needs
}

fn reduce_to(g: &mut Graph, grad: NodeId, target: NodeId) -> Result<NodeId> {
    if g.shape(grad) == g.shape(target) {
        Ok(grad)
    } else {
        let shape = g.shape(target).to_vec();
        g.sum_to_shape(grad, &shape)
    }
}

fn local_gradients(
    g: &mut Graph,
    node: &Node,
    id: NodeId,
    grad: NodeId,
) -> Result<Vec<(NodeId, NodeId)>> {
    let inputs = &node.inputs;
    Ok(match &node.op {
        Op::Input(_) | Op::Full(_) => vec![],
        Op::Add => vec![
            (inputs[0], reduce_to(g, grad, inputs[0])?),
            (inputs[1], reduce_to(g, grad, inputs[1])?),
        ],
        Op::Sub => {
            let negated = g.scale(grad, -1.0)?;
            vec![
                (inputs[0], reduce_to(g, grad, inputs[0])?),
                (inputs[1], reduce_to(g, negated, inputs[1])?),
            ]
        }
        Op::Mul => {
            let (a, b) = (inputs[0], inputs[1]);
            let (for_a, for_b) = (g.mul(grad, b)?, g.mul(grad, a)?);
            vec![(a, reduce_to(g, for_a, a)?), (b, reduce_to(g, for_b, b)?)]
        }
        Op::MatMul => {
            let (a, b) = (inputs[0], inputs[1]);
            vec![(a, g.matmul_nt(grad, b)?), (b, g.matmul_tn(a, grad)?)]
        }
        Op::MatMulNT => {
            let (a, b) = (inputs[0], inputs[1]);
            vec![(a, g.matmul(grad, b)?), (b, g.matmul_tn(grad, a)?)]
        }
        Op::MatMulTN => {
            let (a, b) = (inputs[0], inputs[1]);
            vec![(a, g.matmul_nt(b, grad)?), (b, g.matmul(a, grad)?)]
        }
        Op::Transpose => vec![(inputs[0], g.transpose(grad)?)],
        Op::Relu => vec![(inputs[0], g.relu_grad(inputs[0], grad)?)],
        Op::Scale(s) => vec![(inputs[0], g.scale(grad, *s)?)],
        Op::SumAll | Op::SumToShape => {
            let shape = g.shape(inputs[0]).to_vec();
            vec![(inputs[0], g.broadcast_to(grad, &shape)?)]
        }
        Op::BroadcastTo => {
            let shape = g.shape(inputs[0]).to_vec();
            vec![(inputs[0], g.sum_to_shape(grad, &shape)?)]
        }
        Op::CrossEntropy => vec![(inputs[0], g.cross_entropy_grad(inputs[0], inputs[1], grad)?)],
        Op::Exp => vec![(inputs[0], g.mul(grad, id)?)],
        Op::LogSoftmax => {
            let rows = g.shape(id)[0];
            let row_sums = g.sum_to_shape(grad, &[rows, 1])?;
            let softmax = g.exp(id)?;
            let share = g.mul(softmax, row_sums)?;
            vec![(inputs[0], g.sub(grad, share)?)]
        }
        Op::Clamp { min, max } => vec![(inputs[0], g.clamp_grad(inputs[0], grad, *min, *max)?)],
        Op::Minimum => {
            let (a, b) = (inputs[0], inputs[1]);
            vec![
                (a, g.minimum_grad(a, b, grad, true)?),
                (b, g.minimum_grad(a, b, grad, false)?),
            ]
        }
        Op::ReluGrad
        | Op::CrossEntropyGrad
        | Op::Linear { .. }
        | Op::MatMulNTMasked
        | Op::ClampGrad { .. }
        | Op::MinimumGrad { .. } => {
            return Err(IrError::Invalid(format!(
                "{:?} is not differentiable here",
                node.op
            )))
        }
    })
}
