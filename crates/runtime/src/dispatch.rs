use rt_ir::{Graph, NodeId, Op};
use rt_tensor::kernels;
use rt_tensor::Tensor;

use crate::error::{Result, RuntimeError};
use crate::Feeds;

pub(crate) const MAX_INPUTS: usize = 3;

pub(crate) fn find_feed<'a>(feeds: Feeds<'a>, name: &str) -> Result<&'a Tensor> {
    feeds
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, t)| *t)
        .ok_or_else(|| RuntimeError::MissingInput(name.to_string()))
}

pub(crate) fn check_feeds(graph: &Graph, feeds: Feeds) -> Result<()> {
    for (name, id) in graph.input_names() {
        let tensor = find_feed(feeds, name)?;
        if tensor.shape() != graph.shape(id) {
            return Err(RuntimeError::InputShape {
                name: name.to_string(),
                expected: graph.shape(id).to_vec(),
                actual: tensor.shape().to_vec(),
            });
        }
    }
    Ok(())
}

fn dims2(shape: &[usize]) -> (usize, usize) {
    match shape {
        [a, b] => (*a, *b),
        _ => unreachable!("rank was validated by the IR"),
    }
}

/// Runs one non-input node, reading operand data from `ins` and writing into `out`.
pub(crate) fn execute_node(
    graph: &Graph,
    id: NodeId,
    ins: &[&[f32]; MAX_INPUTS],
    out: &mut [f32],
) -> Result<()> {
    let node = graph.node(id);
    let shape = |k: usize| graph.shape(node.inputs[k]);
    match &node.op {
        Op::Input(_) => unreachable!("inputs are bound by the executor"),
        Op::Full(value) => out.fill(*value),
        Op::Add => kernels::zip_into(
            ins[0],
            shape(0),
            ins[1],
            shape(1),
            out,
            &node.shape,
            |a, b| a + b,
        )?,
        Op::Sub => kernels::zip_into(
            ins[0],
            shape(0),
            ins[1],
            shape(1),
            out,
            &node.shape,
            |a, b| a - b,
        )?,
        Op::Mul => kernels::zip_into(
            ins[0],
            shape(0),
            ins[1],
            shape(1),
            out,
            &node.shape,
            |a, b| a * b,
        )?,
        Op::MatMul => {
            let (m, k) = dims2(shape(0));
            kernels::matmul_into(ins[0], ins[1], m, k, dims2(shape(1)).1, out)?
        }
        Op::Exp => kernels::map_into(ins[0], out, f32::exp)?,
        Op::LogSoftmax => {
            let (rows, cols) = dims2(shape(0));
            kernels::log_softmax_rows_into(ins[0], rows, cols, out)?
        }
        Op::Clamp { min, max } => kernels::map_into(ins[0], out, |x| {
            if x < *min {
                *min
            } else if x > *max {
                *max
            } else {
                x
            }
        })?,
        Op::ClampGrad { min, max } => kernels::zip_into(
            ins[0],
            shape(0),
            ins[1],
            shape(1),
            out,
            &node.shape,
            |x, g| if x >= *min && x <= *max { g } else { 0.0 },
        )?,
        Op::Minimum => kernels::zip_into(
            ins[0],
            shape(0),
            ins[1],
            shape(1),
            out,
            &node.shape,
            f32::min,
        )?,
        Op::MinimumGrad { first } => {
            kernels::minimum_grad_into(ins[0], ins[1], ins[2], *first, out)?
        }
        Op::MatMulNTMasked => {
            let (m, k) = dims2(shape(0));
            let dims = (m, k, dims2(shape(1)).0);
            kernels::matmul_nt_masked_into(ins[0], ins[1], ins[2], dims, out)?
        }
        Op::Linear { relu } => {
            let (m, k) = dims2(shape(0));
            let dims = (m, k, dims2(shape(1)).1);
            kernels::linear_into(ins[0], ins[1], ins[2], *relu, dims, out)?
        }
        Op::MatMulNT => {
            let (m, k) = dims2(shape(0));
            kernels::matmul_nt_into(ins[0], ins[1], m, k, dims2(shape(1)).0, out)?
        }
        Op::MatMulTN => {
            let (k, m) = dims2(shape(0));
            kernels::matmul_tn_into(ins[0], ins[1], m, k, dims2(shape(1)).1, out)?
        }
        Op::Transpose => {
            let (rows, cols) = dims2(shape(0));
            kernels::transpose_into(ins[0], rows, cols, out)?
        }
        Op::Relu => kernels::map_into(ins[0], out, |x| x.max(0.0))?,
        Op::Scale(s) => kernels::map_into(ins[0], out, |x| x * s)?,
        Op::SumAll | Op::SumToShape => kernels::reduce_into(ins[0], shape(0), out, &node.shape)?,
        Op::BroadcastTo => kernels::broadcast_into(ins[0], shape(0), out, &node.shape)?,
        Op::ReluGrad => kernels::zip_into(
            ins[0],
            shape(0),
            ins[1],
            shape(1),
            out,
            &node.shape,
            |x, g| {
                if x > 0.0 {
                    g
                } else {
                    0.0
                }
            },
        )?,
        Op::CrossEntropy => {
            let (rows, cols) = dims2(shape(0));
            out[0] = kernels::cross_entropy(ins[0], rows, cols, ins[1])?;
        }
        Op::CrossEntropyGrad => {
            let (rows, cols) = dims2(shape(0));
            kernels::cross_entropy_grad_into(ins[0], rows, cols, ins[1], ins[2][0], out)?
        }
    }
    Ok(())
}
