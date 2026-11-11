//! Which shader runs a node, and the parameter block it needs.

use rt_ir::{Graph, Node, NodeId, Op};
use rt_tensor::{broadcast_strides, contiguous_strides};

use crate::error::{GpuError, Result};

pub(crate) const WORDS: usize = 64;
pub(crate) type Words = [u32; WORDS];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Kernel {
    Binary,
    Unary,
    Ternary,
    Matmul,
    Reduce,
    Rows,
    AdamW,
}

impl Kernel {
    pub fn name(self) -> &'static str {
        match self {
            Self::Binary => "binary",
            Self::Unary => "unary",
            Self::Ternary => "ternary",
            Self::Matmul => "matmul",
            Self::Reduce => "reduce",
            Self::Rows => "rows",
            Self::AdamW => "adamw",
        }
    }

    pub fn source(self) -> String {
        let body = match self {
            Self::Binary => include_str!("shaders/binary.wgsl"),
            Self::Unary => include_str!("shaders/unary.wgsl"),
            Self::Ternary => include_str!("shaders/ternary.wgsl"),
            Self::Matmul => include_str!("shaders/matmul.wgsl"),
            Self::Reduce => include_str!("shaders/reduce.wgsl"),
            Self::Rows => include_str!("shaders/rows.wgsl"),
            Self::AdamW => include_str!("shaders/adamw.wgsl"),
        };
        format!("{}\n{body}", include_str!("shaders/common.wgsl"))
    }
}

/// Everything needed to launch one node: shader, parameters, the nodes bound to each buffer
/// slot after the parameter block (`None` binds a placeholder), and the dispatch size.
pub(crate) struct Launch {
    pub kernel: Kernel,
    pub words: Words,
    pub bindings: Vec<Option<NodeId>>,
    pub groups: [u32; 3],
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn groups_for(items: usize, per_group: usize) -> [u32; 3] {
    [items.div_ceil(per_group).max(1) as u32, 1, 1]
}

fn set_shape(words: &mut Words, offset: usize, values: &[usize]) {
    for (i, &v) in values.iter().enumerate() {
        words[offset + i] = v as u32;
    }
}

fn binary(
    op: u32,
    constants: (f32, f32),
    shapes: [&[usize]; 2],
    out: &[usize],
) -> Result<(Words, usize)> {
    let mut words = [0; WORDS];
    let (a, b) = (
        broadcast_strides(shapes[0], out)?,
        broadcast_strides(shapes[1], out)?,
    );
    (words[0], words[1], words[2]) = (numel(out) as u32, out.len() as u32, op);
    (words[3], words[4]) = (constants.0.to_bits(), constants.1.to_bits());
    set_shape(&mut words, 8, out);
    set_shape(&mut words, 16, &a[..out.len()]);
    set_shape(&mut words, 24, &b[..out.len()]);
    Ok((words, numel(out)))
}

fn unary(op: u32, f0: f32, f1: f32, n: usize) -> Words {
    let mut words = [0; WORDS];
    (words[0], words[1], words[2], words[3]) = (n as u32, op, f0.to_bits(), f1.to_bits());
    words
}

fn reduce(src: &[usize], out: &[usize]) -> Result<(Words, usize)> {
    let mut words = [0; WORDS];
    let strides = contiguous_strides(src);
    let (mut kept, mut reduced) = (Vec::new(), Vec::new());
    for (d, &size) in src.iter().enumerate() {
        let target = if d + out.len() >= src.len() {
            out[d + out.len() - src.len()]
        } else {
            1
        };
        if target == size {
            if size > 1 {
                kept.push((size, strides[d]));
            }
        } else {
            reduced.push((size, strides[d]));
        }
    }
    let total: usize = reduced.iter().map(|r| r.0).product();
    (words[0], words[1], words[2], words[3]) = (
        numel(out) as u32,
        kept.len() as u32,
        reduced.len() as u32,
        total as u32,
    );
    for (i, (size, stride)) in kept.iter().enumerate() {
        (words[8 + i], words[16 + i]) = (*size as u32, *stride as u32);
    }
    for (i, (size, stride)) in reduced.iter().enumerate() {
        (words[24 + i], words[32 + i]) = (*size as u32, *stride as u32);
    }
    Ok((words, numel(out)))
}

fn matmul(
    m: usize,
    n: usize,
    k: usize,
    a: (usize, usize),
    b: (usize, usize),
    flags: u32,
) -> Launch {
    let mut words = [0; WORDS];
    let values = [m, n, k, a.0, a.1, b.0, b.1, flags as usize];
    for (i, v) in values.iter().enumerate() {
        words[i] = *v as u32;
    }
    Launch {
        kernel: Kernel::Matmul,
        words,
        bindings: Vec::new(),
        groups: [
            n.div_ceil(64).max(1) as u32,
            m.div_ceil(64).max(1) as u32,
            1,
        ],
    }
}

fn rows(rows: usize, cols: usize, mode: u32, groups: [u32; 3]) -> Launch {
    let mut words = [0; WORDS];
    (words[0], words[1], words[2]) = (rows as u32, cols as u32, mode);
    Launch {
        kernel: Kernel::Rows,
        words,
        bindings: Vec::new(),
        groups,
    }
}

fn dims(shape: &[usize]) -> Result<(usize, usize)> {
    match shape {
        [a, b] => Ok((*a, *b)),
        _ => Err(GpuError::Device(format!(
            "expected a matrix, got shape {shape:?}"
        ))),
    }
}

/// One kernel launch of a compiled graph, in execution order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelLaunch {
    pub node: usize,
    pub kernel: &'static str,
    pub groups: [u32; 3],
}

/// The sequence of kernels the GPU executor would run for an already optimised graph.
pub fn schedule(graph: &Graph) -> Result<Vec<KernelLaunch>> {
    let mut launches = Vec::new();
    for (index, node) in graph.nodes().iter().enumerate() {
        if let Some(plan) = launch(graph, node)? {
            launches.push(KernelLaunch {
                node: index,
                kernel: plan.kernel.name(),
                groups: plan.groups,
            });
        }
    }
    Ok(launches)
}
/// Plans the launch of `node`; `None` for nodes that need no work (inputs).
pub(crate) fn launch(graph: &Graph, node: &Node) -> Result<Option<Launch>> {
    let shape = |i: usize| graph.shape(node.inputs[i]);
    let id = |i: usize| Some(node.inputs[i]);
    let out = node.shape.as_slice();
    let elementwise = |words: Words, n: usize, bindings: Vec<Option<NodeId>>| Launch {
        kernel: Kernel::Binary,
        words,
        bindings,
        groups: groups_for(n, 256),
    };
    let binary_launch = |op: u32, constants: (f32, f32)| -> Result<Launch> {
        let (words, n) = binary(op, constants, [shape(0), shape(1)], out)?;
        Ok(elementwise(words, n, vec![id(0), id(1)]))
    };
    let unary_launch = |op: u32, f0: f32, f1: f32| Launch {
        kernel: Kernel::Unary,
        words: unary(op, f0, f1, numel(out)),
        bindings: vec![id(0)],
        groups: groups_for(numel(out), 256),
    };
    Ok(Some(match &node.op {
        Op::Input(_) => return Ok(None),
        Op::Full(value) => Launch {
            kernel: Kernel::Unary,
            words: unary(4, *value, 0.0, numel(out)),
            bindings: vec![None],
            groups: groups_for(numel(out), 256),
        },
        Op::Add => binary_launch(0, (0.0, 0.0))?,
        Op::Sub => binary_launch(1, (0.0, 0.0))?,
        Op::Mul => binary_launch(2, (0.0, 0.0))?,
        Op::Minimum => binary_launch(3, (0.0, 0.0))?,
        Op::ReluGrad => binary_launch(4, (0.0, 0.0))?,
        Op::ClampGrad { min, max } => binary_launch(5, (*min, *max))?,
        Op::Relu => unary_launch(0, 0.0, 0.0),
        Op::Scale(s) => unary_launch(1, *s, 0.0),
        Op::Exp => unary_launch(2, 0.0, 0.0),
        Op::Clamp { min, max } => unary_launch(3, *min, *max),
        Op::BroadcastTo => {
            let (words, n) = binary(6, (0.0, 0.0), [shape(0), shape(0)], out)?;
            elementwise(words, n, vec![id(0), id(0)])
        }
        Op::Transpose => {
            let (rows, cols) = dims(shape(0))?;
            let mut words = [0; WORDS];
            (words[0], words[1], words[2]) = ((rows * cols) as u32, 2, 6);
            set_shape(&mut words, 8, &[cols, rows]);
            set_shape(&mut words, 16, &[1, cols]);
            set_shape(&mut words, 24, &[1, cols]);
            elementwise(words, rows * cols, vec![id(0), id(0)])
        }
        Op::SumAll | Op::SumToShape => {
            let (words, n) = reduce(shape(0), out)?;
            Launch {
                kernel: Kernel::Reduce,
                words,
                bindings: vec![id(0)],
                groups: groups_for(n, 64),
            }
        }
        Op::MatMul => {
            let ((m, k), (_, n)) = (dims(shape(0))?, dims(shape(1))?);
            let mut l = matmul(m, n, k, (k, 1), (n, 1), 0);
            l.bindings = vec![id(0), id(1), None, None];
            l
        }
        Op::MatMulNT => {
            let ((m, k), (n, _)) = (dims(shape(0))?, dims(shape(1))?);
            let mut l = matmul(m, n, k, (k, 1), (1, k), 0);
            l.bindings = vec![id(0), id(1), None, None];
            l
        }
        Op::MatMulTN => {
            let ((k, m), (_, n)) = (dims(shape(0))?, dims(shape(1))?);
            let mut l = matmul(m, n, k, (1, m), (n, 1), 0);
            l.bindings = vec![id(0), id(1), None, None];
            l
        }
        Op::Linear { relu } => {
            let ((m, k), (_, n)) = (dims(shape(0))?, dims(shape(1))?);
            let mut l = matmul(m, n, k, (k, 1), (n, 1), 1 | (u32::from(*relu) << 1));
            l.bindings = vec![id(0), id(1), id(2), None];
            l
        }
        Op::MatMulNTMasked => {
            let ((m, k), (n, _)) = (dims(shape(0))?, dims(shape(1))?);
            let mut l = matmul(m, n, k, (k, 1), (1, k), 4);
            l.bindings = vec![id(0), id(1), None, id(2)];
            l
        }
        Op::CrossEntropy => {
            let (r, c) = dims(shape(0))?;
            let mut l = rows(r, c, 0, [1, 1, 1]);
            l.bindings = vec![id(0), id(1), None];
            l
        }
        Op::CrossEntropyGrad => {
            let (r, c) = dims(shape(0))?;
            let mut l = rows(r, c, 1, groups_for(r, 64));
            l.bindings = vec![id(0), id(1), id(2)];
            l
        }
        Op::LogSoftmax => {
            let (r, c) = dims(shape(0))?;
            let mut l = rows(r, c, 2, groups_for(r, 64));
            l.bindings = vec![id(0), None, None];
            l
        }
        Op::MinimumGrad { first } => {
            let mut words = [0; WORDS];
            (words[0], words[1]) = (numel(out) as u32, u32::from(*first));
            Launch {
                kernel: Kernel::Ternary,
                words,
                bindings: vec![id(0), id(1), id(2)],
                groups: groups_for(numel(out), 256),
            }
        }
    }))
}
