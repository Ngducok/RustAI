use rt_ir::{gradients, Graph, IrError, NodeId, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    Mse,
    CrossEntropy,
}

/// The nodes of an MLP inside a graph: its output and its parameters in `w0, b0, w1, b1, ...` order.
pub struct MlpNodes {
    pub output: NodeId,
    pub params: Vec<NodeId>,
    pub param_names: Vec<String>,
}

fn param_name(prefix: &str, kind: &str, layer: usize) -> String {
    if prefix.is_empty() {
        format!("{kind}{layer}")
    } else {
        format!("{prefix}.{kind}{layer}")
    }
}

/// Adds an MLP (ReLU between layers) applied to `x`. Parameters become inputs named
/// `{prefix}.w0`, `{prefix}.b0`, ... (or `w0`, `b0`, ... for an empty prefix).
pub fn mlp_nodes(g: &mut Graph, prefix: &str, sizes: &[usize], x: NodeId) -> Result<MlpNodes> {
    if sizes.len() < 2 {
        return Err(IrError::Invalid(
            "an MLP needs at least an input and an output width".into(),
        ));
    }
    let (mut hidden, mut params, mut param_names) = (x, Vec::new(), Vec::new());
    for (i, width) in sizes.windows(2).enumerate() {
        let (weight_name, bias_name) = (param_name(prefix, "w", i), param_name(prefix, "b", i));
        let weight = g.input(&weight_name, &[width[0], width[1]])?;
        let bias = g.input(&bias_name, &[width[1]])?;
        params.extend([weight, bias]);
        param_names.extend([weight_name, bias_name]);
        hidden = g.matmul(hidden, weight)?;
        hidden = g.add(hidden, bias)?;
        if i + 2 < sizes.len() {
            hidden = g.relu(hidden)?;
        }
    }
    Ok(MlpNodes {
        output: hidden,
        params,
        param_names,
    })
}

/// IR for one MLP training step.
///
/// Inputs: `x`, `target` (class indices as f32 for cross-entropy), then `w0`, `b0`, `w1`, `b1`, ...
/// Outputs: the loss followed by one gradient per parameter, in the order of `param_names`.
pub struct TrainStep {
    pub graph: Graph,
    pub param_names: Vec<String>,
}

pub fn mlp_train_step(sizes: &[usize], batch: usize, loss: Loss) -> Result<TrainStep> {
    if sizes.len() < 2 {
        return Err(IrError::Invalid(
            "an MLP needs at least an input and an output width".into(),
        ));
    }
    let mut g = Graph::new();
    let x = g.input("x", &[batch, sizes[0]])?;
    let mlp = mlp_nodes(&mut g, "", sizes, x)?;

    let loss_node = match loss {
        Loss::Mse => {
            let target = g.input("target", &[batch, sizes[sizes.len() - 1]])?;
            let diff = g.sub(mlp.output, target)?;
            let squared = g.mul(diff, diff)?;
            g.mean(squared)?
        }
        Loss::CrossEntropy => {
            let target = g.input("target", &[batch])?;
            g.cross_entropy(mlp.output, target)?
        }
    };
    let grads = gradients(&mut g, loss_node, &mlp.params)?;
    g.set_outputs(std::iter::once(loss_node).chain(grads).collect())?;
    Ok(TrainStep {
        graph: g,
        param_names: mlp.param_names,
    })
}

/// IR for evaluating an MLP: input `x` plus the parameters, output the final layer.
pub struct ForwardGraph {
    pub graph: Graph,
    pub param_names: Vec<String>,
}

pub fn mlp_forward(prefix: &str, sizes: &[usize], batch: usize) -> Result<ForwardGraph> {
    let mut g = Graph::new();
    let x = g.input("x", &[batch, sizes.first().copied().unwrap_or(0)])?;
    let mlp = mlp_nodes(&mut g, prefix, sizes, x)?;
    g.set_outputs(vec![mlp.output])?;
    Ok(ForwardGraph {
        graph: g,
        param_names: mlp.param_names,
    })
}
