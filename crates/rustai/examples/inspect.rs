//! Shows how a training step is differentiated, optimised, laid out in memory and scheduled.
//!
//! `cargo run --release -p rustai --features gpu --example inspect -- [--json out.json]`
//! prints a worked autodiff example and a summary of an MLP training step; with `--json` it also
//! writes every node, buffer and kernel launch for `tools/explain_report.py`.

use std::collections::BTreeMap;

use rustai::compiler::{
    eliminate_common_subexpressions, fold_broadcasts, fuse_linear, fuse_relu_grad, optimize,
};
use rustai::ir::{gradients, Graph, Node, Op};
use rustai::memory::{liveness, plan, Plan};
use rustai::nn::{mlp_forward, mlp_train_step, Loss, Mlp};
use rustai::prelude::*;
use rustai::runtime::Executor;
use serde_json::{json, Value};

fn op_name(op: &Op) -> String {
    match op {
        Op::Input(name) => format!("Input({name})"),
        Op::Linear { relu: true } => "Linear+ReLU".into(),
        Op::Linear { relu: false } => "Linear".into(),
        Op::Scale(s) => format!("Scale({s:.4})"),
        other => format!("{other:?}"),
    }
}

fn describe(index: usize, node: &Node) -> String {
    let inputs: Vec<String> = node.inputs.iter().map(|i| format!("%{}", i.0)).collect();
    format!(
        "  %{index:<2} = {}({}) : {:?}",
        op_name(&node.op),
        inputs.join(", "),
        node.shape
    )
}

fn print_graph(graph: &Graph, forward_len: usize) {
    for (i, node) in graph.nodes().iter().enumerate() {
        let tag = match node.op {
            Op::Input(_) => "",
            _ if i < forward_len => "   forward",
            _ => "   backward",
        };
        println!("{}{tag}", describe(i, node));
    }
}

fn loss_and_grads(graph: &Graph, feeds: &[(&str, &Tensor)]) -> rustai::Result<(f32, Vec<Tensor>)> {
    let mut executor = Executor::compile(graph)?;
    executor.run(feeds)?;
    let grads = (1..graph.outputs().len())
        .map(|i| executor.output_tensor(i))
        .collect();
    Ok((executor.output(0)[0], grads))
}

/// A hand-sized graph: differentiate it, print the nodes the autodiff added, and compare the
/// result with central finite differences.
fn walkthrough() -> rustai::Result<Value> {
    let mut g = Graph::new();
    let x = g.input("x", &[4, 2])?;
    let w = g.input("w", &[2, 3])?;
    let b = g.input("b", &[3])?;
    let pre = g.matmul(x, w)?;
    let pre = g.add(pre, b)?;
    let h = g.relu(pre)?;
    let squared = g.mul(h, h)?;
    let loss = g.mean(squared)?;
    let forward_len = g.len();
    let grads = gradients(&mut g, loss, &[w, b])?;
    g.set_outputs([vec![loss], grads].concat())?;

    println!("== autodiff on loss = mean(relu(x*w + b)^2)");
    print_graph(&g, forward_len);

    let xs = [1.0, 2.0, -1.0, 0.5, 0.5, -1.0, 2.0, 1.0];
    let ws = [0.5, -0.3, 0.2, 0.1, 0.4, -0.6];
    let bs = [0.1, 0.3, 0.3];
    let tensor = |data: &[f32], shape: &[usize]| Tensor::from_vec(data.to_vec(), shape);
    let (xt, wt, bt) = (
        tensor(&xs, &[4, 2])?,
        tensor(&ws, &[2, 3])?,
        tensor(&bs, &[3])?,
    );
    let loss_of = |w: &Tensor, b: &Tensor| -> rustai::Result<f32> {
        let feeds = [("x", &xt), ("w", w), ("b", b)];
        Ok(loss_and_grads(&g, &feeds)?.0)
    };
    let (value, analytic) = loss_and_grads(&g, &[("x", &xt), ("w", &wt), ("b", &bt)])?;
    println!("\nloss = {value:.6}");
    println!("  parameter   autodiff   finite difference");
    let eps = 1e-2;
    let mut worst = 0.0f32;
    let mut rows = Vec::new();
    for (which, name, tensor) in [(0, "w", &wt), (1, "b", &bt)] {
        for i in 0..tensor.numel() {
            let shifted = |sign: f32| {
                let mut t = tensor.clone();
                t.data_mut()[i] += sign * eps;
                t
            };
            let (plus, minus) = if which == 0 {
                (loss_of(&shifted(1.0), &bt)?, loss_of(&shifted(-1.0), &bt)?)
            } else {
                (loss_of(&wt, &shifted(1.0))?, loss_of(&wt, &shifted(-1.0))?)
            };
            let numeric = (plus - minus) / (2.0 * eps);
            let exact = analytic[which].data()[i];
            worst = worst.max((exact - numeric).abs());
            println!("  {name}[{i}]  {exact:>10.6}  {numeric:>10.6}");
            rows.push(json!({
                "parameter": format!("{name}[{i}]"),
                "autodiff": exact,
                "finite_difference": numeric,
            }));
        }
    }
    println!("largest difference: {worst:.2e}");
    let nodes: Vec<Value> = g
        .nodes()
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let phase = match node.op {
                Op::Input(_) => "input",
                _ if i < forward_len => "forward",
                _ => "backward",
            };
            json!({
                "id": i,
                "op": op_name(&node.op),
                "inputs": node.inputs.iter().map(|n| n.0).collect::<Vec<_>>(),
                "shape": node.shape,
                "phase": phase,
            })
        })
        .collect();
    Ok(json!({"nodes": nodes, "loss": value, "check": rows, "largest_difference": worst}))
}

fn op_counts(graph: &Graph) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for node in graph.nodes() {
        if !matches!(node.op, Op::Input(_)) {
            *counts.entry(op_name(&node.op)).or_insert(0) += 1;
        }
    }
    counts
}

/// For every parameter tensor, the derivative of the loss along the unit vector of its own
/// gradient: the compiled gradient graph gives `|g|`, finite differences of the loss give the rest.
fn gradient_check(sizes: &[usize], batch: usize) -> rustai::Result<Value> {
    let step = mlp_train_step(sizes, batch, Loss::CrossEntropy)?;
    let mut rng = Rng::new(7);
    let model = Mlp::new(sizes, &mut rng);
    let classes = sizes[sizes.len() - 1];
    let x = Tensor::uniform(&[batch, sizes[0]], -1.0, 1.0, &mut rng);
    let labels = Tensor::from_vec(
        (0..batch).map(|_| rng.below(classes) as f32).collect(),
        &[batch],
    )?;
    let params: Vec<Tensor> = model.params().into_iter().cloned().collect();
    let evaluate = |params: &[Tensor]| -> rustai::Result<(f32, Vec<Tensor>)> {
        let mut feeds = vec![("x", &x), ("target", &labels)];
        feeds.extend(step.param_names.iter().map(String::as_str).zip(params));
        loss_and_grads(&step.graph, &feeds)
    };
    let (_, grads) = evaluate(&params)?;

    let eps = 1e-3f64;
    let mut checks = Vec::new();
    for (k, name) in step.param_names.iter().enumerate() {
        let norm = grads[k]
            .data()
            .iter()
            .map(|&g| f64::from(g).powi(2))
            .sum::<f64>()
            .sqrt();
        let shifted = |sign: f64| -> Vec<Tensor> {
            let mut moved = params.clone();
            for (v, &g) in moved[k].data_mut().iter_mut().zip(grads[k].data()) {
                *v += (sign * eps * f64::from(g) / norm) as f32;
            }
            moved
        };
        let numeric = (f64::from(evaluate(&shifted(1.0))?.0)
            - f64::from(evaluate(&shifted(-1.0))?.0))
            / (2.0 * eps);
        checks.push(json!({
            "parameter": name,
            "autodiff": norm,
            "finite_difference": numeric,
            "relative_error": (norm - numeric).abs() / norm.max(1e-12),
        }));
    }
    Ok(json!(checks))
}

struct Analysis {
    optimised: Graph,
    layout: Plan,
    last_use: Vec<usize>,
    forward: Vec<bool>,
    live: Vec<usize>,
}

impl Analysis {
    fn new(raw: &Graph) -> rustai::Result<Self> {
        let optimised = optimize(raw)?;
        let layout = plan(&optimised);
        layout.validate(&optimised).map_err(Error::Invalid)?;
        let last_use = liveness(&optimised).last_use;

        let mut forward = vec![false; optimised.len()];
        let mut stack = vec![optimised.outputs()[0]];
        while let Some(id) = stack.pop() {
            if !forward[id.0] {
                forward[id.0] = true;
                stack.extend(optimised.node(id).inputs.iter().copied());
            }
        }
        let live = (0..optimised.len())
            .map(|t| {
                (0..=t)
                    .filter(|&i| !is_input(&optimised, i) && last_use[i] >= t)
                    .map(|i| elements(&optimised, i))
                    .sum()
            })
            .collect();
        Ok(Self {
            optimised,
            layout,
            last_use,
            forward,
            live,
        })
    }

    fn planned(&self) -> usize {
        self.layout.planned_elements()
    }

    fn unplanned(&self) -> usize {
        Plan::unplanned_elements(&self.optimised)
    }

    fn peak(&self) -> usize {
        self.live.iter().copied().max().unwrap_or(0)
    }

    fn to_json(&self, name: &str, sizes: &[usize], batch: usize, raw: &Graph) -> Value {
        let nodes: Vec<Value> = self
            .optimised
            .nodes()
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let phase = match node.op {
                    Op::Input(_) => "input",
                    _ if self.forward[i] => "forward",
                    _ => "backward",
                };
                json!({
                    "id": i,
                    "op": op_name(&node.op),
                    "inputs": node.inputs.iter().map(|n| n.0).collect::<Vec<_>>(),
                    "shape": node.shape,
                    "phase": phase,
                    "last_use": if self.last_use[i] == usize::MAX { Value::Null } else { json!(self.last_use[i]) },
                    "buffer": self.layout.buffer_of[i],
                })
            })
            .collect();
        json!({
            "name": name,
            "sizes": sizes,
            "batch": batch,
            "nodes_before": raw.len(),
            "nodes_after": self.optimised.len(),
            "ops_before": op_counts(raw),
            "ops_after": op_counts(&self.optimised),
            "nodes": nodes,
            "buffer_sizes": self.layout.buffer_sizes,
            "planned_elements": self.planned(),
            "unplanned_elements": self.unplanned(),
            "peak_live_elements": self.peak(),
            "live_elements": self.live,
        })
    }
}

fn is_input(graph: &Graph, index: usize) -> bool {
    matches!(graph.nodes()[index].op, Op::Input(_))
}

fn elements(graph: &Graph, index: usize) -> usize {
    graph.nodes()[index].shape.iter().product()
}

fn mib(elements: usize) -> f64 {
    elements as f64 * 4.0 / 1_048_576.0
}

struct Case {
    name: &'static str,
    sizes: Vec<usize>,
    batch: usize,
    training: bool,
}

fn inspect(json_path: Option<&str>, walkthrough: Value) -> rustai::Result<()> {
    let deep: Vec<usize> = [vec![32], vec![128; 8], vec![10]].concat();
    let cases = [
        Case {
            name: "demo network, training step",
            sizes: vec![2, 64, 64, 3],
            batch: 510,
            training: true,
        },
        Case {
            name: "deep network, training step",
            sizes: deep.clone(),
            batch: 256,
            training: true,
        },
        Case {
            name: "deep network, inference",
            sizes: deep,
            batch: 256,
            training: false,
        },
    ];

    println!("\n== memory planning (float32 elements, 4 bytes each)");
    println!(
        "  {:<30}{:>7}{:>7}{:>10}{:>10}{:>10}{:>8}",
        "", "nodes", "bufs", "own MiB", "plan MiB", "peak MiB", "saving"
    );
    let mut documents = Vec::new();
    let mut first = None;
    for case in &cases {
        let raw = if case.training {
            mlp_train_step(&case.sizes, case.batch, Loss::CrossEntropy)?.graph
        } else {
            mlp_forward("", &case.sizes, case.batch)?.graph
        };
        let analysis = Analysis::new(&raw)?;
        println!(
            "  {:<30}{:>7}{:>7}{:>10.2}{:>10.2}{:>10.2}{:>7.1}x",
            case.name,
            analysis.optimised.len(),
            analysis.layout.buffer_sizes.len(),
            mib(analysis.unplanned()),
            mib(analysis.planned()),
            mib(analysis.peak()),
            analysis.unplanned() as f64 / analysis.planned() as f64,
        );
        documents.push(analysis.to_json(case.name, &case.sizes, case.batch, &raw));
        first.get_or_insert((raw, analysis));
    }
    let Some((raw, analysis)) = first else {
        return Ok(());
    };

    println!("\n== optimisation passes on the demo training step");
    let folded = fold_broadcasts(&raw)?;
    let merged = eliminate_common_subexpressions(&folded)?;
    let linear = fuse_linear(&merged)?;
    let fused = fuse_relu_grad(&linear)?;
    let passes = [
        ("autodiff output", raw.len()),
        ("broadcast folding", folded.len()),
        ("common subexpressions", merged.len()),
        ("Linear fusion", linear.len()),
        ("ReLU-gradient fusion", fused.len()),
        ("dead code", analysis.optimised.len()),
    ];
    for (name, count) in passes {
        println!("  {name:<24}{count:>4} nodes");
    }
    println!("  ops before: {:?}", op_counts(&raw));
    println!("  ops after:  {:?}", op_counts(&analysis.optimised));

    let checks = gradient_check(&cases[0].sizes, 64)?;
    println!("\n== autodiff against finite differences, one row per parameter tensor");
    println!("  tensor   autodiff   finite difference   relative error");
    for c in checks.as_array().into_iter().flatten() {
        println!(
            "  {:<6}{:>10.6}{:>18.6}{:>17.1e}",
            c["parameter"].as_str().unwrap_or(""),
            c["autodiff"].as_f64().unwrap_or(f64::NAN),
            c["finite_difference"].as_f64().unwrap_or(f64::NAN),
            c["relative_error"].as_f64().unwrap_or(f64::NAN)
        );
    }

    #[cfg(feature = "gpu")]
    let schedule: Vec<Value> = rustai::gpu::schedule(&analysis.optimised)
        .map_err(|e| Error::Invalid(e.to_string()))?
        .into_iter()
        .map(|k| json!({"node": k.node, "kernel": k.kernel, "groups": k.groups}))
        .collect();
    #[cfg(not(feature = "gpu"))]
    let schedule: Vec<Value> = Vec::new();
    if !schedule.is_empty() {
        let mut per_kernel: BTreeMap<&str, usize> = BTreeMap::new();
        for k in &schedule {
            *per_kernel
                .entry(k["kernel"].as_str().unwrap_or(""))
                .or_insert(0) += 1;
        }
        println!(
            "\n== GPU: {} kernel launches per step, recorded once into one compute pass: {per_kernel:?}",
            schedule.len()
        );
    }

    if let Some(path) = json_path {
        let document = json!({
            "walkthrough": walkthrough,
            "cases": documents,
            "passes": passes.iter().map(|(n, c)| json!({"name": n, "nodes": c})).collect::<Vec<_>>(),
            "gradient_check": checks,
            "gpu_schedule": schedule,
        });
        let text =
            serde_json::to_string_pretty(&document).map_err(|e| Error::Invalid(e.to_string()))?;
        std::fs::write(path, text).map_err(|e| Error::Invalid(e.to_string()))?;
        println!("\nwrote {path}");
    }
    Ok(())
}

fn main() -> rustai::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let json_path = args
        .iter()
        .position(|a| a == "--json")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str);
    let walk = walkthrough()?;
    inspect(json_path, walk)
}
