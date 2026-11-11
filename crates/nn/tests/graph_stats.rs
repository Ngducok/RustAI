use rt_compiler::{eliminate_dead_code, optimize};
use rt_nn::{mlp_train_step, Loss};

#[test]
fn print_node_histogram() {
    for (name, loss) in [("mse", Loss::Mse), ("cross-entropy", Loss::CrossEntropy)] {
        let step = mlp_train_step(&[128, 256, 256, 10], 256, loss).unwrap();
        let plain = eliminate_dead_code(&step.graph).unwrap();
        let graph = optimize(&step.graph).unwrap();
        println!("{name}: dce only {} nodes", plain.len());
        let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
        for node in graph.nodes() {
            let label = format!("{:?}", node.op);
            let label = label.split(['(', ' ']).next().unwrap().to_string();
            *counts.entry(label).or_default() += 1;
        }
        println!("{name}: {} nodes after DCE: {counts:?}", graph.len());
    }
}
