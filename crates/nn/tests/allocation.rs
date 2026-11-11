//! One test per binary: the counting allocator is process-global.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use rt_autograd::Graph;
use rt_nn::{mlp_train_step, mse, Loss, Mlp};
use rt_runtime::{run_reference, Executor};
use rt_tensor::{Rng, Tensor};

struct Counting;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size(), Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn measure<T>(f: impl FnOnce() -> T) -> (usize, usize) {
    let (a, b) = (ALLOCATIONS.load(Relaxed), BYTES.load(Relaxed));
    std::hint::black_box(f());
    (ALLOCATIONS.load(Relaxed) - a, BYTES.load(Relaxed) - b)
}

#[test]
fn planned_steady_state_step_does_not_allocate() {
    let sizes = [16, 64, 64, 4];
    let batch = 32;
    let mut rng = Rng::new(5);
    let mut model = Mlp::new(&sizes, &mut rng);
    let x = Tensor::uniform(&[batch, sizes[0]], -1.0, 1.0, &mut rng);
    let target = Tensor::uniform(&[batch, sizes[3]], -1.0, 1.0, &mut rng);

    let step = mlp_train_step(&sizes, batch, Loss::Mse).unwrap();
    let params = model.params_mut();
    let mut feeds = vec![("x", &x), ("target", &target)];
    feeds.extend(
        step.param_names
            .iter()
            .map(String::as_str)
            .zip(params.iter().map(|p| &**p)),
    );

    let mut executor = Executor::compile(&step.graph).unwrap();
    executor.run(&feeds).unwrap();
    rayon::broadcast(|_| ());
    let planned = measure(|| executor.run(&feeds).unwrap());
    let reference = measure(|| run_reference(&step.graph, &feeds).unwrap());
    drop(params);

    let dynamic = measure(|| {
        let mut g = Graph::new();
        let bound = model.bind(&mut g);
        let input = g.constant(x.clone());
        let out = bound.forward(&mut g, input).unwrap();
        let t = g.constant(target.clone());
        let loss = mse(&mut g, out, t).unwrap();
        g.backward(loss).unwrap()
    });

    println!(
        "dynamic tape : {} allocations, {} bytes",
        dynamic.0, dynamic.1
    );
    println!(
        "reference IR : {} allocations, {} bytes",
        reference.0, reference.1
    );
    println!(
        "planned IR   : {} allocations, {} bytes",
        planned.0, planned.1
    );
    assert_eq!(
        planned.0, 0,
        "planned executor allocated {} times",
        planned.0
    );
}
