use rustai::prelude::*;

fn xor_batch() -> (Tensor, Tensor) {
    (
        Tensor::from_vec(vec![0., 0., 0., 1., 1., 0., 1., 1.], &[4, 2]).unwrap(),
        Tensor::from_vec(vec![0., 1., 1., 0.], &[4]).unwrap(),
    )
}

fn xor_trainer() -> Trainer {
    Trainer::new(TrainerConfig {
        sizes: vec![2, 16, 2],
        batch: 4,
        loss: Loss::CrossEntropy,
        learning_rate: 0.05,
        seed: 42,
        ..TrainerConfig::default()
    })
    .unwrap()
}

fn params_bits(trainer: &Trainer) -> Vec<Vec<u32>> {
    trainer
        .model()
        .params()
        .iter()
        .map(|p| p.data().iter().map(|v| v.to_bits()).collect())
        .collect()
}

fn scratch(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rustai-test-{}-{name}", std::process::id()))
}

#[test]
fn learns_xor_and_predicts_for_any_number_of_rows() {
    let mut trainer = xor_trainer();
    let (x, labels) = xor_batch();
    for _ in 0..300 {
        trainer.step(&x, &labels).unwrap();
    }
    assert_eq!(trainer.steps(), 300);
    let all = trainer.predict(&x).unwrap();
    assert_eq!(all.argmax_rows().unwrap(), [0, 1, 1, 0]);

    let one = Tensor::from_vec(vec![1., 0.], &[1, 2]).unwrap();
    let single = trainer.predict(&one).unwrap();
    assert_eq!(single.shape(), &[1, 2]);
    assert_eq!(
        single.data(),
        &all.data()[4..6],
        "a row is the same alone or in a batch"
    );
}

#[test]
fn resuming_from_a_checkpoint_is_bit_identical_to_never_stopping() {
    let (x, labels) = xor_batch();
    let mut uninterrupted = xor_trainer();
    for _ in 0..10 {
        uninterrupted.step(&x, &labels).unwrap();
    }
    let path = scratch("resume.safetensors");
    uninterrupted.save(&path).unwrap();
    let continued: Vec<f32> = (0..10)
        .map(|_| uninterrupted.step(&x, &labels).unwrap())
        .collect();

    let mut resumed = xor_trainer();
    resumed.load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(resumed.steps(), 10);
    let replayed: Vec<f32> = (0..10)
        .map(|_| resumed.step(&x, &labels).unwrap())
        .collect();

    assert_eq!(continued, replayed);
    assert_eq!(params_bits(&uninterrupted), params_bits(&resumed));
}

#[test]
fn a_checkpoint_saved_before_any_step_restores_a_fresh_optimizer() {
    let path = scratch("fresh.safetensors");
    let (x, labels) = xor_batch();
    let trainer = xor_trainer();
    trainer.save(&path).unwrap();
    let mut other = Trainer::new(TrainerConfig {
        seed: 7,
        ..xor_config()
    })
    .unwrap();
    other.load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(params_bits(&trainer), params_bits(&other));
    other.step(&x, &labels).unwrap();
}

fn xor_config() -> TrainerConfig {
    TrainerConfig {
        sizes: vec![2, 16, 2],
        batch: 4,
        loss: Loss::CrossEntropy,
        ..TrainerConfig::default()
    }
}

#[test]
fn loading_a_checkpoint_of_a_different_network_fails_cleanly() {
    let path = scratch("other.safetensors");
    xor_trainer().save(&path).unwrap();
    let mut wider = Trainer::new(TrainerConfig {
        sizes: vec![2, 32, 2],
        ..xor_config()
    })
    .unwrap();
    assert!(wider.load(&path).is_err());
    assert!(wider.load(&scratch("missing.safetensors")).is_err());
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn rejects_batches_of_the_wrong_shape() {
    let mut trainer = xor_trainer();
    let (_, labels) = xor_batch();
    assert!(trainer.step(&Tensor::zeros(&[3, 2]), &labels).is_err());
    assert!(trainer.predict(&Tensor::zeros(&[5, 2])).is_err());
}

#[test]
fn gradients_are_those_of_the_step_that_just_ran() {
    use rustai::nn::mlp_train_step;
    use rustai::runtime::Executor;

    let (x, labels) = xor_batch();
    let mut trainer = xor_trainer();
    let before: Vec<Tensor> = trainer.model().params().into_iter().cloned().collect();
    let loss = trainer.step(&x, &labels).unwrap();

    let step = mlp_train_step(&[2, 16, 2], 4, Loss::CrossEntropy).unwrap();
    let mut executor = Executor::compile(&step.graph).unwrap();
    let mut feeds = vec![("x", &x), ("target", &labels)];
    feeds.extend(step.param_names.iter().map(String::as_str).zip(&before));
    executor.run(&feeds).unwrap();

    assert_eq!(executor.output(0)[0], loss);
    assert_eq!(trainer.gradients().len(), before.len());
    for (i, grad) in trainer.gradients().iter().enumerate() {
        assert_eq!(grad.data(), executor.output(i + 1));
        assert!(grad.data().iter().any(|&g| g != 0.0));
    }
}
