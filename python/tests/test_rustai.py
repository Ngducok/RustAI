"""Tests for the Python bindings. Run with:
    python tools/build_python.py
    PYTHONPATH=python python -m unittest discover -s python/tests -v
"""
import os
import tempfile
import unittest

import numpy as np
import torch

import rustai


def torch_mlp(sizes, seed):
    torch.manual_seed(seed)
    layers = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(torch.nn.Linear(a, b))
        if i + 2 < len(sizes):
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def parameters_of(model):
    """Torch stores weights as (out, in); rustai uses (in, out)."""
    linears = [m for m in model if isinstance(m, torch.nn.Linear)]
    params = {}
    for i, layer in enumerate(linears):
        params[f"w{i}"] = np.ascontiguousarray(layer.weight.detach().numpy().T)
        params[f"b{i}"] = np.ascontiguousarray(layer.bias.detach().numpy())
    return params


class TrainerAgainstPyTorch(unittest.TestCase):
    sizes, batch = [4, 12, 3], 16

    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.uniform(-1, 1, size=(self.batch, 4)).astype(np.float32)
        self.labels = rng.integers(0, 3, size=self.batch).astype(np.float32)
        self.targets = rng.uniform(-1, 1, size=(self.batch, 3)).astype(np.float32)

    def run_both(self, loss_name, target, steps=25):
        model = torch_mlp(self.sizes, seed=3)
        trainer = rustai.MlpTrainer(self.sizes, self.batch, loss=loss_name, learning_rate=0.01)
        trainer.set_parameters(parameters_of(model))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
        x = torch.from_numpy(self.x)
        losses = []
        for _ in range(steps):
            optimizer.zero_grad()
            out = model(x)
            if loss_name == "mse":
                loss = ((out - torch.from_numpy(target)) ** 2).mean()
            else:
                loss = torch.nn.functional.cross_entropy(out, torch.from_numpy(target).long())
            loss.backward()
            optimizer.step()
            losses.append((trainer.step(self.x, target), loss.item()))
        return model, trainer, losses

    def test_cross_entropy_training_follows_pytorch(self):
        model, trainer, losses = self.run_both("cross_entropy", self.labels)
        for ours, reference in losses:
            self.assertAlmostEqual(ours, reference, delta=1e-4 + 1e-4 * abs(reference))
        for name, expected in parameters_of(model).items():
            np.testing.assert_allclose(trainer.parameters()[name], expected, atol=1e-4, rtol=1e-3)
        self.assertEqual(trainer.steps, 25)

    def test_mse_training_follows_pytorch(self):
        _, _, losses = self.run_both("mse", self.targets)
        for ours, reference in losses:
            self.assertAlmostEqual(ours, reference, delta=1e-4 + 1e-4 * abs(reference))

    def test_predictions_match_pytorch_for_any_row_count(self):
        model = torch_mlp(self.sizes, seed=5)
        trainer = rustai.MlpTrainer(self.sizes, self.batch, loss="mse")
        trainer.set_parameters(parameters_of(model))
        for rows in (1, 5, 16):
            expected = model(torch.from_numpy(self.x[:rows])).detach().numpy()
            got = trainer.predict(self.x[:rows])
            self.assertEqual(got.shape, (rows, 3))
            self.assertEqual(got.dtype, np.float32)
            np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-4)


class Checkpoints(unittest.TestCase):
    def test_python_safetensors_reads_the_checkpoint_and_training_resumes(self):
        from safetensors.numpy import load_file

        x = np.random.default_rng(1).uniform(-1, 1, size=(8, 2)).astype(np.float32)
        y = (x.sum(axis=1, keepdims=True) > 0).astype(np.float32)
        config = dict(sizes=[2, 6, 1], batch=8, loss="mse", learning_rate=0.02)
        trainer = rustai.MlpTrainer(**config)
        for _ in range(10):
            trainer.step(x, y)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.safetensors")
            trainer.save(path)
            stored = load_file(path)
            for name, value in trainer.parameters().items():
                np.testing.assert_array_equal(stored[name], value)
            self.assertIn("adam.m.0", stored)

            continued = [trainer.step(x, y) for _ in range(5)]
            resumed = rustai.MlpTrainer(**config)
            resumed.load(path)
            self.assertEqual(resumed.steps, 10)
            self.assertEqual([resumed.step(x, y) for _ in range(5)], continued)


class InputValidation(unittest.TestCase):
    def test_errors_are_python_exceptions(self):
        trainer = rustai.MlpTrainer([2, 4, 1], 4)
        good = np.zeros((4, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            trainer.step(np.zeros((3, 2), dtype=np.float32), np.zeros((3, 1), dtype=np.float32))
        with self.assertRaises(Exception):
            trainer.step(good.astype(np.float64), np.zeros((4, 1), dtype=np.float32))
        with self.assertRaises(ValueError):
            rustai.MlpTrainer([2, 4, 1], 4, loss="hinge")
        with self.assertRaises(ValueError):
            trainer.set_parameters({"w0": np.zeros((2, 4), dtype=np.float32)})
        with self.assertRaises(ValueError):
            trainer.predict(np.zeros((9, 2), dtype=np.float32))
        with self.assertRaises(RuntimeError):
            trainer.load("does-not-exist.safetensors")


class Reinforcement(unittest.TestCase):
    def test_ppo_learns_cartpole(self):
        score = rustai.train_ppo_cartpole(60_000, seed=1, learning_rate=1e-3)
        self.assertGreater(score, 100.0)

    def test_module_metadata(self):
        self.assertRegex(rustai.__version__, r"^\d+\.\d+\.\d+")


if __name__ == "__main__":
    unittest.main()
