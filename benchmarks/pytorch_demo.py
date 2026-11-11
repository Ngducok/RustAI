"""Trains the demo network with PyTorch from the weights and data the Rust demo wrote.

Usage: python pytorch_demo.py <directory containing rust.json and initial.safetensors>
Writes <directory>/torch.json with the same measurements as the Rust run, plus the difference
between the two frameworks' gradients at step 0.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file


def build(sizes, weights):
    layers = []
    for i, (fan_in, fan_out) in enumerate(zip(sizes, sizes[1:])):
        layer = torch.nn.Linear(fan_in, fan_out)
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(weights[f"w{i}"]).T)
            layer.bias.copy_(torch.from_numpy(weights[f"b{i}"]))
        layers.append(layer)
        if i + 2 < len(sizes):
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def predict(model, points):
    with torch.no_grad():
        return model(points).argmax(dim=1).numpy()


def main():
    directory = Path(sys.argv[1])
    rust = json.loads((directory / "rust.json").read_text())
    config, data, grid = rust["config"], rust["data"], rust["grid"]
    sizes = config["sizes"]

    tensor = lambda values, shape=None: torch.tensor(values, dtype=torch.float32).reshape(shape or -1)
    x, labels = tensor(data["x"], (-1, 2)), tensor(data["labels"]).long()
    test_x, test_labels = tensor(data["test_x"], (-1, 2)), tensor(data["test_labels"]).long()

    model = build(sizes, load_file(directory / "initial.safetensors"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    loss_fn = torch.nn.CrossEntropyLoss()

    def accuracy(points, target):
        return float((predict(model, points) == target.numpy()).mean())

    losses, norms, micros, curve = [], [], [], []
    step0 = None
    for step in range(config["steps"]):
        started = time.perf_counter()
        optimizer.zero_grad()
        loss = loss_fn(model(x), labels)
        loss.backward()
        optimizer.step()
        value = loss.item()
        micros.append((time.perf_counter() - started) * 1e6)
        losses.append(value)
        grads = [p.grad.detach().clone() for p in model.parameters()]
        norms.append(float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads))))
        if step == 0:
            step0 = grads
        if step % 25 == 0 or step + 1 == config["steps"]:
            curve.append([step, accuracy(x, labels)])

    differences = []
    for expected, actual in zip(rust["first_grads"], step0):
        theirs = np.array(expected["data"], dtype=np.float64).reshape(expected["shape"])
        mine = actual.numpy().astype(np.float64)
        mine = mine.T if mine.ndim == 2 else mine
        scale = max(np.abs(mine).max(), 1e-12)
        differences.append({
            "name": expected["name"],
            "max_abs": float(np.abs(theirs - mine).max()),
            "max_relative_to_largest": float(np.abs(theirs - mine).max() / scale),
        })

    count, extent = grid["size"], grid["extent"]
    axis = np.linspace(-extent, extent, count, dtype=np.float32)
    points = np.array([[axis[c], axis[count - 1 - r]] for r in range(count) for c in range(count)], dtype=np.float32)
    result = {
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "loss": losses,
        "grad_norm": norms,
        "micros": micros,
        "total_seconds": sum(micros) / 1e6,
        "accuracy_curve": curve,
        "final_accuracy": accuracy(x, labels),
        "test_accuracy": accuracy(test_x, test_labels),
        "grid_prediction": predict(model, torch.from_numpy(points)).tolist(),
        "step0_gradient_difference": differences,
    }
    (directory / "torch.json").write_text(json.dumps(result))
    print(f"torch {torch.__version__}: {config['steps']} steps in {result['total_seconds']:.3f} s; "
          f"loss {losses[0]:.4f} -> {losses[-1]:.4f}; train accuracy {100 * result['final_accuracy']:.1f}%")


if __name__ == "__main__":
    main()
