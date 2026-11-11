"""Generates PyTorch (float64) reference values for the Rust parity tests.

Usage: python generate.py
Inputs are rounded to float32 first so Rust and PyTorch start from identical values.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.default_rng(0)
OUT = Path(__file__).parent / "fixtures" / "pytorch.json"


def f32(shape, lo=-1.0, hi=1.0):
    return np.asarray(rng.uniform(lo, hi, size=shape), dtype=np.float32)


def away_from_zero(shape, margin=0.1):
    a = f32(shape)
    return np.where(a >= 0, a + margin, a - margin).astype(np.float32)


def enc(a):
    a = np.asarray(a, dtype=np.float64)
    return {"shape": list(a.shape), "data": a.ravel().tolist()}


def leaf(a):
    return torch.tensor(np.asarray(a, dtype=np.float64), requires_grad=True)


def square_sum(t):
    return (t * t).sum()


def mlp(x, w1, b1, w2, b2):
    return torch.relu(x @ w1 + b1) @ w2 + b2


def autograd_case(name, op, arrays, fn, targets=()):
    leaves = [leaf(a) for a in arrays]
    loss = fn(*leaves)
    loss.backward()
    return {
        "name": name,
        "op": op,
        "inputs": [enc(a) for a in arrays],
        "targets": list(targets),
        "loss": float(loss),
        "grads": [enc(t.grad.numpy()) for t in leaves],
    }


def autograd_cases():
    cases = []
    for k in range(3):
        cases += [
            autograd_case(f"add_bcast_{k}", "add_bcast", [f32((3, 4)), f32((4,))],
                          lambda a, b: square_sum(a + b)),
            autograd_case(f"sub_mul_col_{k}", "sub_mul_col", [f32((3, 4)), f32((3, 1))],
                          lambda a, b: square_sum((a - b) * b)),
            autograd_case(f"scalar_bcast_{k}", "scalar_bcast",
                          [f32((2, 3)), np.asarray(rng.uniform(-1, 1), dtype=np.float32)],
                          lambda a, b: square_sum(a * b)),
            autograd_case(f"outer_bcast_{k}", "add_bcast", [f32((4, 1)), f32((1, 5))],
                          lambda a, b: square_sum(a + b)),
            autograd_case(f"matmul_{k}", "matmul", [f32((5, 4)), f32((4, 3))],
                          lambda a, b: square_sum(a @ b)),
            autograd_case(f"scale_{k}", "scale", [f32((3, 3)), f32((3, 3))],
                          lambda a, b: ((a * -2.5) * b).sum()),
            autograd_case(f"relu_mean_{k}", "relu_mean", [away_from_zero((4, 5))],
                          lambda a: (torch.relu(a) ** 2).mean()),
        ]
        cases += [
            autograd_case(f"exp_{k}", "exp", [f32((3, 4), -1.5, 1.5)],
                          lambda a: square_sum(torch.exp(a))),
            autograd_case(f"log_softmax_{k}", "log_softmax", [f32((4, 5), -2, 2)],
                          lambda a: square_sum(F.log_softmax(a, dim=1))),
            autograd_case(f"clamp_{k}", "clamp", [f32((4, 5), -1, 1)],
                          lambda a: (torch.clamp(a, -0.3, 0.4) ** 2).mean()),
            autograd_case(f"minimum_{k}", "minimum", [f32((3, 4)), f32((3, 4))],
                          lambda a, b: square_sum(torch.minimum(a, b))),
        ]
        labels = rng.integers(0, 5, size=6).tolist()
        cases.append(autograd_case(
            f"cross_entropy_{k}", "cross_entropy", [f32((6, 5), -3, 3)],
            lambda a: F.cross_entropy(a, torch.tensor(labels)), labels))
        cases.append(autograd_case(
            f"cross_entropy_extreme_{k}", "cross_entropy", [f32((6, 5), -40, 40)],
            lambda a: F.cross_entropy(a, torch.tensor(labels)), labels))

        labels = rng.integers(0, 3, size=8).tolist()
        mlp_inputs = [f32((8, 4)), f32((4, 6), -0.8, 0.8), f32((6,), -0.3, 0.3),
                      f32((6, 3), -0.8, 0.8), f32((3,), -0.3, 0.3)]
        cases.append(autograd_case(
            f"mlp_ce_{k}", "mlp_ce", mlp_inputs,
            lambda *p: F.cross_entropy(mlp(*p), torch.tensor(labels)), labels))
        cases.append(autograd_case(
            f"mlp_mse_{k}", "mlp_mse", mlp_inputs + [f32((8, 3))],
            lambda *p: ((mlp(*p[:5]) - p[5]) ** 2).mean()))

    labels = rng.integers(0, 4, size=64).tolist()
    large = [f32((64, 16)), f32((16, 32), -0.5, 0.5), f32((32,), -0.2, 0.2),
             f32((32, 4), -0.5, 0.5), f32((4,), -0.2, 0.2)]
    cases.append(autograd_case(
        "mlp_ce_large", "mlp_ce", large,
        lambda *p: F.cross_entropy(mlp(*p), torch.tensor(labels)), labels))
    return cases


def make_optimizer(spec, params):
    if spec["kind"] == "adamw":
        return torch.optim.AdamW(params, lr=spec["lr"], weight_decay=spec["weight_decay"])
    return torch.optim.SGD(params, lr=spec["lr"], momentum=spec["momentum"])


OPTIMIZERS = [
    {"kind": "adamw", "lr": 0.01, "weight_decay": 0.01, "momentum": 0.0},
    {"kind": "adamw", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.0},
    {"kind": "sgd", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.0},
    {"kind": "sgd", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9},
]


def optimizer_cases():
    cases = []
    for i, spec in enumerate(OPTIMIZERS):
        init = [f32((3, 4)), f32((4,))]
        steps = [[f32((3, 4)), f32((4,))] for _ in range(10)]
        params = [leaf(p) for p in init]
        opt = make_optimizer(spec, params)
        trajectory = []
        for grads in steps:
            for p, g in zip(params, grads):
                p.grad = torch.tensor(np.asarray(g, dtype=np.float64))
            opt.step()
            trajectory.append([enc(p.detach().numpy()) for p in params])
        cases.append({
            "name": f"{spec['kind']}_{i}",
            "optimizer": spec,
            "init": [enc(p) for p in init],
            "grads": [[enc(g) for g in step] for step in steps],
            "trajectory": trajectory,
        })
    return cases


def training_case(name, loss_kind, spec, x, y, labels, out_dim, steps=30):
    init = [f32((x.shape[1], 8), -0.7, 0.7), f32((8,), -0.1, 0.1),
            f32((8, out_dim), -0.7, 0.7), f32((out_dim,), -0.1, 0.1)]
    params = [leaf(p) for p in init]
    opt = make_optimizer(spec, params)
    x_t = torch.tensor(x.astype(np.float64))
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        out = mlp(x_t, *params)
        if loss_kind == "mse":
            loss = ((out - torch.tensor(y.astype(np.float64))) ** 2).mean()
        else:
            loss = F.cross_entropy(out, torch.tensor(labels))
        loss.backward()
        losses.append(float(loss))
        opt.step()
    return {
        "name": name,
        "loss_kind": loss_kind,
        "optimizer": spec,
        "x": enc(x),
        "y": enc(y),
        "targets": list(labels),
        "params": [enc(p) for p in init],
        "losses": losses,
        "final": [enc(p.detach().numpy()) for p in params],
    }


def training_cases():
    x = f32((16, 3))
    y = np.sin(x.sum(axis=1, keepdims=True)).astype(np.float32)
    xc = f32((20, 3))
    labels = ((xc[:, 0] > 0).astype(int) + (xc[:, 1] > 0.3).astype(int)).tolist()
    return [
        training_case("train_mse_adamw", "mse", OPTIMIZERS[0], x, y, [], 1),
        training_case("train_ce_sgd_momentum", "cross_entropy", OPTIMIZERS[3],
                      xc, np.zeros((1, 1), dtype=np.float32), labels, 3),
    ]


def mlp_forward(x, params):
    layers = len(params) // 2
    for i in range(layers):
        x = x @ params[2 * i] + params[2 * i + 1]
        if i + 1 < layers:
            x = torch.relu(x)
    return x


PPO_CLIP, PPO_VF, PPO_ENT = 0.2, 0.5, 0.01


def ppo_case(name, batch, obs_dim, actions, hidden, masked):
    actor_sizes, critic_sizes = [obs_dim, hidden, actions], [obs_dim, hidden, 1]

    def params(sizes):
        out = []
        for a, b in zip(sizes, sizes[1:]):
            out += [f32((a, b), -0.8, 0.8), f32((b,), -0.3, 0.3)]
        return out

    actor, critic = params(actor_sizes), params(critic_sizes)
    obs = f32((batch, obs_dim))
    allowed = np.ones((batch, actions), dtype=bool)
    if masked:
        allowed = rng.random((batch, actions)) > 0.4
        allowed[np.arange(batch), rng.integers(0, actions, batch)] = True
    chosen = np.array([rng.choice(np.flatnonzero(row)) for row in allowed])
    onehot = np.eye(actions, dtype=np.float32)[chosen]
    mask_bias = np.where(allowed, 0.0, -1e9).astype(np.float32)

    with torch.no_grad():
        logits = mlp_forward(torch.tensor(obs.astype(np.float64)), [torch.tensor(p.astype(np.float64)) for p in actor])
        logits = logits + torch.tensor(mask_bias.astype(np.float64))
        current = (F.log_softmax(logits, dim=1) * torch.tensor(onehot.astype(np.float64))).sum(1, keepdim=True)
    old_logp = (current.numpy() + rng.normal(0, 0.15, size=(batch, 1))).astype(np.float32)
    advantage = f32((batch, 1), -2, 2)
    returns = f32((batch, 1), -1, 3)

    a_t, c_t = [leaf(p) for p in actor], [leaf(p) for p in critic]
    obs_t = torch.tensor(obs.astype(np.float64))
    logits = mlp_forward(obs_t, a_t) + torch.tensor(mask_bias.astype(np.float64))
    log_probs = F.log_softmax(logits, dim=1)
    logp = (log_probs * torch.tensor(onehot.astype(np.float64))).sum(1, keepdim=True)
    old = torch.tensor(old_logp.astype(np.float64))
    adv = torch.tensor(advantage.astype(np.float64))
    ratio = torch.exp(logp - old)
    surrogate = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv)
    policy_loss = -surrogate.mean()
    value_loss = ((mlp_forward(obs_t, c_t) - torch.tensor(returns.astype(np.float64))) ** 2).mean()
    entropy = -(torch.exp(log_probs) * log_probs).sum(1).mean()
    loss = policy_loss + PPO_VF * value_loss - PPO_ENT * entropy
    loss.backward()
    return {
        "name": name,
        "actor_sizes": actor_sizes, "critic_sizes": critic_sizes, "batch": batch,
        "clip": PPO_CLIP, "vf": PPO_VF, "ent": PPO_ENT,
        "obs": enc(obs), "action_onehot": enc(onehot), "mask_bias": enc(mask_bias),
        "old_logp": enc(old_logp), "advantage": enc(advantage), "returns": enc(returns),
        "params": [enc(p) for p in actor + critic],
        "loss": float(loss), "policy_loss": float(policy_loss), "value_loss": float(value_loss),
        "entropy": float(entropy), "kl": float((old - logp).mean()),
        "grads": [enc(t.grad.numpy()) for t in a_t + c_t],
    }


def ppo_cases():
    return [ppo_case("ppo_plain", 12, 4, 3, 8, False), ppo_case("ppo_masked", 16, 5, 4, 10, True)]


def gae_case(steps, envs, gamma, lam):
    rewards = rng.normal(size=(steps, envs))
    values = rng.normal(size=(steps + 1, envs))
    dones = (rng.random((steps, envs)) < 0.2).astype(np.float64)
    advantages, last = np.zeros((steps, envs)), np.zeros(envs)
    for t in reversed(range(steps)):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
        last = delta + gamma * lam * non_terminal * last
        advantages[t] = last
    return {"steps": steps, "envs": envs, "gamma": gamma, "lambda": lam,
            "rewards": enc(rewards), "values": enc(values), "dones": enc(dones),
            "advantages": enc(advantages), "returns": enc(advantages + values[:steps])}


def cartpole_case():
    gravity, mass_cart, mass_pole, length, force_mag, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    total_mass, pole_length = mass_cart + mass_pole, mass_pole * length
    limit = 12 * 2 * np.pi / 360
    state = [0.01, -0.02, 0.03, 0.015]
    actions = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    trajectory, terminated_at = [], None
    x, x_dot, theta, theta_dot = state
    for step, action in enumerate(actions):
        force = force_mag if action == 1 else -force_mag
        cos, sin = np.cos(theta), np.sin(theta)
        temp = (force + pole_length * theta_dot ** 2 * sin) / total_mass
        theta_acc = (gravity * sin - cos * temp) / (length * (4.0 / 3.0 - mass_pole * cos ** 2 / total_mass))
        x_acc = temp - pole_length * theta_acc * cos / total_mass
        x, x_dot = x + tau * x_dot, x_dot + tau * x_acc
        theta, theta_dot = theta + tau * theta_dot, theta_dot + tau * theta_acc
        trajectory.append([x, x_dot, theta, theta_dot])
        if x < -2.4 or x > 2.4 or theta < -limit or theta > limit:
            terminated_at = step
            break
    return {"initial": state, "actions": actions[:len(trajectory)], "trajectory": trajectory,
            "terminated_at": terminated_at}


if __name__ == "__main__":
    fixtures = {
        "torch_version": torch.__version__,
        "autograd": autograd_cases(),
        "optimizers": optimizer_cases(),
        "training": training_cases(),
        "ppo": ppo_cases(),
        "gae": [gae_case(7, 3, 0.99, 0.95), gae_case(20, 5, 0.97, 0.9)],
        "cartpole": cartpole_case(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(fixtures))
    print(f"wrote {OUT} (torch {torch.__version__})")
