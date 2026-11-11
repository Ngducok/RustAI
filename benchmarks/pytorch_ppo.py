"""PPO on CartPole in PyTorch with the same algorithm and hyperparameters as `rt-bench --bin ppo`.

Environments are plain Python objects stepped in a loop; the network math runs on one CPU thread.
Usage: python pytorch_ppo.py
"""
import statistics
import time

import numpy as np
import torch
import torch.nn as nn

TARGET, WINDOW, MAX_STEPS, SEEDS = 195.0, 3, 400_000, (1, 2, 3)
N_ENVS, N_STEPS, EPOCHS, MINIBATCHES = 8, 128, 4, 4
GAMMA, LAMBDA, CLIP, VF, ENT, MAX_NORM, LR = 0.99, 0.95, 0.2, 0.5, 0.01, 0.5, 1e-3


class CartPole:
    """The dynamics of Gym's CartPole-v1, in float64."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.state, self.steps = np.zeros(4), 0

    def reset(self):
        self.state, self.steps = self.rng.uniform(-0.05, 0.05, size=4), 0
        return self.state.astype(np.float32)

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = 10.0 if action == 1 else -10.0
        cos, sin = np.cos(theta), np.sin(theta)
        temp = (force + 0.05 * theta_dot ** 2 * sin) / 1.1
        theta_acc = (9.8 * sin - cos * temp) / (0.5 * (4.0 / 3.0 - 0.1 * cos ** 2 / 1.1))
        x_acc = temp - 0.05 * theta_acc * cos / 1.1
        self.state = np.array([x + 0.02 * x_dot, x_dot + 0.02 * x_acc,
                               theta + 0.02 * theta_dot, theta_dot + 0.02 * theta_acc])
        self.steps += 1
        limit = 12 * 2 * np.pi / 360
        terminated = abs(self.state[0]) > 2.4 or abs(self.state[2]) > limit
        return self.state.astype(np.float32), 1.0, terminated or self.steps >= 500


def mlp(out):
    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, out))


def run(seed):
    torch.manual_seed(seed)
    envs = [CartPole(seed * 100 + i) for i in range(N_ENVS)]
    obs = np.stack([e.reset() for e in envs])
    actor, critic = mlp(2), mlp(1)
    optimizer = torch.optim.AdamW(list(actor.parameters()) + list(critic.parameters()), lr=LR, weight_decay=0.0)
    returns_so_far, lengths, recent, steps_done = np.zeros(N_ENVS), np.zeros(N_ENVS), [], 0
    start = time.perf_counter()
    while steps_done < MAX_STEPS:
        for group in optimizer.param_groups:
            group["lr"] = LR * (1 - steps_done / MAX_STEPS)
        storage = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
        finished = []
        for _ in range(N_STEPS):
            obs_t = torch.from_numpy(obs.copy())
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=actor(obs_t))
                action = dist.sample()
                logp, value = dist.log_prob(action), critic(obs_t).squeeze(-1)
            rewards, dones = np.zeros(N_ENVS, dtype=np.float32), np.zeros(N_ENVS, dtype=np.float32)
            for i, env in enumerate(envs):
                o, rewards[i], done = env.step(int(action[i]))
                returns_so_far[i] += rewards[i]
                if done:
                    dones[i] = 1.0
                    finished.append(returns_so_far[i])
                    returns_so_far[i] = 0.0
                    o = env.reset()
                obs[i] = o
            for key, v in zip(storage, (obs_t, action, logp, value, torch.from_numpy(rewards), torch.from_numpy(dones))):
                storage[key].append(v.clone() if torch.is_tensor(v) else v)
        with torch.no_grad():
            last_value = critic(torch.from_numpy(obs)).squeeze(-1)
        values = torch.stack(storage["val"] + [last_value])
        rewards, dones = torch.stack(storage["rew"]), torch.stack(storage["done"])
        advantages, last = torch.zeros_like(rewards), torch.zeros(N_ENVS)
        for t in reversed(range(N_STEPS)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + GAMMA * values[t + 1] * non_terminal - values[t]
            last = delta + GAMMA * LAMBDA * non_terminal * last
            advantages[t] = last
        flat = {
            "obs": torch.stack(storage["obs"]).reshape(-1, 4), "act": torch.stack(storage["act"]).reshape(-1),
            "logp": torch.stack(storage["logp"]).reshape(-1), "adv": advantages.reshape(-1),
            "ret": (advantages + values[:-1]).reshape(-1),
        }
        samples = N_STEPS * N_ENVS
        size = samples // MINIBATCHES
        for _ in range(EPOCHS):
            order = torch.randperm(samples)
            for k in range(MINIBATCHES):
                idx = order[k * size:(k + 1) * size]
                adv = flat["adv"][idx]
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
                dist = torch.distributions.Categorical(logits=actor(flat["obs"][idx]))
                logp = dist.log_prob(flat["act"][idx])
                ratio = torch.exp(logp - flat["logp"][idx])
                policy_loss = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv).mean()
                value_loss = ((critic(flat["obs"][idx]).squeeze(-1) - flat["ret"][idx]) ** 2).mean()
                loss = policy_loss + VF * value_loss - ENT * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), MAX_NORM)
                optimizer.step()
        steps_done += samples
        recent.append((sum(finished), len(finished)))
        tail = recent[-WINDOW:]
        episodes = sum(c for _, c in tail)
        if len(recent) >= WINDOW and episodes and sum(s for s, _ in tail) / episodes >= TARGET:
            break
    return time.perf_counter() - start, steps_done


def main():
    torch.set_num_threads(1)
    print(f"torch {torch.__version__}, 1 thread; solved when the mean return of the last {WINDOW} updates >= {TARGET}")
    times = []
    for seed in SEEDS:
        seconds, steps = run(seed)
        print(f"seed {seed}: {seconds:6.2f} s, {steps:7d} env steps, {steps / seconds:8.0f} steps/s")
        times.append(seconds)
    print(f"median time to solve: {statistics.median(times):.2f} s")


if __name__ == "__main__":
    main()
