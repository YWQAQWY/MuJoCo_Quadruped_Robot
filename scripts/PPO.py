"""PPO-Clip for bounded continuous control.

The policy is a tanh-squashed diagonal Gaussian, so sampled actions and the
log-probabilities used by PPO describe the same values applied by the env.
"""
from __future__ import annotations

import numpy as np
import torch


def _init_layer(layer: torch.nn.Linear, gain: float) -> torch.nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, gain)
    torch.nn.init.zeros_(layer.bias)
    return layer


def _activation(name: str):
    activations = {"relu": torch.nn.ReLU, "elu": torch.nn.ELU, "tanh": torch.nn.Tanh}
    if name not in activations:
        raise ValueError(f"不支持的 activation={name!r}，可选: {sorted(activations)}")
    return activations[name]


def _mlp(input_dim, hidden_dims, output_dim, activation, hidden_gain, output_gain):
    layers = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend([_init_layer(torch.nn.Linear(previous, width), hidden_gain),
                       _activation(activation)()])
        previous = width
    layers.append(_init_layer(torch.nn.Linear(previous, output_dim), output_gain))
    return torch.nn.Sequential(*layers)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim, activation="elu",
                 hidden_gain=np.sqrt(2.0), output_gain=0.01, initial_log_std=-1.0):
        super().__init__()
        self.network = _mlp(state_dim, hidden_dims, action_dim, activation,
                            hidden_gain, output_gain)
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), float(initial_log_std)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, activation="elu",
                 hidden_gain=np.sqrt(2.0), output_gain=1.0):
        super().__init__()
        self.network = _mlp(state_dim, hidden_dims, 1, activation, hidden_gain, output_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    episode_ends: torch.Tensor,
    gamma: float,
    lmbda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE without leaking across resets.

    Time-limit truncation bootstraps from next_values, while both termination
    and truncation stop the reverse GAE recursion at the episode boundary.
    """
    deltas = rewards + gamma * next_values * (1.0 - terminated) - values
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), device=rewards.device)
    for t in range(rewards.shape[0] - 1, -1, -1):
        gae = deltas[t] + gamma * lmbda * (1.0 - episode_ends[t]) * gae
        advantages[t] = gae
    return advantages, advantages + values


class PPO:
    def __init__(
        self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
        epoch, eps, batch_size, actor_gamma, critic_gamma, device,
        entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
        target_kl=0.02, value_clip=0.2, hidden_dims=None, activation="elu",
        hidden_init_gain=np.sqrt(2.0), actor_output_gain=0.01,
        critic_output_gain=1.0, initial_log_std=-1.0,
        log_std_bounds=(-5.0, 2.0), numerical_epsilon=1e-6,
    ):
        hidden_dims = list(hidden_dims or [hidden_dim, hidden_dim])
        self.actor_net = PolicyNet(state_dim, hidden_dims, action_dim, activation,
                                   hidden_init_gain, actor_output_gain, initial_log_std).to(device)
        self.critic_net = ValueNet(state_dim, hidden_dims, activation,
                                   hidden_init_gain, critic_output_gain).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.actor_gamma = actor_gamma
        self.critic_gamma = critic_gamma
        self.lmbda = lmbda
        self.epoch = epoch
        self.eps = eps
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.value_clip = value_clip
        self.log_std_bounds = tuple(log_std_bounds)
        self.numerical_epsilon = numerical_epsilon
        self.device = torch.device(device)

    def _distribution(self, states: torch.Tensor):
        mu = self.actor_net(states)
        log_std = self.actor_net.log_std.clamp(*self.log_std_bounds)
        return torch.distributions.Normal(mu, log_std.exp())

    def _squashed_log_prob(self, dist, raw_action: torch.Tensor, action: torch.Tensor):
        correction = torch.log(1.0 - action.square() + self.numerical_epsilon)
        return (dist.log_prob(raw_action) - correction).sum(dim=-1)

    def take_action(self, state, deterministic=False):
        """Return a bounded action in [-1, 1]."""
        with torch.no_grad():
            state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                                      device=self.device).unsqueeze(0)
            dist = self._distribution(state_t)
            raw_action = dist.mean if deterministic else dist.sample()
            return torch.tanh(raw_action).squeeze(0).cpu().numpy()

    def update(self, transition_dict):
        states = torch.as_tensor(np.asarray(transition_dict["states"]), dtype=torch.float32,
                                 device=self.device)
        actions = torch.as_tensor(np.asarray(transition_dict["actions"]), dtype=torch.float32,
                                  device=self.device)
        rewards = torch.as_tensor(transition_dict["rewards"], dtype=torch.float32,
                                  device=self.device)
        next_states = torch.as_tensor(np.asarray(transition_dict["next_states"]),
                                      dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(transition_dict["terminated"], dtype=torch.float32,
                                     device=self.device)
        episode_ends = torch.as_tensor(transition_dict["episode_ends"], dtype=torch.float32,
                                      device=self.device)

        with torch.no_grad():
            old_values = self.critic_net(states).squeeze(-1)
            next_values = self.critic_net(next_states).squeeze(-1)
            advantages, returns = compute_gae(
                rewards, old_values, next_values, terminated, episode_ends,
                self.actor_gamma, self.lmbda,
            )
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + self.numerical_epsilon
            )
            safe_actions = actions.clamp(-1.0 + self.numerical_epsilon,
                                         1.0 - self.numerical_epsilon)
            raw_actions = torch.atanh(safe_actions)
            old_dist = self._distribution(states)
            old_log_probs = self._squashed_log_prob(old_dist, raw_actions, safe_actions)

        metrics = {"policy_loss": [], "value_loss": [], "entropy": [],
                   "approx_kl": [], "clip_fraction": []}
        n = states.shape[0]
        stop_early = False
        for _ in range(self.epoch):
            for idx in torch.randperm(n, device=self.device).split(self.batch_size):
                dist = self._distribution(states[idx])
                log_probs = self._squashed_log_prob(dist, raw_actions[idx], safe_actions[idx])
                log_ratio = log_probs - old_log_probs[idx]
                ratio = log_ratio.exp()
                surr1 = ratio * advantages[idx]
                surr2 = ratio.clamp(1.0 - self.eps, 1.0 + self.eps) * advantages[idx]
                entropy_raw_action = dist.rsample()
                entropy_action = torch.tanh(entropy_raw_action)
                squashed_entropy = -self._squashed_log_prob(
                    dist, entropy_raw_action, entropy_action
                ).mean()
                actor_loss = (-torch.min(surr1, surr2).mean()
                              - self.entropy_coef * squashed_entropy)

                values = self.critic_net(states[idx]).squeeze(-1)
                clipped_values = old_values[idx] + (values - old_values[idx]).clamp(
                    -self.value_clip, self.value_clip
                )
                value_loss = 0.5 * torch.maximum(
                    (values - returns[idx]).square(),
                    (clipped_values - returns[idx]).square(),
                ).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.eps).float().mean()
                for key, value in (
                    ("policy_loss", actor_loss), ("value_loss", value_loss),
                    ("entropy", squashed_entropy), ("approx_kl", approx_kl),
                    ("clip_fraction", clip_fraction),
                ):
                    metrics[key].append(float(value.detach().cpu()))
                if self.target_kl and approx_kl > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        with torch.no_grad():
            prediction = self.critic_net(states).squeeze(-1)
            return_var = torch.var(returns, unbiased=False)
            explained_var = 1.0 - torch.var(returns - prediction, unbiased=False) / (
                return_var + self.numerical_epsilon
            )
        result = {key: float(np.mean(values)) for key, values in metrics.items()}
        result["explained_var"] = float(explained_var.cpu())
        result["action_std"] = float(self.actor_net.log_std.exp().mean().detach().cpu())
        result["early_stop"] = float(stop_early)
        return result
