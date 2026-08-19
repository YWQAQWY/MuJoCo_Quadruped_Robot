"""Behavior-clone the PPO actor from the MuJoCo-validated DogML dataset."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load, load_all  # noqa: E402
from env.quadruped_env import QuadrupedEnv  # noqa: E402
from PPO import PPO  # noqa: E402
from scripts.train_ppo import load_state_with_input_expansion  # noqa: E402


def main():
    cfg = load("expert")
    bc = cfg["behavior_cloning"]
    np.random.seed(bc["seed"])
    torch.manual_seed(bc["seed"])
    data = np.load(ROOT / cfg["output_path"])
    states, actions = data["states"], data["actions"]
    if states.ndim != 2 or states.shape[1] != 54 or actions.shape != (len(states), 12):
        raise ValueError(f"专家数据维度错误: states={states.shape}, actions={actions.shape}")
    if len(states) < bc["minimum_training_samples"]:
        raise RuntimeError(f"专家样本仅 {len(states)}，低于发布门槛 "
                           f"{bc['minimum_training_samples']}；拒绝行为克隆")
    train_cfg = load("train")
    p = train_cfg["ppo"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = QuadrupedEnv(add_noise=False, randomize=False)
    agent = PPO(env.obs_dim, p["hidden_dims"][0], env.act_dim, p["actor_lr"], p["critic_lr"],
                p["lmbda"], p["epochs"], p["eps"], p["batch_size"], p["gamma"], p["gamma"],
                device, hidden_dims=p["hidden_dims"], activation=p["activation"],
                entropy_coef=p["entropy_coef"], value_coef=p["value_coef"],
                max_grad_norm=p["max_grad_norm"], target_kl=p["target_kl"],
                value_clip=p["value_clip"], initial_log_std=p["initial_log_std"],
                log_std_bounds=p["log_std_bounds"], numerical_epsilon=p["numerical_epsilon"])
    source = torch.load(ROOT / bc["init_checkpoint"], map_location=device, weights_only=False)
    load_state_with_input_expansion(agent.actor_net, source["actor"])
    load_state_with_input_expansion(agent.critic_net, source["critic"])
    order = np.random.permutation(len(states))
    split = int(len(states) * (1.0 - bc["validation_fraction"]))
    train_idx, val_idx = order[:split], order[split:]
    optimizer = torch.optim.AdamW(agent.actor_net.parameters(), lr=bc["learning_rate"],
                                  weight_decay=bc["weight_decay"])
    batch_size = int(bc["batch_size"])
    for epoch in range(int(bc["epochs"])):
        np.random.shuffle(train_idx)
        losses = []
        agent.actor_net.train()
        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[start:start + batch_size]
            x = torch.as_tensor(states[idx], device=device)
            y = torch.as_tensor(actions[idx], device=device)
            loss = torch.nn.functional.mse_loss(torch.tanh(agent.actor_net(x)), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor_net.parameters(), bc["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        agent.actor_net.eval()
        with torch.no_grad():
            x = torch.as_tensor(states[val_idx], device=device)
            y = torch.as_tensor(actions[val_idx], device=device)
            val_loss = float(torch.nn.functional.mse_loss(
                torch.tanh(agent.actor_net(x)), y).cpu())
        print(f"BC epoch {epoch + 1:3d}/{bc['epochs']} train={np.mean(losses):.6f} val={val_loss:.6f}")
    output = ROOT / bc["output_checkpoint"]
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = copy.deepcopy(source)
    checkpoint["actor"] = agent.actor_net.state_dict()
    checkpoint.pop("actor_optimizer", None)
    checkpoint["expert_pretraining"] = {"dataset": cfg["output_path"],
                                         "samples": len(states), "validation_loss": val_loss}
    checkpoint["config"] = load_all()
    torch.save(checkpoint, output)
    print(f"行为克隆模型已保存: {output}  validation MSE={val_loss:.6f}")


if __name__ == "__main__":
    main()
