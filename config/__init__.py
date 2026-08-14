"""配置加载：config/ 下的 yaml 是项目所有超参数的唯一来源。"""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CONFIG_DIR.parent


def _require(config: dict, path: str):
    value = config
    for key in path.split("."):
        if key not in value:
            raise ValueError(f"配置缺少必填项: {path}")
        value = value[key]
    return value


def _validate(name: str, config: dict) -> None:
    required = {
        "train": ["ppo.hidden_dims", "ppo.activation", "training.iterations",
                  "evaluation.commands", "logging.print_every", "curriculum.enabled"],
        "env": ["simulation.decimation", "simulation.episode_length", "rewards.weights",
                "domain_randomization.init"],
        "robot": ["xml_path", "default_dof_pos", "dof_lower", "dof_upper", "leg_names"],
        "pd": ["kp", "kd", "torque_limit", "action_scale"],
        "play": ["duration", "fps", "video", "command_script", "keyboard.keys"],
        "plot": ["window", "figure_size", "dpi"],
    }
    for path in required.get(name, []):
        _require(config, path)

    if name == "train" and config["ppo"]["activation"] not in {"relu", "elu", "tanh"}:
        raise ValueError("ppo.activation 只能是 relu / elu / tanh")
    if name == "robot":
        joints = len(config["default_dof_pos"])
        per_leg = len(config["dof_lower"])
        if per_leg == 0 or joints != per_leg * len(config["leg_names"]):
            raise ValueError("default_dof_pos 数量必须等于 每腿关节数 × leg_names 数量")


def load(name: str) -> dict:
    """加载 config/<name>.yaml 并返回字典。"""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    _validate(name, config)
    return config


def load_all() -> dict[str, dict]:
    """加载并校验 config/ 下全部 YAML，便于保存完整实验配置。"""
    return {path.stem: load(path.stem) for path in sorted(CONFIG_DIR.glob("*.yaml"))}
