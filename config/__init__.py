"""配置加载：config/ 下的 yaml 是项目所有超参数的唯一来源。"""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent


def load(name: str) -> dict:
    """加载 config/<name>.yaml 并返回字典。"""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
