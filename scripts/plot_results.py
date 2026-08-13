"""绘制训练曲线（支持多个 run 对比，例如你的实现 vs 参考答案）。

用法：
    python scripts/plot_results.py --logs logs/run_xxx --output logs/run_xxx/curve.png
    python scripts/plot_results.py --logs logs/my_run logs/ref_run --labels 我的实现 参考答案
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def parse_args():
    p = argparse.ArgumentParser(description="绘制 PPO 训练曲线")
    p.add_argument("--logs", nargs="+", required=True, help="一个或多个 logs/<run> 目录")
    p.add_argument("--labels", nargs="+", default=None, help="对应每个 run 的图例名")
    p.add_argument("--window", type=int, default=50, help="滑动平均窗口（迭代数）")
    p.add_argument("--output", type=str, default=None, help="默认保存到第一个 run 目录下")
    return p.parse_args()


def main():
    args = parse_args()
    labels = args.labels or [Path(d).name for d in args.logs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (log_dir, label) in enumerate(zip(args.logs, labels)):
        csv_path = Path(log_dir) / "progress.csv"
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        iters = data["iteration"]
        color = colors[i % len(colors)]

        axes[0, 0].plot(iters, rolling_mean(data["mean_ep_reward"], args.window),
                        label=label, color=color)
        axes[0, 1].plot(iters, rolling_mean(data["mean_ep_len"], args.window),
                        label=label, color=color)
        axes[1, 0].plot(iters, rolling_mean(data["value_loss"], args.window),
                        label=label, color=color)
        axes[1, 1].plot(iters, rolling_mean(data["explained_var"], args.window),
                        label=label, color=color)

    axes[0, 0].set_title(f"mean episode reward (window={args.window})")
    axes[0, 0].set_xlabel("iteration")
    axes[0, 1].set_title("mean episode length (steps)")
    axes[0, 1].set_xlabel("iteration")
    axes[1, 0].set_title("value loss")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 1].set_title("explained variance")
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].set_ylim(-1, 1.1)

    for ax in axes.flat:
        ax.legend()
        ax.grid(alpha=0.3)

    out = Path(args.output) if args.output else Path(args.logs[0]) / "progress.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"曲线已保存: {out}")


if __name__ == "__main__":
    main()
