"""Plot original versus cropped RH20 weak-supervision H2 performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def selected(path):
    report = json.loads(path.read_text(encoding="utf-8"))["H2"]
    return report["models"][report["selected"]]


def recall(metrics):
    matrix = np.asarray(metrics["confusion"], dtype=float)
    return np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--cropped-optical", type=Path, required=True)
    parser.add_argument("--optical-path", type=Path,
                        help="optional within-run isotonic optical-path candidate")
    parser.add_argument("--optical-path-label", default="Optical path, weight 0.05")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = [("Original RH20 ramp", selected(args.original)),
             ("Cropped, unforced optical max", selected(args.cropped_optical))]
    if args.optical_path:
        cases.append((args.optical_path_label, selected(args.optical_path)))
    colours = ["#4C78A8", "#F58518", "#54A24B"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    metrics_names = ["Exact", "Balanced", "Within ±1"]
    x = np.arange(3); width = .8 / len(cases)
    for index, (name, metrics) in enumerate(cases):
        values = [metrics["exact_accuracy"], metrics["stage_balanced_accuracy"],
                  metrics["within_one_step"]]
        axes[0].bar(x + (index - (len(cases) - 1) / 2) * width, values, width,
                    color=colours[index], label=name)
    axes[0].set_xticks(x, metrics_names); axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Held-out score"); axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("All H2 stages")
    for index, (name, metrics) in enumerate(cases):
        axes[1].plot(range(5), recall(metrics), marker="o", linewidth=2,
                     color=colours[index], label=name)
    axes[1].set_xticks(range(5), [f"{i}%" for i in range(5)])
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("Held-out recall")
    axes[1].set_title("Per-stage effect"); axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("RH20 crop replacement: H2 concentration A/B", weight="bold")
    fig.savefig(args.output, dpi=300); plt.close(fig)


if __name__ == "__main__":
    main()
