"""Plot the held-out effect of adding 1_80_2 RH20 as weak H2 supervision."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def selected_metrics(path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))["H2"]
    return report["models"][report["selected"]]


def recalls(confusion):
    matrix = np.asarray(confusion, dtype=float)
    return np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--angle80", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/angle80_h2_ab"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    cases = [("Existing weak runs", selected_metrics(args.baseline)),
             ("+ 1_80_2 RH20", selected_metrics(args.angle80))]
    rows = []
    for name, metrics in cases:
        row = {
            "case": name, "exact_accuracy": metrics["exact_accuracy"],
            "stage_balanced_accuracy": metrics["stage_balanced_accuracy"],
            "within_one_step": metrics["within_one_step"], "mae": metrics["mae"],
            "weak_training_frames": metrics["n_weak_training_frames"],
        }
        for level, recall in enumerate(recalls(metrics["confusion"])):
            row[f"recall_h2_{level}"] = float(recall)
        rows.append(row)

    with (args.output / "angle80_h2_ab_metrics.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    colours = ["#4C78A8", "#F58518"]
    metric_names = ["Exact", "Balanced", "Within ±1"]
    positions = np.arange(len(metric_names)); width = .34
    for index, (name, metrics) in enumerate(cases):
        values = [metrics["exact_accuracy"], metrics["stage_balanced_accuracy"],
                  metrics["within_one_step"]]
        axes[0].bar(positions + (index - .5) * width, values, width, label=name,
                    color=colours[index])
    axes[0].set_xticks(positions, metric_names); axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Held-out score"); axes[0].legend(frameon=False)
    axes[0].set_title("Overall H2 concentration performance")

    stages = np.arange(5)
    for index, (name, metrics) in enumerate(cases):
        axes[1].plot(stages, recalls(metrics["confusion"]), marker="o", linewidth=2,
                     label=name, color=colours[index])
    axes[1].set_xticks(stages, [f"{stage}%" for stage in stages]); axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Held-out recall"); axes[1].set_title("Per-stage recall")
    axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("A/B: 1_80_2 RH20 weak H2 augmentation", weight="bold")
    fig.savefig(args.output / "angle80_h2_ab.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
