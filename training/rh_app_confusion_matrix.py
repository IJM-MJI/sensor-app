"""Build audited RH range confusion matrices from the supplied app screenshots.

The primary matrix retains nominal endpoint times.  A secondary diagnostic
matrix uses transition-safe optical anchors selected after trajectory review;
it must not be reported as independent validation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LEVELS = (25, 45, 65, 85)
LABELS = ("20–30", "40–50", "60–70", "80–90")

# Predictions are the RH endpoint-shadow outputs visible in the supplied app
# screenshots, with the deployed v7 low-chroma guard replayed for response3 2 s.
NOMINAL = (
    ("response3", 2.0, 25, 25), ("response3", 3.0, 25, 25),
    ("response3", 5.0, 45, 45), ("response3", 7.0, 45, 65),
    ("response3", 11.0, 65, 65), ("response3", 25.0, 65, 65),
    ("response3", 28.0, 85, 85), ("response3", 38.0, 85, 85),
    ("response6", 7.0, 25, 25), ("response6", 10.0, 25, 25),
    ("response6", 13.0, 45, 45), ("response6", 14.0, 45, 45),
    ("response6", 16.0, 65, 45), ("response6", 18.0, 65, 65),
    ("response6", 20.0, 85, 85), ("response6", 32.0, 85, 85),
)

TRANSITION_SAFE = (
    ("response3", 2.0, 25, 25), ("response3", 3.0, 25, 25),
    ("response3", 5.0, 45, 45), ("response3", 6.5, 45, 45),
    ("response3", 11.0, 65, 65), ("response3", 25.0, 65, 65),
    ("response3", 28.0, 85, 85), ("response3", 38.0, 85, 85),
    ("response6", 7.0, 25, 25), ("response6", 10.0, 25, 25),
    ("response6", 13.0, 45, 45), ("response6", 14.0, 45, 45),
    ("response6", 17.0, 65, 65), ("response6", 18.0, 65, 65),
    ("response6", 19.0, 85, 85), ("response6", 32.0, 85, 85),
)


def matrix(records):
    index = {value: i for i, value in enumerate(LEVELS)}
    result = np.zeros((4, 4), dtype=int)
    for _, _, reference, prediction in records:
        result[index[reference], index[prediction]] += 1
    return result


def metrics(records, confusion):
    truth = np.asarray([row[2] for row in records])
    prediction = np.asarray([row[3] for row in records])
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    return {
        "n": len(records),
        "exact_accuracy": float(np.mean(truth == prediction)),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_range_recall": {label: float(value) for label, value in zip(LABELS, recalls)},
        "within_one_range": float(np.mean(np.abs(np.searchsorted(LEVELS, truth) -
                                                   np.searchsorted(LEVELS, prediction)) <= 1)),
        "mae_percent_rh": float(np.mean(np.abs(truth - prediction))),
        "confusion": confusion.tolist(),
    }


def draw(ax, confusion, title, subtitle):
    image = ax.imshow(confusion, cmap="Blues", vmin=0, vmax=4)
    for row in range(4):
        for column in range(4):
            value = int(confusion[row, column])
            ax.text(column, row, str(value), ha="center", va="center",
                    color="white" if value >= 3 else "#1f2937", fontsize=13,
                    fontweight="bold")
    ax.set_xticks(range(4), LABELS, rotation=25, ha="right")
    ax.set_yticks(range(4), LABELS)
    ax.set_xlabel("Predicted RH range (%)")
    ax.set_ylabel("Reference RH range (%)")
    ax.set_title(f"{title}\n{subtitle}", fontweight="bold", fontsize=12, pad=10)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/rh_app_confusion_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    protocols = {"nominal_endpoint": NOMINAL, "transition_safe": TRANSITION_SAFE}
    report = {}
    for name, records in protocols.items():
        confusion = matrix(records)
        report[name] = metrics(records, confusion)
        with (args.output / f"{name}_observations.csv").open(
                "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(("run", "time_s", "reference_midpoint", "prediction_midpoint",
                             "reference_range", "prediction_range", "correct"))
            for run, seconds, reference, prediction in records:
                writer.writerow((run, seconds, reference, prediction,
                                 LABELS[LEVELS.index(reference)],
                                 LABELS[LEVELS.index(prediction)], reference == prediction))

    report["interpretation"] = {
        "primary": "nominal_endpoint",
        "secondary": "transition_safe",
        "warning": "Transition-safe anchors were chosen after trajectory review and are not independent validation.",
        "scope": "RH range quantitation shadow output; four-state classification is excluded.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), constrained_layout=True)
    draw(axes[0], np.asarray(report["nominal_endpoint"]["confusion"]),
         "Nominal endpoints", "Accuracy 87.5% · macro recall 87.5% · n=16")
    draw(axes[1], np.asarray(report["transition_safe"]["confusion"]),
         "Transition-safe anchors", "Diagnostic 100% · post-reviewed · n=16")
    fig.suptitle("Place-2 RH range quantitation", fontweight="bold", fontsize=15)
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        fig.savefig(args.output / f"rh_app_confusion_matrix.{suffix}", **kwargs)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
