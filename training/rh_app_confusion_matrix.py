"""Build the audited seven-band Place-2 RH app confirmation matrices.

These records come from user-reviewed app screenshots. They measure deployed
app consistency at tuning anchors, not independent held-out accuracy.
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


LEVELS = (25, 35, 45, 55, 65, 75, 85)
LABELS = ("20–30", "30–40", "40–50", "50–60", "60–70", "70–80", "80–90")

# v36 response3 screenshots, before the profile-specific ordinal correction.
RESPONSE3_PRE = (
    ("response3", 1.5, 25, 25), ("response3", 4.5, 35, 35),
    ("response3", 10.0, 45, 35), ("response3", 20.0, 55, 45),
    ("response3", 27.0, 65, 45), ("response3", 31.5, 75, 65),
    ("response3", 35.0, 85, 75),
)

# v37 response3 plus the six explicitly confirmed final response6 points.
# Unconfirmed response6 bands are deliberately not inferred.
DEPLOYED_CONFIRMATION = (
    ("response3", 1.5, 25, 25), ("response3", 4.5, 35, 35),
    ("response3", 10.0, 45, 45), ("response3", 20.0, 55, 55),
    ("response3", 27.0, 65, 65), ("response3", 31.5, 75, 75),
    ("response3", 35.0, 85, 85),
    ("response6", 9.5, 25, 25), ("response6", 15.0, 55, 55),
    ("response6", 16.0, 55, 55), ("response6", 17.0, 65, 65),
    ("response6", 21.5, 85, 85), ("response6", 24.5, 85, 85),
)


def matrix(records):
    index = {value: i for i, value in enumerate(LEVELS)}
    result = np.zeros((len(LEVELS), len(LEVELS)), dtype=int)
    for _, _, reference, prediction in records:
        result[index[reference], index[prediction]] += 1
    return result


def metrics(records, confusion):
    truth = np.asarray([row[2] for row in records])
    prediction = np.asarray([row[3] for row in records])
    support = confusion.sum(axis=1)
    recalls = np.divide(np.diag(confusion), support,
                        out=np.full(len(LEVELS), np.nan), where=support > 0)
    return {
        "n": len(records),
        "exact_accuracy": float(np.mean(truth == prediction)),
        "balanced_accuracy_observed_classes": float(np.nanmean(recalls)),
        "per_range_recall": {
            label: (None if np.isnan(value) else float(value))
            for label, value in zip(LABELS, recalls)
        },
        "within_one_adjacent_range": float(np.mean(
            np.abs(np.searchsorted(LEVELS, truth) -
                   np.searchsorted(LEVELS, prediction)) <= 1)),
        "mae_percent_rh": float(np.mean(np.abs(truth - prediction))),
        "confusion": confusion.tolist(),
    }


def write_observations(path, records):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run", "time_s", "reference_range", "predicted_range", "correct"))
        for run, seconds, reference, prediction in records:
            writer.writerow((run, seconds, LABELS[LEVELS.index(reference)],
                             LABELS[LEVELS.index(prediction)], reference == prediction))


def draw(axis, confusion, title, subtitle):
    peak = max(1, int(np.max(confusion)))
    axis.imshow(confusion, cmap="Blues", vmin=0, vmax=peak)
    for row in range(len(LEVELS)):
        for column in range(len(LEVELS)):
            value = int(confusion[row, column])
            axis.text(column, row, str(value), ha="center", va="center",
                      color="white" if value == peak else "#1f2937",
                      fontsize=10, fontweight="bold")
    axis.set_xticks(range(len(LEVELS)), LABELS, rotation=35, ha="right")
    axis.set_yticks(range(len(LEVELS)), LABELS)
    axis.set_xlabel("Predicted RH range (%)")
    axis.set_ylabel("Reference RH range (%)")
    axis.set_title(f"{title}\n{subtitle}", fontweight="bold", fontsize=11)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/rh_app_confusion_v2"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    protocols = {
        "response3_before_v37": RESPONSE3_PRE,
        "deployed_anchor_confirmation": DEPLOYED_CONFIRMATION,
    }
    report = {}
    for name, records in protocols.items():
        confusion = matrix(records)
        report[name] = metrics(records, confusion)
        write_observations(args.output / f"{name}_observations.csv", records)
    report["interpretation"] = {
        "scope": "Place-2 seven-band RH app output",
        "status": "tuned deployment-anchor confirmation",
        "independent_accuracy": False,
        "warning": "Do not report 13/13 as held-out model accuracy.",
        "next_gate": "Evaluate unused time blocks or a new independent Place-2 H2O-only run.",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    before = np.asarray(report["response3_before_v37"]["confusion"])
    after = np.asarray(report["deployed_anchor_confirmation"]["confusion"])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.7), constrained_layout=True)
    draw(axes[0], before, "Response3 before ordinal correction",
         "2/7 exact (28.6%) · app screenshots")
    draw(axes[1], after, "Current deployed anchor confirmation",
         "13/13 exact · tuned anchors, not held-out")
    fig.suptitle("Place-2 RH seven-range app confirmation",
                 fontweight="bold", fontsize=15)
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        fig.savefig(args.output / f"rh_app_confusion_matrix.{suffix}", **kwargs)
    plt.close(fig)
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
