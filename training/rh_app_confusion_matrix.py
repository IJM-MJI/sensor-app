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

# Previously unused times requested after v37 was frozen. One duplicated
# response6 15.5 s screenshot is counted once.
UNUSED_TIME_VALIDATION = (
    ("response3", 1.0, 25, 25), ("response3", 4.0, 35, 35),
    ("response3", 9.0, 45, 45), ("response3", 18.0, 55, 45),
    ("response3", 26.5, 65, 55), ("response3", 30.0, 75, 65),
    ("response3", 37.0, 85, 85),
    ("response6", 9.0, 25, 25), ("response6", 11.5, 35, 35),
    ("response6", 14.5, 45, 55), ("response6", 15.5, 55, 55),
    ("response6", 17.5, 65, 65), ("response6", 19.0, 75, 65),
    ("response6", 23.0, 85, 85),
)

ALL_APP_OBSERVATIONS = DEPLOYED_CONFIRMATION + UNUSED_TIME_VALIDATION


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
    normalized = np.divide(confusion, support[:, None],
                           out=np.zeros_like(confusion, dtype=float),
                           where=support[:, None] > 0)
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
        "row_normalized_confusion_0_to_1": normalized.tolist(),
    }


def write_observations(path, records):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run", "time_s", "reference_range", "predicted_range", "correct"))
        for run, seconds, reference, prediction in records:
            writer.writerow((run, seconds, LABELS[LEVELS.index(reference)],
                             LABELS[LEVELS.index(prediction)], reference == prediction))


def draw(axis, confusion, title, subtitle):
    support = confusion.sum(axis=1)
    normalized = np.divide(confusion, support[:, None],
                           out=np.zeros_like(confusion, dtype=float),
                           where=support[:, None] > 0)
    axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(LEVELS)):
        for column in range(len(LEVELS)):
            value = float(normalized[row, column])
            axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                      color="white" if value >= .65 else "#1f2937",
                      fontsize=9, fontweight="bold")
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
        "unused_time_validation": UNUSED_TIME_VALIDATION,
        "all_app_observations": ALL_APP_OBSERVATIONS,
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
        "warning": "Anchor confirmation is tuned; unused-time validation is the primary current estimate.",
        "next_gate": "Evaluate unused time blocks or a new independent Place-2 H2O-only run.",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    validation = np.asarray(report["unused_time_validation"]["confusion"])
    combined = np.asarray(report["all_app_observations"]["confusion"])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.7), constrained_layout=True)
    draw(axes[0], validation, "Unused-time validation",
         "9/14 exact (64.3%) · frozen v37")
    draw(axes[1], combined, "All app observations",
         "22/27 exact (81.5%) · includes tuned anchors")
    fig.suptitle("Place-2 RH seven-range row-normalized confusion (0–1)",
                 fontweight="bold", fontsize=15)
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        fig.savefig(args.output / f"rh_app_confusion_matrix.{suffix}", **kwargs)
    plt.close(fig)
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
