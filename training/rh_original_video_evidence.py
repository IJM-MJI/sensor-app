"""Summarize leakage-safe and development-only RH evidence without mixing protocols."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "training/output/rh_original_video_evidence_v1"
LABELS = ("20–30", "30–40", "40–50", "50–60", "60–70", "70–80", "80–90")


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def matrix(value):
    """Accept both numeric arrays and the compact row strings in older metrics."""
    if value and isinstance(value[0], str):
        return np.asarray([[float(item) for item in row.split()] for row in value])
    return np.asarray(value, dtype=float)


def draw(items):
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.1), constrained_layout=True)
    for axis, item in zip(axes, items):
        values = matrix(item["metrics"]["row_normalized_confusion_0_to_1"])
        axis.imshow(values, cmap="Blues", vmin=0, vmax=1)
        for row in range(7):
            for column in range(7):
                value = values[row, column]
                axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                          fontsize=7.5, color="white" if value >= .65 else "#1f2937")
        axis.set_xticks(range(7), LABELS, rotation=38, ha="right")
        axis.set_yticks(range(7), LABELS)
        axis.set(xlabel="Predicted RH range", ylabel="Reference RH range",
                 title=(f"{item['title']}\n"
                        f"exact={item['metrics']['exact_accuracy']:.2f}, "
                        f"within-one={item['metrics']['within_one_adjacent_range']:.2f}, "
                        f"n={item['metrics']['n']}"))
    fig.suptitle("Original-video RH evidence — row-normalized confusion (0–1)",
                 fontsize=14, fontweight="bold")
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUTPUT / f"rh_original_video_evidence.{suffix}", **kwargs)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    place1 = read_json(ROOT / "training/output/rh_place1_external_validation_v1/metrics.json")
    place2 = read_json(ROOT / "training/output/rh_place2_seven_band_run_holdout_v1/metrics.json")
    burst = read_json(ROOT / "training/output/rh_place2_microburst_validation_v1/metrics.json")

    items = [
        {
            "key": "place1_external_frozen",
            "title": "Place 1 external / frozen",
            "protocol": "independent location; frozen Place-2 model",
            "metrics": place1["frozen_place2_external"],
        },
        {
            "key": "place2_complete_run_holdout",
            "title": "Place 2 / complete-run holdout",
            "protocol": "response3 and response6 held out by complete run; single frame",
            "metrics": place2["results"][place2["decision"]["selected"]],
        },
        {
            "key": "place2_burst3_development",
            "title": "Place 2 / 3-frame development",
            "protocol": "same-run selected points; 30–40 absent; not independent accuracy",
            "metrics": burst["results"]["burst3"],
        },
    ]
    draw(items)

    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "evaluation", "protocol", "n", "exact_accuracy_0_to_1",
            "within_one_accuracy_0_to_1", "mae_percent_rh", "independent_accuracy_claim"))
        writer.writeheader()
        for item in items:
            metrics = item["metrics"]
            writer.writerow({
                "evaluation": item["key"],
                "protocol": item["protocol"],
                "n": metrics["n"],
                "exact_accuracy_0_to_1": f"{metrics['exact_accuracy']:.3f}",
                "within_one_accuracy_0_to_1": f"{metrics['within_one_adjacent_range']:.3f}",
                "mae_percent_rh": f"{metrics['mae_percent_rh']:.2f}",
                "independent_accuracy_claim": "no" if "development" in item["key"] else "yes",
            })

    payload = {
        "matrix_scale": "row-normalized 0-to-1",
        "labels": LABELS,
        "evaluations": {
            item["key"]: {
                "protocol": item["protocol"],
                "n": item["metrics"]["n"],
                "exact_accuracy": item["metrics"]["exact_accuracy"],
                "within_one_adjacent_range": item["metrics"]["within_one_adjacent_range"],
                "mae_percent_rh": item["metrics"]["mae_percent_rh"],
                "row_normalized_confusion_0_to_1": matrix(
                    item["metrics"]["row_normalized_confusion_0_to_1"]).tolist(),
            } for item in items
        },
        "decision": {
            "freeze_rh_accuracy": False,
            "reason": "Independent Place-2 exact accuracy is 0.444; the 1.000 burst result has n=8, omits 30–40, and is same-run development evidence.",
            "monitor_recapture_included": False,
        },
    }
    with (OUTPUT / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
