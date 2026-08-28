"""Build row-normalized 0--1 confusion matrices for the deployed app tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "state": ROOT / "training/output/state_condition/state_metrics.json",
    "h2_a": ROOT / "training/output/h2_app_family_concentration_v3/metrics.json",
    "h2_b": ROOT / "training/output/h2_app_family_b_fixed_mask_v5/metrics.json",
    "rh": ROOT / "training/output/rh_app_confusion_v1/metrics.json",
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(matrix):
    counts = np.asarray(matrix, dtype=float)
    total = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)


def payloads():
    state = read(SOURCES["state"])["best_direct_candidate"]
    h2_a = read(SOURCES["h2_a"])["families"]["A"]["current_single_model"]
    h2_b = read(SOURCES["h2_b"])["result"]
    rh = read(SOURCES["rh"])["nominal_endpoint"]
    return [
        ("Four-state", ["Initial", "H2 only", "RH only", "Simultaneous"], state,
         "complete experiment group held out"),
        ("H2 environment A", ["0", "1–2", "2–3", "4"], h2_a,
         "complete video held out; base concentration model"),
        ("H2 environment B", ["0", "1–2", "2–3"], h2_b,
         "run3/run4 leave-one-run-out; experimental"),
        ("RH range", ["20–30", "40–50", "60–70", "80–90"], rh,
         "nominal endpoints; response3/response6 app audit"),
    ]


def score(metric, probability):
    recalls = np.diag(probability)
    return {
        "n": int(np.asarray(metric["confusion"]).sum()),
        "exact_accuracy": float(metric["exact_accuracy"] if "exact_accuracy" in metric
                                else metric["exact"]),
        "balanced_accuracy": float(metric.get("balanced_accuracy", recalls.mean())),
        "minimum_recall": float(recalls.min()),
        "per_class_recall": recalls.tolist(),
    }


def draw(axis, probability, labels, title, subtitle):
    axis.imshow(probability, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = probability[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                      color="white" if value >= .58 else "#172033",
                      fontsize=10, fontweight="bold")
    axis.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Reference class")
    axis.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "training/output/overall_probability_confusion_v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    report = {"normalization": "row; P(predicted class | reference class)", "tasks": {}}
    long_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.2), constrained_layout=True)
    for axis, (key, labels, metric, protocol) in zip(axes.flat, payloads()):
        counts = np.asarray(metric["confusion"], dtype=int)
        probability = normalize(counts)
        values = score(metric, probability)
        slug = key.lower().replace(" ", "_")
        report["tasks"][slug] = {
            "labels": labels,
            "protocol": protocol,
            "counts": counts.tolist(),
            "probability": probability.tolist(),
            **values,
        }
        for i, reference in enumerate(labels):
            for j, prediction in enumerate(labels):
                long_rows.append((slug, reference, prediction, counts[i, j],
                                  round(float(probability[i, j]), 6)))
        draw(axis, probability, labels, key,
             f"accuracy={values['exact_accuracy']:.3f} · balanced={values['balanced_accuracy']:.3f}")

    fig.suptitle("Sensor app confusion matrices — row-normalized probability (0–1)",
                 fontsize=15, fontweight="bold")
    for suffix, kwargs in (("png", {"dpi": 240}), ("svg", {}), ("pdf", {})):
        fig.savefig(args.output / f"overall_probability_confusion.{suffix}", **kwargs)
    plt.close(fig)

    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "probability_confusion.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("task", "reference", "predicted", "count", "probability_0_1"))
        writer.writerows(long_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
