"""Leakage-safe pairwise ensemble of the two RH20 H2 weak-label models."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


LEVELS = [0, 1, 2, 3, 4]


def read_selected(path: Path, model: str):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["task"] == "H2" and row["protocol"] == "video_holdout"
                and row["model"] == model]
    return rows


def align_rows(baseline, optical):
    """Align equal-rate exports despite sub-frame timestamp rounding changes."""
    by_video_a, by_video_b = {}, {}
    for row in baseline: by_video_a.setdefault(row["video"], []).append(row)
    for row in optical: by_video_b.setdefault(row["video"], []).append(row)
    aligned = []
    for video in sorted(set(by_video_a) & set(by_video_b)):
        left = sorted(by_video_a[video], key=lambda row: float(row["time"]))
        right = sorted(by_video_b[video], key=lambda row: float(row["time"]))
        if len(left) != len(right):
            raise ValueError(f"Frame count differs for {video}: {len(left)} vs {len(right)}")
        aligned.extend(zip(left, right))
    return aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    baseline = read_selected(args.baseline, "ridge_flexible_rounded")
    optical = read_selected(args.optical, "ridge_rounded")
    aligned = align_rows(baseline, optical)
    rows = []
    groups = sorted({base["group"] for base, _ in aligned})
    for held_out in groups:
        train_pairs = [(base, candidate) for base, candidate in aligned
                       if base["group"] != held_out]
        lookup = {}
        for pair in {(int(float(base["prediction"])), int(float(candidate["prediction"])))
                     for base, candidate in train_pairs}:
            truth = [int(float(base["reference"])) for base, candidate in train_pairs
                     if (int(float(base["prediction"])),
                         int(float(candidate["prediction"]))) == pair]
            lookup[pair] = Counter(truth).most_common(1)[0][0]
        for base, candidate in aligned:
            if base["group"] != held_out:
                continue
            pair = (int(float(base["prediction"])), int(float(candidate["prediction"])))
            prediction = lookup.get(pair, pair[0])
            rows.append({
                "video": base["video"], "group": held_out, "time": float(base["time"]),
                "reference": int(float(base["reference"])),
                "baseline_prediction": pair[0], "optical_prediction": pair[1],
                "ensemble_prediction": prediction,
            })
    truth = np.asarray([row["reference"] for row in rows])
    prediction = np.asarray([row["ensemble_prediction"] for row in rows])
    cm = confusion_matrix(truth, prediction, labels=LEVELS)
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    metrics = {
        "protocol": "leave-one-video-out pair lookup; no held-out reference used",
        "n_frames": len(rows), "exact_accuracy": float(np.mean(truth == prediction)),
        "within_one_step": float(np.mean(np.abs(truth - prediction) <= 1)),
        "mae": float(np.mean(np.abs(truth - prediction))),
        "stage_balanced_accuracy": float(np.mean(recall)),
        "recall": recall.tolist(), "confusion": cm.tolist(),
    }
    with (args.output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(5.4, 4.5), constrained_layout=True)
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for i in range(5):
        for j in range(5):
            axis.text(j, i, f"{normalized[i,j]:.2f}\n(n={cm[i,j]})", ha="center", va="center",
                      fontsize=7, color="white" if normalized[i,j] > .55 else "black")
    axis.set_xticks(range(5), LEVELS); axis.set_yticks(range(5), LEVELS)
    axis.set(xlabel="Predicted H2 (%)", ylabel="Reference H2 (%)",
             title=f"LOVO pair ensemble — exact {metrics['exact_accuracy']:.3f}")
    fig.savefig(args.output / "confusion.png", dpi=300); plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
