"""External validation of the deployed Place-2 RH optical model on Place-1 runs.

The Place-2 model is frozen.  Two independent Place-1 rising H2O-only runs are
calibrated from their own initial frame and evaluated at user-confirmed ramp
endpoints.  Place 1 did not reliably reach RH90, so the 80--90 output band is
not scored here.
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

from rh_40_50_cross_run_spatial_analysis import extract
from train_models import CACHE_VERSION, read_csv


LEVELS = np.asarray([25, 35, 45, 55, 65, 75, 85], dtype=int)
LABELS = ("20–30", "30–40", "40–50", "50–60", "60–70", "70–80", "80–90")

# The label is the interval ending at the nominal endpoint.  For example, the
# nominal RH40 endpoint is reported as the 30--40 band.
PROTOCOL = (
    # Place-1 fast run: 20% at 3 s is the calibration state.
    ("place1-fast", "1_90_H2O_only_2_extract.mp4", 3.0, None),
    ("place1-fast", "1_90_H2O_only_2_extract.mp4", 9.0, 25),
    ("place1-fast", "1_90_H2O_only_2_extract.mp4", 15.0, 35),
    ("place1-fast", "1_90_H2O_only_2_extract.mp4", 25.0, 45),
    ("place1-fast", "1_90_H2O_only_2_extract.mp4", 35.0, 55),
    ("place1-fast", "1_90_H2O_only_2_extract.mp4", 45.0, 65),
    ("place1-fast", "1_90_H2O_only_2_extract.mp4", 72.0, 75),
    # Place-1 long is one run split at source t=180 s.  The extra clip begins
    # nine seconds before the RH70 endpoint and ends at the RH80 endpoint.
    ("place1-long", "1_90_H2O_only_extract_3min.mp4", 0.5, None),
    ("place1-long", "1_90_H2O_only_extract_3min.mp4", 25.0, 25),
    ("place1-long", "1_90_H2O_only_extract_3min.mp4", 45.0, 35),
    ("place1-long", "1_90_H2O_only_extract_3min.mp4", 90.0, 45),
    ("place1-long", "1_90_H2O_only_extract_3min.mp4", 120.0, 55),
    ("place1-long", "1_90_H2O_only_extract_extra.mp4", 9.0, 65),
    ("place1-long", "1_90_H2O_only_extract_extra.mp4", 87.0, 75),
)


def nearest_row(rows, video, seconds):
    candidates = [row for row in rows if row.get("video") == video]
    if not candidates:
        raise RuntimeError(f"No cached geometry for {video}")
    return min(candidates, key=lambda row: abs(float(row["time"]) - seconds))


def load_model(path: Path):
    text = path.read_text(encoding="utf-8")
    payload = text[text.index("{"):text.rindex("}") + 1]
    return json.loads(payload)


def control_vector(summary):
    drop = summary["controls"]["drop_lab"]
    substrate = summary["controls"]["substrate_lab"]
    if drop is None or substrate is None:
        raise RuntimeError("Missing droplet or substrate pixels")
    return np.asarray(drop, dtype=float) - np.asarray(substrate, dtype=float)


def predict(vector, model):
    prototypes = np.asarray(model["prototypes"], dtype=float)
    classes = np.asarray(model["classes"], dtype=int)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    distances = np.sqrt(np.mean(((prototypes - vector) / scale) ** 2, axis=1))
    order = np.argsort(distances)
    return int(classes[order[0]]), float(distances[order[0]]), float(
        distances[order[1]] - distances[order[0]])


def confusion(records):
    matrix = np.zeros((len(LEVELS), len(LEVELS)), dtype=int)
    lookup = {value: index for index, value in enumerate(LEVELS)}
    for row in records:
        matrix[lookup[row["reference"]], lookup[row["prediction"]]] += 1
    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float),
                           where=support > 0)
    return matrix, normalized


def draw_matrix(axis, normalized, title, xlabel):
    axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(LEVELS)):
        for column in range(len(LEVELS)):
            value = normalized[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                      fontsize=7, color="white" if value >= .65 else "#1f2937")
    axis.set_xticks(range(7), LABELS, rotation=35, ha="right")
    axis.set_yticks(range(7), LABELS)
    axis.set(xlabel=xlabel, ylabel="Place-1 endpoint reference", title=title)


def draw(normalized, candidate_normalized, output, metrics, candidate_metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.9), constrained_layout=True)
    draw_matrix(axes[0], normalized,
                f"Frozen Place-2 model\nexact={metrics['exact_accuracy']:.3f}",
                "Place-2 model prediction")
    draw_matrix(axes[1], candidate_normalized,
                f"Place-1 complete-run holdout\nexact={candidate_metrics['exact_accuracy']:.3f}",
                "Place-1 candidate prediction")
    names = ("Place-2\nexternal", "Place-1\nrun-held-out")
    exact = (metrics["exact_accuracy"], candidate_metrics["exact_accuracy"])
    adjacent = (metrics["within_one_adjacent_range"],
                candidate_metrics["within_one_adjacent_range"])
    x = np.arange(2); width = .34
    axes[2].bar(x - width / 2, exact, width, label="Exact")
    axes[2].bar(x + width / 2, adjacent, width, label="Adjacent")
    axes[2].axhline(.85, color="crimson", linestyle="--", label="0.85 target")
    axes[2].set_xticks(x, names); axes[2].set_ylim(0, 1.03)
    axes[2].set(ylabel="Score (0–1)", title="Model comparison")
    axes[2].legend()
    fig.suptitle("Place-1 H2O-only external and complete-run-held-out validation",
                 fontweight="bold")
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"rh_place1_external_validation.{suffix}", **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--model", type=Path,
                        default=Path("sensor-rh-place2-seven-band-model.js"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/rh_place1_external_validation_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    cached = read_csv(args.cache)
    items = []
    for group, video, seconds, reference in PROTOCOL:
        items.append({"group": group, "video": video, "time": seconds,
                      "stage": 25 if reference is None else reference,
                      "row": nearest_row(cached, video, seconds),
                      "calibration": reference is None,
                      "reference": reference})
    summaries = extract(items, args.video_root)
    controls = [control_vector(summary) for summary in summaries]
    baselines = {item["group"]: controls[index] for index, item in enumerate(items)
                 if item["calibration"]}
    model = load_model(args.model)
    records = []
    for item, control in zip(items, controls):
        if item["calibration"]:
            continue
        vector = control - baselines[item["group"]]
        prediction, distance, margin = predict(vector, model)
        records.append({"group": item["group"], "video": item["video"],
                        "time_s": item["time"], "reference": item["reference"],
                        "prediction": prediction, "correct": prediction == item["reference"],
                        "distance": distance, "margin": margin,
                        "delta_L": float(vector[0]), "delta_a": float(vector[1]),
                        "delta_b": float(vector[2])})

    matrix, normalized = confusion(records)
    truth = np.asarray([row["reference"] for row in records])
    prediction = np.asarray([row["prediction"] for row in records])
    truth_index = np.asarray([np.where(LEVELS == value)[0][0] for value in truth])
    pred_index = np.asarray([np.where(LEVELS == value)[0][0] for value in prediction])
    exact = float(np.mean(truth == prediction))
    adjacent = float(np.mean(np.abs(truth_index - pred_index) <= 1))
    support = matrix.sum(axis=1)
    recall = np.divide(np.diag(matrix), support, out=np.full(7, np.nan), where=support > 0)
    base_metrics = {
        "n": len(records), "exact_accuracy": exact,
        "within_one_adjacent_range": adjacent,
        "mae_percent_rh": float(np.mean(np.abs(truth - prediction))),
        "balanced_accuracy_observed_classes": float(np.nanmean(recall)),
        "per_range_recall": {label: None if np.isnan(value) else float(value)
                             for label, value in zip(LABELS, recall)},
        "confusion": matrix.tolist(),
        "row_normalized_confusion_0_to_1": normalized.tolist(),
    }

    # Train on one complete Place-1 run and predict the other. This is the
    # smallest leakage-free environment-specific candidate available here.
    vectors = np.asarray([[row["delta_L"], row["delta_a"], row["delta_b"]]
                          for row in records])
    groups = np.asarray([row["group"] for row in records])
    candidate_prediction = np.zeros_like(truth)
    for held in sorted(set(groups)):
        train = groups != held; test = ~train
        scale = np.maximum(np.std(vectors[train], axis=0), .5)
        distances = np.sqrt(np.mean(
            ((vectors[test, None] - vectors[train][None]) / scale) ** 2, axis=2))
        candidate_prediction[test] = truth[train][np.argmin(distances, axis=1)]
    candidate_records = [
        {**row, "prediction": int(value)}
        for row, value in zip(records, candidate_prediction)
    ]
    candidate_matrix, candidate_normalized = confusion(candidate_records)
    candidate_support = candidate_matrix.sum(axis=1)
    candidate_recall = np.divide(
        np.diag(candidate_matrix), candidate_support, out=np.full(7, np.nan),
        where=candidate_support > 0)
    candidate_indices = np.asarray([
        np.where(LEVELS == value)[0][0] for value in candidate_prediction])
    candidate_metrics = {
        "n": len(records),
        "exact_accuracy": float(np.mean(truth == candidate_prediction)),
        "within_one_adjacent_range": float(np.mean(
            np.abs(truth_index - candidate_indices) <= 1)),
        "mae_percent_rh": float(np.mean(np.abs(truth - candidate_prediction))),
        "balanced_accuracy_observed_classes": float(np.nanmean(candidate_recall)),
        "per_range_recall": {
            label: None if np.isnan(value) else float(value)
            for label, value in zip(LABELS, candidate_recall)},
        "confusion": candidate_matrix.tolist(),
        "row_normalized_confusion_0_to_1": candidate_normalized.tolist(),
    }
    payload = {
        "scope": "Frozen Place-2 optical model externally evaluated on two Place-1 rising runs",
        "frozen_place2_external": base_metrics,
        "place1_complete_run_holdout_candidate": candidate_metrics,
        "decision": {
            "apply_place2_model_to_place1": bool(exact >= .85 and np.nanmin(recall) >= .85),
            "deploy_place1_candidate": bool(
                candidate_metrics["exact_accuracy"] >= .85
                and np.nanmin(candidate_recall) >= .85),
            "rule": "exact >= 0.85 and every observed range recall >= 0.85",
            "note": "Place-1 RH90/80–90 is excluded because the experiment did not reliably reach RH90.",
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    draw(normalized, candidate_normalized, args.output, base_metrics, candidate_metrics)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
