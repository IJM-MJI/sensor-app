"""Validate four deployable RH ranges with complete-run holdout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rh_40_50_cross_run_spatial_analysis import build, extract
from rh_paired_pixel_hue_analysis import endpoint_rows
from train_models import CACHE_VERSION, read_csv


BANDS = np.asarray([25.0, 45.0, 65.0, 85.0])
DISPLAY = ("20-30", "40-50", "60-70", "80-90")
STAGES = np.asarray([25.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])


def band(stage):
    if stage == 25:
        return 25.0
    if stage in (40, 50):
        return 45.0
    if stage in (60, 70):
        return 65.0
    return 85.0


def group_band_weights(groups, truth, use):
    weights = np.zeros(len(truth))
    for group in sorted(set(groups[use])):
        for value in BANDS:
            selected = use & (groups == group) & (truth == value)
            if np.any(selected):
                weights[selected] = 1.0 / np.sum(selected)
    return weights


def score(truth, prediction):
    matrix = confusion_matrix(truth, prediction, labels=BANDS)
    recalls = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    indices = {value: index for index, value in enumerate(BANDS)}
    distance = np.asarray([abs(indices[a] - indices[b])
                           for a, b in zip(truth, prediction)])
    return {"exact_accuracy": float(np.mean(truth == prediction)),
            "balanced_accuracy": float(np.mean(recalls)),
            "within_one_band": float(np.mean(distance <= 1)),
            "per_band_recall": recalls.tolist(), "confusion": matrix.tolist()}


def evaluate(x, truth, groups):
    prediction = np.zeros_like(truth); choices = []
    for held in sorted(set(groups)):
        outer = groups != held; best = None
        for c_value in (.001, .003, .01, .03, .1, .3, 1.0, 3.0):
            fold_scores = []
            for inner in sorted(set(groups[outer])):
                train = outer & (groups != inner); test = outer & (groups == inner)
                weights = group_band_weights(groups, truth, train)
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=c_value, max_iter=5000, class_weight="balanced", random_state=42))
                model.fit(x[train], truth[train],
                          logisticregression__sample_weight=weights[train])
                metrics = score(truth[test], model.predict(x[test]))
                fold_scores.append((metrics["balanced_accuracy"],
                                    metrics["exact_accuracy"],
                                    metrics["within_one_band"]))
            candidate = tuple(np.mean(fold_scores, axis=0))
            if best is None or candidate > best[0]:
                best = (candidate, c_value)
        train, test = groups != held, groups == held
        weights = group_band_weights(groups, truth, train)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=best[1], max_iter=5000, class_weight="balanced", random_state=42))
        model.fit(x[train], truth[train],
                  logisticregression__sample_weight=weights[train])
        prediction[test] = model.predict(x[test])
        choices.append({"held_out_group": held, "C": best[1]})
    metrics = score(truth, prediction); metrics["outer_fold_C"] = choices
    return prediction, metrics


def plot(output, results, selected):
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), constrained_layout=True)
    candidates = ("whole_relative", selected)
    for axis, name in zip(axes[:2], candidates):
        matrix = np.asarray(results[name]["confusion"], float)
        norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        axis.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for row in range(4):
            for column in range(4):
                value = norm[row, column]
                axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                          color="white" if value > .55 else "black")
        axis.set_xticks(range(4), DISPLAY, rotation=25)
        axis.set_yticks(range(4), DISPLAY)
        axis.set(xlabel="Predicted RH range", ylabel="Reference RH range", title=name)
    names = list(results); x = np.arange(len(names)); width = .36
    axes[2].bar(x-width/2, [results[name]["exact_accuracy"] for name in names],
                width, label="Exact")
    axes[2].bar(x+width/2, [results[name]["balanced_accuracy"] for name in names],
                width, label="Balanced")
    axes[2].axhline(.85, color="crimson", linestyle="--", linewidth=1, label="0.85 target")
    axes[2].set_xticks(x, [name.replace("_control", " ctrl").replace("_relative", "")
                           for name in names], rotation=32, ha="right")
    axes[2].set_ylim(0, 1); axes[2].legend(fontsize=8)
    axes[2].set(title="Complete-run-held-out A/B", ylabel="Score")
    fig.suptitle("RH four-range held-out validation", fontweight="bold")
    fig.savefig(output / "rh_four_band_validation.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/rh_four_band_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    items = endpoint_rows(read_csv(args.cache))
    summaries = extract(items, args.video_root)
    matrices, audit = build(items, summaries, STAGES)
    truth = np.asarray([band(row["reference"]) for row in audit])
    groups = np.asarray([row["group"] for row in audit])
    results, prediction_rows = {}, []
    for name, matrix in matrices.items():
        prediction, metrics = evaluate(matrix, truth, groups); results[name] = metrics
        prediction_rows.extend({"feature_set": name, **row,
                                "reference_band": band(row["reference"]),
                                "prediction_band": value}
                               for row, value in zip(audit, prediction))
    selected = max(results, key=lambda name: (
        results[name]["balanced_accuracy"], results[name]["exact_accuracy"]))
    chosen = results[selected]
    decision = {"selected": selected, "required_per_band_recall": .85,
                "passes_score_only": bool(chosen["exact_accuracy"] >= .85
                                           and chosen["balanced_accuracy"] >= .85
                                           and min(chosen["per_band_recall"]) >= .85)}
    decision["app_deploy"] = decision["passes_score_only"]
    payload = {"scope": "30 valid rising RH-only endpoints; complete run held out",
               "bands": dict(zip(DISPLAY, BANDS.tolist())),
               "results": results, "decision": decision}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    plot(args.output, results, selected)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
