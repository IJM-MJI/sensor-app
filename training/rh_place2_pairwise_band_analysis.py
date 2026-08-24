"""Train on one place-2 RH response run and test the other, both directions."""

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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rh_40_50_cross_run_spatial_analysis import build, extract
from rh_four_band_analysis import BANDS, DISPLAY, STAGES, band, score
from rh_paired_pixel_hue_analysis import endpoint_rows
from train_models import CACHE_VERSION, read_csv


GROUPS = ("rh-response-3", "rh-response-6")
FIXED_C = .1


def evaluate(x, truth, groups):
    prediction = np.zeros_like(truth)
    folds = []
    for held in GROUPS:
        train, test = groups != held, groups == held
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=FIXED_C, class_weight="balanced", max_iter=5000, random_state=42))
        model.fit(x[train], truth[train]); prediction[test] = model.predict(x[test])
        folds.append({"train_group": str(groups[train][0]), "test_group": held,
                      **score(truth[test], prediction[test])})
    metrics = score(truth, prediction); metrics["folds"] = folds
    return prediction, metrics


def plot(output, results, selected):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), constrained_layout=True)
    for axis, name in zip(axes[:2], ("whole_relative", selected)):
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
    axes[2].axhline(.85, color="crimson", linestyle="--", linewidth=1)
    axes[2].set_xticks(x, [name.replace("_control", " ctrl").replace("_relative", "")
                           for name in names], rotation=32, ha="right")
    axes[2].set_ylim(0, 1); axes[2].legend(fontsize=8)
    axes[2].set(title="Place-2 run-to-run transfer", ylabel="Score")
    fig.suptitle("RH four ranges: response3 <-> response6", fontweight="bold")
    fig.savefig(output / "rh_place2_pairwise_band_validation.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/rh_place2_pairwise_bands_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    items = [item for item in endpoint_rows(read_csv(args.cache))
             if item["group"] in GROUPS]
    summaries = extract(items, args.video_root)
    matrices, audit = build(items, summaries, STAGES)
    truth = np.asarray([band(row["reference"]) for row in audit])
    groups = np.asarray([row["group"] for row in audit])
    results, predictions = {}, []
    for name, matrix in matrices.items():
        pred, metrics = evaluate(matrix, truth, groups); results[name] = metrics
        predictions.extend({"feature_set": name, **row,
                            "reference_band": band(row["reference"]),
                            "prediction_band": value}
                           for row, value in zip(audit, pred))
    selected = max(results, key=lambda name: (
        results[name]["balanced_accuracy"], results[name]["exact_accuracy"]))
    best = results[selected]
    decision = {"selected": selected, "fixed_C": FIXED_C,
                "passes_score_only": bool(best["exact_accuracy"] >= .85
                                           and best["balanced_accuracy"] >= .85
                                           and min(best["per_band_recall"]) >= .85),
                "app_deploy": False,
                "reason": "Diagnostic two-run transfer; feature choice is not an independent third-run validation."}
    payload = {"scope": "place-2 response3/response6, train one complete run and test the other",
               "bands": dict(zip(DISPLAY, BANDS.tolist())),
               "results": results, "decision": decision}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    plot(args.output, results, selected)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
