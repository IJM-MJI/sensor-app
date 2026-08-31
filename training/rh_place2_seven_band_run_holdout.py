"""Seven-band Place-2 RH complete-run-held-out model comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rh_40_50_cross_run_spatial_analysis import extract
from rh_place1_external_validation import control_vector, nearest_row
from train_models import CACHE_VERSION, read_csv


LEVELS = np.asarray([25, 35, 45, 55, 65, 75, 85], dtype=int)
LABELS = ("20–30", "30–40", "40–50", "50–60", "60–70", "70–80", "80–90")
VIDEOS = {
    "response3": "1_90_H2O_only_3(response).mp4",
    "response6": "1_90_H2O_only_6(response).mp4",
}
CALIBRATION = {"response3": .5, "response6": 2.0}

# User-confirmed tuning anchors and the later frozen-v37 unused-time validation.
# Duplicate times are removed; no frame from the held-out run trains its fold.
POINTS = {
    "response3": {
        25: (1.0, 1.5), 35: (4.0, 4.5), 45: (9.0, 10.0),
        55: (18.0, 20.0), 65: (26.5, 27.0), 75: (30.0, 31.5),
        85: (35.0, 37.0),
    },
    "response6": {
        25: (9.0, 9.5), 35: (11.5,), 45: (14.5,),
        55: (15.0, 15.5, 16.0), 65: (17.0, 17.5), 75: (19.0,),
        85: (21.5, 23.0, 24.5),
    },
}


def nearest_level(values):
    values = np.asarray(values, dtype=float)
    return LEVELS[np.argmin(np.abs(values[:, None] - LEVELS[None]), axis=1)]


def matrices(truth, prediction):
    lookup = {value: index for index, value in enumerate(LEVELS)}
    matrix = np.zeros((7, 7), dtype=int)
    for reference, predicted in zip(truth, prediction):
        matrix[lookup[int(reference)], lookup[int(predicted)]] += 1
    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float),
                           where=support > 0)
    recall = np.diag(normalized)
    indices_truth = np.asarray([lookup[int(value)] for value in truth])
    indices_prediction = np.asarray([lookup[int(value)] for value in prediction])
    return {
        "n": int(len(truth)), "exact_accuracy": float(np.mean(truth == prediction)),
        "balanced_accuracy": float(np.mean(recall)),
        "within_one_adjacent_range": float(np.mean(
            np.abs(indices_truth - indices_prediction) <= 1)),
        "mae_percent_rh": float(np.mean(np.abs(truth - prediction))),
        "per_range_recall": {label: float(value) for label, value in zip(LABELS, recall)},
        "confusion": matrix.tolist(),
        "row_normalized_confusion_0_to_1": normalized.tolist(),
    }


def fit_predict(name, train_x, train_y, test_x):
    if name == "standardized_1nn":
        scale = np.maximum(np.std(train_x, axis=0), .5)
        distance = np.sqrt(np.mean(
            ((test_x[:, None] - train_x[None]) / scale) ** 2, axis=2))
        return train_y[np.argmin(distance, axis=1)]
    if name == "ridge_ordinal":
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        return nearest_level(model.fit(train_x, train_y).predict(test_x))
    if name == "multinomial_logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=.1, class_weight="balanced", max_iter=5000, random_state=42))
        return model.fit(train_x, train_y).predict(test_x)
    raise ValueError(name)


def plot(results, selected, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
    normalized = np.asarray(results[selected]["row_normalized_confusion_0_to_1"])
    axes[0].imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(7):
        for column in range(7):
            value = normalized[row, column]
            axes[0].text(column, row, f"{value:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if value >= .65 else "#1f2937")
    axes[0].set_xticks(range(7), LABELS, rotation=35, ha="right")
    axes[0].set_yticks(range(7), LABELS)
    axes[0].set(xlabel="Predicted RH range", ylabel="Reference RH range",
                title=f"{selected} row-normalized confusion (0–1)")
    names = list(results); x = np.arange(len(names)); width = .34
    axes[1].bar(x-width/2, [results[name]["exact_accuracy"] for name in names],
                width, label="Exact")
    axes[1].bar(x+width/2, [results[name]["within_one_adjacent_range"] for name in names],
                width, label="Adjacent")
    axes[1].axhline(.85, color="crimson", linestyle="--", label="0.85 target")
    axes[1].set_xticks(x, [name.replace("_", "\n") for name in names])
    axes[1].set_ylim(0, 1.03); axes[1].legend()
    axes[1].set(ylabel="Score (0–1)", title="Complete-run-held-out candidates")
    fig.suptitle("Place-2 response3 ↔ response6 seven-band RH validation",
                 fontweight="bold")
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"rh_place2_seven_band_run_holdout.{suffix}", **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_place2_seven_band_run_holdout_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cached = read_csv(args.cache)
    items = []
    for group, video in VIDEOS.items():
        seconds = CALIBRATION[group]
        items.append({"group": group, "video": video, "time": seconds,
                      "stage": 25, "calibration": True,
                      "row": nearest_row(cached, video, seconds)})
        for level, times in POINTS[group].items():
            for seconds in times:
                items.append({"group": group, "video": video, "time": seconds,
                              "stage": level, "calibration": False,
                              "row": nearest_row(cached, video, seconds)})
    summaries = extract(items, args.video_root)
    controls = [control_vector(summary) for summary in summaries]
    baseline = {item["group"]: controls[index] for index, item in enumerate(items)
                if item["calibration"]}
    rows = []
    for item, control in zip(items, controls):
        if item["calibration"]:
            continue
        vector = control - baseline[item["group"]]
        rows.append({"group": item["group"], "video": item["video"],
                     "time_s": item["time"], "reference": item["stage"],
                     "delta_L": float(vector[0]), "delta_a": float(vector[1]),
                     "delta_b": float(vector[2])})
    x = np.asarray([[row["delta_L"], row["delta_a"], row["delta_b"]] for row in rows])
    truth = np.asarray([row["reference"] for row in rows], dtype=int)
    groups = np.asarray([row["group"] for row in rows])
    results, prediction_rows = {}, []
    for name in ("standardized_1nn", "ridge_ordinal", "multinomial_logistic"):
        prediction = np.zeros_like(truth)
        folds = []
        for held in sorted(set(groups)):
            train = groups != held; test = ~train
            prediction[test] = fit_predict(name, x[train], truth[train], x[test])
            folds.append({"held_out_run": held,
                          **matrices(truth[test], prediction[test])})
        results[name] = {**matrices(truth, prediction), "folds": folds}
        prediction_rows.extend({"model": name, **row, "prediction": int(value)}
                               for row, value in zip(rows, prediction))
    selected = max(results, key=lambda name: (
        results[name]["balanced_accuracy"], results[name]["exact_accuracy"]))
    chosen = results[selected]
    deploy = bool(chosen["exact_accuracy"] >= .85
                  and min(chosen["per_range_recall"].values()) >= .85
                  and all(fold["exact_accuracy"] >= .85 for fold in chosen["folds"]))
    payload = {
        "scope": "response3 and response6 complete-run-held-out seven-band RH",
        "features": ["drop_minus_substrate_L", "drop_minus_substrate_a",
                     "drop_minus_substrate_b"],
        "results": results,
        "decision": {"selected": selected, "deploy_to_app": deploy,
                     "rule": "overall exact, every recall, and each run fold >= 0.85",
                     "warning": "candidate comparison is diagnostic with only two independent runs"},
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader(); writer.writerows(prediction_rows)
    plot(results, selected, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
