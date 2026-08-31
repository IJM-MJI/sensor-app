"""Profile-specific Place-2 RH models trained from stable optical windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rh_40_50_cross_run_spatial_analysis import extract
from rh_place1_external_validation import control_vector, nearest_row
from rh_place2_seven_band_run_holdout import LABELS, LEVELS, VIDEOS, matrices, nearest_level
from train_models import CACHE_VERSION, read_csv


CALIBRATION = {"response3": .5, "response6": 2.0}
STABLE = {
    "response3": (2.25, 4.25, 10.0, 18.0, 27.25, 31.5, 37.5),
    "response6": (9.0, 11.5, 13.0, 16.5, 17.75, 19.75, 25.0),
}
# User-reviewed frames from the frozen-v37 unused-time audit.
VALIDATION = {
    "response3": ((1.0, 25), (4.0, 35), (9.0, 45), (18.0, 55),
                  (26.5, 65), (30.0, 75), (37.0, 85)),
    "response6": ((9.0, 25), (11.5, 35), (14.5, 45), (15.5, 55),
                  (17.5, 65), (19.0, 75), (23.0, 85)),
}
STEP = .25
RADIUS = .5
MIN_SEPARATION = .75


def window_times(centres):
    values = []
    for centre in centres:
        values.extend(np.arange(centre - RADIUS, centre + RADIUS + STEP / 2, STEP))
    return sorted(set(round(float(value), 2) for value in values if value >= 0))


def median_near(times, vectors, centre):
    chosen = np.abs(times - centre) <= RADIUS + 1e-9
    return np.median(vectors[chosen], axis=0), int(chosen.sum())


def predict_1nn(train_x, train_y, test_x):
    scale = np.maximum(np.std(train_x, axis=0), .5)
    distance = np.sqrt(np.mean(
        ((test_x[:, None] - train_x[None]) / scale) ** 2, axis=2))
    return train_y[np.argmin(distance, axis=1)]


def predict_polyline(train_x, train_y, test_x):
    """Nearest point on the ordered optical path, rounded to an RH band."""
    scale = np.maximum(np.std(train_x, axis=0), .5)
    points, values = [], []
    for index in range(len(train_x) - 1):
        for fraction in np.linspace(0, 1, 41, endpoint=index == len(train_x) - 2):
            points.append(train_x[index] * (1 - fraction) + train_x[index + 1] * fraction)
            values.append(train_y[index] * (1 - fraction) + train_y[index + 1] * fraction)
    points = np.asarray(points); values = np.asarray(values)
    distance = np.sqrt(np.mean(
        ((test_x[:, None] - points[None]) / scale) ** 2, axis=2))
    return nearest_level(values[np.argmin(distance, axis=1)])


def predict_ridge(train_x, train_y, test_x):
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    return nearest_level(model.fit(train_x, train_y).predict(test_x))


def plot(results, best, output):
    matrix = np.asarray(results[best]["row_normalized_confusion_0_to_1"])
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)
    axes[0].imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for row in range(7):
        for column in range(7):
            value = matrix[row, column]
            axes[0].text(column, row, f"{value:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if value >= .65 else "#1f2937")
    axes[0].set_xticks(range(7), LABELS, rotation=35, ha="right")
    axes[0].set_yticks(range(7), LABELS)
    axes[0].set(xlabel="Predicted", ylabel="Reference",
                title=f"{best} row-normalized confusion (0-1)")
    names = list(results); x = np.arange(len(names)); width = .34
    axes[1].bar(x-width/2, [results[n]["exact_accuracy"] for n in names],
                width, label="Exact")
    axes[1].bar(x+width/2, [results[n]["within_one_adjacent_range"] for n in names],
                width, label="Within one")
    axes[1].axhline(.85, color="crimson", linestyle="--", label="0.85 target")
    axes[1].set_xticks(x, [n.replace("_", "\n") for n in names])
    axes[1].set_ylim(0, 1.03); axes[1].legend()
    axes[1].set(ylabel="Score (0-1)", title="Separated profile candidates")
    fig.suptitle("Place-2 RH stable-profile non-neighbour validation",
                 fontweight="bold")
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"rh_place2_profile_stable_model.{suffix}", **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_place2_profile_stable_model_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cached = read_csv(args.cache)

    kept = {group: tuple((time, level) for time, level in VALIDATION[group]
                         if min(abs(time - selected) for selected in STABLE[group])
                         >= MIN_SEPARATION)
            for group in VIDEOS}
    requested = {}
    items = []
    for group, video in VIDEOS.items():
        centres = list(STABLE[group]) + [time for time, _ in kept[group]]
        times = sorted(set([CALIBRATION[group]] + window_times(centres)))
        requested[group] = times
        for time in times:
            items.append({"group": group, "video": video, "time": time,
                          "row": nearest_row(cached, video, time),
                          "calibration": abs(time - CALIBRATION[group]) < 1e-9})
    summaries = extract(items, args.video_root)
    controls = [control_vector(summary) for summary in summaries]
    baseline = {item["group"]: value for item, value in zip(items, controls)
                if item["calibration"]}
    dense = {}
    for group in VIDEOS:
        pairs = sorted((item["time"], value - baseline[group])
                       for item, value in zip(items, controls)
                       if item["group"] == group and not item["calibration"])
        dense[group] = (np.asarray([p[0] for p in pairs]),
                        np.asarray([p[1] for p in pairs]))

    train, validation = {}, []
    for group in VIDEOS:
        times, vectors = dense[group]
        train_x = np.asarray([median_near(times, vectors, time)[0]
                              for time in STABLE[group]])
        train[group] = (train_x, LEVELS.copy())
        for time, level in kept[group]:
            vector, support = median_near(times, vectors, time)
            centre_index = int(np.argmin(np.abs(times - time)))
            validation.append({"group": group, "time_s": time,
                               "reference": level, "support": support,
                               "vector_window": vector,
                               "vector_single": vectors[centre_index]})

    predictors = {"profile_1nn": predict_1nn,
                  "profile_polyline": predict_polyline,
                  "profile_ridge": predict_ridge}
    truth = np.asarray([row["reference"] for row in validation], dtype=int)
    results, prediction_rows = {}, []
    for base_name, predictor in predictors.items():
        for input_mode in ("single", "window"):
            name = f"{base_name}_{input_mode}"
            prediction = []
            for row in validation:
                train_x, train_y = train[row["group"]]
                prediction.append(int(predictor(
                    train_x, train_y, row[f"vector_{input_mode}"][None])[0]))
            prediction = np.asarray(prediction)
            results[name] = matrices(truth, prediction)
            for row, predicted in zip(validation, prediction):
                prediction_rows.append({"model": name, "group": row["group"],
                                        "time_s": row["time_s"],
                                        "reference": row["reference"],
                                        "prediction": int(predicted),
                                        "correct": row["reference"] == predicted})
    single_results = {name: value for name, value in results.items()
                      if name.endswith("_single")}
    best = max(single_results, key=lambda name: (
        results[name]["exact_accuracy"], results[name]["balanced_accuracy"]))
    outer = {25, 35, 65, 75, 85}
    best_rows = [row for row in prediction_rows if row["model"] == best]
    outer_rows = [row for row in best_rows if int(row["reference"]) in outer]
    outer_accuracy = float(np.mean([row["correct"] for row in outer_rows]))
    deploy = bool(results[best]["exact_accuracy"] >= .85 and outer_accuracy >= .85)
    payload = {
        "scope": "profile-separated stable-window RH; single-frame app validation >=0.75 s from selected centre",
        "kept_validation": {group: list(rows) for group, rows in kept.items()},
        "results": results,
        "decision": {"best_model": best, "outer_band_accuracy": outer_accuracy,
                     "deploy_to_app": deploy,
                     "rule": "exact >= 0.85 and frozen outer-band accuracy >= 0.85",
                     "warning": "same-run validation; not an independent experiment; deployment uses single-frame score"},
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader(); writer.writerows(prediction_rows)
    plot(results, best, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
