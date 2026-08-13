"""Video-held-out ordinal concentration analysis for H2-only and H2O-only.

The sensor response is treated as an ordered colour trajectory, not unrelated
colour names. Features are per-chip calibration deltas from the flame (H2) or
droplet (RH), matching the instant single-frame browser application.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_models import CACHE_VERSION, read_csv


TASKS = {
    "H2": {"kind": "h2_only", "label": "h2_value", "levels": [0, 1, 2, 3, 4], "region": "flame"},
    "RH": {"kind": "rh_only", "label": "rh_value", "levels": [20, 30, 40, 50, 60, 70, 80, 90], "region": "drop"},
}


def add_stability(rows):
    by_video = defaultdict(list)
    for row in rows:
        by_video[str(row["video"])].append(row)
    for video_rows in by_video.values():
        video_rows.sort(key=lambda row: float(row["time"]))
        for label in ("h2_value", "rh_value"):
            start, previous = None, object()
            for row in video_rows:
                value = row.get(label)
                if value is None:
                    start, previous = None, object()
                    row[label + "_stable_seconds"] = 0.0
                    continue
                if start is None or value != previous:
                    start = float(row["time"])
                row[label + "_stable_seconds"] = float(row["time"]) - start
                previous = value


def augment(row, region):
    L, a, b = (float(row[f"{region}_{channel}"]) for channel in "Lab")
    chroma = float(np.hypot(a, b))
    hue = float(np.arctan2(b, a))
    return [L, a, b, chroma, np.sin(hue), np.cos(hue)]


def candidates():
    return {
        "multinomial_logistic": make_pipeline(StandardScaler(), LogisticRegression(
            C=.5, max_iter=3000, class_weight="balanced", random_state=42)),
        "extra_trees_classifier": ExtraTreesClassifier(
            n_estimators=400, max_depth=9, min_samples_leaf=7,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "gradient_boosting_classifier": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=.04, max_leaf_nodes=15,
            min_samples_leaf=15, l2_regularization=3, random_state=42),
        "ridge_rounded": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "extra_trees_regression_rounded": ExtraTreesRegressor(
            n_estimators=400, max_depth=9, min_samples_leaf=7, random_state=42, n_jobs=-1),
    }


def nearest_level(values, levels):
    levels = np.asarray(levels, dtype=float)
    return levels[np.argmin(np.abs(np.asarray(values)[:, None] - levels[None, :]), axis=1)]


def evaluate(rows, config, estimator, name):
    levels = np.asarray(config["levels"], dtype=float)
    x = np.asarray([augment(row, config["region"]) for row in rows])
    y = np.asarray([float(row[config["label"]]) for row in rows])
    groups = np.asarray([str(row["group"]) for row in rows])
    prediction = np.full(len(y), np.nan)
    confidence = np.full(len(y), np.nan)
    for group in sorted(set(groups)):
        test, train = groups == group, groups != group
        fitted = clone(estimator).fit(x[train], y[train])
        if name.endswith("rounded"):
            raw = np.asarray(fitted.predict(x[test])).reshape(-1)
            prediction[test] = nearest_level(raw, levels)
            distance = np.abs(raw - prediction[test])
            step = float(np.median(np.diff(levels)))
            confidence[test] = np.clip(1 - distance / max(step / 2, 1e-6), 0, 1)
        else:
            probability = fitted.predict_proba(x[test])
            classes = np.asarray(fitted.classes_, dtype=float)
            best = np.argmax(probability, axis=1)
            prediction[test] = classes[best]
            confidence[test] = probability[np.arange(len(best)), best]
    step = float(np.median(np.diff(levels)))
    level_distance = np.abs(prediction - y) / step
    per_video = []
    for group in sorted(set(groups)):
        use = groups == group
        per_video.append({
            "group": group, "n": int(use.sum()),
            "exact_accuracy": float(np.mean(prediction[use] == y[use])),
            "within_one_step": float(np.mean(level_distance[use] <= 1)),
            "mae": float(np.mean(np.abs(prediction[use] - y[use]))),
        })
    metric = {
        "exact_accuracy": float(np.mean(prediction == y)),
        "within_one_step": float(np.mean(level_distance <= 1)),
        "mae": float(np.mean(np.abs(prediction - y))),
        "video_macro_exact_accuracy": float(np.mean([row["exact_accuracy"] for row in per_video])),
        "video_macro_within_one_step": float(np.mean([row["within_one_step"] for row in per_video])),
        "video_macro_mae": float(np.mean([row["mae"] for row in per_video])),
        "n_frames": len(y), "n_videos": len(per_video), "per_video": per_video,
        "confusion": confusion_matrix(y, prediction, labels=levels).tolist(),
    }
    predictions = [{
        "video": row["video"], "group": group, "time": row["time"],
        "reference": float(truth), "prediction": float(estimate), "confidence": float(conf),
    } for row, group, truth, estimate, conf in zip(rows, groups, y, prediction, confidence)]
    return metric, predictions


def colour_path(rows, config):
    output = []
    for level in config["levels"]:
        use = [row for row in rows if float(row[config["label"]]) == level]
        values = np.asarray([augment(row, config["region"]) for row in use])
        output.append({
            "level": level, "n": len(use), "n_videos": len({str(row["group"]) for row in use}),
            "median_L": float(np.median(values[:, 0])),
            "median_a": float(np.median(values[:, 1])),
            "median_b": float(np.median(values[:, 2])),
            "median_chroma": float(np.median(values[:, 3])),
        })
    return output


def plot(output, reports, paths):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.7), constrained_layout=True)
    for col, task in enumerate(("H2", "RH")):
        config = TASKS[task]; path = paths[task]
        x = [row["level"] for row in path]
        axes[0, col].plot(x, [row["median_L"] for row in path], "o-", label="L*")
        axes[0, col].plot(x, [row["median_a"] for row in path], "o-", label="a* delta")
        axes[0, col].plot(x, [row["median_b"] for row in path], "o-", label="b* delta")
        axes[0, col].set(xlabel=f"{task} reference (%)", ylabel="Median calibrated LAB feature",
                         title=f"{task} ordered colour trajectory")
        axes[0, col].legend(frameon=False, fontsize=7)
        selected = reports[task]["selected"]
        pred = reports[task]["predictions"]
        levels = config["levels"]
        cm = np.asarray(reports[task]["models"][selected]["confusion"], dtype=float)
        cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1)
        axes[1, col].imshow(cm, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(levels)):
            for j in range(len(levels)):
                axes[1, col].text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=6,
                                  color="white" if cm[i,j] > .55 else "black")
        axes[1, col].set_xticks(range(len(levels)), levels)
        axes[1, col].set_yticks(range(len(levels)), levels)
        axes[1, col].set(xlabel="Predicted concentration (%)", ylabel="Reference concentration (%)",
                         title=f"{task}: {selected}")
    fig.suptitle("Single-frame ordinal concentration validation (video-wise holdout)", weight="bold")
    for suffix, kwargs in (("png", {"dpi": 500}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"ordinal_concentration_validation.{suffix}", **kwargs)
    plt.close(fig)


def main():
    output = Path("training/output/ordinal_concentration")
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path("training/cache") / CACHE_VERSION / "features.csv")
    add_stability(rows)
    reports, paths, all_predictions = {}, {}, []
    for task, config in TASKS.items():
        task_rows = [row for row in rows if row["kind"] == config["kind"]
                     and row.get(config["label"]) is not None
                     and float(row[config["label"] + "_stable_seconds"]) >= 4.5]
        paths[task] = colour_path(task_rows, config)
        models, predictions = {}, {}
        for name, estimator in candidates().items():
            models[name], predictions[name] = evaluate(task_rows, config, estimator, name)
        # Prioritize exact step recognition, use one-step accuracy and MAE as tie breakers.
        selected = max(models, key=lambda name: (
            models[name]["video_macro_exact_accuracy"],
            models[name]["video_macro_within_one_step"],
            -models[name]["video_macro_mae"],
        ))
        reports[task] = {"selected": selected, "models": models,
                         "colour_path": paths[task], "predictions": predictions[selected]}
        for row in predictions[selected]:
            all_predictions.append({"task": task, "model": selected, **row})
    (output / "metrics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_predictions[0])); writer.writeheader(); writer.writerows(all_predictions)
    plot(output, reports, paths)
    print(json.dumps({task: {"selected": value["selected"],
                            "metric": value["models"][value["selected"]],
                            "colour_path": value["colour_path"]}
                      for task, value in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
