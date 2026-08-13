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
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from train_models import CACHE_VERSION, read_csv


TASKS = {
    "H2": {"kind": "h2_only", "label": "h2_value", "levels": [0, 1, 2, 3, 4],
           "display_levels": ["0", "1", "2", "3", "4"], "region": "flame"},
    "RH": {"kind": "rh_only", "label": "rh_value", "levels": [25, 40, 50, 60, 70, 80, 90],
           "display_levels": ["20–30", "40", "50", "60", "70", "80", "90"], "region": "drop"},
}
MIN_STABLE_SECONDS = 1.0

# User timelines describe ramp endpoints, not piecewise-constant plateaus.
# (time, concentration reached at that time). Runs 4/5 then recover linearly
# from 4% to 0% by the final frame.
H2_RAMP_ENDPOINTS = {
    "1_90_H2_only_test.mp4": [(0, 0), (15, 1), (25, 2), (30, 3), (40, 4)],
    "1_90_H2_only_test_2.mp4": [(0, 0), (4, 0), (13, 1), (21, 2), (30, 3), (51, 4)],
    "1_90_H2_only_test_3.MOV": [(0, 0), (3, 0), (10, 1), (20, 2), (28, 3), (152, 4)],
    "1_90_H2_only_4.mp4": [(0, 0), (5, 0), (13, 1), (30, 2), (109, 3), (122, 4)],
    "1_90_H2_only_5.mp4": [(0, 0), (5, 0), (8, 1), (13, 2), (21, 3), (130, 4)],
}
H2_RECOVERY_START = {"1_90_H2_only_4.mp4": 122.0, "1_90_H2_only_5.mp4": 130.0}


class OrdinalLogistic(BaseEstimator, ClassifierMixin):
    """Learn P(level >= threshold) so neighbouring stages remain ordered."""

    def __init__(self, C=.5):
        self.C = C

    def fit(self, x, y):
        self.classes_ = np.asarray(sorted(set(y)), dtype=float)
        self.models_ = []
        for threshold in self.classes_[1:]:
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=self.C, max_iter=3000, class_weight="balanced", random_state=42))
            model.fit(x, np.asarray(y) >= threshold)
            self.models_.append(model)
        return self

    def predict_proba(self, x):
        cumulative = np.column_stack([
            model.predict_proba(x)[:, 1] for model in self.models_
        ])
        cumulative = np.minimum.accumulate(cumulative, axis=1)
        probability = np.column_stack([
            1 - cumulative[:, 0],
            *[cumulative[:, j - 1] - cumulative[:, j]
              for j in range(1, cumulative.shape[1])],
            cumulative[:, -1],
        ])
        probability = np.clip(probability, 0, 1)
        return probability / np.maximum(probability.sum(axis=1, keepdims=True), 1e-9)

    def predict(self, x):
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]


def target_value(row, config):
    if config["label"] == "h2_value" and "analysis_stage" in row:
        return float(row["analysis_stage"])
    value = float(row[config["label"]])
    return 25.0 if config["label"] == "rh_value" and value <= 30 else value


def assign_h2_ramp_targets(rows):
    """Replace the earlier plateau interpretation with endpoint interpolation."""
    for row in rows:
        video = str(row["video"])
        if video not in H2_RAMP_ENDPOINTS:
            continue
        points = list(H2_RAMP_ENDPOINTS[video])
        duration = float(row["duration"])
        recovery_start = H2_RECOVERY_START.get(video)
        if recovery_start is not None:
            points.append((duration, 0.0))
        seconds, values = zip(*points)
        continuous = float(np.interp(float(row["time"]), seconds, values))
        row["continuous_target"] = continuous
        row["analysis_stage"] = float(np.clip(np.floor(continuous + .5), 0, 4))


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
    reference = "drop" if region == "flame" else "flame"
    ref_L, ref_a, ref_b = (float(row[f"{reference}_{channel}"]) for channel in "Lab")
    chroma = float(np.hypot(a, b))
    hue = float(np.arctan2(b, a))
    # The target shape remains the sensing signal. The other printed shape is an
    # internal reference that removes colour/brightness motion shared by the frame.
    output = [L, a, b, chroma, np.sin(hue), np.cos(hue),
              L - ref_L, a - ref_a, b - ref_b]
    if region == "flame":
        # H2 benefits slightly from knowing the held-out run's own 0% flame
        # colour in addition to its delta. RH overfits this domain cue, so its
        # model deliberately remains delta-only.
        base = [float(row.get(f"baseline_{region}_{channel}", 0)) for channel in "Lab"]
        base_ref = [float(row.get(f"baseline_{reference}_{channel}", 0)) for channel in "Lab"]
        output.extend([*base, *[value - ref for value, ref in zip(base, base_ref)]])
    return output


def candidates():
    return {
        "ordinal_logistic": OrdinalLogistic(C=.5),
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


def evaluate(rows, config, estimator, name, protocol="video_holdout"):
    levels = np.asarray(config["levels"], dtype=float)
    x = np.asarray([augment(row, config["region"]) for row in rows])
    y = np.asarray([target_value(row, config) for row in rows])
    groups = np.asarray([str(row["group"]) for row in rows])
    prediction = np.full(len(y), np.nan)
    confidence = np.full(len(y), np.nan)
    if protocol == "video_holdout":
        splits = [(groups != group, groups == group) for group in sorted(set(groups))]
    elif protocol == "within_run_blocks":
        # Five-second blocks prevent adjacent frames from being split individually,
        # while retaining the calibration/domain information from each recording.
        blocks = np.asarray([f"{row['video']}::{int(float(row['time']) // 5)}" for row in rows])
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        splits = list(splitter.split(x, y, groups=blocks))
    else:
        raise ValueError(protocol)
    for train, test in splits:
        fitted = clone(estimator)
        fit_kwargs = {}
        if name.endswith("rounded"):
            counts = {level: int(np.sum(y[train] == level)) for level in levels}
            sample_weight = np.asarray([
                len(train) / max(len(levels) * counts[value], 1) for value in y[train]
            ])
            if hasattr(fitted, "steps"):
                fit_kwargs[f"{fitted.steps[-1][0]}__sample_weight"] = sample_weight
            else:
                fit_kwargs["sample_weight"] = sample_weight
        fitted.fit(x[train], y[train], **fit_kwargs)
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
        "protocol": "calibration_aware_video_holdout" if protocol == "video_holdout" else protocol,
        "exact_accuracy": float(np.mean(prediction == y)),
        "within_one_step": float(np.mean(level_distance <= 1)),
        "mae": float(np.mean(np.abs(prediction - y))),
        "video_macro_exact_accuracy": float(np.mean([row["exact_accuracy"] for row in per_video])),
        "video_macro_within_one_step": float(np.mean([row["within_one_step"] for row in per_video])),
        "video_macro_mae": float(np.mean([row["mae"] for row in per_video])),
        "n_frames": len(y), "n_videos": len(per_video), "per_video": per_video,
        "confusion": confusion_matrix(y, prediction, labels=levels).tolist(),
    }
    cm = np.asarray(metric["confusion"], dtype=float)
    metric["stage_balanced_accuracy"] = float(np.mean(np.diag(cm) / np.maximum(cm.sum(axis=1), 1)))
    predictions = [{
        "video": row["video"], "group": group, "time": row["time"],
        "reference": float(truth), "prediction": float(estimate), "confidence": float(conf),
    } for row, group, truth, estimate, conf in zip(rows, groups, y, prediction, confidence)]
    return metric, predictions


def colour_path(rows, config):
    output = []
    for level in config["levels"]:
        use = [row for row in rows if target_value(row, config) == level]
        values = np.asarray([augment(row, config["region"]) for row in use])
        output.append({
            "level": level, "n": len(use), "n_videos": len({str(row["group"]) for row in use}),
            "median_L": float(np.median(values[:, 0])),
            "median_a": float(np.median(values[:, 1])),
            "median_b": float(np.median(values[:, 2])),
            "median_chroma": float(np.median(values[:, 3])),
        })
    return output


def draw_confusion(axis, matrix, levels, title):
    cm = np.asarray(matrix, dtype=float)
    cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1)
    axis.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(levels)):
        for j in range(len(levels)):
            axis.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=6,
                      color="white" if cm[i,j] > .55 else "black")
    axis.set_xticks(range(len(levels)), levels)
    axis.set_yticks(range(len(levels)), levels)
    axis.set(xlabel="Predicted concentration (%)", ylabel="Reference concentration (%)", title=title)


def plot(output, reports, paths):
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 9.5), constrained_layout=True)
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
        display_levels = config["display_levels"]
        draw_confusion(axes[1, col], reports[task]["models"][selected]["confusion"], display_levels,
                       f"{task}: calibration-aware video-held-out")
        draw_confusion(axes[2, col], reports[task]["within_run_models"][selected]["confusion"], display_levels,
                       f"{task}: within-run 5 s blocks")
    fig.suptitle("Limited-data single-frame concentration validation", weight="bold")
    for suffix, kwargs in (("png", {"dpi": 500}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"ordinal_concentration_validation.{suffix}", **kwargs)
    plt.close(fig)


def main():
    output = Path("training/output/ordinal_concentration")
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path("training/cache") / CACHE_VERSION / "features.csv")
    add_stability(rows)
    assign_h2_ramp_targets(rows)
    reports, paths, all_predictions = {}, {}, []
    for task, config in TASKS.items():
        task_rows = [row for row in rows if row["kind"] == config["kind"]
                     and row.get(config["label"]) is not None
                     and (task == "H2" or
                          float(row[config["label"] + "_stable_seconds"]) >= MIN_STABLE_SECONDS)]
        if task == "H2":
            task_rows = [row for row in task_rows if "analysis_stage" in row]
        paths[task] = colour_path(task_rows, config)
        models, predictions, within_run_models, within_run_predictions = {}, {}, {}, {}
        for name, estimator in candidates().items():
            models[name], predictions[name] = evaluate(task_rows, config, estimator, name, "video_holdout")
            within_run_models[name], within_run_predictions[name] = evaluate(
                task_rows, config, estimator, name, "within_run_blocks")
        # Sparse stages count equally. Select a model that works both on an unseen
        # recording and on held-out time blocks from a calibrated recording.
        selected = max(models, key=lambda name: (
            np.sqrt(models[name]["stage_balanced_accuracy"]
                    * within_run_models[name]["stage_balanced_accuracy"]),
            models[name]["stage_balanced_accuracy"],
            models[name]["video_macro_within_one_step"],
            -models[name]["video_macro_mae"],
        ))
        reports[task] = {"selected": selected, "models": models,
                         "within_run_models": within_run_models,
                         "colour_path": paths[task], "predictions": predictions[selected],
                         "within_run_predictions": within_run_predictions[selected]}
        for protocol, source in (("video_holdout", predictions[selected]),
                                 ("within_run_blocks", within_run_predictions[selected])):
            for row in source:
                all_predictions.append({"task": task, "protocol": protocol, "model": selected, **row})
    (output / "metrics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_predictions[0])); writer.writeheader(); writer.writerows(all_predictions)
    plot(output, reports, paths)
    print(json.dumps({task: {"selected": value["selected"],
                            "video_holdout": value["models"][value["selected"]],
                            "within_run_blocks": value["within_run_models"][value["selected"]],
                            "colour_path": value["colour_path"]}
                      for task, value in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
