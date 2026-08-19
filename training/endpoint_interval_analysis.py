"""Rebuild H2/RH supervision as exact endpoints plus censored ramp intervals.

The supplied timelines state the concentration reached at the end of each
interval.  Only a short endpoint window (and a genuine repeated-value hold) is
therefore an exact label.  Ramp-interior frames retain the known lower/upper
stage but are never silently converted to a linear point target.  Recovery is
excluded from quantitative training and evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ordinal_concentration_analysis import (
    H2_RAMP_ENDPOINTS, H2_RECOVERY_START, RH_RAMP_ENDPOINTS, TASKS,
    assign_h2_ramp_targets, assign_rh_ramp_targets, augment,
)
from train_models import CACHE_VERSION, read_csv


ENDPOINT_WINDOW_SECONDS = .55
MAX_HOLD_EVAL_PER_VIDEO_STAGE = 8
MAX_BASELINE_EVAL_PER_VIDEO = 4


def stage_value(task: str, value: float) -> float:
    levels = np.asarray(TASKS[task]["levels"], dtype=float)
    return float(levels[np.argmin(np.abs(levels - float(value)))])


def supervision_for(
    task: str, video: str, time: float, duration: float,
) -> tuple[float, float, float | None, str] | None:
    """Return lower stage, upper stage, optional exact stage, and source."""
    points = H2_RAMP_ENDPOINTS.get(video) if task == "H2" else RH_RAMP_ENDPOINTS.get(video)
    if not points:
        return None
    if task == "H2":
        recovery_start = H2_RECOVERY_START.get(video)
        if recovery_start is not None and time >= recovery_start:
            return None
        # Every H2 run starts at a known 0% calibration condition.  This is a
        # real baseline anchor even when the first ramp starts immediately.
        if time <= min(1.0, points[1][0] if len(points) > 1 else 1.0):
            return 0.0, 0.0, 0.0, "baseline"

    first_time, first_value = points[0]
    if time < first_time:
        return None
    previous_time, previous_value = first_time, first_value
    for end, target in points[1:]:
        if previous_time <= time <= end:
            previous_stage = stage_value(task, previous_value)
            target_stage = stage_value(task, target)
            if previous_stage == target_stage:
                return target_stage, target_stage, target_stage, "hold"
            lower, upper = sorted((previous_stage, target_stage))
            if end - ENDPOINT_WINDOW_SECONDS <= time <= end:
                return target_stage, target_stage, target_stage, "endpoint"
            return lower, upper, None, "interval"
        previous_time, previous_value = end, target

    if task == "H2" and video not in H2_RECOVERY_START and time >= points[-1][0]:
        target = stage_value(task, points[-1][1])
        return target, target, target, "hold"
    return None


def prepare(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    prepared = []
    for row in rows:
        task = "H2" if row["kind"] == "h2_only" else ("RH" if row["kind"] == "rh_only" else None)
        if task is None:
            continue
        label = supervision_for(
            task, str(row["video"]), float(row["time"]), float(row["duration"]))
        if label is None:
            continue
        lower, upper, exact, source = label
        prepared.append({
            "task": task, "row": row, "lower": lower, "upper": upper,
            "exact": exact, "source": source, "group": str(row["group"]),
            "video": str(row["video"]), "time": float(row["time"]),
        })

    # Score every short endpoint window.  Long baseline/hold sequences are
    # deterministically thinned so duration cannot inflate a stage's score.
    buckets = defaultdict(list)
    for item in prepared:
        if item["exact"] is not None:
            buckets[(item["task"], item["video"], item["source"], item["exact"])].append(item)
    for (_, _, source, _), items in buckets.items():
        items.sort(key=lambda item: item["time"])
        if source == "endpoint":
            limit = len(items)
        elif source == "baseline":
            limit = MAX_BASELINE_EVAL_PER_VIDEO
        else:
            limit = MAX_HOLD_EVAL_PER_VIDEO_STAGE
        chosen = set(np.linspace(0, len(items) - 1, min(limit, len(items))).round().astype(int))
        for index, item in enumerate(items):
            item["evaluate"] = index in chosen
    for item in prepared:
        item.setdefault("evaluate", False)
    return prepared


def feature_matrix(items, task):
    region = TASKS[task]["region"]
    return np.asarray([augment(item["row"], region) for item in items], dtype=float)


def supervision_base_weights(items):
    """Give each run/anchor comparable mass regardless of video duration."""
    buckets = defaultdict(list)
    for index, item in enumerate(items):
        key = (item["video"], item["source"], item["lower"], item["upper"])
        buckets[key].append(index)
    weights = np.ones(len(items), dtype=float)
    for (_, source, _, _), indices in buckets.items():
        if source == "endpoint":
            cap = len(indices)
        elif source in ("baseline", "hold"):
            cap = 8
        else:
            cap = 12
        weights[indices] = min(1.0, cap / max(len(indices), 1))
    return weights


def fit_censored(items, task, train, C, interval_weight):
    x = feature_matrix(items, task)
    lower = np.asarray([item["lower"] for item in items], dtype=float)
    upper = np.asarray([item["upper"] for item in items], dtype=float)
    exact = np.asarray([item["exact"] is not None for item in items])
    base_weights = supervision_base_weights(items)
    levels = np.asarray(TASKS[task]["levels"], dtype=float)
    thresholds = (levels[:-1] + levels[1:]) / 2
    models = []
    for threshold in thresholds:
        known = train & ((upper <= threshold) | (lower > threshold))
        y = lower[known] > threshold
        weights = base_weights[known] * np.where(exact[known], 1.0, interval_weight)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, max_iter=3000, class_weight="balanced", random_state=42))
        model.fit(x[known], y, logisticregression__sample_weight=weights)
        models.append(model)
    return models, x


def predict_censored(models, x, levels):
    cumulative = np.column_stack([model.predict_proba(x)[:, 1] for model in models])
    cumulative = np.minimum.accumulate(cumulative, axis=1)
    probability = np.column_stack([
        1 - cumulative[:, 0],
        *[cumulative[:, index - 1] - cumulative[:, index]
          for index in range(1, cumulative.shape[1])],
        cumulative[:, -1],
    ])
    probability = np.clip(probability, 0, 1)
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-9)
    return levels[np.argmax(probability, axis=1)], np.max(probability, axis=1)


def balanced_score(truth, prediction):
    present = sorted(set(truth))
    return float(np.mean([np.mean(prediction[truth == level] == level) for level in present]))


def tune_censored(items, task, outer_group):
    groups = np.asarray([item["group"] for item in items])
    exact = np.asarray([item["exact"] is not None for item in items])
    evaluate = np.asarray([item["evaluate"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    outer_train = groups != outer_group
    best = None
    for C in (.05, .15, .5, 1.5):
        for interval_weight in (0.0, .05, .15, .30):
            fold_scores = []
            for inner_group in sorted(set(groups[outer_train])):
                train = outer_train & (groups != inner_group)
                test = outer_train & (groups == inner_group) & exact & evaluate
                models, x = fit_censored(items, task, train, C, interval_weight)
                prediction, _ = predict_censored(
                    models, x[test], np.asarray(TASKS[task]["levels"], dtype=float))
                fold_scores.append((
                    balanced_score(truth[test], prediction),
                    float(np.mean(truth[test] == prediction)),
                    -float(np.mean(np.abs(truth[test] - prediction))),
                ))
            score = tuple(np.mean(fold_scores, axis=0))
            if best is None or score > best[0]:
                best = (score, C, interval_weight)
    return best[1], best[2]


def evaluate_censored(items, task):
    groups = np.asarray([item["group"] for item in items])
    evaluate = np.asarray([item["evaluate"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    prediction = np.full(len(items), np.nan)
    confidence = np.full(len(items), np.nan)
    chosen = []
    levels = np.asarray(TASKS[task]["levels"], dtype=float)
    for group in sorted(set(groups)):
        C, interval_weight = tune_censored(items, task, group)
        train = groups != group
        test = (groups == group) & evaluate
        models, x = fit_censored(items, task, train, C, interval_weight)
        prediction[test], confidence[test] = predict_censored(models, x[test], levels)
        chosen.append({"held_out_group": group, "C": C,
                       "interval_weight": interval_weight})
    return metric_report(items, task, truth, prediction, confidence, chosen,
                         "nested endpoint-exact + interval-censored ordinal")


def evaluate_exact_multinomial(items, task):
    groups = np.asarray([item["group"] for item in items])
    exact = np.asarray([item["exact"] is not None for item in items])
    evaluate = np.asarray([item["evaluate"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    x = feature_matrix(items, task)
    base_weights = supervision_base_weights(items)
    prediction = np.full(len(items), np.nan)
    confidence = np.full(len(items), np.nan)
    for group in sorted(set(groups)):
        train = (groups != group) & exact
        test = (groups == group) & evaluate
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=.15, max_iter=3000, class_weight="balanced", random_state=42))
        model.fit(x[train], truth[train],
                  logisticregression__sample_weight=base_weights[train])
        probabilities = model.predict_proba(x[test])
        prediction[test] = model.classes_[np.argmax(probabilities, axis=1)]
        confidence[test] = np.max(probabilities, axis=1)
    return metric_report(items, task, truth, prediction, confidence, [],
                         "endpoint/hold exact labels only; multinomial logistic")


def evaluate_linear_baseline(rows, items, task):
    # Reproduce the current linear-ramp point-label policy on the exact same
    # endpoint-only test frames, so the comparison changes supervision rather
    # than the evaluation set.
    if task == "H2":
        assign_h2_ramp_targets(rows)
        training = [row for row in rows if row["kind"] == "h2_only"
                    and row.get("analysis_phase") == "reaction"]
        target_key = "analysis_stage"
    else:
        assign_rh_ramp_targets(rows)
        training = [row for row in rows if row["kind"] == "rh_only"
                    and row.get("rh_analysis_stage") is not None]
        target_key = "rh_analysis_stage"
    groups = np.asarray([str(row["group"]) for row in training])
    x = np.asarray([augment(row, TASKS[task]["region"]) for row in training])
    y = np.asarray([float(row[target_key]) for row in training])
    item_groups = np.asarray([item["group"] for item in items])
    evaluate = np.asarray([item["evaluate"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    prediction = np.full(len(items), np.nan)
    confidence = np.full(len(items), np.nan)
    levels = np.asarray(TASKS[task]["levels"], dtype=float)
    for group in sorted(set(groups)):
        train = groups != group
        counts = {level: max(int(np.sum(y[train] == level)), 1) for level in levels}
        weights = np.asarray([len(y[train]) / (len(levels) * counts[value]) for value in y[train]])
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x[train], y[train], ridge__sample_weight=weights)
        test_indices = np.where((item_groups == group) & evaluate)[0]
        test_x = feature_matrix([items[index] for index in test_indices], task)
        raw = model.predict(test_x)
        prediction[test_indices] = levels[np.argmin(np.abs(raw[:, None] - levels), axis=1)]
    return metric_report(items, task, truth, prediction, confidence, [],
                         "current linear-ramp point labels; endpoint-only scoring")


def metric_report(items, task, truth, prediction, confidence, chosen, policy):
    use = np.asarray([item["evaluate"] for item in items]) & ~np.isnan(prediction)
    levels = np.asarray(TASKS[task]["levels"], dtype=float)
    cm = confusion_matrix(truth[use], prediction[use], labels=levels)
    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    report = {
        "policy": policy,
        "exact_accuracy": float(np.mean(prediction[use] == truth[use])),
        "stage_balanced_accuracy": float(np.mean(recalls)),
        "within_one_step": float(np.mean(
            np.abs(np.searchsorted(levels, prediction[use]) -
                   np.searchsorted(levels, truth[use])) <= 1)),
        "mae": float(np.mean(np.abs(prediction[use] - truth[use]))),
        "n_evaluation_frames": int(use.sum()),
        "n_training_rows": len(items),
        "per_stage_recall": {str(level): float(value)
                             for level, value in zip(levels, recalls)},
        "confusion": cm.tolist(),
        "outer_fold_hyperparameters": chosen,
    }
    predictions = [{
        "task": task, "video": item["video"], "group": item["group"],
        "time": item["time"], "source": item["source"],
        "reference": truth[index], "prediction": prediction[index],
        "confidence": confidence[index],
    } for index, item in enumerate(items) if use[index]]
    return report, predictions


def write_dataset(path, items):
    rows = [{key: item[key] for key in (
        "task", "video", "group", "time", "source", "lower", "upper", "exact", "evaluate")}
            for item in items]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot(output, reports):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    for ax, task in zip(axes, ("H2", "RH")):
        selected = reports[task]["selected"]
        cm = np.asarray(reports[task]["models"][selected]["confusion"], dtype=float)
        cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        for row in range(len(cm)):
            for col in range(len(cm)):
                ax.text(col, row, f"{cm[row, col]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if cm[row, col] > .55 else "black")
        labels = TASKS[task]["display_levels"]
        ax.set_xticks(range(len(labels)), labels, rotation=35 if task == "RH" else 0)
        ax.set_yticks(range(len(labels)), labels)
        ax.set(xlabel="Predicted stage", ylabel="Exact endpoint reference",
               title=f"{task}: {selected}")
    for suffix, kwargs in (("png", {"dpi": 500}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"endpoint_interval_validation.{suffix}", **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("training/cache") / CACHE_VERSION / "features_registered_drop_v2.csv")
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/endpoint_interval"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.cache)
    items = prepare(rows)
    write_dataset(args.output / "endpoint_interval_dataset.csv", items)
    reports, all_predictions = {}, []
    for task in ("H2", "RH"):
        task_items = [item for item in items if item["task"] == task]
        baseline, baseline_predictions = evaluate_linear_baseline(rows, task_items, task)
        exact, exact_predictions = evaluate_exact_multinomial(task_items, task)
        censored, censored_predictions = evaluate_censored(task_items, task)
        models = {
            "linear_ramp_baseline": baseline,
            "exact_multinomial": exact,
            "censored_ordinal": censored,
        }
        selected = max(models, key=lambda name: (
            models[name]["stage_balanced_accuracy"],
            models[name]["exact_accuracy"], -models[name]["mae"]))
        reports[task] = {"selected": selected,
                         "models": models}
        prediction_sets = {
            "linear_ramp_baseline": baseline_predictions,
            "exact_multinomial": exact_predictions,
            "censored_ordinal": censored_predictions,
        }
        all_predictions.extend({"model": selected, **row}
                               for row in prediction_sets[selected])
    (args.output / "metrics.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_predictions[0]))
        writer.writeheader(); writer.writerows(all_predictions)
    plot(args.output, reports)
    summary = {task: {"selected": report["selected"],
                      **{name: {key: model[key] for key in (
                          "exact_accuracy", "stage_balanced_accuracy", "within_one_step", "mae")}
                         for name, model in report["models"].items()}}
               for task, report in reports.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
