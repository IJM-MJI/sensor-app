"""Compare one- and two-anchor run-normalized concentration progress models."""

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

from endpoint_interval_analysis import (
    ConstantBinary, feature_matrix, prepare, supervision_base_weights,
)
from ordinal_concentration_analysis import TASKS
from train_models import CACHE_VERSION, read_csv


def fold_transform(items, task, raw_x, train, mode):
    """Normalize feature coordinates, then center on run-specific calibration."""
    exact = np.asarray([item["exact"] is not None for item in items])
    scaler = StandardScaler().fit(raw_x[train & exact])
    x = scaler.transform(raw_x)
    groups = np.asarray([item["group"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    low_level = float(TASKS[task]["levels"][0])
    output = np.zeros_like(x) if mode == "one_anchor" else np.zeros((len(x), 5))
    anchors = {}
    for group in sorted(set(groups)):
        low_rows = (groups == group) & exact & (truth == low_level)
        if not np.any(low_rows):
            raise RuntimeError(f"{task} {group} has no low calibration anchor")
        low = np.median(x[low_rows], axis=0)
        if mode == "one_anchor":
            output[groups == group] = x[groups == group] - low
            anchors[group] = {"low_level": low_level}
            continue
        group_levels = truth[(groups == group) & exact]
        high_level = float(np.nanmax(group_levels))
        high_rows = (groups == group) & exact & (truth == high_level)
        high = np.median(x[high_rows], axis=0)
        axis = high - low
        axis_norm = max(float(np.linalg.norm(axis)), 1e-9)
        centered = x[groups == group] - low
        progress = centered @ axis / (axis_norm ** 2)
        radial = np.linalg.norm(centered, axis=1) / axis_norm
        projection = progress[:, None] * axis
        orthogonal = np.linalg.norm(centered - projection, axis=1) / axis_norm
        output[groups == group] = np.column_stack([
            progress, progress ** 2, progress ** 3, radial, orthogonal])
        anchors[group] = {"low_level": low_level, "high_level": high_level,
                          "axis_norm": axis_norm}
    return output, anchors


def fit_thresholds(items, task, x, train, C, interval_weight):
    lower = np.asarray([item["lower"] for item in items], dtype=float)
    upper = np.asarray([item["upper"] for item in items], dtype=float)
    exact = np.asarray([item["exact"] is not None for item in items])
    levels = np.asarray(TASKS[task]["levels"], dtype=float)
    thresholds = (levels[:-1] + levels[1:]) / 2
    base_weights = supervision_base_weights(items)
    models = []
    for threshold in thresholds:
        known = train & ((upper <= threshold) | (lower > threshold))
        target = lower[known] > threshold
        weight = base_weights[known] * np.where(exact[known], 1.0, interval_weight)
        active = weight > 0
        if not np.any(active):
            model = ConstantBinary(False)
        elif len(set(target[active])) < 2:
            model = ConstantBinary(bool(target[active][0]))
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=C, max_iter=3000, class_weight="balanced", random_state=42))
            model.fit(x[known], target, logisticregression__sample_weight=weight)
        models.append(model)
    return models


def threshold_predict(models, x, levels):
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


def balanced(truth, prediction):
    return float(np.mean([
        np.mean(prediction[truth == level] == level) for level in sorted(set(truth))
    ]))


def tune(items, task, raw_x, outer_group, mode):
    groups = np.asarray([item["group"] for item in items])
    evaluate = np.asarray([item["evaluate"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    outer_train = groups != outer_group
    best = None
    for C in (.03, .1, .3, 1.0, 3.0):
        for interval_weight in (0.0, .03, .10, .25):
            scores = []
            for inner_group in sorted(set(groups[outer_train])):
                train = outer_train & (groups != inner_group)
                test = outer_train & (groups == inner_group) & evaluate
                x, _ = fold_transform(items, task, raw_x, train, mode)
                models = fit_thresholds(items, task, x, train, C, interval_weight)
                prediction, _ = threshold_predict(
                    models, x[test], np.asarray(TASKS[task]["levels"], dtype=float))
                scores.append((balanced(truth[test], prediction),
                               float(np.mean(truth[test] == prediction)),
                               -float(np.mean(np.abs(truth[test] - prediction)))))
            score = tuple(np.mean(scores, axis=0))
            if best is None or score > best[0]:
                best = (score, C, interval_weight)
    return best[1], best[2]


def evaluate(items, task, mode):
    raw_x = feature_matrix(items, task)
    groups = np.asarray([item["group"] for item in items])
    evaluate_mask = np.asarray([item["evaluate"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    levels = np.asarray(TASKS[task]["levels"], dtype=float)
    prediction = np.full(len(items), np.nan)
    confidence = np.full(len(items), np.nan)
    parameters, anchor_audit = [], []
    for group in sorted(set(groups)):
        C, interval_weight = tune(items, task, raw_x, group, mode)
        train = groups != group
        test = (groups == group) & evaluate_mask
        x, anchors = fold_transform(items, task, raw_x, train, mode)
        models = fit_thresholds(items, task, x, train, C, interval_weight)
        prediction[test], confidence[test] = threshold_predict(models, x[test], levels)
        parameters.append({"held_out_group": group, "C": C,
                           "interval_weight": interval_weight})
        anchor_audit.append({"held_out_group": group, **anchors[group]})
    use = evaluate_mask & ~np.isnan(prediction)
    cm = confusion_matrix(truth[use], prediction[use], labels=levels)
    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    report = {
        "mode": mode,
        "calibration_requirement": (
            "known low endpoint only" if mode == "one_anchor" else
            "known low and run-specific high endpoint"),
        "exact_accuracy": float(np.mean(prediction[use] == truth[use])),
        "stage_balanced_accuracy": float(np.mean(recalls)),
        "within_one_step": float(np.mean(
            np.abs(np.searchsorted(levels, prediction[use]) -
                   np.searchsorted(levels, truth[use])) <= 1)),
        "mae": float(np.mean(np.abs(prediction[use] - truth[use]))),
        "n_evaluation_frames": int(use.sum()),
        "per_stage_recall": recalls.tolist(), "confusion": cm.tolist(),
        "outer_fold_hyperparameters": parameters, "anchor_audit": anchor_audit,
    }
    predictions = [{
        "task": task, "mode": mode, "video": item["video"], "group": item["group"],
        "time": item["time"], "source": item["source"], "reference": truth[index],
        "prediction": prediction[index], "confidence": confidence[index],
    } for index, item in enumerate(items) if use[index]]
    return report, predictions


def tune_exact(items, task, raw_x, outer_group, mode):
    groups = np.asarray([item["group"] for item in items])
    evaluate_mask = np.asarray([item["evaluate"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    exact = ~np.isnan(truth)
    base_weights = supervision_base_weights(items)
    outer_train = groups != outer_group
    best = None
    for C in (.01, .03, .1, .3, 1.0, 3.0):
        scores = []
        for inner_group in sorted(set(groups[outer_train])):
            train = outer_train & (groups != inner_group) & exact
            test = outer_train & (groups == inner_group) & evaluate_mask
            x, _ = fold_transform(items, task, raw_x, train, mode)
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=C, max_iter=3000, class_weight="balanced", random_state=42))
            model.fit(x[train], truth[train],
                      logisticregression__sample_weight=base_weights[train])
            prediction = model.predict(x[test])
            scores.append((balanced(truth[test], prediction),
                           float(np.mean(truth[test] == prediction)),
                           -float(np.mean(np.abs(truth[test] - prediction)))))
        score = tuple(np.mean(scores, axis=0))
        if best is None or score > best[0]:
            best = (score, C)
    return best[1]


def evaluate_exact(items, task, mode):
    raw_x = feature_matrix(items, task)
    groups = np.asarray([item["group"] for item in items])
    evaluate_mask = np.asarray([item["evaluate"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    exact = ~np.isnan(truth)
    base_weights = supervision_base_weights(items)
    prediction = np.full(len(items), np.nan)
    confidence = np.full(len(items), np.nan)
    parameters, anchor_audit = [], []
    for group in sorted(set(groups)):
        C = tune_exact(items, task, raw_x, group, mode)
        train = (groups != group) & exact
        test = (groups == group) & evaluate_mask
        x, anchors = fold_transform(items, task, raw_x, train, mode)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, max_iter=3000, class_weight="balanced", random_state=42))
        model.fit(x[train], truth[train],
                  logisticregression__sample_weight=base_weights[train])
        probabilities = model.predict_proba(x[test])
        prediction[test] = model.classes_[np.argmax(probabilities, axis=1)]
        confidence[test] = np.max(probabilities, axis=1)
        parameters.append({"held_out_group": group, "C": C})
        anchor_audit.append({"held_out_group": group, **anchors[group]})
    use = evaluate_mask & ~np.isnan(prediction)
    levels = np.asarray(TASKS[task]["levels"], dtype=float)
    cm = confusion_matrix(truth[use], prediction[use], labels=levels)
    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    report = {
        "mode": mode + "_exact_multinomial",
        "calibration_requirement": (
            "known low endpoint only" if mode == "one_anchor" else
            "known low and run-specific high endpoint"),
        "exact_accuracy": float(np.mean(prediction[use] == truth[use])),
        "stage_balanced_accuracy": float(np.mean(recalls)),
        "within_one_step": float(np.mean(
            np.abs(np.searchsorted(levels, prediction[use]) -
                   np.searchsorted(levels, truth[use])) <= 1)),
        "mae": float(np.mean(np.abs(prediction[use] - truth[use]))),
        "n_evaluation_frames": int(use.sum()),
        "per_stage_recall": recalls.tolist(), "confusion": cm.tolist(),
        "outer_fold_hyperparameters": parameters, "anchor_audit": anchor_audit,
    }
    predictions = [{
        "task": task, "mode": report["mode"], "video": item["video"],
        "group": item["group"], "time": item["time"], "source": item["source"],
        "reference": truth[index], "prediction": prediction[index],
        "confidence": confidence[index],
    } for index, item in enumerate(items) if use[index]]
    return report, predictions


def trajectory_data(items, task, mode="two_anchor"):
    raw_x = feature_matrix(items, task)
    train = np.ones(len(items), dtype=bool)
    x, _ = fold_transform(items, task, raw_x, train, mode)
    groups = np.asarray([item["group"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    exact = ~np.isnan(truth)
    output = []
    for group in sorted(set(groups)):
        for level in sorted(set(truth[(groups == group) & exact])):
            select = (groups == group) & exact & (truth == level)
            output.append({"group": group, "level": float(level),
                           "progress": float(np.median(x[select, 0]))})
    return output


def normalized_confusion(ax, report, labels, title):
    cm = np.asarray(report["confusion"], dtype=float)
    cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    for row in range(len(cm)):
        for col in range(len(cm)):
            ax.text(col, row, f"{cm[row, col]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cm[row, col] > .55 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=35 if len(labels) > 5 else 0)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted", ylabel="Exact endpoint", title=title)


def plot_summary(output, reports):
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.5), constrained_layout=True)
    for row, task in enumerate(("H2", "RH")):
        labels = TASKS[task]["display_levels"]
        normalized_confusion(axes[row, 0], reports[task]["one_anchor"], labels,
                             f"{task}: Initial anchor only")
        normalized_confusion(axes[row, 1], reports[task]["two_anchor"], labels,
                             f"{task}: Initial + high anchor")
        x = np.arange(len(labels)); width = .34
        axes[row, 2].bar(x - width / 2, reports[task]["one_anchor"]["per_stage_recall"],
                         width, label="Initial only", color="#4472C4")
        axes[row, 2].bar(x + width / 2, reports[task]["two_anchor"]["per_stage_recall"],
                         width, label="Initial + high", color="#ED7D31")
        axes[row, 2].axhline(.85, color="#2E7D32", ls="--", lw=1.4, label="0.85 target")
        axes[row, 2].set_xticks(x, labels, rotation=35 if len(labels) > 5 else 0)
        axes[row, 2].set_ylim(0, 1.03); axes[row, 2].set_ylabel("Recall")
        axes[row, 2].set_title(f"{task}: per-stage recall")
        axes[row, 2].legend(fontsize=8, loc="upper center", ncol=2)
    fig.suptitle("Run-normalized endpoint/interval validation", weight="bold")
    for suffix, kwargs in (("png", {"dpi": 450}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"run_progress_validation.{suffix}", **kwargs)
    plt.close(fig)


def plot_trajectories(output, trajectories):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    for ax, task in zip(axes, ("H2", "RH")):
        task_rows = trajectories[task]
        for group in sorted({row["group"] for row in task_rows}):
            rows = sorted((row for row in task_rows if row["group"] == group),
                          key=lambda row: row["level"])
            ax.plot([row["level"] for row in rows], [row["progress"] for row in rows],
                    marker="o", ms=4, lw=1.2, label=group)
        ax.set(xlabel=f"Exact {task} stage", ylabel="Two-anchor normalized progress",
               title=f"{task}: independent run trajectories")
        ax.grid(alpha=.2); ax.legend(fontsize=6, ncol=2)
    for suffix, kwargs in (("png", {"dpi": 450}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"run_progress_trajectories.{suffix}", **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("training/cache") / CACHE_VERSION / "features_registered_drop_v2.csv")
    parser.add_argument("--output", type=Path, default=Path("training/output/run_progress"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.cache); items = prepare(rows)
    reports, predictions, trajectories = {}, [], {}
    for task in ("H2", "RH"):
        task_items = [item for item in items if item["task"] == task]
        one_censored, one_censored_predictions = evaluate(task_items, task, "one_anchor")
        two_censored, two_censored_predictions = evaluate(task_items, task, "two_anchor")
        one_exact, one_exact_predictions = evaluate_exact(task_items, task, "one_anchor")
        two_exact, two_exact_predictions = evaluate_exact(task_items, task, "two_anchor")
        one_candidates = {"censored": one_censored, "exact": one_exact}
        two_candidates = {"censored": two_censored, "exact": two_exact}
        choose = lambda values: max(values, key=lambda name: (
            values[name]["stage_balanced_accuracy"], values[name]["exact_accuracy"],
            -values[name]["mae"]))
        one_name, two_name = choose(one_candidates), choose(two_candidates)
        reports[task] = {
            "one_anchor": one_candidates[one_name],
            "two_anchor": two_candidates[two_name],
            "selected_candidates": {"one_anchor": one_name, "two_anchor": two_name},
            "all_candidates": {"one_anchor": one_candidates, "two_anchor": two_candidates},
        }
        selected_predictions = {
            ("one_anchor", "censored"): one_censored_predictions,
            ("one_anchor", "exact"): one_exact_predictions,
            ("two_anchor", "censored"): two_censored_predictions,
            ("two_anchor", "exact"): two_exact_predictions,
        }
        predictions.extend(selected_predictions[("one_anchor", one_name)] +
                           selected_predictions[("two_anchor", two_name)])
        trajectories[task] = trajectory_data(task_items, task)
    (args.output / "metrics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    with (args.output / "normalized_trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
        rows_out = [{"task": task, **row} for task, values in trajectories.items() for row in values]
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader(); writer.writerows(rows_out)
    plot_summary(args.output, reports); plot_trajectories(args.output, trajectories)
    print(json.dumps({task: {mode: {key: report[key] for key in (
        "exact_accuracy", "stage_balanced_accuracy", "within_one_step", "mae")}
        for mode, report in task_report.items() if mode in ("one_anchor", "two_anchor")}
        for task, task_report in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
