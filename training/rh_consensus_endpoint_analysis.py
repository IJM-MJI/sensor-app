"""Leave-one-run-out RH endpoint consensus after Initial anchoring.

This experiment keeps the supplied endpoint labels unchanged.  It excludes the
single daylight recovery run so rising Reaction trajectories are not mixed with
descending hysteresis, then compares the current one-anchor logistic model with
group-balanced non-parametric consensus prototypes.
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
from sklearn.metrics import confusion_matrix

from endpoint_interval_analysis import feature_matrix, prepare
from ordinal_concentration_analysis import TASKS
from run_progress_analysis import evaluate_exact, fold_transform
from train_models import CACHE_VERSION, read_csv


RECOVERY_GROUP = "rh-daylight-recovery"
FEATURE_SETS = {
    "drop_lab": (0, 1, 2),
    "shape_reference_delta": (6, 7, 8),
    "drop_lab_plus_reference": (0, 1, 2, 6, 7, 8),
    "all_compact_features": tuple(range(9)),
}
CONSENSUS_MODES = ("median", "nearest_reference", "two_reference_mean")


def balanced(truth, prediction):
    return float(np.mean([
        np.mean(prediction[truth == level] == level) for level in sorted(set(truth))
    ]))


def metrics(truth, prediction, levels):
    matrix = confusion_matrix(truth, prediction, labels=levels)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    return {
        "exact_accuracy": float(np.mean(truth == prediction)),
        "stage_balanced_accuracy": float(np.mean(recall)),
        "within_one_stage": float(np.mean(
            np.abs(np.searchsorted(levels, truth) -
                   np.searchsorted(levels, prediction)) <= 1)),
        "mae": float(np.mean(np.abs(truth - prediction))),
        "per_stage_recall": recall.tolist(),
        "confusion": matrix.tolist(),
        "n_evaluation_frames": int(len(truth)),
    }


def group_stage_references(x, truth, exact, groups, train, feature_indices, levels):
    references = {level: [] for level in levels}
    for group in sorted(set(groups[train])):
        for level in levels:
            use = train & exact & (groups == group) & (truth == level)
            if np.any(use):
                references[level].append(np.median(x[use][:, feature_indices], axis=0))
    return references


def prototype_predict(x, references, levels, feature_indices, mode):
    values = x[:, feature_indices]
    costs = np.full((len(values), len(levels)), np.inf)
    for level_index, level in enumerate(levels):
        prototypes = np.asarray(references[level])
        if not len(prototypes):
            continue
        distances = np.sqrt(np.mean(
            (values[:, None, :] - prototypes[None, :, :]) ** 2, axis=2))
        if mode == "median":
            centre = np.median(prototypes, axis=0)
            costs[:, level_index] = np.sqrt(np.mean((values - centre) ** 2, axis=1))
        elif mode == "nearest_reference":
            costs[:, level_index] = np.min(distances, axis=1)
        else:
            count = min(2, distances.shape[1])
            costs[:, level_index] = np.mean(np.sort(distances, axis=1)[:, :count], axis=1)
    order = np.argsort(costs, axis=1)
    best = costs[np.arange(len(values)), order[:, 0]]
    second = costs[np.arange(len(values)), order[:, 1]]
    confidence = np.clip((second - best) / np.maximum(second, 1e-9), 0, 1)
    return levels[order[:, 0]], confidence


def choose_config(items, raw_x, outer_group):
    groups = np.asarray([item["group"] for item in items])
    exact = np.asarray([item["exact"] is not None for item in items])
    evaluate = np.asarray([item["evaluate"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    levels = np.asarray(TASKS["RH"]["levels"], dtype=float)
    outer_train = groups != outer_group
    candidates = []
    for feature_name, feature_indices in FEATURE_SETS.items():
        feature_indices = np.asarray(feature_indices)
        for mode in CONSENSUS_MODES:
            fold_scores = []
            for inner_group in sorted(set(groups[outer_train])):
                train = outer_train & (groups != inner_group)
                test = outer_train & (groups == inner_group) & evaluate
                x, _ = fold_transform(items, "RH", raw_x, train, "one_anchor")
                references = group_stage_references(
                    x, truth, exact, groups, train, feature_indices, levels)
                prediction, _ = prototype_predict(
                    x[test], references, levels, feature_indices, mode)
                fold_scores.append((
                    balanced(truth[test], prediction),
                    float(np.mean(truth[test] == prediction)),
                    -float(np.mean(np.abs(truth[test] - prediction))),
                ))
            score = tuple(np.mean(fold_scores, axis=0))
            candidates.append((score, feature_name, mode))
    return max(candidates)[1:]


def evaluate_consensus(items):
    raw_x = feature_matrix(items, "RH")
    groups = np.asarray([item["group"] for item in items])
    exact = np.asarray([item["exact"] is not None for item in items])
    evaluate = np.asarray([item["evaluate"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    levels = np.asarray(TASKS["RH"]["levels"], dtype=float)
    prediction = np.full(len(items), np.nan)
    confidence = np.full(len(items), np.nan)
    choices = []
    for held_out in sorted(set(groups)):
        feature_name, mode = choose_config(items, raw_x, held_out)
        feature_indices = np.asarray(FEATURE_SETS[feature_name])
        train = groups != held_out
        test = (groups == held_out) & evaluate
        x, anchors = fold_transform(items, "RH", raw_x, train, "one_anchor")
        references = group_stage_references(
            x, truth, exact, groups, train, feature_indices, levels)
        prediction[test], confidence[test] = prototype_predict(
            x[test], references, levels, feature_indices, mode)
        choices.append({
            "held_out_group": held_out, "feature_set": feature_name,
            "consensus_mode": mode, **anchors[held_out],
        })
    use = evaluate & ~np.isnan(prediction)
    report = metrics(truth[use], prediction[use], levels)
    report.update({
        "policy": "Reaction-only, Initial-centred, nested consensus prototypes",
        "outer_fold_choices": choices,
    })
    rows = [{
        "group": item["group"], "video": item["video"], "time": item["time"],
        "source": item["source"], "reference": truth[index],
        "prediction": prediction[index], "confidence": confidence[index],
    } for index, item in enumerate(items) if use[index]]
    return report, rows


def plot(output, baseline, consensus):
    labels = TASKS["RH"]["display_levels"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True)
    for axis, title, report in (
            (axes[0], "Reaction-only one-anchor logistic", baseline),
            (axes[1], "Reaction-only consensus prototypes", consensus)):
        matrix = np.asarray(report["confusion"], dtype=float)
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for row in range(len(matrix)):
            for column in range(len(matrix)):
                axis.text(column, row, f"{normalized[row, column]:.2f}",
                          ha="center", va="center", fontsize=7,
                          color="white" if normalized[row, column] > .55 else "black")
        axis.set_xticks(range(len(labels)), labels, rotation=35)
        axis.set_yticks(range(len(labels)), labels)
        axis.set(xlabel="Predicted RH", ylabel="Endpoint reference", title=title)
        axis.text(.5, -.26,
                  f"exact={report['exact_accuracy']:.3f}  "
                  f"balanced={report['stage_balanced_accuracy']:.3f}  "
                  f"MAE={report['mae']:.2f}%RH",
                  transform=axis.transAxes, ha="center", fontsize=9)
    fig.suptitle("RH rising-Reaction endpoint held-out validation", fontweight="bold")
    fig.savefig(output / "rh_consensus_endpoint_validation.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_consensus_endpoint_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    all_rows = read_csv(args.cache)
    items = [item for item in prepare(all_rows)
             if item["task"] == "RH" and item["group"] != RECOVERY_GROUP]
    baseline, baseline_rows = evaluate_exact(items, "RH", "one_anchor")
    consensus, consensus_rows = evaluate_consensus(items)
    report = {
        "evaluation_scope": "rising H2O-only Reaction; daylight Recovery excluded",
        "endpoint_labels_changed": False,
        "independent_groups": sorted(set(item["group"] for item in items)),
        "baseline_one_anchor_logistic": baseline,
        "consensus_prototype": consensus,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    combined = ([{"model": "one_anchor_logistic", **row} for row in baseline_rows]
                + [{"model": "consensus_prototype", **row} for row in consensus_rows])
    with (args.output / "predictions.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader(); writer.writerows(combined)
    plot(args.output, baseline, consensus)
    print(json.dumps({
        "baseline": {key: baseline[key] for key in (
            "exact_accuracy", "stage_balanced_accuracy", "within_one_step", "mae",
            "per_stage_recall")},
        "consensus": {key: consensus[key] for key in (
            "exact_accuracy", "stage_balanced_accuracy", "within_one_stage", "mae",
            "per_stage_recall")},
        "fold_choices": consensus["outer_fold_choices"],
    }, indent=2))


if __name__ == "__main__":
    main()
