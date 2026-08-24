"""End-to-end RH coarse-band gate plus 40/50/60 expert.

Place-1 nominal 90% is treated as a 70--80% partial response, while place-2
response runs retain exact 90% endpoints.  All fitting and gate/expert
predictions use complete-run holdout folds.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from endpoint_interval_analysis import feature_matrix, prepare
from ordinal_concentration_analysis import TASKS
from rh_location_mixture_analysis import (
    FEATURE_SETS, MIDDLE_LEVELS, global_middle_prototypes, middle_predict,
)
from run_progress_analysis import evaluate_exact, fold_transform
from train_models import CACHE_VERSION, read_csv


LEVELS = np.asarray(TASKS["RH"]["levels"], dtype=float)
BAND_NAMES = ("20-30", "40-60", "70-90")
PLACE1_PARTIAL_90_GROUP = "rh-indoor-fast"
RECOVERY_GROUP = "rh-daylight-recovery"


def apply_verified_high_policy(items):
    """Remove place-1 nominal 90 from exact fine labels; retain 70--80 bounds."""
    corrected = []
    changes = []
    for original in items:
        item = dict(original)
        if (item["group"] == PLACE1_PARTIAL_90_GROUP
                and item["exact"] == 90.0):
            changes.append({
                "group": item["group"], "video": item["video"],
                "time": item["time"], "old_exact": 90.0,
                "new_bounds": [70.0, 80.0],
            })
            item.update({"lower": 70.0, "upper": 80.0, "exact": None,
                         "source": "verified_partial_70_80", "evaluate": False})
        corrected.append(item)
    return corrected, changes


def band_from_bounds(lower, upper):
    if upper <= 25:
        return 0
    if lower >= 40 and upper <= 60:
        return 1
    if lower >= 70:
        return 2
    return None


def fine_report(truth, prediction):
    matrix = confusion_matrix(truth, prediction, labels=LEVELS)
    recall = np.asarray([
        np.mean(prediction[truth == level] == level) for level in LEVELS
    ])
    return {
        "exact_accuracy": float(np.mean(truth == prediction)),
        "stage_balanced_accuracy": float(np.mean(recall)),
        "within_one_stage": float(np.mean(
            np.abs(np.searchsorted(LEVELS, truth) -
                   np.searchsorted(LEVELS, prediction)) <= 1)),
        "mae": float(np.mean(np.abs(truth - prediction))),
        "per_stage_recall": recall.tolist(), "confusion": matrix.tolist(),
        "n_frames": int(len(truth)),
    }


def band_report(truth, prediction):
    labels = np.arange(3)
    matrix = confusion_matrix(truth, prediction, labels=labels)
    recall = np.asarray([
        np.mean(prediction[truth == level] == level) for level in labels
    ])
    return {
        "exact_accuracy": float(np.mean(truth == prediction)),
        "balanced_accuracy": float(np.mean(recall)),
        "per_band_recall": {name: float(value)
                            for name, value in zip(BAND_NAMES, recall)},
        "confusion": matrix.tolist(), "n_frames": int(len(truth)),
    }


def known_band_labels(items):
    return np.asarray([
        -1 if band_from_bounds(float(item["lower"]), float(item["upper"])) is None
        else band_from_bounds(float(item["lower"]), float(item["upper"]))
        for item in items
    ], dtype=int)


def group_band_weights(groups, labels, train):
    weights = np.zeros(len(groups), dtype=float)
    for group in sorted(set(groups[train])):
        for label in range(3):
            use = train & (groups == group) & (labels == label)
            if np.any(use):
                weights[use] = 1.0 / np.sum(use)
    return weights


def score_band(truth, prediction):
    recalls = [np.mean(prediction[truth == label] == label) for label in range(3)]
    return (float(np.mean(recalls)), float(np.mean(truth == prediction)))


def tune_gate(items, raw_x, groups, band_labels, evaluate, outer_group):
    outer_train = groups != outer_group
    candidates = []
    for feature_name, feature_indices_raw in FEATURE_SETS.items():
        feature_indices = np.asarray(feature_indices_raw)
        for C in (.03, .1, .3, 1.0, 3.0):
            scores = []
            for inner_group in sorted(set(groups[outer_train])):
                train = outer_train & (groups != inner_group) & (band_labels >= 0)
                test = (outer_train & (groups == inner_group) & evaluate
                        & (band_labels >= 0))
                x, _ = fold_transform(items, "RH", raw_x, train, "one_anchor")
                weights = group_band_weights(groups, band_labels, train)
                model = LogisticRegression(
                    C=C, max_iter=3000, class_weight="balanced", random_state=42)
                model.fit(x[train][:, feature_indices], band_labels[train],
                          sample_weight=weights[train])
                prediction = model.predict(x[test][:, feature_indices])
                scores.append(score_band(band_labels[test], prediction))
            candidates.append((tuple(np.mean(scores, axis=0)), feature_name, C))
    return max(candidates)[1:]


def cross_fitted_predictions(items):
    raw_x = feature_matrix(items, "RH")
    groups = np.asarray([item["group"] for item in items])
    fine_truth = np.asarray([
        np.nan if item["exact"] is None else float(item["exact"]) for item in items
    ])
    exact = ~np.isnan(fine_truth)
    evaluate = np.asarray([bool(item["evaluate"]) for item in items])
    band_labels = known_band_labels(items)
    gate_prediction = np.full(len(items), -1, dtype=int)
    gate_confidence = np.full(len(items), np.nan)
    middle_prediction = np.full(len(items), np.nan)
    middle_confidence = np.full(len(items), np.nan)
    choices = []
    middle_features = np.asarray(FEATURE_SETS["drop_lab"])
    for held_out in sorted(set(groups)):
        feature_name, C = tune_gate(
            items, raw_x, groups, band_labels, evaluate, held_out)
        feature_indices = np.asarray(FEATURE_SETS[feature_name])
        train_gate = (groups != held_out) & (band_labels >= 0)
        test = (groups == held_out) & evaluate & exact
        x, _ = fold_transform(items, "RH", raw_x, train_gate, "one_anchor")
        weights = group_band_weights(groups, band_labels, train_gate)
        gate = LogisticRegression(
            C=C, max_iter=3000, class_weight="balanced", random_state=42)
        gate.fit(x[train_gate][:, feature_indices], band_labels[train_gate],
                 sample_weight=weights[train_gate])
        probabilities = gate.predict_proba(x[test][:, feature_indices])
        gate_prediction[test] = gate.classes_[np.argmax(probabilities, axis=1)]
        gate_confidence[test] = np.max(probabilities, axis=1)

        references = global_middle_prototypes(
            x, groups, fine_truth, exact, groups != held_out,
            middle_features)
        middle_prediction[test], middle_confidence[test] = middle_predict(
            x[test], references, middle_features)
        choices.append({"held_out_group": held_out,
                        "gate_feature_set": feature_name, "C": C})
    return (fine_truth, evaluate, band_labels, gate_prediction, gate_confidence,
            middle_prediction, middle_confidence, choices)


def plot(output, gate, baseline, hierarchy):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), constrained_layout=True)
    gate_matrix = np.asarray(gate["confusion"], dtype=float)
    gate_norm = gate_matrix / np.maximum(gate_matrix.sum(axis=1, keepdims=True), 1)
    axes[0].imshow(gate_norm, vmin=0, vmax=1, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, f"{gate_norm[row, column]:.2f}", ha="center",
                         va="center", color="white" if gate_norm[row, column] > .55 else "black")
    axes[0].set_xticks(range(3), BAND_NAMES); axes[0].set_yticks(range(3), BAND_NAMES)
    axes[0].set(xlabel="Predicted band", ylabel="Reference band", title="Coarse gate")
    for axis, title, report in ((axes[1], "Fine baseline", baseline),
                                (axes[2], "Gate + middle expert", hierarchy)):
        matrix = np.asarray(report["confusion"], dtype=float)
        norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        axis.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for row in range(7):
            for column in range(7):
                axis.text(column, row, f"{norm[row, column]:.1f}", ha="center", va="center",
                          fontsize=6, color="white" if norm[row, column] > .55 else "black")
        labels = TASKS["RH"]["display_levels"]
        axis.set_xticks(range(7), labels, rotation=35); axis.set_yticks(range(7), labels)
        axis.set(xlabel="Predicted RH", ylabel="Reference RH", title=title)
        axis.text(.5, -.26, f"exact={report['exact_accuracy']:.3f}  "
                  f"balanced={report['stage_balanced_accuracy']:.3f}  "
                  f"MAE={report['mae']:.2f}%RH",
                  transform=axis.transAxes, ha="center", fontsize=8)
    fig.suptitle("RH complete-run held-out hierarchy", fontweight="bold")
    fig.savefig(output / "rh_coarse_middle_hierarchy.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_coarse_middle_hierarchy_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.cache)
    base_items = [item for item in prepare(rows)
                  if item["task"] == "RH" and item["group"] != RECOVERY_GROUP]
    items, label_changes = apply_verified_high_policy(base_items)
    baseline_metrics_raw, baseline_rows = evaluate_exact(items, "RH", "one_anchor")
    baseline_lookup = {(row["group"], round(float(row["time"]), 6)):
                       float(row["prediction"]) for row in baseline_rows}

    (truth, evaluate, band_labels, gate_prediction, gate_confidence,
     middle_prediction, middle_confidence, choices) = cross_fitted_predictions(items)
    groups = np.asarray([item["group"] for item in items])
    times = np.asarray([float(item["time"]) for item in items])
    use = evaluate & ~np.isnan(truth) & (gate_prediction >= 0)
    baseline_prediction = np.asarray([
        baseline_lookup.get((groups[index], round(times[index], 6)), np.nan)
        for index in range(len(items))
    ])
    use &= ~np.isnan(baseline_prediction)
    hierarchy_prediction = baseline_prediction.copy()
    route_middle = use & (gate_prediction == 1)
    hierarchy_prediction[route_middle] = middle_prediction[route_middle]

    true_band = np.asarray([band_from_bounds(value, value) if np.isfinite(value) else -1
                            for value in truth], dtype=int)
    gate_metrics = band_report(true_band[use], gate_prediction[use])
    baseline = fine_report(truth[use], baseline_prediction[use])
    hierarchy = fine_report(truth[use], hierarchy_prediction[use])
    preserve_middle = np.all(
        np.asarray(hierarchy["per_stage_recall"])[1:4]
        >= np.asarray(baseline["per_stage_recall"])[1:4])
    preserve_high = np.all(
        np.asarray(hierarchy["per_stage_recall"])[4:]
        >= np.asarray(baseline["per_stage_recall"])[4:])
    preserve_all = np.all(
        np.asarray(hierarchy["per_stage_recall"])
        >= np.asarray(baseline["per_stage_recall"]))
    improve_middle = np.any(
        np.asarray(hierarchy["per_stage_recall"])[1:4]
        > np.asarray(baseline["per_stage_recall"])[1:4])
    middle_gate_ready = gate_metrics["per_band_recall"]["40-60"] >= .85
    deploy = bool(preserve_all and improve_middle and middle_gate_ready
                  and hierarchy["exact_accuracy"] >= baseline["exact_accuracy"]
                  and hierarchy["within_one_stage"] >= baseline["within_one_stage"]
                  and hierarchy["mae"] <= baseline["mae"])
    result = {
        "scope": "rising H2O-only Reaction; full run held out",
        "verified_label_policy": {
            "place1_nominal_90": "70-80 partial; excluded from exact fine score",
            "place2_90": "exact 90 retained", "changes": label_changes,
        },
        "coarse_gate": gate_metrics,
        "coarse_gate_outer_choices": choices,
        "fine_baseline": baseline,
        "gate_plus_middle_expert": hierarchy,
        "preserves_40_50_60": bool(preserve_middle),
        "preserves_70_80_90": bool(preserve_high),
        "preserves_every_stage": bool(preserve_all),
        "improves_at_least_one_middle_stage": bool(improve_middle),
        "middle_gate_recall_at_least_0.85": bool(middle_gate_ready),
        "deployment_conditions_passed": deploy,
        "routed_to_middle_frames": int(np.sum(route_middle)),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    prediction_rows = []
    for index in np.where(use)[0]:
        prediction_rows.append({
            "group": groups[index], "video": items[index]["video"],
            "time": times[index], "reference": truth[index],
            "reference_band": BAND_NAMES[true_band[index]],
            "gate_prediction": BAND_NAMES[gate_prediction[index]],
            "gate_confidence": gate_confidence[index],
            "baseline_prediction": baseline_prediction[index],
            "middle_expert_prediction": middle_prediction[index],
            "middle_expert_confidence": middle_confidence[index],
            "hierarchy_prediction": hierarchy_prediction[index],
        })
    with (args.output / "predictions.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader(); writer.writerows(prediction_rows)
    plot(args.output, gate_metrics, baseline, hierarchy)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
