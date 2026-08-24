"""Constrained asymmetric gating for the leakage-safe H2 2/3 expert.

The gate may change only global 2%/3% predictions.  A candidate must improve
2% recall without reducing 3% recall, exact accuracy, or MAE on the meta-
training videos.  Thresholds are selected without using the held-out video's
labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from combine_h2_23_expert import read_global
from h2_23_pairwise_analysis import VERIFIED
from ordinal_concentration_analysis import (
    apply_h2_verified_partial_response, assign_h2_ramp_targets, augment,
)
from train_models import CACHE_VERSION, read_csv


THRESHOLDS = (.50, .55, .60, .65, .70, .75, .80, .85, .90, .95, .99, 1.01)


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    cm = confusion_matrix(truth, prediction, labels=range(5))
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return {
        "exact_accuracy": float(np.mean(truth == prediction)),
        "stage_balanced_accuracy": float(np.mean(recall)),
        "within_one_step": float(np.mean(np.abs(truth - prediction) <= 1)),
        "mae": float(np.mean(np.abs(truth - prediction))),
        "recall": recall.tolist(),
        "confusion": cm.tolist(),
    }


def apply_gate(baseline, probability_2, probability_3, threshold_23, threshold_32):
    """Apply separate confidence requirements to 2->3 and 3->2 changes."""
    result = baseline.copy()
    result[(baseline == 2) & (probability_3 >= threshold_23)] = 3
    result[(baseline == 3) & (probability_2 >= threshold_32)] = 2
    return result


def passes_constraints(candidate: dict, baseline: dict) -> bool:
    eps = 1e-12
    return (
        candidate["recall"][3] + eps >= baseline["recall"][3]
        and candidate["recall"][2] > baseline["recall"][2] + eps
        and candidate["exact_accuracy"] + eps >= baseline["exact_accuracy"]
        and candidate["mae"] <= baseline["mae"] + eps
    )


def rank(candidate: dict, baseline: dict):
    """Maximise 2% gain, then balanced/exact accuracy, then minimise MAE."""
    return (
        candidate["recall"][2] - baseline["recall"][2],
        candidate["stage_balanced_accuracy"],
        candidate["exact_accuracy"],
        -candidate["mae"],
        candidate["recall"][3],
    )


def plot_comparison(path: Path, baseline_metrics: dict, candidate_metrics: dict):
    """Save a compact diagnostic comparison; the candidate is not deployable."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, title, report in (
            (axes[0], "Corrected global baseline", baseline_metrics),
            (axes[1], "Best fixed gate (diagnostic only)", candidate_metrics)):
        matrix = np.asarray(report["confusion"])
        image = axis.imshow(matrix, cmap="Blues", vmin=0,
                            vmax=max(np.asarray(baseline_metrics["confusion"]).max(),
                                     np.asarray(candidate_metrics["confusion"]).max()))
        for row in range(5):
            for column in range(5):
                axis.text(column, row, str(matrix[row, column]), ha="center",
                          va="center", color="white" if matrix[row, column] > 65 else "black")
        axis.set(title=title, xlabel="Predicted H2 (%)", ylabel="Reference H2 (%)")
        axis.set_xticks(range(5)); axis.set_yticks(range(5))
        axis.text(.5, -0.22,
                  f"exact={report['exact_accuracy']:.3f}  "
                  f"R2={report['recall'][2]:.3f}  R3={report['recall'][3]:.3f}  "
                  f"MAE={report['mae']:.3f}",
                  transform=axis.transAxes, ha="center", fontsize=9)
    fig.colorbar(image, ax=axes, shrink=.8, label="Frames")
    fig.suptitle("H2 2/3 asymmetric-gate constraint test", fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features.csv"))
    parser.add_argument("--global-predictions", type=Path, default=Path(
        "training/output/h2_verified_partial_w0p002_v12/predictions.csv"))
    parser.add_argument("--global-model", default="ridge_flexible_rounded")
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/h2_verified_23_expert_constrained_v14"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cache = read_csv(args.cache)
    assign_h2_ramp_targets(cache)
    apply_h2_verified_partial_response(cache, "verified_run4_run5", .002)
    by_key = {(str(row["video"]), round(float(row["time"]), 4)): row
              for row in cache}
    training = [row for row in cache if str(row["video"]) in VERIFIED
                and row.get("analysis_phase") == "reaction"
                and int(float(row["analysis_stage"])) in (2, 3)]
    global_rows = read_global(args.global_predictions, args.global_model)

    groups = np.asarray([row["group"] for row in global_rows])
    probability_2 = np.zeros(len(global_rows), dtype=float)
    probability_3 = np.zeros(len(global_rows), dtype=float)
    for held_out in sorted(set(groups)):
        train = [row for row in training if str(row["group"]) != held_out]
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=.5, class_weight="balanced", max_iter=3000, random_state=42))
        model.fit(np.asarray([augment(row, "flame") for row in train]),
                  np.asarray([int(float(row["analysis_stage"])) for row in train]))
        indices = np.where(groups == held_out)[0]
        test = [by_key[(global_rows[index]["video"],
                        round(float(global_rows[index]["time"]), 4))]
                for index in indices]
        probabilities = model.predict_proba(
            np.asarray([augment(row, "flame") for row in test]))
        classes = list(model.classes_)
        probability_2[indices] = probabilities[:, classes.index(2)]
        probability_3[indices] = probabilities[:, classes.index(3)]

    truth = np.asarray([int(float(row["reference"])) for row in global_rows])
    baseline = np.asarray([int(float(row["prediction"])) for row in global_rows])
    baseline_metrics = metrics(truth, baseline)

    # Honest nested selection: the held-out video's labels never select its gate.
    final = baseline.copy()
    selected_by_group = {}
    for held_out in sorted(set(groups)):
        meta = groups != held_out
        meta_baseline = metrics(truth[meta], baseline[meta])
        feasible = []
        for threshold_23 in THRESHOLDS:
            for threshold_32 in THRESHOLDS:
                candidate_prediction = apply_gate(
                    baseline[meta], probability_2[meta], probability_3[meta],
                    threshold_23, threshold_32)
                candidate_metrics = metrics(truth[meta], candidate_prediction)
                if passes_constraints(candidate_metrics, meta_baseline):
                    feasible.append((rank(candidate_metrics, meta_baseline),
                                     threshold_23, threshold_32, candidate_metrics))
        indices = groups == held_out
        if feasible:
            _, threshold_23, threshold_32, meta_metrics = max(feasible)
            final[indices] = apply_gate(
                baseline[indices], probability_2[indices], probability_3[indices],
                threshold_23, threshold_32)
            selected_by_group[held_out] = {
                "feasible": True, "threshold_2_to_3": threshold_23,
                "threshold_3_to_2": threshold_32,
                "meta_training_metrics": meta_metrics,
            }
        else:
            selected_by_group[held_out] = {
                "feasible": False, "decision": "keep global predictions",
            }

    final_metrics = metrics(truth, final)

    # Diagnostic only: fixed thresholds selected using all cross-fitted labels.
    # This cannot be deployed as an unbiased result, but answers whether any
    # fixed asymmetric pair can satisfy the requested aggregate constraints.
    grid_rows = []
    feasible_global = []
    for threshold_23 in THRESHOLDS:
        for threshold_32 in THRESHOLDS:
            candidate_prediction = apply_gate(
                baseline, probability_2, probability_3,
                threshold_23, threshold_32)
            candidate_metrics = metrics(truth, candidate_prediction)
            passed = passes_constraints(candidate_metrics, baseline_metrics)
            row = {
                "threshold_2_to_3": threshold_23,
                "threshold_3_to_2": threshold_32,
                "passes_constraints": passed,
                "exact_accuracy": candidate_metrics["exact_accuracy"],
                "mae": candidate_metrics["mae"],
                "recall_2": candidate_metrics["recall"][2],
                "recall_3": candidate_metrics["recall"][3],
            }
            grid_rows.append(row)
            if passed:
                feasible_global.append((rank(candidate_metrics, baseline_metrics),
                                        row, candidate_metrics))

    best_global = None
    best_global_prediction = None
    if feasible_global:
        _, best_row, best_metrics = max(feasible_global, key=lambda item: item[0])
        best_global_prediction = apply_gate(
            baseline, probability_2, probability_3,
            float(best_row["threshold_2_to_3"]),
            float(best_row["threshold_3_to_2"]))
        best_global = {"thresholds": best_row, "metrics": best_metrics,
                       "warning": "diagnostic only; selected on evaluation labels"}
        best_global["per_held_out_group"] = {
            group: metrics(truth[groups == group], best_global_prediction[groups == group])
            for group in sorted(set(groups))
        }
        best_global["changed_frames_by_group"] = {
            group: {
                "total": int(np.sum(best_global_prediction[groups == group]
                                    != baseline[groups == group])),
                "3_to_2": int(np.sum((groups == group) & (baseline == 3)
                                     & (best_global_prediction == 2))),
                "2_to_3": int(np.sum((groups == group) & (baseline == 2)
                                     & (best_global_prediction == 3))),
            }
            for group in sorted(set(groups))
        }
        plot_comparison(args.output / "constraint_comparison.png",
                        baseline_metrics, best_metrics)

    per_group = {}
    for group in sorted(set(groups)):
        use = groups == group
        per_group[group] = {
            "baseline": metrics(truth[use], baseline[use]),
            "constrained": metrics(truth[use], final[use]),
        }

    report = {
        "protocol": "nested LOVO asymmetric 2/3 expert with hard constraints",
        "constraints": {
            "recall_3": "not below fold-specific global baseline",
            "recall_2": "strictly above fold-specific global baseline",
            "exact_accuracy": "not below fold-specific global baseline",
            "mae": "not above fold-specific global baseline",
        },
        "n_frames": len(truth),
        "baseline": baseline_metrics,
        "constrained_nested_lovo": final_metrics,
        "selected_by_held_out_group": selected_by_group,
        "all_folds_found_feasible_gate": all(
            item["feasible"] for item in selected_by_group.values()),
        "aggregate_constraints_passed": passes_constraints(
            final_metrics, baseline_metrics),
        "diagnostic_best_fixed_gate": best_global,
        "per_held_out_group": per_group,
        "changed_frames": int(np.sum(final != baseline)),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    with (args.output / "threshold_grid.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid_rows[0]))
        writer.writeheader()
        writer.writerows(grid_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
