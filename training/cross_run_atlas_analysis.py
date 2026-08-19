"""Match concentration trajectories to a trusted cross-run optical atlas.

Only the reference run timelines build the atlas. Candidate timelines remain
hidden during matching and are opened afterwards solely for evaluation.
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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from ordinal_concentration_analysis import (
    H2_RAMP_ENDPOINTS, RH_RAMP_ENDPOINTS, TASKS,
    assign_h2_ramp_targets, assign_rh_ramp_targets,
)
from train_models import CACHE_VERSION, read_csv


H2_REFERENCE_GROUP = "h2-test-2"
RH_REFERENCE_GROUP = "rh-indoor-long"
RH_90_REFERENCE_GROUPS = {"rh-response-3", "rh-response-6"}


def value(row, name, default=0.0):
    item = row.get(name)
    return float(default if item in (None, "") else item)


def trajectory_feature(row, task):
    if task == "H2":
        flame = np.asarray([value(row, f"flame_{c}") for c in "Lab"])
        drop = np.asarray([value(row, f"drop_{c}") for c in "Lab"])
        percentiles = [value(row, f"flame_chroma_p{p}") for p in (10, 25, 50, 75, 90)]
        return np.asarray([*flame, *(flame - drop), *percentiles], dtype=float)
    drop = np.asarray([value(row, f"drop_registered_{c}", value(row, f"drop_{c}"))
                       for c in "Lab"])
    flame = np.asarray([value(row, f"flame_{c}") for c in "Lab"])
    percentiles = [value(row, f"drop_chroma_p{p}") for p in (10, 25, 50, 75, 90)]
    return np.asarray([*drop, *(drop - flame), *percentiles], dtype=float)


def median_near(rows, label, target, tolerance):
    selected = [trajectory_feature(row, label) for row in rows
                if abs(float(row["_reference_value"]) - target) <= tolerance]
    if not selected:
        return None
    return np.median(np.asarray(selected), axis=0)


def build_h2_atlas(rows):
    reference = [row for row in rows if row["group"] == H2_REFERENCE_GROUP
                 and row.get("analysis_phase") == "reaction" and row.get("h2_value") is not None]
    for row in reference:
        row["_reference_value"] = float(row["h2_value"])
    levels = np.linspace(0, 4, 81)
    features = []
    for level in levels:
        feature = median_near(reference, "H2", level, .035)
        if feature is None:
            nearest = min(reference, key=lambda row: abs(float(row["h2_value"]) - level))
            feature = trajectory_feature(nearest, "H2")
        features.append(feature)
    return levels, np.asarray(features), reference


def build_rh_atlas(rows):
    long_rows = [row for row in rows if row["group"] == RH_REFERENCE_GROUP
                 and row.get("rh_value") is not None and float(row["rh_value"]) <= 80]
    for row in long_rows:
        row["_reference_value"] = float(row["rh_value"])
    high_rows = [row for row in rows if row["group"] in RH_90_REFERENCE_GROUPS
                 and row.get("rh_value") is not None and float(row["rh_value"]) >= 89.5]
    for row in high_rows:
        row["_reference_value"] = 90.0
    levels = np.linspace(20, 90, 141)
    features = []
    high = np.median(np.asarray([trajectory_feature(row, "RH") for row in high_rows]), axis=0)
    feature_80 = median_near(long_rows, "RH", 80.0, .3)
    for level in levels:
        if level <= 80:
            feature = median_near(long_rows, "RH", level, .3)
            if feature is None:
                nearest = min(long_rows, key=lambda row: abs(float(row["rh_value"]) - level))
                feature = trajectory_feature(nearest, "RH")
        else:
            fraction = (level - 80) / 10
            feature = (1 - fraction) * feature_80 + fraction * high
        features.append(feature)
    return levels, np.asarray(features), long_rows + high_rows


def robust_scale(reference_features):
    differences = np.diff(reference_features, axis=0)
    scale = np.percentile(np.abs(differences), 75, axis=0)
    global_scale = np.percentile(np.abs(reference_features - reference_features[0]), 75, axis=0)
    return np.maximum(scale * 4, np.maximum(global_scale * .12, .15))


def cost_matrix(candidate_features, atlas_features):
    scale = robust_scale(atlas_features)
    delta = (candidate_features[:, None, :] - atlas_features[None, :, :]) / scale
    euclidean = np.sqrt(np.mean(delta ** 2, axis=2))
    # Direction from Initial is useful while retaining magnitude, so a partial
    # response is not automatically stretched to the atlas maximum.
    c = candidate_features - candidate_features[0]
    a = atlas_features - atlas_features[0]
    dot = c @ a.T
    norms = np.linalg.norm(c, axis=1)[:, None] * np.linalg.norm(a, axis=1)[None, :]
    cosine_cost = 1 - np.divide(dot, norms, out=np.zeros_like(dot), where=norms > 1e-9)
    cosine_cost[0, :] = 0
    return euclidean + .25 * cosine_cost


def monotonic_match(cost, direction="up", jump_penalty=.025, start_penalty=.10):
    if direction == "down":
        reversed_indices = monotonic_match(cost[:, ::-1], "up", jump_penalty, start_penalty)
        return cost.shape[1] - 1 - reversed_indices
    n, m = cost.shape
    back = np.zeros((n, m), dtype=np.int32)
    previous = cost[0] + start_penalty * np.arange(m)
    for i in range(1, n):
        adjusted = previous - jump_penalty * np.arange(m)
        prefix_value = np.minimum.accumulate(adjusted)
        prefix_index = np.zeros(m, dtype=np.int32)
        best = 0
        for j in range(m):
            if adjusted[j] < adjusted[best]:
                best = j
            prefix_index[j] = best
        current = cost[i] + jump_penalty * np.arange(m) + prefix_value
        back[i] = prefix_index
        previous = current
    indices = np.zeros(n, dtype=np.int32); indices[-1] = int(np.argmin(previous))
    for i in range(n - 1, 0, -1):
        indices[i - 1] = back[i, indices[i]]
    return indices


def candidates(rows, task):
    if task == "H2":
        return [row for row in rows if row["kind"] == "h2_only"
                and row["group"] != H2_REFERENCE_GROUP
                and row.get("analysis_phase") == "reaction"]
    return [row for row in rows if row["kind"] == "rh_only"
            and row["group"] != RH_REFERENCE_GROUP and row.get("rh_value") is not None]


def match_groups(rows, task, atlas_levels, atlas_features):
    by_group = defaultdict(list)
    for row in candidates(rows, task):
        by_group[str(row["group"])].append(row)
    output = []
    for group, group_rows in sorted(by_group.items()):
        group_rows.sort(key=lambda row: (str(row["video"]), float(row["time"])))
        # RH recovery is the only descending single-condition trajectory.
        direction = "down" if task == "RH" and group == "rh-daylight-recovery" else "up"
        features = np.asarray([trajectory_feature(row, task) for row in group_rows])
        costs = cost_matrix(features, atlas_features)
        indices = monotonic_match(costs, direction)
        atlas_stages = nearest_stage(task, atlas_levels)
        stage_levels = np.asarray(TASKS[task]["levels"], dtype=float)
        mutual_candidate = np.argmin(costs, axis=0)
        cost_limit = float(np.median(costs[np.arange(len(indices)), indices]))
        for candidate_index, (row, index, row_cost) in enumerate(zip(group_rows, indices, costs)):
            stage_costs = np.asarray([np.min(row_cost[atlas_stages == stage])
                                      for stage in stage_levels])
            order = np.argsort(stage_costs)
            confidence = float(np.clip((stage_costs[order[1]] - stage_costs[order[0]]) /
                                       max(stage_costs[order[1]], 1e-9), 0, 1))
            mutual = bool(mutual_candidate[index] == candidate_index)
            eligible = bool(mutual and confidence >= .10 and row_cost[index] <= cost_limit)
            truth = float(row["h2_value"] if task == "H2" else row["rh_value"])
            endpoints = (H2_RAMP_ENDPOINTS if task == "H2" else RH_RAMP_ENDPOINTS).get(
                str(row["video"]), [])
            is_endpoint = any(abs(float(row["time"]) - float(endpoint_time)) <= .55
                              for endpoint_time, _ in endpoints)
            output.append({
                "task": task, "group": group, "video": row["video"],
                "time": float(row["time"]), "hidden_timeline_value": truth,
                "optical_equivalent": float(atlas_levels[index]),
                "match_cost": float(row_cost[index]), "local_confidence": confidence,
                "mutual_nearest": mutual, "pseudo_label_eligible": eligible,
                "is_timeline_endpoint": is_endpoint,
                "direction": direction,
            })
    return output


def nearest_stage(task, values):
    levels = np.asarray(TASKS[task]["levels"], dtype=float)
    values = np.asarray(values, dtype=float)
    return levels[np.argmin(abs(values[:, None] - levels[None, :]), axis=1)]


def metrics_and_plot(matches, task, output):
    levels = np.asarray(TASKS[task]["levels"], dtype=float)
    truth = nearest_stage(task, [row["hidden_timeline_value"] for row in matches])
    prediction = nearest_stage(task, [row["optical_equivalent"] for row in matches])
    groups = np.asarray([row["group"] for row in matches])
    cm = confusion_matrix(truth, prediction, labels=levels)
    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    report = {
        "reference": H2_REFERENCE_GROUP if task == "H2" else RH_REFERENCE_GROUP,
        "candidate_timeline_used_for_matching": False,
        "n_candidate_frames": len(matches), "n_candidate_groups": len(set(groups)),
        "exact_accuracy": float(np.mean(truth == prediction)),
        "stage_balanced_accuracy": float(np.mean(recalls)),
        "within_one_stage": float(np.mean(abs(np.searchsorted(levels, truth) -
                                                   np.searchsorted(levels, prediction)) <= 1)),
        "mae": float(np.mean(abs(truth - prediction))),
        "per_stage_recall": recalls.tolist(), "confusion": cm.tolist(),
        "per_group": {},
    }
    eligible = np.asarray([str(row["pseudo_label_eligible"]).lower() == "true"
                           if isinstance(row["pseudo_label_eligible"], str)
                           else bool(row["pseudo_label_eligible"]) for row in matches])
    report["high_confidence_pseudo_labels"] = {
        "n": int(eligible.sum()), "coverage": float(np.mean(eligible)),
        "exact_accuracy_after_timeline_reveal": (
            float(np.mean(truth[eligible] == prediction[eligible])) if np.any(eligible) else None),
        "within_one_stage_after_timeline_reveal": (
            float(np.mean(abs(np.searchsorted(levels, truth[eligible]) -
                              np.searchsorted(levels, prediction[eligible])) <= 1))
            if np.any(eligible) else None),
    }
    endpoints = np.asarray([bool(row["is_timeline_endpoint"]) for row in matches])
    report["endpoint_only_after_timeline_reveal"] = {
        "n": int(endpoints.sum()),
        "exact_accuracy": float(np.mean(truth[endpoints] == prediction[endpoints])),
        "within_one_stage": float(np.mean(
            abs(np.searchsorted(levels, truth[endpoints]) -
                np.searchsorted(levels, prediction[endpoints])) <= 1)),
        "mae": float(np.mean(abs(truth[endpoints] - prediction[endpoints]))),
    }
    for group in sorted(set(groups)):
        use = groups == group
        report["per_group"][group] = {
            "n": int(use.sum()), "exact_accuracy": float(np.mean(truth[use] == prediction[use])),
            "mae": float(np.mean(abs(truth[use] - prediction[use]))),
            "predicted_min": float(np.min(np.asarray([row["optical_equivalent"] for row in matches])[use])),
            "predicted_max": float(np.max(np.asarray([row["optical_equivalent"] for row in matches])[use])),
        }

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    ConfusionMatrixDisplay(cm, display_labels=[f"{v:g}" for v in levels]).plot(
        ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"{task}: hidden-timeline validation")
    for group in sorted(set(groups)):
        subset = [row for row in matches if row["group"] == group]
        x = np.arange(len(subset))
        axes[1].plot(x, [row["optical_equivalent"] for row in subset], label=group, linewidth=1.5)
    axes[1].set_ylabel("Reference-equivalent concentration")
    axes[1].set_xlabel("Ordered frame index within run")
    axes[1].set_title("Unforced optical atlas match")
    axes[1].legend(fontsize=7, frameon=False)
    figure.savefig(output / f"{task.lower()}_cross_run_atlas.png", dpi=300)
    plt.close(figure)
    return report


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path,
                        default=Path(f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/cross_run_atlas_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.cache); assign_h2_ramp_targets(rows); assign_rh_ramp_targets(rows)
    all_matches, reports = [], {}
    atlas_rows = []
    for task, builder in (("H2", build_h2_atlas), ("RH", build_rh_atlas)):
        levels, features, reference_rows = builder(rows)
        for level, feature in zip(levels, features):
            atlas_rows.append({"task": task, "reference_value": level,
                               **{f"feature_{i}": value for i, value in enumerate(feature)}})
        matches = match_groups(rows, task, levels, features)
        all_matches.extend(matches)
        reports[task] = metrics_and_plot(matches, task, args.output)
        reports[task]["n_reference_rows"] = len(reference_rows)
    write_csv(args.output / "reference_atlas.csv", atlas_rows)
    write_csv(args.output / "matches.csv", all_matches)
    (args.output / "metrics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
