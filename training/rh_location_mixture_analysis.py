"""Held-out place-aware RH 40/50/60 mixture experiment.

For every outer fold, one complete run is hidden.  Its 20--30% calibration
frame selects place 1 or place 2 using only the other runs.  The middle-stage
expert then uses endpoint prototypes from the remaining run(s) in that place.
No held-out middle/high label participates in domain or concentration fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from endpoint_interval_analysis import feature_matrix, prepare
from ordinal_concentration_analysis import TASKS
from run_progress_analysis import evaluate_exact, fold_transform
from train_models import CACHE_VERSION, read_csv


LOCATIONS = {
    "place1": {"rh-indoor-fast", "rh-indoor-long"},
    "place2": {"rh-response-3", "rh-response-6"},
}
GROUP_LOCATION = {group: location for location, groups in LOCATIONS.items()
                  for group in groups}
MIDDLE_LEVELS = np.asarray([40.0, 50.0, 60.0])
FEATURE_SETS = {
    "drop_lab": (0, 1, 2),
    "drop_lab_plus_reference": (0, 1, 2, 6, 7, 8),
    "all_compact_features": tuple(range(9)),
}


def report(truth, prediction):
    matrix = confusion_matrix(truth, prediction, labels=MIDDLE_LEVELS)
    truth_counts = np.asarray([np.sum(truth == level) for level in MIDDLE_LEVELS])
    # The global baseline may predict 20-30/70/80/90 for a true middle endpoint.
    # Those are errors and must remain in each recall denominator even though the
    # compact 3x3 display has no column for them.
    recall = np.asarray([
        np.mean(prediction[truth == level] == level) for level in MIDDLE_LEVELS
    ])
    return {
        "exact_accuracy": float(np.mean(truth == prediction)),
        "balanced_accuracy": float(np.mean(recall)),
        "within_one_stage": float(np.mean(
            np.abs(np.searchsorted(MIDDLE_LEVELS, truth) -
                   np.searchsorted(MIDDLE_LEVELS, prediction)) <= 1)),
        "mae": float(np.mean(np.abs(truth - prediction))),
        "recall_40_50_60": recall.tolist(),
        "confusion": matrix.tolist(),
        "truth_counts": truth_counts.tolist(),
        "n_endpoints": int(len(truth)),
    }


def calibration_location(raw_x, groups, truth, exact, held_out, feature_indices):
    """Nearest place centroid from run-median Initial calibration features."""
    outer_train = groups != held_out
    train_prototypes, train_locations = [], []
    for group in sorted(set(groups[outer_train])):
        use = outer_train & exact & (groups == group) & (truth == 25.0)
        train_prototypes.append(np.median(raw_x[use][:, feature_indices], axis=0))
        train_locations.append(GROUP_LOCATION[group])
    held = exact & (groups == held_out) & (truth == 25.0)
    held_prototype = np.median(raw_x[held][:, feature_indices], axis=0)
    scaler = StandardScaler().fit(np.asarray(train_prototypes))
    prototypes = scaler.transform(np.asarray(train_prototypes))
    held_scaled = scaler.transform(held_prototype[None, :])[0]
    location_centres = {
        location: np.median(prototypes[np.asarray(train_locations) == location], axis=0)
        for location in sorted(set(train_locations))
    }
    distances = {location: float(np.sqrt(np.mean((held_scaled - centre) ** 2)))
                 for location, centre in location_centres.items()}
    ordered = sorted(distances, key=distances.get)
    margin = ((distances[ordered[1]] - distances[ordered[0]]) /
              max(distances[ordered[1]], 1e-9))
    return ordered[0], float(np.clip(margin, 0, 1)), distances


def middle_prototypes(x, groups, truth, exact, train, location, feature_indices):
    references = {}
    eligible_groups = LOCATIONS[location]
    for level in MIDDLE_LEVELS:
        per_group = []
        for group in sorted(eligible_groups):
            use = train & exact & (groups == group) & (truth == level)
            if np.any(use):
                per_group.append(np.median(x[use][:, feature_indices], axis=0))
        references[level] = np.median(np.asarray(per_group), axis=0)
    return references


def global_middle_prototypes(x, groups, truth, exact, train, feature_indices):
    references = {}
    for level in MIDDLE_LEVELS:
        per_group = []
        for group in sorted(set(groups[train])):
            use = train & exact & (groups == group) & (truth == level)
            if np.any(use):
                per_group.append(np.median(x[use][:, feature_indices], axis=0))
        references[level] = np.median(np.asarray(per_group), axis=0)
    return references


def middle_predict(values, references, feature_indices):
    selected = values[:, feature_indices]
    costs = np.column_stack([
        np.sqrt(np.mean((selected - references[level]) ** 2, axis=1))
        for level in MIDDLE_LEVELS
    ])
    order = np.argsort(costs, axis=1)
    best = costs[np.arange(len(values)), order[:, 0]]
    second = costs[np.arange(len(values)), order[:, 1]]
    confidence = np.clip((second - best) / np.maximum(second, 1e-9), 0, 1)
    return MIDDLE_LEVELS[order[:, 0]], confidence


def evaluate_mixture(items, feature_name, automatic_location=True):
    raw_x = feature_matrix(items, "RH")
    groups = np.asarray([item["group"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    exact = ~np.isnan(truth)
    evaluate = np.asarray([item["evaluate"] for item in items])
    feature_indices = np.asarray(FEATURE_SETS[feature_name])
    prediction = np.full(len(items), np.nan)
    confidence = np.full(len(items), np.nan)
    domain_audit = []
    for held_out in sorted(set(groups)):
        domain_votes, feature_audit = [], {}
        for selector_name, selector_indices_raw in FEATURE_SETS.items():
            selected, margin, distances = calibration_location(
                raw_x, groups, truth, exact, held_out,
                np.asarray(selector_indices_raw))
            domain_votes.append(selected)
            feature_audit[selector_name] = {
                "prediction": selected, "margin": margin, "distances": distances,
            }
        selected_location = Counter(domain_votes).most_common(1)[0][0]
        true_location = GROUP_LOCATION[held_out]
        location = selected_location if automatic_location else true_location
        train = groups != held_out
        test = ((groups == held_out) & evaluate &
                np.isin(truth, MIDDLE_LEVELS))
        x, _ = fold_transform(items, "RH", raw_x, train, "one_anchor")
        references = middle_prototypes(
            x, groups, truth, exact, train, location, feature_indices)
        prediction[test], confidence[test] = middle_predict(
            x[test], references, feature_indices)
        domain_audit.append({
            "held_out_group": held_out, "true_location": true_location,
            "selected_location": selected_location,
            "correct": selected_location == true_location,
            "votes": domain_votes, "selectors": feature_audit,
        })
    use = ~np.isnan(prediction)
    metrics = report(truth[use], prediction[use])
    metrics.update({
        "feature_set": feature_name,
        "location_mode": "calibration-selected" if automatic_location else "oracle",
        "domain_accuracy": float(np.mean([row["correct"] for row in domain_audit])),
        "domain_audit": domain_audit,
    })
    rows = [{
        "feature_set": feature_name,
        "location_mode": metrics["location_mode"],
        "group": item["group"], "video": item["video"], "time": item["time"],
        "reference": truth[index], "prediction": prediction[index],
        "confidence": confidence[index],
    } for index, item in enumerate(items) if use[index]]
    return metrics, rows


def evaluate_global_middle(items, feature_name):
    """Same forced-middle prototype without a place split (control A/B)."""
    raw_x = feature_matrix(items, "RH")
    groups = np.asarray([item["group"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    exact = ~np.isnan(truth)
    evaluate = np.asarray([item["evaluate"] for item in items])
    feature_indices = np.asarray(FEATURE_SETS[feature_name])
    prediction = np.full(len(items), np.nan)
    confidence = np.full(len(items), np.nan)
    for held_out in sorted(set(groups)):
        train = groups != held_out
        test = ((groups == held_out) & evaluate &
                np.isin(truth, MIDDLE_LEVELS))
        x, _ = fold_transform(items, "RH", raw_x, train, "one_anchor")
        references = global_middle_prototypes(
            x, groups, truth, exact, train, feature_indices)
        prediction[test], confidence[test] = middle_predict(
            x[test], references, feature_indices)
    use = ~np.isnan(prediction)
    metrics = report(truth[use], prediction[use])
    metrics.update({"feature_set": feature_name,
                    "location_mode": "global forced-middle control"})
    rows = [{
        "feature_set": feature_name, "location_mode": metrics["location_mode"],
        "group": item["group"], "video": item["video"], "time": item["time"],
        "reference": truth[index], "prediction": prediction[index],
        "confidence": confidence[index],
    } for index, item in enumerate(items) if use[index]]
    return metrics, rows


def baseline_middle(items):
    metrics, rows = evaluate_exact(items, "RH", "one_anchor")
    selected = [row for row in rows if float(row["reference"]) in MIDDLE_LEVELS]
    truth = np.asarray([float(row["reference"]) for row in selected])
    prediction = np.asarray([float(row["prediction"]) for row in selected])
    return report(truth, prediction), selected


def plot(output, baseline, global_control, automatic):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    for axis, title, metrics in (
            (axes[0], "Global one-anchor model", baseline),
            (axes[1], "Global forced-middle control", global_control),
            (axes[2], "Calibration-selected location expert", automatic)):
        matrix = np.asarray(metrics["confusion"], dtype=float)
        normalized = matrix / np.maximum(
            np.asarray(metrics["truth_counts"], dtype=float)[:, None], 1)
        axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{normalized[row, column]:.2f}",
                          ha="center", va="center",
                          color="white" if normalized[row, column] > .55 else "black")
        axis.set_xticks(range(3), ["40", "50", "60"])
        axis.set_yticks(range(3), ["40", "50", "60"])
        axis.set(xlabel="Predicted RH", ylabel="Endpoint reference", title=title)
        axis.text(.5, -.21,
                  f"exact={metrics['exact_accuracy']:.3f}  "
                  f"balanced={metrics['balanced_accuracy']:.3f}  "
                  f"MAE={metrics['mae']:.1f}%RH",
                  transform=axis.transAxes, ha="center", fontsize=9)
    fig.suptitle("RH 40/50/60 complete-run held-out location mixture", fontweight="bold")
    fig.savefig(output / "rh_location_mixture_validation.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_location_mixture_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.cache)
    items = [item for item in prepare(rows)
             if item["task"] == "RH" and item["group"] in GROUP_LOCATION]
    baseline, baseline_rows = baseline_middle(items)
    models, global_controls, all_rows = {}, {}, []
    for feature_name in FEATURE_SETS:
        global_control, global_rows = evaluate_global_middle(items, feature_name)
        oracle, oracle_rows = evaluate_mixture(items, feature_name, False)
        automatic, automatic_rows = evaluate_mixture(items, feature_name, True)
        global_controls[feature_name] = global_control
        models[feature_name] = {"oracle_location": oracle,
                                "calibration_selected_location": automatic}
        all_rows.extend(global_rows + oracle_rows + automatic_rows)
    selected_feature = "drop_lab"  # predeclared from the preceding location audit
    selected = models[selected_feature]["calibration_selected_location"]
    result = {
        "scope": "rising H2O-only Reaction middle endpoints only",
        "locations": {key: sorted(value) for key, value in LOCATIONS.items()},
        "held_out_unit": "complete independent run",
        "baseline_global_one_anchor": baseline,
        "global_forced_middle_controls": global_controls,
        "predeclared_feature_set": selected_feature,
        "location_models": models,
        "selected_calibration_location_model": selected,
        "all_40_50_60_recalls_improved_or_preserved": bool(np.all(
            np.asarray(selected["recall_40_50_60"])
            >= np.asarray(baseline["recall_40_50_60"]))),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    combined = ([{"feature_set": "global", "location_mode": "baseline", **row}
                 for row in baseline_rows] + all_rows)
    with (args.output / "predictions.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader(); writer.writerows(combined)
    plot(args.output, baseline, global_controls[selected_feature], selected)
    print(json.dumps({
        "baseline": baseline,
        "selected": selected,
        "all_middle_recalls_preserved": result[
            "all_40_50_60_recalls_improved_or_preserved"],
        "sensitivity": {name: {
            mode: {key: model[key] for key in (
                "exact_accuracy", "balanced_accuracy", "mae",
                "recall_40_50_60", "domain_accuracy")}
            for mode, model in candidates.items()}
            for name, candidates in models.items()},
        "global_middle_controls": {name: {key: model[key] for key in (
            "exact_accuracy", "balanced_accuracy", "mae", "recall_40_50_60")}
            for name, model in global_controls.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
