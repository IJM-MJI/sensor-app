"""Compare high-RH and response-presence policies for four-state classification."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from train_models import (
    CACHE_VERSION, H2_FEATURES, RH_FEATURES, binary_model, feature_value, read_csv,
)


STATES = ["initial", "H2_only", "H2O_only", "simultaneous"]
HIGH_RATE_STATE_VIDEOS = {
    "1_90_H2O_only_3(response).mp4", "1_90_H2O_only_6(response).mp4",
}
STATE_FEATURES = ["flame_a", "flame_b", "drop_a", "drop_b", "flame_drop_a", "flame_drop_b"]
STATE_FEATURE_SETS = {
    "ab_regions": STATE_FEATURES,
    "lab_regions": [
        "flame_L", "flame_a", "flame_b", "drop_L", "drop_a", "drop_b",
        "top_L", "top_a", "top_b", "flame_drop_a", "flame_drop_b",
    ],
    "temporal_lab": [
        "flame_L", "flame_a", "flame_b", "drop_L", "drop_a", "drop_b",
        "flame_drop_a", "flame_drop_b",
        "flame_a_med5", "flame_b_med5", "drop_a_med5", "drop_b_med5",
        "flame_drop_a_med5", "flame_drop_b_med5",
        "flame_a_slope5", "flame_b_slope5", "drop_a_slope5", "drop_b_slope5",
    ],
    "recovery_response": [
        "flame_L", "flame_a", "flame_b", "drop_L", "drop_a", "drop_b",
        "top_L", "top_a", "top_b", "flame_drop_a", "flame_drop_b",
        "flame_lab_shift", "drop_lab_shift", "flame_chroma_shift", "drop_chroma_shift",
        "joint_lab_shift", "joint_chroma_shift", "flame_warm_shift", "drop_warm_shift",
    ],
}


def state_candidates():
    return {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "extra_trees_h2o_x2": ExtraTreesClassifier(
            n_estimators=400, max_depth=10, min_samples_leaf=5,
            class_weight={"initial": 1, "H2_only": 1, "H2O_only": 2, "simultaneous": 1},
            random_state=42, n_jobs=-1,
        ),
        "extra_trees_h2o_x3": ExtraTreesClassifier(
            n_estimators=400, max_depth=10, min_samples_leaf=5,
            class_weight={"initial": 1, "H2_only": 1, "H2O_only": 3, "simultaneous": 1},
            random_state=42, n_jobs=-1,
        ),
        "extra_trees_sim_x2": ExtraTreesClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=5,
            class_weight={"initial": 1, "H2_only": 1, "H2O_only": 1, "simultaneous": 2},
            random_state=42, n_jobs=-1,
        ),
        "extra_trees_h2o_sim_x2": ExtraTreesClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=5,
            class_weight={"initial": 1, "H2_only": 1, "H2O_only": 2, "simultaneous": 2},
            random_state=42, n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=.04, max_leaf_nodes=15,
            min_samples_leaf=12, l2_regularization=2, random_state=42,
        ),
    }


def response_labels(rows: list[dict[str, object]]) -> None:
    """Add weak labels from known exposure type; exact transition times are unused."""
    for row in rows:
        kind, state = str(row["kind"]), str(row.get("state"))
        h2_value, rh_value = row.get("h2_value"), row.get("rh_value")
        row["rh_present"] = None
        # The concentration timelines are ramp endpoints.  Earlier state
        # training kept only the stable 3--4% anchors, which left verified
        # H2-only 1--2% endpoint/late-ramp frames without a state target and
        # caused a real 1% app frame to be classified as Initial.  Once the
        # nominal ramp has reached 1%, the exposure type and flame response are
        # sufficient H2-presence supervision.  Existing explicit 0/1 labels
        # (including recovery) always take precedence.
        if (kind == "h2_only" and row.get("h2_present") is None
                and h2_value is not None and float(h2_value) >= 1.0):
            row["h2_present"] = 1
        if row.get("h2_present") is None and kind == "simultaneous":
            phase = str(row.get("phase", ""))
            if phase == "reaction" and h2_value is not None and float(h2_value) >= 3:
                row["h2_present"] = 1
            elif phase in ("recovery", "recovered") and h2_value is not None and float(h2_value) <= .5:
                row["h2_present"] = 0
        if kind == "h2_only" and h2_value is not None:
            row["rh_present"] = 0  # chamber held at RH20
        elif kind == "rh_only" and rh_value is not None:
            # RH30 is deliberately left uncertain; >=40 is a robust exposure anchor.
            row["rh_present"] = 0 if float(rh_value) <= 20 else (1 if float(rh_value) >= 40 else None)
        elif kind == "simultaneous":
            if state == "h2_only_condition":
                row["rh_present"] = 0
            elif state == "simultaneous_condition":
                row["rh_present"] = 1
            elif state == "baseline_recovery":
                row["rh_present"] = 0
            else:
                phase = str(row.get("phase", ""))
                if phase == "reaction" and h2_value is not None and float(h2_value) >= 3:
                    row["rh_present"] = 1
                elif phase in ("recovery", "recovered") and h2_value is not None and float(h2_value) <= .5:
                    row["rh_present"] = 0


def state_sampling(rows: list[dict[str, object]], interval_s: float = 4.0) -> list[dict[str, object]]:
    """Keep state metrics invariant when quantitative extraction becomes denser."""
    by_video: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_video.setdefault(str(row["video"]), []).append(row)
    output = []
    for video, video_rows in by_video.items():
        video_rows.sort(key=lambda row: float(row["time"]))
        if video in HIGH_RATE_STATE_VIDEOS:
            output.extend(video_rows)
            continue
        last = -float("inf")
        for row in video_rows:
            seconds = float(row["time"])
            if seconds - last >= interval_s - 1e-6:
                output.append(row); last = seconds
    return output


def add_temporal_features(rows: list[dict[str, object]], window_s: float = 5.0) -> None:
    """Add causal rolling medians/slopes that can be reproduced by the app."""
    bases = ["flame_a", "flame_b", "drop_a", "drop_b", "flame_drop_a", "flame_drop_b"]
    by_video: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_video.setdefault(str(row["video"]), []).append(row)
    for video_rows in by_video.values():
        video_rows.sort(key=lambda row: float(row["time"]))
        times = np.asarray([float(row["time"]) for row in video_rows])
        values = {name: np.asarray([feature_value(row, name) for row in video_rows]) for name in bases}
        for index, row in enumerate(video_rows):
            start = int(np.searchsorted(times, times[index] - window_s, side="left"))
            use = slice(start, index + 1)
            dt = max(times[index] - times[start], 1e-6)
            for name in bases:
                row[f"{name}_med5"] = float(np.median(values[name][use]))
                row[f"{name}_slope5"] = float((values[name][index] - values[name][start]) / dt) if index > start else 0.0


def add_recovery_response_features(rows: list[dict[str, object]]) -> None:
    """Encode weak but joint flame/drop changes relative to the recovery baseline."""
    for row in rows:
        flame_l, flame_a, flame_b = (float(row[f"flame_{c}"]) for c in "Lab")
        drop_l, drop_a, drop_b = (float(row[f"drop_{c}"]) for c in "Lab")
        flame_lab = float(np.linalg.norm([flame_l, flame_a, flame_b]))
        drop_lab = float(np.linalg.norm([drop_l, drop_a, drop_b]))
        flame_chroma = float(np.linalg.norm([flame_a, flame_b]))
        drop_chroma = float(np.linalg.norm([drop_a, drop_b]))
        row.update({
            "flame_lab_shift": flame_lab,
            "drop_lab_shift": drop_lab,
            "flame_chroma_shift": flame_chroma,
            "drop_chroma_shift": drop_chroma,
            # A simultaneous response requires evidence in both regions. The
            # minimum is deliberately used instead of the product so one very
            # strong droplet response cannot hide an unchanged flame.
            "joint_lab_shift": min(flame_lab, drop_lab),
            "joint_chroma_shift": min(flame_chroma, drop_chroma),
            # Positive b and negative a correspond to the observed shift from
            # cooler/neutral toward the warmer droplet/flame response axis.
            "flame_warm_shift": flame_b - flame_a,
            "drop_warm_shift": drop_b - drop_a,
        })


def matrix(rows, label, features):
    use = [row for row in rows if row.get(label) is not None]
    x = np.asarray([[feature_value(row, feature) for feature in features] for row in use])
    y = np.asarray([int(row[label]) for row in use])
    groups = np.asarray([str(row["group"]) for row in use])
    return use, x, y, groups


def truth_state(row: dict[str, object]) -> str | None:
    kind, state = str(row["kind"]), str(row.get("state"))
    if kind == "h2_only" and row.get("h2_present") is not None:
        return "H2_only" if int(row["h2_present"]) else "initial"
    if kind == "rh_only" and row.get("rh_present") is not None:
        return "H2O_only" if int(row["rh_present"]) else "initial"
    if kind == "simultaneous":
        if state == "h2_only_condition": return "H2_only"
        if state == "simultaneous_condition": return "simultaneous"
        if state == "baseline_recovery": return "initial"
        phase = str(row.get("phase", ""))
        if phase == "reaction" and row.get("h2_value") is not None and float(row["h2_value"]) >= 3:
            return "simultaneous"
        if phase in ("recovery", "recovered") and row.get("h2_value") is not None and float(row["h2_value"]) <= .5:
            return "initial"
    return None


def state_scope(row: dict[str, object]) -> bool:
    """State target is simultaneous RH30-80; RH20 is H2-only and RH90 is OOD."""
    return not (
        str(row.get("kind")) == "simultaneous"
        and str(row.get("state")) == "simultaneous_rh90_saturated"
    )


def predict_excluding_group(train_rows, label, features, eval_rows):
    use, x, y, groups = matrix(train_rows, label, features)
    output = np.full(len(eval_rows), np.nan)
    eval_groups = np.asarray([str(row["group"]) for row in eval_rows])
    for group in sorted(set(eval_groups)):
        test = eval_groups == group
        keep = groups != group
        fitted = clone(binary_model()).fit(x[keep], y[keep])
        ex = np.asarray([[feature_value(row, feature) for feature in features]
                         for row, selected in zip(eval_rows, test) if selected])
        output[test] = fitted.predict_proba(ex)[:, 1]
    return output


def evaluate(rows, rh_policy, eval_source=None):
    eval_rows = [row for row in (eval_source or rows) if truth_state(row) is not None]
    h2_prob = predict_excluding_group(rows, "h2_present", H2_FEATURES, eval_rows)
    rh_label = "rh_high" if rh_policy == "high_rh" else "rh_present"
    rh_train = [row for row in rows if row.get(rh_label) is not None]
    rh_prob = predict_excluding_group(rh_train, rh_label, RH_FEATURES, eval_rows)
    truth = [truth_state(row) for row in eval_rows]
    pred = []
    for hp, rp in zip(h2_prob, rh_prob):
        h2, rh = hp >= .5, rp >= .5
        pred.append("simultaneous" if h2 and rh else "H2_only" if h2 else "H2O_only" if rh else "initial")
    cm = confusion_matrix(truth, pred, labels=STATES)
    by_state = {}
    for state in STATES:
        use = np.asarray([value == state for value in truth])
        by_state[state] = {"n": int(use.sum()), "recall": float(np.mean(np.asarray(pred)[use] == state))}
    by_kind = {}
    for kind in ("h2_only", "rh_only", "simultaneous"):
        use = np.asarray([row["kind"] == kind for row in eval_rows])
        by_kind[kind] = {"n": int(use.sum()), "accuracy": float(np.mean(np.asarray(pred)[use] == np.asarray(truth)[use]))}
    predictions = [{
        "video": row["video"], "group": row["group"], "kind": row["kind"], "time": row["time"],
        "truth": actual, "prediction": estimate, "h2_probability": float(hp), "rh_probability": float(rp),
    } for row, actual, estimate, hp, rp in zip(eval_rows, truth, pred, h2_prob, rh_prob)]
    return {
        "exact_accuracy": float(np.mean(np.asarray(truth) == np.asarray(pred))),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "n_frames": len(truth), "n_groups": len({str(row['group']) for row in eval_rows}),
        "confusion": cm.tolist(), "by_state": by_state, "by_kind": by_kind,
    }, predictions


def evaluate_direct(rows, features=STATE_FEATURES, base=None, eval_source=None):
    """Directly compare the combined flame/drop signatures of the four states."""
    train_rows = [row for row in rows if truth_state(row) is not None]
    eval_rows = [row for row in (eval_source or rows) if truth_state(row) is not None]
    truth = np.asarray([truth_state(row) for row in eval_rows])
    groups = np.asarray([str(row["group"]) for row in eval_rows])
    train_truth = np.asarray([truth_state(row) for row in train_rows])
    train_groups = np.asarray([str(row["group"]) for row in train_rows])
    train_x = np.asarray([[feature_value(row, feature) for feature in features] for row in train_rows])
    eval_x = np.asarray([[feature_value(row, feature) for feature in features] for row in eval_rows])
    pred = np.full(len(eval_rows), "", dtype=object)
    prob = np.full((len(eval_rows), len(STATES)), np.nan)
    if base is None:
        base = state_candidates()["extra_trees"]
    for group in sorted(set(groups)):
        test, train = groups == group, train_groups != group
        fitted = clone(base).fit(train_x[train], train_truth[train])
        pred[test] = fitted.predict(eval_x[test])
        fold_prob = fitted.predict_proba(eval_x[test])
        for index, state in enumerate(fitted.classes_):
            prob[test, STATES.index(str(state))] = fold_prob[:, index]
    cm = confusion_matrix(truth, pred, labels=STATES)
    by_state = {}
    for state in STATES:
        use = truth == state
        by_state[state] = {"n": int(use.sum()), "recall": float(np.mean(pred[use] == state))}
    by_kind = {}
    for kind in ("h2_only", "rh_only", "simultaneous"):
        use = np.asarray([row["kind"] == kind for row in eval_rows])
        by_kind[kind] = {"n": int(use.sum()), "accuracy": float(np.mean(pred[use] == truth[use]))}
    predictions = [{
        "video": row["video"], "group": row["group"], "kind": row["kind"], "time": row["time"],
        "truth": actual, "prediction": estimate,
        **{f"p_{state}": float(probability) for state, probability in zip(STATES, probabilities)},
    } for row, actual, estimate, probabilities in zip(eval_rows, truth, pred, prob)]
    segments: dict[tuple[str, str], list[int]] = {}
    for index, (group, actual) in enumerate(zip(groups, truth)):
        segments.setdefault((str(group), str(actual)), []).append(index)
    segment_truth, segment_pred = [], []
    for (_, actual), indices in segments.items():
        segment_truth.append(actual)
        segment_pred.append(STATES[int(np.nanargmax(np.nanmean(prob[indices], axis=0)))])
    confidence = np.nanmax(prob, axis=1)
    selective = {}
    for threshold in (.40, .45, .50, .55, .60):
        accepted = confidence >= threshold
        selective[f"{threshold:.2f}"] = {
            "coverage": float(np.mean(accepted)), "n": int(accepted.sum()),
            "accuracy": float(np.mean(truth[accepted] == pred[accepted])) if accepted.any() else None,
        }
    return {
        "exact_accuracy": float(np.mean(truth == pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "n_frames": len(truth), "n_groups": len(set(groups)), "confusion": cm.tolist(),
        "by_state": by_state, "by_kind": by_kind,
        "stable_segment_aggregation": {
            "n_segments": len(segment_truth),
            "exact_accuracy": float(np.mean(np.asarray(segment_truth) == np.asarray(segment_pred))),
            "balanced_accuracy": float(balanced_accuracy_score(segment_truth, segment_pred)),
        },
        "selective_prediction": selective,
    }, predictions


def export_multiclass_forest(model, features):
    """Serialize a fitted sklearn forest for the dependency-free browser app."""
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        probabilities = []
        for value in tree.value[:, 0, :]:
            total = float(value.sum())
            probabilities.append((value / total).tolist() if total else [1 / len(model.classes_)] * len(model.classes_))
        trees.append({
            "left": tree.children_left.astype(int).tolist(),
            "right": tree.children_right.astype(int).tolist(),
            "feature": tree.feature.astype(int).tolist(),
            "threshold": tree.threshold.tolist(),
            "probability": probabilities,
        })
    return {
        "type": "extra_trees_multiclass", "features": features,
        "classes": [str(value) for value in model.classes_], "trees": trees,
    }


def calibrate_simultaneous_threshold(predictions):
    """Tune the simultaneous probability cutoff without seeing the held-out run."""
    truth = np.asarray([row["truth"] for row in predictions])
    base = np.asarray([row["prediction"] for row in predictions], dtype=object)
    groups = np.asarray([row["group"] for row in predictions])
    probability = np.asarray([float(row["p_simultaneous"]) for row in predictions])
    estimate = base.copy()
    chosen = {}
    grid = np.arange(.25, .451, .025)
    for group in sorted(set(groups)):
        train = groups != group
        scores = []
        for threshold in grid:
            candidate = base[train].copy()
            candidate[probability[train] >= threshold] = "simultaneous"
            scores.append(balanced_accuracy_score(truth[train], candidate))
        threshold = float(grid[int(np.argmax(scores))])
        chosen[str(group)] = threshold
        test = groups == group
        estimate[test & (probability >= threshold)] = "simultaneous"
    cm = confusion_matrix(truth, estimate, labels=STATES)
    by_state = {}
    for state in STATES:
        use = truth == state
        by_state[state] = {"n": int(use.sum()), "recall": float(np.mean(estimate[use] == state))}
    adjusted = [dict(row, thresholded_prediction=str(value))
                for row, value in zip(predictions, estimate)]
    return {
        "exact_accuracy": float(np.mean(truth == estimate)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, estimate)),
        "confusion": cm.tolist(), "by_state": by_state,
        "selection": "simultaneous cutoff selected on all other experiment groups",
        "threshold_by_held_out_group": chosen,
    }, adjusted


def plot(results, output):
    shown = {
        "Independent high-RH": results["high_rh"],
        "Independent response": results["response_presence"],
        "Best direct four-state": results["best_direct_candidate"],
    }
    fig, axes = plt.subplots(1, len(shown), figsize=(4.1 * len(shown), 3.5), constrained_layout=True)
    for ax, (name, result) in zip(axes, shown.items()):
        cm = np.asarray(result["confusion"], dtype=float)
        normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{normalized[i,j]:.2f}\n(n={int(cm[i,j])})", ha="center", va="center",
                        fontsize=7, color="white" if normalized[i,j] > .55 else "black")
        ax.set_xticks(range(4), STATES, rotation=25, ha="right")
        ax.set_yticks(range(4), STATES)
        ax.set(xlabel="Predicted state", ylabel="Reference state",
               title=f"{name}\nBalanced accuracy={result['balanced_accuracy']:.3f}")
    fig.colorbar(image, ax=axes, label="Row-normalized fraction", shrink=.8)
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(f"training/cache/{CACHE_VERSION}/features.csv"))
    parser.add_argument("--angle-cache", type=Path, default=Path(f"training/cache/{CACHE_VERSION}/angle_runs.csv"))
    parser.add_argument("--weak-rh-cache", type=Path, default=None,
                        help="optional weak state-only rows; disabled once response timelines are known")
    parser.add_argument("--output", type=Path, default=Path("training/output/state_condition"))
    args = parser.parse_args()
    # Only recordings with user-supplied timelines and verified ROIs belong in
    # state validation. Untimed legacy recordings are prediction-review data,
    # never ground truth.
    strong_rows = read_csv(args.cache)
    if args.angle_cache.exists():
        strong_rows += read_csv(args.angle_cache)
    strong_rows = state_sampling(strong_rows)
    response_labels(strong_rows)
    strong_rows = [row for row in strong_rows if state_scope(row)]
    add_temporal_features(strong_rows)
    add_recovery_response_features(strong_rows)
    weak_rows = []
    if args.weak_rh_cache is not None and args.weak_rh_cache.exists():
        weak_rows = read_csv(args.weak_rh_cache)
        for row in weak_rows:
            row["kind"] = "rh_only"
            row["h2_present"] = 0
            row["rh_present"] = 1
            row["state"] = "rh_response_unlabeled"
    rows = strong_rows + weak_rows
    results, predictions = {}, {}
    for policy in ("high_rh", "response_presence"):
        results[policy], predictions[policy] = evaluate(rows, policy, strong_rows)
    results["direct_four_state"], predictions["direct_four_state"] = evaluate_direct(rows, eval_source=strong_rows)
    candidate_metrics = {}
    best = None
    best_predictions = None
    best_features = None
    best_estimator = None
    for feature_name, features in STATE_FEATURE_SETS.items():
        for model_name, estimator in state_candidates().items():
            if feature_name == "ab_regions" and "h2o_x" in model_name:
                continue
            key = f"{feature_name}+{model_name}"
            metric, candidate_predictions = evaluate_direct(rows, features, estimator, strong_rows)
            candidate_metrics[key] = {
                "exact_accuracy": metric["exact_accuracy"],
                "balanced_accuracy": metric["balanced_accuracy"],
                "stable_segment_aggregation": metric["stable_segment_aggregation"],
                "by_state": metric["by_state"],
            }
            score = metric["balanced_accuracy"]
            if best is None or score > best[0]:
                best = (score, key, metric)
                best_predictions = candidate_predictions
                best_features = features
                best_estimator = estimator
    results["direct_model_comparison"] = candidate_metrics
    results["best_direct_candidate"] = {"name": best[1], **best[2]}
    threshold_metric, threshold_predictions = calibrate_simultaneous_threshold(best_predictions)
    results["threshold_calibrated_direct"] = threshold_metric
    results["weak_rh_augmentation"] = {
        "training_frames": len(weak_rows), "training_groups": len({str(r["group"]) for r in weak_rows}),
        "evaluation_policy": "training augmentation only; excluded from validation truth",
    }
    results["evaluation_scope"] = {
        "simultaneous_nominal_rh": "30-80 (RH20 reaction is H2-only)",
        "excluded_condition": "nominal RH90 reaction (saturated; H2-dependent change not reliably visible)",
        "simultaneous_rh_quantitation": "H2O-only-calibrated optical-equivalent RH; nominal setpoint never used as target",
    }
    predictions["best_direct_candidate"] = best_predictions
    predictions["threshold_calibrated_direct"] = threshold_predictions
    args.output.mkdir(parents=True, exist_ok=True)
    final_rows = [row for row in rows if truth_state(row) is not None]
    final_x = np.asarray([[feature_value(row, feature) for feature in best_features] for row in final_rows])
    final_y = np.asarray([truth_state(row) for row in final_rows])
    # 320 trees retains essentially the full model's held-out performance while
    # reducing the browser payload by roughly one third (500-tree BA 0.702;
    # 320-tree BA 0.701, simultaneous recall 0.427).
    browser_estimator = clone(best_estimator).set_params(n_estimators=320)
    browser_metric, _ = evaluate_direct(rows, best_features, browser_estimator, strong_rows)
    fitted_state = browser_estimator.fit(final_x, final_y)
    exported = export_multiclass_forest(fitted_state, best_features)
    exported["validation"] = {
        "method": "leave-one-experiment-group-out",
        "exact_accuracy": browser_metric["exact_accuracy"],
        "balanced_accuracy": browser_metric["balanced_accuracy"],
        "simultaneous_recall": browser_metric["by_state"]["simultaneous"]["recall"],
        "stable_segment_accuracy": browser_metric["stable_segment_aggregation"]["exact_accuracy"],
        "trees": 320,
    }
    (args.output / "state_model.json").write_text(json.dumps(exported), encoding="utf-8")
    Path("sensor-state-model.js").write_text(
        "// Generated by training/state_condition_analysis.py\nwindow.SENSOR_STATE_MODEL="
        + json.dumps(exported, separators=(",", ":")) + ";\n", encoding="utf-8")
    (args.output / "state_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for policy, data in predictions.items():
        with (args.output / f"{policy}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    plot(results, args.output / "state_confusion_comparison")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
