"""Validate H2 quantitation from ramp endpoints and interval-censored labels.

The supplied timeline guarantees the concentration reached at each interval end;
it does not guarantee a linear concentration inside that interval. Endpoint
frames therefore carry exact labels, while interior frames carry only the known
ordered range. Long 4% holds are useful training anchors but are reweighted so
they cannot dominate a recording or the validation score.
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
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ordinal_concentration_analysis import (
    H2_RAMP_ENDPOINTS, H2_RECOVERY_START, assign_h2_ramp_targets, augment,
)
from train_models import CACHE_VERSION, read_csv


LEVELS = np.arange(5, dtype=float)
THRESHOLDS = np.arange(.5, 4, 1.0)
ENDPOINT_WINDOW_SECONDS = 1.0
MAX_BASELINE_EVAL_PER_VIDEO = 4
MAX_HOLD_EVAL_PER_VIDEO = 8


def timeline_label(video: str, time: float) -> tuple[float, float, float | None, str] | None:
    """Return lower/upper concentration, exact target if known, and phase."""
    points = H2_RAMP_ENDPOINTS.get(video)
    if not points:
        return None
    recovery = H2_RECOVERY_START.get(video)
    if recovery is not None and time >= recovery:
        return None
    # The initial calibration window is a strong 0% anchor. Runs with an
    # explicit 0% hold contribute their whole hold through the loop below.
    if time <= min(1.0, points[1][0] if len(points) > 1 else 1.0):
        return 0.0, 0.0, 0.0, "baseline"
    previous_time, previous_value = points[0]
    for end, target in points[1:]:
        if previous_time <= time <= end:
            lo, hi = sorted((float(previous_value), float(target)))
            exact = float(target) if end - ENDPOINT_WINDOW_SECONDS <= time <= end else None
            return (float(target), float(target), exact, "endpoint") if exact is not None else (lo, hi, None, "range")
        previous_time, previous_value = end, target
    if recovery is None and time >= points[-1][0]:
        target = float(points[-1][1])
        return target, target, target, "hold"
    return None


def prepare(rows):
    prepared = []
    for row in rows:
        if row["kind"] != "h2_only":
            continue
        label = timeline_label(str(row["video"]), float(row["time"]))
        if label is None:
            continue
        lower, upper, exact, source = label
        prepared.append({
            "row": row, "lower": lower, "upper": upper, "exact": exact,
            "source": source, "group": str(row["group"]),
            "video": str(row["video"]), "time": float(row["time"]),
        })

    # Validation is endpoint-balanced: long 4% holds and long baseline clips do
    # not get thousands of extra votes merely because they last longer.
    by_video_source = defaultdict(list)
    for item in prepared:
        if item["exact"] is not None:
            by_video_source[(item["video"], item["source"])].append(item)
    for (video, source), items in by_video_source.items():
        items.sort(key=lambda item: item["time"])
        limit = MAX_HOLD_EVAL_PER_VIDEO if source == "hold" else (
            MAX_BASELINE_EVAL_PER_VIDEO if source == "baseline" else len(items))
        chosen = set(np.linspace(0, len(items) - 1, min(limit, len(items))).round().astype(int))
        for index, item in enumerate(items):
            item["evaluate"] = index in chosen
    for item in prepared:
        item.setdefault("evaluate", False)
    return prepared


def fit_threshold_models(items, train_mask):
    x = np.asarray([augment(item["row"], "flame") for item in items])
    lower = np.asarray([item["lower"] for item in items])
    upper = np.asarray([item["upper"] for item in items])
    exact = np.asarray([item["exact"] is not None for item in items])
    models = []
    for threshold in THRESHOLDS:
        known = train_mask & ((upper <= threshold) | (lower > threshold))
        y = lower[known] > threshold
        # Exact anchors define the boundary. Range rows still give useful
        # one-sided ordering constraints, but receive less influence.
        weights = np.where(exact[known], 1.0, .20)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=.35, max_iter=3000, class_weight="balanced", random_state=42))
        model.fit(x[known], y, logisticregression__sample_weight=weights)
        models.append(model)
    return models, x


def predict(models, x):
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
    return LEVELS[np.argmax(probability, axis=1)], np.max(probability, axis=1)


def evaluate(items):
    groups = np.asarray([item["group"] for item in items])
    eval_mask = np.asarray([item["evaluate"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    estimate = np.full(len(items), np.nan)
    confidence = np.full(len(items), np.nan)
    per_video = []
    for group in sorted(set(groups)):
        train = groups != group
        test = (groups == group) & eval_mask
        models, x = fit_threshold_models(items, train)
        estimate[test], confidence[test] = predict(models, x[test])
        per_video.append({
            "group": group, "n": int(test.sum()),
            "exact_accuracy": float(np.mean(estimate[test] == truth[test])),
            "within_one_step": float(np.mean(np.abs(estimate[test] - truth[test]) <= 1)),
            "mae": float(np.mean(np.abs(estimate[test] - truth[test]))),
        })
    use = eval_mask & ~np.isnan(estimate)
    cm = confusion_matrix(truth[use], estimate[use], labels=LEVELS)
    report = {
        "policy": "endpoint exact labels + interval-censored ramp constraints; recovery excluded",
        "exact_accuracy": float(np.mean(estimate[use] == truth[use])),
        "stage_balanced_accuracy": float(np.mean(np.diag(cm) / np.maximum(cm.sum(axis=1), 1))),
        "within_one_step": float(np.mean(np.abs(estimate[use] - truth[use]) <= 1)),
        "mae": float(np.mean(np.abs(estimate[use] - truth[use]))),
        "n_evaluation_frames": int(use.sum()),
        "n_training_rows": len(items),
        "confusion": cm.tolist(), "per_video": per_video,
    }
    predictions = [{
        "video": item["video"], "group": item["group"], "time": item["time"],
        "source": item["source"], "reference": truth[index],
        "prediction": estimate[index], "confidence": confidence[index],
    } for index, item in enumerate(items) if use[index]]
    return report, predictions


def metric_report(truth, estimate, use, policy):
    cm = confusion_matrix(truth[use], estimate[use], labels=LEVELS)
    return {
        "policy": policy,
        "exact_accuracy": float(np.mean(estimate[use] == truth[use])),
        "stage_balanced_accuracy": float(np.mean(np.diag(cm) / np.maximum(cm.sum(axis=1), 1))),
        "within_one_step": float(np.mean(np.abs(estimate[use] - truth[use]) <= 1)),
        "mae": float(np.mean(np.abs(estimate[use] - truth[use]))),
        "n_evaluation_frames": int(use.sum()), "confusion": cm.tolist(),
    }


def existing_reaction_baseline(rows, items):
    """Apply the current reaction model to the same endpoint-only test frames."""
    assign_h2_ramp_targets(rows)
    reaction = [row for row in rows if row["kind"] == "h2_only"
                and row.get("analysis_phase") == "reaction"]
    x = np.asarray([augment(row, "flame") for row in reaction])
    y = np.asarray([float(row["analysis_stage"]) for row in reaction])
    groups = np.asarray([str(row["group"]) for row in reaction])
    item_groups = np.asarray([item["group"] for item in items])
    use = np.asarray([item["evaluate"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    estimate = np.full(len(items), np.nan)
    for group in sorted(set(groups)):
        train = groups != group
        counts = {level: int(np.sum(y[train] == level)) for level in LEVELS}
        weights = np.asarray([len(y[train]) / max(len(LEVELS) * counts[value], 1)
                              for value in y[train]])
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x[train], y[train], ridge__sample_weight=weights)
        test_indices = np.where((item_groups == group) & use)[0]
        test_x = np.asarray([augment(items[index]["row"], "flame") for index in test_indices])
        raw = model.predict(test_x)
        estimate[test_indices] = LEVELS[np.argmin(np.abs(raw[:, None] - LEVELS), axis=1)]
    return metric_report(truth, estimate, use,
                         "current linear-ramp reaction model; endpoint-only scoring")


def nested_range_ridge(items):
    """Tune range weight and ridge strength without seeing the outer test run."""
    x = np.asarray([augment(item["row"], "flame") for item in items])
    groups = np.asarray([item["group"] for item in items])
    source = np.asarray([item["source"] for item in items])
    exact = np.asarray([item["exact"] is not None for item in items])
    target = np.asarray([item["exact"] if item["exact"] is not None
                         else (item["lower"] + item["upper"]) / 2 for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    use = np.asarray([item["evaluate"] for item in items])
    estimate = np.full(len(items), np.nan)
    chosen = []

    def weights(mask, range_weight):
        result = np.where(exact, 1.0, range_weight)
        for group in sorted(set(groups[mask])):
            for name in ("baseline", "hold"):
                select = mask & (groups == group) & (source == name)
                if select.sum():
                    result[select] *= min(1.0, 4.0 / result[select].sum())
        return result

    def fit_predict(train, test, range_weight, alpha):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        sample_weight = weights(train, range_weight)
        model.fit(x[train], target[train], ridge__sample_weight=sample_weight[train])
        raw = model.predict(x[test])
        return LEVELS[np.argmin(np.abs(raw[:, None] - LEVELS), axis=1)]

    def balanced(y, prediction):
        present = sorted(set(y))
        return float(np.mean([np.mean(prediction[y == level] == level) for level in present]))

    for outer in sorted(set(groups)):
        outer_train = groups != outer
        best = None
        for range_weight in (0, .02, .05, .1, .2, .4):
            for alpha in (.03, .1, .3, 1, 3, 10):
                scores = []
                for inner in sorted(set(groups[outer_train])):
                    train = outer_train & (groups != inner)
                    test = (groups == inner) & use
                    prediction = fit_predict(train, test, range_weight, alpha)
                    scores.append((balanced(truth[test], prediction),
                                   np.mean(prediction == truth[test]),
                                   -np.mean(np.abs(prediction - truth[test]))))
                score = tuple(np.mean(scores, axis=0))
                if best is None or score > best[0]:
                    best = (score, range_weight, alpha)
        test = (groups == outer) & use
        estimate[test] = fit_predict(outer_train, test, best[1], best[2])
        chosen.append({"held_out_group": outer, "range_weight": best[1], "alpha": best[2]})
    report = metric_report(
        truth, estimate, use,
        "nested video-held-out range-weighted ridge; hold/baseline capped",
    )
    report["outer_fold_hyperparameters"] = chosen
    return report


def plot(output: Path, report):
    cm = np.asarray(report["confusion"], dtype=float)
    cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.7, 4.2), constrained_layout=True)
    ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    for row in range(5):
        for col in range(5):
            ax.text(col, row, f"{cm[row, col]:.2f}", ha="center", va="center",
                    color="white" if cm[row, col] > .55 else "black")
    ax.set_xticks(range(5), range(5)); ax.set_yticks(range(5), range(5))
    ax.set(xlabel="Predicted H2 (%)", ylabel="Endpoint reference H2 (%)",
           title="H2 endpoint/range video-held-out")
    for suffix, kwargs in (("png", {"dpi": 500}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"h2_endpoint_range_validation.{suffix}", **kwargs)
    plt.close(fig)


def main():
    output = Path("training/output/ordinal_concentration")
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path("training/cache") / CACHE_VERSION / "features.csv")
    items = prepare(rows)
    censored_report, predictions = evaluate(items)
    baseline_report = existing_reaction_baseline(rows, items)
    range_report = nested_range_ridge(items)
    candidates = {
        "current_reaction_baseline": baseline_report,
        "censored_ordinal": censored_report,
        "nested_range_ridge": range_report,
    }
    selected = max(candidates, key=lambda name: (
        candidates[name]["stage_balanced_accuracy"],
        candidates[name]["exact_accuracy"], -candidates[name]["mae"],
    ))
    report = {"selected": selected, "models": candidates}
    (output / "h2_endpoint_range_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    with (output / "h2_endpoint_censored_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    plot(output, candidates[selected])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
