"""Evaluate a conservative calibration-relative 0% gate for H2 environment A.

The gate is only allowed to change an existing 1-2% prediction to 0%.  It
therefore cannot disturb the already strong 2-3% and 4% classes.  Thresholds
are selected using training videos only and each complete video is held out.
RH20_2_x2 is intentionally retained as an out-of-domain audit, not training.
"""

from __future__ import annotations

import argparse
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

from h2_app_family_concentration_analysis import (
    DISPLAY, FEATURES, current_model, current_prediction, estimator, load_rows,
    score, stable_blocks,
)


PRIMARY_RUNS = ("test_2", "test_3", "test")
OOD_RUN = "run2"
LABELS = (0, 1, 2, 3)


def balanced_run_class_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(y), dtype=float)
    for label in (0, 1):
        runs = sorted(set(groups[y == label]))
        for run in runs:
            block = (groups == run) & (y == label)
            weights[block] = 1.0 / (len(runs) * max(int(block.sum()), 1))
    weights *= len(weights) / max(weights.sum(), 1e-9)
    return weights


def choose_threshold(y, base, p_zero, max_one_loss=.02):
    """Choose the most accurate train-only threshold under a 1-2 recall guard."""
    base_one = np.mean(base[y == 1] == 1)
    candidates = np.unique(np.r_[0, p_zero, 1])
    choices = []
    for threshold in candidates:
        prediction = base.copy()
        prediction[(base == 1) & (p_zero >= threshold)] = 0
        one_recall = np.mean(prediction[y == 1] == 1)
        zero_recall = np.mean(prediction[y == 0] == 0)
        exact = np.mean(prediction == y)
        if one_recall + 1e-12 >= base_one - max_one_loss:
            choices.append((exact, zero_recall, one_recall, threshold))
    # Prefer accuracy, then zero recall, then the more conservative threshold.
    return max(choices, key=lambda item: (item[0], item[1], item[2], item[3]))[-1]


def evaluate_candidate(x, y, groups, base, feature, c):
    prediction = base.copy()
    folds = {}
    for held_out in PRIMARY_RUNS:
        test = groups == held_out
        train = np.isin(groups, PRIMARY_RUNS) & ~test & np.isin(y, (0, 1))
        binary_y = (y[train] == 0).astype(int)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=4000, random_state=42),
        )
        weights = balanced_run_class_weights(y[train], groups[train])
        model.fit(x[train][:, feature], binary_y,
                  logisticregression__sample_weight=weights)
        p_train = model.predict_proba(x[train][:, feature])[:, 1]
        threshold = choose_threshold(y[train], base[train], p_train)
        p_test = model.predict_proba(x[test][:, feature])[:, 1]
        prediction[test & (base == 1)] = np.where(
            p_test[base[test] == 1] >= threshold, 0, 1)
        folds[held_out] = {"threshold": float(threshold),
                           "support": int(test.sum())}
    use = np.isin(groups, PRIMARY_RUNS)
    return score(y[use], prediction[use], groups[use], LABELS), prediction, folds


def evaluate_multiclass_consensus(x, y, groups, base):
    """Use the reviewed multiclass specialist only as a zero-vote."""
    use = np.isin(groups, PRIMARY_RUNS)
    feature = FEATURES["green_ab"]
    specialist = np.full(len(y), -1)
    folds = {}
    for held_out in PRIMARY_RUNS:
        test = use & (groups == held_out)
        train = use & (groups != held_out)
        selected = stable_blocks(x[:, feature], y, groups, train, LABELS, .70, 20)
        model = estimator("lda")
        model.fit(x[selected][:, feature], y[selected])
        specialist[test] = model.predict(x[test][:, feature])
        folds[held_out] = {"support": int(test.sum()),
                           "retained_train": int(selected.sum())}
    prediction = base.copy()
    prediction[use & (base == 1) & (specialist == 0)] = 0
    return score(y[use], prediction[use], groups[use], LABELS), prediction, folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--current-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    x, y, groups, times = load_rows(args.cache)
    base = current_prediction(current_model(args.current_model), x)
    primary = np.isin(groups, PRIMARY_RUNS)
    baseline = score(y[primary], base[primary], groups[primary], LABELS)

    candidates = {}
    predictions = {}
    for feature_name, feature in FEATURES.items():
        for c in (.02, .05, .1, .2, .5, 1.0):
            metrics, prediction, folds = evaluate_candidate(
                x, y, groups, base, feature, c)
            name = f"{feature_name}_logistic_C{c:g}"
            candidates[name] = {"metrics": metrics, "folds": folds,
                                "feature_indices": list(feature), "C": c}
            predictions[name] = prediction

    consensus_name = "green_ab_lda_multiclass_consensus"
    metrics, prediction, folds = evaluate_multiclass_consensus(x, y, groups, base)
    candidates[consensus_name] = {
        "metrics": metrics, "folds": folds,
        "feature_indices": list(FEATURES["green_ab"]),
        "policy": "override legacy 1-2 only when specialist predicts 0",
    }
    predictions[consensus_name] = prediction

    # Deployment screening is deliberately lexicographic and conservative:
    # first preserve 1-2%, then improve 0%, then improve total accuracy.
    base_one = baseline["recall"]["1–2"]
    eligible = [name for name, item in candidates.items()
                if item["metrics"]["recall"]["1–2"] >= base_one - .02
                and item["metrics"]["recall"]["0"] > baseline["recall"]["0"]]
    selected = max(eligible, key=lambda name: (
        candidates[name]["metrics"]["recall"]["0"],
        candidates[name]["metrics"]["exact"],
        candidates[name]["metrics"]["minimum_recall"],
    )) if eligible else None

    ood = groups == OOD_RUN
    ood_audit = None
    if selected and ood.any():
        # No claim of independent generalisation: this only records how the
        # primary-run gate behaves on the deliberately excluded run.
        ood_audit = score(y[ood], predictions[selected][ood], groups[ood], LABELS)

    payload = {
        "protocol": "complete-video held-out; train-only thresholds; gate may only map 1-2 to 0",
        "primary_runs": list(PRIMARY_RUNS),
        "excluded_ood_run": OOD_RUN,
        "baseline": baseline,
        "selected": selected,
        "selected_result": candidates.get(selected),
        "ood_audit": ood_audit,
        "candidates": candidates,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    chosen = candidates[selected]["metrics"] if selected else baseline
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, title, metrics in zip(axes, ("Baseline", "Conservative zero gate"),
                                  (baseline, chosen)):
        matrix = np.asarray(metrics["confusion"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        ax.set_xticks(range(4), [DISPLAY[i] for i in LABELS])
        ax.set_yticks(range(4), [DISPLAY[i] for i in LABELS])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Reference")
        ax.set_title(f"{title}\nexact={metrics['exact']:.3f}")
    fig.tight_layout()
    fig.savefig(args.output / "zero_gate_confusion.png", dpi=180)
    print(json.dumps({"baseline": baseline, "selected": selected,
                      "selected_result": candidates.get(selected),
                      "ood_audit": ood_audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
