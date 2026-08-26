"""Evaluate user-confirmed H2 environment families without duplicate-run leakage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from h2_four_range_analysis import DISPLAY, ZERO_WINDOWS, in_windows


FAMILIES = {
    "A": ("test_2", "test_3", "test", "run2"),
    "B": ("run3", "run4"),
    "C": ("run5_normal", "run5_x2"),
}


def model(kind: str):
    if kind == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"))
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.2, max_iter=4000, class_weight="balanced", random_state=42))
    return make_pipeline(StandardScaler(), SVC(
        C=1, gamma="scale", class_weight="balanced", random_state=42))


def prepare(path: Path):
    saved = np.load(path, allow_pickle=False)
    x, old_y, groups, times = (saved["x"], saved["y"], saved["groups"],
                               saved["times"])
    keep = np.asarray([int(label) != 0 or in_windows(float(time), ZERO_WINDOWS[str(run)])
                       for label, run, time in zip(old_y, groups, times)])
    y = np.asarray([{0: 0, 2: 1, 3: 2, 4: 3}[int(v)] for v in old_y[keep]])
    return x[keep], y, groups[keep], times[keep]


def report(y, pred, groups, labels):
    matrix = confusion_matrix(y, pred, labels=labels)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    per_run = {run: float(np.mean(pred[groups == run] == y[groups == run]))
               for run in sorted(set(groups))}
    return {
        "exact": float(np.mean(pred == y)),
        "video_macro_exact": float(np.mean(list(per_run.values()))),
        "recall": {DISPLAY[label]: float(recall[i]) for i, label in enumerate(labels)},
        "minimum_recall": float(recall.min()), "confusion": matrix.tolist(),
        "support": {DISPLAY[label]: int(matrix[i].sum()) for i, label in enumerate(labels)},
        "per_run_exact": per_run,
    }


def cross_video(x, y, groups, family, labels, kind, exclude_training=()):
    use = np.isin(groups, family) & np.isin(y, labels)
    fx, fy, fg = x[use], y[use], groups[use]
    pred = np.full(len(fy), -1)
    fold_notes = {}
    for held_out in family:
        test = fg == held_out
        if not test.any():
            continue
        train = (fg != held_out) & ~np.isin(fg, exclude_training)
        available = sorted(set(fy[train]))
        if available != list(labels):
            return None
        estimator = model(kind); estimator.fit(fx[train], fy[train])
        pred[test] = estimator.predict(fx[test])
        fold_notes[held_out] = {"training_runs": sorted(set(fg[train]))}
    result = report(fy, pred, fg, labels)
    result["folds"] = fold_notes
    return result


def choose(results):
    return max(results, key=lambda name: (
        results[name]["minimum_recall"], results[name]["video_macro_exact"],
        results[name]["exact"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    x, y, groups, _ = prepare(args.cache)

    family_a = {}
    for weak_policy, excluded in (("run2_in_training", ()),
                                  ("run2_evaluation_only", ("run2",))):
        for kind in ("lda", "logistic", "svm"):
            result = cross_video(x, y, groups, FAMILIES["A"], tuple(range(4)),
                                 kind, excluded)
            if result is not None:
                family_a[f"{weak_policy}_{kind}"] = result
    selected_a = choose(family_a)

    family_a_shared = {}
    for weak_policy, excluded in (("run2_in_training", ()),
                                  ("run2_evaluation_only", ("run2",))):
        for kind in ("lda", "logistic", "svm"):
            result = cross_video(x, y, groups, FAMILIES["A"], (0, 1, 2),
                                 kind, excluded)
            if result is not None:
                family_a_shared[f"{weak_policy}_{kind}"] = result
    selected_a_shared = choose(family_a_shared)

    # User review indicates test_3 reaches the 2-3 response family rather than
    # an independently verified 4% endpoint. Keep its former upper tail in the
    # middle range and leave 4% support to test/test_2.
    corrected_y = y.copy()
    corrected_y[(groups == "test_3") & (corrected_y == 3)] = 2
    family_a_corrected = {}
    for membership, family in (("with_weak_run2", FAMILIES["A"]),
                               ("without_weak_run2", ("test_2", "test_3", "test"))):
        for kind in ("lda", "logistic", "svm"):
            result = cross_video(x, corrected_y, groups, family,
                                 tuple(range(4)), kind)
            if result is not None:
                family_a_corrected[f"{membership}_{kind}"] = result
    selected_a_corrected = choose(family_a_corrected)

    # B has no independent 4% support in run3. Validate only shared ranges.
    family_b = {}
    for kind in ("lda", "logistic", "svm"):
        result = cross_video(x, y, groups, FAMILIES["B"], (0, 1, 2), kind)
        if result is not None:
            family_b[kind] = result
    selected_b = choose(family_b)

    # C is the same physical run at two playback speeds. This is a consistency
    # diagnostic, never an independent held-out accuracy estimate.
    family_c = {}
    for kind in ("lda", "logistic", "svm"):
        result = cross_video(x, y, groups, FAMILIES["C"], (0, 1, 2), kind)
        if result is not None:
            family_c[kind] = result
    selected_c = choose(family_c)

    payload = {
        "families": {"A": list(FAMILIES["A"]), "B": list(FAMILIES["B"]),
                     "C": list(FAMILIES["C"])},
        "A_independent_four_range": {"selected": selected_a, "models": family_a},
        "A_independent_shared_ranges": {
            "selected": selected_a_shared, "models": family_a_shared,
            "four_percent": "excluded to isolate the disputed test_3 upper reference"},
        "A_corrected_test3_upper": {
            "selected": selected_a_corrected, "models": family_a_corrected,
            "policy": "test_3 former upper tail maps to 2-3; 4 supported by test/test_2"},
        "B_independent_shared_ranges": {
            "selected": selected_b, "models": family_b,
            "four_percent": "not independently validated; run3 has no 4% reference"},
        "C_same_run_speed_consistency_only": {
            "selected": selected_c, "models": family_c,
            "warning": "normal and x2 are the same physical run; metrics are not accuracy"},
        "deployment_ready": False,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    selections = (("A corrected", family_a_corrected[selected_a_corrected], (0, 1, 2, 3)),
                  ("B", family_b[selected_b], (0, 1, 2)),
                  ("C consistency", family_c[selected_c], (0, 1, 2)))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for axis, (title, result, labels) in zip(axes, selections):
        matrix = np.asarray(result["confusion"]); axis.imshow(matrix, cmap="Blues")
        for row in range(len(labels)):
            for column in range(len(labels)):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        ticks = [DISPLAY[label] for label in labels]
        axis.set(xticks=range(len(labels)), xticklabels=ticks,
                 yticks=range(len(labels)), yticklabels=ticks,
                 xlabel="Predicted", ylabel="Reference",
                 title=f"Family {title}\nmin recall={result['minimum_recall']:.2f}")
    fig.savefig(args.output / "family_confusions.png", dpi=190)
    plt.close(fig)
    print(json.dumps({
        "A": {"selected": selected_a, **family_a[selected_a]},
        "A_shared": {"selected": selected_a_shared,
                     **family_a_shared[selected_a_shared]},
        "A_corrected": {"selected": selected_a_corrected,
                        **family_a_corrected[selected_a_corrected]},
        "B": {"selected": selected_b, **family_b[selected_b]},
        "C_consistency": {"selected": selected_c, **family_c[selected_c]},
    }, indent=2))


if __name__ == "__main__":
    main()
