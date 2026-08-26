"""Evaluate H2 operational ranges: 0 / 1-2 / 2-3 / 4%."""

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
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DISPLAY = ("0", "1-2", "2-3", "4")

# The former 0-1 reference windows included early ramp frames.  Retain only
# narrow initial or full-recovery endpoints for the exact-zero class.
ZERO_WINDOWS = {
    "test_2": ((0, 4),),
    "test_3": ((0, 3),),
    "test": ((0, 3),),
    "run2": ((10, 14), (83, 85)),
    "run3": ((0, 4), (95, 97)),
    "run4": ((0, 8), (178, 180)),
    "run5_x2": ((0, 4), (105, 107)),
    "run5_normal": ((0, 6), (212, 214)),
}


def in_windows(value: float, windows: tuple[tuple[float, float], ...]) -> bool:
    return any(start <= value <= end for start, end in windows)


def score(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> dict:
    matrix = confusion_matrix(y, pred, labels=range(4))
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    per_run = {run: float(np.mean(pred[groups == run] == y[groups == run]))
               for run in sorted(set(groups))}
    return {
        "exact": float(np.mean(pred == y)),
        "video_macro_exact": float(np.mean(list(per_run.values()))),
        "mae_ranges": float(np.mean(np.abs(pred - y))),
        "recall": {DISPLAY[i]: float(recall[i]) for i in range(4)},
        "minimum_recall": float(recall.min()),
        "confusion": matrix.tolist(),
        "support": {DISPLAY[i]: int(matrix[i].sum()) for i in range(4)},
        "per_run_exact": per_run,
        "per_run_confusion": {
            run: confusion_matrix(y[groups == run], pred[groups == run],
                                  labels=range(4)).tolist()
            for run in sorted(set(groups))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    saved = np.load(args.cache, allow_pickle=False)
    x, old_y, groups, times = (saved["x"], saved["y"], saved["groups"],
                               saved["times"])
    keep = np.asarray([
        int(label) != 0 or in_windows(float(time), ZERO_WINDOWS[str(run)])
        for label, run, time in zip(old_y, groups, times)
    ])
    x, old_y, groups, times = x[keep], old_y[keep], groups[keep], times[keep]
    # Former optical 2 and 3 anchors become overlapping-report ranges.  The
    # classes themselves remain mutually exclusive during training.
    y = np.asarray([{0: 0, 2: 1, 3: 2, 4: 3}[int(value)] for value in old_y])

    models = {
        "shrinkage_lda": make_pipeline(StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(
            C=.2, max_iter=4000, class_weight="balanced", random_state=42)),
        "rbf_svm": make_pipeline(StandardScaler(), SVC(
            C=1.0, gamma="scale", class_weight="balanced", random_state=42)),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=4,
            class_weight="balanced", random_state=42, n_jobs=1),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=9, min_samples_leaf=4,
            class_weight="balanced_subsample", random_state=42, n_jobs=1),
    }
    results = {}
    predictions = {}
    for name, model in models.items():
        pred = np.full(len(y), -1)
        for held_out in sorted(set(groups)):
            train, test = groups != held_out, groups == held_out
            model.fit(x[train], y[train])
            pred[test] = model.predict(x[test])
        results[name] = score(y, pred, groups)
        predictions[name] = pred

    # Exact zero has a distinct physical meaning and much cleaner references.
    # Test whether separating state detection from response-range estimation
    # avoids forcing one four-way boundary through both tasks.
    for name in ("shrinkage_lda", "logistic", "rbf_svm"):
        pred = np.full(len(y), -1)
        for held_out in sorted(set(groups)):
            train, test = groups != held_out, groups == held_out
            gate = clone(models[name])
            ranger = clone(models[name])
            gate.fit(x[train], (y[train] > 0).astype(int))
            response_train = train & (y > 0)
            ranger.fit(x[response_train], y[response_train])
            response = gate.predict(x[test]).astype(bool)
            fold = np.zeros(int(test.sum()), dtype=int)
            if response.any():
                fold[response] = ranger.predict(x[test][response])
            pred[test] = fold
        hierarchical_name = f"hierarchical_{name}"
        results[hierarchical_name] = score(y, pred, groups)
        predictions[hierarchical_name] = pred

    selected = max(results, key=lambda name: (
        results[name]["minimum_recall"], results[name]["video_macro_exact"],
        results[name]["exact"], -results[name]["mae_ranges"]))
    best = results[selected]
    payload = {
        "definition": {
            "display_ranges": list(DISPLAY),
            "training_mapping": {"0": "confirmed initial/recovery endpoint",
                                 "1-2": "former optical-2 anchor",
                                 "2-3": "former optical-3 anchor",
                                 "4": "upper endpoint candidate"},
            "overlap_note": "2 is a reporting boundary, never a dual training label",
        },
        "validation": "complete video held out",
        "rows": int(len(y)),
        "selected": selected,
        "models": results,
        "deployment_target_all_recalls_0.85": bool(best["minimum_recall"] >= .85),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    matrix = np.asarray(best["confusion"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].imshow(matrix, cmap="Blues")
    for row in range(4):
        for column in range(4):
            axes[0].text(column, row, str(matrix[row, column]),
                         ha="center", va="center")
    axes[0].set(xticks=range(4), xticklabels=DISPLAY,
                yticks=range(4), yticklabels=DISPLAY,
                xlabel="Predicted H2 range", ylabel="Reference H2 range",
                title=f"Video-held-out: {selected}")
    recalls = [best["recall"][label] for label in DISPLAY]
    axes[1].bar(DISPLAY, recalls,
                color=("#66bb6a", "#42a5f5", "#ffa726", "#ef5350"))
    axes[1].axhline(.85, color="#555", ls="--", lw=1, label="0.85 target")
    axes[1].set(ylim=(0, 1), ylabel="Recall", title="Per-range recall")
    axes[1].legend()
    fig.savefig(args.output / "four_range_confusion.png", dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
