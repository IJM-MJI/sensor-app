"""Evaluate user-confirmed H2 bands: 0-1 / 2-3 / 4%."""

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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DISPLAY = ("0-1", "2-3", "4")


def metrics(y, prediction, groups):
    matrix = confusion_matrix(y, prediction, labels=(0, 1, 2))
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    per_run = {run: float(np.mean(prediction[groups == run] == y[groups == run]))
               for run in sorted(set(groups))}
    per_run_confusion = {
        run: confusion_matrix(
            y[groups == run], prediction[groups == run], labels=(0, 1, 2)
        ).tolist()
        for run in sorted(set(groups))
    }
    return {
        "exact": float(np.mean(prediction == y)),
        "video_macro_exact": float(np.mean(list(per_run.values()))),
        "mae_bands": float(np.mean(np.abs(prediction - y))),
        "recall": {DISPLAY[i]: float(recall[i]) for i in range(3)},
        "minimum_recall": float(np.min(recall)),
        "confusion": matrix.tolist(), "per_run_exact": per_run,
        "per_run_confusion": per_run_confusion,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    saved = np.load(args.cache, allow_pickle=False)
    x, old_y, groups = saved["x"], saved["y"], saved["groups"]
    y = np.asarray([{0: 0, 2: 1, 3: 1, 4: 2}[int(value)] for value in old_y])
    models = {
        "shrinkage_lda": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto")),
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
    for name, model in models.items():
        prediction = np.full(len(y), -1)
        for held_out in sorted(set(groups)):
            train, test = groups != held_out, groups == held_out
            model.fit(x[train], y[train]); prediction[test] = model.predict(x[test])
        results[name] = metrics(y, prediction, groups)
    selected = max(results, key=lambda name: (
        results[name]["minimum_recall"], results[name]["video_macro_exact"],
        results[name]["exact"], -results[name]["mae_bands"]))
    best = results[selected]
    payload = {
        "definition": {"bands": list(DISPLAY),
                       "reason": "user-confirmed small green shift is already H2 2-3%"},
        "validation": "complete video held out", "rows": int(len(y)),
        "selected": selected, "models": results,
        "deployment_target_all_recalls_0.85": bool(best["minimum_recall"] >= .85),
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    matrix = np.asarray(best["confusion"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes[0].set(xticks=range(3), xticklabels=DISPLAY, yticks=range(3),
                yticklabels=DISPLAY, xlabel="Predicted H2 band", ylabel="Reference H2 band",
                title=f"Video-held-out: {selected}")
    recalls = [best["recall"][label] for label in DISPLAY]
    axes[1].bar(DISPLAY, recalls, color=("#66bb6a", "#42a5f5", "#ef5350"))
    axes[1].axhline(.85, color="#555", ls="--", lw=1, label="0.85 target")
    axes[1].set(ylim=(0, 1), ylabel="Recall", title="Per-band recall")
    axes[1].legend()
    fig.savefig(args.output / "three_band_confusion.png", dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
