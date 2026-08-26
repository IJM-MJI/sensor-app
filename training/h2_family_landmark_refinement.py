"""Refine adjacent H2 range landmarks inside user-confirmed environments."""

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

from h2_environment_family_analysis import FAMILIES, prepare
from h2_four_range_analysis import DISPLAY


FEATURES = {
    "green_a": (1, 4),
    "green_ab": (1, 2, 4, 5),
    "lab6": tuple(range(6)),
    "all11": tuple(range(11)),
}


def estimator(kind):
    if kind == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"))
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.2, max_iter=4000, class_weight="balanced", random_state=42))
    return make_pipeline(StandardScaler(), SVC(
        C=1, gamma="scale", class_weight="balanced", random_state=42))


def stable_blocks(x, y, groups, train, labels, fraction, cap):
    keep = np.zeros(len(y), dtype=bool)
    for run in sorted(set(groups[train])):
        for label in labels:
            index = np.flatnonzero(train & (groups == run) & (y == label))
            if not len(index):
                continue
            values = x[index]
            center = np.median(values, axis=0)
            scale = np.maximum(np.median(np.abs(values - center), axis=0), .12)
            distance = np.sqrt(np.mean(((values - center) / scale) ** 2, axis=1))
            index = index[distance <= np.quantile(distance, fraction)]
            if len(index) > cap:
                positions = np.linspace(0, len(index) - 1, cap).round().astype(int)
                index = index[np.unique(positions)]
            keep[index] = True
    return keep


def balanced_weights(y, groups, selected, labels):
    weights = np.zeros(len(y), dtype=float)
    for label in labels:
        runs = sorted(set(groups[selected & (y == label)]))
        for run in runs:
            block = selected & (groups == run) & (y == label)
            weights[block] = 1 / (max(len(runs), 1) * max(int(block.sum()), 1))
    weights[selected] *= selected.sum() / weights[selected].sum()
    return weights


def metrics(y, pred, groups, labels):
    matrix = confusion_matrix(y, pred, labels=labels)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    per_run = {run: float(np.mean(pred[groups == run] == y[groups == run]))
               for run in sorted(set(groups))}
    return {
        "exact": float(np.mean(pred == y)),
        "video_macro_exact": float(np.mean(list(per_run.values()))),
        "recall": {DISPLAY[label]: float(recall[i]) for i, label in enumerate(labels)},
        "minimum_recall": float(recall.min()), "confusion": matrix.tolist(),
        "per_run_exact": per_run,
        "per_run_confusion": {
            run: confusion_matrix(y[groups == run], pred[groups == run],
                                  labels=labels).tolist()
            for run in sorted(set(groups))
        },
    }


def evaluate(x, y, groups, family, labels, feature, fraction, cap, kind):
    use = np.isin(groups, family) & np.isin(y, labels)
    fx, fy, fg = x[use][:, feature], y[use], groups[use]
    pred = np.full(len(fy), -1)
    retained = {}
    for held_out in family:
        test = fg == held_out
        if not test.any():
            continue
        train = fg != held_out
        selected = stable_blocks(fx, fy, fg, train, labels, fraction, cap)
        if set(fy[selected]) != set(labels):
            return None
        model = estimator(kind)
        weights = balanced_weights(fy, fg, selected, labels)
        if kind in ("logistic", "svm"):
            model.fit(fx[selected], fy[selected], **{
                f"{model.steps[-1][0]}__sample_weight": weights[selected]})
        else:
            model.fit(fx[selected], fy[selected])
        pred[test] = model.predict(fx[test])
        retained[held_out] = {
            run: {DISPLAY[label]: int(np.sum(selected & (fg == run) & (fy == label)))
                  for label in labels}
            for run in sorted(set(fg[selected]))
        }
    result = metrics(fy, pred, fg, labels)
    result["retained_by_fold"] = retained
    return result


def select(results):
    return max(results, key=lambda name: (
        results[name]["minimum_recall"], results[name]["video_macro_exact"],
        results[name]["exact"]))


def run_family(x, y, groups, family, labels):
    results = {}
    for feature_name, feature in FEATURES.items():
        for fraction in (.50, .70, .90):
            for cap in (6, 12, 20):
                for kind in ("lda", "logistic", "svm"):
                    result = evaluate(x, y, groups, family, labels, feature,
                                      fraction, cap, kind)
                    if result is not None:
                        name = f"{feature_name}_{kind}_f{fraction:.2f}_cap{cap}"
                        results[name] = result
    chosen = select(results)
    return chosen, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    x, y, groups, _ = prepare(args.cache)
    y[(groups == "test_3") & (y == 3)] = 2

    selected_a, results_a = run_family(
        x, y, groups, FAMILIES["A"], (0, 1, 2, 3))
    selected_b, results_b = run_family(
        x, y, groups, FAMILIES["B"], (0, 1, 2))
    payload = {
        "protocol": "family-specific stable landmarks; complete video held out intact",
        "A": {"selected": selected_a, "models": results_a},
        "B": {"selected": selected_b, "models": results_b},
        "deployment_ready": False,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    for axis, title, result, labels in (
        (axes[0], "A", results_a[selected_a], (0, 1, 2, 3)),
        (axes[1], "B", results_b[selected_b], (0, 1, 2)),
    ):
        matrix = np.asarray(result["confusion"]); axis.imshow(matrix, cmap="Blues")
        for row in range(len(labels)):
            for column in range(len(labels)):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        ticks = [DISPLAY[label] for label in labels]
        axis.set(xticks=range(len(labels)), xticklabels=ticks,
                 yticks=range(len(labels)), yticklabels=ticks,
                 xlabel="Predicted", ylabel="Reference",
                 title=f"Family {title}: {result['exact']:.1%}\nmin recall {result['minimum_recall']:.1%}")
    fig.savefig(args.output / "family_landmark_confusions.png", dpi=190)
    plt.close(fig)
    print(json.dumps({
        "A": {"selected": selected_a, **results_a[selected_a]},
        "B": {"selected": selected_b, **results_b[selected_b]},
    }, indent=2))


if __name__ == "__main__":
    main()
