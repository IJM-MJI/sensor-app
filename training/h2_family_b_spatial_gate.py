"""Audit a conservative ensemble of baseline and spatial H2 family-B models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from h2_environment_family_analysis import FAMILIES, prepare
from h2_family_landmark_refinement import (
    balanced_weights, estimator, metrics, stable_blocks,
)


def corrected_base(path: Path):
    x, y, groups, times = prepare(path)
    use = np.isin(groups, FAMILIES["A"] + FAMILIES["B"])
    x, y, groups, times = x[use], y[use], groups[use], times[use]
    y[(groups == "test_3") & (y == 3)] = 2
    y[(groups == "test_3") & (times >= 24) & (times <= 28)] = 1
    y[(groups == "run4") & (times >= 65) & (times <= 78)] = 2
    return x, y, groups


def held_out_predictions(x, y, groups, feature, fraction, cap):
    family, labels = FAMILIES["B"], (0, 1, 2)
    use = np.isin(groups, family) & np.isin(y, labels)
    fx, fy, fg = x[use][:, feature], y[use], groups[use]
    prediction = np.full(len(fy), -1)
    probability = np.zeros((len(fy), len(labels)))
    for held_out in family:
        test, train = fg == held_out, fg != held_out
        selected = stable_blocks(fx, fy, fg, train, labels, fraction, cap)
        model = estimator("lda")
        model.fit(fx[selected], fy[selected])
        prediction[test] = model.predict(fx[test])
        probability[test] = model.predict_proba(fx[test])
    return fy, fg, prediction, probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--spatial-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    base_x, y, groups = corrected_base(args.base_cache)
    spatial = np.load(args.spatial_cache, allow_pickle=False)
    spatial_x = spatial["x"]
    if not (np.array_equal(y, spatial["y"]) and np.array_equal(groups, spatial["groups"])):
        raise RuntimeError("Base and spatial caches are not aligned")

    truth, fold_groups, base_pred, base_prob = held_out_predictions(
        base_x, y, groups, tuple(range(6)), .70, 6)
    _, _, spatial_pred, spatial_prob = held_out_predictions(
        spatial_x, y, groups, tuple(range(spatial_x.shape[1])), .90, 6)

    baseline = metrics(truth, base_pred, fold_groups, (0, 1, 2))
    spatial_only = metrics(truth, spatial_pred, fold_groups, (0, 1, 2))
    candidates = []
    thresholds = np.r_[np.linspace(.50, .99, 50), 1.01]
    for up in thresholds:
        for down in thresholds:
            pred = base_pred.copy()
            promote = (base_pred == 1) & (spatial_pred == 2) & (spatial_prob[:, 2] >= up)
            demote = (base_pred == 2) & (spatial_pred == 1) & (spatial_prob[:, 1] >= down)
            pred[promote], pred[demote] = 2, 1
            result = metrics(truth, pred, fold_groups, (0, 1, 2))
            result.update(up_threshold=float(up), down_threshold=float(down),
                          promoted=int(promote.sum()), demoted=int(demote.sum()))
            candidates.append(result)

    # Preserve every baseline recall; only then maximize the worst class and accuracy.
    feasible = [result for result in candidates if all(
        result["recall"][label] >= baseline["recall"][label]
        for label in ("0", "1-2", "2-3"))]
    chosen = max(feasible or candidates, key=lambda result: (
        result["minimum_recall"], result["video_macro_exact"], result["exact"]))
    payload = {"protocol": "complete-video-held-out; baseline-preserving spatial gate",
               "baseline": baseline, "spatial_only": spatial_only,
               "feasible_count": len(feasible), "selected": chosen}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, title, result in zip(
            axes, ("Baseline", "Spatial only", "Conservative gate"),
            (baseline, spatial_only, chosen)):
        matrix = np.asarray(result["confusion"])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
        for row in range(3):
            for column in range(3):
                axis.text(column, row, str(matrix[row, column]),
                          ha="center", va="center")
        axis.set(xticks=range(3), xticklabels=("0", "1-2", "2-3"),
                 yticks=range(3), yticklabels=("0", "1-2", "2-3"),
                 xlabel="Predicted", ylabel="Reference",
                 title=f"{title}\naccuracy {result['exact']:.1%}")
    fig.savefig(args.output / "family_b_gate_comparison.png", dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
