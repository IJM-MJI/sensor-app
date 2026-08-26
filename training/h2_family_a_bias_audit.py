"""Audit an asymmetric A-family 1-2/2-3 decision bias."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from h2_environment_family_analysis import FAMILIES, prepare
from h2_family_landmark_refinement import estimator, metrics, stable_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    x, y, groups, times = prepare(args.cache)
    y[(groups == "test_3") & (y == 3)] = 2
    y[(groups == "test_3") & (times >= 24) & (times <= 28)] = 1
    y[(groups == "run4") & (times >= 65) & (times <= 78)] = 2

    family = FAMILIES["A"]
    use = np.isin(groups, family)
    fx, fy, fg = x[use][:, (1, 2, 4, 5)], y[use], groups[use]
    scores = np.zeros((len(fy), 4))
    for held_out in family:
        test, train = fg == held_out, fg != held_out
        selected = stable_blocks(fx, fy, fg, train, (0, 1, 2, 3), .90, 6)
        model = estimator("lda"); model.fit(fx[selected], fy[selected])
        scores[test] = model.decision_function(fx[test])

    rows = []
    for bias in np.linspace(-2, 2, 1601):
        adjusted = scores.copy(); adjusted[:, 2] += bias
        result = metrics(fy, adjusted.argmax(axis=1), fg, (0, 1, 2, 3))
        rows.append((float(bias), result))
    eligible = [(bias, result) for bias, result in rows
                if result["recall"]["0"] >= .85
                and result["recall"]["1-2"] >= .85
                and result["recall"]["4"] >= .85]
    best_bias, best = max(eligible, key=lambda item: (
        item[1]["recall"]["2-3"], item[1]["exact"], -abs(item[0])))
    baseline = min(rows, key=lambda item: abs(item[0]))[1]
    payload = {
        "policy": "maximize 2-3 recall while 0, 1-2 and 4 recalls stay >=0.85",
        "baseline": baseline, "selected_bias": best_bias, "selected": best,
        "all_ranges_recall_0.85": bool(min(best["recall"].values()) >= .85),
        "deployment_ready": False,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    axis.plot([bias for bias, _ in rows],
              [result["recall"]["1-2"] for _, result in rows], label="1-2 recall")
    axis.plot([bias for bias, _ in rows],
              [result["recall"]["2-3"] for _, result in rows], label="2-3 recall")
    axis.axhline(.85, color="#555", ls="--", lw=1, label="0.85 target")
    axis.axvline(best_bias, color="#7e57c2", ls=":", label=f"selected {best_bias:.3f}")
    axis.set(xlabel="Added class-2 decision bias", ylabel="Recall", ylim=(0, 1),
             title="Family A adjacent-range threshold trade-off")
    axis.legend()
    fig.savefig(args.output / "bias_tradeoff.png", dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
