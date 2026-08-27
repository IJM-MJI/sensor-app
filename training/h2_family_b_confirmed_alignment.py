"""Validate the user-confirmed family-B H2 optical alignment with added ramp frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from h2_environment_family_analysis import FAMILIES, prepare
from h2_family_landmark_refinement import FEATURES, evaluate, select
from h2_rh20_max_response_analysis import RUNS as RH20_RUNS, extract


def aligned_data(cache: Path, video_root: Path):
    x, y, groups, times = prepare(cache)
    # Confirmed run3 points: 55 s and 59 s are 2-3; late 50-54.5 remains 1-2.
    y[(groups == "run3") & (times >= 50) & (times < 55)] = 1
    # Confirmed run4 points: 50, 55, and 56 s are 2-3.
    y[(groups == "run4") & (times >= 50) & (times <= 78)] = 2

    # The original endpoint cache starts run4's middle band at 55 s. Add the
    # missing pre-boundary interval so advancing the boundary does not remove
    # the held-out 1-2 class or inflate accuracy through support imbalance.
    rows, *_ = extract(video_root, "run4", RH20_RUNS["run4"], 2.0)
    extra = [row for row in rows if 30 <= row["time"] < 55]
    extra_x = np.asarray([row["delta"][:11] for row in extra])
    extra_t = np.asarray([row["time"] for row in extra])
    extra_y = np.where(extra_t < 50, 1, 2)
    return (np.vstack([x, extra_x]), np.r_[y, extra_y],
            np.r_[groups, np.repeat("run4", len(extra))], np.r_[times, extra_t])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    x, y, groups, times = aligned_data(args.cache, args.video_root)

    results = {}
    for feature_name, feature in FEATURES.items():
        for fraction in (.50, .70, .90):
            for cap in (6, 12, 20):
                for kind in ("lda", "logistic", "svm"):
                    result = evaluate(x, y, groups, FAMILIES["B"], (0, 1, 2),
                                      feature, fraction, cap, kind)
                    if result is not None:
                        name = f"{feature_name}_{kind}_f{fraction:.2f}_cap{cap}"
                        results[name] = result
    chosen = select(results); best = results[chosen]
    support = {run: {str(label): int(np.sum((groups == run) & (y == label)))
                     for label in (0, 1, 2)} for run in FAMILIES["B"]}
    payload = {
        "protocol": "complete-video-held-out; user-confirmed optical boundaries",
        "labels": {"run3": "1-2 through 54.5 s; 2-3 from 55 s",
                   "run4": "1-2 at 30-49.5 s; 2-3 from 50 s"},
        "support": support, "selected": chosen, "result": best,
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    matrix = np.asarray(best["confusion"])
    fig, axis = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    axis.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(xticks=range(3), xticklabels=("0", "1-2", "2-3"),
             yticks=range(3), yticklabels=("0", "1-2", "2-3"),
             xlabel="Predicted", ylabel="Reference",
             title=f"Confirmed family B alignment\naccuracy {best['exact']:.1%}")
    fig.savefig(args.output / "confirmed_alignment_confusion.png", dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
