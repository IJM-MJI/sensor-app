"""Sweep user-constrained run3/run4 H2 transition times."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    x, original_y, groups, times = prepare(args.cache)

    runs = []
    for boundary3 in np.arange(50, 55.1, 1):
        for boundary4 in np.arange(55, 65.1, 1):
            y = original_y.copy()
            # User anchors: run3 55/59 and run4 65/84/90 are 2-3.
            y[(groups == "run3") & (times >= 50) & (times < boundary3)] = 1
            y[(groups == "run4") & (times >= boundary4) & (times <= 78)] = 2
            models = {}
            for feature_name, feature in FEATURES.items():
                for fraction in (.50, .70, .90):
                    for cap in (6, 12, 20):
                        for kind in ("lda", "logistic", "svm"):
                            result = evaluate(x, y, groups, FAMILIES["B"],
                                              (0, 1, 2), feature,
                                              fraction, cap, kind)
                            if result is not None:
                                name = (f"{feature_name}_{kind}_f{fraction:.2f}"
                                        f"_cap{cap}")
                                models[name] = result
            if not models:
                continue
            chosen = select(models); result = models[chosen]
            runs.append({"run3_boundary": float(boundary3),
                         "run4_boundary": float(boundary4),
                         "model": chosen, **result})

    selected = max(runs, key=lambda result: (
        result["minimum_recall"], result["video_macro_exact"], result["exact"]))
    current = next(result for result in runs
                   if result["run3_boundary"] == 50 and result["run4_boundary"] == 65)
    payload = {
        "protocol": "complete-video-held-out boundary sweep",
        "constraints": {"run3_boundary_s": [50, 55],
                        "run4_boundary_s": [55, 65],
                        "anchors": ["run3 55/59 = 2-3", "run4 65/84/90 = 2-3"]},
        "current": current, "selected": selected, "all": runs,
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    grid = np.full((6, 11), np.nan)
    for result in runs:
        grid[int(result["run3_boundary"] - 50),
             int(result["run4_boundary"] - 55)] = result["minimum_recall"]
    fig, axis = plt.subplots(figsize=(9, 4.7), constrained_layout=True)
    image = axis.imshow(grid, cmap="YlGnBu", vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            if np.isfinite(grid[row, column]):
                axis.text(column, row, f"{grid[row, column]:.2f}",
                          ha="center", va="center", fontsize=8)
    axis.set(xticks=range(11), xticklabels=range(55, 66),
             yticks=range(6), yticklabels=range(50, 56),
             xlabel="run4 1-2 → 2-3 boundary (s)",
             ylabel="run3 1-2 → 2-3 boundary (s)",
             title="Family B held-out minimum class recall")
    fig.colorbar(image, ax=axis, label="minimum recall")
    fig.savefig(args.output / "boundary_sweep.png", dpi=190)
    plt.close(fig)
    print(json.dumps({"current": current, "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
