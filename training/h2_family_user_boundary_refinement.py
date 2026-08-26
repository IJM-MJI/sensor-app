"""Apply user-reviewed H2 family boundary intervals and re-evaluate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from h2_environment_family_analysis import FAMILIES, prepare
from h2_family_landmark_refinement import run_family
from h2_four_range_analysis import DISPLAY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    x, y, groups, times = prepare(args.cache)

    # Prior review: test_3 does not independently establish 4%; its upper tail
    # belongs to the 2-3 response family.
    y[(groups == "test_3") & (y == 3)] = 2
    # Current point review: the early test_3 optical state is still 1-2, while
    # its late state remains 2-3.
    y[(groups == "test_3") & (times >= 24) & (times <= 28)] = 1
    # Current point review: run4 has entered 2-3 by 65 s. Preserve 55-64.5 s as
    # the lower landmark rather than erasing the class from this run.
    y[(groups == "run4") & (times >= 65) & (times <= 78)] = 2

    selected_a, results_a = run_family(x, y, groups, FAMILIES["A"], (0, 1, 2, 3))
    selected_b, results_b = run_family(x, y, groups, FAMILIES["B"], (0, 1, 2))
    payload = {
        "user_review": {
            "unchanged_2-3": ["test 30-35 s", "test_3 88 s",
                              "run3 55-59 s", "run4 84-90 s"],
            "test_3": "24-28 s -> 1-2",
            "run4": "65-78 s -> 2-3; 55-64.5 s remains 1-2",
        },
        "protocol": "stable family landmarks; complete held-out video unchanged",
        "A": {"selected": selected_a, "models": results_a},
        "B": {"selected": selected_b, "models": results_b},
        "deployment_ready": False,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    selections = (("A", results_a[selected_a], (0, 1, 2, 3)),
                  ("B", results_b[selected_b], (0, 1, 2)))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    for axis, (family, result, labels) in zip(axes, selections):
        matrix = np.asarray(result["confusion"]); axis.imshow(matrix, cmap="Blues")
        for row in range(len(labels)):
            for column in range(len(labels)):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        ticks = [DISPLAY[label] for label in labels]
        axis.set(xticks=range(len(labels)), xticklabels=ticks,
                 yticks=range(len(labels)), yticklabels=ticks,
                 xlabel="Predicted", ylabel="Reference",
                 title=f"Family {family}: {result['exact']:.1%}\n"
                       f"min recall {result['minimum_recall']:.1%}")
    fig.savefig(args.output / "user_boundary_confusions.png", dpi=190)
    plt.close(fig)
    print(json.dumps({
        "A": {"selected": selected_a, **results_a[selected_a]},
        "B": {"selected": selected_b, **results_b[selected_b]},
    }, indent=2))


if __name__ == "__main__":
    main()
