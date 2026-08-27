"""Test whether one physical family-C run improves family-B H2 transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from h2_family_b_calibration_normalization import calibration_contrast
from h2_family_b_confirmed_alignment import aligned_data
from h2_family_landmark_refinement import evaluate, select


FAMILY = ("run3", "run4", "run5_normal")


def sweep(values, y, groups, variants, family=FAMILY):
    results = {}
    for variant, feature in variants.items():
        for fraction in (.50, .70, .90):
            for cap in (6, 12, 20):
                for kind in ("lda", "logistic", "svm"):
                    result = evaluate(values[variant], y, groups, family,
                                      (0, 1, 2), feature, fraction, cap, kind)
                    if result is not None:
                        name = f"{variant}_{kind}_f{fraction:.2f}_cap{cap}"
                        results[name] = result
    chosen = select(results)
    return chosen, results[chosen]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    x, y, groups, _ = aligned_data(args.cache, args.video_root)
    use = np.isin(groups, FAMILY) & np.isin(y, (0, 1, 2))
    x, y, groups = x[use], y[use], groups[use]
    baselines = {run: calibration_contrast(args.video_root, run) for run in FAMILY}

    raw_values = {"raw6": x[:, :6]}
    raw_name, raw = sweep(raw_values, y, groups, {"raw6": tuple(range(6))})

    normalized_values, normalized_variants = {}, {}
    for floor in (1.0, 2.0, 4.0, 8.0):
        denominator = np.asarray([
            np.maximum(np.abs(baselines[str(run)][:6]), floor) for run in groups])
        ratio = x[:, :6] / denominator
        key = f"ratio6_floor{floor:g}"
        normalized_values[key] = ratio
        normalized_variants[key] = tuple(range(6))
    norm_name, normalized = sweep(normalized_values, y, groups, normalized_variants)

    # Same-physical-run check only: this is not counted as independent validation.
    c_family = ("run5_normal", "run5_x2")
    cx, cy, cg, _ = aligned_data(args.cache, args.video_root)
    c_use = np.isin(cg, c_family) & np.isin(cy, (0, 1, 2))
    cx, cy, cg = cx[c_use], cy[c_use], cg[c_use]
    c_raw_name, c_raw = sweep({"raw6": cx[:, :6]}, cy, cg,
                              {"raw6": tuple(range(6))}, c_family)

    payload = {
        "protocol": "three complete physical runs held out; run5_x2 excluded as duplicate",
        "family": list(FAMILY),
        "baseline_contrast": {run: value[:6].tolist() for run, value in baselines.items()},
        "raw": {"selected": raw_name, **raw},
        "normalized": {"selected": norm_name, **normalized},
        "family_c_duplicate_consistency": {
            "note": "same physical run at different playback speed; not independent validation",
            "selected": c_raw_name, **c_raw,
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.1), constrained_layout=True)
    for axis, title, result in ((axes[0], "Raw colour change", raw),
                                (axes[1], "Calibration-normalized", normalized)):
        matrix = np.asarray(result["confusion"]); axis.imshow(matrix, cmap="Blues")
        for row in range(3):
            for column in range(3):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set(xticks=range(3), xticklabels=("0", "1-2", "2-3"),
                 yticks=range(3), yticklabels=("0", "1-2", "2-3"),
                 xlabel="Predicted", ylabel="Reference",
                 title=f"{title}\naccuracy {result['exact']:.1%}")
    fig.savefig(args.output / "family_bc_transfer.png", dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
