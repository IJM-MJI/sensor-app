"""Audit deployable calibration-relative response normalization for H2 family B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from h2_environment_family_analysis import FAMILIES
from h2_family_b_confirmed_alignment import aligned_data
from h2_family_landmark_refinement import evaluate, select
from h2_more_crop_fixed_mask_analysis import frame_at, fixed_shape_mask, substrate, summary
from h2_other_run_reference_matching import resize_roi
from h2_rh20_max_response_analysis import RUNS


def calibration_contrast(video_root: Path, run: str) -> np.ndarray:
    config = RUNS[run]
    cap = cv2.VideoCapture(str(video_root / config["file"]))
    calibration = resize_roi(frame_at(cap, config["cal"]), config["roi"])
    cap.release()
    lab = cv2.cvtColor(calibration, cv2.COLOR_BGR2LAB).astype(float)
    height, width = lab.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    nx, ny = xx / width, yy / height
    fx, fy, rx, ry = config["flame"]
    flame_zone = ((nx - fx) / rx) ** 2 + ((ny - fy) / ry) ** 2 <= 1
    if run == "run3":
        flame_zone &= ny >= .25
    card = (nx >= .02) & (nx <= .90) & (ny >= .02) & (ny <= .95)
    background = substrate(lab, card, flame_zone)
    mask = fixed_shape_mask(lab, flame_zone, background)
    return summary(lab, mask, background)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    x, y, groups, _ = aligned_data(args.cache, args.video_root)
    use = np.isin(groups, FAMILIES["B"])
    x, y, groups = x[use], y[use], groups[use]
    baselines = {run: calibration_contrast(args.video_root, run)
                 for run in FAMILIES["B"]}

    results = {}
    feature_sets = {}
    for floor in (.5, 1.0, 2.0, 4.0, 8.0):
        denominator = np.asarray([
            np.maximum(np.abs(baselines[str(run)][:6]), floor) for run in groups])
        ratio = x[:, :6] / denominator
        combined = np.c_[x[:, :6], ratio]
        variants = {
            f"ratio6_floor{floor:g}": (ratio, tuple(range(6))),
            f"ratio_ab_floor{floor:g}": (ratio, (1, 2, 4, 5)),
            f"raw_ratio_floor{floor:g}": (combined, tuple(range(12))),
            f"raw_ratio_ab_floor{floor:g}": (combined, (1, 2, 4, 5, 7, 8, 10, 11)),
        }
        for variant, (values, feature) in variants.items():
            feature_sets[variant] = values
            for fraction in (.50, .70, .90):
                for cap in (6, 12, 20):
                    for kind in ("lda", "logistic", "svm"):
                        result = evaluate(values, y, groups, FAMILIES["B"],
                                          (0, 1, 2), feature, fraction, cap, kind)
                        if result is not None:
                            name = f"{variant}_{kind}_f{fraction:.2f}_cap{cap}"
                            results[name] = result
    chosen = select(results); best = results[chosen]
    payload = {
        "protocol": "complete-video-held-out; normalization uses calibration frame only",
        "baseline_contrast": {run: value[:6].tolist() for run, value in baselines.items()},
        "selected": chosen, "result": best,
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
             title=f"Calibration-normalized family B\naccuracy {best['exact']:.1%}")
    fig.savefig(args.output / "normalization_confusion.png", dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
