"""Evaluate spatial flame colour features inside H2 environment families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from h2_environment_family_analysis import FAMILIES, prepare
from h2_family_landmark_refinement import evaluate, select
from h2_four_range_analysis import DISPLAY
from h2_more_crop_fixed_mask_analysis import (
    RUNS as TIGHT_RUNS, canonical, content_crop, fixed_shape_mask, frame_at,
    substrate, zones,
)
from h2_other_run_reference_matching import OTHER_RUNS, card_zones, resize_roi
from h2_rh20_max_response_analysis import RUNS as RH20_RUNS


REGIONS = ("global", "top", "bottom", "left", "right")
STATS = ("mean_l", "mean_a", "mean_b", "median_l", "median_a", "median_b",
         "q25_a", "q75_a", "q25_b", "q75_b")


class VideoContext:
    def __init__(self, root: Path, run: str):
        self.run = run
        if run in TIGHT_RUNS:
            config = TIGHT_RUNS[run]
            self.cap = cv2.VideoCapture(str(root / config["file"]))
            calibration_raw = frame_at(self.cap, 2.0)
            _, self.bounds = content_crop(calibration_raw)
            self.kind, self.config = "tight", config
            calibration = canonical(calibration_raw, self.bounds)
            flame_zone, drop_zone, card = zones(calibration.shape[:2])
        elif run == "test":
            config = OTHER_RUNS[run]
            self.cap = cv2.VideoCapture(str(root / config["file"]))
            self.kind, self.config = "other", config
            calibration = resize_roi(frame_at(self.cap, 2.0), config["roi"])
            flame_zone, drop_zone, card = card_zones(
                calibration.shape[:2], config["flame_y"], config["drop_y"])
        else:
            config = RH20_RUNS[run]
            self.cap = cv2.VideoCapture(str(root / config["file"]))
            self.kind, self.config = "rh20", config
            calibration = resize_roi(frame_at(self.cap, config["cal"]), config["roi"])
            height, width = calibration.shape[:2]
            yy, xx = np.ogrid[:height, :width]; nx, ny = xx / width, yy / height
            fx, fy, rx, ry = config["flame"]
            flame_zone = ((nx - fx) / rx) ** 2 + ((ny - fy) / ry) ** 2 <= 1
            if run == "run3":
                flame_zone &= ny >= .25
            drop_zone = np.zeros_like(flame_zone)
            card = (nx >= .02) & (nx <= .90) & (ny >= .02) & (ny <= .95)
        if not self.cap.isOpened():
            raise FileNotFoundError(run)
        lab0 = cv2.cvtColor(calibration, cv2.COLOR_BGR2LAB).astype(float)
        background0 = substrate(lab0, card, flame_zone | drop_zone)
        self.mask = fixed_shape_mask(lab0, flame_zone, background0)
        self.card, self.drop_zone = card, drop_zone
        self.region_masks = split_regions(self.mask)
        self.baseline = spatial_summary(lab0, self.region_masks, background0)

    def image(self, seconds: float):
        raw = frame_at(self.cap, seconds)
        if self.kind == "tight":
            return canonical(raw, self.bounds)
        return resize_roi(raw, self.config["roi"])

    def feature(self, seconds: float):
        image = self.image(seconds)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
        background = substrate(lab, self.card, self.mask | self.drop_zone)
        return spatial_summary(lab, self.region_masks, background) - self.baseline

    def close(self):
        self.cap.release()


def split_regions(mask):
    yy, xx = np.indices(mask.shape)
    y_mid = float(np.median(yy[mask])); x_mid = float(np.median(xx[mask]))
    return (mask, mask & (yy <= y_mid), mask & (yy > y_mid),
            mask & (xx <= x_mid), mask & (xx > x_mid))


def spatial_summary(lab, masks, background):
    output = []
    relative = lab.astype(float) - background
    for mask in masks:
        pixels = relative[mask]
        output.extend((
            *np.mean(pixels, axis=0), *np.median(pixels, axis=0),
            *np.percentile(pixels[:, 1], (25, 75)),
            *np.percentile(pixels[:, 2], (25, 75)),
        ))
    return np.asarray(output)


def extract(root, cache, base_cache):
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        return saved["x"], saved["y"], saved["groups"], saved["times"]
    _, y, groups, times = prepare(base_cache)
    use = np.isin(groups, FAMILIES["A"] + FAMILIES["B"])
    y, groups, times = y[use], groups[use], times[use]
    y[(groups == "test_3") & (y == 3)] = 2
    y[(groups == "test_3") & (times >= 24) & (times <= 28)] = 1
    y[(groups == "run4") & (times >= 65) & (times <= 78)] = 2
    x = np.zeros((len(y), len(REGIONS) * len(STATS)))
    for run in sorted(set(groups)):
        context = VideoContext(root, run)
        indices = np.flatnonzero(groups == run)
        for index in indices:
            x[index] = context.feature(float(times[index]))
        context.close()
    np.savez_compressed(cache, x=x, y=y, groups=groups, times=times)
    return x, y, groups, times


def feature_sets():
    sets = {"global": tuple(range(len(STATS)))}
    a_offsets = (1, 4, 6, 7)
    ab_offsets = (1, 2, 4, 5, 6, 7, 8, 9)
    sets["spatial_a"] = tuple(region * len(STATS) + offset
                               for region in range(len(REGIONS)) for offset in a_offsets)
    sets["spatial_ab"] = tuple(region * len(STATS) + offset
                                for region in range(len(REGIONS)) for offset in ab_offsets)
    sets["all_spatial"] = tuple(range(len(REGIONS) * len(STATS)))
    return sets


def run_family(x, y, groups, family, labels):
    results = {}
    for feature_name, feature in feature_sets().items():
        for fraction in (.50, .70, .90):
            for cap in (6, 12):
                for kind in ("lda", "logistic", "svm"):
                    result = evaluate(x, y, groups, family, labels, feature,
                                      fraction, cap, kind)
                    if result is not None:
                        results[f"{feature_name}_{kind}_f{fraction:.2f}_cap{cap}"] = result
    chosen = select(results)
    return chosen, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "spatial_flame_rows.npz"
    x, y, groups, _ = extract(args.video_root, cache, args.base_cache)
    selected_a, results_a = run_family(x, y, groups, FAMILIES["A"], (0, 1, 2, 3))
    selected_b, results_b = run_family(x, y, groups, FAMILIES["B"], (0, 1, 2))
    payload = {
        "features": {"regions": list(REGIONS), "statistics": list(STATS)},
        "protocol": "calibration-relative fixed-mask spatial colour; complete video held out",
        "A": {"selected": selected_a, "models": results_a},
        "B": {"selected": selected_b, "models": results_b},
        "deployment_ready": False,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    for axis, family, result, labels in (
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
                 title=f"Spatial family {family}: {result['exact']:.1%}\n"
                       f"min recall {result['minimum_recall']:.1%}")
    fig.savefig(args.output / "spatial_family_confusions.png", dpi=190)
    plt.close(fig)
    print(json.dumps({
        "A": {"selected": selected_a, **results_a[selected_a]},
        "B": {"selected": selected_b, **results_b[selected_b]},
    }, indent=2))


if __name__ == "__main__":
    main()
