"""Evaluate a calibration-locked flame mask for app-domain family-B H2 ranges."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from h2_app_fixed_mask_router_analysis import (
    GRID_X, GRID_Y, calibration_grid, metadata, prepared,
)
from h2_app_family_concentration_analysis import (
    DISPLAY, FEATURES as DYNAMIC_FEATURES, WINDOWS, balanced_weights, estimator,
    label_at, score, stable_blocks,
)
from train_models import frame_at


RUNS = {
    "1_90_RH20_3_x2.mp4": ("run3", 1.0, "1_90_RH20_3_x2.mp4"),
    # Use the higher-quality normal-speed source. Geometry is identical to the
    # x2 encoding, so its calibration circle metadata remains applicable.
    "1_90_RH20_4.mp4": ("run4", 2.0, "1_90_RH20_4_x2.mp4"),
}
LABELS = (0, 1, 2)
SUMMARY_NAMES = ["mean_L", "mean_a", "mean_b", "median_L", "median_a", "median_b",
                 "chroma_p10", "chroma_p25", "chroma_p50", "chroma_p75", "chroma_p90"]
NAMES = [f"d_{name}" for name in SUMMARY_NAMES] + [f"baseline_{name}" for name in SUMMARY_NAMES]
FEATURES = {
    "mean": (0, 1, 2), "ab": (1, 2, 4, 5),
    "lab6": tuple(range(6)), "delta11": tuple(range(11)),
    "all22": tuple(range(22)),
}


def labelled_times(cache: Path):
    output = {run: [] for run, _, _ in RUNS.values()}
    video_to_run = {video: run for video, (run, _, _) in RUNS.items()}
    with cache.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            run = video_to_run.get(row["video"])
            if run is None:
                continue
            seconds = float(row["time"])
            label = label_at(run, seconds)
            if label is not None:
                output[run].append((seconds, label))
    # The cache contains the x2 encoding, while this audit deliberately uses
    # the clearer normal-speed run4. Restore the user-reviewed normal timeline.
    output["run4"] = []
    normal_windows = {0: ((0, 8),), 1: ((30, 50),), 2: ((50, 90),)}
    for seconds in np.arange(0, 90.01, .5):
        hits = [label for label, windows in normal_windows.items()
                if any(start <= seconds <= end for start, end in windows)]
        if len(hits) == 1:
            output["run4"].append((float(seconds), hits[0]))
    return output


def registered_summary(frame, circle, orientation, grid):
    lab, _, zone, nx, ny = prepared(frame, circle, orientation)
    gx = np.floor((nx + .55) / .90 * GRID_X).astype(int)
    gy = np.floor((ny + .62) / .76 * GRID_Y).astype(int)
    valid = zone & (gx >= 0) & (gx < GRID_X) & (gy >= 0) & (gy < GRID_Y)
    selected = np.zeros(zone.shape, dtype=bool)
    selected[valid] = grid[gy[valid], gx[valid]]
    local = zone & ~selected
    substrate = lab[local]
    center = np.median(substrate, axis=0)
    distance = np.sqrt((.25 * (substrate[:, 0] - center[0])) ** 2
                       + (substrate[:, 1] - center[1]) ** 2
                       + (substrate[:, 2] - center[2]) ** 2)
    background = np.median(substrate[distance <= np.percentile(distance, 45)], axis=0)
    pixels = lab[selected] - background
    chroma = np.hypot(pixels[:, 1], pixels[:, 2])
    return np.r_[np.mean(pixels, axis=0), np.median(pixels, axis=0),
                 np.percentile(chroma, (10, 25, 50, 75, 90))]


def vector(current, baseline):
    return np.r_[current - baseline, baseline]


def extract(video_root: Path, source_cache: Path, label_cache: Path, cache: Path):
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        return saved["x"], saved["y"], saved["groups"], saved["times"]
    geometry = metadata(source_cache)
    times = labelled_times(label_cache)
    rows = []
    for video, (run, calibration_time, metadata_video) in RUNS.items():
        cap = cv2.VideoCapture(str(video_root / video))
        if not cap.isOpened():
            raise FileNotFoundError(video_root / video)
        meta = geometry[metadata_video]
        calibration = frame_at(cap, calibration_time)
        grid = calibration_grid(calibration, meta["circle"], meta["orientation"])
        baseline = registered_summary(calibration, meta["circle"], meta["orientation"], grid)
        for seconds, label in times[run]:
            current = registered_summary(frame_at(cap, seconds), meta["circle"],
                                         meta["orientation"], grid)
            rows.append((vector(current, baseline), label, run, seconds))
        cap.release()
    x = np.asarray([row[0] for row in rows]); y = np.asarray([row[1] for row in rows])
    groups = np.asarray([row[2] for row in rows]); seconds = np.asarray([row[3] for row in rows])
    np.savez_compressed(cache, x=x, y=y, groups=groups, times=seconds)
    return x, y, groups, seconds


def evaluate(x, y, groups, feature, fraction, cap, kind):
    fx = x[:, feature]
    prediction = np.full(len(y), -1)
    for held_out in sorted(set(groups)):
        test, train = groups == held_out, groups != held_out
        selected = stable_blocks(fx, y, groups, train, LABELS, fraction, cap)
        if set(y[selected]) != set(LABELS):
            return None
        weights = balanced_weights(y, groups, selected, LABELS)
        if kind.startswith("hierarchical_"):
            base_kind = kind.split("_", 1)[1]
            gate = estimator(base_kind); stage = estimator(base_kind)
            positive = selected & (y > 0)
            if base_kind in ("logistic", "svm"):
                gate.fit(fx[selected], y[selected] > 0, **{
                    f"{gate.steps[-1][0]}__sample_weight": weights[selected]})
                stage.fit(fx[positive], y[positive], **{
                    f"{stage.steps[-1][0]}__sample_weight": weights[positive]})
            else:
                gate.fit(fx[selected], y[selected] > 0)
                stage.fit(fx[positive], y[positive])
            response = gate.predict(fx[test]).astype(bool)
            fold = np.zeros(int(test.sum()), dtype=int)
            if response.any():
                fold[response] = stage.predict(fx[test][response])
            prediction[test] = fold
            continue
        model = estimator(kind)
        if kind in ("logistic", "svm"):
            model.fit(fx[selected], y[selected], **{
                f"{model.steps[-1][0]}__sample_weight": weights[selected]})
        else:
            model.fit(fx[selected], y[selected])
        prediction[test] = model.predict(fx[test])
    return score(y, prediction, groups, LABELS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--label-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-js", type=Path)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    x, y, groups, times = extract(
        args.video_root, args.source_cache, args.label_cache,
        args.output / "fixed_mask_b_rows.npz")
    results = {}
    variants = {name: (x, feature) for name, feature in FEATURES.items()}
    baseline_contrast = x[:, 11:17]
    for floor in (.5, 1.0, 2.0, 4.0, 8.0):
        denominator = np.maximum(np.abs(baseline_contrast), floor)
        ratio6 = x[:, :6] / denominator
        combined = np.c_[x[:, :6], ratio6]
        variants.update({
            f"ratio6_floor{floor:g}": (ratio6, tuple(range(6))),
            f"ratio_ab_floor{floor:g}": (ratio6, (1, 2, 4, 5)),
            f"raw_ratio_floor{floor:g}": (combined, tuple(range(12))),
            f"raw_ratio_ab_floor{floor:g}": (combined, (1, 2, 4, 5, 7, 8, 10, 11)),
        })
    for feature_name, (values, feature) in variants.items():
        for fraction in (.50, .70, .90):
            for cap in (6, 12, 20):
                for kind in ("lda", "logistic", "svm",
                             "hierarchical_lda", "hierarchical_logistic"):
                    result = evaluate(values, y, groups, feature, fraction, cap, kind)
                    if result is not None:
                        results[f"{feature_name}_{kind}_f{fraction:.2f}_cap{cap}"] = result
    selected = max(results, key=lambda name: (
        results[name]["minimum_recall"], results[name]["video_macro_exact"],
        results[name]["exact"]))
    best = results[selected]
    variant_name = next(name for name in variants if selected.startswith(name + "_"))
    remainder = selected[len(variant_name) + 1:].split("_")
    kind, fraction, cap = remainder[0], float(remainder[1][1:]), int(remainder[2][3:])
    values, feature = variants[variant_name]
    selected_rows = stable_blocks(values[:, feature], y, groups, np.ones(len(y), dtype=bool),
                                  LABELS, fraction, cap)
    final_model = estimator(kind)
    weights = balanced_weights(y, groups, selected_rows, LABELS)
    if kind in ("logistic", "svm"):
        final_model.fit(values[selected_rows][:, feature], y[selected_rows], **{
            f"{final_model.steps[-1][0]}__sample_weight": weights[selected_rows]})
    else:
        final_model.fit(values[selected_rows][:, feature], y[selected_rows])
    export = None
    if kind in ("lda", "logistic"):
        scaler = final_model.named_steps["standardscaler"]
        fitted = final_model.steps[-1][1]
        coefficients = fitted.coef_ / scaler.scale_[None, :]
        intercepts = fitted.intercept_ - coefficients @ scaler.mean_
        export = {
            "schema_version": 1, "type": "fixed_flame_linear_scores",
            "environment": "B", "feature_profile": variant_name,
            "source_features": [NAMES[index] if values is x else f"derived_{index}"
                                for index in feature],
            "classes": list(LABELS), "display_levels": [DISPLAY[label] for label in LABELS],
            "coefficients": coefficients.tolist(), "intercepts": intercepts.tolist(),
            "grid": {"width": GRID_X, "height": GRID_Y,
                     "x_min": -.55, "x_max": .35, "y_min": -.62, "y_max": .14},
            "ratio_floor": float(variant_name.split("floor", 1)[1])
                           if "ratio6_floor" in variant_name else None,
            "validation": {key: best[key] for key in ("exact", "minimum_recall", "recall")},
        }
        raw = values[:, feature] @ coefficients.T + intercepts
        parity = np.asarray(LABELS)[np.argmax(raw, axis=1)]
        export["full_fit_exact"] = float(np.mean(parity == final_model.predict(values[:, feature])))
        js = "// Generated by training/h2_app_family_b_fixed_mask_analysis.py; do not edit by hand.\n"
        js += "window.SENSOR_H2_FAMILY_B_MODEL=" + json.dumps(export, separators=(",", ":")) + ";\n"
        (args.output / "sensor-h2-family-b-model.js").write_text(js, encoding="utf-8")
        if args.export_js:
            args.export_js.write_text(js, encoding="utf-8")
    payload = {"protocol": "calibration-locked 36x32 flame mask; complete-video-held-out",
               "labels": {"run3": "0; 1-2 at 35-55 s; 2-3 at 55-60 s",
                          "run4": "normal-speed source; 0; 1-2 at 30-50 s; 2-3 at 50-90 s"},
               "selected": selected, "result": best, "features": NAMES,
               "models": results, "export": export}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    matrix = np.asarray(best["confusion"])
    fig, axis = plt.subplots(figsize=(5.0, 4.4), constrained_layout=True)
    axis.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    ticks = [DISPLAY[label] for label in LABELS]
    axis.set(xticks=range(3), xticklabels=ticks, yticks=range(3), yticklabels=ticks,
             xlabel="Predicted", ylabel="Reference",
             title=f"Family B fixed flame mask\naccuracy {best['exact']:.1%}")
    fig.savefig(args.output / "fixed_mask_b_confusion.png", dpi=190); plt.close(fig)
    print(json.dumps({"selected": selected, **best}, indent=2))


if __name__ == "__main__":
    main()
