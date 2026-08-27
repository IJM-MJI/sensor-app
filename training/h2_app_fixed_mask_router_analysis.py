"""Audit an app-compatible, calibration-locked flame mask for A/B routing."""

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train_models import (
    frame_at, normalized_coordinates, patch_balance_lab, resize_for_app,
    shape_pixel_mask,
)


RUNS = {
    "1_90_H2_only_test_2.mp4": ("A", (0.0, 3.0), 2.0),
    "1_90_H2_only_test_3.MOV": ("A", (0.0, 2.5), 2.0),
    "1_90_H2_only_test.mp4": ("A", (0.0, 3.0), 2.0),
    "1_90_RH20_2_x2.mp4": ("A", (10.0, 14.0), 12.0),
    "1_90_RH20_3_x2.mp4": ("B", (0.0, 4.0), 1.0),
    "1_90_RH20_4_x2.mp4": ("B", (0.0, 8.0), 2.0),
    "1_90_RH20_5_x2.mp4": ("C", (0.0, 4.0), 1.0),
}

# Browser-sized occupancy grid over the normalised flame search zone.
GRID_X, GRID_Y = 36, 32
NAMES = [
    "flame_L_mean", "flame_a_mean", "flame_b_mean",
    "flame_L_p25", "flame_L_p50", "flame_L_p75",
    "flame_a_p10", "flame_a_p25", "flame_a_p50", "flame_a_p75", "flame_a_p90",
    "flame_b_p10", "flame_b_p25", "flame_b_p50", "flame_b_p75", "flame_b_p90",
    "flame_chroma_p25", "flame_chroma_p50", "flame_chroma_p75",
    "bg_L", "bg_a", "bg_b", "mask_fraction",
]
for colour_space in ("lab", "hsv"):
    for channel in range(3):
        for statistic in ("p10", "p25", "p50", "p75", "p90", "mean", "std"):
            NAMES.append(f"env_{colour_space}_{channel}_{statistic}")
FEATURES = {
    "mean": (0, 1, 2),
    "ab_distribution": tuple(range(6, 19)),
    "mean_distribution": tuple(range(19)),
    "flame_all": tuple(range(23)),
    "env_lab": tuple(range(23, 44)),
    "env_hsv": tuple(range(44, 65)),
    "env_all": tuple(range(23, 65)),
    "flame_env": tuple(range(len(NAMES))),
}


def metadata(cache: Path):
    output = {}
    with cache.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            video = row["video"]
            if video in RUNS and video not in output:
                output[video] = {
                    "circle": (int(float(row["circle_x"])), int(float(row["circle_y"])),
                               int(float(row["circle_r"]))),
                    "orientation": int(float(row.get("orientation_quarters") or 0)),
                }
    missing = sorted(set(RUNS) - set(output))
    if missing:
        raise RuntimeError(f"Missing geometry metadata: {missing}")
    return output


def prepared(frame, circle, orientation):
    image = resize_for_app(frame)
    raw_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    x, y, r = circle
    yy, xx = np.ogrid[:raw_lab.shape[0], :raw_lab.shape[1]]
    chamber = (xx - x) ** 2 + (yy - y) ** 2 <= round(r * .90) ** 2
    nx, ny = normalized_coordinates(raw_lab.shape, circle, orientation)
    nx = np.broadcast_to(nx, raw_lab.shape[:2])
    ny = np.broadcast_to(ny, raw_lab.shape[:2])
    # Match the deployed path: chamber-neutral balance, not patch-specific balance.
    lab = patch_balance_lab(raw_lab, chamber)
    pixels = lab[chamber].astype(float)
    chroma = np.hypot(pixels[:, 1] - 128, pixels[:, 2] - 128)
    bg = pixels[np.argsort(chroma)[:max(1, int(len(pixels) * .50))]].mean(axis=0)
    flame_zone = chamber & (nx >= -.55) & (nx <= .35) & (ny >= -.62) & (ny <= .14)
    return lab.astype(float), bg, flame_zone, nx, ny


def calibration_grid(frame, circle, orientation):
    lab, bg, zone, nx, ny = prepared(frame, circle, orientation)
    # Calibration is the yellow/olive H2=0 flame. A broad percentile mask also
    # selected substrate texture across the whole search rectangle. Restrict to
    # the physical upper mark and require positive b* contrast, then retain only
    # substantial connected ink components.
    tight = zone & (nx >= -.40) & (nx <= .25) & (ny >= -.58) & (ny <= .08)
    selected = shape_pixel_mask(lab, tight, bg, percentile=55.0)
    selected &= (lab[:, :, 2] - bg[2]) >= 3.0
    binary = selected.astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros(selected.shape, dtype=bool)
    ranked = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        cx, cy = centroids[component]
        local_x = (cx - circle[0]) / max(circle[2], 1)
        local_y = (cy - circle[1]) / max(circle[2], 1)
        position = np.exp(-2.0 * ((local_x + .08) ** 2 + (local_y + .27) ** 2))
        ranked.append((area * position, component))
    for _, component in sorted(ranked, reverse=True)[:6]:
        keep |= labels == component
    if int(keep.sum()) >= 20:
        selected = keep
    gx = np.floor((nx + .55) / .90 * GRID_X).astype(int)
    gy = np.floor((ny + .62) / .76 * GRID_Y).astype(int)
    valid = tight & (gx >= 0) & (gx < GRID_X) & (gy >= 0) & (gy < GRID_Y)
    total = np.zeros((GRID_Y, GRID_X), dtype=int)
    chosen = np.zeros_like(total)
    np.add.at(total, (gy[valid], gx[valid]), 1)
    np.add.at(chosen, (gy[selected & valid], gx[selected & valid]), 1)
    # Keep cells whose calibration pixels were predominantly part of the colour mask.
    grid = (chosen >= 2) & (chosen / np.maximum(total, 1) >= .35)
    if int(grid.sum()) < 12:
        grid = chosen > 0
    return grid


def fixed_features(frame, circle, orientation, grid):
    image = resize_for_app(frame)
    lab, bg, zone, nx, ny = prepared(frame, circle, orientation)
    gx = np.floor((nx + .55) / .90 * GRID_X).astype(int)
    gy = np.floor((ny + .62) / .76 * GRID_Y).astype(int)
    valid = zone & (gx >= 0) & (gx < GRID_X) & (gy >= 0) & (gy < GRID_Y)
    selected = np.zeros(zone.shape, dtype=bool)
    selected[valid] = grid[gy[valid], gx[valid]]
    pixels = lab[selected]
    if len(pixels) < 20:
        raise RuntimeError("Calibration-locked flame mask has too few pixels")
    chroma = np.hypot(pixels[:, 1] - 128, pixels[:, 2] - 128)
    values = {
        "flame_L_mean": pixels[:, 0].mean(),
        "flame_a_mean": pixels[:, 1].mean(),
        "flame_b_mean": pixels[:, 2].mean(),
        "flame_L_p25": np.percentile(pixels[:, 0], 25),
        "flame_L_p50": np.percentile(pixels[:, 0], 50),
        "flame_L_p75": np.percentile(pixels[:, 0], 75),
    }
    for channel, data in (("a", pixels[:, 1]), ("b", pixels[:, 2])):
        for percentile in (10, 25, 50, 75, 90):
            values[f"flame_{channel}_p{percentile}"] = np.percentile(data, percentile)
    for percentile in (25, 50, 75):
        values[f"flame_chroma_p{percentile}"] = np.percentile(chroma, percentile)
    values.update({"bg_L": bg[0], "bg_a": bg[1], "bg_b": bg[2],
                   "mask_fraction": selected.sum() / max(zone.sum(), 1)})
    height, width = image.shape[:2]
    crop = image[int(.08 * height):int(.92 * height),
                 int(.08 * width):int(.92 * width)]
    for colour_space, converted in (
        ("lab", cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)),
        ("hsv", cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)),
    ):
        converted = converted.reshape(-1, 3).astype(float)
        for channel in range(3):
            vector = converted[:, channel]
            stats = (*np.percentile(vector, (10, 25, 50, 75, 90)),
                     np.mean(vector), np.std(vector))
            for statistic, result in zip(
                    ("p10", "p25", "p50", "p75", "p90", "mean", "std"), stats):
                values[f"env_{colour_space}_{channel}_{statistic}"] = result
    return np.asarray([float(values[name]) for name in NAMES])


def extract(video_root: Path, source_cache: Path, cache: Path):
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        return saved["x"], saved["y"], saved["groups"]
    meta = metadata(source_cache)
    rows = []
    for video, (family, window, calibration_time) in RUNS.items():
        cap = cv2.VideoCapture(str(video_root / video))
        if not cap.isOpened():
            raise FileNotFoundError(video_root / video)
        geometry = meta[video]
        grid = calibration_grid(frame_at(cap, calibration_time), geometry["circle"],
                                geometry["orientation"])
        for seconds in np.linspace(window[0], window[1], 8):
            vector = fixed_features(frame_at(cap, float(seconds)), geometry["circle"],
                                    geometry["orientation"], grid)
            rows.append((vector, family, video))
        cap.release()
    x = np.asarray([row[0] for row in rows])
    y = np.asarray([row[1] for row in rows])
    groups = np.asarray([row[2] for row in rows])
    np.savez_compressed(cache, x=x, y=y, groups=groups)
    return x, y, groups


def estimator(kind):
    if kind == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"))
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.2, max_iter=4000, class_weight="balanced", random_state=42))
    return make_pipeline(StandardScaler(), SVC(
        C=1, gamma="scale", class_weight="balanced", random_state=42))


def evaluate(x, y, groups, index, kind):
    use = np.isin(y, ("A", "B"))
    x, y, groups = x[use][:, index], y[use], groups[use]
    prediction = np.full(len(y), "?", dtype="<U1")
    for held_out in sorted(set(groups)):
        test, train = groups == held_out, groups != held_out
        model = estimator(kind)
        model.fit(x[train], y[train])
        prediction[test] = model.predict(x[test])
    matrix = confusion_matrix(y, prediction, labels=("A", "B"))
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    per_run = {run: float(np.mean(prediction[groups == run] == y[groups == run]))
               for run in sorted(set(groups))}
    return {"exact": float(np.mean(prediction == y)),
            "video_macro_exact": float(np.mean(list(per_run.values()))),
            "recall": {"A": float(recall[0]), "B": float(recall[1])},
            "minimum_recall": float(recall.min()), "confusion": matrix.tolist(),
            "per_run_exact": per_run}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-js", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x, y, groups = extract(args.video_root, args.source_cache,
                           args.output / "fixed_mask_router_rows.npz")
    results = {}
    for feature_name, index in FEATURES.items():
        for kind in ("lda", "logistic", "svm"):
            results[f"{feature_name}_{kind}"] = evaluate(x, y, groups, index, kind)
    selected = max(results, key=lambda name: (
        results[name]["minimum_recall"], results[name]["video_macro_exact"],
        results[name]["exact"]))
    best = results[selected]
    passed = best["recall"]["A"] >= .85 and best["recall"]["B"] >= .85
    feature_name, kind = selected.rsplit("_", 1)
    use = np.isin(y, ("A", "B"))
    final_model = estimator(kind)
    final_model.fit(x[use][:, FEATURES[feature_name]], y[use])
    export = None
    if kind == "svm":
        scaler = final_model.named_steps["standardscaler"]
        svc = final_model.named_steps["svc"]
        export = {
            "schema_version": 1,
            "type": "standardized_rbf_svm",
            "scope": "calibration frame only",
            "feature_set": feature_name,
            "features": [NAMES[index] for index in FEATURES[feature_name]],
            "classes": svc.classes_.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "support_vectors": svc.support_vectors_.tolist(),
            "dual_coef": svc.dual_coef_[0].tolist(),
            "intercept": float(svc.intercept_[0]),
            "gamma": float(svc._gamma),
        }
        js = "// Generated by training/h2_app_fixed_mask_router_analysis.py; do not edit by hand.\n"
        js += "window.SENSOR_H2_ENVIRONMENT_ROUTER=" + json.dumps(export, separators=(",", ":")) + ";\n"
        (args.output / "sensor-h2-environment-router.js").write_text(js, encoding="utf-8")
        if args.export_js:
            args.export_js.write_text(js, encoding="utf-8")
    payload = {
        "protocol": "calibration frame only; candidate fixed flame mask and full-frame colour features; complete video held out",
        "acceptance": "A recall >= 0.85 and B recall >= 0.85",
        "passed": passed, "selected": selected, "A_B": best,
        "features": NAMES, "models": results, "export": export,
        "C_independent_validation": False,
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    matrix = np.asarray(best["confusion"])
    fig, axis = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(xticks=(0, 1), xticklabels=("A", "B"), yticks=(0, 1), yticklabels=("A", "B"),
             xlabel="Predicted", ylabel="Reference",
             title=f"Calibration-locked flame mask\naccuracy {best['exact']:.1%}")
    fig.savefig(args.output / "fixed_mask_router_confusion.png", dpi=190)
    plt.close(fig)
    print(json.dumps({"passed": passed, "selected": selected, **best}, indent=2))


if __name__ == "__main__":
    main()
