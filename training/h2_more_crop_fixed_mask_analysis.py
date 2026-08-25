"""Audit user-supplied tight H2 crops with calibration-locked shape masks.

This is deliberately an evaluation script, not an app-model exporter.  It asks
whether removing chamber/circle geometry from the two trustworthy H2 ramps
makes their calibrated flame-colour trajectories transferable across runs.
"""

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RUNS = {
    "test_2": {
        "file": "1_90_H2_only_test_2_more_cropped.mp4",
        "points": [(0, 0), (4, 0), (13, 1), (21, 2), (30, 3), (51, 4)],
    },
    "test_3": {
        "file": "1_90_H2_only_test_3_more_cropped.mp4",
        "points": [(0, 0), (3, 0), (10, 1), (20, 2), (28, 3), (152, 4)],
    },
}


def frame_at(cap: cv2.VideoCapture, seconds: float) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame at {seconds:.2f} s")
    return frame


def content_crop(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Remove the editor's black side bars without following image content."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    occupied = np.mean(gray > 8, axis=0) > .35
    indices = np.flatnonzero(occupied)
    if len(indices) < frame.shape[1] * .2:
        return frame, (0, frame.shape[1])
    left, right = int(indices[0]), int(indices[-1] + 1)
    return frame[:, left:right], (left, right)


def canonical(frame: np.ndarray, bounds: tuple[int, int], width: int = 480) -> np.ndarray:
    cropped = frame[:, bounds[0]:bounds[1]]
    height = round(cropped.shape[0] * width / cropped.shape[1])
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)


def zones(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    nx, ny = xx / width, yy / height
    # The tight crops share the same editor framing.  Shape-centred ellipses
    # exclude the diagonal cable and gray patch above the flame, the white
    # patch left of the droplet, and the physical gap between the two shapes.
    flame = ((nx - .50) / .225) ** 2 + ((ny - .415) / .195) ** 2 <= 1.0
    main_drop = ((nx - .47) / .19) ** 2 + ((ny - .735) / .17) ** 2 <= 1.0
    satellite = ((nx - .68) / .09) ** 2 + ((ny - .80) / .105) ** 2 <= 1.0
    drop = main_drop | satellite
    card = (nx >= .08) & (nx <= .90) & (ny >= .02) & (ny <= .92)
    return flame, drop, card


def substrate(lab: np.ndarray, card: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    pixels = lab[card & ~excluded].astype(float)
    center = np.median(pixels, axis=0)
    distance = np.sqrt((.25 * (pixels[:, 0] - center[0])) ** 2
                       + (pixels[:, 1] - center[1]) ** 2
                       + (pixels[:, 2] - center[2]) ** 2)
    return np.median(pixels[distance <= np.percentile(distance, 45)], axis=0)


def fixed_shape_mask(lab: np.ndarray, zone: np.ndarray, background: np.ndarray) -> np.ndarray:
    distance = np.sqrt((.30 * (lab[:, :, 0] - background[0])) ** 2
                       + (lab[:, :, 1] - background[1]) ** 2
                       + (lab[:, :, 2] - background[2]) ** 2)
    values = distance[zone]
    cutoff = max(5.0, float(np.percentile(values, 62)))
    raw = (distance >= cutoff) & zone
    raw = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN,
                           np.ones((3, 3), np.uint8)).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), 8)
    keep = np.zeros_like(raw)
    components = sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    for component in components[:5]:
        if stats[component, cv2.CC_STAT_AREA] >= 18:
            keep |= labels == component
    if keep.sum() < 80:
        raise RuntimeError("Calibration shape mask is too small")
    return keep


def summary(lab: np.ndarray, mask: np.ndarray, background: np.ndarray) -> np.ndarray:
    pixels = lab[mask].astype(float) - background
    chroma = np.hypot(pixels[:, 1], pixels[:, 2])
    return np.asarray([
        *np.mean(pixels, axis=0),
        *np.median(pixels, axis=0),
        *np.percentile(chroma, [10, 25, 50, 75, 90]),
    ])


def target(points: list[tuple[float, float]], seconds: float) -> tuple[float, int]:
    times, values = zip(*points)
    continuous = float(np.interp(seconds, times, values))
    return continuous, int(np.clip(np.floor(continuous + .5), 0, 4))


def extract_run(video_root: Path, name: str, config: dict, sample_hz: float):
    cap = cv2.VideoCapture(str(video_root / config["file"]))
    if not cap.isOpened():
        raise FileNotFoundError(video_root / config["file"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    calibration_time = 2.0
    calibration_raw = frame_at(cap, calibration_time)
    _, bounds = content_crop(calibration_raw)
    calibration = canonical(calibration_raw, bounds)
    lab0 = cv2.cvtColor(calibration, cv2.COLOR_BGR2LAB).astype(float)
    flame_zone, drop_zone, card = zones(lab0.shape[:2])
    background0 = substrate(lab0, card, flame_zone | drop_zone)
    flame_mask = fixed_shape_mask(lab0, flame_zone, background0)
    drop_mask = fixed_shape_mask(lab0, drop_zone, background0)
    base_flame = summary(lab0, flame_mask, background0)
    base_drop = summary(lab0, drop_mask, background0)
    rows = []
    # OpenCV commonly reports a fractional duration just beyond the last
    # decodable timestamp.  Stop half a frame early.
    for seconds in np.arange(0, max(0, duration - .5 / fps), 1 / sample_hz):
        raw = frame_at(cap, float(seconds))
        image = canonical(raw, bounds)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
        background = substrate(lab, card, flame_zone | drop_zone)
        flame = summary(lab, flame_mask, background)
        drop = summary(lab, drop_mask, background)
        continuous, stage = target(config["points"], float(seconds))
        # Calibrated flame change plus the flame-minus-inert-droplet change.
        feature = np.r_[flame - base_flame, (flame - drop) - (base_flame - base_drop)]
        rows.append({"run": name, "time": float(seconds), "continuous": continuous,
                     "stage": stage, "feature": feature})
    cap.release()
    return rows, calibration, flame_mask, drop_mask


def montage(items, output: Path) -> None:
    panels = []
    for name, image, flame, drop in items:
        overlay = image.copy()
        overlay[flame] = (.35 * overlay[flame] + .65 * np.array([0, 0, 255])).astype(np.uint8)
        overlay[drop] = (.35 * overlay[drop] + .65 * np.array([255, 180, 0])).astype(np.uint8)
        cv2.putText(overlay, f"{name}: red=flame, blue=drop", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(overlay)
    height = min(panel.shape[0] for panel in panels)
    panels = [cv2.resize(panel, (round(panel.shape[1] * height / panel.shape[0]), height))
              for panel in panels]
    cv2.imwrite(str(output), np.hstack(panels))


def trajectory_plot(rows, output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.0), sharex=False)
    for name, color in (("test_2", "#1565c0"), ("test_3", "#ef6c00")):
        selected = [row for row in rows if row["run"] == name]
        time = np.asarray([row["time"] for row in selected])
        feature = np.asarray([row["feature"] for row in selected])
        for channel, label in enumerate(("L*", "a*", "b*")):
            axes[channel].plot(time, feature[:, channel], color=color, lw=1.5, label=name)
            axes[channel].set_ylabel(f"flame Δ{label}")
            axes[channel].axhline(0, color="#999", lw=.6)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("video time (s)")
    fig.suptitle("Calibration-locked flame trajectory from tight crops")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def optical_reference_match(rows):
    """Match test_3 frames to test_2 optical stages without its timeline."""
    reference = [row for row in rows if row["run"] == "test_2"]
    query = [row for row in rows if row["run"] == "test_3"]
    x = np.asarray([row["feature"][:6] for row in reference])
    scale = np.maximum(np.median(np.abs(x - np.median(x, axis=0)), axis=0), .5)
    centroids = {
        stage: np.median([row["feature"][:6] for row in reference if row["stage"] == stage], axis=0)
        for stage in range(5)
    }
    output = []
    for seconds in (2, 10, 20, 28, 40, 60, 90, 120, 150):
        row = min(query, key=lambda item: abs(item["time"] - seconds))
        distance = {stage: float(np.linalg.norm((row["feature"][:6] - center) / scale))
                    for stage, center in centroids.items()}
        optical = min(distance, key=distance.get)
        output.append({"time": row["time"], "timeline_stage": row["stage"],
                       "test2_optical_stage": optical,
                       "distance": distance[optical]})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=2.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows, review = [], []
    for name, config in RUNS.items():
        extracted, image, flame, drop = extract_run(args.video_root, name, config, args.sample_hz)
        rows.extend(extracted)
        review.append((name, image, flame, drop))
    montage(review, args.output / "fixed_shape_masks.jpg")
    trajectory_plot(rows, args.output / "calibrated_flame_trajectory.png")
    optical_match = optical_reference_match(rows)

    models = {
        "multinomial_logistic": make_pipeline(StandardScaler(), LogisticRegression(
            C=.3, max_iter=3000, class_weight="balanced", random_state=42)),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=.04, max_leaf_nodes=15,
            min_samples_leaf=12, l2_regularization=3, random_state=42),
    }
    y = np.asarray([row["stage"] for row in rows])
    x = np.asarray([row["feature"] for row in rows])
    groups = np.asarray([row["run"] for row in rows])
    results = {}
    predictions = {}
    for model_name, model in models.items():
        predicted = np.full(len(y), -1)
        per_run = {}
        for held_out in RUNS:
            train, test = groups != held_out, groups == held_out
            model.fit(x[train], y[train])
            predicted[test] = model.predict(x[test])
            per_run[held_out] = float(accuracy_score(y[test], predicted[test]))
        exact = float(accuracy_score(y, predicted))
        within_one = float(np.mean(np.abs(y - predicted) <= 1))
        mae = float(np.mean(np.abs(y - predicted)))
        band_truth = np.select([y <= 1, y <= 3], [0, 1], default=2)
        band_pred = np.select([predicted <= 1, predicted <= 3], [0, 1], default=2)
        results[model_name] = {"exact": exact, "within_one": within_one, "mae": mae,
                               "range_exact_0-1_2-3_4": float(accuracy_score(band_truth, band_pred)),
                               "per_run_exact": per_run,
                               "confusion": confusion_matrix(y, predicted, labels=range(5)).tolist()}
        predictions[model_name] = predicted

    best_name = max(results, key=lambda name: results[name]["exact"])
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    ConfusionMatrixDisplay.from_predictions(y, predictions[best_name], labels=range(5),
                                             normalize="true", cmap="Blues", ax=ax)
    ax.set_title(f"Tight-crop fixed-mask held-out: {best_name}")
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    with (args.output / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "time", "continuous", "stage", *[f"f{i}" for i in range(x.shape[1])]])
        for row in rows:
            writer.writerow([row["run"], row["time"], row["continuous"], row["stage"], *row["feature"]])
    (args.output / "metrics.json").write_text(json.dumps(
        {"protocol": "train one complete video, test the other", "best": best_name,
         "models": results, "test3_to_test2_optical_match": optical_match}, indent=2),
         encoding="utf-8")
    print(json.dumps({"best": best_name, **results[best_name]}, indent=2))


if __name__ == "__main__":
    main()
