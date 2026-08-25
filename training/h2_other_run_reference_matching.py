"""Match other H2-only videos to the fixed-mask test_2 optical reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from h2_more_crop_fixed_mask_analysis import (
    RUNS as TIGHT_RUNS,
    extract_run as extract_tight_run,
    fixed_shape_mask,
    frame_at,
    substrate,
    summary,
)


OTHER_RUNS = {
    "test": {
        "file": "1_90_H2_only_test_cropped.mp4",
        "roi": (730, 140, 1320, 860),
        "times": [2, 15, 25, 30, 40, 100],
        "flame_y": .43, "drop_y": .74,
    },
    "run4": {
        "file": "1_90_H2_only_4_cropped.mp4",
        "roi": (660, 150, 1280, 930),
        "times": [2, 13, 30, 90, 109, 122],
        "flame_y": .42, "drop_y": .74,
    },
    "run5": {
        "file": "1_90_H2_only_5_cropped.mp4",
        "roi": (540, 0, 1240, 870),
        "times": [2, 8, 13, 21, 80, 130],
        "flame_y": .32, "drop_y": .70,
    },
}


def resize_roi(frame: np.ndarray, roi, width=480):
    x1, y1, x2, y2 = roi
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (width, round(cropped.shape[0] * width / cropped.shape[1])),
                      interpolation=cv2.INTER_AREA)


def card_zones(shape, flame_y=.42, drop_y=.74):
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    nx, ny = xx / width, yy / height
    flame = (((nx - .50) / .20) ** 2 + ((ny - flame_y) / .205) ** 2 <= 1) & (nx <= .69)
    main = ((nx - .46) / .19) ** 2 + ((ny - drop_y) / .155) ** 2 <= 1
    satellite = ((nx - .69) / .09) ** 2 + ((ny - (drop_y + .06)) / .095) ** 2 <= 1
    drop = main | satellite
    card = (nx >= .04) & (nx <= .95) & (ny >= .02) & (ny <= .96)
    return flame, drop, card


def extract_other(video_root: Path, name: str, config: dict, sample_hz: float):
    cap = cv2.VideoCapture(str(video_root / config["file"]))
    if not cap.isOpened():
        raise FileNotFoundError(video_root / config["file"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    calibration = resize_roi(frame_at(cap, 2.0), config["roi"])
    lab0 = cv2.cvtColor(calibration, cv2.COLOR_BGR2LAB).astype(float)
    flame_zone, drop_zone, card = card_zones(
        lab0.shape[:2], config["flame_y"], config["drop_y"])
    bg0 = substrate(lab0, card, flame_zone | drop_zone)
    flame_mask = fixed_shape_mask(lab0, flame_zone, bg0)
    drop_mask = fixed_shape_mask(lab0, drop_zone, bg0)
    base_flame = summary(lab0, flame_mask, bg0)
    base_drop = summary(lab0, drop_mask, bg0)
    rows = []
    for seconds in np.arange(0, max(0, duration - .5 / fps), 1 / sample_hz):
        image = resize_roi(frame_at(cap, float(seconds)), config["roi"])
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
        bg = substrate(lab, card, flame_zone | drop_zone)
        flame = summary(lab, flame_mask, bg)
        drop = summary(lab, drop_mask, bg)
        feature = np.r_[flame - base_flame, (flame - drop) - (base_flame - base_drop)]
        rows.append({"run": name, "time": float(seconds), "feature": feature})
    cap.release()
    return rows, calibration, flame_mask, drop_mask


def save_masks(review, output):
    panels = []
    for name, image, flame, drop in review:
        overlay = image.copy()
        overlay[flame] = (.3 * overlay[flame] + .7 * np.asarray([0, 0, 255])).astype(np.uint8)
        overlay[drop] = (.3 * overlay[drop] + .7 * np.asarray([255, 180, 0])).astype(np.uint8)
        cv2.putText(overlay, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(overlay)
    height = min(panel.shape[0] for panel in panels)
    panels = [cv2.resize(panel, (round(panel.shape[1] * height / panel.shape[0]), height))
              for panel in panels]
    cv2.imwrite(str(output), np.hstack(panels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=2.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows, review = [], []
    for name in ("test_2", "test_3"):
        extracted, image, flame, drop = extract_tight_run(
            args.video_root, name, TIGHT_RUNS[name], args.sample_hz)
        rows.extend(extracted)
        review.append((name, image, flame, drop))
    for name, config in OTHER_RUNS.items():
        extracted, image, flame, drop = extract_other(args.video_root, name, config, args.sample_hz)
        rows.extend(extracted)
        review.append((name, image, flame, drop))
    save_masks(review, args.output / "all_run_fixed_masks.jpg")

    reference = [row for row in rows if row["run"] == "test_2"]
    reference_windows = {0: (0, 3), 1: (12, 14), 2: (20, 22),
                         3: (29, 31), 4: (51, 90)}
    centroids = {
        stage: np.median([row["feature"][:6] for row in reference
                          if start <= row["time"] <= end], axis=0)
        for stage, (start, end) in reference_windows.items()
    }
    reference_x = np.asarray([row["feature"][:6] for row in reference])
    scale = np.maximum(np.median(np.abs(reference_x - np.median(reference_x, axis=0)), axis=0), .5)

    matches = {}
    plot_rows = {}
    requested = {**{name: config["times"] for name, config in OTHER_RUNS.items()},
                 "test_3": [2, 10, 20, 28, 40, 60, 90, 120, 150]}
    for name in requested:
        selected = [row for row in rows if row["run"] == name]
        trajectory = []
        for row in selected:
            distances = {stage: float(np.linalg.norm((row["feature"][:6] - center) / scale))
                         for stage, center in centroids.items()}
            optical = min(distances, key=distances.get)
            trajectory.append((row["time"], optical, distances[optical]))
        # A 2.5 s median rejects isolated exposure/keyframe changes.
        stages = np.asarray([item[1] for item in trajectory], dtype=float)
        radius = max(1, round(args.sample_hz * 1.25))
        smooth = np.asarray([np.median(stages[max(0, i-radius):i+radius+1])
                             for i in range(len(stages))])
        plot_rows[name] = (np.asarray([item[0] for item in trajectory]), smooth)
        matches[name] = []
        for seconds in requested[name]:
            index = int(np.argmin(np.abs(plot_rows[name][0] - seconds)))
            matches[name].append({"time": float(plot_rows[name][0][index]),
                                  "test2_optical_stage": float(smooth[index]),
                                  "distance": float(trajectory[index][2])})

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for name, (time, optical) in plot_rows.items():
        ax.plot(time, optical, lw=1.4, label=name)
    ax.set(yticks=range(5), ylim=(-.2, 4.2), xlabel="video time (s)",
           ylabel="test_2-equivalent optical H2 stage")
    ax.legend(ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "test2_reference_matching.png", dpi=180)
    plt.close(fig)
    (args.output / "matches.json").write_text(json.dumps(matches, indent=2), encoding="utf-8")
    print(json.dumps(matches, indent=2))


if __name__ == "__main__":
    main()
