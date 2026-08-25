"""Find RH20 runs that reach a test_2-equivalent high H2 optical state."""

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

from h2_more_crop_fixed_mask_analysis import frame_at, fixed_shape_mask, substrate, summary
from h2_other_run_reference_matching import resize_roi


RUNS = {
    "run2": {"file": "1_90_RH20_2_x2_cropped.mp4", "roi": (650, 50, 1320, 970),
             "cal": 2, "reaction": (10, 60), "recovery": (60, 85),
             "flame": (.34, .38, .22, .19)},
    "run3": {"file": "1_90_RH20_3_x2_cropped.mp4", "roi": (720, 150, 1270, 860),
             "cal": 0, "reaction": (0, 60), "recovery": (60, 97),
             "flame": (.31, .41, .20, .17)},
    "run4": {"file": "1_90_RH20_4_cropped.mp4", "roi": (660, 150, 1280, 930),
             "cal": 0, "reaction": (0, 120), "recovery": (120, 180),
             "flame": (.32, .35, .22, .18)},
    "run5_x2": {"file": "1_90_RH20_5_x2_cropped.mp4", "roi": (680, 100, 1280, 970),
                "cal": 0, "reaction": (0, 54), "recovery": (54, 107),
                "flame": (.47, .42, .20, .18)},
    "run5_normal": {"file": "1_90_RH20_cropped.mp4", "roi": (680, 100, 1280, 970),
                    "cal": 0, "reaction": (0, 90), "recovery": (90, 214),
                    "flame": (.47, .42, .20, .18)},
}


def reference(reference_csv: Path):
    rows = []
    with reference_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["run"] == "test_2":
                rows.append((float(row["time"]), np.asarray(
                    [float(row[f"f{i}"]) for i in range(6)])))
    windows = {0: (0, 3), 1: (12, 14), 2: (20, 22), 3: (29, 31), 4: (51, 90)}
    centers = {stage: np.median([value for time, value in rows if start <= time <= end], axis=0)
               for stage, (start, end) in windows.items()}
    # Mean and median a* are the physically relevant yellow-to-green channels.
    values = np.asarray([[value[1], value[4]] for _, value in rows])
    scale = np.maximum(np.median(np.abs(values - np.median(values, axis=0)), axis=0), .35)
    return centers, scale


def extract(video_root: Path, name: str, config: dict, sample_hz: float):
    cap = cv2.VideoCapture(str(video_root / config["file"]))
    if not cap.isOpened():
        raise FileNotFoundError(video_root / config["file"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    calibration = resize_roi(frame_at(cap, config["cal"]), config["roi"])
    lab0 = cv2.cvtColor(calibration, cv2.COLOR_BGR2LAB).astype(float)
    height, width = lab0.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    nx, ny = xx / width, yy / height
    fx, fy, rx, ry = config["flame"]
    flame_zone = ((nx - fx) / rx) ** 2 + ((ny - fy) / ry) ** 2 <= 1
    if name == "run3":
        flame_zone &= ny >= .25
    card = (nx >= .02) & (nx <= .90) & (ny >= .02) & (ny <= .95)
    background0 = substrate(lab0, card, flame_zone)
    flame_mask = fixed_shape_mask(lab0, flame_zone, background0)
    baseline = summary(lab0, flame_mask, background0)
    rows = []
    for seconds in np.arange(0, min(duration - .1, config["recovery"][1]), 1 / sample_hz):
        image = resize_roi(frame_at(cap, float(seconds)), config["roi"])
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
        background = substrate(lab, card, flame_zone)
        delta = summary(lab, flame_mask, background) - baseline
        rows.append({"time": float(seconds), "delta": delta})
    cap.release()
    return rows, calibration, flame_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=1.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    centers, scale = reference(args.reference_csv)
    results, trajectories, reviews = {}, {}, []
    for name, config in RUNS.items():
        rows, image, mask = extract(args.video_root, name, config, args.sample_hz)
        reviews.append((name, image, mask))
        time = np.asarray([row["time"] for row in rows])
        delta_a = np.asarray([row["delta"][1] for row in rows])
        stages, distances = [], []
        for row in rows:
            query = np.asarray([row["delta"][1], row["delta"][4]])
            distance = {stage: float(np.linalg.norm(
                (query - np.asarray([center[1], center[4]])) / scale))
                for stage, center in centers.items()}
            stages.append(min(distance, key=distance.get))
            distances.append(min(distance.values()))
        stages = np.asarray(stages, dtype=float)
        radius = max(1, round(args.sample_hz * 2))
        smooth = np.asarray([np.median(stages[max(0, i-radius):i+radius+1])
                             for i in range(len(stages))])
        reaction_end = config["reaction"][1]
        late = (time >= reaction_end - 10) & (time <= reaction_end)
        recovery_end = min(config["recovery"][1], time[-1])
        # User timelines mark the recovery endpoint as the fully recovered
        # state; the preceding interval is still a descending optical response.
        recovered = time >= recovery_end - 2
        results[name] = {
            "late_window": [reaction_end - 10, reaction_end],
            "late_median_stage": float(np.median(smooth[late])),
            "late_stage_counts": {str(int(stage)): int(count) for stage, count in
                                  zip(*np.unique(smooth[late], return_counts=True))},
            "late_delta_a_median": float(np.median(delta_a[late])),
            "late_delta_a_range": [float(np.min(delta_a[late])), float(np.max(delta_a[late]))],
            "late_distance_median": float(np.median(np.asarray(distances)[late])),
            "recovery_tail_delta_a_median": float(np.median(delta_a[recovered])),
            "recovery_tail_stage_median": float(np.median(smooth[recovered])),
        }
        trajectories[name] = (time, delta_a, smooth)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=False)
    for name, (time, delta_a, stage) in trajectories.items():
        axes[0].plot(time, delta_a, label=name, lw=1.3)
        axes[1].plot(time, stage, label=name, lw=1.3)
    for stage in (2, 3, 4):
        axes[0].axhline(centers[stage][1], color="#999", ls="--", lw=.7)
        axes[0].text(axes[0].get_xlim()[1], centers[stage][1], f" test2 {stage}%",
                     va="center", fontsize=8)
    axes[0].set_ylabel("calibration-relative flame Δa*")
    axes[0].legend(ncol=3)
    axes[1].set(yticks=range(5), ylim=(-.2, 4.2), xlabel="video time (s)",
                ylabel="test2-equivalent stage (a* only)")
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "rh20_max_response.png", dpi=180)
    plt.close(fig)

    panels = []
    for name, image, mask in reviews:
        overlay = image.copy()
        overlay[mask] = (.3 * overlay[mask] + .7 * np.asarray([0, 0, 255])).astype(np.uint8)
        cv2.putText(overlay, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(overlay)
    height = min(panel.shape[0] for panel in panels)
    panels = [cv2.resize(panel, (round(panel.shape[1] * height / panel.shape[0]), height))
              for panel in panels]
    cv2.imwrite(str(args.output / "fixed_flame_masks.jpg"), np.hstack(panels))
    payload = {"test2_reference_delta_a": {str(stage): float(center[1])
                                             for stage, center in centers.items()},
               "runs": results}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
