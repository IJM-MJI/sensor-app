"""Validate an immediate 3/5-frame burst against Place-2 RH frame outliers."""

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

from make_endpoint_mask_review import source_path
from rh_40_50_cross_run_spatial_analysis import extract
from rh_place1_external_validation import control_vector, nearest_row
from rh_place2_profile_stable_model import (
    CALIBRATION, LABELS, LEVELS, MIN_SEPARATION, RADIUS, STABLE, STEP,
    VALIDATION, VIDEOS, median_near, predict_1nn,
)
from rh_place2_seven_band_run_holdout import matrices
from train_models import CACHE_VERSION, read_csv


BURSTS = {"single": (0,), "burst3": (-1, 0, 1), "burst5": (-2, -1, 0, 1, 2)}


def video_fps(video_root, video):
    capture = cv2.VideoCapture(str(source_path(video_root, video)))
    fps = float(capture.get(cv2.CAP_PROP_FPS)); capture.release()
    if not np.isfinite(fps) or fps < 10:
        raise RuntimeError(f"Invalid FPS for {video}: {fps}")
    return fps


def draw(results, output):
    names = list(results); x = np.arange(len(names)); width = .34
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    best = max(names, key=lambda name: results[name]["exact_accuracy"])
    matrix = np.asarray(results[best]["row_normalized_confusion_0_to_1"])
    axes[0].imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for row in range(7):
        for column in range(7):
            value = matrix[row, column]
            axes[0].text(column, row, f"{value:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if value >= .65 else "#1f2937")
    axes[0].set_xticks(range(7), LABELS, rotation=35, ha="right")
    axes[0].set_yticks(range(7), LABELS)
    axes[0].set(xlabel="Predicted", ylabel="Reference",
                title=f"{best} row-normalized confusion (0-1)")
    axes[1].bar(x-width/2, [results[n]["exact_accuracy"] for n in names],
                width, label="Exact")
    axes[1].bar(x+width/2, [results[n]["within_one_adjacent_range"] for n in names],
                width, label="Within one")
    axes[1].axhline(.85, color="crimson", linestyle="--", label="0.85 target")
    axes[1].set_xticks(x, names); axes[1].set_ylim(0, 1.03); axes[1].legend()
    axes[1].set(ylabel="Score (0-1)", title="Immediate micro-burst comparison")
    fig.suptitle("Place-2 RH single-frame outlier suppression", fontweight="bold")
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"rh_place2_microburst_validation.{suffix}", **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_place2_microburst_validation_v1"))
    parser.add_argument("--app-model", type=Path, default=Path(
        "sensor-rh-place2-stable-profile-model.js"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cached = read_csv(args.cache)
    fps = {group: video_fps(args.video_root, video)
           for group, video in VIDEOS.items()}
    kept = {group: tuple((time, level) for time, level in VALIDATION[group]
                         if min(abs(time - selected) for selected in STABLE[group])
                         >= MIN_SEPARATION)
            for group in VIDEOS}

    required = {}
    for group in VIDEOS:
        times = {CALIBRATION[group]}
        for centre in STABLE[group]:
            times.update(np.arange(centre - RADIUS,
                                   centre + RADIUS + STEP / 2, STEP))
        for centre, _ in kept[group]:
            for offsets in BURSTS.values():
                times.update(centre + offset / fps[group] for offset in offsets)
        required[group] = sorted(round(float(value), 5) for value in times if value >= 0)

    items = []
    for group, video in VIDEOS.items():
        for time in required[group]:
            items.append({"group": group, "video": video, "time": time,
                          "row": nearest_row(cached, video, time),
                          "calibration": abs(time - CALIBRATION[group]) < 1e-6})
    summaries = extract(items, args.video_root)
    controls = [control_vector(summary) for summary in summaries]
    baseline = {item["group"]: value for item, value in zip(items, controls)
                if item["calibration"]}
    dense = {}
    for group in VIDEOS:
        pairs = sorted((item["time"], value - baseline[group])
                       for item, value in zip(items, controls)
                       if item["group"] == group and not item["calibration"])
        dense[group] = (np.asarray([pair[0] for pair in pairs]),
                        np.asarray([pair[1] for pair in pairs]))

    train = {}
    for group in VIDEOS:
        times, vectors = dense[group]
        train[group] = np.asarray([median_near(times, vectors, centre)[0]
                                   for centre in STABLE[group]])

    truth = np.asarray([level for group in VIDEOS for _, level in kept[group]], dtype=int)
    results, rows = {}, []
    for burst, offsets in BURSTS.items():
        predictions = []
        for group in VIDEOS:
            times, vectors = dense[group]
            for centre, level in kept[group]:
                wanted = np.asarray([centre + offset / fps[group] for offset in offsets])
                chosen = np.asarray([vectors[np.argmin(np.abs(times - time))]
                                     for time in wanted])
                feature = np.median(chosen, axis=0)
                prediction = int(predict_1nn(train[group], LEVELS,
                                             feature[None])[0])
                predictions.append(prediction)
                rows.append({"method": burst, "group": group, "time_s": centre,
                             "fps": fps[group], "span_ms": round(
                                 (max(offsets) - min(offsets)) / fps[group] * 1000, 1),
                             "reference": level, "prediction": prediction,
                             "correct": level == prediction})
        results[burst] = matrices(truth, np.asarray(predictions))

    outer = {25, 35, 65, 75, 85}
    decision_rows = {}
    for burst in BURSTS:
        subset = [row for row in rows if row["method"] == burst and
                  row["reference"] in outer]
        decision_rows[burst] = float(np.mean([row["correct"] for row in subset]))
    best = max(results, key=lambda name: (
        results[name]["exact_accuracy"], -len(BURSTS[name])))
    deploy = bool(best == "burst3" and results[best]["exact_accuracy"] >= .85
                  and decision_rows[best] >= .85)
    payload = {
        "scope": "profile-specific stable RH prototypes with immediate adjacent-frame burst",
        "fps": fps,
        "results": results,
        "outer_band_accuracy": decision_rows,
        "decision": {"best": best, "deploy_burst3": deploy,
                     "rule": "burst3 exact and outer-band accuracy >= 0.85",
                     "warning": "same-run validation; monitor/video frames, not live camera burst"},
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    app_model = {
        "version": "2026-08-31-place2-stable-profile-burst3-v1",
        "type": "profile_standardized_1nn",
        "levels": LEVELS.tolist(),
        "display_levels": LABELS,
        "input": "three-frame RGB median, then calibrated droplet-substrate LAB delta",
        "profiles": {
            group: {
                "calibration_top_a": 130.0 if group == "response3" else 128.9,
                "prototypes": train[group].tolist(),
                "scaler_scale": np.maximum(np.std(train[group], axis=0), .5).tolist(),
                "stable_centres_s": list(STABLE[group]),
            }
            for group in VIDEOS
        },
    }
    args.app_model.write_text(
        "window.SENSOR_RH_PLACE2_STABLE_PROFILE_MODEL=" +
        json.dumps(app_model, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    draw(results, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
