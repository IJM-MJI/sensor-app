"""Frozen Place-2 RH model validation on the independent daylight recovery run."""

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
from rh_place1_external_validation import control_vector, load_model, nearest_row, predict
from rh_place2_seven_band_run_holdout import LABELS, LEVELS, matrices
from train_models import CACHE_VERSION, read_csv


VIDEO = "1_90_H2O_only.MOV"
CALIBRATION_TIME = 38.0  # RH20 recovery tail
# Predeclared: 0.5 s before each descending segment boundary.
POINTS = (
    (7.5, 85), (10.5, 75), (12.5, 65), (14.5, 55),
    (19.5, 45), (22.5, 35), (29.5, 25),
)
BURSTS = {"single": (0,), "latest_trailing3": (-2, -1, 0)}


def video_fps(video_root: Path) -> float:
    cap = cv2.VideoCapture(str(source_path(video_root, VIDEO)))
    fps = float(cap.get(cv2.CAP_PROP_FPS)); cap.release()
    if not np.isfinite(fps) or fps < 10:
        raise RuntimeError(f"Invalid FPS: {fps}")
    return fps


def draw(results, output):
    best = max(results, key=lambda key: results[key]["exact_accuracy"])
    values = np.asarray(results[best]["row_normalized_confusion_0_to_1"])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), constrained_layout=True)
    axes[0].imshow(values, cmap="Blues", vmin=0, vmax=1)
    for row in range(7):
        for column in range(7):
            value = values[row, column]
            axes[0].text(column, row, f"{value:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if value >= .65 else "#1f2937")
    axes[0].set_xticks(range(7), LABELS, rotation=35, ha="right")
    axes[0].set_yticks(range(7), LABELS)
    axes[0].set(xlabel="Predicted", ylabel="Reference",
                title=f"{best} row-normalized confusion (0–1)")
    names = list(results); x = np.arange(len(names)); width = .32
    axes[1].bar(x-width, [results[n]["exact_accuracy"] for n in names], width,
                label="Exact")
    axes[1].bar(x, [results[n]["balanced_accuracy"] for n in names], width,
                label="Balanced")
    axes[1].bar(x+width, [results[n]["within_one_adjacent_range"] for n in names], width,
                label="Within one")
    axes[1].axhline(.85, color="crimson", linestyle="--", label="0.85 target")
    axes[1].set_xticks(x, [n.replace("_", "\n") for n in names])
    axes[1].set_ylim(0, 1.03); axes[1].legend()
    axes[1].set(ylabel="Score (0–1)", title="Frozen model external result")
    fig.suptitle("Daylight RH90→20 recovery external validation", fontweight="bold")
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"rh_daylight_recovery_external_validation.{suffix}", **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--model", type=Path,
                        default=Path("sensor-rh-place2-seven-band-model.js"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_daylight_recovery_external_validation_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cache = read_csv(args.cache); fps = video_fps(args.video_root)
    requested = {CALIBRATION_TIME}
    for centre, _ in POINTS:
        for offsets in BURSTS.values():
            requested.update(centre + offset / fps for offset in offsets)
    items = [{"group": "daylight-recovery", "video": VIDEO, "time": seconds,
              "row": nearest_row(cache, VIDEO, seconds)} for seconds in sorted(requested)]
    summaries = extract(items, args.video_root)
    controls = {round(item["time"], 6): control_vector(summary)
                for item, summary in zip(items, summaries)}
    baseline = controls[round(CALIBRATION_TIME, 6)]
    model = load_model(args.model)

    results, rows = {}, []
    truth = np.asarray([level for _, level in POINTS], dtype=int)
    for name, offsets in BURSTS.items():
        prediction = []
        for centre, level in POINTS:
            vectors = np.asarray([
                controls[round(centre + offset / fps, 6)] - baseline
                for offset in offsets])
            vector = np.median(vectors, axis=0)
            predicted, distance, margin = predict(vector, model)
            prediction.append(predicted)
            rows.append({"method": name, "video": VIDEO, "time_s": centre,
                         "reference": level, "prediction": predicted,
                         "correct": level == predicted, "distance": distance,
                         "margin": margin, "delta_L": vector[0],
                         "delta_a": vector[1], "delta_b": vector[2]})
        results[name] = matrices(truth, np.asarray(prediction))
    best = max(results, key=lambda key: (
        results[key]["exact_accuracy"], results[key]["balanced_accuracy"]))
    chosen = results[best]
    payload = {
        "scope": "frozen Place-2 seven-band model on independent daylight descending recovery",
        "video": VIDEO,
        "calibration_time_s": CALIBRATION_TIME,
        "points": POINTS,
        "results": results,
        "decision": {
            "best": best,
            "passes_all_0_85": bool(
                chosen["exact_accuracy"] >= .85
                and chosen["balanced_accuracy"] >= .85
                and min(chosen["per_range_recall"].values()) >= .85),
            "deploy": False,
            "warning": "Descending recovery tests cross-direction generalization and hysteresis; no timing was tuned on this video.",
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    draw(results, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
