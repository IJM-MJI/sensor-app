"""Test place-1 indoor-long RH ordering on late one-second frame blocks.

These overlapping blocks are pseudo-replicates for stability/noise analysis,
not independent experimental runs.  Every prediction recomputes hue midpoints
after withholding the tested block.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from make_endpoint_mask_review import source_path
from rh_paired_pixel_hue_analysis import balanced_frame_and_masks, endpoint_rows
from rh_tight_relative_analysis import angle, tight_masks
from train_models import CACHE_VERSION, read_csv, resize_for_app


LEVELS = np.asarray([25.0, 40.0, 50.0, 60.0])
DISPLAY = ("20-30", "40", "50", "60")


def classify(hue, class_hues):
    """Nearest ordered hue centroid; midpoint-equivalent for this trajectory."""
    ordered = sorted(class_hues.items())
    return min(ordered, key=lambda item: abs(hue - item[1]))[0]


def extract_blocks(items, video_root, span=3.0, duration=1.0, step=.5):
    output = []
    for item in items:
        path = source_path(video_root, item["video"])
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        starts = np.arange(item["time"] - span, item["time"] - duration + 1e-6, step)
        for start in starts:
            hues, lightness, chroma, counts = [], [], [], []
            # Stay just inside the block so a decoder cannot cross the nominal
            # endpoint into the next ramp interval.
            margin = min(.08, duration * .10)
            for seconds in np.linspace(start + margin, start + duration - margin, 5):
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0, seconds) * 1000)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode {path.name} at {seconds:.2f}s")
                frame = resize_for_app(frame)
                lab, _, selected = balanced_frame_and_masks(frame, item["row"])
                circle = tuple(int(float(item["row"][name]))
                               for name in ("circle_x", "circle_y", "circle_r"))
                main, _ = tight_masks(lab.shape, circle, item["row"], selected)
                pixels = lab[main].astype(float)
                if len(pixels) < 12:
                    continue
                a, b = pixels[:, 1] - 128, pixels[:, 2] - 128
                weights = np.maximum(np.hypot(a, b), 3.0)
                summary = {"circular": np.asarray([
                    np.average(np.sin(np.arctan2(b, a)), weights=weights),
                    np.average(np.cos(np.arctan2(b, a)), weights=weights),
                ])}
                hues.append(np.degrees(angle(summary)))
                lightness.append(float(np.median(pixels[:, 0])))
                chroma.append(float(np.median(np.hypot(a, b))))
                counts.append(int(len(pixels)))
            if len(hues) < 3:
                continue
            output.append({
                "video": item["video"], "reference": float(item["stage"]),
                "endpoint_time": float(item["time"]), "block_start": float(start),
                "block_end": float(start + duration), "hue": float(np.median(hues)),
                "lightness": float(np.median(lightness)),
                "chroma": float(np.median(chroma)),
                "n_frames": len(hues), "median_pixels": float(np.median(counts)),
            })
        cap.release()
    return output


def leave_one_block_out(rows):
    truth, prediction, details = [], [], []
    for index, row in enumerate(rows):
        class_hues = {}
        for level in LEVELS:
            values = [value["hue"] for other, value in enumerate(rows)
                      if other != index and value["reference"] == level]
            class_hues[float(level)] = float(np.median(values))
        pred = classify(row["hue"], class_hues)
        truth.append(row["reference"]); prediction.append(pred)
        details.append({**row, "prediction": pred,
                        "correct": int(pred == row["reference"])})
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    matrix = confusion_matrix(truth, prediction, labels=LEVELS)
    recalls = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    return details, {
        "exact_accuracy": float(np.mean(truth == prediction)),
        "balanced_accuracy": float(np.mean(recalls)),
        "mae_percent_rh": float(np.mean(abs(truth - prediction))),
        "per_stage_recall": recalls.tolist(), "confusion": matrix.tolist(),
    }


def summaries(rows):
    result = {}
    for level in LEVELS:
        use = [row for row in rows if row["reference"] == level]
        result[str(int(level))] = {
            name: {"median": float(np.median([row[name] for row in use])),
                   "min": float(np.min([row[name] for row in use])),
                   "max": float(np.max([row[name] for row in use])),
                   "std": float(np.std([row[name] for row in use]))}
            for name in ("hue", "lightness", "chroma")
        }
    medians = [result[str(int(level))]["hue"]["median"] for level in LEVELS]
    result["hue_midpoint_boundaries"] = [
        float((medians[index] + medians[index + 1]) / 2)
        for index in range(len(medians) - 1)]
    result["adjacent_hue_ranges_overlap"] = [
        not (result[str(int(LEVELS[index]))]["hue"]["min"] >
             result[str(int(LEVELS[index + 1]))]["hue"]["max"])
        for index in range(len(LEVELS) - 1)]
    result["monotonic_more_orange_order"] = bool(
        all(medians[index] > medians[index + 1]
            for index in range(len(medians) - 1)))
    return result


def plot(output, rows, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1), constrained_layout=True)
    for index, level in enumerate(LEVELS):
        use = [row for row in rows if row["reference"] == level]
        axes[0].scatter([index] * len(use), [row["hue"] for row in use], s=30)
        axes[0].plot([index-.18, index+.18], [np.median([r["hue"] for r in use])]*2,
                     color="black", linewidth=2)
    axes[0].set_xticks(range(4), DISPLAY); axes[0].invert_yaxis()
    axes[0].set(xlabel="Reference RH (%)", ylabel="Large-droplet hue (degrees)",
                title="Late-block stability\n(lower = more orange)")
    matrix = np.asarray(metrics["confusion"], dtype=float)
    norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    axes[1].imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for row in range(4):
        for column in range(4):
            value = norm[row, column]
            axes[1].text(column, row, f"{value:.2f}", ha="center", va="center",
                         color="white" if value > .55 else "black")
    axes[1].set_xticks(range(4), DISPLAY); axes[1].set_yticks(range(4), DISPLAY)
    axes[1].set(xlabel="Predicted RH (%)", ylabel="Reference RH (%)",
                title="Leave-one-block-out")
    fig.suptitle("Place-1 indoor-long relative-hue audit", fontweight="bold")
    fig.savefig(output / "rh_indoor_long_late_block_validation.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_indoor_long_late_blocks_v1"))
    parser.add_argument("--span", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=.5)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    items = [item for item in endpoint_rows(read_csv(args.cache))
             if item["group"] == "rh-indoor-long" and item["stage"] <= 60
             and item["video"] == "1_90_H2O_only_extract_3min.mp4"]
    rows = extract_blocks(items, args.video_root, args.span, args.duration, args.step)
    predictions, metrics = leave_one_block_out(rows)
    summary = summaries(rows)
    decision = {
        "pseudo_block_gate_pass": bool(
            metrics["balanced_accuracy"] >= .85
            and min(metrics["per_stage_recall"]) >= .85
            and not any(summary["adjacent_hue_ranges_overlap"])
            and summary["monotonic_more_orange_order"]),
        "app_deploy": False,
        "reason": "Pseudo-blocks are correlated within one run; independent-run evidence is still required.",
    }
    payload = {"scope": (f"place-1 indoor-long endpoint-final {args.span:g} s; "
                         f"{args.duration:g} s blocks, {args.step:g} s step"),
               "warning": "Stability audit only, not independent-run validation",
               "metrics": metrics, "stage_summary": summary, "decision": decision}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.output / "blocks.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    plot(args.output, rows, metrics)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
