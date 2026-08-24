"""A/B test conservative droplet ROIs and within-yellow relative colour.

The large sensing droplet and its small satellite are measured separately.  An
inner registered template suppresses chamber hardware/board edges, while hue,
chroma and lightness are expressed relative to each run's 20-30 % baseline.
Evaluation is the same nested complete-run holdout used by the paired-hue audit.
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

from make_endpoint_mask_review import source_path
from rh_human_color_path_analysis import LEVELS, DISPLAY
from rh_paired_pixel_hue_analysis import (
    balanced_frame_and_masks, build_features, deployment_decision, endpoint_rows,
    extract, paired_summary, report, tune_and_evaluate,
)
from train_models import CACHE_VERSION, normalized_coordinates, read_csv, resize_for_app


FEATURE_NAMES = (
    "legacy_delta_lab", "paired_named", "tight_main_relative",
    "tight_main_satellite_relative",
)


def tight_masks(shape, circle, row, selected):
    """Return conservative cores in calibration-registered droplet coordinates."""
    orientation = int(float(row["orientation_quarters"]))
    nx, ny = normalized_coordinates(shape, circle, orientation)
    values = [row.get(f"drop_registration_{name}") for name in ("x", "y", "angle")]
    if all(value not in (None, "") for value in values):
        center_x, center_y, angle = (float(value) for value in values)
        dx, dy = nx - center_x, ny - center_y
        cosine, sine = np.cos(angle), np.sin(angle)
        local_x = cosine * dx - sine * dy
        local_y = sine * dx + cosine * dy
    else:
        # Same semantic coordinate system as the unregistered template.
        local_x, local_y = nx + .08, ny - .43
    # Cores are deliberately smaller than the .25/.29 and .14/.18 masks used
    # for discovery.  Discovery finds ink; quantification avoids its boundary.
    main = (local_x / .215) ** 2 + (local_y / .245) ** 2 <= 1.0
    satellite = ((local_x - .30) / .105) ** 2 + ((local_y - .02) / .135) ** 2 <= 1.0
    return selected & main, selected & satellite


def extract_tight(items, video_root):
    by_video = defaultdict(list)
    for index, item in enumerate(items):
        by_video[item["video"]].append((index, item))
    summaries = [None] * len(items)
    for video, indexed in by_video.items():
        cap = cv2.VideoCapture(str(source_path(video_root, video)))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        for index, item in sorted(indexed, key=lambda pair: pair[1]["time"]):
            cap.set(cv2.CAP_PROP_POS_MSEC, item["time"] * 1000)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode {video} at {item['time']:.2f}s")
            frame = resize_for_app(frame)
            lab, _, selected = balanced_frame_and_masks(frame, item["row"])
            x, y, radius = (int(float(item["row"][name]))
                            for name in ("circle_x", "circle_y", "circle_r"))
            main, satellite = tight_masks(lab.shape, (x, y, radius), item["row"], selected)
            # If a very small satellite is not reliably selected, retain a
            # neutral placeholder and expose its pixel count to the audit.
            main_summary = paired_summary(lab, main)
            satellite_summary = paired_summary(lab, satellite) if satellite.sum() >= 12 else None
            summaries[index] = {"main": main_summary, "satellite": satellite_summary,
                                "main_pixels": int(main.sum()),
                                "satellite_pixels": int(satellite.sum())}
        cap.release()
        print(f"tight relative: {video} ({len(indexed)} endpoints)")
    return summaries


def angle(summary):
    return float(np.arctan2(summary["circular"][0], summary["circular"][1]))


def median_summary(values, part):
    valid = [value[part] for value in values if value[part] is not None]
    return {name: np.median(np.asarray([value[name] for value in valid]), axis=0)
            for name in ("named", "circular", "chroma", "lightness")}


def relative(summary, baseline):
    if summary is None:
        # Missing-satellite flag is appended by the caller; zero deltas prevent
        # the model from inventing a colour response from missing pixels.
        return np.zeros(19, dtype=float)
    hue_shift = np.arctan2(np.sin(angle(summary) - angle(baseline)),
                           np.cos(angle(summary) - angle(baseline)))
    return np.concatenate([
        summary["named"], summary["named"] - baseline["named"],
        [hue_shift], summary["circular"] - baseline["circular"],
        summary["chroma"] - baseline["chroma"],
        summary["lightness"] - baseline["lightness"],
    ])


def build_tight_features(items, tight, original_matrices):
    groups = np.asarray([item["group"] for item in items])
    truth = np.asarray([item["stage"] for item in items])
    baselines = {}
    for group in sorted(set(groups)):
        values = [tight[index] for index in np.where((groups == group) & (truth == 25))[0]]
        baselines[group] = {"main": median_summary(values, "main")}
        satellites = [value for value in values if value["satellite"] is not None]
        baselines[group]["satellite"] = (median_summary(satellites, "satellite")
                                          if satellites else baselines[group]["main"])
    main_features, both_features, audit = [], [], []
    for item, value in zip(items, tight):
        base = baselines[item["group"]]
        main = relative(value["main"], base["main"])
        satellite = relative(value["satellite"], base["satellite"])
        present = float(value["satellite"] is not None)
        main_features.append(main)
        both_features.append(np.concatenate([main, satellite, [present]]))
        audit.append({"group": item["group"], "video": item["video"],
                      "time": item["time"], "reference": item["stage"],
                      "main_pixels": value["main_pixels"],
                      "satellite_pixels": value["satellite_pixels"],
                      "main_hue_degrees": np.degrees(angle(value["main"])),
                      "main_chroma_median": value["main"]["chroma"][1],
                      "main_lightness_median": value["main"]["lightness"][1]})
    return {
        "legacy_delta_lab": original_matrices["legacy_delta_lab"],
        "paired_named": original_matrices["paired_named"],
        "tight_main_relative": np.asarray(main_features),
        "tight_main_satellite_relative": np.asarray(both_features),
    }, audit


def plot(output, results, best):
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3), constrained_layout=True)
    for axis, name in zip(axes[:2], ("paired_named", best)):
        matrix = np.asarray(results[name]["confusion"], dtype=float)
        norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        axis.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for row in range(len(LEVELS)):
            for column in range(len(LEVELS)):
                value = norm[row, column]
                axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                          fontsize=7, color="white" if value > .55 else "black")
        axis.set_xticks(range(7), DISPLAY, rotation=35)
        axis.set_yticks(range(7), DISPLAY)
        axis.set(xlabel="Predicted RH", ylabel="Reference RH", title=name)
    names = list(results); x = np.arange(len(names)); width = .36
    axes[2].bar(x-width/2, [results[name]["exact_accuracy"] for name in names],
                width, label="Exact")
    axes[2].bar(x+width/2, [results[name]["balanced_accuracy"] for name in names],
                width, label="Balanced")
    axes[2].axhline(.85, color="crimson", linestyle="--", linewidth=1, label="0.85 target")
    axes[2].set_xticks(x, [name.replace("tight_", "") for name in names],
                       rotation=28, ha="right")
    axes[2].set_ylim(0, 1); axes[2].legend(fontsize=8)
    axes[2].set(title="Complete-run-held-out A/B", ylabel="Score")
    fig.suptitle("RH inner ROI + within-yellow relative colour", fontweight="bold")
    fig.savefig(output / "rh_tight_relative_validation.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_tight_relative_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    items = endpoint_rows(read_csv(args.cache))
    original = extract(items, args.video_root)
    original_matrices, _ = build_features(items, original)
    tight = extract_tight(items, args.video_root)
    matrices, audit = build_tight_features(items, tight, original_matrices)
    truth = np.asarray([item["stage"] for item in items])
    groups = np.asarray([item["group"] for item in items])
    results, prediction_rows = {}, []
    for name in FEATURE_NAMES:
        metrics, predictions, confidence, choices = tune_and_evaluate(
            matrices[name], truth, groups)
        metrics["outer_fold_C"] = choices; results[name] = metrics
        for item, prediction, score in zip(items, predictions, confidence):
            prediction_rows.append({"feature_set": name, "group": item["group"],
                                    "video": item["video"], "time": item["time"],
                                    "reference": item["stage"], "prediction": prediction,
                                    "confidence": score})
    candidates = ("tight_main_relative", "tight_main_satellite_relative")
    best = max(candidates, key=lambda name: (
        results[name]["balanced_accuracy"], results[name]["exact_accuracy"]))
    decision = deployment_decision(results["paired_named"], results[best])
    decision["selected_candidate"] = best
    decision["apply_to_app"] = bool(
        decision["improves_exact"] and decision["improves_balanced"]
        and decision["preserves_every_stage"] and decision["improves_middle_stage"]
        and decision["all_stage_recall_at_least_0.85"])
    decision["proceed_to_h2"] = decision["apply_to_app"]
    payload = {"scope": "30 valid RH-only rising endpoints; nested complete-run holdout",
               "results": results, "decision": decision, "n_endpoints": len(items)}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for filename, rows in (("predictions.csv", prediction_rows), ("roi_audit.csv", audit)):
        with (args.output / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plot(args.output, results, best)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
