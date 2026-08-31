"""Audit predeclared frame-time policies under complete-run RH holdout."""

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
from rh_place2_seven_band_run_holdout import (
    CALIBRATION, LABELS, LEVELS, POINTS, VIDEOS, fit_predict, matrices,
)
from train_models import CACHE_VERSION, read_csv


CHOICES = {
    "earliest": lambda values: values[0],
    "middle": lambda values: values[len(values) // 2],
    "latest": lambda values: values[-1],
}
BURSTS = {
    "single": (0,),
    "trailing3": (-2, -1, 0),
    "centered3": (-1, 0, 1),
}


def fps_for(video_root: Path, video: str) -> float:
    cap = cv2.VideoCapture(str(source_path(video_root, video)))
    fps = float(cap.get(cv2.CAP_PROP_FPS)); cap.release()
    if not np.isfinite(fps) or fps < 10:
        raise RuntimeError(f"Invalid FPS for {video}: {fps}")
    return fps


def draw(results, best, output):
    values = np.asarray(results[best]["row_normalized_confusion_0_to_1"])
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)
    axes[0].imshow(values, cmap="Blues", vmin=0, vmax=1)
    for row in range(7):
        for column in range(7):
            value = values[row, column]
            axes[0].text(column, row, f"{value:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if value >= .65 else "#1f2937")
    axes[0].set_xticks(range(7), LABELS, rotation=35, ha="right")
    axes[0].set_yticks(range(7), LABELS)
    axes[0].set(xlabel="Predicted", ylabel="Reference",
                title=f"Best policy: {best}\nrow-normalized confusion (0–1)")
    names = list(results); x = np.arange(len(names))
    axes[1].bar(x, [results[name]["exact_accuracy"] for name in names],
                color="#3b82f6", label="Exact")
    axes[1].plot(x, [results[name]["within_one_adjacent_range"] for name in names],
                 color="#16a34a", marker="o", label="Within one")
    axes[1].axhline(.85, color="crimson", linestyle="--", label="0.85 target")
    axes[1].set_xticks(x, [name.replace("_", "\n") for name in names], fontsize=7)
    axes[1].set_ylim(0, 1.03); axes[1].legend()
    axes[1].set(ylabel="Score (0–1)", title="Predeclared timing policies")
    fig.suptitle("Place-2 seven-band RH timing-policy audit", fontweight="bold")
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"rh_place2_timing_policy_audit.{suffix}", **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_place2_timing_policy_audit_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cache = read_csv(args.cache)
    fps = {group: fps_for(args.video_root, video) for group, video in VIDEOS.items()}

    requests = {}
    for group, video in VIDEOS.items():
        requests[(group, "calibration", 0, 0)] = CALIBRATION[group]
        for choice, selector in CHOICES.items():
            for level, anchors in POINTS[group].items():
                centre = float(selector(anchors))
                for burst, offsets in BURSTS.items():
                    for offset in offsets:
                        requests[(group, choice, level, (burst, offset))] = (
                            centre + offset / fps[group])

    unique = {}
    items = []
    for key, seconds in requests.items():
        group = key[0]; video = VIDEOS[group]
        lookup = (group, round(seconds, 6))
        if lookup not in unique:
            unique[lookup] = len(items)
            items.append({"group": group, "video": video, "time": seconds,
                          "row": nearest_row(cache, video, seconds)})
    summaries = extract(items, args.video_root)
    controls = [control_vector(summary) for summary in summaries]
    by_time = {(item["group"], round(item["time"], 6)): value
               for item, value in zip(items, controls)}
    baseline = {group: by_time[(group, round(CALIBRATION[group], 6))]
                for group in VIDEOS}

    results, rows = {}, []
    for choice, selector in CHOICES.items():
        for burst, offsets in BURSTS.items():
            name = f"{choice}_{burst}"
            features, truth, groups, details = [], [], [], []
            for group in VIDEOS:
                for level, anchors in POINTS[group].items():
                    centre = float(selector(anchors))
                    vectors = [by_time[(group, round(centre + offset / fps[group], 6))]
                               - baseline[group] for offset in offsets]
                    feature = np.median(np.asarray(vectors), axis=0)
                    features.append(feature); truth.append(level); groups.append(group)
                    details.append((group, centre, level))
            x = np.asarray(features); y = np.asarray(truth); group_values = np.asarray(groups)
            prediction = np.zeros_like(y)
            fold_metrics = []
            for held in sorted(set(groups)):
                train = group_values != held; test = ~train
                prediction[test] = fit_predict("standardized_1nn", x[train], y[train], x[test])
                fold_metrics.append({"held_out_run": held, **matrices(y[test], prediction[test])})
            result = {**matrices(y, prediction), "folds": fold_metrics}
            results[name] = result
            for (group, seconds, level), predicted in zip(details, prediction):
                rows.append({"policy": name, "group": group, "time_s": seconds,
                             "reference": level, "prediction": int(predicted),
                             "correct": level == predicted})

    best = max(results, key=lambda name: (
        results[name]["exact_accuracy"], results[name]["balanced_accuracy"]))
    chosen = results[best]
    passes = bool(chosen["exact_accuracy"] >= .85
                  and chosen["balanced_accuracy"] >= .85
                  and min(chosen["per_range_recall"].values()) >= .85
                  and all(fold["exact_accuracy"] >= .85 for fold in chosen["folds"]))
    payload = {
        "scope": "one predeclared time per RH band; response3/response6 complete-run holdout",
        "policies": results,
        "decision": {
            "best_policy": best,
            "passes_all_0_85": passes,
            "deploy": False,
            "rule": "exact, balanced, every range recall, and each held-out run exact >= 0.85",
            "warning": "Only two runs; policies are a timing sensitivity audit, not an independent model-selection set.",
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    draw(results, best, args.output)
    print(json.dumps(payload["decision"], indent=2, ensure_ascii=False))
    for name, result in results.items():
        print(name, f"exact={result['exact_accuracy']:.3f}",
              f"balanced={result['balanced_accuracy']:.3f}",
              f"within1={result['within_one_adjacent_range']:.3f}")


if __name__ == "__main__":
    main()
