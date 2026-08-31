"""Test whether stable-window selection fixes Place-2 seven-band RH transfer.

The label order is fixed before looking at the held-out run.  For every range,
we compare the user-confirmed single-frame anchors with (a) a local median at
the same anchor and (b) the lowest-variance one-second window inside a broad,
predeclared time interval.  Each fold still excludes one complete video.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rh_40_50_cross_run_spatial_analysis import extract
from rh_place1_external_validation import control_vector, nearest_row
from rh_place2_seven_band_run_holdout import (
    CALIBRATION, LABELS, LEVELS, VIDEOS, fit_predict, matrices,
)
from train_models import CACHE_VERSION, read_csv


# Optical guide centres come from the user's later seven-band inspection.
# Boundaries are midpoints between centres, clipped to the usable response.
GUIDES = {
    "response3": (1.5, 4.5, 10.0, 20.0, 27.0, 31.5, 35.0),
    "response6": (9.0, 11.5, 14.5, 15.5, 17.5, 19.0, 23.0),
}
LIMITS = {"response3": (.75, 37.5), "response6": (7.5, 25.0)}
STEP = .25
RADIUS = .5


def intervals(group):
    centres = np.asarray(GUIDES[group], dtype=float)
    mids = (centres[:-1] + centres[1:]) / 2
    starts = np.r_[LIMITS[group][0], mids]
    ends = np.r_[mids, LIMITS[group][1]]
    return list(zip(LEVELS, starts, ends, centres))


def rolling(rows, times, radius=RADIUS):
    """Return robust local medians and a LAB-MAD stability score."""
    values = np.asarray(rows, dtype=float)
    medians, spread, support = [], [], []
    for time in times:
        chosen = np.abs(times - time) <= radius + 1e-9
        window = values[chosen]
        median = np.median(window, axis=0)
        medians.append(median)
        spread.append(float(np.sqrt(np.mean(np.median(
            np.abs(window - median), axis=0) ** 2))))
        support.append(int(chosen.sum()))
    return np.asarray(medians), np.asarray(spread), np.asarray(support)


def stable_indices(group, times, spread):
    """Choose a stable point per ordered interval without using class colour."""
    selected = []
    for _, start, end, centre in intervals(group):
        possible = np.where((times >= start) & (times <= end))[0]
        if not len(possible):
            raise RuntimeError(f"No candidates for {group} {start:.2f}-{end:.2f}")
        local = spread[possible]
        robust_scale = max(float(np.median(local)), .05)
        half_width = max((end - start) / 2, .25)
        score = local / robust_scale + .15 * np.abs(times[possible] - centre) / half_width
        selected.append(int(possible[np.argmin(score)]))
    return selected


def evaluate(method_rows):
    x = np.asarray([row["vector"] for row in method_rows], dtype=float)
    truth = np.asarray([row["reference"] for row in method_rows], dtype=int)
    groups = np.asarray([row["group"] for row in method_rows])
    predictions = np.zeros_like(truth)
    folds = []
    for held in sorted(set(groups)):
        train = groups != held
        test = ~train
        predictions[test] = fit_predict(
            "standardized_1nn", x[train], truth[train], x[test])
        folds.append({"held_out_run": held,
                      **matrices(truth[test], predictions[test])})
    return {**matrices(truth, predictions), "folds": folds}, predictions


def draw(results, output):
    methods = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, method in zip(axes[:2], ("anchor_window", "stable_window")):
        matrix = np.asarray(results[method]["row_normalized_confusion_0_to_1"])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        for row in range(7):
            for column in range(7):
                value = matrix[row, column]
                axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                          fontsize=7, color="white" if value >= .65 else "#1f2937")
        axis.set_xticks(range(7), LABELS, rotation=35, ha="right")
        axis.set_yticks(range(7), LABELS)
        axis.set(xlabel="Predicted", ylabel="Reference",
                 title=method.replace("_", " "))
    x = np.arange(len(methods)); width = .35
    axes[2].bar(x-width/2, [results[m]["exact_accuracy"] for m in methods],
                width, label="Exact")
    axes[2].bar(x+width/2,
                [results[m]["within_one_adjacent_range"] for m in methods],
                width, label="Within one")
    axes[2].axhline(.85, color="crimson", linestyle="--", label="0.85 target")
    axes[2].set_xticks(x, [m.replace("_", "\n") for m in methods])
    axes[2].set_ylim(0, 1.03); axes[2].legend()
    axes[2].set(ylabel="Score (0-1)", title="Complete-run held-out")
    fig.suptitle("Place-2 RH: effect of stable-window interval selection",
                 fontweight="bold")
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"rh_place2_stable_window_alignment.{suffix}", **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_place2_stable_window_alignment_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cached = read_csv(args.cache)

    items = []
    dense_times = {}
    for group, video in VIDEOS.items():
        times = np.arange(LIMITS[group][0], LIMITS[group][1] + STEP / 2, STEP)
        dense_times[group] = times
        all_times = np.unique(np.r_[CALIBRATION[group], times])
        for time in all_times:
            items.append({"group": group, "video": video, "time": float(time),
                          "row": nearest_row(cached, video, float(time)),
                          "calibration": abs(time - CALIBRATION[group]) < 1e-9})
    summaries = extract(items, args.video_root)
    controls = [control_vector(summary) for summary in summaries]
    baseline = {item["group"]: control for item, control in zip(items, controls)
                if item["calibration"]}
    dense = {}
    for group in VIDEOS:
        pairs = [(item["time"], control - baseline[group])
                 for item, control in zip(items, controls)
                 if item["group"] == group and not item["calibration"]]
        pairs.sort()
        times = np.asarray([pair[0] for pair in pairs])
        vectors = np.asarray([pair[1] for pair in pairs])
        median, spread, support = rolling(vectors, times)
        dense[group] = {"times": times, "raw": vectors, "median": median,
                        "spread": spread, "support": support}

    methods = {"single_anchor": [], "anchor_window": [], "stable_window": []}
    selection_rows = []
    for group in VIDEOS:
        data = dense[group]
        stable = stable_indices(group, data["times"], data["spread"])
        for band_index, (level, start, end, guide) in enumerate(intervals(group)):
            anchor_index = int(np.argmin(np.abs(data["times"] - guide)))
            stable_index = stable[band_index]
            common = {"group": group, "reference": int(level)}
            methods["single_anchor"].append(
                {**common, "time_s": float(data["times"][anchor_index]),
                 "vector": data["raw"][anchor_index]})
            methods["anchor_window"].append(
                {**common, "time_s": float(data["times"][anchor_index]),
                 "vector": data["median"][anchor_index]})
            methods["stable_window"].append(
                {**common, "time_s": float(data["times"][stable_index]),
                 "vector": data["median"][stable_index]})
            selection_rows.append({
                "group": group, "range": LABELS[band_index],
                "interval_start_s": float(start), "interval_end_s": float(end),
                "guide_s": float(guide),
                "selected_s": float(data["times"][stable_index]),
                "window_mad_lab": float(data["spread"][stable_index]),
                "support": int(data["support"][stable_index]),
                "delta_L": float(data["median"][stable_index, 0]),
                "delta_a": float(data["median"][stable_index, 1]),
                "delta_b": float(data["median"][stable_index, 2]),
            })

    results, prediction_rows = {}, []
    for method, rows in methods.items():
        result, prediction = evaluate(rows)
        results[method] = result
        for row, predicted in zip(rows, prediction):
            prediction_rows.append({
                "method": method, "group": row["group"],
                "time_s": row["time_s"], "reference": row["reference"],
                "prediction": int(predicted),
            })
    best = max(results, key=lambda name: (
        results[name]["exact_accuracy"], results[name]["balanced_accuracy"]))
    deploy = bool(results[best]["exact_accuracy"] >= .85
                  and min(results[best]["per_range_recall"].values()) >= .85
                  and all(fold["exact_accuracy"] >= .85
                          for fold in results[best]["folds"]))
    payload = {
        "scope": "Place-2 response3-response6 complete-run holdout",
        "window": {"sample_step_s": STEP, "median_radius_s": RADIUS,
                   "stable_selection": "minimum local LAB MAD plus 0.15 guide-distance penalty"},
        "results": results,
        "decision": {"best_method": best, "deploy_to_app": deploy,
                     "rule": "overall exact, every recall, and each fold >= 0.85"},
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for filename, rows in (("selected_windows.csv", selection_rows),
                           ("predictions.csv", prediction_rows)):
        with (args.output / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    draw(results, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
