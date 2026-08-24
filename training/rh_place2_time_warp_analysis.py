"""Test whether place-2 run differences are consistent temporal lag/advance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rh_40_50_cross_run_spatial_analysis import build, extract
from rh_four_band_analysis import BANDS, DISPLAY, STAGES, band
from rh_paired_pixel_hue_analysis import endpoint_rows
from rh_place2_pairwise_band_analysis import FIXED_C
from rh_place2_pairwise_band_analysis import evaluate as pairwise_evaluate
from train_models import CACHE_VERSION, read_csv


RUNS = {
    "rh-response-3": {
        "video": "1_90_H2O_only_3(response).mp4", "end": 38.0,
        "other": "rh-response-6",
        "segments": ((0,2,20),(2,3,30),(3,5,40),(5,7,50),
                     (7,11,60),(11,25,70),(25,28,80),(28,38,90)),
    },
    "rh-response-6": {
        "video": "1_90_H2O_only_6(response).mp4", "end": 32.0,
        "other": "rh-response-3",
        "segments": ((0,7,20),(7,10,30),(10,13,40),(13,14,50),
                     (14,16,60),(16,18,70),(18,20,80),(20,32,90)),
    },
}


def nominal(segments, seconds):
    previous = float(segments[0][2])
    for start, end, target in segments:
        if start <= seconds <= end:
            fraction = (seconds - start) / max(end - start, 1e-9)
            return previous + fraction * (target - previous)
        previous = float(target)
    return previous


def full_run_items(cache, endpoints, group):
    info = RUNS[group]
    training = [item for item in endpoints if item["group"] == info["other"]]
    baseline = [item for item in endpoints if item["group"] == group and item["stage"] == 25]
    rows = [row for row in cache if row.get("video") == info["video"]
            and 0 <= float(row["time"]) <= info["end"]]
    candidates = [{"video": info["video"], "group": group,
                   "time": float(row["time"]), "requested_time": float(row["time"]),
                   "stage": 40.0, "row": row, "candidate": True} for row in rows]
    return training + baseline + candidates, {round(item["time"], 6) for item in candidates}


def smooth(probabilities, radius=2):
    result = np.zeros_like(probabilities)
    for index in range(len(probabilities)):
        lo, hi = max(0, index-radius), min(len(probabilities), index+radius+1)
        result[index] = np.mean(probabilities[lo:hi], axis=0)
    return result


def crossings(rows):
    output = []
    predicted = [row["predicted_band"] for row in rows]
    for target, nominal_boundary in ((45.0, 35.0), (65.0, 55.0), (85.0, 75.0)):
        found = None
        for index in range(len(rows)-2):
            if all(value >= target for value in predicted[index:index+3]):
                found = rows[index]
                break
        output.append({"target_band": target, "nominal_boundary_rh": nominal_boundary,
                       "crossing_time": None if found is None else found["time"],
                       "nominal_rh_at_crossing": None if found is None else found["nominal_rh"],
                       "rh_shift_at_crossing": None if found is None else
                           found["nominal_rh"] - nominal_boundary})
    return output


def analyse_run(cache, endpoints, video_root, group):
    items, candidate_times = full_run_items(cache, endpoints, group)
    summaries = extract(items, video_root)
    matrices, audit = build(items, summaries, STAGES)
    groups = np.asarray([row["group"] for row in audit])
    truth = np.asarray([band(row["reference"]) for row in audit])
    train = groups == RUNS[group]["other"]
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=FIXED_C, class_weight="balanced", max_iter=5000, random_state=42))
    model.fit(matrices["background_control"][train], truth[train])
    # Baseline endpoint rows can share a timestamp with trajectory candidates.
    # Keep one feature vector per timestamp so smoothing is not biased there.
    held_by_time = {round(row["time"], 6): index for index, row in enumerate(audit)
                    if row["group"] == group and round(row["time"], 6) in candidate_times}
    held_indices = [held_by_time[key] for key in sorted(held_by_time)]
    probabilities = model.predict_proba(matrices["background_control"][held_indices])
    probabilities = smooth(probabilities)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    rows = []
    for index, prediction, probability in zip(held_indices, predictions, probabilities):
        row = audit[index]; seconds = float(row["time"])
        rows.append({"group": group, "video": RUNS[group]["video"], "time": seconds,
                     "nominal_rh": nominal(RUNS[group]["segments"], seconds),
                     "predicted_band": float(prediction),
                     "confidence": float(np.max(probability)),
                     "drop_minus_bg_L": row["drop_minus_bg_L"],
                     "drop_minus_bg_a": row["drop_minus_bg_a"],
                     "drop_minus_bg_b": row["drop_minus_bg_b"]})
    return rows, crossings(rows)


def plot(output, all_rows):
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.0), constrained_layout=True)
    for axis, group in zip(axes, RUNS):
        rows = [row for row in all_rows if row["group"] == group]
        times = [row["time"] for row in rows]
        axis.plot(times, [row["nominal_rh"] for row in rows], label="Nominal RH ramp",
                  linewidth=2)
        axis.step(times, [row["predicted_band"] for row in rows], where="mid",
                  label="Other-run optical band", linewidth=1.6)
        axis.set_yticks((20,25,40,45,60,65,80,85,90))
        axis.set_ylim(15,95); axis.grid(alpha=.2)
        axis.set(title=group, xlabel="Video time (s)", ylabel="RH / optical band")
        axis.legend()
    fig.suptitle("Place-2 full-trajectory temporal alignment audit", fontweight="bold")
    fig.savefig(output / "rh_place2_time_warp_audit.png", dpi=220)
    plt.close(fig)


def shifted_endpoint_items(cache, shifts):
    """Return endpoint items with explicitly stated run-local sensitivity shifts."""
    items = [item for item in endpoint_rows(cache)
             if item["group"] in RUNS]
    for item in items:
        key = (item["group"], float(item["stage"]))
        if key not in shifts:
            continue
        requested = float(shifts[key])
        candidates = [row for row in cache if row.get("video") == item["video"]]
        row = min(candidates, key=lambda value: abs(float(value["time"]) - requested))
        item.update(time=float(row["time"]), requested_time=requested, row=row)
    return items


def timeline_sensitivity(cache, video_root):
    # These are diagnostic, pre-specified shifts derived from the full-trajectory
    # crossing audit.  They are not independent validation and cannot be deployed.
    variants = {
        "original": {},
        "response3_40_later": {("rh-response-3", 40.0): 6.13},
        "response6_70_earlier": {("rh-response-6", 70.0): 16.67},
        "both_shifts": {
            ("rh-response-3", 40.0): 6.13,
            ("rh-response-6", 70.0): 16.67,
        },
    }
    output = {}
    for name, shifts in variants.items():
        items = shifted_endpoint_items(cache, shifts)
        summaries = extract(items, video_root)
        matrices, audit = build(items, summaries, STAGES)
        truth = np.asarray([band(row["reference"]) for row in audit])
        groups = np.asarray([row["group"] for row in audit])
        _, metrics = pairwise_evaluate(matrices["background_control"], truth, groups)
        output[name] = {"shifts_seconds": [
            {"group": group, "stage": stage, "new_time": seconds}
            for (group, stage), seconds in shifts.items()], **metrics}
    return output


def plot_sensitivity(output, variants):
    names = list(variants)
    exact = [variants[name]["exact_accuracy"] for name in names]
    minimum = [min(variants[name]["per_band_recall"]) for name in names]
    x = np.arange(len(names)); width = .36
    fig, axis = plt.subplots(figsize=(9.0, 4.4), constrained_layout=True)
    axis.bar(x-width/2, exact, width, label="Exact accuracy")
    axis.bar(x+width/2, minimum, width, label="Minimum band recall")
    axis.axhline(.85, color="crimson", linestyle="--", linewidth=1, label="0.85 target")
    axis.set_xticks(x, [name.replace("_", "\n") for name in names])
    axis.set_ylim(0, 1); axis.grid(axis="y", alpha=.2); axis.legend()
    axis.set(title="Run-local timeline-shift sensitivity (not independent validation)",
             ylabel="Score")
    fig.savefig(output / "rh_place2_timeline_sensitivity.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/rh_place2_time_warp_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cache = read_csv(args.cache); endpoints = endpoint_rows(cache)
    all_rows, result = [], {}
    for group in RUNS:
        rows, found = analyse_run(cache, endpoints, args.video_root, group)
        all_rows.extend(rows); result[group] = found
    shifts = [item["rh_shift_at_crossing"] for values in result.values()
              for item in values if item["rh_shift_at_crossing"] is not None]
    decision = {"consistent_global_time_warp": False,
                "reason": "A time warp requires all ordered boundaries in each run and similar signed shifts.",
                "observed_shift_range_percent_rh": None if not shifts else
                    [float(min(shifts)), float(max(shifts))]}
    # Require every boundary and a within-run shift spread no larger than 5%RH.
    complete = all(all(item["rh_shift_at_crossing"] is not None for item in values)
                   for values in result.values())
    stable = complete and all(
        max(item["rh_shift_at_crossing"] for item in values)
        - min(item["rh_shift_at_crossing"] for item in values) <= 5
        for values in result.values())
    decision["consistent_global_time_warp"] = bool(stable)
    sensitivity = timeline_sensitivity(cache, args.video_root)
    payload = {"scope": "full response3/response6 trajectories predicted by the opposite run",
               "crossings": result, "decision": decision,
               "timeline_shift_sensitivity": sensitivity}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.output / "trajectory.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    plot(args.output, all_rows)
    plot_sensitivity(args.output, sensitivity)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
