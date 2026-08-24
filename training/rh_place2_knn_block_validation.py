"""Validate the place-2 1-NN candidate on late-ramp endpoint blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rh_40_50_cross_run_spatial_analysis import build, extract
from rh_four_band_analysis import STAGES, band, score
from rh_place2_time_warp_analysis import RUNS, shifted_endpoint_items
from train_models import CACHE_VERSION, read_csv


RADII = (0.0, 0.27, 0.53)


def neighbourhood_items(cache, endpoint, radius):
    # Supplied times are the instant the target is reached.  Frames after an
    # endpoint already belong to the next ramp, so only sample the late side.
    offsets = (0.0,) if radius == 0 else np.linspace(-radius, 0.0, 5)
    rows = [row for row in cache if row.get("video") == endpoint["video"]]
    output, seen = [], set()
    for offset in offsets:
        requested = max(0.0, endpoint["requested_time"] + float(offset))
        row = min(rows, key=lambda value: abs(float(value["time"]) - requested))
        key = (round(float(row["time"]), 6), endpoint["stage"])
        if key in seen:
            continue
        seen.add(key)
        output.append({**endpoint, "time": float(row["time"]),
                       "requested_time": requested, "row": row})
    return output


def evaluate_radius(cache, video_root, endpoints, radius, train_radius=0.0):
    predictions, truth_all = [], []
    folds = []
    for held in RUNS:
        train_endpoints = [item for item in endpoints if item["group"] != held]
        train_items = [candidate for endpoint in train_endpoints
                       for candidate in neighbourhood_items(cache, endpoint, train_radius)]
        held_endpoints = [item for item in endpoints if item["group"] == held]
        test_items = [candidate for endpoint in held_endpoints
                      for candidate in neighbourhood_items(cache, endpoint, radius)]
        items = train_items + test_items
        summaries = extract(items, video_root)
        matrices, audit = build(items, summaries, STAGES)
        groups = np.asarray([row["group"] for row in audit])
        truth = np.asarray([band(row["reference"]) for row in audit])
        train, test = groups != held, groups == held
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
        model.fit(matrices["background_control"][train], truth[train])
        prediction = model.predict(matrices["background_control"][test])
        predictions.extend(prediction.tolist()); truth_all.extend(truth[test].tolist())
        folds.append({"held": held, "n_test": int(np.sum(test)),
                      **score(truth[test], prediction)})
    return {**score(np.asarray(truth_all), np.asarray(predictions)), "folds": folds}


def plot(output, results):
    labels = list(results); x = np.arange(len(labels)); width = .36
    fig, axis = plt.subplots(figsize=(11.8, 5.0), constrained_layout=True)
    axis.bar(x-width/2, [results[label]["exact_accuracy"] for label in labels],
             width, label="Exact")
    axis.bar(x+width/2, [min(results[label]["per_band_recall"]) for label in labels],
             width, label="Minimum band recall")
    axis.axhline(.85, color="crimson", linestyle="--", linewidth=1)
    axis.set_xticks(x, [label.replace("_", " ") for label in labels],
                    rotation=18, ha="right")
    axis.set_ylim(0, 1); axis.grid(axis="y", alpha=.2)
    axis.set(title="Place-2 1-NN late-ramp endpoint robustness", ylabel="Score")
    axis.legend()
    fig.savefig(output / "rh_place2_knn_block_validation.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/rh_place2_knn_blocks_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cache = read_csv(args.cache)
    endpoints = shifted_endpoint_items(cache, {("rh-response-6", 70.0): 16.67})
    results = {("endpoint" if radius == 0 else f"last_{radius:.2f}s"): evaluate_radius(
        cache, args.video_root, endpoints, radius) for radius in RADII}
    augmented = {
        f"train_last_{radius:.2f}s_test_last_0.27s": evaluate_radius(
            cache, args.video_root, endpoints, 0.27, train_radius=radius)
        for radius in (0.27, 0.53)
    }
    all_candidates = {**results, **augmented}
    selected_name = max(augmented, key=lambda name: (
        min(augmented[name]["per_band_recall"]),
        augmented[name]["balanced_accuracy"]))
    selected = augmented[selected_name]
    decision = {
        "selected": selected_name,
        "passes_selected_0_85": bool(selected["exact_accuracy"] >= .85
                                      and selected["balanced_accuracy"] >= .85
                                      and min(selected["per_band_recall"]) >= .85),
        "app_deploy": False,
        "reason": "Late-ramp robustness is still based on only two source runs.",
    }
    payload = {"model": "background-control StandardScaler + 1-NN",
               "response6_70_time_seconds": 16.67,
               "results": results, "late_ramp_augmented_training": augmented,
               "decision": decision}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(args.output, all_candidates)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
