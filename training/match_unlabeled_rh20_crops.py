"""Create timeline-free H2 pseudo-label candidates for cropped RH20 runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cross_run_atlas_analysis import (
    build_h2_atlas, cost_matrix, monotonic_match, nearest_stage, trajectory_feature,
)
from ordinal_concentration_analysis import assign_h2_ramp_targets
from train_models import CACHE_VERSION, read_csv


def smooth(values, width=9):
    width = min(width, len(values) // 2 * 2 + 1)
    if width < 3:
        return values
    half = width // 2
    padded = np.pad(values, (half, half), mode="edge")
    median = np.asarray([np.median(padded[index:index + width])
                         for index in range(len(values))])
    return np.convolve(np.pad(median, (half, half), mode="edge"),
                       np.ones(width) / width, mode="valid")


def automatic_response_segment(features, atlas_features):
    """Use only the known Initial -> Reaction -> Recovery phase order."""
    window = min(9, max(3, len(features) // 12))
    initial = np.median(features[:window], axis=0)
    recovery = np.median(features[-window:], axis=0)
    scale = np.maximum(np.percentile(abs(atlas_features - atlas_features[0]), 75, axis=0), .2)
    from_initial = np.sqrt(np.mean(((features - initial) / scale) ** 2, axis=1))
    from_recovery = np.sqrt(np.mean(((features - recovery) / scale) ** 2, axis=1))
    # A recovery endpoint may not equal the initial colour exactly. Distance to
    # the nearer endpoint baseline still isolates the intervening response.
    strength = np.minimum(from_initial, from_recovery)
    filtered = smooth(strength)
    peak = int(np.argmax(filtered))
    baseline_values = np.r_[filtered[:window], filtered[-window:]]
    floor = float(np.percentile(baseline_values, 90))
    threshold = floor + .20 * max(float(filtered[peak]) - floor, 0)
    before = np.where(filtered[:peak + 1] <= threshold)[0]
    start = int(before[-1] + 1) if len(before) else 0
    after = np.where(filtered[peak:] <= threshold)[0]
    end = int(peak + after[0]) if len(after) else len(features) - 1
    if end - start < 6:
        start, end = max(0, peak - 3), min(len(features) - 1, peak + 3)
    return start, end, peak, strength, filtered, threshold


def stage_confidence(costs, atlas_levels, selected_indices):
    stages = np.asarray([0., 1., 2., 3., 4.])
    atlas_stages = nearest_stage("H2", atlas_levels)
    confidence, mutual = [], []
    nearest_candidate = np.argmin(costs, axis=0)
    for index, (selected, row_cost) in enumerate(zip(selected_indices, costs)):
        per_stage = np.asarray([np.min(row_cost[atlas_stages == stage]) for stage in stages])
        order = np.argsort(per_stage)
        confidence.append(float(np.clip(
            (per_stage[order[1]] - per_stage[order[0]]) / max(per_stage[order[1]], 1e-9), 0, 1)))
        mutual.append(bool(nearest_candidate[selected] == index))
    return np.asarray(confidence), np.asarray(mutual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-cache", type=Path,
                        default=Path(f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--candidate-cache", type=Path,
                        default=Path(f"training/cache/{CACHE_VERSION}/rh20_cropped_unlabeled_v1.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/rh20_cropped_atlas_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    reference_rows = read_csv(args.reference_cache); assign_h2_ramp_targets(reference_rows)
    atlas_levels, atlas_features, _ = build_h2_atlas(reference_rows)
    candidates = read_csv(args.candidate_cache)
    by_group = defaultdict(list)
    for row in candidates:
        by_group[str(row["group"])].append(row)

    output_rows, report = [], {"reference": "h2-test-2", "candidate_timeline_used": False,
                               "groups": {}}
    fig, axes = plt.subplots(len(by_group), 2, figsize=(12, 2.8 * len(by_group)),
                             constrained_layout=True, squeeze=False)
    for axis_row, (group, rows) in enumerate(sorted(by_group.items())):
        rows.sort(key=lambda row: float(row["time"]))
        features = np.asarray([trajectory_feature(row, "H2") for row in rows])
        start, end, peak, strength, filtered, threshold = automatic_response_segment(
            features, atlas_features)
        rising_rows = rows[start:end + 1]; rising_features = features[start:end + 1]
        costs = cost_matrix(rising_features, atlas_features)
        indices = monotonic_match(costs, "up")
        confidence, mutual = stage_confidence(costs, atlas_levels, indices)
        matched = atlas_levels[indices]
        cost_values = costs[np.arange(len(indices)), indices]
        cost_limit = float(np.median(cost_values))
        eligible = mutual & (confidence >= .10) & (cost_values <= cost_limit)
        stages = nearest_stage("H2", matched)
        for row, optical, stage, cost, conf, is_mutual, use in zip(
                rising_rows, matched, stages, cost_values, confidence, mutual, eligible):
            output_rows.append({
                "video": row["video"], "group": group, "time": float(row["time"]),
                "optical_equivalent_h2": float(optical), "pseudo_stage": float(stage),
                "match_cost": float(cost), "stage_confidence": float(conf),
                "mutual_nearest": bool(is_mutual), "training_candidate": bool(use),
                "reaction_start_time_detected": float(rows[start]["time"]),
                "reaction_end_time_detected": float(rows[end]["time"]),
            })
        report["groups"][group] = {
            "video": str(rows[0]["video"]), "n_total_frames": len(rows),
            "n_rising_frames": len(rising_rows),
            "reaction_start_time_detected": float(rows[start]["time"]),
            "reaction_end_time_detected": float(rows[end]["time"]),
            "optical_max_h2": float(np.max(matched)),
            "pseudo_stage_max": float(np.max(stages)),
            "training_candidates": int(np.sum(eligible)),
            "training_candidate_coverage": float(np.mean(eligible)),
        }
        times = np.asarray([float(row["time"]) for row in rows])
        axes[axis_row, 0].plot(times, strength, alpha=.4, label="raw change")
        axes[axis_row, 0].plot(times, filtered, label="smoothed change")
        axes[axis_row, 0].axhline(threshold, color="gray", linestyle=":", label="phase threshold")
        axes[axis_row, 0].axvspan(float(rows[start]["time"]), float(rows[end]["time"]),
                                 color="crimson", alpha=.10, label="auto Reaction")
        axes[axis_row, 0].set_title(group); axes[axis_row, 0].set_ylabel("Flame change")
        axes[axis_row, 0].legend(fontsize=7, frameon=False)
        axes[axis_row, 1].plot([float(row["time"]) for row in rising_rows], matched,
                               color="#4C78A8", label="test_2 equivalent")
        axes[axis_row, 1].scatter(
            np.asarray([float(row["time"]) for row in rising_rows])[eligible], matched[eligible],
            color="#F58518", s=18, label="training candidate")
        axes[axis_row, 1].set_ylim(-.1, 4.1); axes[axis_row, 1].set_ylabel("H2 (%)")
        axes[axis_row, 1].legend(fontsize=7, frameon=False)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    fig.suptitle("Timeline-free RH20 crop matching to H2 test_2", weight="bold")
    fig.savefig(args.output / "rh20_cropped_atlas_review.png", dpi=300)
    plt.close(fig)

    with (args.output / "matches.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0])); writer.writeheader()
        writer.writerows(output_rows)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
