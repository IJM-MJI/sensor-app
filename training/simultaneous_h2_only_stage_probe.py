"""Probe simultaneous H2 stages using the H2-only run-5 optical trajectory.

This is an audit candidate, not deployable ground truth.  Each video is
endpoint-normalized to reduce lighting/exposure differences, then its flame
progress is matched to the reviewed 0/1/2/3/4% anchors in H2-only run 5.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_models import CACHE_VERSION, read_csv, simultaneous_clips


FEATURES = ("flame_L", "flame_a", "flame_b")
REFERENCE_VIDEO = "1_90_H2_only_5.mp4"
REFERENCE_TIMES = {0: 5.0, 1: 8.0, 2: 13.0, 3: 21.0, 4: 130.0}
FRACTIONS = tuple(np.linspace(0.0, 1.0, 9))


def vector(row):
    return np.asarray([float(row[name]) for name in FEATURES])


def nearest(rows, seconds):
    return min(rows, key=lambda row: abs(float(row["time"]) - seconds))


def progress(values, low, high):
    axis = high - low
    return float(np.dot(values - low, axis) / max(float(np.dot(axis, axis)), 1e-9))


def endpoint(rows, seconds, side):
    ordered = sorted(rows, key=lambda row: abs(float(row["time"]) - seconds))[:5]
    values = np.asarray([vector(row) for row in ordered])
    # A median guards against one decoded boundary frame from the adjacent state.
    return np.median(values, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path,
                        default=Path(f"training/cache/{CACHE_VERSION}/features.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/simultaneous_stage_review/h2_only_stage_probe.json"))
    args = parser.parse_args()
    rows = read_csv(args.cache)

    reference = [row for row in rows if row["video"] == REFERENCE_VIDEO]
    ref_low = vector(nearest(reference, REFERENCE_TIMES[0]))
    ref_high = vector(nearest(reference, REFERENCE_TIMES[4]))
    reference_progress = {
        stage: progress(vector(nearest(reference, seconds)), ref_low, ref_high)
        for stage, seconds in REFERENCE_TIMES.items()
    }

    predictions = []
    for clip in simultaneous_clips():
        if clip.group != "sim-run-5" or clip.rh is None or clip.rh > 80:
            continue
        use = [row for row in rows if row["video"] == clip.name]
        start, end = float(clip.reaction_start), float(clip.reaction_end)
        low, high = endpoint(use, start, "low"), endpoint(use, end, "high")
        for fraction in FRACTIONS:
            seconds = start + fraction * (end - start)
            optical_progress = progress(vector(nearest(use, seconds)), low, high)
            stage = min(reference_progress,
                        key=lambda level: abs(optical_progress - reference_progress[level]))
            predictions.append({
                "rh_metadata": int(clip.rh), "video": clip.name,
                "time_s": round(seconds, 3), "search_fraction": round(float(fraction), 3),
                "endpoint_normalized_flame_progress": round(optical_progress, 4),
                "nearest_h2_only_stage_pct": stage,
            })

    report = {
        "status": "audit candidate; requires visual review",
        "reference_video": REFERENCE_VIDEO,
        "reference_times_s": REFERENCE_TIMES,
        "reference_endpoint_normalized_progress": reference_progress,
        "warning": (
            "The simultaneous endpoint contains both H2 and RH effects. "
            "A non-monotonic row or skipped stage must not be promoted to the app."
        ),
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader(); writer.writerows(predictions)
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))
    print(f"Wrote {len(predictions)} candidate predictions to {csv_path}")


if __name__ == "__main__":
    main()
