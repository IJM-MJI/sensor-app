"""Estimate per-video H2 response lag and shift nominal concentration labels.

The nominal H2 sequence comes from the supplied experiment timelines.  At a
camera time t the colour responds to gas introduced earlier, so the corrected
target is nominal_h2(t - lag).  Lags are selected against an out-of-video colour
proxy: the model producing that proxy never sees the video whose lag it scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

from train_models import H2_FEATURES, feature_value, read_legacy_continuous


# Exact nominal H2 steps supplied with the experiment timelines.  Values apply
# from the listed second until the next boundary.  The final zero in runs 4 and
# 5 is the start of Recovery, not a gradual 4 -> 0 concentration ramp.
H2_STEP_TIMELINES: dict[str, list[tuple[float, float]]] = {
    "1_90_H2_only_test.mp4": [(0, 1), (15, 2), (25, 3), (30, 4)],
    "1_90_H2_only_test_2.mp4": [(0, 0), (4, 1), (13, 2), (21, 3), (30, 4)],
    "1_90_H2_only_test_3.MOV": [(0, 0), (3, 1), (10, 2), (20, 3), (28, 4)],
    "1_90_H2_only_4.mp4": [(0, 0), (5, 1), (13, 2), (30, 3), (109, 4), (122, 0)],
    "1_90_H2_only_5.mp4": [(0, 0), (5, 1), (8, 2), (13, 3), (21, 4), (130, 0)],
}
RECOVERY_START = {
    "1_90_H2_only_4.mp4": 122.0,
    "1_90_H2_only_5.mp4": 130.0,
}


def apply_supplied_nominal_timeline(row: dict[str, object]) -> None:
    """Replace legacy interpolation with the supplied nominal experiment state."""
    video = str(row["video"])
    seconds = float(row["time"])
    steps = H2_STEP_TIMELINES.get(video)
    if steps:
        value = steps[0][1]
        for start, candidate in steps:
            if seconds < start:
                break
            value = candidate
        row["h2_value"] = float(value)
        row["phase"] = "recovery" if seconds >= RECOVERY_START.get(video, float("inf")) else "reaction"
    elif str(row.get("phase")) == "recovery":
        # Recovery was RH 20%, H2 0%; it was not a linear concentration ramp.
        row["h2_value"] = 0.0


def step_value_at(video: str, seconds: float) -> float:
    """Return the piecewise-constant nominal value, including pre-run H2 0%."""
    if seconds < 0:
        return 0.0
    value = 0.0
    for start, candidate in H2_STEP_TIMELINES[video]:
        if seconds < start:
            break
        value = candidate
    return float(value)


def regression_model() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=4,
        random_state=42, n_jobs=-1,
    )


def phase_runs(rows: list[dict[str, object]]) -> list[list[int]]:
    """Split repeated reaction/recovery cycles into monotonic contiguous runs."""
    order = sorted(range(len(rows)), key=lambda i: float(rows[i]["time"]))
    times = np.asarray([float(rows[i]["time"]) for i in order])
    typical = float(np.median(np.diff(times))) if len(times) > 1 else 5.0
    runs: list[list[int]] = []
    current: list[int] = []
    for index in order:
        if current:
            previous = rows[current[-1]]
            gap = float(rows[index]["time"]) - float(previous["time"])
            phase_changed = rows[index]["phase"] != previous["phase"]
            h2 = float(rows[index]["h2_value"])
            old_h2 = float(previous["h2_value"])
            direction_broke = (
                (rows[index]["phase"] == "reaction" and h2 + .15 < old_h2)
                or (rows[index]["phase"] == "recovery" and h2 > old_h2 + .15)
            )
            if phase_changed or gap > typical * 2.5 or direction_broke:
                runs.append(current)
                current = []
        current.append(index)
    if current:
        runs.append(current)
    return runs


def shifted_targets(rows: list[dict[str, object]], lag_by_phase: dict[str, float]) -> np.ndarray:
    result = np.asarray([float(row["h2_value"]) for row in rows])
    for run in phase_runs(rows):
        phase = str(rows[run[0]]["phase"])
        lag = lag_by_phase.get(phase, 0.0)
        if phase not in ("reaction", "recovery") or lag <= 0 or len(run) < 3:
            continue
        time = np.asarray([float(rows[i]["time"]) for i in run])
        value = np.asarray([float(rows[i]["h2_value"]) for i in run])
        video = str(rows[run[0]]["video"])
        if phase == "reaction" and video in H2_STEP_TIMELINES:
            result[run] = [step_value_at(video, seconds - lag) for seconds in time]
            continue
        left = 0.0 if phase == "reaction" else 4.0
        result[run] = np.interp(time - lag, time, value, left=left, right=float(value[-1]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("training/output/h2_lag_report.json"))
    parser.add_argument("--max-lag", type=int, default=30)
    args = parser.parse_args()

    rows = [r for r in read_legacy_continuous(args.input) if r["kind"] != "rh_only"]
    for row in rows:
        apply_supplied_nominal_timeline(row)
    x = np.asarray([[feature_value(row, name) for name in H2_FEATURES] for row in rows])
    nominal = np.asarray([float(row["h2_value"]) for row in rows])
    videos = np.asarray([str(row["video"]) for row in rows])
    proxy = np.full(len(rows), np.nan)
    for video in sorted(set(videos)):
        test = videos == video
        train = ~test
        proxy[test] = clone(regression_model()).fit(x[train], nominal[train]).predict(x[test])

    lag_report: dict[str, object] = {}
    corrected = nominal.copy()
    for video in sorted(set(videos)):
        mask = videos == video
        video_rows = [row for row, keep in zip(rows, mask) if keep]
        video_proxy = proxy[mask]
        chosen: dict[str, float] = {}
        details: dict[str, object] = {}
        for phase in ("reaction", "recovery"):
            phase_count = sum(str(row["phase"]) == phase for row in video_rows)
            if phase_count < 6:
                continue
            # Search twice as far as the admissible range.  If the optimum
            # keeps moving when the window expands, colour-response amplitude
            # is being mistaken for lag and the estimate must be rejected.
            scores = []
            for lag in range(args.max_lag * 2 + 1):
                target = shifted_targets(video_rows, {phase: float(lag)})
                use = np.asarray([str(row["phase"]) == phase for row in video_rows])
                scores.append(float(mean_absolute_error(target[use], video_proxy[use])))
            best = int(np.argmin(scores[:args.max_lag + 1]))
            expanded_best = int(np.argmin(scores))
            improvement = scores[0] - scores[best]
            # A one-frame-scale improvement is required. Otherwise the data do
            # not identify a lag and zero is safer than an arbitrary shift.
            step = np.median(np.diff(sorted(float(row["time"]) for row in video_rows)))
            boundary_stable = bool(abs(expanded_best - best) <= max(1.0, step / 2))
            reliable = bool(
                best > 0 and best <= args.max_lag and boundary_stable
                and improvement >= .08 and best >= min(2, step)
            )
            chosen[phase] = float(best if reliable else 0)
            details[phase] = {
                "selected_seconds": chosen[phase],
                "raw_best_seconds": best,
                "expanded_best_seconds": expanded_best,
                "mae_at_zero": scores[0],
                "mae_at_best": scores[best],
                "improvement": improvement,
                "reliable": reliable,
                "boundary_stable": boundary_stable,
                "labelled_frames": phase_count,
            }
        corrected[mask] = shifted_targets(video_rows, chosen)
        lag_report[video] = details

    # Write the original extracted rows with both nominal and corrected labels.
    with args.input.open(encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    lookup: dict[tuple[str, float], tuple[float, float, str]] = {
        (str(row["video"]), float(row["time"])): (
            float(row["h2_value"]), float(value), str(row["phase"]),
        )
        for row, value in zip(rows, corrected)
    }
    for row in raw:
        key = (str(row["video"]), float(row["time"]))
        if key in lookup:
            nominal_value, corrected_value, phase = lookup[key]
            row["h2_value_nominal"] = f"{nominal_value:.6f}"
            row["h2_value"] = f"{corrected_value:.6f}"
            row["phase"] = phase
        else:
            row["h2_value_nominal"] = row["h2_value"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)

    before_prediction = proxy
    summary = {
        "method": "per-video phase lag against leave-one-video-out colour proxy",
        "nominal_policy": "supplied H2-only steps; Recovery is H2 0%; simultaneous Reaction remains 0-4% ramp",
        "before_proxy_mae": float(mean_absolute_error(nominal, before_prediction)),
        "after_alignment_proxy_mae": float(mean_absolute_error(corrected, before_prediction)),
        "videos": lag_report,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
