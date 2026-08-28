"""Construct provisional seven-band Place-2 RH optical anchors.

Response3 labels are the user-reviewed sensor-equivalent ranges, not chamber
ground truth.  A missing 70-80 anchor is selected between the reviewed 60-70
and 80-90 anchors.  Response6 anchors are then matched to the response3 colour
path with an ordered dynamic program.  Outputs are diagnostic until app-frame
tests confirm the proposed times.
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

from rh_40_50_cross_run_spatial_analysis import build, extract
from train_models import CACHE_VERSION, read_csv


VIDEOS = {
    "response3": "1_90_H2O_only_3(response).mp4",
    "response6": "1_90_H2O_only_6(response).mp4",
}
LEVELS = np.asarray([25, 35, 45, 55, 65, 75, 85], dtype=float)
DISPLAY = ["20–30", "30–40", "40–50", "50–60", "60–70", "70–80", "80–90"]

# User-reviewed response3 optical-equivalent ranges.  The 75 midpoint is the
# only missing band and is searched inside 27--35 s below.
R3_REVIEWED = {
    25: (1.5, 2.5), 35: (4.5, 6.5), 45: (10.0,), 55: (20.0,),
    65: (27.0,), 85: (35.0,),
}
R6_WINDOWS = {
    25: (2.0, 10.0), 35: (7.0, 13.0), 45: (10.0, 15.0),
    55: (12.0, 17.0), 65: (14.0, 19.0), 75: (16.0, 22.0),
    85: (19.0, 32.0),
}


def nearest_row(rows, video, seconds):
    candidates = [row for row in rows if row.get("video") == video]
    return min(candidates, key=lambda row: abs(float(row["time"]) - seconds))


def item(rows, group, seconds, stage):
    video = VIDEOS[group]
    row = nearest_row(rows, video, seconds)
    return {"video": video, "group": group, "time": float(row["time"]),
            "requested_time": seconds, "stage": float(stage), "row": row}


def standardize(reference, candidates):
    joined = np.vstack([reference, candidates])
    scale = np.maximum(np.std(joined, axis=0), .5)
    return reference / scale, candidates / scale, scale


def ordered_match(times, candidates, prototypes):
    ref, values, scale = standardize(prototypes, candidates)
    cost = np.sqrt(np.mean((values[:, None, :] - ref[None, :, :]) ** 2, axis=2))
    allowed = np.zeros_like(cost, dtype=bool)
    for column, level in enumerate(LEVELS):
        lo, hi = R6_WINDOWS[int(level)]
        allowed[:, column] = (times >= lo) & (times <= hi)
    cost[~allowed] = np.inf
    n, k = cost.shape
    dp = np.full((n, k), np.inf); parent = np.full((n, k), -1, dtype=int)
    dp[:, 0] = cost[:, 0]
    for column in range(1, k):
        for row in range(n):
            prior = np.flatnonzero(times[:row] <= times[row] - .5)
            if not len(prior) or not np.isfinite(cost[row, column]):
                continue
            best = prior[np.argmin(dp[prior, column - 1])]
            dp[row, column] = dp[best, column - 1] + cost[row, column]
            parent[row, column] = best
    last = int(np.argmin(dp[:, -1]))
    if not np.isfinite(dp[last, -1]):
        raise RuntimeError("No ordered response6 path found")
    indices = [last]
    for column in range(k - 1, 0, -1):
        indices.append(parent[indices[-1], column])
    indices.reverse()
    return np.asarray(indices), cost, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_place2_seven_band_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.cache)

    # Extract reviewed response3 anchors plus every 0.5 s response6 candidate.
    items = []
    for level, seconds_values in R3_REVIEWED.items():
        items.extend(item(rows, "response3", seconds, level)
                     for seconds in seconds_values)
    r3_75_scan = np.arange(27.5, 35.0, .5)
    items.extend(item(rows, "response3", seconds, 999) for seconds in r3_75_scan)
    r6_baselines = (2.0, 6.0)
    items.extend(item(rows, "response6", seconds, 25) for seconds in r6_baselines)
    r6_scan = np.arange(2.0, 32.01, .5)
    items.extend(item(rows, "response6", seconds, 999) for seconds in r6_scan)

    summaries = extract(items, args.video_root)
    matrices, audit = build(items, summaries, included_levels=np.asarray([*LEVELS, 999]))
    x = matrices["background_control"]
    audit_groups = np.asarray([row["group"] for row in audit])
    audit_stage = np.asarray([row["reference"] for row in audit])
    audit_time = np.asarray([row["time"] for row in audit])

    # Interpolate the missing response3 70-80 colour between reviewed 65/85.
    prototypes = {}
    for level in LEVELS:
        if level == 75:
            continue
        use = (audit_groups == "response3") & (audit_stage == level)
        prototypes[level] = np.median(x[use], axis=0)
    target75 = (prototypes[65] + prototypes[85]) / 2
    use75 = (audit_groups == "response3") & (audit_stage == 999)
    scale75 = np.maximum(np.std(x[use75], axis=0), .5)
    distances75 = np.sqrt(np.mean(((x[use75] - target75) / scale75) ** 2, axis=1))
    best75 = np.flatnonzero(use75)[int(np.argmin(distances75))]
    r3_75_time = float(audit_time[best75]); prototypes[75] = x[best75]
    prototype_matrix = np.asarray([prototypes[level] for level in LEVELS])

    r6_use = (audit_groups == "response6") & (audit_stage == 999)
    r6_times, r6_values = audit_time[r6_use], x[r6_use]
    order = np.argsort(r6_times); r6_times, r6_values = r6_times[order], r6_values[order]
    matched, costs, scale = ordered_match(r6_times, r6_values, prototype_matrix)
    r6_anchor_times = r6_times[matched]

    anchors = []
    for level in LEVELS:
        reviewed = list(R3_REVIEWED.get(int(level), (r3_75_time,)))
        for seconds in reviewed:
            anchors.append({"run": "response3", "time_s": seconds,
                            "midpoint": int(level), "range": DISPLAY[list(LEVELS).index(level)],
                            "source": "user-reviewed" if level != 75 else "interpolated-candidate"})
    for level, seconds, index in zip(LEVELS, r6_anchor_times, matched):
        anchors.append({"run": "response6", "time_s": float(seconds),
                        "midpoint": int(level), "range": DISPLAY[list(LEVELS).index(level)],
                        "source": "ordered-colour-match",
                        "distance": float(costs[index, list(LEVELS).index(level)])})

    payload = {
        "status": "diagnostic_not_deployed",
        "meaning": "sensor-equivalent optical RH range; not independent chamber ground truth",
        "levels": LEVELS.astype(int).tolist(), "display_levels": DISPLAY,
        "response3_missing_70_80_candidate_seconds": r3_75_time,
        "response6_matched_times_seconds": {
            label: float(seconds) for label, seconds in zip(DISPLAY, r6_anchor_times)},
        "anchors": anchors,
        "next_gate": "Confirm response3 70-80 and all response6 matched times in app before export",
    }

    # Build an experimental app model from the reviewed/matched anchors.  It is
    # written to the analysis output only; deployment remains a separate step.
    model_vectors, model_classes, model_groups = [], [], []
    for level in LEVELS:
        for seconds in R3_REVIEWED.get(int(level), (r3_75_time,)):
            choices = np.flatnonzero(audit_groups == "response3")
            chosen = choices[int(np.argmin(np.abs(audit_time[choices] - seconds)))]
            model_vectors.append(x[chosen]); model_classes.append(int(level))
            model_groups.append("rh-response-3")
    for level, scan_index in zip(LEVELS, matched):
        model_vectors.append(r6_values[scan_index]); model_classes.append(int(level))
        model_groups.append("rh-response-6")
    model_vectors = np.asarray(model_vectors)
    endpoint_model = {
        "schema_version": 2, "task": "RH",
        "profile": "place2_seven_band_optical_experimental",
        "type": "standardized_1nn",
        "feature_extractor": "app-tight-drop-minus-near-substrate-v1",
        "features": ["drop_minus_substrate_L", "drop_minus_substrate_a",
                     "drop_minus_substrate_b"],
        "classes": model_classes,
        "levels": LEVELS.astype(int).tolist(), "display_levels": DISPLAY,
        "low_chroma_abs_ab_max": 1.25,
        "scaler_mean": np.mean(model_vectors, axis=0).astype(float).tolist(),
        "scaler_scale": np.maximum(np.std(model_vectors, axis=0), .5).astype(float).tolist(),
        "prototypes": model_vectors.astype(float).tolist(),
        "source_groups": model_groups,
        "scope": "H2O-only sensor-equivalent optical ranges; diagnostic pending app confirmation",
    }
    (args.output / "experimental_model.json").write_text(
        json.dumps(endpoint_model, ensure_ascii=False, indent=2), encoding="utf-8")
    js = "// Generated diagnostic model; confirm candidate times before deployment.\n"
    js += "window.SENSOR_RH_PLACE2_SEVEN_BAND_MODEL=" + json.dumps(
        endpoint_model, ensure_ascii=False, separators=(",", ":")) + ";\n"
    (args.output / "sensor-rh-place2-seven-band-model.js").write_text(js, encoding="utf-8")
    (args.output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "candidate_anchors.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in anchors for key in row}))
        writer.writeheader(); writer.writerows(anchors)

    fig, axis = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    axis.plot([np.mean(R3_REVIEWED.get(int(level), (r3_75_time,))) for level in LEVELS],
              LEVELS, "o-", label="response3 reviewed/candidate")
    axis.plot(r6_anchor_times, LEVELS, "s-", label="response6 colour matched")
    axis.set(yticks=LEVELS, yticklabels=DISPLAY, xlabel="Video time (s)",
             ylabel="Sensor-equivalent RH range", title="Place-2 seven-band optical anchors")
    axis.grid(alpha=.2); axis.legend()
    fig.savefig(args.output / "seven_band_anchor_times.png", dpi=220); plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
