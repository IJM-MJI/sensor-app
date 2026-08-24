"""Compare alternate place-2 RH frames with place-1 60% endpoints.

The requested 5 s / 13 s frames are diagnostics only.  Supplied timeline labels
remain unchanged; the script asks whether those optical states are closer to the
two place-1 60% endpoints than the nominal place-2 60% endpoints are.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from endpoint_interval_analysis import feature_matrix, prepare
from make_endpoint_mask_review import frame_at, masks_for, source_path
from make_middle_endpoint_review import rotate_and_crop
from run_progress_analysis import fold_transform
from train_models import CACHE_VERSION, read_csv


PLACE1 = (
    ("1_90_H2O_only_2_extract.mp4", 35.0, "place1 fast, nominal 60%"),
    ("1_90_H2O_only_extract_3min.mp4", 120.0, "place1 long, nominal 60%"),
)
PLACE2 = (
    ("1_90_H2O_only_3(response).mp4", 5.0, "place2 response-3, requested"),
    ("1_90_H2O_only_3(response).mp4", 10.9333333333, "place2 response-3, nominal 60%"),
    ("1_90_H2O_only_6(response).mp4", 13.0, "place2 response-6, requested"),
    ("1_90_H2O_only_6(response).mp4", 16.0, "place2 response-6, nominal 60%"),
)
FEATURE_SETS = {
    "drop_lab": (0, 1, 2),
    "drop_lab_plus_reference": (0, 1, 2, 6, 7, 8),
    "all_compact_features": tuple(range(9)),
}


def nearest(items, video, seconds):
    candidates = [item for item in items if item["video"] == video]
    return min(candidates, key=lambda item: abs(float(item["time"]) - seconds))


def put(image, text, origin, scale=.40, colour=(25, 25, 25)):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                1, cv2.LINE_AA)


def render_cell(video_root, item, label):
    row = item["row"]
    frame = frame_at(source_path(video_root, item["video"]), float(item["time"]))
    flame_zone, drop_zone, flame, drop, circle = masks_for(
        frame, row, 65.0, False, True)
    marked = frame.copy(); tint = np.zeros_like(frame); tint[drop] = (255, 175, 0)
    marked[drop] = cv2.addWeighted(frame[drop], .30, tint[drop], .70, 0)
    contours, _ = cv2.findContours(drop_zone.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(marked, contours, -1, (255, 255, 0), 2)
    quarters = int(float(row["orientation_quarters"]))
    raw = rotate_and_crop(frame, circle, quarters, size=210)
    mask = rotate_and_crop(marked, circle, quarters, size=210)
    cell = np.full((282, 420, 3), 248, np.uint8)
    cell[:210, :210] = raw; cell[:210, 210:] = mask
    put(cell, label, (8, 232), .43)
    put(cell, f"{Path(item['video']).name}  t={float(item['time']):.2f}s", (8, 252), .37)
    put(cell, "RAW / REGISTERED DROPLET PIXELS", (8, 272), .36)
    return cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_location_60_candidates_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.cache)
    items = [item for item in prepare(rows)
             if item["task"] == "RH" and item["group"] != "rh-daylight-recovery"]
    raw_x = feature_matrix(items, "RH")
    train = np.ones(len(items), dtype=bool)
    x, _ = fold_transform(items, "RH", raw_x, train, "one_anchor")
    item_index = {id(item): index for index, item in enumerate(items)}

    selected = []
    for video, seconds, label in (*PLACE1, *PLACE2):
        item = nearest(items, video, seconds)
        selected.append((item, label))

    # Place-1 40/50/60 endpoints form location-specific comparison prototypes.
    groups = np.asarray([item["group"] for item in items])
    truth = np.asarray([np.nan if item["exact"] is None else item["exact"] for item in items])
    place1_groups = {"rh-indoor-fast", "rh-indoor-long"}
    records = []
    for feature_name, feature_indices_raw in FEATURE_SETS.items():
        feature_indices = np.asarray(feature_indices_raw)
        prototypes = {}
        for level in (40.0, 50.0, 60.0):
            vectors = []
            for group in sorted(place1_groups):
                use = (groups == group) & (truth == level)
                if np.any(use):
                    vectors.append(np.median(x[use][:, feature_indices], axis=0))
            prototypes[level] = np.median(np.asarray(vectors), axis=0)
        reference_60 = prototypes[60.0]
        for item, label in selected:
            vector = x[item_index[id(item)], feature_indices]
            distances = {level: float(np.sqrt(np.mean((vector - prototype) ** 2)))
                         for level, prototype in prototypes.items()}
            predicted = min(distances, key=distances.get)
            records.append({
                "feature_set": feature_name, "location": "place1" if "place1" in label else "place2",
                "label": label, "video": item["video"], "requested_time": (
                    next(seconds for video, seconds, name in (*PLACE1, *PLACE2)
                         if video == item["video"] and name == label)),
                "actual_cache_time": float(item["time"]),
                "timeline_exact": "" if item["exact"] is None else item["exact"],
                "distance_to_place1_40": distances[40.0],
                "distance_to_place1_50": distances[50.0],
                "distance_to_place1_60": distances[60.0],
                "nearest_place1_stage": predicted,
            })

    with (args.output / "comparison.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    cells = [render_cell(args.video_root, item, label) for item, label in selected]
    header = np.full((68, 1260, 3), 250, np.uint8)
    put(header, "RH place comparison: requested frames are optical diagnostics, not relabels",
        (14, 28), .58)
    put(header, "Each tile: aligned RAW / registered droplet sensing pixels", (14, 53), .46)
    sheet = np.vstack([header, np.hstack(cells[:3]), np.hstack(cells[3:])])
    cv2.imwrite(str(args.output / "rh_location_60_candidate_atlas.jpg"), sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 96])

    summary = defaultdict(dict)
    for feature_name in FEATURE_SETS:
        subset = [row for row in records if row["feature_set"] == feature_name
                  and row["location"] == "place2"]
        for video in sorted(set(row["video"] for row in subset)):
            video_rows = [row for row in subset if row["video"] == video]
            requested = next(row for row in video_rows if "requested" in row["label"])
            nominal = next(row for row in video_rows if "nominal" in row["label"])
            summary[feature_name][video] = {
                "requested_distance_to_place1_60": requested["distance_to_place1_60"],
                "nominal_distance_to_place1_60": nominal["distance_to_place1_60"],
                "requested_is_closer": bool(requested["distance_to_place1_60"]
                                            < nominal["distance_to_place1_60"]),
                "requested_nearest_stage": requested["nearest_place1_stage"],
                "nominal_nearest_stage": nominal["nearest_place1_stage"],
            }
    report = {
        "timeline_labels_changed": False,
        "normalization": "registered droplet features, centred on each run's 20-30% Initial",
        "place1_reference": [label for _, _, label in PLACE1],
        "comparison": summary,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
