"""Build a compact, decision-oriented atlas for H2 0% and 4% endpoints."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from make_endpoint_mask_review import frame_at, masks_for, outline
from ordinal_concentration_analysis import (
    H2_RAMP_ENDPOINTS, H2_RECOVERY_START, apply_h2_frame_quality_profile,
    assign_h2_ramp_targets,
)
from train_models import CACHE_VERSION, read_csv


def read_predictions(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_video = defaultdict(list)
    for row in rows:
        by_video[str(row["video"])].append(row)
    return by_video


def original_source(video_root: Path, logical_video: str) -> Path:
    path = video_root / logical_video
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def endpoint_times(video: str, duration: float):
    points = H2_RAMP_ENDPOINTS[video]
    zero = max(time for time, value in points if value == 0)
    # The indoor test starts its ramp immediately, but the real app test used a
    # 2 s calibration photo. Avoid the camera-start/exposure transient at t=0;
    # 2 s still rounds to the trained 0% stage on the 0-to-1% ramp.
    if zero == 0:
        zero = min(2.0, points[1][0] * .25)
    high = max(time for time, value in points if value == 4)
    # Stay on the reaction side when recovery starts at the nominal 4% endpoint.
    if H2_RECOVERY_START.get(video) == high:
        high = max(0.0, high - .5)
    else:
        high = min(high, duration - .5)
    return [("H2 0% anchor", float(zero), 0), ("H2 4% endpoint", float(high), 4)]


def crop_square(image, circle, scale=1.22):
    x, y, radius = circle
    half = int(round(radius * scale))
    x0, x1 = max(0, x - half), min(image.shape[1], x + half)
    y0, y1 = max(0, y - half), min(image.shape[0], y + half)
    return image[y0:y1, x0:x1]


def render_tile(frame, cache_row, record):
    flame_zone, _, flame, _, circle = masks_for(
        frame, cache_row, 65.0, False, False)
    overlay = frame.copy()
    tint = np.zeros_like(frame); tint[flame] = (0, 50, 255)
    overlay[flame] = cv2.addWeighted(frame[flame], .25, tint[flame], .75, 0)
    outline(overlay, flame_zone, (0, 255, 255), 1)
    cv2.circle(overlay, circle[:2], circle[2], (0, 255, 0), 1)
    isolated = np.full_like(frame, 24)
    isolated[flame] = frame[flame]

    panel, image_h, caption_h = 205, 205, 118
    views = [crop_square(item, circle) for item in (frame, overlay, isolated)]
    views = [cv2.resize(item, (panel, image_h), interpolation=cv2.INTER_AREA)
             for item in views]
    tile = np.full((image_h + caption_h, panel * 3, 3), 246, np.uint8)
    for index, view in enumerate(views):
        tile[:image_h, index * panel:(index + 1) * panel] = view
    lines = [
        "RAW                         FLAME MASK                  SELECTED PIXELS",
        f"{record['video']} | {record['endpoint']} | t={record['time']:.1f}s",
        f"reference={record['reference']}%  prediction={record['prediction']}%  "
        f"error={record['absolute_error']} stage",
        f"dLAB=({record['flame_L']:+.2f}, {record['flame_a']:+.2f}, "
        f"{record['flame_b']:+.2f})  mask={record['flame_pixels']} px",
        f"local-path score={record['outlier_score']:.2f}  "
        f"within-run percentile={record['outlier_percentile']:.1f}%",
        f"AUTO REVIEW: {record['auto_review']}",
    ]
    for index, text in enumerate(lines):
        colour = (20, 20, 190) if index == len(lines) - 1 else (25, 25, 25)
        cv2.putText(tile, text, (7, image_h + 17 + index * 17),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, colour, 1, cv2.LINE_AA)
    return tile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path,
                        default=Path("training/cache") / CACHE_VERSION / "features.csv")
    parser.add_argument("--predictions", type=Path,
                        default=Path("training/output/rh20_selective_confidence_lovo_v9/predictions.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/h2_endpoint_decision_atlas_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.cache)
    assign_h2_ramp_targets(rows)
    apply_h2_frame_quality_profile(rows, "local_path")
    by_video = defaultdict(list)
    for row in rows:
        if str(row["video"]) in H2_RAMP_ENDPOINTS:
            by_video[str(row["video"])].append(row)
    predictions = read_predictions(args.predictions)

    records, tiles = [], []
    for video in H2_RAMP_ENDPOINTS:
        video_rows = by_video[video]
        duration = max(float(row["duration"]) for row in video_rows)
        scores = np.asarray([float(row.get("h2_frame_outlier_score", 0))
                             for row in video_rows if "h2_frame_outlier_score" in row])
        for endpoint, seconds, reference in endpoint_times(video, duration):
            cache_row = min(video_rows, key=lambda row: abs(float(row["time"]) - seconds))
            actual_time = float(cache_row["time"])
            prediction_row = min(predictions[video],
                                 key=lambda row: abs(float(row["time"]) - actual_time))
            prediction = int(float(prediction_row["ensemble_prediction"]))
            score = float(cache_row.get("h2_frame_outlier_score", 0))
            percentile = float(100 * np.mean(scores <= score)) if len(scores) else 0.0
            error = abs(prediction - reference)
            if percentile > 95:
                review = "REVIEW - optically unstable"
            elif error == 0:
                review = "CONSISTENT"
            elif error == 1:
                review = "AMBIGUOUS - adjacent stage"
            else:
                review = "REVIEW - reference/prediction mismatch"
            frame = frame_at(original_source(args.video_root, video), actual_time)
            flame, _, flame_mask, _, _ = masks_for(frame, cache_row, 65.0, False, False)
            del flame
            record = {
                "video": video, "endpoint": endpoint, "time": actual_time,
                "reference": reference, "prediction": prediction,
                "absolute_error": error,
                "flame_L": float(cache_row["flame_L"]),
                "flame_a": float(cache_row["flame_a"]),
                "flame_b": float(cache_row["flame_b"]),
                "flame_pixels": int(flame_mask.sum()),
                "outlier_score": score, "outlier_percentile": percentile,
                "auto_review": review,
            }
            records.append(record); tiles.append(render_tile(frame, cache_row, record))

    columns = 2; blank = np.full_like(tiles[0], 246)
    while len(tiles) % columns: tiles.append(blank.copy())
    body = np.vstack([np.hstack(tiles[index:index + columns])
                      for index in range(0, len(tiles), columns)])
    header = np.full((72, body.shape[1], 3), 250, np.uint8)
    cv2.putText(header, "H2 0% / 4% ENDPOINT DECISION ATLAS",
                (18, 27), cv2.FONT_HERSHEY_SIMPLEX, .78, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(header, "Human decision requested: VALID / PARTIAL RESPONSE / INVALID REFERENCE",
                (18, 56), cv2.FONT_HERSHEY_SIMPLEX, .54, (30, 30, 160), 1, cv2.LINE_AA)
    cv2.imwrite(str(args.output / "h2_endpoint_decision_atlas.jpg"), np.vstack([header, body]),
                [cv2.IMWRITE_JPEG_QUALITY, 97])
    with (args.output / "endpoint_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    summary = {
        "n_endpoints": len(records),
        "auto_review_counts": {name: sum(row["auto_review"] == name for row in records)
                               for name in sorted({row["auto_review"] for row in records})},
        "policy": "automatic flags are diagnostic only; human endpoint validity controls training",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
