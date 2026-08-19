"""Render aligned visual QA sheets for difficult middle concentration endpoints."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from make_endpoint_mask_review import frame_at, masks_for, source_path
from train_models import CACHE_VERSION, read_csv


TARGETS = {"H2": (2.0, 3.0), "RH": (40.0, 50.0, 60.0, 80.0)}
STAGE_STEP = {"H2": 1.0, "RH": 10.0}


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def rotate_and_crop(image: np.ndarray, circle: tuple[int, int, int], quarters: int,
                    size: int = 220, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Crop a fixed chamber square and rotate it into flame-up coordinates."""
    x, y, radius = circle
    half = max(8, int(round(radius * 1.08)))
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_CONSTANT,
                                value=0)
    cx, cy = x + half, y + half
    crop = padded[cy - half:cy + half, cx - half:cx + half]
    crop = np.rot90(crop, k=quarters % 4).copy()
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def paint_mask(frame: np.ndarray, cache_row: dict[str, object], task: str):
    flame_zone, drop_zone, flame, drop, circle = masks_for(
        frame, cache_row, 65.0, False, True)
    selected = flame if task == "H2" else drop
    zone = flame_zone if task == "H2" else drop_zone
    colour = (0, 55, 255) if task == "H2" else (255, 175, 0)
    outline_colour = (0, 255, 255) if task == "H2" else (255, 255, 0)
    marked = frame.copy()
    tint = np.zeros_like(frame); tint[selected] = colour
    marked[selected] = cv2.addWeighted(frame[selected], .30, tint[selected], .70, 0)
    contours, _ = cv2.findContours(zone.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(marked, contours, -1, outline_colour, 2)
    return marked, circle


def put_text(image: np.ndarray, text: str, origin: tuple[int, int], scale=.43,
             colour=(25, 25, 25), thickness=1):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thickness, cv2.LINE_AA)


def prediction_lookup(rows: list[dict[str, object]]):
    result = {}
    for row in rows:
        if str(row["mode"]).startswith("one_anchor") and row["source"] == "endpoint":
            key = (str(row["task"]), str(row["group"]), float(row["reference"]))
            result[key] = row
    return result


def trajectory_lookup(rows: list[dict[str, object]]):
    return {(str(row["task"]), str(row["group"]), float(row["level"])):
            float(row["progress"]) for row in rows}


def endpoint_rows(rows: list[dict[str, object]], task: str, level: float):
    candidates = [row for row in rows
                  if row["task"] == task and row["source"] == "endpoint"
                  and row.get("exact") not in (None, "")
                  and float(row["exact"]) == level]
    # Exactly one stated endpoint per independent run/group.
    unique = {}
    for row in candidates:
        unique[str(row["group"])] = row
    return [unique[group] for group in sorted(unique)]


def cache_nearest(by_video, video: str, seconds: float):
    return min(by_video[video], key=lambda row: abs(float(row["time"]) - seconds))


def render_cell(frame: np.ndarray, cache_row: dict[str, object], endpoint: dict[str, object],
                prediction: dict[str, object] | None, progress: float | None,
                task: str, level: float) -> tuple[np.ndarray, dict[str, object]]:
    marked, circle = paint_mask(frame, cache_row, task)
    quarters = int(float(cache_row["orientation_quarters"]))
    raw = rotate_and_crop(frame, circle, quarters)
    mask = rotate_and_crop(marked, circle, quarters)
    pred = float(prediction["prediction"]) if prediction else float("nan")
    confidence = float(prediction["confidence"]) if prediction else float("nan")
    error = abs(pred - level) if np.isfinite(pred) else float("inf")
    step_error = error / STAGE_STEP[task]
    border = (40, 150, 40) if step_error < .5 else ((0, 170, 255) if step_error <= 1.0 else
                                                     (40, 40, 220))

    caption_h = 92
    cell = np.full((220 + caption_h, 440, 3), 247, np.uint8)
    cell[:220, :220] = raw; cell[:220, 220:] = mask
    cv2.rectangle(cell, (1, 1), (438, 310), border, 3)
    group = str(endpoint["group"])
    video = Path(str(endpoint["video"])).name
    put_text(cell, f"{group}", (8, 239), .45)
    put_text(cell, f"{video}  t={float(endpoint['time']):.1f}s", (8, 258), .39)
    pred_text = "n/a" if prediction is None else f"{pred:g} (conf {confidence:.2f})"
    progress_text = "n/a" if progress is None else f"{progress:.2f}"
    put_text(cell, f"reference {level:g}  |  held-out prediction {pred_text}", (8, 278), .41)
    put_text(cell, f"two-anchor path progress {progress_text}  |  RAW / SENSING PIXELS",
             (8, 298), .39)
    return cell, {
        "task": task, "reference": level, "group": group, "video": video,
        "time": float(endpoint["time"]), "orientation_quarters": quarters,
        "prediction": "" if prediction is None else pred,
        "confidence": "" if prediction is None else confidence,
        "stage_error": "" if prediction is None else pred - level,
        "two_anchor_progress": "" if progress is None else progress,
        "review_priority": "high" if step_error > 1.0 else ("medium" if step_error >= .5 else "low"),
    }


def write_csv(path: Path, rows: list[dict[str, object]]):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path,
                        default=Path(f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--endpoint-dataset", type=Path,
                        default=Path("training/output/endpoint_interval_registered_v3/endpoint_interval_dataset.csv"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("training/output/run_progress_v3/predictions.csv"))
    parser.add_argument("--trajectories", type=Path,
                        default=Path("training/output/run_progress_v3/normalized_trajectories.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/middle_endpoint_review"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    cache = read_csv(args.cache); endpoints = read_table(args.endpoint_dataset)
    predictions = prediction_lookup(read_table(args.predictions))
    trajectories = trajectory_lookup(read_table(args.trajectories))
    by_video = defaultdict(list)
    for row in cache:
        by_video[str(row["video"])].append(row)

    review_rows = []
    for task, levels in TARGETS.items():
        rows_by_level = {level: endpoint_rows(endpoints, task, level) for level in levels}
        groups = sorted({str(row["group"]) for rows in rows_by_level.values() for row in rows})
        sheets = []
        for level in levels:
            row_map = {str(row["group"]): row for row in rows_by_level[level]}
            cells = []
            for group in groups:
                endpoint = row_map.get(group)
                if endpoint is None:
                    cells.append(np.full((312, 440, 3), 235, np.uint8)); continue
                seconds = float(endpoint["time"]); video = str(endpoint["video"])
                cache_row = cache_nearest(by_video, video, seconds)
                frame = frame_at(source_path(args.video_root, video), seconds)
                cell, record = render_cell(
                    frame, cache_row, endpoint, predictions.get((task, group, level)),
                    trajectories.get((task, group, level)), task, level)
                cells.append(cell); review_rows.append(record)
            header = np.full((56, 440 * len(groups), 3), 250, np.uint8)
            put_text(header, f"{task} endpoint {level:g} | green=exact, amber=adjacent, red=>1 stage error",
                     (10, 24), .58, thickness=1)
            put_text(header, "All chamber crops are rotated to flame-up; coloured pixels are model inputs.",
                     (10, 47), .47)
            sheets.append(np.vstack([header, np.hstack(cells)]))
        sheet = np.vstack(sheets)
        cv2.imwrite(str(args.output / f"{task.lower()}_middle_endpoint_review.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 96])

    write_csv(args.output / "middle_endpoint_review.csv", review_rows)
    print(args.output)
    for row in review_rows:
        if row["review_priority"] != "low":
            print(row)


if __name__ == "__main__":
    main()
