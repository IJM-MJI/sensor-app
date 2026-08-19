"""Render an easy-to-read endpoint review with the actual sensing pixels."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from train_models import (
    CACHE_VERSION, droplet_template_zone, normalized_coordinates, patch_balance_lab,
    read_csv, registered_droplet_template_zone, resize_for_app, shape_pixel_mask,
)


def source_path(video_root: Path, logical_video: str) -> Path:
    cropped = video_root / f"{Path(logical_video).stem}_cropped.mp4"
    return cropped if cropped.exists() else video_root / logical_video


def frame_at(path: Path, seconds: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read(); cap.release()
    if not ok:
        raise RuntimeError(f"Cannot decode {path.name} at {seconds:.2f}s")
    return resize_for_app(frame)


def masks_for(
    frame: np.ndarray,
    row: dict[str, object],
    drop_percentile: float,
    drop_template: bool,
    registered_drop_template: bool,
):
    x, y, radius = (int(float(row[name])) for name in ("circle_x", "circle_y", "circle_r"))
    orientation = int(float(row["orientation_quarters"]))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    yy, xx = np.ogrid[:lab.shape[0], :lab.shape[1]]
    chamber = (xx - x) ** 2 + (yy - y) ** 2 <= (radius * .90) ** 2
    nx, ny = normalized_coordinates(lab.shape, (x, y, radius), orientation)
    balanced = patch_balance_lab(lab, chamber)
    chamber_pixels = balanced[chamber].astype(float)
    chroma = np.hypot(chamber_pixels[:, 1] - 128, chamber_pixels[:, 2] - 128)
    background = chamber_pixels[np.argsort(chroma)[:max(1, len(chroma) // 2)]].mean(axis=0)
    central = (nx >= -.55) & (nx <= .35)
    flame_zone = chamber & central & (ny >= -.62) & (ny <= .14)
    registration_values = [row.get(f"drop_registration_{name}")
                           for name in ("x", "y", "angle")]
    if registered_drop_template and all(value not in (None, "") for value in registration_values):
        registration = tuple(float(value) for value in registration_values)
        drop_zone = registered_droplet_template_zone(chamber, nx, ny, registration)
    elif drop_template:
        drop_zone = droplet_template_zone(chamber, nx, ny)
    else:
        drop_zone = chamber & central & (ny >= .18) & (ny <= .68)
    return flame_zone, drop_zone, shape_pixel_mask(balanced, flame_zone, background), \
        shape_pixel_mask(balanced, drop_zone, background, drop_percentile), (x, y, radius)


def outline(image: np.ndarray, mask: np.ndarray, colour, thickness=1):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, colour, thickness)


def render_tile(
    frame: np.ndarray,
    cache_row: dict[str, object],
    audit_row: dict[str, str],
    drop_percentile: float,
    drop_template: bool,
    registered_drop_template: bool,
) -> np.ndarray:
    flame_zone, drop_zone, flame, drop, circle = masks_for(
        frame, cache_row, drop_percentile, drop_template, registered_drop_template)
    overlay = frame.copy()
    colour = np.zeros_like(frame)
    colour[flame] = (0, 40, 255)       # red/orange: actual H2 pixels
    colour[drop] = (255, 170, 0)       # blue/cyan: actual RH pixels
    selected = flame | drop
    overlay[selected] = cv2.addWeighted(frame[selected], .35, colour[selected], .65, 0)
    outline(overlay, flame_zone, (0, 255, 255), 1)
    outline(overlay, drop_zone, (255, 255, 0), 1)
    cv2.circle(overlay, circle[:2], circle[2], (0, 255, 0), 1)

    panel_w, panel_h, caption_h = 300, 205, 86
    raw = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    marked = cv2.resize(overlay, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    tile = np.full((panel_h + caption_h, panel_w * 2, 3), 245, np.uint8)
    tile[:panel_h, :panel_w] = raw; tile[:panel_h, panel_w:] = marked
    lines = [
        f"RAW                                  MASK",
        f"{audit_row['group']}  {Path(audit_row['representative_video']).name}",
        f"t={float(audit_row['representative_time']):.1f}s  timeline={float(audit_row['nominal_stage']):g}",
        f"other-run nearest={float(audit_row['nearest_consensus_stage']):g}  red=flame blue=drop",
    ]
    for index, text in enumerate(lines):
        cv2.putText(tile, text, (7, panel_h + 18 + index * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (25, 25, 25), 1, cv2.LINE_AA)
    return tile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path,
                        default=Path(f"training/cache/{CACHE_VERSION}/features_cropped_centered_v3.csv"))
    parser.add_argument("--audit", type=Path,
                        default=Path("training/output/single_condition_trajectory_audit_v2/trajectory_audit.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/endpoint_mask_review"))
    parser.add_argument("--per-task", type=int, default=12)
    parser.add_argument("--drop-percentile", type=float, default=65.0)
    parser.add_argument("--drop-template", action="store_true")
    parser.add_argument("--registered-drop-template", action="store_true")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    cache = read_csv(args.cache)
    by_video = defaultdict(list)
    for row in cache:
        by_video[str(row["video"])].append(row)
    with args.audit.open(encoding="utf-8-sig") as handle:
        audit = list(csv.DictReader(handle))

    for task in ("H2", "RH"):
        candidates = [row for row in audit if row["task"] == task and row["review"] == "mismatch"]
        candidates.sort(key=lambda row: (-float(row["distance_margin"]),
                                         -abs(float(row["stage_error"]))))
        tiles = []
        for audit_row in candidates[:args.per_task]:
            video = audit_row["representative_video"]
            seconds = float(audit_row["representative_time"])
            cache_row = min(by_video[video], key=lambda row: abs(float(row["time"]) - seconds))
            frame = frame_at(source_path(args.video_root, video), seconds)
            tiles.append(render_tile(frame, cache_row, audit_row,
                                     args.drop_percentile, args.drop_template,
                                     args.registered_drop_template))
        if not tiles:
            continue
        columns = 2
        blank = np.full_like(tiles[0], 245)
        while len(tiles) % columns:
            tiles.append(blank.copy())
        sheet = np.vstack([np.hstack(tiles[index:index + columns])
                           for index in range(0, len(tiles), columns)])
        cv2.imwrite(str(args.output / f"{task.lower()}_endpoint_mask_review.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 96])
    print(args.output)


if __name__ == "__main__":
    main()
