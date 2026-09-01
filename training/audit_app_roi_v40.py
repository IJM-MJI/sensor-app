"""Mirror the v40 browser ROI selector on tight, cropped, and uncropped inputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


VIDEO_CASES = (
    ("daylight-cropped-cal", "1_90_H2O_only_cropped.mp4", 38.0),
    ("daylight-uncropped-cal", "1_90_H2O_only.MOV", 38.0),
    ("response3-cropped-cal", "1_90_H2O_only_3(response)_cropped.mp4", .5),
    ("response3-uncropped-cal", "1_90_H2O_only_3(response).mp4", .5),
    ("response6-cropped-cal", "1_90_H2O_only_6(response)_cropped.mp4", 2.0),
    ("response6-uncropped-cal", "1_90_H2O_only_6(response).mp4", 2.0),
)


def resize(frame):
    scale = min(1.0, 480 / max(frame.shape[:2]))
    return cv2.resize(frame, (round(frame.shape[1] * scale),
                              round(frame.shape[0] * scale))) if scale < 1 else frame.copy()


def select(frame):
    frame = resize(frame); h, w = frame.shape[:2]; side = min(h, w)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    found = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, 1.2, round(side * .12),
        param1=80, param2=35, minRadius=round(side * .07),
        maxRadius=round(side * .60))
    circles = [] if found is None else [tuple(map(int, np.round(row))) for row in found[0]]
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    aa = lab[:, :, 1].astype(float) - 128; bb = lab[:, :, 2].astype(float) - 128
    warm = (np.hypot(aa, bb) > 15) & (lab[:, :, 2] > 133)
    yy, xx = np.ogrid[:h, :w]
    valid, best, outer = [], None, None
    tight = max(w, h) / side < 1.35
    for cx, cy, radius in circles:
        nx, ny, nr = cx / w, cy / h, radius / side
        x_lo, x_hi = ((.18, .82) if tight else (.24, .76))
        y_lo, y_hi = ((.10, .78) if tight else (.15, .65))
        if not (x_lo <= nx <= x_hi and y_lo <= ny <= y_hi and .20 <= nr <= .60):
            continue
        valid.append((cx, cy, radius))
        if tight and nr >= .43:
            distance = math.hypot(nx - .50, ny - .52)
            if outer is None or distance < outer[0]:
                outer = (distance, cx, cy, radius)
        inner = round(radius * .5)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= inner ** 2
        coverage = float(np.mean(warm[mask])) if mask.any() else 0.0
        warm_y = np.where(mask & warm)[0]
        span = ((warm_y.max() - warm_y.min()) / max(radius, 1)
                if warm_y.size else 0.0)
        center_prior = math.exp(-.5 * (((nx - .50) / .30) ** 2 +
                                       ((ny - .43) / .30) ** 2))
        radius_prior = math.exp(-.5 * ((nr - .30) / .14) ** 2)
        pair_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (.9 * radius) ** 2
        pair_total = int(np.sum(warm & pair_mask))
        upper = int(np.sum(warm & pair_mask & (yy < cy - .08 * radius)))
        lower = int(np.sum(warm & pair_mask & (yy > cy + .08 * radius)))
        pair_balance = min(upper, lower) / max(pair_total, 1)
        aperture_center = math.exp(-.5 * (((nx - .50) / .18) ** 2 +
                                          ((ny - .48) / .22) ** 2))
        score = ((coverage + .05 * min(span, 1.8)) * math.sqrt(nr) +
                 .08 * center_prior + .10 * radius_prior +
                 .35 * pair_balance + .08 * aperture_center)
        if best is None or score > best[0]:
            best = (score, cx, cy, radius)
    if tight and outer is not None:
        _, ox, oy, outer_r = outer
        circle = (round(.75 * (w * .50) + .25 * ox),
                  round(.75 * (h * .52) + .25 * oy),
                  round(.75 * (side * .40) + .25 * (outer_r * .75)))
        source = "tight-crop-aperture"
    elif best is not None:
        circle = best[1:]; source = "auto"
    else:
        circle = None; source = "none"
    return frame, circles, valid, circle, source


def decorate(frame, candidates, circle, label):
    view = frame.copy()
    for cx, cy, radius in candidates:
        cv2.circle(view, (cx, cy), radius, (0, 0, 255), 1)
    if circle:
        cv2.circle(view, circle[:2], circle[2], (0, 255, 0), 3)
    cv2.rectangle(view, (0, 0), (min(view.shape[1], 330), 28), (0, 0, 0), -1)
    cv2.putText(view, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, .48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path,
                        default=Path(r"C:\Users\Administrator\Downloads\app_test"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/app_roi_v40_audit"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows, views = [], []
    still_paths = sorted(args.image_root.glob("1_90_H2O_only_*.png"),
                         key=lambda path: ("calibrate" not in path.stem, path.name))
    cases = [(path.stem, cv2.imread(str(path)), "still") for path in still_paths]
    for label, video, seconds in VIDEO_CASES:
        cap = cv2.VideoCapture(str(args.video_root / video))
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000); ok, frame = cap.read(); cap.release()
        if ok:
            cases.append((label, frame, "video"))
    registration = None
    for label, raw, kind in cases:
        frame, candidates, valid, auto_circle, auto_source = select(raw)
        h, w = frame.shape[:2]; side = min(h, w)
        circle, source = auto_circle, auto_source
        aspect = w / max(h, 1)
        if kind == "still" and "calibrate" in label and auto_circle is not None:
            registration = (auto_circle[0] / w, auto_circle[1] / h,
                            auto_circle[2] / side, aspect)
        elif kind == "still" and registration is not None:
            rcx, rcy, rr, registered_aspect = registration
            if abs(math.log(aspect / registered_aspect)) <= .08:
                circle = (round(rcx * w), round(rcy * h), round(rr * side))
                source = "calibration-lock"
        nx = None if circle is None else circle[0] / w
        ny = None if circle is None else circle[1] / h
        nr = None if circle is None else circle[2] / side
        tight = max(w, h) / side < 1.35
        passed = circle is not None and (not tight or (
            source in ("tight-crop-aperture", "calibration-lock") and .38 <= nx <= .62 and
            .39 <= ny <= .65 and .32 <= nr <= .46))
        rows.append({"case": label, "kind": kind, "width": w, "height": h,
                     "all_candidates": len(candidates), "valid_candidates": len(valid),
                     "auto_source": auto_source, "roi_source": source,
                     "cx": "" if circle is None else circle[0],
                     "cy": "" if circle is None else circle[1],
                     "radius": "" if circle is None else circle[2],
                     "normalized_x": nx, "normalized_y": ny,
                     "normalized_radius": nr, "geometry_pass": passed})
        views.append(decorate(frame, candidates, circle,
                              f"{label} | {source} | pass={passed}"))
    with (args.output / "roi_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    tile_w, tile_h = 480, 270
    tiles = [cv2.resize(view, (tile_w, tile_h)) for view in views]
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[index:index + 3])
                       for index in range(0, len(tiles), 3)])
    cv2.imwrite(str(args.output / "roi_audit.jpg"), sheet)
    print(f"cases={len(rows)} passed={sum(row['geometry_pass'] for row in rows)}")
    for row in rows:
        print(row["case"], row["roi_source"], row["cx"], row["cy"], row["radius"],
              "PASS" if row["geometry_pass"] else "FAIL")


if __name__ == "__main__":
    main()
