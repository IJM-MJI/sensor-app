"""Audit central-chamber circle selection on the supplied response3 endpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


TIMES = (0.5, 2, 3, 5, 7, 11, 25, 28, 38)


def frame_at(capture: cv2.VideoCapture, seconds: float) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read t={seconds}s")
    scale = 480 / max(frame.shape[:2])
    return cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))


def candidates(frame: np.ndarray) -> list[tuple[int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape
    base = min(h, w)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, 1.2, round(base * .12),
        param1=80, param2=35, minRadius=round(base * .07),
        maxRadius=round(base * .60),
    )
    if circles is None:
        return []
    return [tuple(int(round(v)) for v in circle) for circle in circles[0]]


def central(circle: tuple[int, int, int], shape: tuple[int, ...]) -> bool:
    x, y, radius = circle
    h, w = shape[:2]
    base = min(h, w)
    return .24 <= x / w <= .76 and .15 <= y / h <= .65 and .20 <= radius / base <= .60


def select(frame: np.ndarray, circles: list[tuple[int, int, int]]) -> tuple[int, int, int] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    h, w = gray.shape
    base = min(h, w)
    best: tuple[float, tuple[int, int, int]] | None = None
    for circle in circles:
        if not central(circle, frame.shape):
            continue
        cx, cy, radius = circle
        inner = round(radius * .5)
        ys, xs = np.ogrid[:h, :w]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= inner ** 2
        aa = lab[..., 1].astype(float) - 128
        bb = lab[..., 2].astype(float) - 128
        warm = mask & (np.hypot(aa, bb) > 15) & (lab[..., 2] > 133)
        coverage = warm.sum() / max(mask.sum(), 1)
        warm_y = np.where(warm)[0]
        span = (warm_y.max() - warm_y.min()) / max(radius, 1) if warm_y.size else 0
        nx, ny, nr = cx / w, cy / h, radius / base
        center_prior = np.exp(-.5 * (((nx - .50) / .30) ** 2 + ((ny - .43) / .30) ** 2))
        score = (coverage + .05 * min(span, 1.8)) * np.sqrt(nr) + .08 * center_prior
        if best is None or score > best[0]:
            best = (float(score), circle)
    return None if best is None else best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("training/output/response3_roi_audit"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(args.video)

    rows, tiles = [], []
    for seconds in TIMES:
        frame = frame_at(capture, seconds)
        found = candidates(frame)
        valid = [circle for circle in found if central(circle, frame.shape)]
        chosen = select(frame, found)
        view = frame.copy()
        for x, y, radius in found:
            cv2.circle(view, (x, y), radius, (0, 0, 255), 1)
        if chosen:
            cv2.circle(view, chosen[:2], chosen[2], (0, 255, 0), 3)
        cv2.putText(view, f"t={seconds:g}s  valid={len(valid)}/{len(found)}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append({"time_s": seconds, "all_candidates": len(found), "central_candidates": len(valid),
                     "selected_x": "" if chosen is None else chosen[0],
                     "selected_y": "" if chosen is None else chosen[1],
                     "selected_r": "" if chosen is None else chosen[2]})
        tiles.append(view)

    with (args.output / "response3_circle_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    sheet = np.hstack(tiles[:3])
    sheet = np.vstack((sheet, np.hstack(tiles[3:6]), np.hstack(tiles[6:9])))
    cv2.imwrite(str(args.output / "response3_circle_audit.jpg"), sheet)
    print(*rows, sep="\n")


if __name__ == "__main__":
    main()
