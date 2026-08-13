"""Create an aligned review sheet for the scarce H2 0/1/2% stages."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from align_h2_lag import H2_STEP_TIMELINES, RECOVERY_START
from train_models import CACHE_VERSION, resize_for_app


def intervals(video: str):
    steps = H2_STEP_TIMELINES[video]
    stop = RECOVERY_START.get(video, float("inf"))
    out = {}
    for index, (start, level) in enumerate(steps):
        end = steps[index + 1][0] if index + 1 < len(steps) else stop
        if level in (0, 1, 2) and end < float("inf") and start < stop:
            out[int(level)] = (float(start), float(end))
    return out


def frame_at(path: Path, seconds: float):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    return resize_for_app(frame) if ok else None


def main():
    root = Path(r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1")
    output = Path("training/output/ordinal_concentration/h2_low_stage_review.png")
    with (Path("training/cache") / CACHE_VERSION / "features.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_video = {}
    for row in rows:
        by_video.setdefault(row["video"], []).append(row)

    videos = list(H2_STEP_TIMELINES)
    cell, header = 260, 62
    sheet = np.full((header + len(videos) * cell, 3 * cell, 3), 245, np.uint8)
    for col, level in enumerate((0, 1, 2)):
        cv2.putText(sheet, f"H2 {level}% (late nominal stage)", (col * cell + 12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, .62, (20, 20, 20), 2, cv2.LINE_AA)
    for row_index, video in enumerate(videos):
        ranges = intervals(video)
        for col, level in enumerate((0, 1, 2)):
            y0, x0 = header + row_index * cell, col * cell
            if level not in ranges:
                cv2.putText(sheet, "not recorded", (x0 + 55, y0 + 135),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (90, 90, 90), 2, cv2.LINE_AA)
                continue
            start, end = ranges[level]
            seconds = max(start, end - min(.7, (end - start) * .12))
            frame = frame_at(root / video, seconds)
            candidates = by_video[video]
            nearest = min(candidates, key=lambda item: abs(float(item["time"]) - seconds))
            cx, cy, radius = (int(float(nearest[key])) for key in ("circle_x", "circle_y", "circle_r"))
            pad = int(radius * 1.05)
            crop = frame[max(0, cy-pad):min(frame.shape[0], cy+pad),
                         max(0, cx-pad):min(frame.shape[1], cx+pad)]
            crop = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_AREA)
            sheet[y0:y0+cell, x0:x0+cell] = crop
            cv2.rectangle(sheet, (x0, y0), (x0 + cell - 1, y0 + cell - 1), (30, 30, 30), 1)
            cv2.putText(sheet, f"{video}  t={seconds:.1f}s", (x0 + 7, y0 + cell - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, .39, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(sheet, f"{video}  t={seconds:.1f}s", (x0 + 7, y0 + cell - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, .39, (20, 20, 20), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)
    print(output)


if __name__ == "__main__":
    main()
