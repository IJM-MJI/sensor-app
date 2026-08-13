"""Render concentration-level frames with the ROI used by feature extraction."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from train_models import CACHE_VERSION, read_csv, resize_for_app


def frame_at(path: Path, seconds: float):
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read(); cap.release()
    if not ok: raise RuntimeError(f"Cannot decode {path.name} at {seconds}")
    return resize_for_app(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(f"training/cache/{CACHE_VERSION}/features.csv"))
    parser.add_argument("--output", type=Path, default=Path("training/output/single_condition_review"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.cache)
    by_video = defaultdict(list)
    for row in rows:
        if row["kind"] in ("h2_only", "rh_only"):
            by_video[str(row["video"])].append(row)
    for video, video_rows in by_video.items():
        label = "h2_value" if video_rows[0]["kind"] == "h2_only" else "rh_value"
        labelled = [row for row in video_rows if row.get(label) is not None]
        if not labelled: continue
        levels = defaultdict(list)
        for row in labelled: levels[float(row[label])].append(row)
        panels = []
        crops = []
        for level, level_rows in sorted(levels.items()):
            level_rows.sort(key=lambda row: float(row["time"]))
            row = level_rows[max(0, int(len(level_rows) * .85) - 1)]
            image = frame_at(args.video_root / video, float(row["time"]))
            x, y, radius = (int(float(row[name])) for name in ("circle_x", "circle_y", "circle_r"))
            cv2.circle(image, (x, y), radius, (0, 0, 255), 3)
            cv2.circle(image, (x, y), int(radius*.9), (0, 255, 255), 2)
            text = f"{label.replace('_value','')}={level:g} t={float(row['time']):.1f}s ROI=({x},{y},{radius})"
            cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1, cv2.LINE_AA)
            panels.append(image)
            margin = int(radius * 1.15)
            padded = cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_CONSTANT)
            crop = padded[y:y + 2 * margin, x:x + 2 * margin]
            crop = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_AREA)
            cv2.putText(crop, f"{label.replace('_value','')}={level:g} t={float(row['time']):.1f}s",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(crop, f"{label.replace('_value','')}={level:g} t={float(row['time']):.1f}s",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1, cv2.LINE_AA)
            crops.append(crop)
        width = max(panel.shape[1] for panel in panels); height = max(panel.shape[0] for panel in panels)
        normalized=[]
        for panel in panels:
            canvas=np.zeros((height,width,3),np.uint8); canvas[:panel.shape[0],:panel.shape[1]]=panel; normalized.append(canvas)
        sheet=np.concatenate(normalized,axis=1)
        cv2.imwrite(str(args.output/(Path(video).stem+'.jpg')),sheet,[int(cv2.IMWRITE_JPEG_QUALITY),94])
        cv2.imwrite(str(args.output/(Path(video).stem+'_crop.jpg')),
                    np.concatenate(crops, axis=1), [int(cv2.IMWRITE_JPEG_QUALITY),96])
    print(f"Wrote {len(by_video)} review sheets to {args.output}")


if __name__ == "__main__": main()
