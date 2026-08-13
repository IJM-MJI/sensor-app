"""Create a review montage of high-confidence four-state classification errors."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


ALIASES = {
    "1_90_H2_only_13.mp4": "1_90_H2_only_5.mp4",
    "1_H2_only_test.MOV": "1_90_H2_only_test.mp4",
    "1_H2_only_test_2.MOV": "1_90_H2_only_test_2.mp4",
    "1_H2_only_test_3.MOV": "1_90_H2_only_test_3.MOV",
}


def fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
    canvas = np.full((height, width, 3), 245, np.uint8)
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("training/output/state_condition/best_direct_candidate_predictions.csv"))
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("training/output/state_condition/state_error_montage.jpg"))
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args()
    with args.predictions.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    probability_columns = [name for name in rows[0] if name.startswith("p_")]
    errors = [row for row in rows if row["truth"] != row["prediction"]]
    for row in errors:
        row["confidence"] = max(float(row[name]) for name in probability_columns)
    # At most one frame per group/truth/prediction combination prevents one long
    # recording from filling the entire montage.
    selected, seen = [], set()
    for row in sorted(errors, key=lambda item: float(item["confidence"]), reverse=True):
        key = (row["group"], row["truth"], row["prediction"])
        if key in seen:
            continue
        name = ALIASES.get(row["video"], row["video"])
        path = args.video_root / name
        if not path.exists():
            continue
        row["path"] = path
        selected.append(row); seen.add(key)
        if len(selected) >= args.count:
            break

    tile_w, tile_h, caption_h, columns = 360, 220, 56, 4
    rows_count = (len(selected) + columns - 1) // columns
    montage = np.full((rows_count * (tile_h + caption_h), columns * tile_w, 3), 255, np.uint8)
    for index, row in enumerate(selected):
        cap = cv2.VideoCapture(str(row["path"]))
        cap.set(cv2.CAP_PROP_POS_MSEC, float(row["time"]) * 1000)
        ok, frame = cap.read(); cap.release()
        if not ok:
            continue
        tile = fit(frame, tile_w, tile_h)
        y, x = divmod(index, columns)
        y0, x0 = y * (tile_h + caption_h), x * tile_w
        montage[y0:y0 + tile_h, x0:x0 + tile_w] = tile
        cv2.putText(montage, f"{row['video']}  t={float(row['time']):.1f}s", (x0 + 5, y0 + tile_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(montage, f"truth={row['truth']}  pred={row['prediction']}  p={float(row['confidence']):.2f}",
                    (x0 + 5, y0 + tile_h + 40), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 180), 1, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), montage, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"wrote {len(selected)} reviewed errors -> {args.output}")


if __name__ == "__main__":
    main()
