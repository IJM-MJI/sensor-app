"""Render representative detected chamber circles for visual QA."""

from pathlib import Path

import cv2
import numpy as np

from train_models import detect_circle, frame_at, resize_for_app


ROOT = Path(r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1")
FILES = [
    "1_90_H2_only_test_2.mp4", "1_90_H2_only_test_3.MOV",
    "1_90_H2O_only_2_extract.mp4", "1_90_H2O_only_extract_3min.mp4",
    "1_90_RH30_2.mp4", "1_90_RH30_3_x2.mp4",
    "1_90_RH30_4.mp4", "1_90_RH30_5_x2.mp4",
]


def main() -> None:
    tiles = []
    for name in FILES:
        cap = cv2.VideoCapture(str(ROOT / name))
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
        frame = frame_at(cap, min(5, duration / 5))
        cap.release()
        if frame is None:
            continue
        small = resize_for_app(frame)
        x, y, r = detect_circle(frame)
        cv2.circle(small, (x, y), r, (0, 0, 255), 3)
        cv2.circle(small, (x, y), round(r * 0.9), (0, 255, 255), 2)
        cv2.putText(small, f"{name} r={r}", (8, small.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(small, f"{name} r={r}", (8, small.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(small)
    width = 480
    height = max(t.shape[0] for t in tiles)
    padded = []
    for tile in tiles:
        canvas = np.zeros((height, width, 3), np.uint8)
        canvas[:tile.shape[0], :tile.shape[1]] = tile
        padded.append(canvas)
    montage = np.vstack([np.hstack(padded[i:i + 2]) for i in range(0, len(padded), 2)])
    out = Path("training/output/roi_montage.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), montage)
    print(out.resolve())


if __name__ == "__main__":
    main()
