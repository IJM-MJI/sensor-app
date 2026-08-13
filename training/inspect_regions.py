"""Visual QA montage for flame/drop masks at low and high response."""

from pathlib import Path

import cv2
import numpy as np

from train_models import detect_circle, frame_at, resize_for_app, split_lower_by_y


ROOT = Path(r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1")
CASES = [
    ("RH fast 20", "1_90_H2O_only_2_extract.mp4", 4.5),
    ("RH fast 90", "1_90_H2O_only_2_extract.mp4", 100),
    ("RH daylight 90", "1_90_H2O_only.MOV", 4),
    ("RH daylight 20", "1_90_H2O_only.MOV", 34),
    ("RH long 20", "1_90_H2O_only_extract_3min.mp4", 7),
    ("RH long 70", "1_90_H2O_only_extract_3min.mp4", 150),
    ("RH long 80", "1_90_H2O_only_extract_extra.mp4", 50),
    ("H2 daylight 4", "1_90_H2_only_5.mp4", 60),
]


def tile(name: str, filename: str, seconds: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(ROOT / filename))
    frame = frame_at(cap, seconds)
    cap.release()
    image = resize_for_app(frame)
    x, y, r = detect_circle(frame)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    yy, xx = np.ogrid[:lab.shape[0], :lab.shape[1]]
    chamber = (xx - x) ** 2 + (yy - y) ** 2 <= round(r * .9) ** 2
    chroma = np.sqrt((lab[:, :, 1].astype(float) - 128) ** 2 + (lab[:, :, 2].astype(float) - 128) ** 2)
    high = chamber & (chroma > 15)
    ys, xs = np.where(high)
    lower = split_lower_by_y(ys.astype(float)) if len(ys) > 20 else np.zeros(len(ys), bool)
    overlay = image.copy()
    overlay[ys[~lower], xs[~lower]] = (30, 30, 255)  # flame red
    overlay[ys[lower], xs[lower]] = (255, 80, 20)    # droplet blue
    image = cv2.addWeighted(image, .55, overlay, .45, 0)
    cv2.circle(image, (x, y), r, (0, 255, 255), 2)
    cv2.putText(image, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    tiles = [tile(*case) for case in CASES]
    height, width = 360, 360
    canvas_tiles = []
    for image in tiles:
        scale = min(width / image.shape[1], height / image.shape[0])
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
        canvas = np.zeros((height, width, 3), np.uint8)
        canvas[: image.shape[0], : image.shape[1]] = image
        canvas_tiles.append(canvas)
    montage = np.vstack([np.hstack(canvas_tiles[i:i + 4]) for i in range(0, len(canvas_tiles), 4)])
    output = Path("training/output/region_montage.jpg")
    cv2.imwrite(str(output), montage)
    print(output.resolve())


if __name__ == "__main__":
    main()
