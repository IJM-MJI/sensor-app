"""Visual QA for the fixed semantic white/gray reference-patch cores."""

from pathlib import Path

import cv2
import numpy as np

from train_models import (
    CACHE_VERSION, frame_at, normalized_coordinates, read_csv,
    reference_patch_measurements, resize_for_app,
    register_coordinates_from_patches,
)


ROOT = Path(r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1")
CASES = [
    ("1_90_H2_only_test.mp4", 2), ("1_90_H2_only_test_2.mp4", 20),
    ("1_90_H2_only_test_3.MOV", 20), ("1_90_H2_only_4.mp4", 60),
    ("1_90_H2_only_5.mp4", 60), ("1_90_H2O_only_2_extract.mp4", 50),
    ("1_90_H2O_only.MOV", 10), ("1_90_H2O_only_extract_3min.mp4", 100),
]


def main():
    rows = read_csv(Path("training/cache") / CACHE_VERSION / "features.csv")
    tiles = []
    for video, seconds in CASES:
        video_rows = [row for row in rows if row["video"] == video]
        row = min(video_rows, key=lambda value: abs(float(value["time"]) - seconds))
        cap = cv2.VideoCapture(str(ROOT / video)); frame = frame_at(cap, seconds); cap.release()
        image = resize_for_app(frame)
        circle = tuple(int(float(row[name])) for name in ("circle_x", "circle_y", "circle_r"))
        orientation = int(float(row["orientation_quarters"]))
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        x, y, radius = circle
        yy, xx = np.ogrid[:lab.shape[0], :lab.shape[1]]
        chamber = (xx - x) ** 2 + (yy - y) ** 2 <= (radius * .9) ** 2
        nx, ny = normalized_coordinates(lab.shape, circle, orientation)
        nx, ny, registration = register_coordinates_from_patches(lab, chamber, nx, ny)
        masks, info = reference_patch_measurements(lab, chamber, nx, ny)
        overlay = image.copy(); overlay[masks["upper_right"]] = (0, 255, 255)
        overlay[masks["lower_left"]] = (255, 0, 255)
        image = cv2.addWeighted(image, .60, overlay, .40, 0)
        text = (f"{Path(video).stem} q={orientation} gap={info['brightness_gap']:.0f} "
                f"patch={int(info['reliable'])} reg={int(registration['reliable'])}")
        cv2.putText(image, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(image)
    height, width = 360, 480
    canvases = []
    for tile in tiles:
        scale = min(width / tile.shape[1], height / tile.shape[0])
        tile = cv2.resize(tile, (round(tile.shape[1] * scale), round(tile.shape[0] * scale)))
        canvas = np.zeros((height, width, 3), np.uint8); canvas[:tile.shape[0], :tile.shape[1]] = tile
        canvases.append(canvas)
    montage = np.vstack([np.hstack(canvases[index:index + 2]) for index in range(0, len(canvases), 2)])
    output = Path("training/output/reference_patch_montage.jpg")
    cv2.imwrite(str(output), montage, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(output.resolve())


if __name__ == "__main__":
    main()
