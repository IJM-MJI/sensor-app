"""Compare verified H2 2/3% endpoints with reviewed RH20 middle response."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from make_endpoint_mask_review import frame_at, source_path
from make_middle_endpoint_review import rotate_and_crop
from train_models import CACHE_VERSION, read_csv


H2_POINTS = {
    "1_90_H2_only_test.mp4": {"2%": 25.0, "3%": 30.0},
    "1_90_H2_only_test_2.mp4": {"2%": 21.0, "3%": 30.0},
    "1_90_H2_only_test_3.MOV": {"2%": 20.0, "3%": 28.0},
    "1_90_H2_only_4.mp4": {"2%": 30.0, "partial 2-3%": 121.5},
    "1_90_H2_only_5.mp4": {"2%": 13.0, "partial 2-3%": 129.5},
}

RH20_POINTS = {
    "1_90_RH20_3_x2_cropped.mp4": {"reviewed stage 2": 18.0, "reviewed stage 3": 44.0},
    "1_90_RH20_4_cropped.mp4": {"reviewed stage 2": 87.0, "reviewed stage 3": 114.5},
}


def enhanced_display(image: np.ndarray) -> np.ndarray:
    """Display-only mild local contrast and chroma boost; never a model input."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    channels = list(cv2.split(lab))
    channels[0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6, 6)).apply(channels[0])
    for index in (1, 2):
        value = channels[index].astype(np.float32)
        channels[index] = np.clip(128 + 1.25 * (value - 128), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge(channels), cv2.COLOR_LAB2BGR)


def put(image, text, xy, scale=.40, colour=(25, 25, 25), thickness=1):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--h2-cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_cropped_centered_v4.csv"))
    parser.add_argument("--rh20-cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/rh20_cropped_unlabeled_v1.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/h2_23_rh20_comparison_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    by_video = defaultdict(list)
    for row in read_csv(args.h2_cache) + read_csv(args.rh20_cache):
        by_video[str(row["video"])].append(row)

    sources = [("H2-only verified", video, points) for video, points in H2_POINTS.items()]
    sources += [("RH20 reviewed weak", video, points) for video, points in RH20_POINTS.items()]
    cells, records = [], []
    for source, video, points in sources:
        row_cells = []
        for label, seconds in points.items():
            cache_row = min(by_video[video], key=lambda row: abs(float(row["time"]) - seconds))
            actual = float(cache_row["time"])
            frame = frame_at(source_path(args.video_root, video), actual)
            circle = tuple(int(float(cache_row[key])) for key in ("circle_x", "circle_y", "circle_r"))
            quarters = int(float(cache_row["orientation_quarters"]))
            raw = rotate_and_crop(frame, circle, quarters, size=235)
            enhanced = enhanced_display(raw)
            sharpness = float(cv2.Laplacian(
                cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            caption = 92
            cell = np.full((235 + caption, 470, 3), 247, np.uint8)
            cell[:235, :235] = raw; cell[:235, 235:] = enhanced
            put(cell, "RAW", (8, 20), .48, (245, 245, 245), 1)
            put(cell, "DISPLAY ENHANCED (not ML input)", (244, 20), .40, (245, 245, 245), 1)
            put(cell, f"{source} | {label}", (8, 255), .43)
            put(cell, f"{video}  t={actual:.1f}s", (8, 275), .39)
            put(cell, f"calibrated flame dLAB=({float(cache_row['flame_L']):+.2f}, "
                f"{float(cache_row['flame_a']):+.2f}, {float(cache_row['flame_b']):+.2f})",
                (8, 295), .38)
            quality = "low-quality partial run" if "partial" in label else "reference/reviewed"
            put(cell, f"{quality} | sharpness={sharpness:.1f}", (8, 315), .39,
                (30, 30, 175) if "partial" in label else (30, 110, 30))
            row_cells.append(cell)
            records.append({"source": source, "video": video, "label": label, "time": actual,
                            "flame_L": cache_row["flame_L"], "flame_a": cache_row["flame_a"],
                            "flame_b": cache_row["flame_b"], "sharpness": sharpness,
                            "use": quality})
        cells.append(np.hstack(row_cells))

    header = np.full((76, 940, 3), 250, np.uint8)
    put(header, "H2 2% / 3% vs RH20 REVIEWED MIDDLE RESPONSE", (14, 29), .72, thickness=2)
    put(header, "Enhanced panels are human-review aids only; ML features remain calibrated raw LAB.",
        (14, 58), .46, (35, 35, 155))
    cv2.imwrite(str(args.output / "h2_23_rh20_comparison.jpg"),
                np.vstack([header, *cells]), [cv2.IMWRITE_JPEG_QUALITY, 97])
    with (args.output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(args.output / "h2_23_rh20_comparison.jpg")


if __name__ == "__main__":
    main()
