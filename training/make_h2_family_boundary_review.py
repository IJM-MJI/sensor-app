"""Create a compact atlas of remaining family-specific H2 boundary errors."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from make_h2_environment_review import image_at


CASES = (
    ("A", "test_2", 20.0, "reference 1-2"),
    ("A", "test_2", 25.0, "reference 2-3"),
    ("A", "test", 30.0, "ref 2-3 / predicted 1-2"),
    ("A", "test", 35.0, "ref 2-3 / predicted 1-2"),
    ("A", "test_3", 26.0, "ref 2-3 / predicted 1-2"),
    ("A", "test_3", 88.0, "upper remapped 2-3 / predicted 1-2"),
    ("B", "run3", 45.0, "reference 1-2"),
    ("B", "run3", 55.0, "ref 2-3 / predicted 1-2"),
    ("B", "run4", 65.0, "ref 1-2 / predicted 2-3"),
    ("B", "run4", 84.0, "reference 2-3 boundary"),
    ("B", "run4", 90.0, "reference 2-3"),
    ("B", "run3", 59.0, "late 2-3 / predicted 1-2"),
)


def panel(image, title, subtitle, width=430, height=430):
    scale = min(width / image.shape[1], (height - 64) / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale),
                                 round(image.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 245, np.uint8)
    x = (width - resized.shape[1]) // 2
    y = 60 + (height - 60 - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(canvas, title, (9, 24), cv2.FONT_HERSHEY_SIMPLEX, .56,
                (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (9, 48), cv2.FONT_HERSHEY_SIMPLEX, .43,
                (45, 45, 45), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    panels = []
    for family, run, seconds, note in CASES:
        image = image_at(args.video_root, run, seconds)
        panels.append(panel(image, f"Family {family} | {run} | t={seconds:g}s", note))
    rows = [np.hstack(panels[index:index + 3]) for index in range(0, len(panels), 3)]
    cv2.imwrite(str(args.output), np.vstack(rows))
    print(args.output)


if __name__ == "__main__":
    main()
