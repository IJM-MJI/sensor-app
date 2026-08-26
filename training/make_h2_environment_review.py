"""Build a human-review atlas and calibration-only H2 environment clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from h2_more_crop_fixed_mask_analysis import (RUNS as TIGHT_RUNS, canonical,
                                               content_crop, frame_at)
from h2_other_run_reference_matching import OTHER_RUNS, resize_roi
from h2_rh20_max_response_analysis import RUNS as RH20_RUNS


RUNS = {
    "test_2": ("tight", 2.0, 30.0),
    "test_3": ("tight", 2.0, 28.0),
    "test": ("other", 2.0, 30.0),
    "run2": ("rh20", 2.0, 55.0),
    "run3": ("rh20", 1.0, 55.0),
    "run4": ("rh20", 1.0, 90.0),
    "run5_x2": ("rh20", 1.0, 42.0),
    "run5_normal": ("rh20", 1.0, 80.0),
}


def image_at(root: Path, run: str, seconds: float) -> np.ndarray:
    kind = RUNS[run][0]
    if kind == "tight":
        config = TIGHT_RUNS[run]
        cap = cv2.VideoCapture(str(root / config["file"]))
        calibration = frame_at(cap, 2.0)
        _, bounds = content_crop(calibration)
        image = canonical(frame_at(cap, seconds), bounds)
    elif kind == "other":
        config = OTHER_RUNS[run]
        cap = cv2.VideoCapture(str(root / config["file"]))
        image = resize_roi(frame_at(cap, seconds), config["roi"])
    else:
        config = RH20_RUNS[run]
        cap = cv2.VideoCapture(str(root / config["file"]))
        image = resize_roi(frame_at(cap, seconds), config["roi"])
    cap.release()
    return image


def environment_feature(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    crop = image[int(.08 * height):int(.92 * height),
                 int(.08 * width):int(.92 * width)]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    values = []
    for pixels in (lab, hsv):
        for channel in range(3):
            vector = pixels[:, channel]
            values.extend(np.percentile(vector, (10, 25, 50, 75, 90)))
            values.extend((np.mean(vector), np.std(vector)))
    return np.asarray(values)


def panel(image: np.ndarray, title: str, width=390, height=430) -> np.ndarray:
    scale = min(width / image.shape[1], (height - 42) / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale),
                                 round(image.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 245, np.uint8)
    x = (width - resized.shape[1]) // 2
    y = 38 + (height - 38 - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(canvas, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, .58,
                (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    initial, response, features = {}, {}, []
    for run, (_, initial_time, response_time) in RUNS.items():
        initial[run] = image_at(args.video_root, run, initial_time)
        response[run] = image_at(args.video_root, run, response_time)
        features.append(environment_feature(initial[run]))
    names = list(RUNS)
    scaled = StandardScaler().fit_transform(np.asarray(features))
    clusters = {}
    for count in (2, 3, 4):
        labels = AgglomerativeClustering(n_clusters=count, linkage="ward").fit_predict(scaled)
        clusters[str(count)] = {
            f"group_{label + 1}": [names[i] for i in range(len(names)) if labels[i] == label]
            for label in sorted(set(labels))
        }
    payload = {
        "basis": "calibration-frame Lab/HSV distribution only",
        "automatic_candidates": clusters,
        "review_note": "response frame is shown for context but is not used in clustering",
    }
    (args.output / "environment_candidates.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for run in names:
        _, initial_time, response_time = RUNS[run]
        rows.append(np.hstack((
            panel(initial[run], f"{run}  INITIAL  t={initial_time:g}s"),
            panel(response[run], f"{run}  RESPONSE CHECK  t={response_time:g}s"),
        )))
    cv2.imwrite(str(args.output / "h2_environment_review_atlas.jpg"), np.vstack(rows))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
