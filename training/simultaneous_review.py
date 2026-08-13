"""Build run-wise visual review sheets for questionable simultaneous labels.

The supplied timeline is treated as a guide.  For every independent run and RH
step this script selects the late-reaction frame used by validation, joins its
leave-one-run-out prediction, and renders the chamber in a common semantic
orientation (flame above droplet).  It does not change labels automatically.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from train_models import CACHE_VERSION, read_csv, resize_for_app


STATES = ("initial", "H2_only", "H2O_only", "simultaneous")
REVIEW_STATUS = {
    ("1_90_RH50_3.mp4", 68): "confirmed simultaneous: both shapes slightly darker",
    ("1_90_RH60_3.mp4", 64): "confirmed simultaneous: droplet darker than recovery at 147 s",
    ("1_90_RH70_3.mp4", 76): "confirmed simultaneous: both darker than recovery at 137 s",
    ("1_80_2.MOV", 292): "confirmed simultaneous: both shapes differ from initial",
    ("1_80_2.MOV", 656): "confirmed simultaneous: both shapes differ from initial",
    ("1_80_2.MOV", 840): "confirmed simultaneous: both shapes differ from initial",
    ("1_70_2.MOV", 452): "confirmed simultaneous: flame darker; droplet cooler-to-warmer",
    ("1_70_2.MOV", 648): "confirmed simultaneous: flame darker; droplet cooler-to-warmer",
    ("1_90_RH50_5_x2.mp4", 76): "confirmed simultaneous vs recovery; model missed weak flame chroma",
    ("1_90_RH60_5_x2.mp4", 68): "confirmed simultaneous vs recovery; model missed weak flame chroma",
}


def key(video: object, time: object) -> tuple[str, float]:
    return str(video), round(float(time), 3)


def rotate_crop(frame: np.ndarray, row: dict[str, object], size: int = 360) -> np.ndarray:
    frame = resize_for_app(frame)
    x, y, radius = (int(float(row[name])) for name in ("circle_x", "circle_y", "circle_r"))
    margin = int(radius * 1.18)
    padded = cv2.copyMakeBorder(frame, margin, margin, margin, margin, cv2.BORDER_CONSTANT)
    x, y = x + margin, y + margin
    crop = padded[y - margin:y + margin, x - margin:x + margin]
    quarter = int(float(row.get("orientation_quarters", 0))) % 4
    for _ in range(quarter):
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def frame_at(path: Path, seconds: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not decode {path.name} at {seconds:.2f}s")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path,
                        default=Path(f"training/cache/{CACHE_VERSION}/features.csv"))
    parser.add_argument("--angle-cache", type=Path,
                        default=Path(f"training/cache/{CACHE_VERSION}/angle_runs.csv"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("training/output/state_condition/best_direct_candidate_predictions.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/simultaneous_review"))
    args = parser.parse_args()

    rows = read_csv(args.cache)
    if args.angle_cache.exists():
        rows += read_csv(args.angle_cache)
    with args.predictions.open(encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    prediction_map = {key(row["video"], row["time"]): row for row in predictions}

    candidates: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("kind")) != "simultaneous" or row.get("rh_setpoint") is None:
            continue
        rh = float(row["rh_setpoint"])
        if not 20 <= rh <= 80:
            continue
        prediction = prediction_map.get(key(row["video"], row["time"]))
        if prediction and prediction["truth"] in ("H2_only", "simultaneous"):
            merged = dict(row)
            merged.update({name: prediction[name] for name in prediction if name.startswith("p_")})
            merged["prediction"] = prediction["prediction"]
            merged["truth"] = prediction["truth"]
            candidates[(str(row["group"]), rh)].append(merged)

    selected = []
    for (_, _), segment_rows in sorted(candidates.items()):
        # Median late-reaction frame is more robust than selecting the most
        # extreme error and is directly interpretable by a human reviewer.
        segment_rows.sort(key=lambda row: float(row["time"]))
        selected.append(segment_rows[len(segment_rows) // 2])

    args.output.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "group", "video", "rh_setpoint", "time", "truth", "prediction",
        "p_initial", "p_H2_only", "p_H2O_only", "p_simultaneous", "review",
    ]
    with (args.output / "review_index.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in selected:
            lookup = (str(row["video"]), round(float(row["time"])))
            row["review"] = REVIEW_STATUS.get(lookup, "")
            writer.writerow({name: row.get(name, "") for name in summary_fields})

    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        by_group[str(row["group"])].append(row)
    for group, group_rows in by_group.items():
        group_rows.sort(key=lambda row: float(row["rh_setpoint"]))
        panels = []
        for row in group_rows:
            path = args.video_root / str(row["video"])
            image = rotate_crop(frame_at(path, float(row["time"])), row)
            confidence = max(float(row[f"p_{state}"]) for state in STATES)
            correct = str(row["prediction"]) == str(row["truth"])
            color = (55, 180, 55) if correct else (40, 60, 230)
            cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), color, 7)
            lines = [
                f"RH{float(row['rh_setpoint']):.0f}  t={float(row['time']):.1f}s",
                f"pred={row['prediction']}  p={confidence:.2f}",
                str(row["video"]),
            ]
            for index, text in enumerate(lines):
                scale = .72 if index < 2 else .45
                y = 27 + index * 27
                cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(image)
        sheet = np.concatenate(panels, axis=1)
        safe_group = group.replace("/", "_")
        cv2.imwrite(str(args.output / f"{safe_group}.jpg"), sheet,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 94])

    # The two run-5 cases that looked RH-only to the model are shown directly
    # beside their own recovery-tail baselines for user adjudication.
    pending_panels = []
    for video, target_second in (("1_90_RH50_5_x2.mp4", 76), ("1_90_RH60_5_x2.mp4", 68)):
        video_rows = [row for row in rows if str(row["video"]) == video]
        target = min(video_rows, key=lambda row: abs(float(row["time"]) - target_second))
        recovery = max(video_rows, key=lambda row: float(row["time"]))
        for label, row in (("late reaction", target), ("recovery tail", recovery)):
            image = rotate_crop(frame_at(args.video_root / video, float(row["time"])), row)
            title = f"{video}  {label}  t={float(row['time']):.1f}s"
            cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), (80, 80, 80), 5)
            cv2.putText(image, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .48,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .48,
                        (255, 255, 255), 1, cv2.LINE_AA)
            pending_panels.append(image)
    cv2.imwrite(str(args.output / "run5_pending_vs_recovery.jpg"),
                np.concatenate(pending_panels, axis=1), [int(cv2.IMWRITE_JPEG_QUALITY), 96])

    print(f"Wrote {len(selected)} review frames in {len(by_group)} sheets to {args.output}")


if __name__ == "__main__":
    main()
