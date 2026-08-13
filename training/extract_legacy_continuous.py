"""Re-extract the existing continuous-label videos with the v4 phone pipeline."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

from train_models import FEATURE_NAMES, circle_track, corrected, extract_features, frame_at, video_info


ALIASES = {
    "1_90_3.MOV": "1_90_3.mp4",
    "1_90_H2_only_13.mp4": "1_90_H2_only_5.mp4",
    "1_H2_only_test.MOV": "1_90_H2_only_test.mp4",
    "1_H2_only_test_2.MOV": "1_90_H2_only_test_2.mp4",
    "1_H2_only_test_3.MOV": "1_90_H2_only_test_3.MOV",
}


def read_labels(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["video"]].append(row)
    return grouped


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("training/cache/v4-patch-balanced-shape-masks/legacy_continuous.csv"))
    args = parser.parse_args()
    grouped = read_labels(args.labels)
    cache_dir = args.output.parent / "legacy_clips"
    combined: list[dict[str, object]] = []
    for source_name, labels in grouped.items():
        actual_name = ALIASES.get(source_name, source_name)
        video_path = args.video_root / actual_name
        if not video_path.exists():
            print(f"SKIP missing {source_name} -> {actual_name}")
            continue
        cache_path = cache_dir / f"{actual_name.replace('.','_')}.csv"
        if cache_path.exists():
            with cache_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            print(f"{actual_name}: loaded {len(rows)} cached rows")
            combined.extend(rows)
            continue
        duration, _, _, _ = video_info(video_path)
        cap = cv2.VideoCapture(str(video_path))
        track = circle_track(cap, duration)
        rows: list[dict[str, object]] = []
        for label in sorted(labels, key=lambda row: float(row["time_sec"])):
            seconds = float(label["time_sec"])
            frame = frame_at(cap, seconds)
            if frame is None:
                continue
            circle = min(track, key=lambda item: abs(item[0] - seconds))[1]
            values = corrected(extract_features(frame, circle))
            row: dict[str, object] = {
                "video": actual_name,
                "source_video": source_name,
                "condition": label["condition"],
                "time": seconds,
                "phase": label["phase"],
                "h2_value": float(label["h2_pct"]),
                "rh_value": float(label["rh_pct"]),
            }
            row.update({name: values[name] for name in FEATURE_NAMES})
            rows.append(row)
        cap.release()
        write_rows(cache_path, rows)
        combined.extend(rows)
        print(f"{actual_name}: extracted {len(rows)} rows")
    write_rows(args.output, combined)
    print(f"wrote {len(combined)} rows from {len(set(str(r['video']) for r in combined))} videos -> {args.output}")


if __name__ == "__main__":
    main()
