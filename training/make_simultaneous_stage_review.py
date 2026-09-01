"""Render simultaneous H2 stage-review sheets without assigning time labels.

Each row is one nominal-RH clip and each column is only a search candidate at a
fixed fraction of the known reaction window.  Fractions are deliberately not
converted to H2 concentrations: a reviewer must match the flame appearance to
the independently reviewed H2-only optical stages.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from simultaneous_review import frame_at, rotate_crop
from train_models import CACHE_VERSION, read_csv, simultaneous_clips


FRACTIONS = tuple(np.linspace(0.0, 1.0, 9))


def put(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = .42) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def nearest_row(rows: list[dict[str, object]], seconds: float) -> dict[str, object]:
    return min(rows, key=lambda row: abs(float(row["time"]) - seconds))


def tile(path: Path, seconds: float, feature_row: dict[str, object], size: int) -> np.ndarray:
    image = rotate_crop(frame_at(path, seconds), feature_row, size=size)
    footer = np.full((38, size, 3), 28, np.uint8)
    put(footer, f"t={seconds:.1f}s", (8, 25), .48)
    return np.vstack((image, footer))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path,
                        default=Path(f"training/cache/{CACHE_VERSION}/features.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/simultaneous_stage_review"))
    parser.add_argument("--tile-size", type=int, default=170)
    args = parser.parse_args()

    feature_rows = read_csv(args.cache)
    by_video: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in feature_rows:
        if str(row.get("kind")) == "simultaneous":
            by_video[str(row["video"])].append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    by_run = defaultdict(list)
    for clip in simultaneous_clips():
        if clip.rh is not None and clip.rh <= 80:
            by_run[clip.group].append(clip)

    for group, clips in sorted(by_run.items()):
        clips.sort(key=lambda clip: float(clip.rh or 0))
        rows = []
        for clip in clips:
            cached = by_video.get(clip.name, [])
            if not cached:
                raise RuntimeError(f"No cached ROI rows for {clip.name}")
            start, end = float(clip.reaction_start), float(clip.reaction_end)
            panels = []
            for fraction in FRACTIONS:
                seconds = start + fraction * (end - start)
                panels.append(tile(args.video_root / clip.name, seconds,
                                   nearest_row(cached, seconds), args.tile_size))
                index_rows.append({
                    "group": group, "video": clip.name,
                    "nominal_rh_metadata": int(clip.rh),
                    "candidate_fraction": f"{fraction:.3f}",
                    "candidate_time_s": f"{seconds:.3f}",
                    "reviewed_h2_stage": "", "review_note": "",
                })
            row = np.hstack(panels)
            put(row, f"RH{int(clip.rh)} metadata | {clip.name}", (8, 20), .45)
            rows.append(row)

        header = np.full((52, args.tile_size * len(FRACTIONS), 3), 28, np.uint8)
        for column, fraction in enumerate(FRACTIONS):
            put(header, f"search {fraction * 100:.1f}%", (column * args.tile_size + 8, 31), .42)
        sheet = np.vstack([header, *rows])
        cv2.imwrite(str(args.output / f"{group}_stage_search.jpg"), sheet,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    fields = ("group", "video", "nominal_rh_metadata", "candidate_fraction",
              "candidate_time_s", "reviewed_h2_stage", "review_note")
    with (args.output / "candidate_index.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index_rows)

    readme = args.output / "README.txt"
    readme.write_text(
        "Columns are search positions, not H2 labels.\n"
        "For each RH row, record the candidate time whose flame most closely matches "
        "the reviewed H2-only 0/1/2/3/4% optical stage.\n"
        "Do not infer a concentration by linearly interpolating elapsed time.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(by_run)} sheets and {len(index_rows)} candidates to {args.output}")


if __name__ == "__main__":
    main()
