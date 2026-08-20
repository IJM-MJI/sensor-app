"""Create side-by-side review sheets for uncertain RH20 H2 stage boundaries."""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import cv2
import numpy as np

from ordinal_concentration_analysis import assign_rh20_h2_weak_targets
from train_models import CACHE_VERSION, read_csv


REFERENCE_TIMES = {0: 4.0, 1: 13.0, 2: 21.0, 3: 30.0, 4: 51.0}


def frame_at(path: Path, seconds: float):
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read(); cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read {path} at {seconds:.1f}s")
    return frame


def tile(frame, label, size=260):
    h, w = frame.shape[:2]; scale = min(size / w, (size - 36) / h)
    resized = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    y = 36 + (size - 36 - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(canvas, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 1,
                cv2.LINE_AA)
    return canvas


def closest(rows, boundary):
    return min(rows, key=lambda row: abs(float(row["continuous_target"]) - boundary))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--normal-cache", type=Path,
                        default=Path(f"training/cache/{CACHE_VERSION}/rh20_cropped_unlabeled_v1.csv"))
    parser.add_argument("--x2-cache", type=Path,
                        default=Path(f"training/cache/{CACHE_VERSION}/rh20_cropped_unlabeled_run5_x2_v2.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/rh20_boundary_review_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    normal = read_csv(args.normal_cache)
    x2 = [row for row in read_csv(args.x2_cache)
          if str(row["video"]) == "1_90_RH20_5_x2_cropped.mp4"]
    source = normal + x2
    time_rows, optical_rows = copy.deepcopy(source), copy.deepcopy(source)
    assign_rh20_h2_weak_targets(time_rows, .05, "time")
    assign_rh20_h2_weak_targets(optical_rows, .05, "optical")
    by_video_time, by_video_optical = {}, {}
    for row in time_rows:
        if row.get("analysis_phase") == "reaction":
            by_video_time.setdefault(str(row["video"]), []).append(row)
    for row in optical_rows:
        if row.get("analysis_phase") == "reaction":
            by_video_optical.setdefault(str(row["video"]), []).append(row)

    reference = args.video_root / "1_90_H2_only_test_2_cropped.mp4"
    audit = []
    master_sections = []
    for video in sorted(by_video_time):
        time_path, optical_path = by_video_time[video], by_video_optical[video]
        maximum = min(max(float(row["continuous_target"]) for row in time_path),
                      max(float(row["continuous_target"]) for row in optical_path))
        boundaries = [value for value in (.5, 1.5, 2.5, 3.5) if value <= maximum + .05]
        strips = []
        for boundary in boundaries:
            lower, upper = int(boundary - .5), int(boundary + .5)
            timed, optical = closest(time_path, boundary), closest(optical_path, boundary)
            time_t, optical_t = float(timed["time"]), float(optical["time"])
            strips.append(np.hstack([
                tile(frame_at(reference, REFERENCE_TIMES[lower]), f"REF {lower}%  t={REFERENCE_TIMES[lower]:.1f}s"),
                tile(frame_at(args.video_root / video, time_t), f"TIME boundary  t={time_t:.1f}s"),
                tile(frame_at(args.video_root / video, optical_t), f"OPTICAL boundary  t={optical_t:.1f}s"),
                tile(frame_at(reference, REFERENCE_TIMES[upper]), f"REF {upper}%  t={REFERENCE_TIMES[upper]:.1f}s"),
            ]))
            audit.append({"video": video, "lower_h2": lower, "upper_h2": upper,
                          "time_boundary_s": time_t, "optical_boundary_s": optical_t,
                          "difference_s": optical_t - time_t})
        if not strips:
            continue
        body = np.vstack(strips)
        header = np.full((44, body.shape[1], 3), 225, dtype=np.uint8)
        cv2.putText(header, video, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, .72,
                    (15, 15, 15), 2, cv2.LINE_AA)
        sheet = np.vstack([header, body]); master_sections.append(sheet)
        cv2.imwrite(str(args.output / f"{Path(video).stem}_boundary_review.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
    master = np.vstack(master_sections)
    cv2.imwrite(str(args.output / "all_rh20_boundary_review.jpg"), master,
                [cv2.IMWRITE_JPEG_QUALITY, 94])
    with (args.output / "boundary_candidates.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0])); writer.writeheader(); writer.writerows(audit)
    print(f"wrote {len(audit)} boundary comparisons for {len(by_video_time)} videos -> {args.output}")


if __name__ == "__main__":
    main()
