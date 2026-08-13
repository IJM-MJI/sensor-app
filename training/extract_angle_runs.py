"""Re-extract the user-timed 70/80-degree run-2 videos with rotation-aware ROIs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from train_models import FEATURE_NAMES, corrected, extract_features, recovery_tail_start, video_info, write_csv


RUNS = {
    "1_80_2.MOV": {
        "group": "angle-80-run2", "circle": (176, 300, 50),
        "timeline": [
            (20, 8, 138, 138, 173), (30, 173, 308, 308, 364),
            (40, 364, 494, 494, 548), (50, 548, 670, 670, 720),
            (60, 720, 854, 854, 916), (70, 916, 1038, 1038, 1081),
            (80, 1081, 1227, 1227, 1279), (90, 1279, 1384, 1384, 1441),
        ],
    },
    "1_70_2.MOV": {
        "group": "angle-70-run2", "circle": (52, 292, 66),
        "timeline": [
            (20, 8, 128, 128, 185), (30, 185, 262, 262, 370),
            (40, 370, 462, 462, 540), (50, 540, 663, 663, 709),
            (60, 709, 833, 854, 894),  # 13:53-14:14 intentionally absent
            (70, 894, 1002, 1002, 1095), (80, 1095, 1249, 1249, 1326),
            (90, 1326, 1465, 1465, 1543),
        ],
    },
}


def label_at(t: float, timeline):
    for rh, reaction_start, reaction_end, recovery_start, recovery_end in timeline:
        reaction_span = reaction_end - reaction_start
        if reaction_start + .78 * reaction_span <= t <= reaction_end - 2:
            if rh >= 90:
                return None, None, "simultaneous_rh90_saturated", float(rh)
            if rh == 20:
                return 1, 0, "h2_only_condition", float(rh)
            return 1, None, "simultaneous_condition", float(rh)
        if recovery_tail_start(recovery_start, recovery_end) <= t <= recovery_end - 1:
            return 0, None, "baseline_recovery", 20.0
    return None, None, None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=.5)
    parser.add_argument("--output", type=Path, default=Path("training/cache/v7-verified-orientation-recovery-tail/angle_runs.csv"))
    args = parser.parse_args()
    combined = []
    for name, config in RUNS.items():
        path = args.video_root / name
        duration, fps, width, height = video_info(path)
        cap = cv2.VideoCapture(str(path))
        # Both user-reviewed 70/80-degree run-2 recordings have the flame on the
        # right; rotate 90 degrees counter-clockwise to place it at the top.
        orientation_lock, orientation_lock_confidence = 1, 1.0
        raw = []
        every = max(1, round(fps / args.sample_hz))
        index = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if index % every == 0:
                ok, frame = cap.retrieve()
                if ok:
                    t = index / fps
                    values = corrected(extract_features(frame, config["circle"], orientation_lock))
                    raw.append((t, values, orientation_lock, 1.0))
            index += 1
        cap.release()
        baseline_rows = [values for t, values, _, _ in raw if 0 <= t <= 6]
        baseline = {key: float(np.median([row[key] for row in baseline_rows])) for key in FEATURE_NAMES}
        for t, values, orientation, orientation_confidence in raw:
            h2_present, rh_high, state, rh_setpoint = label_at(t, config["timeline"])
            row = {
                "video": name, "group": config["group"], "kind": "simultaneous",
                "time": t, "duration": duration, "width": width, "height": height,
                "h2_present": h2_present, "rh_high": rh_high, "state": state,
                "h2_value": None, "rh_value": None, "rh_setpoint": rh_setpoint,
                "circle_x": config["circle"][0], "circle_y": config["circle"][1], "circle_r": config["circle"][2],
                "orientation_quarters": orientation,
                "orientation_confidence": orientation_confidence,
            }
            row.update({key: float(values[key] - baseline[key]) for key in FEATURE_NAMES})
            combined.append(row)
        print(
            f"{name}: extracted {len(raw)} frames, fixed ROI={config['circle']}, "
            f"orientation={orientation_lock} ({orientation_lock_confidence:.2f})"
        )
    write_csv(args.output, combined)
    print(f"wrote {len(combined)} rows -> {args.output}")


if __name__ == "__main__":
    main()
