"""Compare v41 single-frame colour-path features across three environments."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from rh_place2_seven_band_run_holdout import matrices
from rh_v41_still_feature_audit import feature, resize, select_circle


ROOT = Path(r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1")
OUTPUT = Path("training/output/rh_v41_color_path_feature_ab")
VIDEOS = {
    "response3": ("1_90_H2O_only_3(response)_cropped.mp4", .5, None),
    "response6": ("1_90_H2O_only_6(response)_cropped.mp4", 2.0, None),
    "daylight": ("1_90_H2O_only_cropped.mp4", 38.0, None),
}
CURRENT = {
    "response3": {25: 1.5, 35: 4.5, 45: 10.0, 55: 20.0, 65: 27.0, 75: 31.5, 85: 35.0},
    "response6": {25: 9.0, 35: 11.5, 45: 14.5, 55: 15.5, 65: 17.5, 75: 19.0, 85: 23.0},
}
SHIFTED = {
    **CURRENT,
    "response3": {**CURRENT["response3"], 45: 7.0},
    "response6": {**CURRENT["response6"], 45: 13.0},
}
POLICIES = {"current": CURRENT, "shift_40_50": SHIFTED}
DAYLIGHT = {35: (4.0,), 45: (7.0, 9.0, 13.0), 55: (17.0,),
            65: (21.0,), 75: (25.0, 27.0)}
FEATURES = {
    "tight_lab": ("tight_L", "tight_a", "tight_b"),
    "flame_ab": ("flame_a", "flame_b"),
    "flame_lab": ("flame_L", "flame_a", "flame_b"),
    "drop_ab": ("drop_a", "drop_b"),
    "flame_drop_ab": ("flame_a", "flame_b", "drop_a", "drop_b"),
    "flame_tight": ("flame_a", "flame_b", "tight_L", "tight_a", "tight_b"),
    "drop_minus_flame_ab": ("drop_flame_a", "drop_flame_b"),
    "registered_minus_flame_ab": ("registered_flame_a", "registered_flame_b"),
    "tight_minus_flame_ab": ("tight_flame_a", "tight_flame_b"),
    "relative_colour_path": ("drop_flame_a", "drop_flame_b",
                             "registered_flame_a", "registered_flame_b"),
    "drop_hue_direction": ("drop_unit_a", "drop_unit_b"),
    "registered_hue_direction": ("registered_unit_a", "registered_unit_b"),
    "flame_hue_direction": ("flame_unit_a", "flame_unit_b"),
    "dual_hue_direction": ("drop_unit_a", "drop_unit_b",
                           "flame_unit_a", "flame_unit_b"),
}


def decoded_frame(cap, seconds, rotation):
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot decode frame at {seconds:.2f}s")
    frame = frame if rotation is None else cv2.rotate(frame, rotation)
    # Reproduce the near-square saved-frame crops used in the app review. The
    # source mp4 remains 16:9 even though the reviewed PNG keeps the centred
    # chamber area only.
    h, w = frame.shape[:2]; side = min(h, w)
    x0 = (w - side) // 2; y0 = (h - side) // 2
    return frame[y0:y0 + side, x0:x0 + side]


def run_vectors(times):
    output = {}
    for group, (filename, calibration_time, rotation) in VIDEOS.items():
        cap = cv2.VideoCapture(str(ROOT / filename))
        cal_frame, circle, source = select_circle(decoded_frame(cap, calibration_time, rotation))
        cal_tight, registration, _, _, _, cal_diag = feature(cal_frame, circle)
        output[group] = {"_roi": {"circle": circle, "source": source}}
        for seconds in sorted(times[group]):
            frame = resize(decoded_frame(cap, seconds, rotation))
            h, w = frame.shape[:2]; ch, cw = cal_frame.shape[:2]
            locked = (round(circle[0] / cw * w), round(circle[1] / ch * h),
                      round(circle[2] / min(ch, cw) * min(h, w)))
            tight, _, _, _, _, diag = feature(frame, locked, registration)
            values = {"tight_L": float(tight[0] - cal_tight[0]),
                      "tight_a": float(tight[1] - cal_tight[1]),
                      "tight_b": float(tight[2] - cal_tight[2])}
            for key, value in diag.items():
                values[key] = float(value - cal_diag[key])
            for prefix in ("drop", "registered"):
                for channel in ("a", "b"):
                    values[f"{prefix}_flame_{channel}"] = (
                        values[f"{prefix}_{channel}"] - values[f"flame_{channel}"])
            for channel in ("a", "b"):
                values[f"tight_flame_{channel}"] = (
                    values[f"tight_{channel}"] - values[f"flame_{channel}"])
            output[group][seconds] = values
        cap.release()
    return output


def predict(train_x, train_y, test_x):
    scale = np.maximum(np.std(train_x, axis=0), .5)
    distance = np.sqrt(np.mean(((test_x[:, None] - train_x[None]) / scale) ** 2, axis=2))
    return train_y[np.argmin(distance, axis=1)]


def score(truth, prediction):
    result = matrices(truth, prediction)
    present = sorted(set(map(int, truth)))
    recalls = [float(np.mean(prediction[truth == level] == level)) for level in present]
    result["present_balanced_accuracy"] = float(np.mean(recalls))
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    requested = {group: {calibration} for group, (_, calibration, _) in VIDEOS.items()}
    for policy in POLICIES.values():
        for group, levels in policy.items():
            requested[group].update(levels.values())
    requested["daylight"].update(t for times in DAYLIGHT.values() for t in times)
    vectors = run_vectors(requested)
    payload = {"scope": "v41 single-frame, user-reviewed optical stages",
               "roi": {group: values["_roi"] for group, values in vectors.items()},
               "policies": {}}
    for policy_name, policy in POLICIES.items():
        payload["policies"][policy_name] = {}
        train_rows = [(group, level, seconds) for group, levels in policy.items()
                      for level, seconds in levels.items()]
        daylight_rows = [(level, seconds) for level, times in DAYLIGHT.items() for seconds in times]
        for feature_name, names in FEATURES.items():
            train_x = np.asarray([[vectors[group][seconds][name] for name in names]
                                  for group, _, seconds in train_rows])
            train_y = np.asarray([level for _, level, _ in train_rows], dtype=int)
            groups = np.asarray([group for group, _, _ in train_rows])
            folds = []
            for held in ("response3", "response6"):
                use_train = groups != held; use_test = ~use_train
                folds.append({"held_out": held, **score(
                    train_y[use_test], predict(train_x[use_train], train_y[use_train], train_x[use_test]))})
            daylight_x = np.asarray([[vectors["daylight"][seconds][name] for name in names]
                                     for _, seconds in daylight_rows])
            daylight_y = np.asarray([level for level, _ in daylight_rows], dtype=int)
            daylight_prediction = predict(train_x, train_y, daylight_x)
            payload["policies"][policy_name][feature_name] = {
                "features": names, "folds": folds,
                "daylight": score(daylight_y, daylight_prediction),
                "daylight_predictions": [
                    {"time_s": seconds, "reference": int(level), "prediction": int(prediction)}
                    for (level, seconds), prediction in zip(daylight_rows, daylight_prediction)
                ],
            }
    (OUTPUT / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for policy, candidates in payload["policies"].items():
        for name, result in candidates.items():
            print(policy, name,
                  "folds", [round(fold["exact_accuracy"], 3) for fold in result["folds"]],
                  "daylight", round(result["daylight"]["exact_accuracy"], 3),
                  "daylight_bal", round(result["daylight"]["present_balanced_accuracy"], 3))


if __name__ == "__main__":
    main()
