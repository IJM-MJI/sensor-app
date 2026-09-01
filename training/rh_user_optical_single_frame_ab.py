"""Compare user-reviewed single-frame optical RH anchors.

The labels in this audit describe visually reviewed sensor response stages.  The
commanded chamber timeline is deliberately not used as supervision.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from rh_40_50_cross_run_spatial_analysis import extract
from rh_place1_external_validation import control_vector, nearest_row
from rh_place2_seven_band_run_holdout import LABELS, LEVELS, matrices
from train_models import CACHE_VERSION, read_csv


VIDEO_ROOT = Path(r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1")
OUTPUT = Path("training/output/rh_user_optical_single_frame_ab")
VIDEOS = {
    "response3": "1_90_H2O_only_3(response).mp4",
    "response6": "1_90_H2O_only_6(response).mp4",
    "daylight": "1_90_H2O_only.MOV",
}
CALIBRATION = {"response3": .5, "response6": 2.0, "daylight": 38.0}

# One reviewed still per optical stage.  Only the disputed 40--50 anchors vary.
BASE = {
    "response3": {25: 1.5, 35: 4.5, 45: 10.0, 55: 20.0,
                  65: 27.0, 75: 31.5, 85: 35.0},
    "response6": {25: 9.0, 35: 11.5, 45: 14.5, 55: 15.5,
                  65: 17.5, 75: 19.0, 85: 23.0},
}
SHIFTED = {
    **BASE,
    "response3": {**BASE["response3"], 45: 7.0},
    "response6": {**BASE["response6"], 45: 13.0},
}
CANDIDATES = {"current": BASE, "shift_40_50": SHIFTED}

# User-reviewed optical-stage labels for the independent daylight recording.
DAYLIGHT = (
    (4.0, 35),
    (7.0, 45), (9.0, 45), (13.0, 45),
    (17.0, 55),
    (21.0, 65),
    (25.0, 75), (27.0, 75),
)


def predict_1nn(train_x, train_y, test_x):
    scale = np.maximum(np.std(train_x, axis=0), .5)
    distance = np.sqrt(np.mean(
        ((test_x[:, None] - train_x[None]) / scale) ** 2, axis=2))
    order = np.argsort(distance, axis=1)
    prediction = train_y[order[:, 0]]
    margin = distance[np.arange(len(test_x)), order[:, 1]] - distance[np.arange(len(test_x)), order[:, 0]]
    return prediction, distance[np.arange(len(test_x)), order[:, 0]], margin


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = read_csv(Path(f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    requested = {group: {CALIBRATION[group]} for group in VIDEOS}
    for candidate in CANDIDATES.values():
        for group, levels in candidate.items():
            requested[group].update(levels.values())
    requested["daylight"].update(time for time, _ in DAYLIGHT)

    items = []
    for group, times in requested.items():
        for seconds in sorted(times):
            video = VIDEOS[group]
            items.append({"group": group, "video": video, "time": seconds,
                          "row": nearest_row(cache, video, seconds)})
    summaries = extract(items, VIDEO_ROOT)
    raw = {(item["group"], item["time"]): control_vector(summary)
           for item, summary in zip(items, summaries)}
    vectors = {
        key: value - raw[(key[0], CALIBRATION[key[0]])]
        for key, value in raw.items()
    }

    daylight_x = np.asarray([vectors[("daylight", time)] for time, _ in DAYLIGHT])
    daylight_y = np.asarray([level for _, level in DAYLIGHT], dtype=int)
    payload = {
        "scope": "single-frame user-reviewed optical stages; commanded RH timeline excluded",
        "feature_warning": (
            "Legacy registered_drop_v2 audit features; daylight results must not be "
            "treated as v41 browser accuracy until exact app features are reproduced."
        ),
        "calibration_s": CALIBRATION,
        "daylight_review": [{"time_s": time, "level": level}
                            for time, level in DAYLIGHT],
        "candidates": {},
    }
    prediction_rows = []
    for name, candidate in CANDIDATES.items():
        train_rows = [(group, level, time)
                      for group, levels in candidate.items()
                      for level, time in levels.items()]
        train_x = np.asarray([vectors[(group, time)] for group, _, time in train_rows])
        train_y = np.asarray([level for _, level, _ in train_rows], dtype=int)

        external_pred, external_dist, external_margin = predict_1nn(
            train_x, train_y, daylight_x)
        folds = []
        for held in ("response3", "response6"):
            train = np.asarray([group != held for group, _, _ in train_rows])
            test = ~train
            pred, _, _ = predict_1nn(train_x[train], train_y[train], train_x[test])
            folds.append({"held_out_run": held, **matrices(train_y[test], pred)})

        separation = {}
        for group in ("response3", "response6"):
            x45 = vectors[(group, candidate[group][45])]
            x55 = vectors[(group, candidate[group][55])]
            scale = np.maximum(np.std(train_x, axis=0), .5)
            separation[group] = float(np.sqrt(np.mean(((x45 - x55) / scale) ** 2)))

        payload["candidates"][name] = {
            "anchors_s": candidate,
            "external_daylight": matrices(daylight_y, external_pred),
            "complete_run_holdout": folds,
            "standardized_40_50_to_50_60_distance": separation,
        }
        for (time, truth), vector, pred, distance, margin in zip(
                DAYLIGHT, daylight_x, external_pred, external_dist, external_margin):
            prediction_rows.append({
                "candidate": name, "time_s": time, "reference": int(truth),
                "prediction": int(pred), "correct": bool(truth == pred),
                "distance": float(distance), "margin": float(margin),
                "delta_L": float(vector[0]), "delta_a": float(vector[1]),
                "delta_b": float(vector[2]),
            })

    (OUTPUT / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUTPUT / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader(); writer.writerows(prediction_rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
