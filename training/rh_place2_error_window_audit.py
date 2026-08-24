"""Audit temporal neighbourhoods of the two place-2 band errors."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from make_endpoint_mask_review import source_path
from rh_40_50_cross_run_spatial_analysis import build, extract
from rh_four_band_analysis import DISPLAY, STAGES, band
from rh_paired_pixel_hue_analysis import balanced_frame_and_masks, endpoint_rows
from rh_place2_pairwise_band_analysis import FIXED_C
from rh_tight_relative_analysis import tight_masks
from train_models import CACHE_VERSION, read_csv, resize_for_app


WINDOWS = (
    {"group": "rh-response-3", "video": "1_90_H2O_only_3(response).mp4",
     "stage": 40.0, "endpoint": 5.0, "start": 4.0, "end": 6.0,
     "train_group": "rh-response-6"},
    {"group": "rh-response-6", "video": "1_90_H2O_only_6(response).mp4",
     "stage": 70.0, "endpoint": 18.0, "start": 17.0, "end": 19.0,
     "train_group": "rh-response-3"},
)


def nominal_rh(window, seconds):
    if window["group"] == "rh-response-3":
        return 30 + (seconds - 3) / 2 * 10 if seconds <= 5 else 40 + (seconds - 5) / 2 * 10
    return 60 + (seconds - 16) / 2 * 10 if seconds <= 18 else 70 + (seconds - 18) / 2 * 10


def candidate_items(cache, endpoints, window):
    by_video = [row for row in cache if row.get("video") == window["video"]]
    baseline = [item for item in endpoints
                if item["group"] == window["group"] and item["stage"] == 25]
    training = [item for item in endpoints if item["group"] == window["train_group"]]
    candidates = []; seen = set()
    for seconds in np.arange(window["start"], window["end"] + 1e-6, .25):
        row = min(by_video, key=lambda value: abs(float(value["time"]) - seconds))
        key = round(float(row["time"]), 6)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"video": window["video"], "group": window["group"],
                           "time": float(row["time"]), "requested_time": float(seconds),
                           "stage": window["stage"], "row": row, "candidate": True})
    return training + baseline + candidates, candidates


def predict_window(cache, endpoints, video_root, window):
    items, candidates = candidate_items(cache, endpoints, window)
    summaries = extract(items, video_root)
    matrices, audit = build(items, summaries, STAGES)
    groups = np.asarray([row["group"] for row in audit])
    truth = np.asarray([band(row["reference"]) for row in audit])
    train = groups == window["train_group"]
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=FIXED_C, class_weight="balanced", max_iter=5000, random_state=42))
    model.fit(matrices["background_control"][train], truth[train])
    probabilities = model.predict_proba(matrices["background_control"])
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    rows = []
    candidate_times = {round(item["time"], 6) for item in candidates}
    for audit_row, prediction, probability in zip(audit, predictions, probabilities):
        if audit_row["group"] != window["group"] or round(audit_row["time"], 6) not in candidate_times:
            continue
        rows.append({"group": window["group"], "video": window["video"],
                     "time": audit_row["time"],
                     "offset_from_endpoint": audit_row["time"] - window["endpoint"],
                     "nominal_rh": nominal_rh(window, audit_row["time"]),
                     "endpoint_reference": window["stage"],
                     "predicted_band": prediction,
                     "confidence": float(np.max(probability)),
                     "drop_minus_bg_L": audit_row["drop_minus_bg_L"],
                     "drop_minus_bg_a": audit_row["drop_minus_bg_a"],
                     "drop_minus_bg_b": audit_row["drop_minus_bg_b"]})
    return rows


def render_review(cache, video_root, output):
    tiles = []
    for window in WINDOWS:
        video_rows = [row for row in cache if row.get("video") == window["video"]]
        cap = cv2.VideoCapture(str(source_path(video_root, window["video"])))
        for seconds in (window["endpoint"] - .5, window["endpoint"], window["endpoint"] + .5):
            row = min(video_rows, key=lambda value: abs(float(value["time"]) - seconds))
            cap.set(cv2.CAP_PROP_POS_MSEC, float(row["time"]) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = resize_for_app(frame)
            lab, _, selected = balanced_frame_and_masks(frame, row)
            circle = tuple(int(float(row[name]))
                           for name in ("circle_x", "circle_y", "circle_r"))
            main, _ = tight_masks(lab.shape, circle, row, selected)
            ys, xs = np.where(main); pad = 18
            crop = frame[max(0, ys.min()-pad):min(frame.shape[0], ys.max()+pad),
                         max(0, xs.min()-pad):min(frame.shape[1], xs.max()+pad)]
            crop = cv2.resize(crop, (300, 245), interpolation=cv2.INTER_CUBIC)
            tile = np.full((292, 300, 3), 245, np.uint8); tile[:245] = crop
            text = f"{window['group']} t={float(row['time']):.2f}s ({float(row['time'])-window['endpoint']:+.2f})"
            cv2.putText(tile, text, (6, 269), cv2.FONT_HERSHEY_SIMPLEX, .44,
                        (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(tile, f"nominal RH={nominal_rh(window,float(row['time'])):.1f}%",
                        (6, 287), cv2.FONT_HERSHEY_SIMPLEX, .42,
                        (20, 20, 20), 1, cv2.LINE_AA)
            tiles.append(tile)
        cap.release()
    sheet = np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:])))
    cv2.imwrite(str(output / "rh_place2_error_window_review.jpg"), sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 97])


def plot(output, rows):
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.6), constrained_layout=True)
    for row_index, window in enumerate(WINDOWS):
        use = [row for row in rows if row["group"] == window["group"]]
        x = [row["offset_from_endpoint"] for row in use]
        axes[row_index, 0].plot(x, [row["drop_minus_bg_L"] for row in use], "o-", label="L*")
        axes[row_index, 0].plot(x, [row["drop_minus_bg_a"] for row in use], "o-", label="a*")
        axes[row_index, 0].plot(x, [row["drop_minus_bg_b"] for row in use], "o-", label="b*")
        axes[row_index, 0].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[row_index, 0].set(title=f"{window['group']} droplet - substrate",
                               xlabel="Seconds from supplied endpoint", ylabel="Relative LAB")
        axes[row_index, 0].legend(fontsize=8)
        axes[row_index, 1].step(x, [row["predicted_band"] for row in use], where="mid")
        axes[row_index, 1].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[row_index, 1].set_yticks((25,45,65,85), DISPLAY)
        axes[row_index, 1].set(title="Prediction from the other place-2 run",
                               xlabel="Seconds from supplied endpoint", ylabel="Predicted RH range")
    fig.suptitle("Place-2 error endpoint temporal audit", fontweight="bold")
    fig.savefig(output / "rh_place2_error_window_audit.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/rh_place2_error_window_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cache = read_csv(args.cache); endpoints = endpoint_rows(cache)
    rows = []
    for window in WINDOWS:
        rows.extend(predict_window(cache, endpoints, args.video_root, window))
    with (args.output / "candidate_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    payload = {"scope": "diagnostic +/-1 s around the two place-2 band errors",
               "warning": "Candidate predictions do not authorize timeline relabeling",
               "rows": rows}
    (args.output / "audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(args.output, rows); render_review(cache, args.video_root, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
