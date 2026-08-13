"""Analyse untimed monotonic H2O-only response recordings safely.

The script never treats inferred RH as reference truth.  It extracts the same
patch-balanced droplet features as the app, estimates an optical-equivalent RH
only for visualization, detects stable plateaus, and writes high-confidence
H2O-response rows that may augment state training but not RH regression.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_models import (
    CACHE_VERSION, FEATURE_NAMES, corrected, circle_track, extract_features,
    frame_at, read_csv,
)


VIDEO_CONFIG = {
    # Media players apply container rotation metadata, but OpenCV exposes the
    # decoded pixel layout.  Visual inspection of decoded crops found the flame
    # on the left in run 3 and on the right in run 6.
    "1_90_H2O_only_3(response).mp4": 3,
    "1_90_H2O_only_6(response).mp4": 1,
}
DEFAULT_VIDEOS = list(VIDEO_CONFIG)


def robust_track(cap: cv2.VideoCapture, duration: float):
    """Reject one-off Hough detections around fittings and background circles."""
    candidates = circle_track(cap, duration, interval=3.0)
    if len(candidates) < 3:
        return candidates
    best: list[tuple[float, tuple[int, int, int]]] = []
    for anchor in candidates:
        _, (ax, ay, ar) = anchor
        cluster = []
        for item in candidates:
            _, (x, y, r) = item
            if math.hypot(x - ax, y - ay) <= max(18, .65 * ar) and abs(r - ar) <= max(12, .45 * ar):
                cluster.append(item)
        if len(cluster) > len(best):
            best = cluster
    return sorted(best, key=lambda item: item[0])


def rolling_median(values: np.ndarray, width: int = 5) -> np.ndarray:
    radius = width // 2
    return np.asarray([
        np.median(values[max(0, i - radius):min(len(values), i + radius + 1)])
        for i in range(len(values))
    ])


def fit_optical_reference(reference_rows: list[dict[str, object]]):
    rows = [r for r in reference_rows if r.get("kind") == "rh_only" and r.get("rh_value") is not None]
    x = np.asarray([[float(r["drop_a"]), float(r["drop_b"]), float(r["drop_L"])] for r in rows])
    y = np.asarray([float(r["rh_value"]) for r in rows])
    model = make_pipeline(StandardScaler(), Ridge(alpha=8.0)).fit(x, y)
    reference = {}
    for level in sorted(set(y)):
        values = model.predict(x[y == level])
        reference[str(int(level))] = {
            "n": int((y == level).sum()),
            "median_score": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }
    ninety = model.predict(x[y >= 90])
    threshold = float(np.percentile(ninety, 95)) if len(ninety) else 92.0
    return model, reference, threshold


def extract_video(path: Path, sample_hz: float, orientation_quarters: int) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(fps, 1e-6)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    track = robust_track(cap, duration)
    times = np.arange(0.5, max(0.6, duration - .25), 1.0 / sample_hz)
    raw, preview = [], []
    for index, t in enumerate(times):
        frame = frame_at(cap, float(t))
        if frame is None:
            continue
        circle = min(track, key=lambda item: abs(item[0] - t))[1]
        try:
            feat = corrected(extract_features(frame, circle, orientation_quarters=orientation_quarters))
        except RuntimeError:
            continue
        raw.append({
            "video": path.name, "group": f"rh-untimed-{path.stem}", "kind": "rh_only_weak",
            "time": float(t), "duration": duration, "width": width, "height": height,
            "circle_x": circle[0], "circle_y": circle[1], "circle_r": circle[2],
            "orientation_quarters": orientation_quarters, "orientation_confidence": 1.0,
            **feat,
        })
        if index % max(1, round(len(times) / 8)) == 0:
            resized = cv2.resize(frame, (270, 480), interpolation=cv2.INTER_AREA)
            cx, cy, cr = circle
            pad = round(cr * 1.25)
            x0, x1 = max(0, cx - pad), min(resized.shape[1], cx + pad)
            y0, y1 = max(0, cy - pad), min(resized.shape[0], cy + pad)
            crop = resized[y0:y1, x0:x1]
            crop = cv2.resize(crop, (260, 260), interpolation=cv2.INTER_CUBIC)
            cv2.putText(crop, f"{t:.0f} s", (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(crop, f"{t:.0f} s", (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(crop, "F", (126, 42), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(crop, "F", (126, 42), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(crop, "D", (126, 235), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(crop, "D", (126, 235), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
            preview.append(crop)
    cap.release()
    if not raw:
        raise RuntimeError(f"No usable frames extracted from {path}")
    # A monotonic response recording is expected to begin at its dry/low-RH
    # condition.  Use a robust early window, not one frame, as the video baseline.
    baseline_limit = min(8.0, max(4.0, duration * .10))
    baseline_rows = [r for r in raw if float(r["time"]) <= baseline_limit]
    baseline = {name: float(np.median([float(r[name]) for r in baseline_rows])) for name in FEATURE_NAMES}
    for row in raw:
        for name in FEATURE_NAMES:
            row[name] = float(row[name]) - baseline[name]
    return raw, preview


def stable_segments(rows: list[dict[str, object]], scores: np.ndarray, above_threshold: float):
    times = np.asarray([float(r["time"]) for r in rows])
    smooth = rolling_median(scores, 5)
    slope = np.gradient(smooth, times)
    cutoff = max(0.8, float(np.percentile(np.abs(slope), 45)))
    stable = np.abs(slope) <= cutoff
    segments, start = [], None
    for i, value in enumerate(np.r_[stable, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            end = i
            if end - start >= 4 and times[end - 1] - times[start] >= 3:
                median = float(np.median(smooth[start:end]))
                segments.append({
                    "start_s": float(times[start]), "end_s": float(times[end - 1]),
                    "duration_s": float(times[end - 1] - times[start]),
                    "optical_rh_median": median,
                    "optical_rh_iqr": float(np.percentile(smooth[start:end], 75) - np.percentile(smooth[start:end], 25)),
                    "range_status": "above_90_similarity_candidate" if median > above_threshold else "unresolved_without_timeline",
                })
            start = None
    return smooth, slope, stable, segments, cutoff


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(f"training/cache/{CACHE_VERSION}/features.csv"))
    parser.add_argument("--output", type=Path, default=Path("training/output/untimed_rh_response"))
    parser.add_argument("--sample-hz", type=float, default=1.0)
    parser.add_argument("--videos", nargs="*", default=DEFAULT_VIDEOS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference_rows = read_csv(args.cache)
    model, reference_summary, above_threshold = fit_optical_reference(reference_rows)
    all_rows, weak_rows, report, previews = [], [], {}, []
    fig, axes = plt.subplots(len(args.videos), 1, figsize=(8.0, 3.0 * len(args.videos)), squeeze=False, constrained_layout=True)
    for axis, filename in zip(axes[:, 0], args.videos):
        orientation_quarters = VIDEO_CONFIG.get(filename)
        if orientation_quarters is None:
            raise RuntimeError(f"Decoded-pixel orientation has not been verified for {filename}")
        rows, preview = extract_video(args.video_root / filename, args.sample_hz, orientation_quarters)
        x = np.asarray([[float(r["drop_a"]), float(r["drop_b"]), float(r["drop_L"])] for r in rows])
        scores = model.predict(x)
        smooth, slope, stable, segments, cutoff = stable_segments(rows, scores, above_threshold)
        delta_e = np.asarray([math.sqrt((.35 * float(r["drop_L"])) ** 2
                                        + float(r["drop_a"]) ** 2 + float(r["drop_b"]) ** 2) for r in rows])
        baseline_delta = delta_e[np.asarray([float(r["time"]) <= min(8.0, max(4.0, float(r["duration"]) * .10)) for r in rows])]
        response_threshold = max(float(np.percentile(baseline_delta, 95)) + .5,
                                 float(np.percentile(delta_e, 65)))
        response_stable = stable & (delta_e >= response_threshold)
        for row, score, smoothed, derivative, is_stable, de, is_response in zip(
                rows, scores, smooth, slope, stable, delta_e, response_stable):
            row.update({
                "optical_rh_raw": float(score), "optical_rh_smoothed": float(smoothed),
                "optical_rh_slope": float(derivative), "stable": int(is_stable),
                "drop_response_delta_e": float(de),
                "range_status": "above_90_similarity_candidate" if smoothed > above_threshold else "unresolved_without_timeline",
            })
            # These are safe only for state augmentation.  They are explicitly
            # excluded from RH concentration fitting because their times are unknown.
            if is_response:
                weak = dict(row)
                weak.update({"h2_present": 0, "rh_high": None, "rh_present": 1, "state": "rh_response_unlabeled",
                             "h2_value": None, "rh_value": None, "rh_setpoint": None})
                weak_rows.append(weak)
        all_rows.extend(rows)
        previews.append((filename, preview))
        report[filename] = {
            "duration_s": float(rows[0]["duration"]), "frames_analyzed": len(rows),
            "baseline_policy": "median of first 4-8 seconds",
            "display_orientation": "user-verified flame top / droplet bottom",
            "decoded_pixel_orientation_quarters": orientation_quarters,
            "stable_slope_cutoff_rh_per_s": cutoff,
            "above_90_score_threshold": above_threshold,
            "above_90_similarity_candidate_frames": int(np.sum(smooth > above_threshold)),
            "known_recording_contains_above_90": True,
            "above_90_timing_status": "unknown until timeline is supplied",
            "drop_response_delta_e_threshold": response_threshold,
            "weak_state_training_frames": int(np.sum(response_stable)),
            "segments": segments,
        }
        times = [float(r["time"]) for r in rows]
        axis.plot(times, smooth, color="#1769aa", lw=1.8, label="optical RH-equivalent")
        axis.scatter(np.asarray(times)[stable], smooth[stable], s=8, color="#43a047", label="stable candidate")
        axis.axhline(above_threshold, color="#d32f2f", ls="--", lw=1.2, label="90% reference upper bound")
        axis.fill_between(times, above_threshold, max(max(smooth) + 3, above_threshold + 3),
                          where=smooth > above_threshold, color="#ef5350", alpha=.16)
        axis.set(title=filename, xlabel="Time (s)", ylabel="Optical RH-equivalent (%)")
        axis.grid(alpha=.2); axis.legend(fontsize=8, loc="best")
    fig.savefig(args.output / "untimed_rh_response.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output / "untimed_rh_response.pdf", bbox_inches="tight")
    plt.close(fig)
    for filename, images in previews:
        if images:
            cv2.imwrite(str(args.output / f"{Path(filename).stem}_contact_sheet.jpg"), np.hstack(images))
    write_rows(args.output / "frame_features.csv", all_rows)
    write_rows(args.output / "weak_state_features.csv", weak_rows)
    summary = {
        "warning": "Optical RH values are similarity estimates, not concentration ground truth.",
        "reference_levels": reference_summary,
        "videos": report,
    }
    (args.output / "analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
