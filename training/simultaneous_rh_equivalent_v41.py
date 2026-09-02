"""Apply the v41 RH-only optical profiles to uncropped simultaneous clips.

Nominal RH in the filename is diagnostic metadata only.  Each clip is
calibrated from its RH20/H2=0 recovery tail and every reaction frame is
classified independently; time is used only to select audit frames.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import Ridge

from rh_v41_still_feature_audit import feature, load_model, resize, select_circle
from train_models import simultaneous_clips


FRACTIONS = tuple(np.linspace(0.0, 1.0, 9))


def frame_at(path: Path, seconds: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(seconds, 0) * 1000)
    ok, frame = cap.read(); cap.release()
    if not ok:
        raise RuntimeError(f"Cannot decode {path.name} at {seconds:.2f}s")
    return frame


def classify(model, delta):
    candidates = []
    for profile_name, profile in model["profiles"].items():
        prototypes = np.asarray(profile["prototypes"], dtype=float)
        scale = np.maximum(np.asarray(profile["scaler_scale"], dtype=float), 1e-9)
        distances = np.sqrt(np.mean(((prototypes - delta) / scale) ** 2, axis=1))
        for index, distance in enumerate(distances):
            candidates.append((float(distance), profile_name, index))
    candidates.sort()
    best, profile, index = candidates[0]
    return int(model["levels"][index]), profile, best, candidates[1][0] - best


def mode(values):
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], value))


def centered_matrices(rows, groups=None):
    selected = rows if groups is None else [row for row in rows if row["group"] in groups]
    x_parts, y_parts = [], []
    by_video = defaultdict(list)
    for row in selected:
        by_video[row["video"]].append(row)
    for items in by_video.values():
        x = np.asarray([[row[f"flame_delta_{channel}"] for channel in "Lab"]
                        for row in items], dtype=float)
        y = np.asarray([[row[f"delta_{channel}"] for channel in "Lab"]
                        for row in items], dtype=float)
        x_parts.append(x - np.median(x, axis=0))
        y_parts.append(y - np.median(y, axis=0))
    return np.vstack(x_parts), np.vstack(y_parts)


def select_interference_model(rows):
    groups = sorted({row["group"] for row in rows})
    candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        folds = []
        for held in groups:
            train = set(groups) - {held}
            train_x, train_y = centered_matrices(rows, train)
            test_x, test_y = centered_matrices(rows, {held})
            model = Ridge(alpha=alpha, fit_intercept=False).fit(train_x, train_y)
            residual = test_y - model.predict(test_x)
            raw_norm = np.linalg.norm(test_y, axis=1)
            residual_norm = np.linalg.norm(residual, axis=1)
            reduction = 1 - np.median(residual_norm) / max(np.median(raw_norm), 1e-9)
            folds.append({"held_out_run": held, "residual_reduction_fraction": float(reduction)})
        candidates.append({"alpha": alpha, "folds": folds,
                           "median_reduction": float(np.median(
                               [fold["residual_reduction_fraction"] for fold in folds]))})
    selected = max(candidates, key=lambda item: item["median_reduction"])
    x, y = centered_matrices(rows)
    model = Ridge(alpha=selected["alpha"], fit_intercept=False).fit(x, y)
    return model, selected, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/simultaneous_rh_equivalent_v41"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_model()

    rows = []
    for clip in simultaneous_clips():
        if clip.rh is None or clip.rh > 80:
            continue
        path = args.video_root / clip.name
        # Recovery ends at nominal RH20 and H2=0 and is the physically relevant
        # single-frame reference for an RH-only-equivalent delta.
        calibration_s = max(float(clip.reaction_end) + .5,
                            float(clip.duration_hint or clip.reaction_end) - 2.0)
        calibration_frame, circle, roi_source = select_circle(frame_at(path, calibration_s))
        calibration_vector, registration, _, _, _, calibration_diagnostic = feature(
            calibration_frame, circle)

        start, end = float(clip.reaction_start), float(clip.reaction_end)
        for fraction in FRACTIONS:
            seconds = start + float(fraction) * (end - start)
            current = resize(frame_at(path, seconds))
            vector, _, top_a, n_drop, n_substrate, diagnostic = feature(
                current, circle, registration)
            delta = vector - calibration_vector
            prediction, profile, distance, margin = classify(model, delta)
            rows.append({
                "group": clip.group, "video": clip.name,
                "nominal_rh_metadata": int(clip.rh),
                "time_s": round(seconds, 3), "search_fraction": round(float(fraction), 3),
                "rh_only_equivalent_midpoint": prediction,
                "rh_only_equivalent_band": f"{prediction - 5}-{prediction + 5}",
                "profile": profile, "distance": round(distance, 5),
                "margin": round(margin, 5), "roi_source": roi_source,
                "circle_x": circle[0], "circle_y": circle[1], "circle_r": circle[2],
                "delta_L": float(delta[0]), "delta_a": float(delta[1]),
                "delta_b": float(delta[2]), "top_a": top_a,
                "drop_pixels": n_drop, "substrate_pixels": n_substrate,
                "flame_delta_L": diagnostic["flame_L"] - calibration_diagnostic["flame_L"],
                "flame_delta_a": diagnostic["flame_a"] - calibration_diagnostic["flame_a"],
                "flame_delta_b": diagnostic["flame_b"] - calibration_diagnostic["flame_b"],
            })

    interference_model, selected_interference, interference_candidates = \
        select_interference_model(rows)
    for row in rows:
        flame = np.asarray([[row[f"flame_delta_{channel}"] for channel in "Lab"]])
        nuisance = interference_model.predict(flame)[0]
        corrected = np.asarray([row[f"delta_{channel}"] for channel in "Lab"]) - nuisance
        prediction, profile, distance, margin = classify(model, corrected)
        row.update({
            "corrected_rh_only_equivalent_midpoint": prediction,
            "corrected_rh_only_equivalent_band": f"{prediction - 5}-{prediction + 5}",
            "corrected_profile": profile, "corrected_distance": round(distance, 5),
            "corrected_margin": round(margin, 5),
            "predicted_h2_nuisance_L": float(nuisance[0]),
            "predicted_h2_nuisance_a": float(nuisance[1]),
            "predicted_h2_nuisance_b": float(nuisance[2]),
        })

    fields = list(rows[0])
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video"]].append(row)
    videos = []
    for video, items in sorted(by_video.items()):
        predictions = [int(item["rh_only_equivalent_midpoint"]) for item in items]
        corrected = [int(item["corrected_rh_only_equivalent_midpoint"]) for item in items]
        selected = mode(predictions); selected_corrected = mode(corrected)
        videos.append({
            "video": video, "group": items[0]["group"],
            "nominal_rh_metadata": items[0]["nominal_rh_metadata"],
            "modal_rh_only_equivalent_band": f"{selected - 5}-{selected + 5}",
            "within_video_mode_fraction": float(np.mean(np.asarray(predictions) == selected)),
            "unique_predicted_bands": sorted(set(predictions)),
            "corrected_modal_band": f"{selected_corrected - 5}-{selected_corrected + 5}",
            "corrected_within_video_mode_fraction": float(
                np.mean(np.asarray(corrected) == selected_corrected)),
            "corrected_unique_bands": sorted(set(corrected)),
            "median_distance": float(np.median([float(item["distance"]) for item in items])),
            "corrected_median_distance": float(np.median(
                [float(item["corrected_distance"]) for item in items])),
            "roi_source": items[0]["roi_source"],
        })

    by_nominal = defaultdict(list)
    for item in videos:
        by_nominal[int(item["nominal_rh_metadata"])].append(item)
    nominal_summary = []
    for nominal, items in sorted(by_nominal.items()):
        mids = [int(item["modal_rh_only_equivalent_band"].split("-")[0]) + 5 for item in items]
        selected = mode(mids)
        nominal_summary.append({
            "nominal_rh_metadata": nominal,
            "run_modal_bands": [item["modal_rh_only_equivalent_band"] for item in items],
            "cross_run_consensus_band": f"{selected - 5}-{selected + 5}",
            "cross_run_consensus_fraction": float(np.mean(np.asarray(mids) == selected)),
        })

    report = {
        "status": "baseline complete; tested linear H2 correction rejected",
        "target": "RH-only-equivalent seven-band optical response",
        "supervision": "nominal simultaneous RH excluded",
        "input": "single frames from uncropped RH-labelled clips",
        "interference_correction": {
            "method": "within-video centered droplet change regressed on flame change; nominal RH unused",
            "selected": selected_interference,
            "candidates": interference_candidates,
            "coefficients": interference_model.coef_.tolist(),
            "decision": "reject" if selected_interference["median_reduction"] <= 0 else "retain for review",
            "required_direction": "positive residual reduction in every held-out run",
        },
        "videos": videos,
        "nominal_group_diagnostic": nominal_summary,
        "deployment": {
            "allowed": False,
            "reason": (
                "No simultaneous RH-only-equivalent ground truth and the tested "
                "H2 correction worsened held-out residuals."
            ),
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
