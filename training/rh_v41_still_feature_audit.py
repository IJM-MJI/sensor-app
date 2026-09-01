"""Audit single saved frames with the browser v41 ROI/feature geometry."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

from rh_place2_seven_band_run_holdout import matrices


VIDEO_PATH = Path(
    r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1\1_90_H2O_only_cropped.mp4"
)
OUTPUT = Path("training/output/rh_v41_still_feature_audit")
LABELS = {4: 35, 7: 45, 9: 45, 13: 45, 17: 55, 21: 65, 25: 75, 27: 75}


def resize(frame):
    scale = min(1.0, 480 / max(frame.shape[:2]))
    if scale == 1:
        return frame.copy()
    return cv2.resize(frame, (round(frame.shape[1] * scale),
                              round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def select_circle(frame):
    """Mirror the v41 compact-aperture and tight-crop selection rules."""
    frame = resize(frame); h, w = frame.shape[:2]; side = min(h, w)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    found = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, 1.2, round(side * .12),
        param1=80, param2=35, minRadius=round(side * .07),
        maxRadius=round(side * .60))
    circles = [] if found is None else [tuple(map(int, np.round(row))) for row in found[0]]
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    aa = lab[:, :, 1].astype(float) - 128; bb = lab[:, :, 2].astype(float) - 128
    warm = (np.hypot(aa, bb) > 15) & (lab[:, :, 2] > 133)
    yy, xx = np.ogrid[:h, :w]
    tight = max(w, h) / side < 1.35
    best = None; compact = None; outer = None
    for cx, cy, radius in circles:
        nx, ny, nr = cx / w, cy / h, radius / side
        xlo, xhi = ((.18, .82) if tight else (.24, .76))
        ylo, yhi = ((.10, .78) if tight else (.15, .65))
        if not (xlo <= nx <= xhi and ylo <= ny <= yhi and .13 <= nr <= .60):
            continue
        if tight and nr >= .43:
            distance = math.hypot(nx - .50, ny - .52)
            if outer is None or distance < outer[0]:
                outer = (distance, cx, cy, radius)
        inner = round(radius * .5)
        inner_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= inner ** 2
        inner_grid = (((xx - cx + inner) % 3) == 0) & (((yy - cy + inner) % 3) == 0)
        sampled_inner = inner_mask & inner_grid
        sample = warm & sampled_inner
        coverage = float(sample.sum() / max(sampled_inner.sum(), 1))
        warm_y = np.where(sample)[0]
        span = ((warm_y.max() - warm_y.min()) / max(radius, 1)) if warm_y.size else 0
        pair_r = round(radius * .90)
        pair_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= pair_r ** 2
        pair_grid = (((xx - cx + pair_r) % 3) == 0) & (((yy - cy + pair_r) % 3) == 0)
        sampled_pair = pair_mask & pair_grid
        pair = warm & sampled_pair; pair_count = int(pair.sum())
        upper = int((pair & (yy < cy - .08 * radius)).sum())
        lower = int((pair & (yy > cy + .08 * radius)).sum())
        balance = min(upper, lower) / max(pair_count, 1)
        pair_coverage = pair_count / max(int(sampled_pair.sum()), 1)
        center = math.exp(-.5 * (((nx - .50) / .30) ** 2 + ((ny - .43) / .30) ** 2))
        radius_prior = math.exp(-.5 * ((nr - .30) / .14) ** 2)
        aperture = math.exp(-.5 * (((nx - .50) / .18) ** 2 + ((ny - .48) / .22) ** 2))
        score = ((coverage + .05 * min(span, 1.8)) * math.sqrt(nr) +
                 .08 * center + .10 * radius_prior + .35 * balance + .08 * aperture)
        if best is None or score > best[0]:
            best = (score, cx, cy, radius)
        if nr <= .22 and ny <= .55 and balance >= .12 and pair_coverage >= .018:
            compact_score = score + .65 * pair_coverage + .20 * balance
            if compact is None or compact_score > compact[0]:
                compact = (compact_score, cx, cy, radius)
    if compact is not None:
        return frame, compact[1:], "compact-aperture"
    if tight and outer is not None:
        _, ox, oy, outer_r = outer
        return frame, (round(.75 * w * .50 + .25 * ox),
                       round(.75 * h * .52 + .25 * oy),
                       round(.75 * side * .40 + .25 * outer_r * .75)), "tight-crop-aperture"
    if best is None:
        raise RuntimeError("No v41 chamber circle")
    return frame, best[1:], "auto"


def corrected_pixels(frame, circle):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(float)
    h, w = lab.shape[:2]; cx, cy, radius = circle
    yy, xx = np.ogrid[:h, :w]
    chamber = (xx - cx) ** 2 + (yy - cy) ** 2 <= round(radius * .90) ** 2
    ys, xs = np.where(chamber)
    values = lab[ys, xs].copy()
    chroma = np.hypot(values[:, 1] - 128, values[:, 2] - 128)
    neutral = (chroma < 8) & (values[:, 0] > 140)
    if neutral.sum() >= 50:
        da = np.median(values[neutral, 1]) - 128
        db = np.median(values[neutral, 2]) - 128
        white = np.percentile(values[neutral, 0], 90)
        scale = float(np.clip(210 / max(white, 1), .8, 1.5))
        values[:, 0] = np.clip(values[:, 0] * scale, 0, 255)
        values[:, 1] = np.clip(values[:, 1] - da, 0, 255)
        values[:, 2] = np.clip(values[:, 2] - db, 0, 255)
    chroma = np.hypot(values[:, 1] - 128, values[:, 2] - 128)
    return {"L": values[:, 0], "a": values[:, 1], "b": values[:, 2], "c": chroma,
            "x": xs, "y": ys, "nx": (xs - cx) / radius, "ny": (ys - cy) / radius}


def shape_pixels(p, zone):
    background = np.argsort(p["c"])[:len(p["c"]) // 2]
    bg = np.asarray([p[channel][background].mean() for channel in ("L", "a", "b")])
    use = zone(p["nx"], p["ny"])
    idx = np.where(use)[0]
    distance = np.sqrt((.35 * (p["L"][idx] - bg[0])) ** 2 +
                       (p["a"][idx] - bg[1]) ** 2 + (p["b"][idx] - bg[2]) ** 2)
    order = np.argsort(distance); start = math.floor(len(order) * .65)
    chosen = order[start:][distance[order[start:]] >= 6]
    if len(chosen) < 12:
        chosen = order[max(0, len(order) - max(12, math.floor(len(order) * .2))):]
    return idx[chosen]


def landmark(p, zone):
    background = np.argsort(p["c"])[:len(p["c"]) // 2]
    bg = np.asarray([p[channel][background].mean() for channel in ("L", "a", "b")])
    idx = np.where(zone(p["nx"], p["ny"]))[0]
    distance = np.sqrt((.35 * (p["L"][idx] - bg[0])) ** 2 +
                       (p["a"][idx] - bg[1]) ** 2 + (p["b"][idx] - bg[2]) ** 2)
    if len(idx) < 30:
        return None
    cutoff = max(5, np.percentile(distance, 75)); use = distance >= cutoff
    weight = np.maximum(distance[use] - cutoff, 1) ** 1.5
    if use.sum() < 12:
        return None
    return np.asarray([np.average(p["nx"][idx][use], weights=weight),
                       np.average(p["ny"][idx][use], weights=weight)])


def estimate_registration(p):
    flame = landmark(p, lambda x, y: (x >= -.35) & (x <= .25) & (y >= -.60) & (y <= .10))
    drop = landmark(p, lambda x, y: (x >= -.22) & (x <= .14) & (y >= .20) & (y <= .68))
    if flame is None or drop is None:
        return None
    vx, vy = drop - flame; distance = math.hypot(vx, vy); angle = math.atan2(vx, vy)
    if not (.35 <= distance <= 1.15 and abs(angle) <= math.radians(25)):
        return None
    return float(drop[0]), float(drop[1]), float(angle)


def feature(frame, circle, registration=None):
    p = corrected_pixels(frame, circle)
    if registration is None:
        registration = estimate_registration(p)
    if registration is None:
        local_x, local_y = p["nx"] + .08, p["ny"] - .43
    else:
        dx, dy = p["nx"] - registration[0], p["ny"] - registration[1]
        cosine, sine = math.cos(registration[2]), math.sin(registration[2])
        local_x, local_y = cosine * dx - sine * dy, sine * dx + cosine * dy
    drop_shape = shape_pixels(
        p, lambda x, y: (x >= -.55) & (x <= .35) & (y >= .18) & (y <= .68))
    registered = shape_pixels(p, lambda _x, _y:
        (local_x / .25) ** 2 + (local_y / .29) ** 2 <= 1)
    tight = registered[((local_x[registered] / .215) ** 2 +
                        (local_y[registered] / .245) ** 2) <= 1]
    radius = np.sqrt((local_x / .215) ** 2 + (local_y / .245) ** 2)
    satellite = ((local_x - .30) / .14) ** 2 + ((local_y - .02) / .18) ** 2 <= 1
    flame = shape_pixels(p, lambda x, y: (x >= -.55) & (x <= .35) & (y >= -.62) & (y <= .14))
    flame_set = np.zeros(len(p["L"]), dtype=bool); flame_set[flame] = True
    substrate = ((radius >= 1.12) & (radius <= 1.58) & ~satellite & ~flame_set &
                 (p["L"] >= 55) & (p["L"] <= 235) & (p["c"] <= 35))
    if substrate.sum() >= 30:
        lo, hi = np.percentile(p["L"][substrate], (15, 85))
        substrate &= (p["L"] >= lo) & (p["L"] <= hi)
    drop_lab = np.asarray([np.median(p[ch][tight]) for ch in ("L", "a", "b")])
    substrate_lab = np.asarray([np.median(p[ch][substrate]) for ch in ("L", "a", "b")])
    top = np.argsort(p["c"])[math.floor(len(p["c"]) * .97):]
    background = np.argsort(p["c"])[:len(p["c"]) // 2]
    diagnostic = {}
    for name, indices in (("drop", drop_shape), ("registered", registered),
                          ("flame", flame), ("top", top), ("background", background)):
        for channel in ("L", "a", "b"):
            diagnostic[f"{name}_{channel}"] = float(p[channel][indices].mean())
        chroma = np.maximum(p["c"][indices], 3.0)
        diagnostic[f"{name}_unit_a"] = float(np.mean((p["a"][indices] - 128) / chroma))
        diagnostic[f"{name}_unit_b"] = float(np.mean((p["b"][indices] - 128) / chroma))
    return (drop_lab - substrate_lab, registration, float(p["a"][top].mean()),
            len(tight), int(substrate.sum()), diagnostic)


def load_model(path=Path("sensor-rh-place2-stable-profile-model.js")):
    text = path.read_text(encoding="utf-8")
    return json.loads(re.search(r"=(\{.*\});\s*$", text).group(1))


def video_frame(cap, seconds):
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot decode {VIDEO_PATH.name} at {seconds:.2f}s")
    h, w = frame.shape[:2]; side = min(h, w)
    x0 = (w - side) // 2; y0 = (h - side) // 2
    return frame[y0:y0 + side, x0:x0 + side]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")
    calibration_time = 38.0
    cal_frame, cal_circle, cal_source = select_circle(video_frame(cap, calibration_time))
    cal_vector, registration, top_a, _, _, cal_diagnostic = feature(cal_frame, cal_circle)
    model = load_model(); rows = []
    truth = []; prediction = []
    for seconds, reference in LABELS.items():
        frame = resize(video_frame(cap, seconds))
        h, w = frame.shape[:2]; ch, cw = cal_frame.shape[:2]
        circle = (round(cal_circle[0] / cw * w), round(cal_circle[1] / ch * h),
                  round(cal_circle[2] / min(ch, cw) * min(h, w)))
        vector, _, frame_top_a, n_drop, n_substrate, diagnostic = feature(frame, circle, registration)
        delta = vector - cal_vector
        candidates = []
        for profile_name, profile in model["profiles"].items():
            prototypes = np.asarray(profile["prototypes"], dtype=float)
            scale = np.maximum(np.asarray(profile["scaler_scale"], dtype=float), 1e-9)
            distances = np.sqrt(np.mean(((prototypes - delta) / scale) ** 2, axis=1))
            for index, distance in enumerate(distances):
                candidates.append((float(distance), profile_name, index))
        candidates.sort(); best, profile, index = candidates[0]
        second = candidates[1][0]
        predicted = int(model["levels"][index])
        truth.append(reference); prediction.append(predicted)
        diagnostic_delta = {f"d_{key}": value - cal_diagnostic[key]
                            for key, value in diagnostic.items()}
        rows.append({"time_s": seconds, "reference": reference, "prediction": predicted,
                     "profile": profile, "distance": best, "margin": second - best,
                     "delta_L": delta[0], "delta_a": delta[1], "delta_b": delta[2],
                     "top_a": frame_top_a, "drop_pixels": n_drop,
                     "substrate_pixels": n_substrate, **diagnostic_delta})
    correlations = {}
    labels = np.asarray(truth, dtype=float)
    for key in rows[0]:
        if not key.startswith("d_"):
            continue
        values = np.asarray([row[key] for row in rows], dtype=float)
        correlations[key] = float(np.corrcoef(labels, values)[0, 1]) if np.std(values) else 0.0
    result = {"scope": "v41 single saved-frame feature audit",
              "calibration": {"video": str(VIDEO_PATH), "time_s": calibration_time,
                              "roi": cal_circle,
                              "roi_source": cal_source, "top_a": top_a,
                              "registration": registration},
              "metrics": matrices(np.asarray(truth), np.asarray(prediction)),
              "feature_label_correlations": dict(sorted(
                  correlations.items(), key=lambda item: abs(item[1]), reverse=True)),
              "rows": rows}
    cap.release()
    (OUTPUT / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUTPUT / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
