"""Test the user-observed yellow -> orange -> scarlet -> purple RH path.

This experiment is deliberately self-contained (NumPy only).  It reconstructs
the calibrated absolute droplet colour from the cached initial colour plus the
per-frame delta, evaluates ordered ridge models with complete-run holdout, and
writes a compact SVG that does not require a plotting runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


LEVELS = np.asarray([25, 40, 50, 60, 70, 80, 90], dtype=float)
DISPLAY = ["20-30", "40", "50", "60", "70", "80", "90"]
PLACE1_PARTIAL = ("1_90_H2O_only_2_extract.mp4", 140.0)
ENDPOINT_TOLERANCE_SECONDS = .75

# Supplied times are endpoints: the named RH is reached at the listed time.
ENDPOINTS = {
    "1_90_H2O_only_2_extract.mp4": [
        (6, 20), (9, 30), (15, 40), (25, 50), (35, 60), (45, 70),
        (72, 80), (140, 90),
    ],
    "1_90_H2O_only_extract_3min.mp4": [
        (14, 20), (25, 30), (45, 40), (90, 50), (120, 60), (189, 70),
    ],
    "1_90_H2O_only_extract_extra.mp4": [(9, 70), (87, 80)],
    "1_90_H2O_only_6(response).mp4": [
        (7, 20), (10, 30), (13, 40), (14, 50), (16, 60), (18, 70),
        (20, 80), (32, 90),
    ],
    "1_90_H2O_only_3(response).mp4": [
        (2, 20), (3, 30), (5, 40), (7, 50), (11, 60), (25, 70),
        (28, 80), (38, 90),
    ],
}


def number(row, name):
    return float(row[name])


def circular_difference(now, baseline):
    return math.atan2(math.sin(now - baseline), math.cos(now - baseline))


def colour_features(row):
    dL, da, db = (number(row, f"drop_registered_{c}") for c in "Lab")
    baseL, basea, baseb = (
        number(row, f"baseline_drop_registered_{c}") for c in "Lab"
    )
    L, a, b = baseL + dL, basea + da, baseb + db
    base_chroma = math.hypot(basea, baseb)
    chroma = math.hypot(a, b)
    base_hue, hue = math.atan2(baseb, basea), math.atan2(b, a)
    hue_progress = circular_difference(hue, base_hue)
    warm_delta = number(row, "drop_warm_fraction")
    purple_delta = number(row, "drop_purple_fraction")
    # Distribution statistics retain the actual OpenCV-LAB sensing-pixel colour
    # that a person sees.  Reconstruct it from cached initial + delta and centre
    # a*/b* on 128.  The registered mean above is background-subtracted and is
    # useful as a sensor signal, but must not be mistaken for a literal colour.
    visual = {}
    for percentile in (25, 50, 75):
        for channel in "Lab":
            key = f"drop_{channel}_p{percentile}"
            visual[(channel, percentile)] = number(row, f"baseline_{key}") + number(row, key)
    visual_a = visual[("a", 50)] - 128.0
    visual_b = visual[("b", 50)] - 128.0
    visual_hue = math.atan2(visual_b, visual_a)
    base_visual_a = number(row, "baseline_drop_a_p50") - 128.0
    base_visual_b = number(row, "baseline_drop_b_p50") - 128.0
    base_visual_hue = math.atan2(base_visual_b, base_visual_a)
    visual_hue_progress = circular_difference(visual_hue, base_visual_hue)
    actual_warm = number(row, "baseline_drop_warm_fraction") + warm_delta
    actual_purple = number(row, "baseline_drop_purple_fraction") + purple_delta
    quantile_delta = []
    quantile_absolute = []
    for percentile in (25, 50, 75):
        for channel in "Lab":
            key = f"drop_{channel}_p{percentile}"
            quantile_delta.append(number(row, key))
            value = visual[(channel, percentile)]
            quantile_absolute.append(value if channel == "L" else value - 128.0)
    return {
        "delta_lab": np.asarray([dL, da, db]),
        "hue_chroma_path": np.asarray([
            dL, da, db, chroma - base_chroma, math.sin(hue_progress),
            math.cos(hue_progress) - 1.0,
        ]),
        "human_colour_path": np.asarray([
            da, db, chroma - base_chroma, hue_progress,
            warm_delta, purple_delta,
        ]),
        "absolute_plus_path": np.asarray([
            dL, da, db, L, a, b, chroma, math.sin(hue), math.cos(hue),
            hue_progress, warm_delta, purple_delta,
        ]),
        "visual_colour_path": np.asarray([
            number(row, "drop_L_p50"), number(row, "drop_a_p50"),
            number(row, "drop_b_p50"), visual_a, visual_b,
            math.hypot(visual_a, visual_b), math.sin(visual_hue),
            math.cos(visual_hue), visual_hue_progress,
            actual_warm, actual_purple,
        ]),
        "visual_quantile_path": np.asarray([
            *quantile_delta, *quantile_absolute, visual_hue_progress,
            actual_warm, actual_purple,
        ]),
        "audit": {"L": L, "a": a, "b": b, "chroma": chroma,
                  "hue_deg": math.degrees(hue),
                  "hue_progress_deg": math.degrees(hue_progress),
                  "delta_e": math.sqrt(dL * dL + da * da + db * db),
                  "warm_delta": warm_delta, "purple_delta": purple_delta,
                  "visual_a": visual_a, "visual_b": visual_b,
                  "visual_hue_deg": math.degrees(visual_hue),
                  "visual_hue_progress_deg": math.degrees(visual_hue_progress),
                  "actual_warm": actual_warm, "actual_purple": actual_purple},
    }


def load_endpoints(cache):
    with cache.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_video = defaultdict(list)
    for row in rows:
        if row.get("kind") == "rh_only" and row.get("video") in ENDPOINTS:
            by_video[row["video"]].append(row)
    items = []
    for video, points in ENDPOINTS.items():
        available = by_video[video]
        if not available:
            continue
        for seconds, supplied_rh in points:
            row = min(available, key=lambda r: abs(number(r, "time") - seconds))
            if (video, float(seconds)) == PLACE1_PARTIAL:
                continue
            if abs(number(row, "time") - float(seconds)) > ENDPOINT_TOLERANCE_SECONDS:
                continue
            stage = 25.0 if supplied_rh in (20, 30) else float(supplied_rh)
            features = colour_features(row)
            items.append({
                "video": video, "group": row["group"],
                "time": number(row, "time"), "requested_time": float(seconds),
                "stage": stage, **features,
            })
    return items


def group_stage_weights(groups, truth, use):
    weights = np.zeros(len(groups))
    for group in sorted(set(groups[use])):
        for stage in LEVELS:
            select = use & (groups == group) & (truth == stage)
            if np.any(select):
                weights[select] = 1.0 / np.sum(select)
    return weights


def fit_ridge(x, y, weights, alpha):
    active = weights > 0
    xa, ya, wa = x[active], y[active], weights[active]
    mean = np.average(xa, axis=0, weights=wa)
    scale = np.sqrt(np.average((xa - mean) ** 2, axis=0, weights=wa))
    scale = np.maximum(scale, 1e-6)
    z = (xa - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    root_w = np.sqrt(wa)[:, None]
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(
        (design * root_w).T @ (design * root_w) + penalty,
        (design * root_w).T @ (ya * root_w[:, 0]),
    )
    return mean, scale, beta


def predict_ridge(model, x):
    mean, scale, beta = model
    z = (x - mean) / scale
    continuous = np.column_stack([np.ones(len(z)), z]) @ beta
    prediction = LEVELS[np.argmin(abs(continuous[:, None] - LEVELS), axis=1)]
    return prediction, continuous


def fit_ordinal(x, y, weights, alpha):
    """Fit ordered P(RH >= threshold) boundaries with weighted NumPy logistic."""
    active = weights > 0
    xa, ya, wa = x[active], y[active], weights[active]
    mean = np.average(xa, axis=0, weights=wa)
    scale = np.sqrt(np.average((xa - mean) ** 2, axis=0, weights=wa))
    scale = np.maximum(scale, 1e-6)
    design = np.column_stack([np.ones(len(xa)), (xa - mean) / scale])
    wa = wa / max(float(np.sum(wa)), 1e-9)
    betas = []
    for threshold in LEVELS[1:]:
        target = (ya >= threshold).astype(float)
        beta = np.zeros(design.shape[1])
        for iteration in range(800):
            score = np.clip(design @ beta, -30, 30)
            probability = 1.0 / (1.0 + np.exp(-score))
            gradient = design.T @ ((probability - target) * wa)
            gradient[1:] += alpha * beta[1:]
            learning_rate = .25 / math.sqrt(1.0 + iteration / 300.0)
            update = learning_rate * gradient
            beta -= update
            if np.max(abs(update)) < 1e-8:
                break
        betas.append(beta)
    return mean, scale, np.asarray(betas)


def predict_ordinal(model, x):
    mean, scale, betas = model
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    cumulative = 1.0 / (1.0 + np.exp(-np.clip(design @ betas.T, -30, 30)))
    cumulative = np.minimum.accumulate(cumulative, axis=1)
    probabilities = np.column_stack([
        1 - cumulative[:, 0],
        *[cumulative[:, index - 1] - cumulative[:, index]
          for index in range(1, cumulative.shape[1])],
        cumulative[:, -1],
    ])
    probabilities = np.clip(probabilities, 0, 1)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-9)
    index = np.argmax(probabilities, axis=1)
    return LEVELS[index], probabilities @ LEVELS


def metric_tuple(truth, prediction):
    recalls = [np.mean(prediction[truth == level] == level)
               for level in LEVELS if np.any(truth == level)]
    stage_distance = abs(np.searchsorted(LEVELS, truth)
                         - np.searchsorted(LEVELS, prediction))
    return (float(np.mean(recalls)), float(np.mean(truth == prediction)),
            float(np.mean(stage_distance <= 1)),
            -float(np.mean(abs(truth - prediction))))


def confusion(truth, prediction):
    out = np.zeros((len(LEVELS), len(LEVELS)), dtype=int)
    for actual, estimated in zip(truth, prediction):
        out[np.where(LEVELS == actual)[0][0],
            np.where(LEVELS == estimated)[0][0]] += 1
    return out


def report(truth, prediction):
    matrix = confusion(truth, prediction)
    recalls = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    distance = abs(np.searchsorted(LEVELS, truth)
                   - np.searchsorted(LEVELS, prediction))
    return {
        "exact_accuracy": float(np.mean(truth == prediction)),
        "balanced_accuracy": float(np.mean(recalls)),
        "within_one_stage": float(np.mean(distance <= 1)),
        "mae": float(np.mean(abs(truth - prediction))),
        "per_stage_recall": recalls.tolist(), "confusion": matrix.tolist(),
        "n_endpoints": int(len(truth)),
    }


def cross_fitted(items, feature_name):
    groups = np.asarray([item["group"] for item in items])
    truth = np.asarray([item["stage"] for item in items])
    x = np.asarray([item[feature_name] for item in items])
    prediction = np.full(len(items), np.nan)
    continuous = np.full(len(items), np.nan)
    choices = []
    for held_out in sorted(set(groups)):
        outer_train = groups != held_out
        best = None
        for alpha in (.01, .1, 1.0, 10.0, 100.0):
            scores = []
            for inner in sorted(set(groups[outer_train])):
                train = outer_train & (groups != inner)
                test = outer_train & (groups == inner)
                weights = group_stage_weights(groups, truth, train)
                pred, _ = predict_ridge(fit_ridge(x, truth, weights, alpha), x[test])
                scores.append(metric_tuple(truth[test], pred))
            score = tuple(np.mean(scores, axis=0))
            if best is None or score > best[0]:
                best = (score, alpha)
        train, test = groups != held_out, groups == held_out
        weights = group_stage_weights(groups, truth, train)
        prediction[test], continuous[test] = predict_ridge(
            fit_ridge(x, truth, weights, best[1]), x[test])
        choices.append({"held_out_group": held_out, "alpha": best[1]})
    return report(truth, prediction), prediction, continuous, choices


def cross_fitted_ordinal(items, feature_name):
    groups = np.asarray([item["group"] for item in items])
    truth = np.asarray([item["stage"] for item in items])
    x = np.asarray([item[feature_name] for item in items])
    prediction = np.full(len(items), np.nan)
    continuous = np.full(len(items), np.nan)
    choices = []
    for held_out in sorted(set(groups)):
        outer_train = groups != held_out
        best = None
        for alpha in (.001, .03, .3):
            scores = []
            for inner in sorted(set(groups[outer_train])):
                train = outer_train & (groups != inner)
                test = outer_train & (groups == inner)
                weights = group_stage_weights(groups, truth, train)
                pred, _ = predict_ordinal(fit_ordinal(x, truth, weights, alpha), x[test])
                scores.append(metric_tuple(truth[test], pred))
            score = tuple(np.mean(scores, axis=0))
            if best is None or score > best[0]:
                best = (score, alpha)
        train, test = groups != held_out, groups == held_out
        weights = group_stage_weights(groups, truth, train)
        prediction[test], continuous[test] = predict_ordinal(
            fit_ordinal(x, truth, weights, best[1]), x[test])
        choices.append({"held_out_group": held_out, "alpha": best[1]})
    return report(truth, prediction), prediction, continuous, choices


def monotonic_audit(items):
    output = []
    for group in sorted(set(item["group"] for item in items)):
        group_items = [item for item in items if item["group"] == group]
        medians = []
        for stage in LEVELS:
            values = [item for item in group_items if item["stage"] == stage]
            if values:
                medians.append((stage, np.median([
                    item["audit"]["hue_progress_deg"] for item in values]),
                    np.median([item["audit"]["delta_e"] for item in values])))
        stage = np.asarray([value[0] for value in medians])
        hue = np.asarray([value[1] for value in medians])
        magnitude = np.asarray([value[2] for value in medians])
        hue_direction = np.sign(np.median(np.diff(hue))) if len(hue) > 1 else 0
        hue_order = float(np.mean(np.diff(hue) * hue_direction >= 0)) if len(hue) > 1 else 1
        magnitude_order = float(np.mean(np.diff(magnitude) >= 0)) if len(magnitude) > 1 else 1
        output.append({"group": group, "stages": stage.tolist(),
                       "hue_progress_deg": hue.tolist(),
                       "delta_e": magnitude.tolist(),
                       "hue_order_fraction": hue_order,
                       "delta_e_order_fraction": magnitude_order,
                       "visual_hue_progress_deg": [float(np.median([
                           item["audit"]["visual_hue_progress_deg"]
                           for item in group_items if item["stage"] == level
                       ])) for level in stage]})
    return output


def svg_text(x, y, value, size=12, anchor="middle", weight="normal"):
    safe = str(value).replace("&", "&amp;").replace("<", "&lt;")
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-family="Arial" font-size="{size}" font-weight="{weight}">{safe}</text>')


def make_svg(path, items, reports):
    width, height = 1500, 620
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(svg_text(width / 2, 30, "RH human-guided colour-path validation", 22, weight="bold"))
    colours = {"rh-indoor-fast": "#1f77b4", "rh-indoor-long": "#17a589",
               "rh-response-3": "#e67e22", "rh-response-6": "#8e44ad"}
    # Panel 1: absolute a*-b* trajectory.
    x0, y0, w, h = 70, 85, 390, 400
    all_a = np.asarray([item["audit"]["a"] for item in items]); all_b = np.asarray([item["audit"]["b"] for item in items])
    amin, amax = np.min(all_a), np.max(all_a); bmin, bmax = np.min(all_b), np.max(all_b)
    sx = lambda a: x0 + 20 + (a - amin) / max(amax - amin, 1e-6) * (w - 40)
    sy = lambda b: y0 + h - 20 - (b - bmin) / max(bmax - bmin, 1e-6) * (h - 40)
    parts += [f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fafafa" stroke="#999"/>',
              svg_text(x0 + w/2, y0 - 12, "Absolute calibrated droplet a*-b* path", 15, weight="bold")]
    for group in sorted(colours):
        points = []
        for stage in LEVELS:
            candidates = [item for item in items if item["group"] == group and item["stage"] == stage]
            if candidates:
                a = float(np.median([v["audit"]["a"] for v in candidates])); b = float(np.median([v["audit"]["b"] for v in candidates]))
                points.append((sx(a), sy(b), stage))
        parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y,_ in points)}" fill="none" stroke="{colours[group]}" stroke-width="2"/>')
        for x, y, stage in points:
            parts += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colours[group]}"/>', svg_text(x+8, y-7, int(stage), 9, anchor="start")]
    parts += [svg_text(x0+w/2, y0+h+30, "a*  (more red to the right)", 12),
              svg_text(x0+5, y0+h/2, "b* (yellow up / blue down)", 12, anchor="start")]
    # Panel 2: per-run hue progress.
    x1, y1, w1, h1 = 530, 85, 390, 400
    parts += [f'<rect x="{x1}" y="{y1}" width="{w1}" height="{h1}" fill="#fafafa" stroke="#999"/>',
              svg_text(x1+w1/2, y1-12, "Hue change from each run's initial colour", 15, weight="bold")]
    vals = np.asarray([item["audit"]["hue_progress_deg"] for item in items]); vmin, vmax = vals.min(), vals.max()
    px = lambda stage: x1+25 + np.where(LEVELS == stage)[0][0] / (len(LEVELS)-1) * (w1-50)
    py = lambda value: y1+h1-25-(value-vmin)/max(vmax-vmin,1e-6)*(h1-50)
    for group in sorted(colours):
        points=[]
        for stage in LEVELS:
            candidates=[item for item in items if item["group"]==group and item["stage"]==stage]
            if candidates:
                value=float(np.median([v["audit"]["hue_progress_deg"] for v in candidates])); points.append((px(stage),py(value)))
        parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in points)}" fill="none" stroke="{colours[group]}" stroke-width="2"/>')
        for x,y in points: parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colours[group]}"/>')
    for stage in LEVELS: parts.append(svg_text(px(stage), y1+h1+22, DISPLAY[np.where(LEVELS==stage)[0][0]], 10))
    for index,(group,colour) in enumerate(colours.items()):
        yy=515+index*20; parts += [f'<line x1="{x1+10}" y1="{yy}" x2="{x1+35}" y2="{yy}" stroke="{colour}" stroke-width="3"/>', svg_text(x1+42,yy+4,group,10,anchor="start")]
    # Panel 3: normalized confusion for the best human-path candidate.
    best_name=max(reports, key=lambda name: (reports[name]["balanced_accuracy"], reports[name]["exact_accuracy"]))
    best=reports[best_name]; matrix=np.asarray(best["confusion"],dtype=float); norm=matrix/np.maximum(matrix.sum(axis=1,keepdims=True),1)
    x2,y2,cell=1015,105,54
    parts.append(svg_text(x2+cell*3.5,y2-20,f"Best ordered model: {best_name}",15,weight="bold"))
    for r in range(7):
        for c in range(7):
            shade=int(250-165*norm[r,c]); parts.append(f'<rect x="{x2+c*cell}" y="{y2+r*cell}" width="{cell}" height="{cell}" fill="rgb({shade},{shade+10 if shade<245 else 250},250)" stroke="white"/>'); parts.append(svg_text(x2+(c+.5)*cell,y2+(r+.58)*cell,f"{norm[r,c]:.2f}",10))
    for i,label in enumerate(DISPLAY):
        parts += [svg_text(x2+(i+.5)*cell,y2+7*cell+18,label,9), svg_text(x2-8,y2+(i+.58)*cell,label,9,anchor="end")]
    parts += [svg_text(x2+cell*3.5,y2+7*cell+45,"Predicted RH",11),
              svg_text(x2+cell*3.5,535,f"exact={best['exact_accuracy']:.3f}  balanced={best['balanced_accuracy']:.3f}  within1={best['within_one_stage']:.3f}  MAE={best['mae']:.2f}",12),
              svg_text(width/2,600,"Complete-run held-out endpoints; place-1 nominal 90 excluded as verified 70-80 partial",11)]
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        "training/cache/v7-verified-orientation-recovery-tail/features_registered_drop_v2.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_human_color_path_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    items = load_endpoints(args.cache)
    feature_names = ["delta_lab", "hue_chroma_path", "human_colour_path",
                     "absolute_plus_path", "visual_colour_path",
                     "visual_quantile_path"]
    reports, prediction_rows = {}, []
    for feature_name in feature_names:
        for model_name, evaluator in (("ridge", cross_fitted),
                                      ("ordinal", cross_fitted_ordinal)):
            name = f"{feature_name}_{model_name}"
            result, prediction, continuous, choices = evaluator(items, feature_name)
            result["outer_fold_alpha"] = choices; reports[name] = result
            for item, pred, value in zip(items, prediction, continuous):
                prediction_rows.append({"feature_set": name, "group": item["group"],
                                    "video": item["video"], "time": item["time"],
                                    "reference": item["stage"], "prediction": pred,
                                    "continuous_prediction": value})
    result = {"scope": "RH-only rising Reaction exact endpoints; complete run held out",
              "label_policy": {"place1_nominal_90": "70-80 partial, excluded",
                               "place2_90": "exact retained"},
              "models": reports, "trajectory_monotonicity": monotonic_audit(items),
              "deployment_target": ">=0.85 recall at every reported stage",
              "deployment_passed": any(all(value >= .85 for value in model["per_stage_recall"])
                                       for model in reports.values()),
              "n_endpoint_rows": len(items)}
    (args.output / "metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    with (args.output / "predictions.csv").open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    make_svg(args.output / "rh_human_color_path_validation.svg", items, reports)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
