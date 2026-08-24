"""Extract paired-pixel RH hue histograms and run endpoint-held-out A/B.

Unlike independent LAB channel quantiles, each hue observation here comes from
one actual sensing pixel inside the registered droplet mask.  The test uses
complete-run holdout and the verified place-1 high-RH ceiling policy.
"""

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
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from make_endpoint_mask_review import masks_for, source_path
from rh_human_color_path_analysis import ENDPOINTS, LEVELS, DISPLAY, PLACE1_PARTIAL
from train_models import CACHE_VERSION, patch_balance_lab, read_csv, resize_for_app


FEATURE_NAMES = (
    "legacy_delta_lab", "paired_named", "paired_hist_delta",
    "paired_hist_absolute_delta",
)
NAMED_BINS = ("yellow", "orange", "scarlet", "purple", "green")
HUE_EDGES = np.linspace(-180.0, 180.0, 13)
ENDPOINT_TOLERANCE_SECONDS = .75


def balanced(truth, prediction):
    present = [level for level in LEVELS if np.any(truth == level)]
    return float(np.mean([
        np.mean(prediction[truth == level] == level) for level in present
    ]))


def report(truth, prediction):
    matrix = confusion_matrix(truth, prediction, labels=LEVELS)
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


def group_stage_weights(groups, truth, use):
    weights = np.zeros(len(groups))
    for group in sorted(set(groups[use])):
        for stage in LEVELS:
            select = use & (groups == group) & (truth == stage)
            if np.any(select):
                weights[select] = 1.0 / np.sum(select)
    return weights


def endpoint_rows(cache_rows):
    by_video = defaultdict(list)
    for row in cache_rows:
        if row.get("kind") == "rh_only" and row.get("video") in ENDPOINTS:
            by_video[str(row["video"])].append(row)
    output = []
    for video, endpoints in ENDPOINTS.items():
        rows = by_video.get(video, [])
        for seconds, supplied in endpoints:
            if not rows or (video, float(seconds)) == PLACE1_PARTIAL:
                continue
            row = min(rows, key=lambda value: abs(float(value["time"]) - seconds))
            if abs(float(row["time"]) - float(seconds)) > ENDPOINT_TOLERANCE_SECONDS:
                # Never substitute a clip's last available frame for a stated
                # endpoint outside that clip (notably 189 s vs 180 s).
                continue
            output.append({
                "video": video, "group": str(row["group"]),
                "time": float(row["time"]),
                "requested_time": float(seconds),
                "stage": 25.0 if supplied in (20, 30) else float(supplied),
                "row": row,
            })
    return output


def balanced_frame_and_masks(frame, row):
    _, _, flame_mask, drop_mask, circle = masks_for(
        frame, row, 65.0, False, True)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    x, y, radius = circle
    yy, xx = np.ogrid[:lab.shape[0], :lab.shape[1]]
    chamber = (xx - x) ** 2 + (yy - y) ** 2 <= (radius * .90) ** 2
    return patch_balance_lab(lab, chamber), flame_mask, drop_mask


def paired_summary(lab, mask):
    pixels = lab[mask].astype(float)
    if len(pixels) < 12:
        raise RuntimeError("Too few selected sensing pixels")
    a, b = pixels[:, 1] - 128.0, pixels[:, 2] - 128.0
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    # Chroma weighting retains visibly coloured sensor pixels without deleting
    # the gray/low-chroma response.  A small floor keeps every selected pixel.
    weights = np.maximum(chroma, 3.0)
    histogram, _ = np.histogram(hue, bins=HUE_EDGES, weights=weights)
    histogram = histogram / max(float(np.sum(histogram)), 1e-9)
    bins = {
        "yellow": (hue >= 55) & (hue < 115),
        "orange": (hue >= 25) & (hue < 55),
        "scarlet": (hue >= -5) & (hue < 25),
        "purple": (hue >= -110) & (hue < -5),
        "green": (hue >= 115) & (hue <= 180),
    }
    named = np.asarray([
        float(np.sum(weights[select]) / np.sum(weights)) for select in bins.values()
    ])
    circular = np.asarray([
        float(np.sum(weights * np.sin(np.radians(hue))) / np.sum(weights)),
        float(np.sum(weights * np.cos(np.radians(hue))) / np.sum(weights)),
    ])
    chroma_stats = np.percentile(chroma, [25, 50, 75])
    lightness = np.percentile(pixels[:, 0], [25, 50, 75])
    return {"histogram": histogram, "named": named, "circular": circular,
            "chroma": chroma_stats, "lightness": lightness,
            "n_pixels": int(len(pixels))}


def extract(items, video_root):
    by_video = defaultdict(list)
    for index, item in enumerate(items):
        by_video[item["video"]].append((index, item))
    summaries = [None] * len(items)
    for video, indexed in by_video.items():
        path = source_path(video_root, video)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        for index, item in sorted(indexed, key=lambda pair: pair[1]["time"]):
            cap.set(cv2.CAP_PROP_POS_MSEC, item["time"] * 1000)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode {path.name} at {item['time']:.2f}s")
            frame = resize_for_app(frame)
            lab, _, drop = balanced_frame_and_masks(frame, item["row"])
            summaries[index] = paired_summary(lab, drop)
        cap.release()
        print(f"paired hue: {video} ({len(indexed)} endpoints)")
    return summaries


def build_features(items, summaries):
    groups = np.asarray([item["group"] for item in items])
    truth = np.asarray([item["stage"] for item in items])
    baseline = {}
    for group in sorted(set(groups)):
        use = np.where((groups == group) & (truth == 25))[0]
        baseline[group] = {
            name: np.median(np.asarray([summaries[index][name] for index in use]), axis=0)
            for name in ("histogram", "named", "circular", "chroma", "lightness")
        }
    matrices = defaultdict(list)
    audit = []
    for item, summary in zip(items, summaries):
        base = baseline[item["group"]]
        row = item["row"]
        legacy = np.asarray([float(row[f"drop_registered_{channel}"])
                             for channel in "Lab"])
        named_delta = summary["named"] - base["named"]
        hist_delta = summary["histogram"] - base["histogram"]
        common = np.concatenate([
            summary["circular"] - base["circular"],
            summary["chroma"] - base["chroma"],
            summary["lightness"] - base["lightness"],
        ])
        matrices["legacy_delta_lab"].append(legacy)
        matrices["paired_named"].append(np.concatenate([
            summary["named"], named_delta, common]))
        matrices["paired_hist_delta"].append(np.concatenate([hist_delta, common]))
        matrices["paired_hist_absolute_delta"].append(np.concatenate([
            summary["histogram"], hist_delta, summary["named"], named_delta,
            common]))
        audit.append({
            "group": item["group"], "video": item["video"],
            "time": item["time"], "reference": item["stage"],
            "n_pixels": summary["n_pixels"],
            **{name: float(value) for name, value in zip(NAMED_BINS, summary["named"])},
        })
    return {name: np.asarray(values) for name, values in matrices.items()}, audit


def tune_and_evaluate(x, truth, groups):
    prediction = np.full(len(truth), np.nan)
    confidence = np.full(len(truth), np.nan)
    choices = []
    for held_out in sorted(set(groups)):
        outer = groups != held_out
        best = None
        for C in (.01, .03, .1, .3, 1.0, 3.0):
            scores = []
            for inner in sorted(set(groups[outer])):
                train = outer & (groups != inner)
                test = outer & (groups == inner)
                weights = group_stage_weights(groups, truth, train)
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=C, max_iter=5000, class_weight="balanced", random_state=42))
                model.fit(x[train], truth[train],
                          logisticregression__sample_weight=weights[train])
                pred = model.predict(x[test])
                scores.append((balanced(truth[test], pred),
                               float(np.mean(truth[test] == pred)),
                               -float(np.mean(abs(truth[test] - pred)))))
            score = tuple(np.mean(scores, axis=0))
            if best is None or score > best[0]:
                best = (score, C)
        train, test = groups != held_out, groups == held_out
        weights = group_stage_weights(groups, truth, train)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=best[1], max_iter=5000, class_weight="balanced", random_state=42))
        model.fit(x[train], truth[train],
                  logisticregression__sample_weight=weights[train])
        probabilities = model.predict_proba(x[test])
        prediction[test] = model.classes_[np.argmax(probabilities, axis=1)]
        confidence[test] = np.max(probabilities, axis=1)
        choices.append({"held_out_group": held_out, "C": best[1]})
    return report(truth, prediction), prediction, confidence, choices


def deployment_decision(baseline, candidate):
    old = np.asarray(baseline["per_stage_recall"])
    new = np.asarray(candidate["per_stage_recall"])
    return {
        "improves_exact": candidate["exact_accuracy"] > baseline["exact_accuracy"],
        "improves_balanced": candidate["balanced_accuracy"] > baseline["balanced_accuracy"],
        "preserves_every_stage": bool(np.all(new >= old)),
        "improves_middle_stage": bool(np.any(new[1:5] > old[1:5])),
        "all_stage_recall_at_least_0.85": bool(np.all(new >= .85)),
    }


def plot(output, results, best_name):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for axis, name in zip(axes[:2], ("legacy_delta_lab", best_name)):
        matrix = np.asarray(results[name]["confusion"], dtype=float)
        norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        axis.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for row in range(7):
            for column in range(7):
                axis.text(column, row, f"{norm[row, column]:.2f}", ha="center",
                          va="center", fontsize=7,
                          color="white" if norm[row, column] > .55 else "black")
        axis.set_xticks(range(7), DISPLAY, rotation=35); axis.set_yticks(range(7), DISPLAY)
        axis.set(xlabel="Predicted RH", ylabel="Reference RH", title=name)
    names = list(results)
    x = np.arange(len(names)); width = .36
    axes[2].bar(x-width/2, [results[name]["exact_accuracy"] for name in names],
                width, label="Exact")
    axes[2].bar(x+width/2, [results[name]["balanced_accuracy"] for name in names],
                width, label="Balanced")
    axes[2].axhline(.85, color="crimson", linestyle="--", linewidth=1, label="0.85 target")
    axes[2].set_xticks(x, [name.replace("paired_", "") for name in names], rotation=30,
                       ha="right")
    axes[2].set_ylim(0, 1); axes[2].set(title="Complete-run-held-out A/B", ylabel="Score")
    axes[2].legend(fontsize=8)
    fig.suptitle("RH registered-droplet paired-pixel hue validation", fontweight="bold")
    fig.savefig(output / "rh_paired_pixel_hue_validation.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_paired_pixel_hue_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    items = endpoint_rows(read_csv(args.cache))
    summaries = extract(items, args.video_root)
    matrices, audit = build_features(items, summaries)
    truth = np.asarray([item["stage"] for item in items])
    groups = np.asarray([item["group"] for item in items])
    results, predictions = {}, []
    for name in FEATURE_NAMES:
        metrics, pred, confidence, choices = tune_and_evaluate(
            matrices[name], truth, groups)
        metrics["outer_fold_C"] = choices; results[name] = metrics
        for item, value, score in zip(items, pred, confidence):
            predictions.append({"feature_set": name, "group": item["group"],
                                "video": item["video"], "time": item["time"],
                                "reference": item["stage"], "prediction": value,
                                "confidence": score})
    candidates = [name for name in FEATURE_NAMES if name != "legacy_delta_lab"]
    best_name = max(candidates, key=lambda name: (
        results[name]["balanced_accuracy"], results[name]["exact_accuracy"]))
    decision = deployment_decision(results["legacy_delta_lab"], results[best_name])
    decision["selected_candidate"] = best_name
    decision["apply_to_app"] = bool(
        decision["improves_exact"] and decision["improves_balanced"]
        and decision["preserves_every_stage"] and decision["improves_middle_stage"]
        and decision["all_stage_recall_at_least_0.85"])
    decision["proceed_to_h2"] = decision["apply_to_app"]
    payload = {
        "scope": "RH-only rising Reaction endpoints; complete run held out",
        "label_policy": {"place1_nominal_90": "70-80 partial, excluded",
                         "place2_90": "exact retained"},
        "results": results, "decision": decision, "n_endpoints": len(items),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    for filename, rows in (("predictions.csv", predictions), ("pixel_audit.csv", audit)):
        with (args.output / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plot(args.output, results, best_name)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
