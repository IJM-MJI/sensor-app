"""Cross-run RH40/50 A/B using spatial colour inside the large droplet."""

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

from make_endpoint_mask_review import source_path
from rh_paired_pixel_hue_analysis import balanced_frame_and_masks, endpoint_rows, paired_summary
from rh_tight_relative_analysis import angle, tight_masks
from train_models import CACHE_VERSION, normalized_coordinates, read_csv, resize_for_app


LEVELS = np.asarray([40.0, 50.0])
REGIONS = ("whole", "core", "rim", "top", "bottom", "left", "right")


def local_coordinates(shape, circle, row):
    nx, ny = normalized_coordinates(shape, circle, int(float(row["orientation_quarters"])))
    values = [row.get(f"drop_registration_{name}") for name in ("x", "y", "angle")]
    if all(value not in (None, "") for value in values):
        cx, cy, theta = (float(value) for value in values)
        dx, dy = nx - cx, ny - cy
        c, s = np.cos(theta), np.sin(theta)
        return c * dx - s * dy, s * dx + c * dy
    return nx + .08, ny - .43


def safe_summary(lab, mask):
    return paired_summary(lab, mask) if int(mask.sum()) >= 12 else None


def extract(items, video_root):
    by_video = defaultdict(list)
    for index, item in enumerate(items):
        by_video[item["video"]].append((index, item))
    output = [None] * len(items)
    for video, indexed in by_video.items():
        cap = cv2.VideoCapture(str(source_path(video_root, video)))
        for index, item in sorted(indexed, key=lambda pair: pair[1]["time"]):
            cap.set(cv2.CAP_PROP_POS_MSEC, item["time"] * 1000)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode {video} at {item['time']:.2f}s")
            frame = resize_for_app(frame)
            lab, _, selected = balanced_frame_and_masks(frame, item["row"])
            circle = tuple(int(float(item["row"][name]))
                           for name in ("circle_x", "circle_y", "circle_r"))
            main, _ = tight_masks(lab.shape, circle, item["row"], selected)
            lx, ly = local_coordinates(lab.shape, circle, item["row"])
            radius = np.sqrt((lx / .215) ** 2 + (ly / .245) ** 2)
            zones = {
                "whole": main,
                "core": main & (radius <= .58),
                "rim": main & (radius > .58),
                "top": main & (ly <= 0),
                "bottom": main & (ly > 0),
                "left": main & (lx <= 0),
                "right": main & (lx > 0),
            }
            output[index] = {name: safe_summary(lab, mask) for name, mask in zones.items()}
            output[index]["counts"] = {name: int(mask.sum()) for name, mask in zones.items()}
        cap.release()
    return output


def vector(summary, baseline):
    if summary is None or baseline is None:
        return np.zeros(19), 0.0
    hue = np.arctan2(np.sin(angle(summary) - angle(baseline)),
                     np.cos(angle(summary) - angle(baseline)))
    values = np.concatenate([
        [hue], summary["circular"] - baseline["circular"],
        summary["chroma"] - baseline["chroma"],
        summary["lightness"] - baseline["lightness"],
        summary["named"] - baseline["named"],
        summary["named"],
    ])
    return values, 1.0


def build(items, summaries):
    groups = np.asarray([item["group"] for item in items])
    truth = np.asarray([item["stage"] for item in items])
    baseline = {}
    for group in sorted(set(groups)):
        indices = np.where((groups == group) & (truth == 25))[0]
        baseline[group] = {}
        for region in REGIONS:
            valid = [summaries[index][region] for index in indices
                     if summaries[index][region] is not None]
            baseline[group][region] = None if not valid else {
                key: np.median(np.asarray([value[key] for value in valid]), axis=0)
                for key in ("circular", "chroma", "lightness", "named")}
    matrices = defaultdict(list); audit = []
    for item, summary in zip(items, summaries):
        if item["stage"] not in LEVELS:
            continue
        whole, _ = vector(summary["whole"], baseline[item["group"]]["whole"])
        spatial, present = [], []
        for region in REGIONS:
            values, flag = vector(summary[region], baseline[item["group"]][region])
            spatial.extend(values); present.append(flag)
        matrices["whole_relative"].append(whole)
        matrices["spatial_relative"].append(np.concatenate([spatial, present]))
        audit.append({"group": item["group"], "video": item["video"],
                      "time": item["time"], "reference": item["stage"],
                      **{f"{region}_pixels": summary["counts"][region]
                         for region in REGIONS}})
    return {name: np.asarray(value) for name, value in matrices.items()}, audit


def evaluate(x, truth, groups):
    prediction = np.zeros_like(truth)
    choices = []
    for held in sorted(set(groups)):
        outer = groups != held
        best = None
        for c_value in (.001, .003, .01, .03, .1, .3, 1.0):
            scores = []
            for inner in sorted(set(groups[outer])):
                train = outer & (groups != inner); test = outer & (groups == inner)
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=c_value, class_weight="balanced", max_iter=5000, random_state=42))
                model.fit(x[train], truth[train]); pred = model.predict(x[test])
                recalls = [np.mean(pred[truth[test] == level] == level)
                           for level in LEVELS if np.any(truth[test] == level)]
                scores.append((np.mean(recalls), np.mean(pred == truth[test])))
            score = tuple(np.mean(scores, axis=0))
            if best is None or score > best[0]:
                best = (score, c_value)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=best[1], class_weight="balanced", max_iter=5000, random_state=42))
        model.fit(x[outer], truth[outer]); prediction[~outer] = model.predict(x[~outer])
        choices.append({"held_out_group": held, "C": best[1]})
    matrix = confusion_matrix(truth, prediction, labels=LEVELS)
    recalls = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    return prediction, {"exact_accuracy": float(np.mean(truth == prediction)),
                        "balanced_accuracy": float(np.mean(recalls)),
                        "per_class_recall": recalls.tolist(),
                        "confusion": matrix.tolist(), "outer_fold_C": choices}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_40_50_cross_run_spatial_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    items = [item for item in endpoint_rows(read_csv(args.cache)) if item["stage"] in (25, 40, 50)]
    summaries = extract(items, args.video_root)
    matrices, audit = build(items, summaries)
    truth = np.asarray([row["reference"] for row in audit])
    groups = np.asarray([row["group"] for row in audit])
    results, predictions = {}, []
    for name, matrix in matrices.items():
        pred, metrics = evaluate(matrix, truth, groups); results[name] = metrics
        predictions.extend({"feature_set": name, **row, "prediction": value}
                           for row, value in zip(audit, pred))
    decision = {"selected": max(results, key=lambda name: results[name]["balanced_accuracy"]),
                "app_deploy": False, "required_accuracy": .85,
                "reason": "Independent held-out-run threshold not yet passed."}
    selected = decision["selected"]
    decision["passes_score_only"] = bool(
        results[selected]["balanced_accuracy"] >= .85
        and min(results[selected]["per_class_recall"]) >= .85)
    payload = {"scope": "RH40 vs RH50 exact endpoints; every run held out",
               "results": results, "decision": decision}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for filename, rows in (("roi_audit.csv", audit), ("predictions.csv", predictions)):
        with (args.output / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), constrained_layout=True)
    for axis, name in zip(axes, results):
        matrix = np.asarray(results[name]["confusion"], float)
        norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        axis.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2): axis.text(c, r, f"{norm[r,c]:.2f}", ha="center", va="center")
        axis.set_xticks((0,1),("40","50")); axis.set_yticks((0,1),("40","50"))
        axis.set(xlabel="Predicted RH", ylabel="Reference RH", title=name)
    fig.suptitle("Cross-run RH40/50 large-droplet spatial A/B", fontweight="bold")
    fig.savefig(args.output / "rh_40_50_cross_run_spatial_validation.png", dpi=220)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
