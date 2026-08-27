"""Evaluate H2 environment routing with app-mask flame distribution features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train_models import extract_features, frame_at


RUNS = {
    "1_90_H2_only_test_2.mp4": ("A", (0, 3)),
    "1_90_H2_only_test_3.MOV": ("A", (0, 2.5)),
    "1_90_H2_only_test.mp4": ("A", (0, 3)),
    "1_90_RH20_2_x2.mp4": ("A", (10, 14)),
    "1_90_RH20_3_x2.mp4": ("B", (0, 4)),
    "1_90_RH20_4_x2.mp4": ("B", (0, 8)),
    "1_90_RH20_5_x2.mp4": ("C", (0, 4)),
}
NAMES = [
    "flame_L", "flame_a", "flame_b",
    "flame_L_p50", "flame_a_p25", "flame_a_p50", "flame_a_p75",
    "flame_b_p25", "flame_b_p50", "flame_b_p75",
    "flame_chroma_p25", "flame_chroma_p50", "flame_chroma_p75",
]
FEATURES = {
    "mean": (0, 1, 2),
    "ab_distribution": (1, 2, 4, 5, 6, 7, 8, 9),
    "global10": tuple(range(10)),
    "all13": tuple(range(13)),
}


def metadata(cache: Path):
    output = {}
    with cache.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["video"] in RUNS and row["video"] not in output:
                output[row["video"]] = {
                    "circle": (int(float(row["circle_x"])), int(float(row["circle_y"])),
                               int(float(row["circle_r"]))),
                    "orientation": int(float(row.get("orientation_quarters") or 0)),
                }
    return output


def extract(video_root: Path, source_cache: Path, cache: Path):
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        return saved["x"], saved["y"], saved["groups"]
    meta = metadata(source_cache); rows = []
    for video, (family, window) in RUNS.items():
        cap = cv2.VideoCapture(str(video_root / video))
        if not cap.isOpened():
            raise FileNotFoundError(video_root / video)
        for seconds in np.linspace(window[0], window[1], 8):
            values = extract_features(frame_at(cap, float(seconds)), meta[video]["circle"],
                                      meta[video]["orientation"])
            rows.append(([float(values[name]) for name in NAMES], family, video))
        cap.release()
    x = np.asarray([row[0] for row in rows]); y = np.asarray([row[1] for row in rows])
    groups = np.asarray([row[2] for row in rows])
    np.savez_compressed(cache, x=x, y=y, groups=groups)
    return x, y, groups


def estimator(kind):
    if kind == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"))
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.2, class_weight="balanced", random_state=42))
    return make_pipeline(StandardScaler(), SVC(
        C=1, gamma="scale", class_weight="balanced", random_state=42))


def evaluate(x, y, groups, index, kind):
    use = np.isin(y, ("A", "B")); x, y, groups = x[use][:, index], y[use], groups[use]
    prediction = np.full(len(y), "?", dtype="<U1")
    for held_out in sorted(set(groups)):
        test, train = groups == held_out, groups != held_out
        model = estimator(kind); model.fit(x[train], y[train])
        prediction[test] = model.predict(x[test])
    matrix = confusion_matrix(y, prediction, labels=("A", "B"))
    recall = np.diag(matrix) / matrix.sum(axis=1)
    per_run = {run: float(np.mean(prediction[groups == run] == y[groups == run]))
               for run in sorted(set(groups))}
    return {"exact": float(np.mean(prediction == y)),
            "video_macro_exact": float(np.mean(list(per_run.values()))),
            "recall": {"A": float(recall[0]), "B": float(recall[1])},
            "minimum_recall": float(recall.min()), "confusion": matrix.tolist(),
            "per_run_exact": per_run}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    x, y, groups = extract(args.video_root, args.source_cache,
                           args.output / "app_router_distribution_rows.npz")
    results = {}
    for feature_name, index in FEATURES.items():
        for kind in ("lda", "logistic", "svm"):
            results[f"{feature_name}_{kind}"] = evaluate(x, y, groups, index, kind)
    selected = max(results, key=lambda name: (
        results[name]["minimum_recall"], results[name]["video_macro_exact"],
        results[name]["exact"]))
    best = results[selected]
    payload = {"protocol": "current app mask; initial frames; complete video held out",
               "features": NAMES, "selected": selected, "A_B": best,
               "C_independent_validation": False, "models": results}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    matrix = np.asarray(best["confusion"])
    fig, axis = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(xticks=(0, 1), xticklabels=("A", "B"), yticks=(0, 1), yticklabels=("A", "B"),
             xlabel="Predicted", ylabel="Reference",
             title=f"App-mask distribution router\naccuracy {best['exact']:.1%}")
    fig.savefig(args.output / "distribution_router_confusion.png", dpi=190); plt.close(fig)
    print(json.dumps({"selected": selected, **best}, indent=2))


if __name__ == "__main__":
    main()
