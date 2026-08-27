"""Validate an H2 A/B environment router in the deployed app feature domain."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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


RUNS = {
    "1_90_H2_only_test_2.mp4": "A",
    "1_90_H2_only_test_3.MOV": "A",
    "1_90_H2_only_test.mp4": "A",
    "1_90_RH20_2_x2.mp4": "A",
    "1_90_RH20_3_x2.mp4": "B",
    "1_90_RH20_4_x2.mp4": "B",
    "1_90_RH20_5_x2.mp4": "C",
}
BASE = [f"baseline_{region}_{channel}" for region in ("flame", "drop", "top")
        for channel in ("L", "a", "b")]
FEATURES = {
    "flame": (0, 1, 2),
    "flame_drop": (0, 1, 2, 3, 4, 5),
    "all_colour": tuple(range(9)),
    "colour_geometry": tuple(range(13)),
}


def load(path: Path):
    by_video = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            video = row["video"]
            if video not in RUNS or video in by_video:
                continue
            colour = [float(row[name]) for name in BASE]
            geometry = [float(row["circle_x"]) / float(row["width"]),
                        float(row["circle_y"]) / float(row["height"]),
                        float(row["circle_r"]) / min(float(row["width"]), float(row["height"])),
                        float(row.get("orientation_quarters") or 0)]
            by_video[video] = np.asarray(colour + geometry)
    videos = np.asarray(sorted(by_video))
    return np.asarray([by_video[video] for video in videos]), np.asarray(
        [RUNS[video] for video in videos]), videos


def estimator(kind):
    if kind == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"))
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.2, class_weight="balanced", random_state=42))
    return make_pipeline(StandardScaler(), SVC(
        C=1, gamma="scale", class_weight="balanced", random_state=42))


def evaluate(x, y, videos, feature, kind):
    use = np.isin(y, ("A", "B")); x, y, videos = x[use][:, feature], y[use], videos[use]
    prediction = np.full(len(y), "?", dtype="<U1")
    for index in range(len(y)):
        train = np.arange(len(y)) != index
        model = estimator(kind); model.fit(x[train], y[train])
        prediction[index] = model.predict(x[index:index + 1])[0]
    matrix = confusion_matrix(y, prediction, labels=("A", "B"))
    recall = np.diag(matrix) / matrix.sum(axis=1)
    return {"exact": float(np.mean(prediction == y)),
            "recall": {"A": float(recall[0]), "B": float(recall[1])},
            "minimum_recall": float(recall.min()), "confusion": matrix.tolist(),
            "per_video": {video: {"reference": reference, "predicted": predicted}
                          for video, reference, predicted in zip(videos, y, prediction)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    x, y, videos = load(args.cache)
    results = {}
    for feature_name, feature in FEATURES.items():
        for kind in ("lda", "logistic", "svm"):
            results[f"{feature_name}_{kind}"] = evaluate(x, y, videos, feature, kind)
    selected = max(results, key=lambda name: (
        results[name]["minimum_recall"], results[name]["exact"]))
    best = results[selected]

    # C is represented by one physical run; report distances but do not call it validation.
    feature_name = selected.rsplit("_", 1)[0]; index = FEATURES[feature_name]
    z = x[:, index]
    mean = z[y != "C"].mean(axis=0); scale = np.maximum(z[y != "C"].std(axis=0), .25)
    centers = {label: z[y == label].mean(axis=0) for label in ("A", "B", "C")}
    c_distance = {label: float(np.linalg.norm((centers["C"] - center) / scale))
                  for label, center in centers.items() if label != "C"}
    payload = {"protocol": "one calibration baseline per video; leave-one-video-out",
               "selected": selected, "A_B": best,
               "C": {"independent_validation": False,
                     "reason": "one physical run", "distance_to_A_B": c_distance},
               "models": results}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    matrix = np.asarray(best["confusion"])
    fig, axis = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    axis.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(xticks=(0, 1), xticklabels=("A", "B"), yticks=(0, 1), yticklabels=("A", "B"),
             xlabel="Predicted", ylabel="Reference",
             title=f"App-domain environment router\naccuracy {best['exact']:.1%}")
    fig.savefig(args.output / "app_router_confusion.png", dpi=190); plt.close(fig)
    print(json.dumps({"selected": selected, "A_B": best, "C": payload["C"]}, indent=2))


if __name__ == "__main__":
    main()
