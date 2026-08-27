"""Evaluate calibration-frame routing to H2 environment-specific models."""

from __future__ import annotations

import argparse
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

from h2_environment_family_analysis import FAMILIES
from h2_spatial_flame_analysis import VideoContext, spatial_summary
from h2_more_crop_fixed_mask_analysis import substrate
from make_h2_environment_review import environment_feature


WINDOWS = {
    "test_2": (0, 3), "test_3": (0, 2.5), "test": (0, 3),
    "run2": (10, 14), "run3": (0, 4), "run4": (0, 8),
    "run5_x2": (0, 4), "run5_normal": (0, 6),
}
FAMILY_ID = {run: family for family, runs in FAMILIES.items() for run in runs}
FEATURES = {
    "frame": tuple(range(42)),
    "flame": tuple(range(42, 52)),
    "frame_flame": tuple(range(52)),
    "all": tuple(range(99)),
}


def feature(context: VideoContext, seconds: float) -> np.ndarray:
    image = context.image(seconds)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
    background = substrate(lab, context.card, context.mask | context.drop_zone)
    spatial = spatial_summary(lab, context.region_masks, background)
    yy, xx = np.indices(context.mask.shape)
    geometry = np.asarray([
        context.mask.mean(), xx[context.mask].mean() / context.mask.shape[1],
        yy[context.mask].mean() / context.mask.shape[0],
        context.mask.shape[1] / context.mask.shape[0],
    ])
    return np.r_[environment_feature(image), spatial, background, geometry]


def extract(video_root: Path, cache: Path):
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        return saved["x"], saved["y"], saved["groups"]
    rows = []
    for run, (start, end) in WINDOWS.items():
        context = VideoContext(video_root, run)
        for seconds in np.linspace(start, end, 8):
            rows.append((feature(context, float(seconds)), FAMILY_ID[run], run))
        context.close()
    x = np.asarray([row[0] for row in rows])
    y = np.asarray([row[1] for row in rows])
    groups = np.asarray([row[2] for row in rows])
    np.savez_compressed(cache, x=x, y=y, groups=groups)
    return x, y, groups


def estimator(kind):
    if kind == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"))
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.2, max_iter=4000, class_weight="balanced", random_state=42))
    return make_pipeline(StandardScaler(), SVC(
        C=1, gamma="scale", class_weight="balanced", random_state=42))


def metrics(y, pred, groups, labels):
    matrix = confusion_matrix(y, pred, labels=labels)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    per_run = {run: float(np.mean(pred[groups == run] == y[groups == run]))
               for run in sorted(set(groups))}
    return {"exact": float(np.mean(pred == y)),
            "video_macro_exact": float(np.mean(list(per_run.values()))),
            "recall": {labels[i]: float(recall[i]) for i in range(len(labels))},
            "minimum_recall": float(recall.min()), "confusion": matrix.tolist(),
            "per_run_exact": per_run}


def held_out(x, y, groups, family_labels, feature_index, kind):
    use = np.isin(y, family_labels)
    x, y, groups = x[use][:, feature_index], y[use], groups[use]
    pred = np.full(len(y), "?", dtype="<U1")
    for held_out_run in sorted(set(groups)):
        test, train = groups == held_out_run, groups != held_out_run
        if set(y[train]) != set(family_labels):
            return None
        model = estimator(kind); model.fit(x[train], y[train])
        pred[test] = model.predict(x[test])
    return metrics(y, pred, groups, family_labels)


def sweep(x, y, groups, labels):
    results = {}
    for feature_name, index in FEATURES.items():
        for kind in ("lda", "logistic", "svm"):
            result = held_out(x, y, groups, labels, index, kind)
            if result is not None:
                results[f"{feature_name}_{kind}"] = result
    selected = max(results, key=lambda name: (
        results[name]["minimum_recall"], results[name]["video_macro_exact"],
        results[name]["exact"]))
    return selected, results[selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    x, y, groups = extract(args.video_root, args.output / "calibration_router_rows.npz")
    ab_name, ab = sweep(x, y, groups, ("A", "B"))
    abc_name, abc = sweep(x, y, groups, ("A", "B", "C"))
    payload = {
        "protocol": "calibration/initial frames only; complete video held out",
        "A_B_independent": {"selected": ab_name, **ab},
        "A_B_C_technical": {
            "note": "C is optimistic because normal/x2 are the same physical recording",
            "selected": abc_name, **abc,
        },
        "C_independent_validation": "unavailable: only one physical C run",
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    for axis, title, result, labels in (
        (axes[0], "A/B independent", ab, ("A", "B")),
        (axes[1], "A/B/C technical", abc, ("A", "B", "C")),
    ):
        matrix = np.asarray(result["confusion"]); axis.imshow(matrix, cmap="Blues")
        for row in range(len(labels)):
            for column in range(len(labels)):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set(xticks=range(len(labels)), xticklabels=labels,
                 yticks=range(len(labels)), yticklabels=labels,
                 xlabel="Predicted", ylabel="Reference",
                 title=f"{title}\naccuracy {result['exact']:.1%}")
    fig.savefig(args.output / "environment_router_confusions.png", dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
