"""Audit fixed model families for the weak response3 middle-RH signature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from rh_40_50_cross_run_spatial_analysis import build, extract
from rh_four_band_analysis import STAGES, band, score
from rh_place2_time_warp_analysis import RUNS, shifted_endpoint_items
from train_models import CACHE_VERSION, read_csv


MODELS = {
    "nearest_centroid": make_pipeline(StandardScaler(), NearestCentroid()),
    "knn_1": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)),
    "knn_3_distance": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=3, weights="distance")),
    "lda_shrinkage": make_pipeline(
        StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    "gaussian_nb": make_pipeline(StandardScaler(), GaussianNB(var_smoothing=1e-8)),
    "svc_linear": make_pipeline(
        StandardScaler(), SVC(C=1.0, kernel="linear", class_weight="balanced")),
    "svc_rbf": make_pipeline(
        StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")),
    "random_forest": RandomForestClassifier(
        n_estimators=500, max_depth=3, class_weight="balanced", random_state=42),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=500, max_depth=3, class_weight="balanced", random_state=42),
}


def evaluate(model, x, truth, groups):
    prediction = np.zeros_like(truth)
    folds = []
    for held in RUNS:
        train, test = groups != held, groups == held
        fitted = clone(model).fit(x[train], truth[train])
        prediction[test] = fitted.predict(x[test])
        folds.append({"held": held, **score(truth[test], prediction[test])})
    return prediction, {**score(truth, prediction), "folds": folds}


def plot(output, ranked):
    top = ranked[:12]; labels = [row["candidate"] for row in top]
    x = np.arange(len(top)); width = .36
    fig, axis = plt.subplots(figsize=(12.0, 5.2), constrained_layout=True)
    axis.bar(x-width/2, [row["exact_accuracy"] for row in top], width, label="Exact")
    axis.bar(x+width/2, [min(row["per_band_recall"]) for row in top], width,
             label="Minimum band recall")
    axis.axhline(.85, color="crimson", linestyle="--", linewidth=1)
    axis.set_xticks(x, [label.replace("::", "\n") for label in labels],
                    rotation=35, ha="right")
    axis.set_ylim(0, 1); axis.grid(axis="y", alpha=.2); axis.legend()
    axis.set(title="Place-2 fixed model-family audit (response6 70% at 16.67 s)",
             ylabel="Complete-run-held-out score")
    fig.savefig(output / "rh_place2_model_family_audit.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/rh_place2_model_family_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cache = read_csv(args.cache)
    items = shifted_endpoint_items(cache, {("rh-response-6", 70.0): 16.67})
    summaries = extract(items, args.video_root)
    matrices, audit = build(items, summaries, STAGES)
    truth = np.asarray([band(row["reference"]) for row in audit])
    groups = np.asarray([row["group"] for row in audit])
    ranked = []
    for feature_name, matrix in matrices.items():
        for model_name, model in MODELS.items():
            _, metrics = evaluate(model, matrix, truth, groups)
            ranked.append({"candidate": f"{feature_name}::{model_name}", **metrics})
    ranked.sort(key=lambda row: (min(row["per_band_recall"]),
                                 row["balanced_accuracy"], row["exact_accuracy"]),
                reverse=True)
    best = ranked[0]
    decision = {
        "passes_0_85_all_classes": bool(best["exact_accuracy"] >= .85
                                         and best["balanced_accuracy"] >= .85
                                         and min(best["per_band_recall"]) >= .85),
        "app_deploy": False,
        "reason": "Model-family selection uses the same two diagnostic runs; an independent run is required.",
    }
    payload = {"scope": "response3 original endpoints; response6 70% moved 18.0 to 16.67 s",
               "best": best, "decision": decision, "ranked": ranked}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(args.output, ranked)
    print(json.dumps({"best": best, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
