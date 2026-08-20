"""Leakage-safe H2 2% versus 3% expert on verified high-quality runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ordinal_concentration_analysis import (
    apply_h2_verified_partial_response, assign_h2_ramp_targets,
    assign_rh20_h2_weak_targets, augment,
)
from train_models import CACHE_VERSION, read_csv


VERIFIED = {
    "1_90_H2_only_test.mp4", "1_90_H2_only_test_2.mp4",
    "1_90_H2_only_test_3.MOV",
}


def candidates():
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(
            C=.5, class_weight="balanced", max_iter=3000, random_state=42)),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
    }


def fit_kwargs(model, weight):
    if hasattr(model, "steps"):
        return {f"{model.steps[-1][0]}__sample_weight": weight}
    return {"sample_weight": weight}


def evaluate(strong, weak):
    rows = strong + weak
    x = np.asarray([augment(row, "flame") for row in rows])
    y = np.asarray([int(float(row["analysis_stage"])) for row in rows])
    groups = np.asarray([str(row["group"]) for row in rows])
    is_strong = np.asarray([index < len(strong) for index in range(len(rows))])
    reports = {}
    for name, estimator in candidates().items():
        prediction = np.full(len(rows), -1)
        for held_out in sorted(set(groups[is_strong])):
            train = groups != held_out
            test = (groups == held_out) & is_strong
            weights = np.asarray([float(row.get("sample_weight_factor", 1.0))
                                  for row in rows])[train]
            model = clone(estimator)
            model.fit(x[train], y[train], **fit_kwargs(model, weights))
            raw = np.asarray(model.predict(x[test]))
            prediction[test] = np.where(raw < 2.5, 2, 3)
        use = is_strong & (prediction >= 0)
        recall = {level: float(np.mean(prediction[use & (y == level)] == level))
                  for level in (2, 3)}
        reports[name] = {
            "exact_accuracy": float(np.mean(prediction[use] == y[use])),
            "balanced_accuracy": float(np.mean(list(recall.values()))),
            "recall": recall, "n_verified_frames": int(use.sum()),
            "n_weak_training_frames": len(weak),
        }
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features.csv"))
    parser.add_argument("--rh20-cache", type=Path)
    parser.add_argument("--rh20-weight", type=float, default=.002)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.cache); assign_h2_ramp_targets(rows)
    apply_h2_verified_partial_response(rows, "verified_run4_run5", .002)
    strong = [row for row in rows if str(row["video"]) in VERIFIED
              and row.get("analysis_phase") == "reaction"
              and int(float(row["analysis_stage"])) in (2, 3)]
    weak = []
    if args.rh20_cache:
        candidates_rows = read_csv(args.rh20_cache)
        assign_rh20_h2_weak_targets(
            candidates_rows, interior_weight=args.rh20_weight,
            progress_mode="reviewed", reviewed_quality_profile="user",
            reviewed_stages=(2, 3))
        weak = [row for row in candidates_rows if row.get("weak_supervision")
                and int(float(row["analysis_stage"])) in (2, 3)]
    report = evaluate(strong, weak)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
