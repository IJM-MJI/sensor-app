"""Combine the corrected H2 model with a leakage-safe verified 2/3 expert."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from h2_23_pairwise_analysis import VERIFIED
from ordinal_concentration_analysis import (
    apply_h2_verified_partial_response, assign_h2_ramp_targets, augment,
)
from train_models import CACHE_VERSION, read_csv


def read_global(path: Path, model: str):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle)
                if row["task"] == "H2" and row["protocol"] == "video_holdout"
                and row["model"] == model]


def score(truth, prediction):
    cm = confusion_matrix(truth, prediction, labels=range(5))
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return float(np.mean(truth == prediction) + np.mean(recall)
                 + np.mean(np.abs(truth - prediction) <= 1)
                 - .25 * np.mean(np.abs(truth - prediction)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features.csv"))
    parser.add_argument("--global-predictions", type=Path, required=True)
    parser.add_argument("--global-model", default="ridge_flexible_rounded")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    cache = read_csv(args.cache); assign_h2_ramp_targets(cache)
    apply_h2_verified_partial_response(cache, "verified_run4_run5", .002)
    by_key = {(str(row["video"]), round(float(row["time"]), 4)): row for row in cache}
    training = [row for row in cache if str(row["video"]) in VERIFIED
                and row.get("analysis_phase") == "reaction"
                and int(float(row["analysis_stage"])) in (2, 3)]
    global_rows = read_global(args.global_predictions, args.global_model)

    expert_prediction = np.zeros(len(global_rows), dtype=int)
    groups = np.asarray([row["group"] for row in global_rows])
    for held_out in sorted(set(groups)):
        train = [row for row in training if str(row["group"]) != held_out]
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=.5, class_weight="balanced", max_iter=3000, random_state=42))
        model.fit(np.asarray([augment(row, "flame") for row in train]),
                  np.asarray([int(float(row["analysis_stage"])) for row in train]))
        indices = np.where(groups == held_out)[0]
        test = [by_key[(global_rows[index]["video"],
                        round(float(global_rows[index]["time"]), 4))] for index in indices]
        expert_prediction[indices] = model.predict(
            np.asarray([augment(row, "flame") for row in test])).astype(int)

    truth = np.asarray([int(float(row["reference"])) for row in global_rows])
    baseline = np.asarray([int(float(row["prediction"])) for row in global_rows])
    confidence = np.asarray([float(row["confidence"]) for row in global_rows])
    thresholds = (0.0, .3, .5, .7, .9, 1.1)
    scopes = {"both": (2, 3), "predicted_2": (2,), "predicted_3": (3,)}
    final = baseline.copy(); chosen = {}
    for held_out in sorted(set(groups)):
        meta_train = groups != held_out
        best_scope, best_threshold = max(
            ((scope, threshold) for scope in scopes for threshold in thresholds),
            key=lambda candidate: score(
                truth[meta_train], np.where(
                    np.isin(baseline[meta_train], scopes[candidate[0]])
                    & (confidence[meta_train] < candidate[1]),
                    expert_prediction[meta_train], baseline[meta_train])))
        chosen[held_out] = {"scope": best_scope, "threshold": best_threshold}
        use = ((groups == held_out) & np.isin(baseline, scopes[best_scope])
               & (confidence < best_threshold))
        final[use] = expert_prediction[use]

    cm = confusion_matrix(truth, final, labels=range(5))
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    report = {
        "protocol": "LOVO global + nested confidence-gated verified 2/3 expert",
        "threshold_by_held_out_group": chosen, "n_frames": len(final),
        "exact_accuracy": float(np.mean(final == truth)),
        "stage_balanced_accuracy": float(np.mean(recall)),
        "within_one_step": float(np.mean(np.abs(final - truth) <= 1)),
        "mae": float(np.mean(np.abs(final - truth))),
        "recall": recall.tolist(), "confusion": cm.tolist(),
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
