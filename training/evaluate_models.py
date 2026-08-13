"""Detailed leave-one-recording-group-out checks for the two region models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from train_models import (
    CACHE_VERSION, H2_FEATURES, RH_FEATURES, binary_model, merge_training_rows,
    model_matrix, read_csv, read_legacy_continuous,
)


def evaluate(rows, label: str, features: list[str]) -> tuple[dict, dict[int, float]]:
    use, x, y, groups = model_matrix(rows, label, features)
    probability = np.full(len(y), np.nan)
    for group in sorted(set(groups)):
        test = groups == group
        train = ~test
        model = clone(binary_model()).fit(x[train], y[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
    prediction = probability >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    summary = {
        "frames": len(y),
        "groups": len(set(groups)),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "auc": roc_auc_score(y, probability),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "precision": tp / (tp + fp),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "by_kind": {},
        "by_group": {},
    }
    for field in ("kind", "group"):
        target = summary[f"by_{field}"]
        values = np.asarray([str(row[field]) for row in use])
        for value in sorted(set(values)):
            mask = values == value
            target[value] = {
                "frames": int(mask.sum()),
                "accuracy": float(np.mean(y[mask] == prediction[mask])),
                "actual_positive_rate": float(np.mean(y[mask])),
                "predicted_positive_rate": float(np.mean(prediction[mask])),
            }
    return summary, {id(row): float(p) for row, p in zip(use, probability)}


def main() -> None:
    rows = read_csv(Path(f"training/cache/{CACHE_VERSION}/features.csv"))
    new_rows = rows
    rows = merge_training_rows(
        new_rows,
        read_legacy_continuous(Path(f"training/cache/{CACHE_VERSION}/legacy_continuous.csv")),
    )
    h2, h2_probability = evaluate(new_rows, "h2_present", H2_FEATURES)
    rh, rh_probability = evaluate(rows, "rh_high", RH_FEATURES)
    clear_truth, clear_prediction = [], []
    for row in new_rows:
        if row["kind"] == "h2_only" and row["h2_present"] is not None:
            truth = (int(row["h2_present"]), 0)
            prediction = (int(h2_probability[id(row)] >= 0.5), 0)
        elif row["kind"] == "rh_only" and row["rh_high"] is not None:
            if id(row) not in rh_probability:
                continue
            truth = (0, int(row["rh_high"]))
            prediction = (
                int(h2_probability[id(row)] >= 0.5),
                int(rh_probability[id(row)] >= 0.5),
            )
        else:
            continue
        clear_truth.append(truth)
        clear_prediction.append(prediction)
    report = {
        "h2_presence": h2,
        "rh_high": rh,
        "clear_state_combination": {
            "frames": len(clear_truth),
            "exact_accuracy": float(np.mean(np.all(np.asarray(clear_truth) == np.asarray(clear_prediction), axis=1))),
            "note": "H2-only and RH-only labelled extremes; simultaneous RH setpoints excluded.",
        },
    }
    output = Path("training/output/detailed_metrics.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
