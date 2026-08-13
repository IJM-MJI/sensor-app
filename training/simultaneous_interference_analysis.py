"""Diagnose and correct H2 colour leakage into the simultaneous droplet ROI.

The correction is learned only from H2-only recordings: flame changes predict the
apparent droplet change caused by H2/camera coupling.  That predicted component is
subtracted before applying the H2O-only RH calibration.  Simultaneous nominal RH is
used only to inspect ordering; it is never a regression target.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_models import CACHE_VERSION, read_csv


FLAME = ["flame_L", "flame_a", "flame_b"]
DROP = ["drop_L", "drop_a", "drop_b"]


def matrix(rows, names):
    return np.asarray([[float(row[name]) for name in names] for row in rows])


def fit_h2_interference(rows):
    use = [row for row in rows if row["kind"] == "h2_only" and row.get("h2_value") is not None]
    model = make_pipeline(StandardScaler(), Ridge(alpha=20.0)).fit(matrix(use, FLAME), matrix(use, DROP))
    return model, use


def fit_rh_calibration(rows):
    use = [row for row in rows if row["kind"] == "rh_only" and row.get("rh_value") is not None
           and 70 <= float(row["rh_value"]) <= 90]
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
        matrix(use, DROP), np.asarray([float(row["rh_value"]) for row in use]))
    return model, use


def h2_only_lovo(rows):
    groups = sorted({str(row["group"]) for row in rows})
    truth, raw, corrected = [], [], []
    for group in groups:
        train = [row for row in rows if str(row["group"]) != group]
        test = [row for row in rows if str(row["group"]) == group]
        if not train or not test:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=20.0)).fit(matrix(train, FLAME), matrix(train, DROP))
        predicted = model.predict(matrix(test, FLAME))
        observed = matrix(test, DROP)
        truth.extend([group] * len(test))
        raw.extend(np.linalg.norm(observed, axis=1))
        corrected.extend(np.linalg.norm(observed - predicted, axis=1))
    return {
        "groups": len(set(truth)),
        "raw_drop_shift_median": float(np.median(raw)),
        "residual_drop_shift_median": float(np.median(corrected)),
        "residual_reduction_fraction": float(1 - np.median(corrected) / max(np.median(raw), 1e-9)),
    }


def main():
    root = Path("training/output/simultaneous_interference")
    root.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path("training/cache") / CACHE_VERSION / "features.csv")
    angle = Path("training/cache") / CACHE_VERSION / "angle_runs.csv"
    if angle.exists():
        rows += read_csv(angle)

    interference, h2_rows = fit_h2_interference(rows)
    rh_model, rh_rows = fit_rh_calibration(rows)
    simultaneous = [row for row in rows if row["kind"] == "simultaneous"
                    and row.get("state") in ("h2_only_condition", "simultaneous_condition")
                    and row.get("rh_setpoint") is not None and float(row["rh_setpoint"]) <= 80]
    flame = matrix(simultaneous, FLAME)
    drop = matrix(simultaneous, DROP)
    predicted_h2_drop = interference.predict(flame)

    output = []
    for row, raw_drop, h2_drop in zip(simultaneous, drop, predicted_h2_drop):
        raw_rh = float(rh_model.predict(raw_drop.reshape(1, -1))[0])
        corrected_drop = raw_drop - h2_drop
        corrected_rh = float(rh_model.predict(corrected_drop.reshape(1, -1))[0])
        output.append({
            "video": row["video"], "group": row["group"], "time": row["time"],
            "nominal_rh_setpoint_metadata": row["rh_setpoint"],
            "raw_h2o_only_equivalent_rh": raw_rh,
            "corrected_h2o_only_equivalent_rh": corrected_rh,
            **{f"predicted_h2_drop_{name[-1]}": float(value) for name, value in zip(DROP, h2_drop)},
        })
    with (root / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)

    buckets = defaultdict(list)
    for row in output:
        buckets[float(row["nominal_rh_setpoint_metadata"])].append(row)
    summary = []
    for nominal, values in sorted(buckets.items()):
        summary.append({
            "nominal_rh_setpoint_metadata": nominal, "n": len(values),
            "raw_median": float(np.median([row["raw_h2o_only_equivalent_rh"] for row in values])),
            "corrected_median": float(np.median([row["corrected_h2o_only_equivalent_rh"] for row in values])),
        })
    x = np.asarray([row["nominal_rh_setpoint_metadata"] for row in summary])
    raw = np.asarray([row["raw_median"] for row in summary])
    corrected = np.asarray([row["corrected_median"] for row in summary])
    report = {
        "policy": "H2 interference learned only from H2-only; RH learned only from H2O-only; simultaneous setpoint is diagnostic metadata",
        "h2_only_interference_lovo": h2_only_lovo(h2_rows),
        "simultaneous_level_summary": summary,
        "diagnostic_ordering_only": {
            "raw_spearman": float(spearmanr(x, raw).statistic),
            "corrected_spearman": float(spearmanr(x, corrected).statistic),
            "warning": "not an RH accuracy metric",
        },
    }
    (root / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(x, raw, "o-", label="Direct H2O-only transfer", color="#0072B2")
    ax.plot(x, corrected, "o-", label="After H2-only interference subtraction", color="#009E73")
    ax.set(xlabel="Nominal simultaneous RH setpoint (metadata only, %)",
           ylabel="H2O-only-equivalent RH estimate (%)",
           title="Simultaneous RH interference diagnostic")
    ax.legend(frameon=False)
    ax.text(.02, .02, "Setpoint is not optical RH ground truth", transform=ax.transAxes, fontsize=8)
    for suffix, kwargs in (("png", {"dpi": 400}), ("pdf", {}), ("svg", {})):
        fig.savefig(root / f"interference_correction.{suffix}", **kwargs)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
