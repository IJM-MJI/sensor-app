"""Apply independently calibrated H2/RH regressions to simultaneous reactions."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from train_models import CACHE_VERSION, feature_value, read_csv


def estimate(model, row):
    return float(model["intercept"] + sum(c * feature_value(row, f)
                 for c, f in zip(model["coefficients"], model["features"])))


def main():
    root = Path("training/output")
    models = json.loads((root / "models.json").read_text(encoding="utf-8"))
    rows = read_csv(Path("training/cache") / CACHE_VERSION / "features.csv")
    output = []
    for row in rows:
        if row["kind"] != "simultaneous" or row.get("state") not in ("h2_only_condition", "simultaneous_condition"):
            continue
        h2 = estimate(models["h2_concentration"], row)
        rh = estimate(models["rh_regression"], row)
        output.append({
            "video": row["video"], "group": row["group"], "time": row["time"],
            "nominal_rh_setpoint": row["rh_setpoint"], "state_label": row["state"],
            "h2_raw": h2, "h2_display": min(4, max(3, h2)) if 3 <= h2 <= 4.5 else "",
            "rh_h2o_only_equivalent_raw": rh,
            "rh_display": min(90, max(70, rh)) if 65 <= rh <= 95 else "",
        })
    out_dir = root / "simultaneous_quantitation"; out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    buckets = defaultdict(list)
    for row in output: buckets[float(row["nominal_rh_setpoint"])].append(row)
    summary = []
    for nominal, values in sorted(buckets.items()):
        h2 = np.asarray([row["h2_raw"] for row in values]); rh = np.asarray([row["rh_h2o_only_equivalent_raw"] for row in values])
        summary.append({
            "nominal_rh_setpoint": nominal, "n": len(values),
            "median_h2_raw": float(np.median(h2)), "h2_in_display_range_fraction": float(np.mean((h2 >= 3) & (h2 <= 4.5))),
            "median_h2o_only_equivalent_rh": float(np.median(rh)),
            "rh_in_display_range_fraction": float(np.mean((rh >= 65) & (rh <= 95))),
            "interpretation": "nominal RH is grouping metadata, never RH regression truth",
        })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figure(output, summary, out_dir)
    print(json.dumps(summary, indent=2))


def make_figure(rows, summary, out_dir):
    """Show whether single-condition calibrations preserve simultaneous levels."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
    nominal = np.asarray([float(row["nominal_rh_setpoint"]) for row in rows])
    rh = np.asarray([float(row["rh_h2o_only_equivalent_raw"]) for row in rows])
    h2 = np.asarray([float(row["h2_raw"]) for row in rows])
    rng = np.random.default_rng(7)
    jitter = rng.uniform(-0.8, 0.8, len(rows))

    axes[0].scatter(nominal + jitter, rh, s=17, alpha=.42, color="#0072B2", edgecolors="none")
    x = np.asarray([item["nominal_rh_setpoint"] for item in summary])
    y = np.asarray([item["median_h2o_only_equivalent_rh"] for item in summary])
    axes[0].plot(x, y, "o-", color="#003B5C", lw=2, label="Level median")
    axes[0].plot([20, 90], [20, 90], "--", color=".25", lw=1.5, label="Ideal transfer")
    axes[0].set(xlabel="Nominal simultaneous RH setpoint (%)",
                ylabel="H₂O-only-equivalent RH estimate (%)", ylim=(55, 100))
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].text(.03, .96, "A", transform=axes[0].transAxes, va="top", weight="bold", fontsize=15)

    axes[1].scatter(nominal + jitter, h2, s=17, alpha=.42, color="#D55E00", edgecolors="none")
    h2_median = np.asarray([item["median_h2_raw"] for item in summary])
    axes[1].plot(x, h2_median, "o-", color="#8C2D04", lw=2, label="Level median")
    axes[1].axhspan(3, 4, color="#D55E00", alpha=.09, label="App display range")
    axes[1].set(xlabel="Nominal simultaneous RH setpoint (%)",
                ylabel="H₂-only H₂ estimate (%)", ylim=(2.8, 4.5))
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].text(.03, .96, "B", transform=axes[1].transAxes, va="top", weight="bold", fontsize=15)
    fig.suptitle("Transfer of single-condition calibration to simultaneous reactions", weight="bold")
    for suffix, kwargs in (("png", {"dpi": 400}), ("pdf", {}), ("svg", {})):
        fig.savefig(out_dir / f"simultaneous_transfer_diagnostic.{suffix}", **kwargs)
    plt.close(fig)


if __name__ == "__main__": main()
