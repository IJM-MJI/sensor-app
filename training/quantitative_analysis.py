"""Evaluate concentration models and create a publication-ready validation figure.

All predictions are leave-one-video-out (LOVO): every point is predicted by a
model that did not see any frame from that recording.  Metrics are macro-
averaged over videos so long recordings do not dominate the result.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_models import (
    CACHE_VERSION, H2_FEATURES, RH_FEATURES, feature_value, merge_training_rows,
    read_csv, read_legacy_continuous,
)


def candidates() -> dict[str, object]:
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500, max_depth=10, min_samples_leaf=5,
            max_features=1.0, random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=250, learning_rate=.04, max_leaf_nodes=15,
            min_samples_leaf=15, l2_regularization=2.0, random_state=42,
        ),
    }


def oof(rows: list[dict[str, object]], label: str, features: list[str], estimator: object):
    x = np.asarray([[feature_value(row, name) for name in features] for row in rows])
    y = np.asarray([float(row[label]) for row in rows])
    groups = np.asarray([str(row["group"]) for row in rows])
    pred = np.full(len(rows), np.nan)
    for group in sorted(set(groups)):
        test = groups == group
        train = ~test
        fitted = clone(estimator).fit(x[train], y[train])
        pred[test] = fitted.predict(x[test])
    return y, pred, groups


def video_metrics(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, span: float) -> dict[str, object]:
    per_video = []
    for group in sorted(set(groups)):
        use = groups == group
        per_video.append({
            "video": group,
            "mae": float(mean_absolute_error(y[use], pred[use])),
            "rmse": float(mean_squared_error(y[use], pred[use]) ** .5),
            "bias": float(np.mean(pred[use] - y[use])),
            "n": int(use.sum()),
        })
    maes = np.asarray([row["mae"] for row in per_video])
    rng = np.random.default_rng(42)
    boot = np.asarray([np.mean(rng.choice(maes, len(maes), replace=True)) for _ in range(5000)])
    return {
        "video_macro_mae": float(maes.mean()),
        "video_macro_mae_95ci": np.quantile(boot, [.025, .975]).tolist(),
        "normalized_mae_percent_range": float(100 * maes.mean() / span),
        "frame_mae": float(mean_absolute_error(y, pred)),
        "frame_rmse": float(mean_squared_error(y, pred) ** .5),
        "frame_r2": float(r2_score(y, pred)),
        "n_frames": int(len(y)),
        "n_videos": int(len(per_video)),
        "per_video": per_video,
    }


def aggregate_points(rows, y, pred, groups, task: str):
    buckets: dict[tuple[str, float, str], list[int]] = {}
    for index, (row, truth, group) in enumerate(zip(rows, y, groups)):
        if task == "H2":
            level = round(float(truth) * 2) / 2
        else:
            level = round(float(truth) / 10) * 10
        buckets.setdefault((str(group), level, str(row["kind"])), []).append(index)
    output = []
    for (group, level, kind), indices in buckets.items():
        values = pred[indices]
        output.append({
            "task": task, "video": group, "kind": kind,
            "reference": float(level), "prediction": float(np.mean(values)),
            "prediction_sd": float(np.std(values)), "n_frames": len(indices),
        })
    return output


def concentration_mae(points: list[dict[str, object]], task: str):
    levels = sorted({float(p["reference"]) for p in points if p["task"] == task})
    return levels, [
        float(np.mean([abs(float(p["prediction"]) - level) for p in points
                       if p["task"] == task and float(p["reference"]) == level]))
        for level in levels
    ]


def make_figure(path: Path, predictions, comparison, selected):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
        "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    colors = {"h2_only": "#D55E00", "simultaneous": "#7B3294", "rh_only": "#0072B2"}
    labels = {"h2_only": "H₂ only", "simultaneous": "Simultaneous", "rh_only": "H₂O only"}

    for ax, task, limit, unit in ((axes[0, 0], "H2", (0, 4), "%"), (axes[0, 1], "RH", (20, 90), "%RH")):
        task_points = [p for p in predictions if p["task"] == task]
        for kind in sorted({str(p["kind"]) for p in task_points}):
            pts = [p for p in task_points if p["kind"] == kind]
            ax.scatter([p["reference"] for p in pts], [p["prediction"] for p in pts],
                       s=18, alpha=.55, color=colors[kind], edgecolor="none", label=labels[kind])
        ax.plot(limit, limit, "--", color="#333333", lw=1)
        pad = (limit[1] - limit[0]) * .06
        ax.set(xlim=(limit[0] - pad, limit[1] + pad), ylim=(limit[0] - pad, limit[1] + pad),
               xlabel=f"Reference concentration ({unit})", ylabel=f"LOVO prediction ({unit})",
               title=f"{'H₂' if task == 'H2' else task} concentration")
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1, 0]
    for task, color, marker in (("H2", "#D55E00", "o"), ("RH", "#0072B2", "s")):
        levels, errors = concentration_mae(predictions, task)
        span = 4 if task == "H2" else 70
        origin = 0 if task == "H2" else 20
        normalized_levels = (np.asarray(levels) - origin) / span * 100
        ax.plot(normalized_levels, np.asarray(errors) / span * 100, marker=marker, ms=4,
                lw=1.4, color=color, label="H₂" if task == "H2" else task)
    ax.set(xlabel="Reference level (% of measurement range)", ylabel="MAE (% of measurement range)",
           title="Error by concentration level")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    names = list(candidates())
    x = np.arange(len(names))
    width = .35
    for offset, task, color in ((-.5, "H2", "#D55E00"), (.5, "RH", "#0072B2")):
        vals = [comparison[task][name]["normalized_mae_percent_range"] for name in names]
        ci = [comparison[task][name]["video_macro_mae_95ci"] for name in names]
        span = 4 if task == "H2" else 70
        low = [value - interval[0] / span * 100 for value, interval in zip(vals, ci)]
        high = [interval[1] / span * 100 - value for value, interval in zip(vals, ci)]
        ax.bar(x + offset * width, vals, width, yerr=np.asarray([low, high]), capsize=2,
               color=color, alpha=.85, label="H₂" if task == "H2" else task, error_kw={"lw": .8})
    ax.set_xticks(x, names, rotation=18, ha="right")
    ax.set(ylabel="Video-macro MAE (% of range)", title="Model comparison (LOVO)")
    ax.legend(frameon=False)

    for label, ax in zip("ABCD", axes.flat):
        ax.text(.97, .96, label, transform=ax.transAxes, fontweight="bold", fontsize=12,
                ha="right", va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1})
    fig.suptitle(
        f"Quantitative optical sensor validation\nSelected: H₂ {selected['H2']}; RH {selected['RH']}",
        fontsize=12, fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-cache", type=Path, default=Path(f"training/cache/{CACHE_VERSION}/features.csv"))
    parser.add_argument("--legacy-cache", type=Path, default=Path(f"training/cache/{CACHE_VERSION}/legacy_continuous_lag_corrected.csv"))
    parser.add_argument("--output", type=Path, default=Path("training/output/quantitative"))
    args = parser.parse_args()

    new_rows = read_csv(args.new_cache)
    rows = merge_training_rows(new_rows, read_legacy_continuous(args.legacy_cache))
    tasks = {
        "H2": ([r for r in rows if r.get("h2_value") is not None and r.get("kind") != "rh_only"],
               "h2_value", H2_FEATURES, 4.0),
        "RH": ([r for r in rows if r.get("rh_value") is not None and r.get("kind") == "rh_only"],
               "rh_value", RH_FEATURES, 70.0),
    }
    comparison: dict[str, dict[str, object]] = {"H2": {}, "RH": {}}
    all_outputs = {}
    for task, (task_rows, label, features, span) in tasks.items():
        for name, estimator in candidates().items():
            y, pred, groups = oof(task_rows, label, features, estimator)
            comparison[task][name] = video_metrics(y, pred, groups, span)
            all_outputs[(task, name)] = (task_rows, y, pred, groups)

    selected = {
        task: min(comparison[task], key=lambda name: comparison[task][name]["video_macro_mae"])
        for task in tasks
    }
    points = []
    raw_predictions = []
    for task, model_name in selected.items():
        task_rows, y, pred, groups = all_outputs[(task, model_name)]
        points.extend(aggregate_points(task_rows, y, pred, groups, task))
        for row, truth, estimate in zip(task_rows, y, pred):
            raw_predictions.append({
                "task": task, "model": model_name, "video": row["video"], "group": row["group"],
                "kind": row["kind"], "time": row["time"], "reference": float(truth),
                "prediction": float(estimate), "residual": float(estimate - truth),
            })

    args.output.mkdir(parents=True, exist_ok=True)
    segment_metrics = {}
    for task, tolerance in (("H2", 1.0), ("RH", 10.0)):
        task_points = [point for point in points if point["task"] == task]
        errors = np.asarray([abs(float(point["prediction"]) - float(point["reference"])) for point in task_points])
        segment_metrics[task] = {
            "n_group_level_segments": len(task_points),
            "mae": float(np.mean(errors)), "median_absolute_error": float(np.median(errors)),
            f"fraction_within_{tolerance:g}_percentage_points": float(np.mean(errors <= tolerance)),
        }
    report = {"evaluation": "leave-one-video-out; metrics macro-averaged by video",
              "selected": selected, "models": comparison,
              "stable_group_level_aggregation": segment_metrics}
    (args.output / "quantitative_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for filename, data in (("quantitative_predictions.csv", raw_predictions),
                           ("quantitative_figure_points.csv", points)):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    make_figure(args.output / "quantitative_validation", points, comparison, selected)
    print(json.dumps({
        "selected": selected,
        "H2": comparison["H2"][selected["H2"]],
        "RH": comparison["RH"][selected["RH"]],
    }, indent=2))


if __name__ == "__main__":
    main()
