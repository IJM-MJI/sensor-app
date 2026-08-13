"""Stable-segment quantitation for independent H2-only and H2O-only videos."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_models import CACHE_VERSION, feature_value, read_csv


FEATURE_SETS = {
    "H2": {
        "flame_ab": ["flame_a", "flame_b"],
        "flame_lab": ["flame_L", "flame_a", "flame_b"],
        "flame_local_reference": ["flame_L", "flame_a", "flame_b", "top_L", "top_a", "top_b"],
    },
    "RH": {
        "drop_ab": ["drop_a", "drop_b"],
        "drop_lab": ["drop_L", "drop_a", "drop_b"],
        "drop_local_reference": ["drop_L", "drop_a", "drop_b", "top_L", "top_a", "top_b"],
    },
}


def candidates() -> dict[str, object]:
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
        "PLS-1": make_pipeline(StandardScaler(), PLSRegression(n_components=1, scale=False)),
        "PLS-2": make_pipeline(StandardScaler(), PLSRegression(n_components=2, scale=False)),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500, max_depth=6, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, max_depth=6, min_samples_leaf=2, random_state=42, n_jobs=-1),
    }


def contiguous_segments(rows: list[dict[str, object]], label: str) -> list[list[dict[str, object]]]:
    by_video: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get(label) is not None:
            by_video[str(row["video"])].append(row)
    output = []
    for video_rows in by_video.values():
        video_rows.sort(key=lambda row: float(row["time"]))
        segment = []
        previous = None
        for row in video_rows:
            value = float(row[label])
            if segment and value != previous:
                output.append(segment)
                segment = []
            segment.append(row)
            previous = value
        if segment:
            output.append(segment)
    return output


def segment_rows(rows: list[dict[str, object]], task: str, min_duration: float = 5.0):
    label = "h2_value" if task == "H2" else "rh_value"
    kind = "h2_only" if task == "H2" else "rh_only"
    features = sorted({name for values in FEATURE_SETS[task].values() for name in values})
    output, excluded = [], []
    for segment in contiguous_segments([row for row in rows if row["kind"] == kind], label):
        times = np.asarray([float(row["time"]) for row in segment])
        dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
        duration = float(times[-1] - times[0] + dt)
        summary = {
            "task": task, "video": segment[0]["video"], "group": segment[0]["group"],
            "reference": float(segment[0][label]), "start": float(times[0]),
            "end": float(times[-1] + dt), "duration": duration, "n_raw_frames": len(segment),
        }
        if duration < min_duration or len(segment) < 3:
            summary["reason"] = "shorter than 5 s accumulation window"
            excluded.append(summary)
            continue
        # Use the latter half, capped at the final five seconds. This omits the
        # setpoint edge while matching the app's causal five-second observation.
        tail_start = max(times[0] + duration * .5, times[-1] - 5.0)
        tail = [row for row in segment if float(row["time"]) >= tail_start]
        for name in features:
            values = [feature_value(row, name) for row in tail]
            summary[name] = float(np.median(values))
            summary[f"{name}_sd"] = float(np.std(values))
        summary["n_tail_frames"] = len(tail)
        output.append(summary)
    return output, excluded


def oof(segments, features, estimator):
    x = np.asarray([[float(row[name]) for name in features] for row in segments])
    y = np.asarray([float(row["reference"]) for row in segments])
    groups = np.asarray([str(row["group"]) for row in segments])
    prediction = np.full(len(y), np.nan)
    for group in sorted(set(groups)):
        test = groups == group
        train = ~test
        fitted = clone(estimator).fit(x[train], y[train])
        prediction[test] = np.asarray(fitted.predict(x[test])).reshape(-1)
    return y, prediction, groups


def metrics(y, prediction, groups):
    per_video = []
    for group in sorted(set(groups)):
        use = groups == group
        per_video.append({
            "group": str(group), "n_segments": int(use.sum()),
            "mae": float(mean_absolute_error(y[use], prediction[use])),
            "bias": float(np.mean(prediction[use] - y[use])),
        })
    return {
        "video_macro_mae": float(np.mean([row["mae"] for row in per_video])),
        "segment_mae": float(mean_absolute_error(y, prediction)),
        "segment_r2": float(r2_score(y, prediction)),
        "n_segments": len(y), "n_videos": len(per_video), "per_video": per_video,
    }


def level_metrics(predictions, task):
    tolerance = 0.75 if task == "H2" else 10.0
    output = []
    task_predictions = [row for row in predictions if row["task"] == task]
    for level in sorted({float(row["reference"]) for row in task_predictions}):
        use = [row for row in task_predictions if float(row["reference"]) == level]
        errors = np.asarray([abs(float(row["prediction"]) - level) for row in use])
        output.append({
            "reference": level, "n_videos": len({row["group"] for row in use}),
            "n_segments": len(use), "mae": float(errors.mean()),
            "within_tolerance": float(np.mean(errors <= tolerance)),
            "display_candidate": bool(len({row["group"] for row in use}) >= 3 and errors.mean() <= tolerance),
        })
    return output


def plot(path: Path, predictions, levels, selected):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), constrained_layout=True)
    for ax, task, limits, color in (
        (axes[0], "H2", (0, 4), "#D55E00"), (axes[1], "RH", (20, 90), "#0072B2")):
        use = [row for row in predictions if row["task"] == task]
        for group in sorted({row["group"] for row in use}):
            points = [row for row in use if row["group"] == group]
            ax.scatter([row["reference"] for row in points], [row["prediction"] for row in points],
                       s=28, alpha=.72, label=group)
        ax.plot(limits, limits, "--", color="#333", lw=1)
        pad = (limits[1] - limits[0]) * .07
        ax.set(xlim=(limits[0]-pad, limits[1]+pad), ylim=(limits[0]-pad, limits[1]+pad),
               xlabel="Reference", ylabel="Held-out-video prediction",
               title=f"{task} stable segments\n{selected[task]}")
        candidate = [row["reference"] for row in levels[task] if row["display_candidate"]]
        ax.text(.03, .97, "display candidates: " + (", ".join(map(lambda x: f"{x:g}", candidate)) or "none"),
                transform=ax.transAxes, va="top", fontsize=8)
    axes[1].legend(frameon=False, fontsize=6, loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(f"training/cache/{CACHE_VERSION}/features.csv"))
    parser.add_argument("--output", type=Path, default=Path("training/output/single_condition_quantitation"))
    args = parser.parse_args()
    rows = read_csv(args.cache)
    segments, excluded = {}, {}
    comparison, outputs = {"H2": {}, "RH": {}}, {}
    for task in ("H2", "RH"):
        segments[task], excluded[task] = segment_rows(rows, task)
        for feature_name, features in FEATURE_SETS[task].items():
            for model_name, estimator in candidates().items():
                if model_name == "PLS-2" and len(features) < 2:
                    continue
                y, prediction, groups = oof(segments[task], features, estimator)
                key = f"{feature_name}+{model_name}"
                comparison[task][key] = metrics(y, prediction, groups)
                outputs[(task, key)] = (y, prediction, groups)
    selected = {task: min(comparison[task], key=lambda key: comparison[task][key]["video_macro_mae"])
                for task in ("H2", "RH")}
    predictions, levels = [], {}
    for task in ("H2", "RH"):
        y, prediction, groups = outputs[(task, selected[task])]
        for row, truth, estimate, group in zip(segments[task], y, prediction, groups):
            predictions.append({
                "task": task, "video": row["video"], "group": str(group),
                "reference": float(truth), "prediction": float(estimate),
                "absolute_error": float(abs(estimate-truth)), "duration": row["duration"],
                "start": row["start"], "end": row["end"],
            })
        levels[task] = level_metrics(predictions, task)
    report = {
        "policy": "single-condition only; >=5 s steps; median of final <=5 s; leave-one-video-group-out",
        "selected": selected, "models": comparison, "level_metrics": levels,
        "included_segments": {task: len(segments[task]) for task in segments},
        "excluded_transient_segments": excluded,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    plot(args.output / "stable_segment_validation", predictions, levels, selected)
    print(json.dumps({
        "selected": selected,
        "metrics": {task: comparison[task][selected[task]] for task in selected},
        "level_metrics": levels,
        "excluded_transient_count": {task: len(excluded[task]) for task in excluded},
    }, indent=2))


if __name__ == "__main__":
    main()
