"""Audit run-to-run H2/RH optical trajectories and render review frames.

The audit is deliberately model-free.  Each run/stage median is compared with a
leave-one-run-out consensus in robustly scaled LAB-delta space.  A mismatch means
that the optical state is closer to another supplied concentration than to its
timeline label; it is a review target, not an automatic relabel.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ordinal_concentration_analysis import (
    H2_RAMP_ENDPOINTS, RH_RAMP_ENDPOINTS, TASKS,
    assign_h2_ramp_targets, assign_rh_ramp_targets,
)
from train_models import CACHE_VERSION, read_csv, resize_for_app


FEATURES = {
    "H2": ("flame_L", "flame_a", "flame_b"),
    "RH": ("drop_L", "drop_a", "drop_b"),
}


def stage_value(row: dict[str, object], task: str) -> float | None:
    if "audit_stage" in row:
        return float(row["audit_stage"])
    key = "analysis_stage" if task == "H2" else "rh_analysis_stage"
    value = row.get(key)
    return None if value is None else float(value)


def usable_rows(rows: list[dict[str, object]], task: str) -> list[dict[str, object]]:
    """Return endpoint frames, never rounded ramp-interior pseudo-stages."""
    kind = TASKS[task]["kind"]
    timelines = H2_RAMP_ENDPOINTS if task == "H2" else RH_RAMP_ENDPOINTS
    selected = []
    for row in rows:
        if row["kind"] != kind:
            continue
        endpoints = timelines.get(str(row["video"]))
        if not endpoints:
            continue
        time = float(row["time"])
        nearest_time, concentration = min(endpoints, key=lambda point: abs(time - point[0]))
        # All quantitative clips were sampled at >=2 Hz (response clips at 4 Hz).
        # A 0.55 s window retains at least one frame on each side without turning
        # the preceding ramp into a nominal plateau.
        if abs(time - float(nearest_time)) > .55:
            continue
        stage = float(concentration)
        if task == "RH":
            levels = np.asarray(TASKS[task]["levels"], dtype=float)
            stage = float(levels[np.argmin(np.abs(levels - stage))])
        copied = dict(row)
        copied["audit_stage"] = stage
        copied["endpoint_time"] = float(nearest_time)
        copied["endpoint_offset"] = time - float(nearest_time)
        selected.append(copied)
    return selected


def robust_scale(rows: list[dict[str, object]], features: tuple[str, ...]) -> np.ndarray:
    values = np.asarray([[float(row[name]) for name in features] for row in rows])
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    return np.maximum(q75 - q25, 1.0)


def stage_medians(
    rows: list[dict[str, object]], task: str, features: tuple[str, ...], scale: np.ndarray,
) -> dict[tuple[str, float], dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["group"]), float(stage_value(row, task)))].append(row)
    output = {}
    for key, stage_rows in grouped.items():
        values = np.asarray([[float(row[name]) for name in features] for row in stage_rows])
        median = np.median(values, axis=0)
        distances = np.linalg.norm((values - median) / scale, axis=1)
        representative = stage_rows[int(np.argmin(distances))]
        output[key] = {
            "median": median, "n": len(stage_rows), "representative": representative,
            "videos": sorted({str(row["video"]) for row in stage_rows}),
        }
    return output


def consensus_for(
    medians: dict[tuple[str, float], dict[str, object]], excluded_group: str, level: float,
) -> np.ndarray | None:
    candidates = [item["median"] for (group, stage), item in medians.items()
                  if group != excluded_group and stage == level]
    return None if not candidates else np.median(np.asarray(candidates), axis=0)


def audit_task(rows: list[dict[str, object]], task: str) -> tuple[list[dict[str, object]], dict]:
    task_rows = usable_rows(rows, task)
    features = FEATURES[task]
    scale = robust_scale(task_rows, features)
    medians = stage_medians(task_rows, task, features, scale)
    levels = [float(level) for level in TASKS[task]["levels"]]
    report_rows: list[dict[str, object]] = []
    for (group, nominal), item in sorted(medians.items()):
        available = []
        for level in levels:
            consensus = consensus_for(medians, group, level)
            if consensus is not None:
                distance = float(np.linalg.norm((item["median"] - consensus) / scale))
                available.append((distance, level, consensus))
        if not available:
            continue
        available.sort()
        distance, nearest, _ = available[0]
        own = next((candidate[0] for candidate in available if candidate[1] == nominal), None)
        second = available[1][0] if len(available) > 1 else distance
        row = {
            "task": task, "group": group, "videos": ";".join(item["videos"]),
            "nominal_stage": nominal, "nearest_consensus_stage": nearest,
            "stage_error": nearest - nominal, "nearest_distance": distance,
            "own_stage_distance": own, "distance_margin": second - distance,
            "n_frames": item["n"],
            "representative_video": str(item["representative"]["video"]),
            "representative_time": float(item["representative"]["time"]),
            "review": "mismatch" if nearest != nominal else "match",
        }
        for index, name in enumerate(features):
            row[f"median_{name}"] = float(item["median"][index])
        report_rows.append(row)
    mismatch = [row for row in report_rows if row["review"] == "mismatch"]
    progress = []
    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in report_rows:
        by_group[str(row["group"])].append(row)
    for group, group_rows in sorted(by_group.items()):
        group_rows.sort(key=lambda row: float(row["nominal_stage"]))
        baseline = np.asarray([float(group_rows[0][f"median_{name}"]) for name in features])
        magnitudes = np.asarray([
            np.linalg.norm((np.asarray([float(row[f"median_{name}"]) for name in features]) - baseline) / scale)
            for row in group_rows
        ])
        ranks = np.argsort(np.argsort(magnitudes)).astype(float)
        expected = np.arange(len(group_rows), dtype=float)
        correlation = float(np.corrcoef(expected, ranks)[0, 1]) if len(group_rows) > 1 else 0.0
        progress.append({
            "group": group, "spearman_progress": correlation,
            "quality": "ordered" if correlation >= .8 else "mixed" if correlation >= .5 else "weak",
            "endpoint_response_magnitude": [
                {"stage": float(row["nominal_stage"]), "magnitude": float(magnitude)}
                for row, magnitude in zip(group_rows, magnitudes)
            ],
        })
    summary = {
        "task": task, "features": list(features), "robust_scale": scale.tolist(),
        "n_run_stages": len(report_rows), "n_matches": len(report_rows) - len(mismatch),
        "n_mismatches": len(mismatch),
        "matching_rate": (len(report_rows) - len(mismatch)) / max(len(report_rows), 1),
        "within_run_progress": progress, "mismatches": mismatch,
    }
    return report_rows, summary


def source_path(video_root: Path, logical_video: str) -> Path:
    cropped = video_root / f"{Path(logical_video).stem}_cropped.mp4"
    return cropped if cropped.exists() else video_root / logical_video


def frame_at(path: Path, seconds: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot decode {path.name} at {seconds:.2f}s")
    return resize_for_app(frame)


def render_review(
    audit_rows: list[dict[str, object]], video_root: Path, output: Path, task: str,
) -> None:
    mismatches = [row for row in audit_rows if row["review"] == "mismatch"]
    mismatches.sort(key=lambda row: (
        -abs(float(row["stage_error"])), -float(row["distance_margin"]), str(row["group"])
    ))
    tile_w, image_h, caption_h, columns = 320, 250, 82, 3
    rows_count = max(1, int(np.ceil(len(mismatches) / columns)))
    sheet = np.full((rows_count * (image_h + caption_h), columns * tile_w, 3), 242, np.uint8)
    for index, row in enumerate(mismatches):
        y0 = (index // columns) * (image_h + caption_h)
        x0 = (index % columns) * tile_w
        logical = str(row["representative_video"])
        path = source_path(video_root, logical)
        image = frame_at(path, float(row["representative_time"]))
        scale = min(tile_w / image.shape[1], image_h / image.shape[0])
        resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
        iy, ix = y0 + (image_h - resized.shape[0]) // 2, x0 + (tile_w - resized.shape[1]) // 2
        sheet[iy:iy + resized.shape[0], ix:ix + resized.shape[1]] = resized
        lines = [
            f"{row['group']}  {Path(logical).name}",
            f"t={float(row['representative_time']):.1f}s  nominal={float(row['nominal_stage']):g}",
            f"optical-nearest={float(row['nearest_consensus_stage']):g}  d={float(row['nearest_distance']):.2f}",
        ]
        for line_index, text in enumerate(lines):
            cv2.putText(sheet, text, (x0 + 6, y0 + image_h + 20 + line_index * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .43, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.imwrite(str(output / f"{task.lower()}_trajectory_review.jpg"), sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 95])


def plot_trajectories(
    rows: list[dict[str, object]], summaries: dict[str, dict], output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for axis, task in zip(axes, ("H2", "RH")):
        features = FEATURES[task]
        task_rows = usable_rows(rows, task)
        scale = robust_scale(task_rows, features)
        medians = stage_medians(task_rows, task, features, scale)
        groups = sorted({key[0] for key in medians})
        for group in groups:
            points = sorted((stage, item["median"]) for (run, stage), item in medians.items() if run == group)
            if not points:
                continue
            stages = [point[0] for point in points]
            values = np.asarray([point[1] for point in points])
            axis.plot(values[:, 1], values[:, 2], marker="o", linewidth=1.4, label=group)
            for stage, x, y in zip(stages, values[:, 1], values[:, 2]):
                axis.annotate(f"{stage:g}", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        axis.axhline(0, color="0.85", linewidth=.8); axis.axvline(0, color="0.85", linewidth=.8)
        axis.set_xlabel(f"calibrated {features[1]}"); axis.set_ylabel(f"calibrated {features[2]}")
        axis.set_title(f"{task}: run trajectories (match {summaries[task]['matching_rate']:.1%})")
        axis.legend(fontsize=6, loc="best")
    fig.tight_layout()
    for suffix, kwargs in (("png", {"dpi": 400}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"single_condition_trajectory_audit.{suffix}", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path,
                        default=Path(f"training/cache/{CACHE_VERSION}/features_cropped_centered.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/output/single_condition_trajectory_audit"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.cache)
    assign_h2_ramp_targets(rows)
    assign_rh_ramp_targets(rows)
    all_rows: list[dict[str, object]] = []
    summaries = {}
    for task in ("H2", "RH"):
        task_audit, summary = audit_task(rows, task)
        all_rows.extend(task_audit)
        summaries[task] = summary
        render_review(task_audit, args.video_root, args.output, task)
    fields = list(dict.fromkeys(key for row in all_rows for key in row))
    with (args.output / "trajectory_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(all_rows)
    (args.output / "trajectory_audit.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_trajectories(rows, summaries, args.output)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
