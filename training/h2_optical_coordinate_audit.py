"""Run-balanced H2 2/3 audit in a calibration-relative optical coordinate.

The coordinate measures progress along the shared optical-2 to optical-3
yellow-to-green direction, orthogonal residual, and response magnitude.  It is
computed from one calibration-relative measurement vector, so runtime remains
single-shot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from h2_optical_pseudolabel_analysis import optical_labels
from h2_optical_23_specialist_audit import (
    THRESHOLDS, crossfit_baseline, gate, rank, report, safe, safe_per_video,
)
from h2_run_balanced_trajectory_audit import (
    equal_run_class_weights, even_cap,
)
from h2_trajectory_teacher_audit import trajectory_teacher


def equal_run_center(x, labels, groups, selected, label):
    centers = []
    for run in sorted(set(groups[selected & (labels == label)])):
        use = selected & (labels == label) & (groups == run)
        centers.append(np.median(x[use], axis=0))
    return np.mean(centers, axis=0)


def coordinate_fit(x, labels, groups, selected):
    rows = x[selected]
    center = np.median(rows, axis=0)
    scale = np.maximum(np.median(np.abs(rows - center), axis=0), .15)
    z = x / scale
    c2 = equal_run_center(z, labels, groups, selected, 2)
    c3 = equal_run_center(z, labels, groups, selected, 3)
    direction = c3 - c2
    direction /= max(np.linalg.norm(direction), 1e-9)
    return scale, c2, c3, direction


def coordinates(x, scale, c2, c3, direction):
    z = x / scale
    delta = z - c2
    projection = delta @ direction
    residual = delta - projection[:, None] * direction
    orthogonal = np.sqrt(np.mean(residual * residual, axis=1))
    magnitude = np.sqrt(np.mean(z * z, axis=1))
    cosine = projection / np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
    d2 = np.sqrt(np.mean((z - c2) ** 2, axis=1))
    d3 = np.sqrt(np.mean((z - c3) ** 2, axis=1))
    relative_distance = (d2 - d3) / np.maximum(d2 + d3, 1e-9)
    # Preserve separate mean/median/chroma-distribution response summaries in
    # addition to the shared direction coordinate.
    return np.column_stack([
        projection, orthogonal, magnitude, cosine, d2, d3, relative_distance,
        z[:, 0:3].mean(axis=1), z[:, 3:6].mean(axis=1), z[:, 6:11].mean(axis=1),
    ])


def coordinate_probabilities(base_x, base_groups, pseudo_x, pseudo_y,
                             pseudo_groups, selected, weights):
    p2 = np.zeros(len(base_x), dtype=float)
    p3 = np.zeros(len(base_x), dtype=float)
    audit = {}
    for held_out in sorted(set(base_groups)):
        train = selected & (pseudo_groups != held_out)
        scale, c2, c3, direction = coordinate_fit(
            pseudo_x, pseudo_y, pseudo_groups, train)
        train_x = coordinates(pseudo_x[train], scale, c2, c3, direction)
        scaler = StandardScaler()
        train_z = scaler.fit_transform(train_x, sample_weight=weights[train])
        model = LogisticRegression(C=.2, max_iter=4000, random_state=42)
        model.fit(train_z, pseudo_y[train], sample_weight=weights[train])
        test = base_groups == held_out
        test_x = coordinates(base_x[test], scale, c2, c3, direction)
        probability = model.predict_proba(scaler.transform(test_x))
        classes = list(model.classes_)
        p2[test] = probability[:, classes.index(2)]
        p3[test] = probability[:, classes.index(3)]
        audit[held_out] = {
            "direction": direction.tolist(),
            "center_separation": float(np.linalg.norm(c3 - c2)),
        }
    return p2, p3, audit


def nested_gate(y, groups, baseline, p2, p3):
    final = baseline.copy(); choices = {}
    for held_out in sorted(set(groups)):
        meta = groups != held_out
        old = report(y[meta], baseline[meta], groups[meta])
        feasible = []
        for threshold_23 in THRESHOLDS:
            for threshold_32 in THRESHOLDS:
                prediction = gate(baseline[meta], p2[meta], p3[meta],
                                  threshold_23, threshold_32)
                metrics = report(y[meta], prediction, groups[meta])
                if (safe(metrics, old)
                        and safe_per_video(y[meta], prediction,
                                           baseline[meta], groups[meta])):
                    feasible.append((rank(metrics, old, threshold_23, threshold_32),
                                     threshold_23, threshold_32))
        use = groups == held_out
        if feasible:
            _, t23, t32 = max(feasible)
            final[use] = gate(baseline[use], p2[use], p3[use], t23, t32)
            choices[held_out] = {"threshold_2_to_3": t23, "threshold_3_to_2": t32}
        else:
            choices[held_out] = {"action": "keep baseline"}
    return final, choices


def plot(path, baseline, variants):
    names = ["baseline"] + list(variants)
    values = {
        "Exact": [baseline["exact"]] + [variants[n]["metrics"]["exact"] for n in variants],
        "Recall 2%": [baseline["recall"]["2"]] +
                     [variants[n]["metrics"]["recall"]["2"] for n in variants],
        "Recall 3%": [baseline["recall"]["3"]] +
                     [variants[n]["metrics"]["recall"]["3"] for n in variants],
    }
    x = np.arange(len(names)); width = .25
    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for offset, (label, scores) in zip((-width, 0, width), values.items()):
        axis.bar(x + offset, scores, width, label=label)
    axis.axhline(.85, color="#555", ls="--", lw=1, label="0.85 target")
    axis.set(xticks=x, xticklabels=names, ylim=(0, 1), ylabel="Score",
             title="Calibration-relative yellow-to-green optical coordinate")
    axis.legend(ncol=4)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--four-band-cache", type=Path, required=True)
    parser.add_argument("--pseudo-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    base = np.load(args.four_band_cache, allow_pickle=False)
    bx, by, bg = base["x"], base["y"], base["groups"]
    pseudo = np.load(args.pseudo_cache, allow_pickle=False)
    px, pg, pt = pseudo["x"], pseudo["groups"], pseudo["times"]
    initial_y, initial_selected, *_ = optical_labels(px, pg, pt)
    baseline_all = crossfit_baseline(bx, by, bg)
    evaluated = bg != "test_2"
    y, groups, baseline = by[evaluated], bg[evaluated], baseline_all[evaluated]
    baseline_metrics = report(y, baseline, groups)

    variants = {}
    for certainty in (.75, .85):
        teacher_y, teacher_selected, *_ = trajectory_teacher(
            px, initial_y, pg, pt, initial_selected, certainty)
        for cap in (6, 12, 20):
            balanced = even_cap(teacher_selected, pg, teacher_y, pt, cap)
            weights = equal_run_class_weights(pg, teacher_y, balanced)
            p2, p3, direction_audit = coordinate_probabilities(
                bx, bg, px, teacher_y, pg, balanced, weights)
            prediction, choices = nested_gate(
                y, groups, baseline, p2[evaluated], p3[evaluated])
            metrics = report(y, prediction, groups)
            name = f"c{certainty:.2f}_cap{cap}"
            variants[name] = {
                "certainty": certainty, "cap": cap, "rows": int(balanced.sum()),
                "metrics": metrics, "changed_frames": int(np.sum(prediction != baseline)),
                "constraints_passed": bool(safe(metrics, baseline_metrics)
                                           and safe_per_video(y, prediction,
                                                              baseline, groups)),
                "choices": choices, "direction_audit": direction_audit,
            }
    selected = max(variants, key=lambda name: (
        variants[name]["constraints_passed"], variants[name]["metrics"]["recall"]["2"],
        variants[name]["metrics"]["recall"]["3"],
        variants[name]["metrics"]["video_macro_exact"],
        -variants[name]["metrics"]["mae_percent_points"]))
    payload = {
        "protocol": "run-balanced trajectory labels; yellow-to-green coordinate; LOVO",
        "runtime": "single-shot calibration-relative student",
        "baseline": baseline_metrics, "selected": selected,
        "selected_result": variants[selected], "variants": variants,
        "deployment_ready": bool(variants[selected]["constraints_passed"]),
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(args.output / "optical_coordinate_comparison.png", baseline_metrics, variants)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
