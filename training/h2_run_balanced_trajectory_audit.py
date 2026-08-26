"""Run-balanced H2 2/3 student trained from monotonic trajectory labels.

Each (source run, optical class) block has equal total training weight. Highly
correlated frames are also capped by evenly spaced temporal subsampling. This
prevents a long weak-response video from dominating the 2% boundary.
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
from h2_trajectory_teacher_audit import trajectory_teacher


def even_cap(mask, groups, labels, times, cap):
    """Keep at most cap temporally even rows from every run/class block."""
    keep = np.zeros(len(mask), dtype=bool)
    for run in sorted(set(groups[mask])):
        for label in (2, 3):
            index = np.flatnonzero(mask & (groups == run) & (labels == label))
            index = index[np.argsort(times[index])]
            if len(index) > cap:
                positions = np.linspace(0, len(index) - 1, cap).round().astype(int)
                index = index[np.unique(positions)]
            keep[index] = True
    return keep


def equal_run_class_weights(groups, labels, selected):
    """Give every class equal weight and every contributing run equal share."""
    weights = np.zeros(len(selected), dtype=float)
    for label in (2, 3):
        runs = sorted(set(groups[selected & (labels == label)]))
        for run in runs:
            block = selected & (labels == label) & (groups == run)
            weights[block] = 1.0 / (len(runs) * max(int(block.sum()), 1))
    # Normalise for conventional solver scale; relative weights are unchanged.
    weights[selected] *= selected.sum() / weights[selected].sum()
    return weights


def weighted_probabilities(base_x, base_groups, pseudo_x, pseudo_y,
                           pseudo_groups, selected, weights):
    p2 = np.zeros(len(base_x), dtype=float)
    p3 = np.zeros(len(base_x), dtype=float)
    for held_out in sorted(set(base_groups)):
        train = selected & (pseudo_groups != held_out)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(pseudo_x[train], sample_weight=weights[train])
        model = LogisticRegression(C=.2, max_iter=4000, random_state=42)
        model.fit(scaled, pseudo_y[train], sample_weight=weights[train])
        test = base_groups == held_out
        probability = model.predict_proba(scaler.transform(base_x[test]))
        classes = list(model.classes_)
        p2[test] = probability[:, classes.index(2)]
        p3[test] = probability[:, classes.index(3)]
    return p2, p3


def nested_gate(y, groups, baseline, p2, p3):
    final = baseline.copy()
    choices = {}
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
            _, threshold_23, threshold_32 = max(feasible)
            final[use] = gate(baseline[use], p2[use], p3[use],
                              threshold_23, threshold_32)
            choices[held_out] = {"threshold_2_to_3": threshold_23,
                                 "threshold_3_to_2": threshold_32}
        else:
            choices[held_out] = {"action": "keep baseline"}
    return final, choices


def block_counts(groups, labels, selected):
    return {run: {str(label): int(np.sum(selected & (groups == run)
                                            & (labels == label)))
                  for label in (2, 3)}
            for run in sorted(set(groups))}


def plot(path, baseline, variants):
    names = ["baseline"] + list(variants)
    exact = [baseline["exact"]] + [variants[name]["metrics"]["exact"] for name in variants]
    r2 = [baseline["recall"]["2"]] + [variants[name]["metrics"]["recall"]["2"]
                                          for name in variants]
    r3 = [baseline["recall"]["3"]] + [variants[name]["metrics"]["recall"]["3"]
                                          for name in variants]
    x = np.arange(len(names)); width = .25
    fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.bar(x - width, exact, width, label="Exact")
    axis.bar(x, r2, width, label="Recall 2%")
    axis.bar(x + width, r3, width, label="Recall 3%")
    axis.axhline(.85, color="#555", ls="--", lw=1, label="0.85 target")
    axis.set(xticks=x, xticklabels=names, ylim=(0, 1), ylabel="Score",
             title="Run-balanced trajectory teacher: complete-video held out")
    axis.legend(ncol=4)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--four-band-cache", type=Path, required=True)
    parser.add_argument("--pseudo-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

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
            p2, p3 = weighted_probabilities(
                bx, bg, px, teacher_y, pg, balanced, weights)
            prediction, choices = nested_gate(
                y, groups, baseline, p2[evaluated], p3[evaluated])
            metrics = report(y, prediction, groups)
            passed = bool(safe(metrics, baseline_metrics)
                          and safe_per_video(y, prediction, baseline, groups))
            name = f"c{certainty:.2f}_cap{cap}"
            variants[name] = {
                "certainty": certainty, "cap_per_run_class": cap,
                "rows": int(balanced.sum()),
                "counts": block_counts(pg, teacher_y, balanced),
                "weighting": "equal class; equal contributing run within class",
                "metrics": metrics, "changed_frames": int(np.sum(prediction != baseline)),
                "constraints_passed": passed, "choices": choices,
            }

    selected_name = max(variants, key=lambda name: (
        variants[name]["constraints_passed"],
        variants[name]["metrics"]["recall"]["2"],
        variants[name]["metrics"]["recall"]["3"],
        variants[name]["metrics"]["video_macro_exact"],
        -variants[name]["metrics"]["mae_percent_points"],
    ))
    payload = {
        "protocol": "run/class-equal weighted trajectory teacher; nested video held out",
        "runtime": "single measurement image; balancing is training-only",
        "baseline": baseline_metrics,
        "selected": selected_name,
        "selected_result": variants[selected_name],
        "variants": variants,
        "deployment_ready": bool(variants[selected_name]["constraints_passed"]),
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(args.output / "run_balanced_comparison.png", baseline_metrics, variants)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
