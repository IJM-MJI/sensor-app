"""Use monotonic video trajectories to clean H2 optical 2/3 training labels.

The trajectory is a training-time teacher only.  The student specialist still
uses one calibration-relative image feature vector, so browser inference does
not require an accumulating video or a user decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression

from h2_optical_pseudolabel_analysis import REACTION_BOUNDS, optical_labels
from h2_optical_23_specialist_audit import (
    THRESHOLDS, crossfit_baseline, estimator, gate, rank, report, safe,
    safe_per_video, specialist_probabilities,
)


def trajectory_teacher(x, labels, groups, times, selected, certainty):
    """Cross-run probability followed by monotonic smoothing within each run."""
    raw_p3 = np.zeros(len(x), dtype=float)
    for held_out in sorted(set(groups)):
        train = selected & (groups != held_out)
        model = estimator("logistic")
        model.fit(x[train], labels[train])
        test = groups == held_out
        classes = list(model.classes_)
        raw_p3[test] = model.predict_proba(x[test])[:, classes.index(3)]

    monotonic_p3 = raw_p3.copy()
    reaction = np.zeros(len(x), dtype=bool)
    for run, (start, end) in REACTION_BOUNDS.items():
        use = (groups == run) & (times >= start) & (times <= end)
        reaction |= use
        order = np.flatnonzero(use)[np.argsort(times[use])]
        if len(order) >= 2:
            monotonic_p3[order] = IsotonicRegression(
                y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit_transform(
                    times[order], raw_p3[order])

    # Retain only endpoints of the smoothed 2/3 probability, not uncertain
    # transition frames.  Original test_2 anchor rows remain fixed metrology.
    confident = (monotonic_p3 <= 1 - certainty) | (monotonic_p3 >= certainty)
    teacher_selected = selected & reaction & confident
    teacher_labels = np.where(monotonic_p3 >= .5, 3, 2)
    anchor2 = (groups == "test_2") & (times >= 20) & (times <= 22)
    anchor3 = (groups == "test_2") & (times >= 29) & (times <= 31)
    teacher_selected |= anchor2 | anchor3
    teacher_labels[anchor2], teacher_labels[anchor3] = 2, 3
    return teacher_labels, teacher_selected, raw_p3, monotonic_p3


def probability_families(bx, bg, px, py, pg, selected, evaluated):
    output = {}
    for kind in ("lda", "logistic"):
        p2, p3 = specialist_probabilities(bx, bg, px, py, pg, selected, kind)
        output[kind] = (p2[evaluated], p3[evaluated])
    output["mean_consensus"] = (
        np.mean([value[0] for value in output.values()], axis=0),
        np.mean([value[1] for value in output.values()], axis=0),
    )
    output["strict_consensus"] = (
        np.min([output["lda"][0], output["logistic"][0]], axis=0),
        np.min([output["lda"][1], output["logistic"][1]], axis=0),
    )
    return output


def nested_gate(y, groups, baseline, probabilities):
    final = baseline.copy()
    choices = {}
    for held_out in sorted(set(groups)):
        meta = groups != held_out
        meta_baseline = report(y[meta], baseline[meta], groups[meta])
        feasible = []
        for kind, (p2, p3) in probabilities.items():
            for threshold_23 in THRESHOLDS:
                for threshold_32 in THRESHOLDS:
                    prediction = gate(baseline[meta], p2[meta], p3[meta],
                                      threshold_23, threshold_32)
                    metrics = report(y[meta], prediction, groups[meta])
                    if (safe(metrics, meta_baseline)
                            and safe_per_video(y[meta], prediction,
                                               baseline[meta], groups[meta])):
                        feasible.append((rank(metrics, meta_baseline,
                                              threshold_23, threshold_32),
                                         kind, threshold_23, threshold_32))
        use = groups == held_out
        if feasible:
            _, kind, threshold_23, threshold_32 = max(feasible)
            p2, p3 = probabilities[kind]
            final[use] = gate(baseline[use], p2[use], p3[use],
                              threshold_23, threshold_32)
            choices[held_out] = {"model": kind, "threshold_2_to_3": threshold_23,
                                 "threshold_3_to_2": threshold_32}
        else:
            choices[held_out] = {"action": "keep baseline"}
    return final, choices


def attainable_safe_point(y, groups, baseline, probabilities):
    baseline_metrics = report(y, baseline, groups)
    feasible = []
    for kind, (p2, p3) in probabilities.items():
        for threshold_23 in THRESHOLDS:
            for threshold_32 in THRESHOLDS:
                prediction = gate(baseline, p2, p3, threshold_23, threshold_32)
                metrics = report(y, prediction, groups)
                if (safe(metrics, baseline_metrics)
                        and safe_per_video(y, prediction, baseline, groups)):
                    feasible.append((rank(metrics, baseline_metrics,
                                          threshold_23, threshold_32), {
                        "model": kind, "threshold_2_to_3": threshold_23,
                        "threshold_3_to_2": threshold_32, "metrics": metrics,
                    }))
    return max(feasible, key=lambda item: item[0])[1] if feasible else None


def plot_trajectories(path, groups, times, raw, smooth, selected, labels):
    runs = sorted(set(groups))
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 1.35 * len(runs)))
    for axis, run in zip(axes, runs):
        start, end = REACTION_BOUNDS[run]
        use = (groups == run) & (times >= start) & (times <= end)
        order = np.flatnonzero(use)[np.argsort(times[use])]
        axis.plot(times[order], raw[order], color="#bbb", lw=.7, label="frame probability")
        axis.plot(times[order], smooth[order], color="#512da8", lw=1.4,
                  label="monotonic teacher")
        for value, color in ((2, "#42a5f5"), (3, "#ff8f00")):
            chosen = order[selected[order] & (labels[order] == value)]
            axis.scatter(times[chosen], smooth[chosen], s=7, color=color)
        axis.set(ylabel=run, ylim=(-.04, 1.04))
        axis.grid(alpha=.15)
    axes[0].legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Video time (s)")
    fig.suptitle("Training-only monotonic H2 2/3 trajectory teacher")
    fig.tight_layout()
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
    plot_values = None
    for certainty in (.65, .75, .85, .90):
        teacher_y, teacher_selected, raw, smooth = trajectory_teacher(
            px, initial_y, pg, pt, initial_selected, certainty)
        probabilities = probability_families(
            bx, bg, px, teacher_y, pg, teacher_selected, evaluated)
        prediction, choices = nested_gate(y, groups, baseline, probabilities)
        metrics = report(y, prediction, groups)
        variants[str(certainty)] = {
            "teacher_rows": int(teacher_selected.sum()),
            "teacher_2": int((teacher_selected & (teacher_y == 2)).sum()),
            "teacher_3": int((teacher_selected & (teacher_y == 3)).sum()),
            "metrics": metrics,
            "changed_frames": int(np.sum(prediction != baseline)),
            "constraints_passed": bool(safe(metrics, baseline_metrics)
                                       and safe_per_video(y, prediction, baseline, groups)),
            "diagnostic_safe_point": attainable_safe_point(
                y, groups, baseline, probabilities),
            "choices": choices,
        }
        if certainty == .75:
            plot_values = (raw, smooth, teacher_selected, teacher_y)

    selected_name = max(variants, key=lambda name: (
        variants[name]["constraints_passed"],
        variants[name]["metrics"]["recall"]["2"],
        variants[name]["metrics"]["recall"]["3"],
        variants[name]["metrics"]["video_macro_exact"],
        -variants[name]["metrics"]["mae_percent_points"],
    ))
    selected_variant = variants[selected_name]
    payload = {
        "protocol": "training-only monotonic trajectory teacher; single-frame student",
        "runtime": "one calibration image plus one measurement image; no accumulation",
        "baseline": baseline_metrics,
        "selected_certainty": selected_name,
        "selected": selected_variant,
        "variants": variants,
        "deployment_ready": bool(selected_variant["constraints_passed"]),
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    raw, smooth, chosen, teacher_y = plot_values
    plot_trajectories(args.output / "trajectory_teacher.png", pg, pt, raw, smooth,
                      chosen, teacher_y)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
