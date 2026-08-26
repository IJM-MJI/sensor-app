"""Nested four-range H2 trajectory matching with a single-frame student."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from h2_four_range_analysis import DISPLAY, ZERO_WINDOWS, in_windows
from h2_optical_pseudolabel_analysis import REACTION_BOUNDS


def estimator(kind: str):
    if kind == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"))
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=.2, max_iter=4000, class_weight="balanced", random_state=42))


def prepare_base(saved):
    x, old_y, groups, times = (saved["x"], saved["y"], saved["groups"],
                               saved["times"])
    keep = np.asarray([int(label) != 0 or in_windows(float(time), ZERO_WINDOWS[str(run)])
                       for label, run, time in zip(old_y, groups, times)])
    mapping = {0: 0, 2: 1, 3: 2, 4: 3}
    return (x[keep], np.asarray([mapping[int(v)] for v in old_y[keep]]),
            groups[keep], times[keep])


def teacher(initial, px, pg, pt, held_out, tolerance, cap, allowed_runs):
    """Create ordered optical landmarks without using the held-out run."""
    train_groups = (pg != held_out) & np.isin(pg, tuple(allowed_runs))
    probability = initial.predict_proba(px)
    classes = np.asarray(initial.classes_, dtype=float)
    raw = probability @ classes
    smooth = raw.copy()
    selected = np.zeros(len(px), dtype=bool)
    labels = np.full(len(px), -1, dtype=int)

    for run in sorted(set(pg[train_groups])):
        run_mask = (pg == run) & train_groups
        first = float(pt[run_mask].min())
        zero = run_mask & (pt <= first + 2)
        selected[zero] = True
        labels[zero] = 0
        if run not in REACTION_BOUNDS:
            continue
        start, end = REACTION_BOUNDS[run]
        reaction = run_mask & (pt >= start) & (pt <= end)
        order = np.flatnonzero(reaction)[np.argsort(pt[reaction])]
        if len(order) < 3:
            continue
        smooth[order] = IsotonicRegression(
            y_min=0, y_max=3, increasing=True, out_of_bounds="clip"
        ).fit_transform(pt[order], raw[order])
        nearest = np.rint(smooth[order]).astype(int)
        confident = np.abs(smooth[order] - nearest) <= tolerance
        for label in (1, 2, 3):
            candidates = order[confident & (nearest == label)]
            if len(candidates) > cap:
                positions = np.linspace(0, len(candidates) - 1, cap).round().astype(int)
                candidates = candidates[np.unique(positions)]
            selected[candidates] = True
            labels[candidates] = label

    # Cap exact-zero blocks too, so long initial regions cannot dominate.
    for run in sorted(set(pg[selected])):
        candidates = np.flatnonzero(selected & (pg == run) & (labels == 0))
        if len(candidates) > cap:
            keep = candidates[np.linspace(0, len(candidates) - 1, cap).round().astype(int)]
            selected[candidates] = False
            selected[np.unique(keep)] = True
    return labels, selected, raw, smooth


def balanced_weights(labels, groups, selected):
    weights = np.zeros(len(labels), dtype=float)
    for label in range(4):
        runs = sorted(set(groups[selected & (labels == label)]))
        for run in runs:
            block = selected & (groups == run) & (labels == label)
            weights[block] = 1 / (max(len(runs), 1) * max(int(block.sum()), 1))
    weights[selected] *= selected.sum() / weights[selected].sum()
    return weights


def metrics(y, prediction, groups):
    matrix = confusion_matrix(y, prediction, labels=range(4))
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    per_run = {run: float(np.mean(prediction[groups == run] == y[groups == run]))
               for run in sorted(set(groups))}
    return {
        "exact": float(np.mean(prediction == y)),
        "video_macro_exact": float(np.mean(list(per_run.values()))),
        "mae_ranges": float(np.mean(np.abs(prediction - y))),
        "recall": {DISPLAY[i]: float(recall[i]) for i in range(4)},
        "minimum_recall": float(recall.min()),
        "confusion": matrix.tolist(), "per_run_exact": per_run,
    }


def evaluate(bx, by, bg, px, pg, pt, kind, tolerance, cap, allowed_runs):
    prediction = np.full(len(by), -1)
    teacher_counts = {}
    for held_out in sorted(set(bg)):
        base_train = bg != held_out
        initial = estimator(kind)
        initial.fit(bx[base_train], by[base_train])
        labels, selected, *_ = teacher(
            initial, px, pg, pt, held_out, tolerance, cap, allowed_runs)
        if set(labels[selected]) != set(range(4)):
            return None
        student = estimator(kind)
        weights = balanced_weights(labels, pg, selected)
        if kind == "logistic":
            student.fit(px[selected], labels[selected], **{
                f"{student.steps[-1][0]}__sample_weight": weights[selected]})
        else:
            # LDA has no sample_weight argument. Temporal caps still keep each
            # run/class block approximately balanced.
            student.fit(px[selected], labels[selected])
        test = bg == held_out
        prediction[test] = student.predict(bx[test])
        teacher_counts[held_out] = {
            DISPLAY[label]: int(np.sum(selected & (labels == label)))
            for label in range(4)
        }
    result = metrics(by, prediction, bg)
    result["teacher_counts_by_fold"] = teacher_counts
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--pseudo-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    bx, by, bg, _ = prepare_base(np.load(args.base_cache, allow_pickle=False))
    pseudo = np.load(args.pseudo_cache, allow_pickle=False)
    px, pg, pt = pseudo["x"], pseudo["groups"], pseudo["times"]

    variants = {}
    pools = {
        "all": set(pg),
        "official": set(bg),
        "trusted90": set(bg) - {"run2"},
    }
    for pool_name, allowed_runs in pools.items():
        for kind in ("lda", "logistic"):
            for tolerance in (.20, .30, .40):
                for cap in (8, 16, 24):
                    result = evaluate(bx, by, bg, px, pg, pt,
                                      kind, tolerance, cap, allowed_runs)
                    if result is not None:
                        name = f"{pool_name}_{kind}_tol{tolerance:.2f}_cap{cap}"
                        variants[name] = result
    selected = max(variants, key=lambda name: (
        variants[name]["minimum_recall"], variants[name]["video_macro_exact"],
        variants[name]["exact"], -variants[name]["mae_ranges"]))
    best = variants[selected]
    payload = {
        "protocol": "nested complete-video holdout; monotonic teacher; equal run/class weight",
        "runtime": "single calibration-relative frame; trajectory used only for training labels",
        "selected": selected, "selected_result": best, "variants": variants,
        "deployment_target_all_recalls_0.85": bool(best["minimum_recall"] >= .85),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    matrix = np.asarray(best["confusion"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].imshow(matrix, cmap="Blues")
    for row in range(4):
        for column in range(4):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes[0].set(xticks=range(4), xticklabels=DISPLAY, yticks=range(4),
                yticklabels=DISPLAY, xlabel="Predicted H2 range",
                ylabel="Reference H2 range", title=selected)
    recall = [best["recall"][label] for label in DISPLAY]
    axes[1].bar(DISPLAY, recall, color=("#66bb6a", "#42a5f5", "#ffa726", "#ef5350"))
    axes[1].axhline(.85, color="#555", ls="--", lw=1, label="0.85 target")
    axes[1].set(ylim=(0, 1), ylabel="Recall", title="Trajectory-matched recall")
    axes[1].legend()
    fig.savefig(args.output / "trajectory_matching_confusion.png", dpi=190)
    plt.close(fig)
    print(json.dumps({"selected": selected, **best}, indent=2))


if __name__ == "__main__":
    main()
