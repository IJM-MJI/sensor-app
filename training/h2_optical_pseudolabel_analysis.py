"""Select H2 2/3% frames by optical state, using timelines only for phase bounds.

The fixed test_2 flame appearances at 20--22 s and 29--31 s are the only 2/3
class anchors. Other H2-only and RH20 frames receive a label only when their
fixed-mask colour is close to one anchor and clearly farther from the other.
"""

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
import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from h2_more_crop_fixed_mask_analysis import (RUNS as TIGHT_RUNS, canonical, content_crop,
                                               extract_run as extract_tight, fixed_shape_mask,
                                               frame_at, substrate, summary)
from h2_other_run_reference_matching import (OTHER_RUNS, card_zones, extract_other,
                                              resize_roi)
from h2_rh20_max_response_analysis import RUNS as RH20_RUNS, extract as extract_rh20


REACTION_BOUNDS = {
    "test_2": (4, 31),       # exclude the anomalously strong tail
    "test_3": (10, 150),     # weak run remains near the common 3 state
    "test": (15, 100),
    "h2_run4": (13, 109),
    "h2_run5": (8, 130),
    **{name: tuple(config["reaction"]) for name, config in RH20_RUNS.items()},
    "angle80_run2": (8, 138),
}

ANGLE80 = {"file": "1_80_2.MOV", "roi": (1000, 160, 1370, 570)}


def extract_angle80(video_root: Path, sample_hz: float):
    cap = cv2.VideoCapture(str(video_root / ANGLE80["file"]))
    if not cap.isOpened():
        raise FileNotFoundError(video_root / ANGLE80["file"])
    fps = cap.get(cv2.CAP_PROP_FPS)

    def image(seconds):
        rotated = cv2.rotate(frame_at(cap, seconds), cv2.ROTATE_90_COUNTERCLOCKWISE)
        return resize_roi(rotated, ANGLE80["roi"])

    calibration = image(2.0)
    lab0 = cv2.cvtColor(calibration, cv2.COLOR_BGR2LAB).astype(float)
    flame_zone, drop_zone, card = card_zones(lab0.shape[:2], flame_y=.34, drop_y=.72)
    background0 = substrate(lab0, card, flame_zone | drop_zone)
    flame_mask = fixed_shape_mask(lab0, flame_zone, background0)
    baseline = summary(lab0, flame_mask, background0)
    rows = []
    for seconds in np.arange(0, 139, 1 / sample_hz):
        lab = cv2.cvtColor(image(float(seconds)), cv2.COLOR_BGR2LAB).astype(float)
        background = substrate(lab, card, flame_zone | drop_zone)
        rows.append({"run": "angle80_run2", "time": float(seconds),
                     "x": (summary(lab, flame_mask, background) - baseline)[:11]})
    cap.release()
    return rows


def extract_all(video_root: Path, sample_hz: float):
    rows = []
    for run in ("test_2", "test_3"):
        extracted, *_ = extract_tight(video_root, run, TIGHT_RUNS[run], sample_hz)
        rows.extend({"run": run, "time": row["time"], "x": row["feature"][:11]}
                    for row in extracted)
    for source, target in (("test", "test"), ("run4", "h2_run4"), ("run5", "h2_run5")):
        extracted, *_ = extract_other(video_root, source, OTHER_RUNS[source], sample_hz)
        rows.extend({"run": target, "time": row["time"], "x": row["feature"][:11]}
                    for row in extracted)
    for run, config in RH20_RUNS.items():
        extracted, *_ = extract_rh20(video_root, run, config, sample_hz)
        rows.extend({"run": run, "time": row["time"], "x": row["delta"][:11]}
                    for row in extracted)
    rows.extend(extract_angle80(video_root, sample_hz))
    return rows


def load_or_extract(args):
    cache = args.output / "full_reaction_fixed_mask_rows_v2_angle80.npz"
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        return saved["x"], saved["groups"], saved["times"]
    rows = extract_all(args.video_root, args.sample_hz)
    x = np.asarray([row["x"] for row in rows])
    groups = np.asarray([row["run"] for row in rows])
    times = np.asarray([row["time"] for row in rows])
    np.savez_compressed(cache, x=x, groups=groups, times=times)
    return x, groups, times


def optical_labels(x, groups, times, margin_threshold=.16, distance_multiplier=3.0):
    anchor2 = (groups == "test_2") & (times >= 20) & (times <= 22)
    anchor3 = (groups == "test_2") & (times >= 29) & (times <= 31)
    center2, center3 = np.median(x[anchor2], axis=0), np.median(x[anchor3], axis=0)
    # Robust scaling is fixed from the original 90-degree pool.  This covers
    # its lighting variation while ensuring a later angle-80 candidate cannot
    # move the metric and create a false A/B improvement.
    reaction = np.zeros(len(x), dtype=bool)
    for run, (start, end) in REACTION_BOUNDS.items():
        reaction |= (groups == run) & (times >= start) & (times <= end)
    pool = x[reaction & (groups != "angle80_run2")]
    pool_center = np.median(pool, axis=0)
    scale = np.maximum(np.median(np.abs(pool - pool_center), axis=0), .18)
    d2 = np.sqrt(np.mean(((x - center2) / scale) ** 2, axis=1))
    d3 = np.sqrt(np.mean(((x - center3) / scale) ** 2, axis=1))
    anchor_distance = np.r_[d2[anchor2], d3[anchor3]]
    distance_limit = float(np.quantile(anchor_distance, .95) * distance_multiplier)
    nearest = np.minimum(d2, d3)
    margin = np.abs(d2 - d3) / np.maximum(d2 + d3, 1e-9)
    # Response magnitude is relative to each run's own first three seconds,
    # not to test_2's absolute initial colour.  The minimum magnitude is tied
    # to 80% of the trustworthy test_2 2% anchor response.
    response_distance = np.zeros(len(x), dtype=float)
    baseline_centers = {}
    for run in sorted(set(groups)):
        use = groups == run
        first = float(np.min(times[use]))
        baseline = np.median(x[use & (times <= first + 2)], axis=0)
        baseline_centers[run] = baseline
        response_distance[use] = np.sqrt(np.mean(((x[use] - baseline) / scale) ** 2, axis=1))
    test2_baseline = baseline_centers["test_2"]
    anchor2_response = np.sqrt(np.mean(((x[anchor2] - test2_baseline) / scale) ** 2,
                                       axis=1))
    minimum_response = float(np.median(anchor2_response) * .80)
    changed_from_initial = response_distance >= minimum_response
    selected = (reaction & changed_from_initial & (nearest <= distance_limit)
                & (margin >= margin_threshold))
    labels = np.where(d2 <= d3, 2, 3)
    # The anchors are retained regardless of a boundary-frame confidence dip.
    selected |= anchor2 | anchor3
    labels[anchor2], labels[anchor3] = 2, 3
    return labels, selected, d2, d3, margin, distance_limit


def heldout(x, y, groups, selected):
    predicted = np.full(len(y), -1)
    evaluated = selected & (groups != "test_2")
    per_run = {}
    for run in sorted(set(groups[evaluated])):
        test = evaluated & (groups == run)
        # test_2 remains a fixed metrology anchor in every fold; the held-out
        # run itself never supplies training rows.
        train = selected & (groups != run)
        estimator = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"))
        estimator.fit(x[train], y[train])
        predicted[test] = estimator.predict(x[test])
        per_run[run] = float(accuracy_score(y[test], predicted[test]))
    cm = confusion_matrix(y[evaluated], predicted[evaluated], labels=(2, 3))
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return predicted, {
        "reference_matching_exact": float(accuracy_score(y[evaluated], predicted[evaluated])),
        "video_macro_exact": float(np.mean(list(per_run.values()))),
        "recall": {"2": float(recall[0]), "3": float(recall[1])},
        "confusion": cm.tolist(),
        "per_run_exact": per_run,
        "evaluated_rows": int(evaluated.sum()),
    }


def broad_reference_check(x, y, groups, selected, cache: Path, include_angle=True):
    """Train on optical labels, evaluate untouched broad 2/3 interval labels."""
    saved = np.load(cache, allow_pickle=False)
    bx, by, bg = saved["x"], saved["y"], saved["groups"]
    evaluated = np.isin(by, (2, 3)) & (bg != "test_2")
    predicted = np.full(len(by), -1)
    per_run = {}
    for run in sorted(set(bg[evaluated])):
        test = evaluated & (bg == run)
        train = selected & (groups != run)
        if not include_angle:
            train &= groups != "angle80_run2"
        estimator = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"))
        estimator.fit(x[train], y[train])
        predicted[test] = estimator.predict(bx[test])
        per_run[run] = float(accuracy_score(by[test], predicted[test]))
    cm = confusion_matrix(by[evaluated], predicted[evaluated], labels=(2, 3))
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return {
        "exact": float(accuracy_score(by[evaluated], predicted[evaluated])),
        "video_macro_exact": float(np.mean(list(per_run.values()))),
        "recall": {"2": float(recall[0]), "3": float(recall[1])},
        "confusion": cm.tolist(), "per_run_exact": per_run,
        "evaluated_rows": int(evaluated.sum()),
        "note": "independent broad interval labels may themselves contain timeline error",
    }


def plot(x, groups, times, labels, selected, output):
    runs = sorted(set(groups))
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 1.45 * len(runs)), sharex=False)
    for ax, run in zip(axes, runs):
        use = groups == run
        order = np.flatnonzero(use)[np.argsort(times[use])]
        ax.plot(times[order], x[order, 1], color="#aaa", lw=.7)
        for label, color in ((2, "#42a5f5"), (3, "#ffa726")):
            chosen = order[selected[order] & (labels[order] == label)]
            ax.scatter(times[chosen], x[chosen, 1], s=8, color=color, label=str(label))
        ax.set_ylabel(f"{run}\nΔa*")
        ax.grid(alpha=.15)
    axes[0].legend(ncol=2, title="optical pseudo-label")
    axes[-1].set_xlabel("video time (s)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def review_frame(video_root, run, seconds):
    if run in TIGHT_RUNS:
        cap = cv2.VideoCapture(str(video_root / TIGHT_RUNS[run]["file"]))
        calibration = frame_at(cap, 2.0)
        _, bounds = content_crop(calibration)
        image = canonical(frame_at(cap, seconds), bounds)
    elif run in ("test", "h2_run4", "h2_run5"):
        source = {"test": "test", "h2_run4": "run4", "h2_run5": "run5"}[run]
        config = OTHER_RUNS[source]
        cap = cv2.VideoCapture(str(video_root / config["file"]))
        image = resize_roi(frame_at(cap, seconds), config["roi"])
    elif run == "angle80_run2":
        cap = cv2.VideoCapture(str(video_root / ANGLE80["file"]))
        rotated = cv2.rotate(frame_at(cap, seconds), cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = resize_roi(rotated, ANGLE80["roi"])
        cap.release()
    else:
        config = RH20_RUNS[run]
        cap = cv2.VideoCapture(str(video_root / config["file"]))
        image = resize_roi(frame_at(cap, seconds), config["roi"])
    cap.release()
    return image


def review_montage(video_root, x, groups, times, labels, selected, margin, output):
    panels = []
    for run in sorted(set(groups)):
        for label in (2, 3):
            candidates = np.flatnonzero(selected & (groups == run) & (labels == label))
            if not len(candidates):
                continue
            # Median-confidence example avoids showing only an extreme anchor.
            order = candidates[np.argsort(margin[candidates])]
            index = order[len(order) // 2]
            image = review_frame(video_root, run, float(times[index]))
            cv2.rectangle(image, (0, 0), (image.shape[1], 32), (20, 20, 20), -1)
            text = (f"{run} t={times[index]:.1f}s optical={label}% "
                    f"margin={margin[index]:.2f}")
            cv2.putText(image, text, (7, 22), cv2.FONT_HERSHEY_SIMPLEX, .46,
                        (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(image)
    if not panels:
        return
    width, columns = 400, 2
    panels = [cv2.resize(panel, (width, round(panel.shape[0] * width / panel.shape[1])))
              for panel in panels]
    height = min(panel.shape[0] for panel in panels)
    panels = [cv2.resize(panel, (width, height)) for panel in panels]
    blank = np.zeros_like(panels[0])
    while len(panels) % columns:
        panels.append(blank.copy())
    rows = [np.hstack(panels[i:i + columns]) for i in range(0, len(panels), columns)]
    cv2.imwrite(str(output), np.vstack(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=1.0)
    parser.add_argument("--broad-cache", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x, groups, times = load_or_extract(args)
    labels, selected, d2, d3, margin, limit = optical_labels(x, groups, times)
    _, result = heldout(x, labels, groups, selected)
    coverage = {}
    for run, (start, end) in REACTION_BOUNDS.items():
        reaction = (groups == run) & (times >= start) & (times <= end)
        coverage[run] = {
            "reaction_rows": int(reaction.sum()), "selected_rows": int((reaction & selected).sum()),
            "coverage": float((reaction & selected).sum() / max(reaction.sum(), 1)),
            "label_2": int((reaction & selected & (labels == 2)).sum()),
            "label_3": int((reaction & selected & (labels == 3)).sum()),
        }
    payload = {
        "anchor": {"2": "test_2 20-22 s", "3": "test_2 29-31 s"},
        "selection": {"two_three_margin_threshold": .16,
                      "minimum_response": "80% of test_2 2% anchor magnitude",
                      "distance_limit": limit},
        "coverage": coverage,
        "heldout_reference_matching": result,
    }
    if args.broad_cache:
        payload["independent_broad_reference_check"] = {
            "without_angle80": broad_reference_check(
                x, labels, groups, selected, args.broad_cache, include_angle=False),
            "with_angle80": broad_reference_check(
                x, labels, groups, selected, args.broad_cache, include_angle=True),
        }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(x, groups, times, labels, selected, args.output / "optical_2_3_pseudolabels.png")
    review_montage(args.video_root, x, groups, times, labels, selected, margin,
                   args.output / "optical_2_3_review_montage.jpg")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
