"""Evaluate a consensus optical H2 scale: 0-1 / 2 / 3 / 4%.

This is an audit model, not an application exporter.  The new 4% band means
the strongest *repeatable* response shared by independent runs.  The unusually
strong tail of H2_only_test_2 is deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Avoid joblib's Windows physical-core probe, which emits a localized subprocess
# message that some cp949 consoles cannot decode.  Models remain single-process.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from h2_more_crop_fixed_mask_analysis import RUNS as TIGHT_RUNS, extract_run as extract_tight
from h2_other_run_reference_matching import OTHER_RUNS, extract_other
from h2_rh20_max_response_analysis import RUNS as RH20_RUNS, extract as extract_rh20


LABELS = (0, 2, 3, 4)
DISPLAY = ("0-1", "2", "3", "4")

# Explicit optical-response windows.  They are intentionally narrower than the
# supplied ramps: endpoint neighbourhoods and unambiguous ramp intervals only.
# The test_2 tail after 31 s is absent because it is stronger than the common
# maximum seen in the other runs.
WINDOWS = {
    "test_2": {0: [(0, 13)], 2: [(18.5, 22.5)], 3: [(24, 27.5)], 4: [(29, 31)]},
    "test_3": {0: [(0, 10)], 2: [(18, 22)], 3: [(24, 28)], 4: [(60, 150)]},
    "test": {0: [(0, 15)], 2: [(22, 27)], 3: [(28, 38)], 4: [(70, 100)]},
    # RH20 clips contribute only where the earlier fixed-mask audit found a
    # corresponding optical state.  Weak runs are never promoted to 4%.
    "run2": {0: [(10, 25), (83, 85)], 2: [(42, 60)]},
    "run3": {0: [(0, 15), (95, 97)], 2: [(35, 50)], 3: [(50, 60)]},
    "run4": {0: [(0, 25), (178, 180)], 2: [(55, 78)], 3: [(82, 105)], 4: [(110, 120)]},
    "run5_x2": {0: [(0, 10), (105, 107)], 2: [(22, 32)], 3: [(34, 43)], 4: [(44, 54)]},
    "run5_normal": {0: [(0, 18), (212, 214)], 2: [(42, 62)], 3: [(68, 90)]},
}


def in_windows(seconds: float, windows: list[tuple[float, float]]) -> bool:
    return any(start <= seconds <= end for start, end in windows)


def label_rows(rows: list[dict], run: str, feature_key: str) -> list[dict]:
    output = []
    for row in rows:
        hits = [label for label, windows in WINDOWS[run].items()
                if in_windows(row["time"], windows)]
        if len(hits) == 1:
            output.append({"run": run, "time": row["time"], "y": hits[0],
                           "x": np.asarray(row[feature_key], dtype=float)})
    return output


def extract_all(video_root: Path, sample_hz: float) -> list[dict]:
    rows = []
    for run in ("test_2", "test_3"):
        extracted, *_ = extract_tight(video_root, run, TIGHT_RUNS[run], sample_hz)
        rows.extend(label_rows(extracted, run, "feature"))
    extracted, *_ = extract_other(video_root, "test", OTHER_RUNS["test"], sample_hz)
    rows.extend(label_rows(extracted, "test", "feature"))
    for run in RH20_RUNS:
        extracted, *_ = extract_rh20(video_root, run, RH20_RUNS[run], sample_hz)
        # RH20 extractor has a single flame delta; pad to the shared 22-column
        # representation.  Model evaluation below uses only the common first
        # six fixed-flame statistics.
        for row in extracted:
            row["feature"] = row["delta"]
        rows.extend(label_rows(extracted, run, "feature"))
    return rows


def metrics(y: np.ndarray, predicted: np.ndarray, groups: np.ndarray) -> dict:
    cm = confusion_matrix(y, predicted, labels=LABELS)
    recall = np.divide(np.diag(cm), cm.sum(axis=1),
                       out=np.zeros(len(LABELS), dtype=float), where=cm.sum(axis=1) > 0)
    per_run = {run: float(accuracy_score(y[groups == run], predicted[groups == run]))
               for run in sorted(set(groups))}
    return {
        "exact": float(accuracy_score(y, predicted)),
        "video_macro_exact": float(np.mean(list(per_run.values()))),
        "mae_percent_points": float(np.mean(np.abs(y - predicted))),
        "per_class_recall": {DISPLAY[i]: float(recall[i]) for i in range(len(LABELS))},
        "per_run_exact": per_run,
        "confusion": cm.tolist(),
        "support": {DISPLAY[i]: int(cm[i].sum()) for i in range(len(LABELS))},
        "three_to_four_rate": float(cm[2, 3] / max(cm[2].sum(), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=2.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # v2 also retains timestamps so later landmark/trajectory audits can trace
    # every prediction back to a reviewable video frame.
    cache = args.output / "labelled_fixed_mask_rows_v3.npz"
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        x, y, groups, times = (saved["x"], saved["y"], saved["groups"],
                               saved["times"])
        row_count = len(y)
    else:
        rows = extract_all(args.video_root, args.sample_hz)
        # Mean/median Lab plus five within-flame chroma percentiles are common
        # to every fixed-mask extractor.  Percentiles preserve partial colour
        # change that is hidden by a single ROI average.
        x = np.asarray([row["x"][:11] for row in rows])
        y = np.asarray([row["y"] for row in rows])
        groups = np.asarray([row["run"] for row in rows])
        times = np.asarray([row["time"] for row in rows])
        row_count = len(rows)
        np.savez_compressed(cache, x=x, y=y, groups=groups, times=times)
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(
            C=.2, max_iter=4000, class_weight="balanced", random_state=42)),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=4,
            class_weight="balanced", random_state=42, n_jobs=1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=.035, max_leaf_nodes=11,
            min_samples_leaf=10, l2_regularization=4, random_state=42),
        "rbf_svm": make_pipeline(StandardScaler(), SVC(
            C=1.0, gamma="scale", class_weight="balanced", random_state=42)),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=9, min_samples_leaf=4,
            class_weight="balanced_subsample", random_state=42, n_jobs=1),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=17, weights="distance", p=2)),
        "shrinkage_lda": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto")),
    }
    results = {}
    predictions = {}
    for name, model in models.items():
        predicted = np.full(len(y), -1)
        for held_out in sorted(set(groups)):
            train, test = groups != held_out, groups == held_out
            # A fold is valid only if its training runs contain all four bands.
            if set(y[train]) != set(LABELS):
                raise RuntimeError(f"Fold {held_out} lacks a training class")
            model.fit(x[train], y[train])
            predicted[test] = model.predict(x[test])
        results[name] = metrics(y, predicted, groups)
        predictions[name] = predicted

    # Select by video-macro accuracy, then by 3% recall and low 3->4 leakage.
    selected = max(results, key=lambda name: (
        results[name]["video_macro_exact"],
        results[name]["per_class_recall"]["3"],
        -results[name]["three_to_four_rate"]))
    payload = {
        "definition": {
            "bands": list(DISPLAY),
            "four_percent": "common stable optical maximum",
            "excluded": "test_2 >31 s over-strong tail and all recovery ramps",
            "validation": "complete run held out; reference-anchored optical labels",
        },
        "rows": row_count,
        "selected": selected,
        "models": results,
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best = results[selected]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    image = axes[0].imshow(np.asarray(best["confusion"]), cmap="Blues")
    for i, row in enumerate(best["confusion"]):
        for j, value in enumerate(row):
            axes[0].text(j, i, value, ha="center", va="center")
    axes[0].set(xticks=range(4), xticklabels=DISPLAY, yticks=range(4),
                yticklabels=DISPLAY, xlabel="Predicted H2 band", ylabel="Reference H2 band",
                title=f"Video-held-out: {selected}")
    fig.colorbar(image, ax=axes[0], fraction=.046)
    recall = [best["per_class_recall"][label] for label in DISPLAY]
    axes[1].bar(DISPLAY, recall, color=["#66bb6a", "#42a5f5", "#ffa726", "#ef5350"])
    axes[1].axhline(.85, color="#444", ls="--", lw=1, label="target 0.85")
    axes[1].set(ylim=(0, 1), xlabel="H2 band", ylabel="Recall", title="Held-out recall")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.output / "consensus_four_band_confusion.png", dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
