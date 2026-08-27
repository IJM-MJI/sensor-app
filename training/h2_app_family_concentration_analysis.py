"""Train deployable app-domain H2 range models for routed environments A/B."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ordinal_concentration_analysis import FEATURE_NAMES, augment


VIDEO_RUN = {
    "1_90_H2_only_test_2.mp4": "test_2",
    "1_90_H2_only_test_3.MOV": "test_3",
    "1_90_H2_only_test.mp4": "test",
    "1_90_RH20_2_x2.mp4": "run2",
    "1_90_RH20_3_x2.mp4": "run3",
    "1_90_RH20_4_x2.mp4": "run4",
}
FAMILIES = {"A": ("test_2", "test_3", "test", "run2"),
            "B": ("run3", "run4")}
DISPLAY = {0: "0", 1: "1–2", 2: "2–3", 3: "4"}

# Final user-reviewed optical landmarks. These are deliberately narrower than
# nominal ramps and contain no recovery path except confirmed exact-zero tails.
WINDOWS = {
    "test_2": {0: ((0, 4),), 1: ((18.5, 22.5),),
               2: ((24, 27.5),), 3: ((29, 31),)},
    "test_3": {0: ((0, 3),), 1: ((18, 28),), 2: ((60, 150),)},
    "test": {0: ((0, 3),), 1: ((22, 27),),
             2: ((28, 38),), 3: ((70, 100),)},
    "run2": {0: ((10, 14), (83, 85)), 1: ((42, 60),)},
    "run3": {0: ((0, 4), (95, 97)), 1: ((35, 55),), 2: ((55, 60),)},
    # App cache uses the 2x-speed file. User-confirmed boundaries at 30/50 s
    # in the 181 s normal-speed run therefore become 15/25 s here.
    "run4": {0: ((0, 4),), 1: ((15, 25),), 2: ((25, 45),)},
}

FEATURES = {
    "flame_lab": (0, 1, 2),
    "green_ab": (1, 2, 7, 8),
    "flame_reference_lab": (0, 1, 2, 6, 7, 8),
    "lab_baseline": tuple(range(15)),
    "all20": tuple(range(20)),
}


def label_at(run: str, seconds: float):
    hits = [label for label, windows in WINDOWS[run].items()
            if any(start <= seconds < end or np.isclose(seconds, end)
                   for start, end in windows)]
    return hits[0] if len(hits) == 1 else None


def load_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            run = VIDEO_RUN.get(row["video"])
            if run is None:
                continue
            label = label_at(run, float(row["time"]))
            if label is None:
                continue
            vector = np.asarray(augment(row, "flame"), dtype=float)
            if len(vector) != 20:
                raise RuntimeError(f"Expected 20 app features, got {len(vector)}")
            rows.append((vector, label, run, float(row["time"])))
    return (np.asarray([row[0] for row in rows]),
            np.asarray([row[1] for row in rows]),
            np.asarray([row[2] for row in rows]),
            np.asarray([row[3] for row in rows]))


def estimator(kind):
    if kind == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"))
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.2, max_iter=4000, class_weight="balanced", random_state=42))
    return make_pipeline(StandardScaler(), SVC(
        C=1, gamma="scale", class_weight="balanced", random_state=42))


def stable_blocks(x, y, groups, train, labels, fraction, cap):
    keep = np.zeros(len(y), dtype=bool)
    for run in sorted(set(groups[train])):
        for label in labels:
            index = np.flatnonzero(train & (groups == run) & (y == label))
            if not len(index):
                continue
            values = x[index]
            center = np.median(values, axis=0)
            scale = np.maximum(np.median(np.abs(values - center), axis=0), .12)
            distance = np.sqrt(np.mean(((values - center) / scale) ** 2, axis=1))
            index = index[distance <= np.quantile(distance, fraction)]
            if len(index) > cap:
                positions = np.linspace(0, len(index) - 1, cap).round().astype(int)
                index = index[np.unique(positions)]
            keep[index] = True
    return keep


def balanced_weights(y, groups, selected, labels):
    weights = np.zeros(len(y), dtype=float)
    for label in labels:
        runs = sorted(set(groups[selected & (y == label)]))
        for run in runs:
            block = selected & (groups == run) & (y == label)
            weights[block] = 1 / (max(len(runs), 1) * max(int(block.sum()), 1))
    weights[selected] *= selected.sum() / max(weights[selected].sum(), 1e-9)
    return weights


def score(y, prediction, groups, labels):
    matrix = confusion_matrix(y, prediction, labels=labels)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    per_run = {run: float(np.mean(prediction[groups == run] == y[groups == run]))
               for run in sorted(set(groups))}
    return {"exact": float(np.mean(prediction == y)),
            "video_macro_exact": float(np.mean(list(per_run.values()))),
            "minimum_recall": float(recall.min()),
            "recall": {DISPLAY[label]: float(recall[i])
                       for i, label in enumerate(labels)},
            "confusion": matrix.tolist(), "per_run_exact": per_run,
            "support": {DISPLAY[label]: int(matrix[i].sum())
                        for i, label in enumerate(labels)}}


def evaluate(x, y, groups, family, labels, feature, fraction, cap, kind):
    use = np.isin(groups, family) & np.isin(y, labels)
    fx, fy, fg = x[use][:, feature], y[use], groups[use]
    prediction = np.full(len(fy), -1)
    for held_out in family:
        test, train = fg == held_out, fg != held_out
        if not test.any():
            continue
        selected = stable_blocks(fx, fy, fg, train, labels, fraction, cap)
        if set(fy[selected]) != set(labels):
            return None
        model = estimator(kind)
        weights = balanced_weights(fy, fg, selected, labels)
        if kind in ("logistic", "svm"):
            model.fit(fx[selected], fy[selected], **{
                f"{model.steps[-1][0]}__sample_weight": weights[selected]})
        else:
            model.fit(fx[selected], fy[selected])
        prediction[test] = model.predict(fx[test])
    return score(fy, prediction, fg, labels)


def current_model(path: Path):
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text.split("=", 1)[1].rsplit(";", 1)[0])
    return payload["models"]["h2"]


def current_prediction(model, x):
    raw = model["intercept"] + x @ np.asarray(model["coefficients"])
    exact = np.asarray(model["levels"])[
        np.argmin(np.abs(raw[:, None] - np.asarray(model["levels"])[None, :]), axis=1)]
    return np.asarray([{0: 0, 1: 1, 2: 1, 3: 2, 4: 3}[int(value)] for value in exact])


def train_final(x, y, groups, family, labels, feature, fraction, cap, kind):
    use = np.isin(groups, family) & np.isin(y, labels)
    selected = stable_blocks(x[:, feature], y, groups, use, labels, fraction, cap)
    model = estimator(kind)
    weights = balanced_weights(y, groups, selected, labels)
    if kind in ("logistic", "svm"):
        model.fit(x[selected][:, feature], y[selected], **{
            f"{model.steps[-1][0]}__sample_weight": weights[selected]})
    else:
        model.fit(x[selected][:, feature], y[selected])
    return model, selected


def export_model(model, kind, feature, labels):
    scaler = model.named_steps["standardscaler"]
    common = {"features": [FEATURE_NAMES["H2"][index] for index in feature],
              "classes": list(labels),
              "display_levels": [DISPLAY[label] for label in labels]}
    if kind in ("lda", "logistic"):
        fitted = model.steps[-1][1]
        coefficient = fitted.coef_ / scaler.scale_[None, :]
        intercept = fitted.intercept_ - coefficient @ scaler.mean_
        return {**common, "type": "linear_scores", "coefficients": coefficient.tolist(),
                "intercepts": intercept.tolist()}
    svc = model.named_steps["svc"]
    return {**common, "type": "standardized_rbf_svm",
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "support_vectors": svc.support_vectors_.tolist(),
            "dual_coef": svc.dual_coef_.tolist(), "intercept": svc.intercept_.tolist(),
            "n_support": svc.n_support_.tolist(), "gamma": float(svc._gamma)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--current-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-js", type=Path)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    x, y, groups, times = load_rows(args.cache)
    old = current_model(args.current_model)
    payload = {"protocol": "app-domain reviewed optical landmarks; complete video held out",
               "families": {}, "deployment_ready": True}
    exports = {"schema_version": 1, "models": {}}
    figures = []
    for family_name, labels in (("A", (0, 1, 2, 3)), ("B", (0, 1, 2))):
        family = FAMILIES[family_name]
        use = np.isin(groups, family) & np.isin(y, labels)
        baseline = score(y[use], current_prediction(old, x[use]), groups[use], labels)
        results = {}
        for feature_name, feature in FEATURES.items():
            for fraction in (.50, .70, .90):
                for cap in (6, 12, 20):
                    for kind in ("lda", "logistic", "svm"):
                        result = evaluate(x, y, groups, family, labels, feature,
                                          fraction, cap, kind)
                        if result is not None:
                            results[f"{feature_name}_{kind}_f{fraction:.2f}_cap{cap}"] = result
        selected = max(results, key=lambda name: (
            results[name]["minimum_recall"], results[name]["video_macro_exact"],
            results[name]["exact"]))
        best = results[selected]
        feature_name, kind, fraction_text, cap_text = selected.split("_")[-4:]
        # Feature profile names contain underscores; recover them from the known prefix.
        feature_name = next(name for name in FEATURES if selected.startswith(name + "_"))
        remainder = selected[len(feature_name) + 1:].split("_")
        kind, fraction, cap = remainder[0], float(remainder[1][1:]), int(remainder[2][3:])
        fitted, retained = train_final(x, y, groups, family, labels,
                                      FEATURES[feature_name], fraction, cap, kind)
        exports["models"][family_name] = export_model(
            fitted, kind, FEATURES[feature_name], labels)
        improved = (best["exact"] > baseline["exact"] and
                    best["minimum_recall"] >= baseline["minimum_recall"])
        payload["families"][family_name] = {
            "runs": list(family), "labels": [DISPLAY[label] for label in labels],
            "selected": selected, "held_out": best, "current_single_model": baseline,
            "improved": improved, "retained_final": int(retained.sum()), "models": results}
        payload["deployment_ready"] &= improved
        figures.append((family_name, labels, best))
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    exports["validation"] = {name: {key: payload["families"][name]["held_out"][key]
                                    for key in ("exact", "minimum_recall", "recall")}
                             for name in ("A", "B")}
    js = "// Generated by training/h2_app_family_concentration_analysis.py; do not edit by hand.\n"
    js += "window.SENSOR_H2_FAMILY_MODEL=" + json.dumps(exports, separators=(",", ":")) + ";\n"
    (args.output / "sensor-h2-family-model.js").write_text(js, encoding="utf-8")
    if args.export_js and payload["deployment_ready"]:
        args.export_js.write_text(js, encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
    for axis, (family_name, labels, result) in zip(axes, figures):
        matrix = np.asarray(result["confusion"]); axis.imshow(matrix, cmap="Blues")
        for row in range(len(labels)):
            for column in range(len(labels)):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        ticks = [DISPLAY[label] for label in labels]
        axis.set(xticks=range(len(labels)), xticklabels=ticks,
                 yticks=range(len(labels)), yticklabels=ticks,
                 xlabel="Predicted", ylabel="Reference",
                 title=f"Family {family_name}: {result['exact']:.1%}\nmin recall {result['minimum_recall']:.1%}")
    fig.savefig(args.output / "app_family_confusions.png", dpi=190); plt.close(fig)
    print(json.dumps({"deployment_ready": payload["deployment_ready"],
                      **{name: {"selected": payload["families"][name]["selected"],
                                "held_out": payload["families"][name]["held_out"],
                                "current": payload["families"][name]["current_single_model"],
                                "improved": payload["families"][name]["improved"]}
                         for name in ("A", "B")}}, indent=2))


if __name__ == "__main__":
    main()
