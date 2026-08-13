"""Compare model families with leave-one-video-out predictions."""

from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from train_models import H2_FEATURES, RH_FEATURES, feature_value, read_csv


def oof(model, x, y, groups):
    prob = np.full(len(y), np.nan)
    for group in sorted(set(groups)):
        test = groups == group
        train = ~test
        fitted = clone(model).fit(x[train], y[train])
        prob[test] = fitted.predict_proba(x[test])[:, 1]
    return roc_auc_score(y, prob), balanced_accuracy_score(y, prob >= .5)


def main():
    rows = read_csv(Path("training/cache/v3-region-specific-expanded-timelines/features.csv"))
    models = {
        "linear": make_pipeline(StandardScaler(), LogisticRegression(C=.3, max_iter=3000, class_weight="balanced")),
        "poly2": make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                               LogisticRegression(C=.1, max_iter=5000, class_weight="balanced")),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                     class_weight="balanced", random_state=42, n_jobs=-1),
        "extra": ExtraTreesClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                      class_weight="balanced", random_state=42, n_jobs=-1),
    }
    for task in ("h2_present", "rh_high"):
        features = H2_FEATURES if task == "h2_present" else RH_FEATURES
        use = [r for r in rows if r[task] is not None]
        x = np.asarray([[feature_value(r, n) for n in features] for r in use])
        y = np.asarray([int(r[task]) for r in use])
        groups = np.asarray([str(r["group"]) for r in use])
        print(f"\n{task}: n={len(y)}, groups={len(set(groups))}")
        for name, model in models.items():
            auc, bal = oof(model, x, y, groups)
            print(f"  {name:8s} AUC={auc:.3f} balanced_accuracy={bal:.3f}")


if __name__ == "__main__":
    main()
