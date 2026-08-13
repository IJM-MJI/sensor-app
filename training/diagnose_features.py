"""Summarise calibrated feature separation by task and recording run."""

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from train_models import MODEL_FEATURES, read_csv


def main() -> None:
    rows = read_csv(Path("training/cache/v3-region-specific-expanded-timelines/features.csv"))
    for task in ("h2_present", "rh_high"):
        print(f"\n[{task}]")
        labelled = [r for r in rows if r[task] is not None]
        categories = ["all"] + sorted({str(r["group"]).split("-rh")[0] for r in labelled})
        for category in categories:
            use = labelled if category == "all" else [r for r in labelled if str(r["group"]).split("-rh")[0] == category]
            y = np.asarray([int(r[task]) for r in use])
            if len(set(y)) < 2:
                continue
            scored = []
            for name in MODEL_FEATURES:
                x = np.asarray([float(r[name]) for r in use])
                auc = roc_auc_score(y, x)
                scored.append((max(auc, 1 - auc), name, "+" if auc >= .5 else "-"))
            best = sorted(scored, reverse=True)[:4]
            print(f"  {category:12s} n={len(use):4d} pos={y.mean():.2f}  " +
                  " ".join(f"{name}{direction}:{auc:.3f}" for auc, name, direction in best))


if __name__ == "__main__":
    main()
