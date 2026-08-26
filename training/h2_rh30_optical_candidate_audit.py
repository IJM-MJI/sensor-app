"""Audit RH30 simultaneous clips for test_2-like H2 2/3 optical states.

This uses the existing rotation-aware fixed-boundary cache so every candidate
and the test_2 anchor share the same legacy feature extractor.  It is a data
screening audit only; RH30 rows are not exported to the concentration model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FEATURES = [
    "flame_L", "flame_a", "flame_b", "flame_L_p50", "flame_a_p50",
    "flame_b_p50", "flame_chroma_p10", "flame_chroma_p25",
    "flame_chroma_p50", "flame_chroma_p75", "flame_chroma_p90",
]


def read(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    time = np.asarray([float(row["time"]) for row in raw])
    values = np.asarray([[float(row[name]) for name in FEATURES] for row in raw])
    baseline = np.median(values[time <= 3], axis=0)
    return raw, time, values - baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference_path = next(args.cache_dir.glob(
        "1_90_H2_only_test_2_cropped.mp4.cropped-v4-centered-smooth."
        "fixed-boundary-v2.drop-p85.quant-2hz-v2-distribution.csv"))
    ref_raw, ref_time, ref_x = read(reference_path)
    anchor2 = (ref_time >= 20) & (ref_time <= 22)
    anchor3 = (ref_time >= 29) & (ref_time <= 31)
    center2, center3 = np.median(ref_x[anchor2], axis=0), np.median(ref_x[anchor3], axis=0)

    clips = []
    for path in sorted(args.cache_dir.glob("1_90_RH30*.fixed-boundary-v2.drop-p85.csv")):
        raw, time, values = read(path)
        clips.append((path.stem.split(".mp4")[0], raw, time, values))
    pool = np.vstack([ref_x, *[values for _, _, _, values in clips]])
    center = np.median(pool, axis=0)
    scale = np.maximum(np.median(np.abs(pool - center), axis=0), .2)
    anchor_distance = np.r_[
        np.sqrt(np.mean(((ref_x[anchor2] - center2) / scale) ** 2, axis=1)),
        np.sqrt(np.mean(((ref_x[anchor3] - center3) / scale) ** 2, axis=1)),
    ]
    distance_limit = float(np.quantile(anchor_distance, .95) * 3)
    results, trajectories = {}, []
    for name, raw, time, values in clips:
        d2 = np.sqrt(np.mean(((values - center2) / scale) ** 2, axis=1))
        d3 = np.sqrt(np.mean(((values - center3) / scale) ** 2, axis=1))
        margin = np.abs(d2 - d3) / np.maximum(d2 + d3, 1e-9)
        # Existing state cache marks only the late, stable reaction part as H2
        # present.  Concentration stage is not taken from its timeline.
        late_reaction = np.asarray([row["h2_present"] == "1" for row in raw])
        selected = late_reaction & (np.minimum(d2, d3) <= distance_limit) & (margin >= .16)
        label = np.where(d2 <= d3, 2, 3)
        results[name] = {
            "late_reaction_rows": int(late_reaction.sum()),
            "selected_rows": int(selected.sum()),
            "label_2": int((selected & (label == 2)).sum()),
            "label_3": int((selected & (label == 3)).sum()),
            "selected_times_2": [float(value) for value in time[selected & (label == 2)]],
            "selected_times_3": [float(value) for value in time[selected & (label == 3)]],
        }
        trajectories.append((name, time, values[:, 1], selected, label))

    fig, axes = plt.subplots(len(trajectories), 1, figsize=(9, 2 * len(trajectories)))
    for ax, (name, time, delta_a, selected, label) in zip(axes, trajectories):
        ax.plot(time, delta_a, color="#aaa", lw=.8)
        for stage, color in ((2, "#42a5f5"), (3, "#ffa726")):
            use = selected & (label == stage)
            ax.scatter(time[use], delta_a[use], s=13, color=color, label=str(stage))
        ax.set_ylabel(f"{name}\nflame Δa*")
        ax.grid(alpha=.2)
    axes[0].legend(title="test2-like stage")
    axes[-1].set_xlabel("clip time (s)")
    fig.tight_layout()
    fig.savefig(args.output / "rh30_optical_candidates.png", dpi=180)
    plt.close(fig)
    payload = {"distance_limit": distance_limit, "runs": results}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
