"""Extract timeline-free H2 features from user-cropped RH20 recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

from train_models import Clip, apply_shared_baselines, sample_clip, write_csv


BASE_VIDEOS = {
    "1_90_RH20_2_x2_cropped.mp4": "rh20-crop-run-2",
    "1_90_RH20_3_x2_cropped.mp4": "rh20-crop-run-3",
    "1_90_RH20_4_cropped.mp4": "rh20-crop-run-4",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=2.0)
    parser.add_argument("--run5-source", choices=("normal", "x2"), default="normal",
                        help="choose one run-5 export; both map to the same source group")
    parser.add_argument("--output", type=Path,
                        default=Path("training/cache/v7-verified-orientation-recovery-tail/"
                                     "rh20_cropped_unlabeled_v1.csv"))
    args = parser.parse_args()
    videos = dict(BASE_VIDEOS)
    videos["1_90_RH20_cropped.mp4" if args.run5_source == "normal"
           else "1_90_RH20_5_x2_cropped.mp4"] = "rh20-crop-run-5"
    rows = []
    for name, group in videos.items():
        clip = Clip(name=name, kind="h2_only", group=group, centered_crop=True,
                    cache_tag="timeline-free-rh20-crop-v1")
        rows.extend(sample_clip(args.video_root, clip, args.sample_hz))
    rows = apply_shared_baselines(rows)
    # Explicitly preserve that no concentration/reaction timeline supervised
    # extraction. Downstream code may add optical-equivalent pseudo-labels only.
    for row in rows:
        row["kind"] = "h2_candidate_unlabeled"
        row["h2_value"] = None
        row["state"] = None
    write_csv(args.output, rows)
    print(f"wrote {len(rows)} timeline-free rows -> {args.output}")


if __name__ == "__main__":
    main()
