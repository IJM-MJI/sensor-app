"""Create the human-review sheet required for simultaneous H2 quantitation."""

from __future__ import annotations

import csv
from pathlib import Path

from train_models import simultaneous_clips


OUTPUT = Path(__file__).with_name("simultaneous_stage_anchors.csv")


def main():
    fields = (
        "group", "video", "nominal_rh_metadata", "reaction_start_s",
        "h2_0pct_s", "h2_1pct_s", "h2_2pct_s", "h2_3pct_s", "h2_4pct_s",
        "reaction_end_s", "review_status", "review_note",
    )
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for clip in simultaneous_clips():
            if clip.rh > 80:
                continue  # simultaneous RH90 is saturated/out of quantitative scope
            # Reaction boundaries are known. Interior concentration anchors are
            # deliberately blank rather than inferred from elapsed time.
            writer.writerow({
                "group": clip.group,
                "video": clip.name,
                "nominal_rh_metadata": int(clip.rh),
                "reaction_start_s": f"{clip.reaction_start:g}",
                "h2_0pct_s": f"{clip.reaction_start:g}",
                "h2_1pct_s": "",
                "h2_2pct_s": "",
                "h2_3pct_s": "",
                "h2_4pct_s": f"{clip.reaction_end:g}",
                "reaction_end_s": f"{clip.reaction_end:g}",
                "review_status": "needs_1_2_3pct_times",
                "review_note": "",
            })


if __name__ == "__main__":
    main()
