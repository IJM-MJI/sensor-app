"""Create the optical review strip for the proposed run4 H2 boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from make_h2_environment_review import image_at
from make_h2_family_boundary_review import panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    times = (0, 10, 20, 50, 55, 56, 60, 65, 84, 90, 178, 180)
    def note(seconds):
        if seconds == 0:
            return "initial reference"
        if seconds >= 178:
            return "recovery-end reference"
        if seconds <= 20:
            return "early reaction"
        return "candidate boundary review"
    panels = [panel(image_at(args.video_root, "run4", seconds),
                    f"run4 | t={seconds}s",
                    note(seconds), width=360, height=390)
              for seconds in times]
    blank = np.full_like(panels[0], 245)
    rows = [np.hstack(panels[:6]),
            np.hstack(panels[6:] + [blank] * (6 - len(panels[6:])))]
    cv2.imwrite(str(args.output), np.vstack(rows))
    print(args.output)


if __name__ == "__main__":
    main()
