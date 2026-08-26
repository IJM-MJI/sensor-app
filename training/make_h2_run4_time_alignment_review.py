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
    times = (50, 55, 56, 60, 65, 84, 90)
    panels = [panel(image_at(args.video_root, "run4", seconds),
                    f"run4 | t={seconds}s",
                    "candidate boundary review", width=360, height=390)
              for seconds in times]
    blank = np.full_like(panels[0], 245)
    rows = [np.hstack(panels[:4]), np.hstack(panels[4:] + [blank])]
    cv2.imwrite(str(args.output), np.vstack(rows))
    print(args.output)


if __name__ == "__main__":
    main()
