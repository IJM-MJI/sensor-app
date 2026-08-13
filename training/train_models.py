"""Train calibration-delta colour models from the dual-sensor recordings.

The browser application uses the same feature extractor: a circular chamber ROI,
LAB colour, background correction, and per-chip dry calibration deltas.  Video
frames are never copied into this repository.  Only compact CSV features, model
coefficients and evaluation reports are generated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "flame_L", "flame_a", "flame_b",
    "drop_L", "drop_a", "drop_b",
    "top_L", "top_a", "top_b",
]
MODEL_FEATURES = [
    "flame_a", "flame_b", "drop_a", "drop_b", "top_a", "top_b",
]
H2_FEATURES = ["flame_a", "flame_b", "flame_drop_a", "flame_drop_b"]
RH_FEATURES = ["drop_a", "drop_b"]
H2_QUANT_FEATURES = ["flame_L", "flame_a", "flame_b"]
RH_QUANT_FEATURES = ["drop_L", "drop_a", "drop_b"]
CACHE_VERSION = "v7-verified-orientation-recovery-tail"
TIMED_LEGACY_SOURCES = {
    # Older full simultaneous recordings require separate fixed ROIs and
    # orientation maps. Their derived RH clips are visually verified and used
    # by the state pipeline, but the full-video legacy features are excluded.
    "1_90_H2_only_13.mp4", "1_90_H2_only_4.mp4",
    "1_H2_only_test.MOV", "1_H2_only_test_2.MOV", "1_H2_only_test_3.MOV",
    "1_90_H2O_only.MOV",
}


@dataclass(frozen=True)
class Clip:
    name: str
    kind: str
    group: str
    reaction_start: float = 0.0
    reaction_end: float | None = None
    duration_hint: float | None = None
    rh: float | None = None
    segments: tuple[tuple[float, float, float], ...] = ()
    h2_segments: tuple[tuple[float, float, float], ...] = ()
    orientation_quarters: int = 0
    fixed_circle: tuple[int, int, int] | None = None
    minimum_sample_hz: float = 0.0
    cache_tag: str = ""
    analysis_end: float | None = None


def simultaneous_clips() -> list[Clip]:
    # Reaction/recovery boundaries reconstructed from the experiment timelines.
    boundaries = {
        "2": {
            20: (60, 85), 30: (123, 175), 40: (60.5, 89.5), 50: (123, 183),
            # RH60 clip includes about 3 s of pre-reaction footage: the source
            # timestamp was corrected from 12:04 to 12:07.
            60: (3, 127, 191), 70: (63.5, 101), 80: (63, 96.5), 90: (74, 104),
        },
        "3": {
            20: (60, 97.5), 30: (55, 95), 40: (60, 135), 50: (75, 135),
            60: (70, 147), 70: (83, 138), 80: (85, 165), 90: (120, 172),
        },
        "4": {
            20: (60, 90), 30: (120, 180), 40: (90, 148), 50: (122, 152),
            60: (75, 104.5), 70: (125, 187), 80: (68, 98.5), 90: (127, 188),
        },
        "5": {
            20: (54, 107.5), 30: (51, 113.5), 40: (70.5, 127), 50: (84.5, 169),
            60: (74, 115.5), 70: (106, 158), 80: (74, 126.5), 90: (105.5, 137),
        },
    }
    x2 = {
        "2": {20, 40, 70, 80, 90},
        "3": {20, 30},
        "4": {20, 60, 80},
        "5": set(range(20, 100, 10)),
    }
    clips: list[Clip] = []
    for run, rows in boundaries.items():
        for rh, timing in rows.items():
            if len(timing) == 3:
                reaction_start, reaction, duration = timing
            else:
                reaction_start, (reaction, duration) = 0.0, timing
            suffix = "_x2" if rh in x2[run] else ""
            clips.append(Clip(
                name=f"1_90_RH{rh}_{run}{suffix}.mp4",
                kind="simultaneous",
                # Every RH clip from one run comes from the same original
                # recording. Hold the complete run out together to prevent
                # recording-style leakage across train and validation.
                group=f"sim-run-{run}",
                reaction_start=reaction_start,
                reaction_end=reaction,
                duration_hint=duration,
                rh=float(rh),
            ))
    return clips


def manifest() -> list[Clip]:
    rh_fast = (
        (3, 6, 20), (6, 9, 30), (9, 15, 40), (15, 25, 50),
        (25, 35, 60), (35, 45, 70), (45, 72, 80), (72, 140, 90),
    )
    rh_long_a = (
        (0, 14, 20), (14, 25, 30), (25, 45, 40), (45, 90, 50),
        (90, 120, 60), (120, 180, 70),
    )
    rh_long_b = ((0, 9, 70), (9, 87, 80))
    rh_daylight_recovery = (
        (0, 8, 90), (8, 11, 80), (11, 13, 70), (13, 15, 60),
        (15, 20, 50), (20, 23, 40), (23, 30, 30), (30, 39, 20),
    )
    rh_response_6 = (
        (0, 7, 20), (7, 10, 30), (10, 13, 40), (13, 14, 50),
        (14, 16, 60), (16, 18, 70), (18, 20, 80), (20, 32, 90),
    )
    rh_response_3 = (
        (0, 2, 20), (2, 3, 30), (3, 5, 40), (5, 7, 50),
        (7, 11, 60), (11, 25, 70), (25, 28, 80), (28, 38, 90),
    )
    h2_test = ((0, 15, 1), (15, 25, 2), (25, 30, 3), (30, 201, 4))
    h2_test_2 = ((0, 4, 0), (4, 13, 1), (13, 21, 2), (21, 30, 3), (30, 150, 4))
    h2_test_3 = ((0, 3, 0), (3, 10, 1), (10, 20, 2), (20, 28, 3), (28, 152, 4))
    h2_indoor_4 = ((0, 5, 0), (5, 13, 1), (13, 30, 2), (30, 109, 3), (109, 122, 4), (122, 266, 0))
    h2_daylight_5 = ((0, 5, 0), (5, 8, 1), (8, 13, 2), (13, 21, 3), (21, 130, 4), (130, 272, 0))
    clips = [
        Clip("1_90_H2_only_test.mp4", "h2_only", "h2-test-indoor", h2_segments=h2_test,
             minimum_sample_hz=2.0, cache_tag="quant-2hz-v1"),
        Clip("1_90_H2_only_test_2.mp4", "h2_only", "h2-test-2", h2_segments=h2_test_2,
             minimum_sample_hz=2.0, cache_tag="quant-2hz-v1"),
        Clip("1_90_H2_only_test_3.MOV", "h2_only", "h2-test-3", h2_segments=h2_test_3,
             minimum_sample_hz=2.0, cache_tag="quant-2hz-v1"),
        Clip("1_90_H2_only_4.mp4", "h2_only", "h2-indoor-4", h2_segments=h2_indoor_4,
             minimum_sample_hz=2.0, cache_tag="quant-2hz-v1"),
        Clip("1_90_H2_only_5.mp4", "h2_only", "h2-daylight-5", h2_segments=h2_daylight_5,
             minimum_sample_hz=2.0, cache_tag="quant-2hz-v1"),
        Clip("1_90_H2O_only_2_extract.mp4", "rh_only", "rh-indoor-fast", segments=rh_fast,
             minimum_sample_hz=2.0, cache_tag="quant-2hz-v1"),
        Clip("1_90_H2O_only.MOV", "rh_only", "rh-daylight-recovery", segments=rh_daylight_recovery,
             minimum_sample_hz=2.0, cache_tag="quant-2hz-v1"),
        Clip("1_90_H2O_only_extract_3min.mp4", "rh_only", "rh-indoor-long", segments=rh_long_a,
             minimum_sample_hz=2.0, cache_tag="quant-2hz-v1"),
        Clip("1_90_H2O_only_extract_extra.mp4", "rh_only", "rh-indoor-long", segments=rh_long_b,
             minimum_sample_hz=2.0, cache_tag="quant-2hz-v1"),
        Clip("1_90_H2O_only_6(response).mp4", "rh_only", "rh-response-6",
             segments=rh_response_6, orientation_quarters=1, fixed_circle=(164, 274, 61),
             minimum_sample_hz=4.0, cache_tag="timeline-v2", analysis_end=32.0),
        Clip("1_90_H2O_only_3(response).mp4", "rh_only", "rh-response-3",
             segments=rh_response_3, orientation_quarters=3, fixed_circle=(135, 190, 72),
             minimum_sample_hz=4.0, cache_tag="timeline-v2", analysis_end=38.0),
    ]
    return clips + simultaneous_clips()


def video_info(path: Path) -> tuple[float, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frames / max(fps, 1), fps, width, height


def frame_at(cap: cv2.VideoCapture, seconds: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(seconds, 0) * 1000)
    ok, frame = cap.read()
    return frame if ok else None


def resize_for_app(frame: np.ndarray, max_side: int = 480) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def circle_score(lab: np.ndarray, gray: np.ndarray, x: int, y: int, r: int) -> float:
    h, w = gray.shape
    ir = max(2, round(r * 0.5))
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= ir ** 2
    a = lab[:, :, 1].astype(np.float32) - 128
    b = lab[:, :, 2].astype(np.float32) - 128
    chroma = np.sqrt(a * a + b * b)
    vals = chroma[mask]
    if not vals.size:
        return -1
    # At least the flame mark remains yellow/green even when the droplet turns
    # blue/purple. Warm chroma rejects blue tubes, labels and metal fittings.
    warm = (chroma > 15) & (lab[:, :, 2] > 133)
    ys, xs = np.where(mask & warm)
    coverage = float(np.mean(warm[mask]))
    if len(xs) < 20:
        return coverage
    # The actual window contains coloured flame/drop marks spread vertically.
    # Small fittings may be colourful too, but their colour is spatially compact.
    vertical_span = float((np.percentile(ys, 90) - np.percentile(ys, 10)) / max(r, 1))
    horizontal_span = float((np.percentile(xs, 90) - np.percentile(xs, 10)) / max(r, 1))
    layout = coverage + 0.05 * min(vertical_span, 1.8) + 0.02 * min(horizontal_span, 1.8)
    # In all controlled recordings the chamber is in the upper 65% of frame.
    # This prevents brightly coloured hoses/labels near the bottom from winning.
    if y > h * 0.50:
        return 0.0
    position = 1.0
    size_prior = math.sqrt(max(r, 1) / max(min(h, w), 1))
    return layout * position * size_prior


def circle_candidates(frame: np.ndarray) -> list[tuple[int, int, int, float]]:
    small = resize_for_app(frame)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    side = min(gray.shape)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=round(side * 0.12),
        param1=80, param2=35, minRadius=round(side * 0.07), maxRadius=round(side * 0.42),
    )
    if circles is None:
        return []
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    candidates = [tuple(map(int, np.round(c))) for c in circles[0]]
    return [(x, y, r, circle_score(lab, gray, x, y, r)) for x, y, r in candidates]


def detect_circle(frame: np.ndarray) -> tuple[int, int, int]:
    candidates = circle_candidates(frame)
    if not candidates:
        raise RuntimeError("No circular chamber found")
    return tuple(max(candidates, key=lambda c: c[3])[:3])


def lock_circle(cap: cv2.VideoCapture, duration: float) -> tuple[int, int, int]:
    """Lock one chamber circle from several probe frames.

    Candidates are clustered across time. A real chamber stays in the same place;
    one-frame Hough false positives around fittings do not.
    """
    probe_times = sorted({min(duration - 0.2, t) for t in (1, 3, 6, 10, 20, duration * .5, duration * .85) if t < duration})
    clusters: list[list[tuple[int, int, int, float]]] = []
    for t in probe_times:
        frame = frame_at(cap, t)
        if frame is None:
            continue
        for candidate in circle_candidates(frame):
            x, y, r, _ = candidate
            placed = False
            for cluster in clusters:
                cx, cy, cr = np.median(np.asarray([c[:3] for c in cluster]), axis=0)
                if math.hypot(x - cx, y - cy) <= max(14, .35 * cr) and abs(r - cr) <= max(12, .50 * cr):
                    cluster.append(candidate)
                    placed = True
                    break
            if not placed:
                clusters.append([candidate])
    if not clusters:
        raise RuntimeError("No circular chamber found in probe frames")
    # A real window persists and contains the coloured sensor pattern. Multiplying
    # the two signals prevents a persistent grey fitting from winning.
    cluster = max(clusters, key=lambda cs: math.sqrt(len(cs)) * max(0.001, float(np.median([c[3] for c in cs]))))
    x, y, r = np.median(np.asarray([c[:3] for c in cluster]), axis=0)
    return int(round(x)), int(round(y)), int(round(r))


def circle_track(cap: cv2.VideoCapture, duration: float, interval: float = 8.0) -> list[tuple[float, tuple[int, int, int]]]:
    """Detect periodically, then reuse the nearest circle between probes."""
    track: list[tuple[float, tuple[int, int, int]]] = []
    for t in np.arange(0.5, max(0.6, duration), interval):
        frame = frame_at(cap, float(t))
        if frame is None:
            continue
        try:
            track.append((float(t), detect_circle(frame)))
        except RuntimeError:
            pass
    if not track:
        raise RuntimeError("No circular chamber found in tracking probes")
    return track


def split_lower_by_y(ys: np.ndarray, iterations: int = 12) -> np.ndarray:
    lo, hi = float(ys.min()), float(ys.max())
    c0, c1 = lo, hi
    for _ in range(iterations):
        lower = np.abs(ys - c1) < np.abs(ys - c0)
        if lower.all() or (~lower).all():
            break
        c0, c1 = ys[~lower].mean(), ys[lower].mean()
    return np.abs(ys - c1) < np.abs(ys - c0)


def patch_balance_lab(lab: np.ndarray, chamber_mask: np.ndarray) -> np.ndarray:
    """Neutralise a/b using the white/gray pieces printed beside the shapes.

    The original research pipeline explicitly detects the rectangular pieces.
    For the phone-resolution path, the neutral pixels inside the chamber are a
    more stable equivalent: coloured flame/drop pixels are rejected by chroma,
    while the white/gray pieces and substrate vote for the illumination offset.
    """
    out = lab.astype(np.float64)
    chroma = np.sqrt((out[:, :, 1] - 128) ** 2 + (out[:, :, 2] - 128) ** 2)
    neutral = chamber_mask & (chroma < 8) & (out[:, :, 0] > 140)
    if int(neutral.sum()) < 50:
        return out
    da = float(np.median(out[:, :, 1][neutral]) - 128)
    db = float(np.median(out[:, :, 2][neutral]) - 128)
    white_l = float(np.percentile(out[:, :, 0][neutral], 90))
    scale = float(np.clip(210 / max(white_l, 1), .8, 1.5))
    out[:, :, 0] = np.clip(out[:, :, 0] * scale, 0, 255)
    out[:, :, 1] = np.clip(out[:, :, 1] - da, 0, 255)
    out[:, :, 2] = np.clip(out[:, :, 2] - db, 0, 255)
    return out


def infer_orientation_quarters(
    frame: np.ndarray, circle: tuple[int, int, int]
) -> tuple[int | None, float]:
    """Infer the quarter-turn that places the flame above the droplet.

    The flame is the largest warm-chromatic printed component. White/gray
    reference patches are rejected by chroma. A confidence score close to zero
    means that the two leading components are ambiguous or the flame lies near
    a diagonal; those frames must not supervise state learning.
    """
    small = resize_for_app(frame)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    x, y, r = circle
    yy, xx = np.ogrid[:lab.shape[0], :lab.shape[1]]
    chamber = (xx - x) ** 2 + (yy - y) ** 2 <= (r * .82) ** 2
    a = lab[:, :, 1].astype(np.float32) - 128
    b = lab[:, :, 2].astype(np.float32) - 128
    chroma = np.hypot(a, b)
    warm = (chamber & (chroma >= 11) & (b >= 3)).astype(np.uint8)
    warm = cv2.morphologyEx(warm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(warm)
    candidates = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        cx, cy = centroids[index]
        distance = math.hypot(cx - x, cy - y)
        if area < max(5, r * r * .002) or distance < r * .08 or distance > r * .82:
            continue
        mean_chroma = float(np.mean(chroma[labels == index]))
        candidates.append((area * mean_chroma, float(cx), float(cy)))
    if not candidates:
        return None, 0.0
    candidates.sort(reverse=True)
    score, cx, cy = candidates[0]
    separation = 1.0 if len(candidates) == 1 else max(0.0, 1 - candidates[1][0] / max(score, 1e-6))
    dx, dy = cx - x, cy - y
    axial = abs(abs(dx) - abs(dy)) / max(abs(dx) + abs(dy), 1e-6)
    confidence = float(np.clip(.55 * separation + .45 * axial, 0, 1))
    if abs(dx) >= abs(dy):
        quarters = 1 if dx > 0 else 3
    else:
        quarters = 0 if dy < 0 else 2
    return quarters, confidence


def lock_orientation(
    cap: cv2.VideoCapture,
    duration: float,
    circle_at,
) -> tuple[int, float]:
    """Lock one semantic orientation per recording from weighted probe votes."""
    votes = np.zeros(4, dtype=float)
    for t in np.linspace(min(2.0, duration * .05), max(2.1, duration - 2), 15):
        frame = frame_at(cap, float(t))
        if frame is None:
            continue
        circle = circle_at(float(t))
        orientation, confidence = infer_orientation_quarters(frame, circle)
        if orientation is not None:
            votes[orientation] += max(.05, confidence) ** 2
    if votes.sum() <= 0:
        raise RuntimeError("Could not infer sample orientation")
    chosen = int(np.argmax(votes))
    return chosen, float(votes[chosen] / votes.sum())


def masked_shape_mean(lab: np.ndarray, zone: np.ndarray, background: np.ndarray) -> np.ndarray:
    pixels = lab[zone]
    if len(pixels) < 20:
        return background
    distance = np.sqrt(
        (.35 * (pixels[:, 0] - background[0])) ** 2
        + (pixels[:, 1] - background[1]) ** 2
        + (pixels[:, 2] - background[2]) ** 2
    )
    cutoff = max(6.0, float(np.percentile(distance, 65)))
    shape = pixels[distance >= cutoff]
    if len(shape) < 12:
        shape = pixels[np.argsort(distance)[-max(12, len(pixels) // 5):]]
    return np.mean(shape, axis=0)


def extract_features(
    frame: np.ndarray,
    circle: tuple[int, int, int],
    orientation_quarters: int = 0,
) -> dict[str, float]:
    frame = resize_for_app(frame)
    raw_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    x, y, r = circle
    inner = round(r * 0.90)
    yy, xx = np.ogrid[:raw_lab.shape[0], :raw_lab.shape[1]]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= inner ** 2
    lab = patch_balance_lab(raw_lab, mask)
    px = lab[mask].astype(np.float64)
    if len(px) < 100:
        raise RuntimeError("Too few pixels in chamber")
    chroma = np.sqrt((px[:, 1] - 128) ** 2 + (px[:, 2] - 128) ** 2)
    order = np.argsort(chroma)
    top = px[order[math.floor(len(px) * 0.97):]].mean(axis=0)
    bg = px[order[:math.floor(len(px) * 0.50)]].mean(axis=0)
    # Fixed shape zones prevent the gray/white calibration pieces from becoming
    # sensor features. Distance from the local substrate also retains the gray
    # dry droplet, which a chroma-only mask would discard.
    nx = (xx - x) / max(r, 1)
    ny = (yy - y) / max(r, 1)
    # Normalize in-plane sample rotation before applying the semantic shape
    # zones. A +1 quarter-turn rotates the observed right-side flame to the top.
    for _ in range(orientation_quarters % 4):
        nx, ny = ny, -nx
    central_x = (nx >= -.55) & (nx <= .35)
    flame_zone = mask & central_x & (ny >= -.62) & (ny <= -.02)
    drop_zone = mask & central_x & (ny >= .02) & (ny <= .68)
    flame = masked_shape_mean(lab, flame_zone, bg)
    drop = masked_shape_mean(lab, drop_zone, bg)
    return {
        "flame_L": flame[0], "flame_a": flame[1], "flame_b": flame[2],
        "drop_L": drop[0], "drop_a": drop[1], "drop_b": drop[2],
        "bg_L": bg[0], "bg_a": bg[1], "bg_b": bg[2],
        "top_L": top[0], "top_a": top[1], "top_b": top[2],
    }


def corrected(feat: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for region in ("flame", "drop", "top"):
        for channel in ("L", "a", "b"):
            out[f"{region}_{channel}"] = feat[f"{region}_{channel}"] - feat[f"bg_{channel}"]
    return out


def stable_segment_value(segments: Iterable[tuple[float, float, float]], t: float) -> float | None:
    for start, end, value in segments:
        margin = min(1.5, max(0.25, (end - start) * 0.15))
        if start + margin <= t <= end - margin:
            return float(value)
    return None


def recovery_tail_start(start: float, end: float) -> float:
    """Use only the final stable recovery window, capped at twelve seconds."""
    return max(start + .70 * (end - start), end - 12.0)


def labels_for(clip: Clip, t: float, duration: float) -> tuple[int | None, int | None, str | None, float | None, float | None]:
    h2_present: int | None = None
    rh_high: int | None = None
    state: str | None = None
    h2_value = stable_segment_value(clip.h2_segments, t)
    rh_value = stable_segment_value(clip.segments, t)
    if clip.kind == "h2_only":
        rh_high = None
        if h2_value is not None:
            final_recovery = (
                bool(clip.h2_segments) and clip.h2_segments[-1][2] == 0
                and any(value > 0 for _, _, value in clip.h2_segments[:-1])
                and t >= clip.h2_segments[-1][0]
            )
            if final_recovery:
                start, end, _ = clip.h2_segments[-1]
                if t >= recovery_tail_start(start, end):
                    h2_present, state = 0, "baseline_recovery"
            else:
                h2_present = 1 if h2_value >= 3 else (0 if h2_value < 0.5 else None)
                state = "h2_only" if h2_value > 0 else "none"
    elif clip.kind == "rh_only":
        h2_present = 0
        if rh_value is not None:
            rh_high = 1 if rh_value >= 70 else (0 if rh_value <= 30 else None)
            state = "rh_only" if rh_value > 20 else "none"
    elif clip.kind == "simultaneous" and clip.reaction_end is not None:
        # Ramp step times are unknown.  Use only the late reaction portion where H2
        # is expected to be high.  The filename RH is a flow-controller setpoint,
        # not an optical RH label: simultaneous flow is lower and setpoint 90 may
        # look like RH-only 60--80.  Therefore simultaneous clips NEVER supervise
        # the RH model.  Recovery is RH20 + H2 0%, the optical baseline condition.
        reaction_span = clip.reaction_end - clip.reaction_start
        if t >= clip.reaction_start + reaction_span * 0.78 and t <= clip.reaction_end - 2:
            if clip.rh is not None and clip.rh >= 90:
                # At nominal simultaneous RH90 the droplet response is saturated
                # enough to obscure the H2 flame change. Keep it as an explicit
                # out-of-scope condition, never as H2/state supervision.
                h2_present, rh_high, state = None, None, "simultaneous_rh90_saturated"
            else:
                h2_present = 1
                rh_high = None
                state = "h2_only_condition" if clip.rh == 20 else "simultaneous_condition"
        elif t >= recovery_tail_start(clip.reaction_end, duration) and t <= duration - 1:
            h2_present, rh_high, state = 0, None, "baseline_recovery"
    return h2_present, rh_high, state, h2_value, rh_value


def sample_clip(root: Path, clip: Clip, sample_hz: float) -> list[dict[str, object]]:
    path = root / clip.name
    if not path.exists():
        raise FileNotFoundError(path)
    duration, fps, width, height = video_info(path)
    if clip.duration_hint and abs(duration - clip.duration_hint) > 3:
        raise RuntimeError(f"{clip.name}: duration {duration:.2f}s does not match {clip.duration_hint:.2f}s")
    cap = cv2.VideoCapture(str(path))
    track = ([(0.0, clip.fixed_circle), (duration, clip.fixed_circle)]
             if clip.fixed_circle is not None else circle_track(cap, duration))
    # All user-timed 90-degree recordings were visually verified with the flame
    # above the droplet. Semantic orientation must never be inferred from color
    # area because a reacted droplet can be larger/more chromatic than the flame.
    orientation_lock, orientation_lock_confidence = clip.orientation_quarters, 1.0
    raw: list[tuple[float, dict[str, float], int, float, tuple[int, int, int]]] = []
    detected_circles: list[tuple[int, int, int]] = []
    # Seeking repeatedly in phone MP4s can decode from the previous keyframe (or
    # even from the start) for every sample.  A single sequential pass is much
    # faster and has deterministic frame timing.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    effective_sample_hz = max(sample_hz, clip.minimum_sample_hz)
    sample_every = max(1, round(fps / effective_sample_hz))
    frame_index = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if frame_index % sample_every == 0:
            ok, frame = cap.retrieve()
            if not ok:
                frame_index += 1
                continue
            t = frame_index / fps
            if clip.analysis_end is not None and t > clip.analysis_end:
                break
            try:
                circle = min(track, key=lambda item: abs(item[0] - t))[1]
                raw.append((float(t), extract_features(frame, circle, orientation_lock), orientation_lock, 1.0, circle))
                detected_circles.append(circle)
            except RuntimeError:
                pass
        frame_index += 1
    cap.release()
    if not raw:
        raise RuntimeError(f"No features extracted from {clip.name}")
    rows: list[dict[str, object]] = []
    for t, feat, orientation, orientation_confidence, circle in raw:
        corr = corrected(feat)
        hp, rhp, state, h2v, rhv = labels_for(clip, t, duration)
        row: dict[str, object] = {
            "video": clip.name, "group": clip.group, "kind": clip.kind,
            "time": t, "duration": duration, "width": width, "height": height,
            "h2_present": hp, "rh_high": rhp, "state": state,
            "h2_value": h2v, "rh_value": rhv, "rh_setpoint": clip.rh,
            "circle_x": circle[0], "circle_y": circle[1], "circle_r": circle[2],
            "orientation_quarters": orientation,
            "orientation_confidence": orientation_confidence,
        }
        for name in FEATURE_NAMES:
            row[name] = corr[name]
        rows.append(row)
    median_circle = tuple(int(v) for v in np.median(np.asarray(detected_circles), axis=0))
    print(
        f"{clip.name}: {len(rows)} frames, median_circle={median_circle}, "
        f"orientation={orientation_lock} ({orientation_lock_confidence:.2f}), duration={duration:.1f}s"
    )
    return rows


def apply_shared_baselines(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Apply calibration deltas after all split clips have been combined."""
    by_video = {str(r["video"]): [] for r in rows}
    for row in rows:
        by_video[str(row["video"])].append(row)

    long_reference = by_video.get("1_90_H2O_only_extract_3min.mp4", [])
    indoor_h2_reference = by_video.get("1_90_H2_only_test_2.mp4", [])
    baselines: dict[str, dict[str, float]] = {}
    for video, video_rows in by_video.items():
        kind = str(video_rows[0]["kind"])
        if video == "1_90_H2O_only_extract_extra.mp4":
            candidates = [r for r in long_reference if r["rh_value"] == 20]
        elif video == "1_90_H2_only_test.mp4":
            # This recording starts at H2 1%; use the matching indoor 0% clip.
            candidates = [r for r in indoor_h2_reference if float(r["time"]) <= 2.5]
        elif kind == "h2_only":
            candidates = [r for r in video_rows if float(r["time"]) <= 2.5]
        elif kind == "rh_only":
            candidates = [r for r in video_rows if r["rh_value"] == 20]
        else:
            candidates = [r for r in video_rows if r["state"] == "baseline_recovery"]
        if len(candidates) < 1:
            raise RuntimeError(f"No trustworthy baseline for {video}")
        baselines[video] = {
            name: float(np.median([float(r[name]) for r in candidates]))
            for name in FEATURE_NAMES
        }

    normalized: list[dict[str, object]] = []
    for row in rows:
        out = dict(row)
        baseline = baselines[str(row["video"])]
        for name in FEATURE_NAMES:
            out[name] = float(row[name]) - baseline[name]
        normalized.append(out)
    return normalized


def relabel_rows(rows: list[dict[str, object]], clip: Clip) -> list[dict[str, object]]:
    for row in rows:
        hp, rhp, state, h2v, rhv = labels_for(clip, float(row["time"]), float(row["duration"]))
        row.update({"video": clip.name, "group": clip.group, "kind": clip.kind,
                    "h2_present": hp, "rh_high": rhp, "state": state,
                    "h2_value": h2v, "rh_value": rhv, "rh_setpoint": clip.rh})
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("h2_present", "rh_high"):
            row[key] = None if row[key] in ("", "None") else int(float(row[key]))
        for key in ("h2_value", "rh_value", "rh_setpoint"):
            row[key] = None if row[key] in ("", "None") else float(row[key])
    return rows


def read_legacy_continuous(path: Path) -> list[dict[str, object]]:
    """Load v4-reextracted continuous labels and apply per-video calibration."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    by_video: dict[str, list[dict[str, object]]] = {}
    for source in raw:
        # A filename/condition is not a concentration label. Only recordings
        # with a user-supplied timeline can supervise training or validation.
        if str(source.get("source_video")) not in TIMED_LEGACY_SOURCES:
            continue
        row: dict[str, object] = dict(source)
        for name in FEATURE_NAMES:
            row[name] = float(source[name])
        row["h2_value"] = float(source["h2_value"])
        row["rh_value"] = float(source["rh_value"])
        row["time"] = float(source["time"])
        condition = str(source["condition"])
        row["kind"] = "h2_only" if condition == "H2_only" else "rh_only" if condition == "H2O_only" else "simultaneous"
        row["group"] = f"legacy-{source['source_video']}"
        if condition == "H2O_only":
            row["h2_present"] = 0
            row["rh_high"] = 1 if row["rh_value"] >= 70 else (0 if row["rh_value"] <= 30 else None)
        else:
            row["h2_present"] = 1 if row["h2_value"] >= 3 else (0 if row["h2_value"] < .5 else None)
            row["rh_high"] = None
        by_video.setdefault(str(row["video"]), []).append(row)

    normalized: list[dict[str, object]] = []
    for video, video_rows in by_video.items():
        kind = str(video_rows[0]["kind"])
        if kind == "rh_only":
            baseline_rows = [r for r in video_rows if float(r["rh_value"]) <= 25]
        else:
            baseline_rows = [r for r in video_rows if float(r["h2_value"]) <= .25]
        if not baseline_rows:
            print(f"legacy {video}: skipped (no baseline frames)")
            continue
        baseline = {name: float(np.median([float(r[name]) for r in baseline_rows])) for name in FEATURE_NAMES}
        for row in video_rows:
            out = dict(row)
            for name in FEATURE_NAMES:
                out[name] = float(row[name]) - baseline[name]
            normalized.append(out)
    return normalized


def merge_training_rows(new_rows: list[dict[str, object]], legacy_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Prefer dense legacy labels where the same source video occurs in both."""
    legacy_videos = {str(r["video"]) for r in legacy_rows}
    return [r for r in new_rows if str(r["video"]) not in legacy_videos] + legacy_rows


def feature_value(row: dict[str, object], name: str) -> float:
    if name == "flame_drop_a":
        return float(row["flame_a"]) - float(row["drop_a"])
    if name == "flame_drop_b":
        return float(row["flame_b"]) - float(row["drop_b"])
    return float(row[name])


def model_matrix(rows: list[dict[str, object]], label: str, features: list[str] | None = None):
    features = features or MODEL_FEATURES
    use = [r for r in rows if r[label] is not None]
    x = np.asarray([[feature_value(r, n) for n in features] for r in use])
    y = np.asarray([int(r[label]) for r in use])
    groups = np.asarray([str(r["group"]) for r in use])
    return use, x, y, groups


def binary_model():
    return ExtraTreesClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )


def grouped_binary_oof(x: np.ndarray, y: np.ndarray, groups: np.ndarray):
    prob = np.full(len(y), np.nan)
    pred = np.full(len(y), -1)
    base = binary_model()
    for group in sorted(set(groups)):
        test = groups == group
        train = ~test
        if len(set(y[train])) < 2:
            continue
        fitted = clone(base).fit(x[train], y[train])
        prob[test] = fitted.predict_proba(x[test])[:, 1]
        pred[test] = (prob[test] >= 0.5).astype(int)
    valid = ~np.isnan(prob)
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y[valid], pred[valid])),
        "auc": float(roc_auc_score(y[valid], prob[valid])),
        "n_frames": int(valid.sum()),
        "n_videos": int(len(set(groups[valid]))),
    }
    return base.fit(x, y), metrics, y[valid], pred[valid]


def export_forest(model: ExtraTreesClassifier, name: str, features: list[str]) -> dict[str, object]:
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        leaf_probability = []
        for value in tree.value[:, 0, :]:
            total = float(value.sum())
            leaf_probability.append(float(value[1] / total) if total else 0.5)
        trees.append({
            "left": tree.children_left.astype(int).tolist(),
            "right": tree.children_right.astype(int).tolist(),
            "feature": tree.feature.astype(int).tolist(),
            "threshold": tree.threshold.tolist(),
            "probability": leaf_probability,
        })
    return {
        "name": name, "type": "extra_trees", "features": features,
        "threshold": 0.5, "trees": trees,
    }


def evaluate_regression(rows: list[dict[str, object]], label: str, features: list[str]) -> tuple[dict[str, object], dict[str, float]] | None:
    use = [r for r in rows if r[label] is not None and (label != "rh_value" or r["kind"] == "rh_only")]
    # The deployed classifiers are validated only for strong H2 response and
    # high humidity. Fit the displayed number over that same operating range;
    # lower concentrations remain explicitly unquantified in the app.
    lower = 3.0 if label == "h2_value" else 70.0
    use = [r for r in use if float(r[label]) >= lower]
    # Match the app's five-second observation: do not train or validate a
    # concentration before that nominal step has persisted for 4.5 seconds.
    stable_use: list[dict[str, object]] = []
    by_video: dict[str, list[dict[str, object]]] = {}
    for row in use:
        by_video.setdefault(str(row["video"]), []).append(row)
    for video_rows in by_video.values():
        video_rows.sort(key=lambda row: float(row["time"]))
        segment_start = float(video_rows[0]["time"])
        previous = float(video_rows[0][label])
        for row in video_rows:
            value, seconds = float(row[label]), float(row["time"])
            if value != previous:
                segment_start = seconds
            if seconds - segment_start >= 4.5:
                stable_use.append(row)
            previous = value
    use = stable_use
    groups = sorted({str(r["group"]) for r in use})
    if not use:
        return None
    x = np.asarray([[feature_value(r, n) for n in features] for r in use])
    y = np.asarray([float(r[label]) for r in use])
    g = np.asarray([str(r["group"]) for r in use])
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    pred = np.full(len(y), np.nan)
    if len(groups) >= 2:
        for group in groups:
            test = g == group
            train = ~test
            model.fit(x[train], y[train])
            pred[test] = model.predict(x[test])
    metrics = {
        "mae": float(mean_absolute_error(y[~np.isnan(pred)], pred[~np.isnan(pred)])) if np.any(~np.isnan(pred)) else float("nan"),
        "n_frames": int(len(y)), "n_videos": len(groups),
        "evaluation": "leave-one-video-out" if len(groups) >= 2 else "not independently validated",
    }
    model.fit(x, y)
    scaler: StandardScaler = model.named_steps["standardscaler"]
    ridge: Ridge = model.named_steps["ridge"]
    coef = ridge.coef_ / scaler.scale_
    intercept = float(ridge.intercept_ - np.dot(ridge.coef_, scaler.mean_ / scaler.scale_))
    exported = {
        "type": "ridge_regression", "features": features,
        "intercept": intercept, "coefficients": coef.tolist(),
        "calibrated_range": [3, 4] if label == "h2_value" else [70, 90],
        "validation": metrics["evaluation"], "mae": metrics["mae"],
        "status": "experimental",
    }
    return exported, metrics


def plot_confusion(output: Path, title: str, truth: np.ndarray, pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(truth, pred, normalize="true", cmap="Blues", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("training/output"))
    parser.add_argument("--cache", type=Path, default=Path(f"training/cache/{CACHE_VERSION}/features.csv"))
    parser.add_argument("--reuse-cache", action="store_true")
    nominal_legacy = Path(f"training/cache/{CACHE_VERSION}/legacy_continuous.csv")
    lag_corrected_legacy = Path(f"training/cache/{CACHE_VERSION}/legacy_continuous_lag_corrected.csv")
    parser.add_argument(
        "--legacy-cache",
        type=Path,
        default=lag_corrected_legacy if lag_corrected_legacy.exists() else nominal_legacy,
        help="continuous-label CSV; prefers the per-video H2 lag-corrected cache when available",
    )
    args = parser.parse_args()

    clips = manifest()
    if args.reuse_cache and args.cache.exists():
        rows = read_csv(args.cache)
    else:
        rows = []
        clip_cache = args.cache.parent / "clips"
        for clip in clips:
            cache_identity = clip.name + (f".{clip.cache_tag}" if clip.cache_tag else "")
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", cache_identity) + ".csv"
            cached = clip_cache / safe_name
            if cached.exists():
                clip_rows = relabel_rows(read_csv(cached), clip)
                print(f"{clip.name}: loaded {len(clip_rows)} cached frames")
            else:
                clip_rows = relabel_rows(sample_clip(args.video_root, clip, args.sample_hz), clip)
                write_csv(cached, clip_rows)
            rows.extend(clip_rows)
        rows = apply_shared_baselines(rows)
        write_csv(args.cache, rows)

    new_rows = rows
    legacy_rows = read_legacy_continuous(args.legacy_cache)
    if legacy_rows:
        rows = merge_training_rows(rows, legacy_rows)
        print(f"merged dataset: {len(rows)} rows ({len(legacy_rows)} continuous-label rows)")

    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "data": {
            "frames": len(rows), "videos": len({str(r['video']) for r in rows}),
            "sampling_hz": args.sample_hz,
            "training_policy": {
                "h2_presence": "new stable H2 0% and 3-4% intervals; continuous ramps excluded because sensor lag degraded held-out performance",
                "rh_high": "merged new and legacy H2O-only stable/extreme labels",
                "quantitative": "merged continuous and stable labels, evaluation only",
            },
            "excluded": [
                "1_90_RH20.mp4 ... 1_90_RH90.mp4 (duplicate 1x copies of run 5)",
                "1_70.mp4, 1_80.mp4, 1_80_3.mp4, 1_90_3.mp4 (uncertain timeline)",
                "1_90_H2O_only_2.mp4 (same labelled 3-140 s source as the extract; excluded to prevent duplication)",
            ],
        }
    }
    models: dict[str, object] = {"schema_version": 3, "feature_extractor": "app-v7-patch-balanced-shape-masks"}
    for label, key, title, features in (
        ("h2_present", "h2_presence", "H2 presence (video-wise OOF)", H2_FEATURES),
        ("rh_high", "rh_high", "High humidity (video-wise OOF)", RH_FEATURES),
    ):
        task_rows = new_rows if label == "h2_present" else rows
        _, x, y, groups = model_matrix(task_rows, label, features)
        fitted, metrics, truth, pred = grouped_binary_oof(x, y, groups)
        models[key] = export_forest(fitted, key, features)
        report[key] = metrics
        plot_confusion(args.output / f"{key}_confusion.png", title, truth, pred)

    experimental_quantitative: dict[str, object] = {}
    for label, key, features in (
        ("h2_value", "h2_concentration", H2_QUANT_FEATURES),
        ("rh_value", "rh_regression", RH_QUANT_FEATURES),
    ):
        result = evaluate_regression(rows, label, features)
        if result:
            exported, metrics = result
            models[key] = exported
            experimental_quantitative[key] = {
                **metrics,
                "status": "experimental_only",
                "reason": "Exported with an explicit error label; not validated as a measurement instrument.",
            }
    report["experimental_quantitative_models"] = experimental_quantitative

    (args.output / "models.json").write_text(json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8")
    Path("sensor-model.js").write_text(
        "// Generated by training/train_models.py; do not edit by hand.\n" +
        "window.SENSOR_MODEL=" + json.dumps(models, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
