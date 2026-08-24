"""Render RH endpoint RAW / selected-mask / colour-family audit sheets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from make_endpoint_mask_review import source_path
from make_middle_endpoint_review import rotate_and_crop
from rh_paired_pixel_hue_analysis import (
    NAMED_BINS, balanced_frame_and_masks, endpoint_rows,
)
from train_models import CACHE_VERSION, read_csv, resize_for_app


FAMILY_COLOURS = {
    "yellow": (0, 225, 255),
    "orange": (0, 125, 255),
    "scarlet": (25, 30, 230),
    "purple": (180, 55, 185),
    "green": (45, 205, 65),
    "other": (185, 185, 185),
}


def family_masks(lab: np.ndarray, selected: np.ndarray):
    a = lab[:, :, 1].astype(float) - 128.0
    b = lab[:, :, 2].astype(float) - 128.0
    hue = np.degrees(np.arctan2(b, a))
    masks = {
        "yellow": selected & (hue >= 55) & (hue < 115),
        "orange": selected & (hue >= 25) & (hue < 55),
        "scarlet": selected & (hue >= -5) & (hue < 25),
        "purple": selected & (hue >= -110) & (hue < -5),
        "green": selected & (hue >= 115) & (hue <= 180),
    }
    assigned = np.zeros(selected.shape, dtype=bool)
    for mask in masks.values():
        assigned |= mask
    masks["other"] = selected & ~assigned
    chroma = np.hypot(a, b)
    weights = np.maximum(chroma, 3.0)
    total = max(float(np.sum(weights[selected])), 1e-9)
    fractions = {name: float(np.sum(weights[mask]) / total)
                 for name, mask in masks.items()}
    return masks, fractions


def overlay_selected(frame, selected):
    result = frame.copy()
    colour = np.zeros_like(frame); colour[selected] = (255, 190, 0)
    result[selected] = cv2.addWeighted(frame[selected], .30, colour[selected], .70, 0)
    contours, _ = cv2.findContours(selected.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 0), 1)
    return result


def overlay_families(frame, masks):
    result = frame.copy()
    colour = np.zeros_like(frame)
    selected = np.zeros(frame.shape[:2], dtype=bool)
    for name, mask in masks.items():
        colour[mask] = FAMILY_COLOURS[name]
        selected |= mask
    result[selected] = cv2.addWeighted(frame[selected], .28, colour[selected], .72, 0)
    return result


def chamber_panel(image, circle, orientation, size=190):
    return rotate_and_crop(image, circle, orientation, size=size,
                           interpolation=cv2.INTER_NEAREST)


def drop_zoom(image, selected, size=190):
    ys, xs = np.where(selected)
    if len(xs) == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = max(x1 - x0 + 1, y1 - y0 + 1, 24)
    pad = max(12, int(round(side * .65)))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = side // 2 + pad
    padded = cv2.copyMakeBorder(image, half, half, half, half,
                                cv2.BORDER_CONSTANT, value=0)
    cx += half; cy += half
    crop = padded[cy-half:cy+half, cx-half:cx+half]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)


def put(image, text, origin, scale=.43, colour=(25, 25, 25), thickness=1):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thickness, cv2.LINE_AA)


def render_tile(frame, row, lab, selected, masks, fractions, stage, seconds):
    circle = tuple(int(float(row[name])) for name in ("circle_x", "circle_y", "circle_r"))
    orientation = int(float(row["orientation_quarters"]))
    selected_view = overlay_selected(frame, selected)
    family_view = overlay_families(frame, masks)
    panels = [
        chamber_panel(frame, circle, orientation),
        chamber_panel(selected_view, circle, orientation),
        chamber_panel(family_view, circle, orientation),
        drop_zoom(family_view, selected),
    ]
    panel = 190; caption = 72
    tile = np.full((panel + caption, panel * 4, 3), 247, dtype=np.uint8)
    for index, image in enumerate(panels):
        tile[:panel, index*panel:(index+1)*panel] = image
    for index, label in enumerate(("RAW", "SELECTED", "FAMILY", "FAMILY ZOOM")):
        put(tile, label, (index*panel + 7, 16), .38, (255, 255, 255), 1)
    put(tile, f"t={seconds:.1f}s   RH={stage:g}   selected={int(selected.sum())} px",
        (8, panel + 20), .46)
    ranked = sorted(((name, fractions[name]) for name in NAMED_BINS),
                    key=lambda pair: pair[1], reverse=True)
    first = "  ".join(f"{name[:3].upper()} {value*100:.1f}%" for name, value in ranked[:3])
    second = "  ".join(f"{name[:3].upper()} {value*100:.1f}%" for name, value in ranked[3:])
    put(tile, first, (8, panel + 42), .39)
    put(tile, second, (8, panel + 62), .39)
    return tile


def sheet_for(group, tiles, output):
    tile_h, tile_w = tiles[0].shape[:2]
    columns = 2; rows = int(np.ceil(len(tiles) / columns))
    header = 78
    sheet = np.full((header + rows * tile_h, columns * tile_w, 3), 238,
                    dtype=np.uint8)
    put(sheet, f"RH colour-family endpoint audit: {group}", (20, 28), .75,
        (20, 20, 20), 2)
    x = 20
    for name in (*NAMED_BINS, "other"):
        cv2.rectangle(sheet, (x, 43), (x + 18, 61), FAMILY_COLOURS[name], -1)
        put(sheet, name, (x + 23, 58), .38)
        x += 105
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        y0, x0 = header + row * tile_h, column * tile_w
        sheet[y0:y0+tile_h, x0:x0+tile_w] = tile
    path = output / f"{group}_colour_family_atlas.jpg"
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        f"training/cache/{CACHE_VERSION}/features_registered_drop_v2.csv"))
    parser.add_argument("--video-root", type=Path, default=Path(
        r"C:\Users\Administrator\Downloads\dual_sensor\dual_sensor\recordings\1"))
    parser.add_argument("--output", type=Path, default=Path(
        "training/output/rh_colour_family_atlas_v1"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    items = endpoint_rows(read_csv(args.cache))
    by_video = defaultdict(list)
    for item in items:
        by_video[item["video"]].append(item)
    tiles = defaultdict(list); audit = []
    for video, video_items in by_video.items():
        path = source_path(args.video_root, video)
        cap = cv2.VideoCapture(str(path))
        for item in sorted(video_items, key=lambda value: value["time"]):
            cap.set(cv2.CAP_PROP_POS_MSEC, item["time"] * 1000)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode {path.name} at {item['time']:.2f}s")
            frame = resize_for_app(frame)
            lab, _, selected = balanced_frame_and_masks(frame, item["row"])
            masks, fractions = family_masks(lab, selected)
            tiles[item["group"]].append(render_tile(
                frame, item["row"], lab, selected, masks, fractions,
                item["stage"], item["time"]))
            audit.append({"group": item["group"], "video": video,
                          "time": item["time"], "reference": item["stage"],
                          "selected_pixels": int(selected.sum()), **fractions})
        cap.release()
    paths = [sheet_for(group, group_tiles, args.output)
             for group, group_tiles in sorted(tiles.items())]
    (args.output / "atlas_manifest.json").write_text(json.dumps({
        "family_colours_bgr": FAMILY_COLOURS,
        "sheets": [str(path) for path in paths], "endpoints": audit,
        "reading_order": "RAW | selected sensing pixels | colour family | zoom",
    }, indent=2), encoding="utf-8")
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
