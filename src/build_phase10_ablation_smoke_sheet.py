"""Build a side-by-side contact sheet comparing the 4 Phase 10 ablation
variants against the Phase 4 baseline reference at the same sources.

For each (set_id, view), tile rows by variant in the order:
    baseline (Phase 4 reference from data/rgbd_videos), wall, camera, harness_hsv, objects

Camera variant uses view0..view5 slot names — we can't pick the "front" view
for direct comparison; for camera we just show its first slot view0.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SMOKE_ROOT = PROJECT_ROOT / "results" / "phase10_ablation" / "smokes"
PHASE4_ROOT = PROJECT_ROOT / "data" / "rgbd_videos"

SETS = ["000", "002", "003", "004", "006"]
NON_CAMERA_VIEW = "front"
CAMERA_VIEW = "view0"
OUT_PATH = SMOKE_ROOT / "smoke_contact_sheet.png"


def _load_rgb(p: Path):
    if not p.is_file():
        return np.zeros((480, 640, 3), dtype=np.uint8)
    return cv2.imread(str(p))


def _label_bar(img, text, height=28):
    bar = np.full((height, img.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    rows = []
    row_labels = [
        ("baseline (Phase 4 / rgbd_videos)", PHASE4_ROOT, NON_CAMERA_VIEW, "ph4"),
        ("wall (Phase 4 + 5-wall enclosure)", SMOKE_ROOT / "wall", NON_CAMERA_VIEW, "wall"),
        ("camera (Phase 4 + random views)", SMOKE_ROOT / "camera", CAMERA_VIEW, "cam"),
        ("harness_hsv (Phase 4 + HSV jitter)", SMOKE_ROOT / "harness_hsv", NON_CAMERA_VIEW, "hsv"),
        ("objects (Phase 4 + Phase 9 lib + fg)", SMOKE_ROOT / "objects", NON_CAMERA_VIEW, "obj"),
    ]
    for label, root, view, _ in row_labels:
        tiles = []
        for s in SETS:
            base = root / "train" / s / "rgb" / f"0000_00_{view}.png"
            img = _load_rgb(base)
            img = _label_bar(img, f"set={s} view={view}", height=22)
            tiles.append(img)
        row_img = np.hstack(tiles)
        row_img = _label_bar(row_img, label, height=30)
        rows.append(row_img)

    out = np.vstack(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PATH), out)
    print(f"Wrote {OUT_PATH} ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
