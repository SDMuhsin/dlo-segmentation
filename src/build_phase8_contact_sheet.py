#!/usr/bin/env python3
"""Build a contact sheet from the Phase 8 smoke render output.

Produces a single PNG showing the new harder synthetic dataset across
multiple sources × views × (rgb / label) for visual inspection.

Usage::

    python src/build_phase8_contact_sheet.py results/phase8_smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


CLASS_PALETTE_BGR = [
    (0, 0, 0),          # 0 = bg → black
    (180, 180, 180),    # 1 = wire → gray
    (0, 0, 255),        # 2 = endpoint → red
    (255, 0, 0),        # 3 = bifurcation → blue
    (0, 255, 0),        # 4 = connector → green
    (0, 255, 255),      # 5 = noise → yellow
]


def label_to_rgb(L: np.ndarray) -> np.ndarray:
    out = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.uint8)
    for i, bgr in enumerate(CLASS_PALETTE_BGR):
        out[L == i] = bgr
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("smoke_dir", type=Path)
    parser.add_argument("--set-id", type=int, default=0)
    parser.add_argument("--sources", type=int, nargs="+",
                        default=[0, 30, 60, 90, 120])
    parser.add_argument("--anim", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=None, help="Output PNG path.")
    args = parser.parse_args()

    base = args.smoke_dir / "train" / f"{args.set_id:03d}"
    if not base.is_dir():
        print(f"set dir not found: {base}")
        return 1

    views = ["front", "back", "left", "right", "top", "bottom"]
    panel_h, panel_w = 240, 320  # 1/2 size
    n_rows = 2 * len(args.sources)  # 2 rows per source: rgb + label
    n_cols = len(views)
    sheet = np.zeros((panel_h * n_rows, panel_w * n_cols, 3),
                      dtype=np.uint8) + 220

    for i, src in enumerate(args.sources):
        for j, vn in enumerate(views):
            stem = f"{src:04d}_{args.anim:02d}_{vn}"
            rgb = cv2.imread(str(base / "rgb" / f"{stem}.png"))
            lbl = cv2.imread(str(base / "label" / f"{stem}.png"),
                              cv2.IMREAD_UNCHANGED)
            if rgb is None or lbl is None:
                continue
            rgb_p = cv2.resize(rgb, (panel_w, panel_h))
            lbl_rgb = label_to_rgb(lbl)
            lbl_p = cv2.resize(lbl_rgb, (panel_w, panel_h))

            r0 = i * 2
            r1 = r0 + 1
            sheet[r0 * panel_h:(r0 + 1) * panel_h,
                  j * panel_w:(j + 1) * panel_w] = rgb_p
            sheet[r1 * panel_h:(r1 + 1) * panel_h,
                  j * panel_w:(j + 1) * panel_w] = lbl_p
            cv2.putText(sheet, f"src{src:03d} {vn} rgb",
                        (j * panel_w + 6, r0 * panel_h + panel_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1, cv2.LINE_AA)
            cv2.putText(sheet, f"src{src:03d} {vn} lbl",
                        (j * panel_w + 6, r1 * panel_h + panel_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1, cv2.LINE_AA)

    out = args.out or (args.smoke_dir / "contact_sheet.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"saved {out}  shape={sheet.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
