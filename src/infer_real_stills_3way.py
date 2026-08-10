"""Run a 3-way {bg, wire, connector} model over the real still photographs.

These are real camera images with no connector labels anywhere in the project,
so nothing here is scored. The output is for looking at: each image is written
as a side-by-side pair, original on the left and overlay on the right, plus
contact sheets so a reviewer can scan many at once.

Preprocessing matches the real-video path exactly (centre-crop to 4:3, resize to
640x480, ImageNet normalisation), so what is seen here is what the model sees on
the real videos too.

Usage:
  python src/infer_real_stills_3way.py --ckpt <path> --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_rgb_only_sota_gifs import load_model  # noqa: E402
from infer_video_rgb_only import center_crop_to_aspect, preprocess  # noqa: E402
from train_rgb_only_sota import RGB_MEAN, RGB_STD  # noqa: E402

WIRE_BGR = (0, 255, 0)
CON_BGR = (200, 0, 200)


@torch.no_grad()
def predict(model, bgr_640x480, device):
    rgb = bgr_640x480[:, :, ::-1].copy()
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device).float() / 255.0
    t = (t - RGB_MEAN.to(device)) / RGB_STD.to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(t).float()
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def overlay(bgr, mask, alpha=0.5):
    out = bgr.copy()
    for cls, col in ((1, WIRE_BGR), (2, CON_BGR)):
        sel = mask == cls
        if sel.any():
            layer = np.zeros_like(bgr)
            layer[sel] = col
            out[sel] = (out[sel] * (1 - alpha) + layer[sel] * alpha).astype(np.uint8)
    return out


def bar(w, text, h=30, shade=245):
    b = np.full((h, w, 3), shade, np.uint8)
    cv2.putText(b, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1,
                cv2.LINE_AA)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--imgs", default="data/real_wires_valset/imgs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sheet-cols", type=int, default=2)
    ap.add_argument("--sheet-rows", type=int, default=4)
    args = ap.parse_args()

    pairs_dir = os.path.join(args.out, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)
    device = torch.device(args.device)
    model = load_model(args.ckpt, device)

    files = sorted(glob.glob(os.path.join(args.imgs, "*.jpg")) +
                   glob.glob(os.path.join(args.imgs, "*.png")))
    print(f"[stills] {len(files)} images from {args.imgs}")

    rows, stats = [], []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        raw = cv2.imread(f)
        if raw is None:
            print(f"  skip unreadable {f}")
            continue
        proc = preprocess(raw)                       # centre-crop 4:3 -> 640x480
        mask = predict(model, proc, device)
        ov = overlay(proc, mask)

        wire_pct = 100.0 * float((mask == 1).mean())
        con_pct = 100.0 * float((mask == 2).mean())
        stats.append((name, wire_pct, con_pct))

        left = np.vstack([bar(proc.shape[1], f"{name}   camera image"), proc])
        right = np.vstack([bar(proc.shape[1],
                               f"wire {wire_pct:.1f}%   connector {con_pct:.2f}%"), ov])
        sep = np.full((left.shape[0], 4, 3), 255, np.uint8)
        pair = np.hstack([left, sep, right])
        cv2.imwrite(os.path.join(pairs_dir, f"{name}.png"), pair)
        rows.append(pair)

    # contact sheets
    per = args.sheet_cols * args.sheet_rows
    nsheet = 0
    for i0 in range(0, len(rows), per):
        chunk = rows[i0:i0 + per]
        h, w = chunk[0].shape[:2]
        canvas = np.full((h * args.sheet_rows, w * args.sheet_cols, 3), 255, np.uint8)
        for k, im in enumerate(chunk):
            r, c = divmod(k, args.sheet_cols)
            canvas[r * h:r * h + im.shape[0], c * w:c * w + im.shape[1]] = im
        nsheet += 1
        cv2.imwrite(os.path.join(args.out, f"contact_sheet_{nsheet:02d}.png"), canvas)

    with open(os.path.join(args.out, "coverage.csv"), "w") as fh:
        fh.write("image,wire_coverage_pct,connector_coverage_pct\n")
        for n, wpc, cpc in stats:
            fh.write(f"{n},{wpc:.3f},{cpc:.3f}\n")

    con_found = sum(1 for _, _, c in stats if c > 0.01)
    print(f"[stills] wrote {len(rows)} pairs, {nsheet} contact sheets -> {args.out}")
    print(f"[stills] frames with any connector predicted: {con_found}/{len(stats)}")
    print(f"[stills] mean wire coverage {np.mean([w for _, w, _ in stats]):.2f}%  "
          f"mean connector coverage {np.mean([c for _, _, c in stats]):.3f}%")


if __name__ == "__main__":
    main()
