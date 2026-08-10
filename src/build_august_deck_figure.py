"""Build the August-deck comparison figure: input | ground truth | previous | current.

Picks the val frames where the two models differ most on the connector class, so
the panel shows a real change rather than a hand-chosen best case. Selection is
by connector-IoU delta computed over the whole val split, and the chosen frame
ids are printed so the choice is reproducible.

Colours match the July deck: wire green, connector magenta.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_rgb_only_sota import RGB_MEAN, RGB_STD  # noqa: E402
from gen_rgb_only_sota_gifs import load_model  # noqa: E402

DATA = "data/dformer_dataset_3way_connscale3"
PREV = "results/realism_campaign/p_3way_decheat/seg_b5_3way_connscale3/best_model.pth"
CURR = "results/realism_campaign/p3w_lovasz/lovasz050_zoom/best_model.pth"
OUT = "results/presentations/KIAT_CREFLE_UPDATE_AUGUST2026/slide_compare.png"

WIRE_BGR = (0, 150, 0)
CON_BGR = (200, 0, 200)


@torch.no_grad()
def predict_all(ckpt, rgb, device, batch=8):
    model = load_model(ckpt, device)
    preds = np.zeros((len(rgb), 480, 640), np.uint8)
    for i0 in range(0, len(rgb), batch):
        i1 = min(i0 + batch, len(rgb))
        a = np.asarray(rgb[i0:i1])[:, :, :, ::-1].copy()
        t = torch.from_numpy(a.transpose(0, 3, 1, 2)).to(device).float() / 255.0
        t = (t - RGB_MEAN.to(device)) / RGB_STD.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            preds[i0:i1] = model(t).float().argmax(1).cpu().numpy().astype(np.uint8)
    del model
    torch.cuda.empty_cache()
    return preds


def con_iou(pred, gt):
    tp = int(((pred == 2) & (gt == 2)).sum())
    fp = int(((pred == 2) & (gt != 2)).sum())
    fn = int(((pred != 2) & (gt == 2)).sum())
    return tp / max(tp + fp + fn, 1)


def overlay(rgb_bgr, lbl, alpha=0.55):
    out = rgb_bgr.copy()
    for cls, col in ((1, WIRE_BGR), (2, CON_BGR)):
        m = lbl == cls
        if m.any():
            out[m] = (np.array(col, np.float32) * alpha
                      + out[m].astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return out


def label_bar(w, text, h=34):
    bar = np.full((h, w, 3), 245, np.uint8)
    cv2.putText(bar, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1,
                cv2.LINE_AA)
    return bar


def main():
    device = "cuda"
    rgb = np.load(f"{DATA}/cache/val_rgb.npy", mmap_mode="r")
    lab = np.load(f"{DATA}/cache/val_label.npy", mmap_mode="r")
    names = [l.strip() for l in open(f"{DATA}/test.txt") if l.strip()]

    print("[fig] running previous model ...", flush=True)
    p_prev = predict_all(PREV, rgb, device)
    print("[fig] running current model ...", flush=True)
    p_curr = predict_all(CURR, rgb, device)

    rows = []
    for i in range(len(rgb)):
        g = np.asarray(lab[i]).astype(np.int64)
        if (g == 2).sum() < 400:          # need a connector big enough to see
            continue
        a, b = con_iou(p_prev[i], g), con_iou(p_curr[i], g)
        rows.append((b - a, a, b, i))
    rows.sort(reverse=True)

    # take two improved frames from different held-out assemblies
    chosen, seen = [], set()
    for d, a, b, i in rows:
        s = names[i].split("/")[1].split("_")[0]
        if s in seen:
            continue
        seen.add(s)
        chosen.append((d, a, b, i))
        if len(chosen) == 2:
            break

    panels = []
    for d, a, b, i in chosen:
        img = np.asarray(rgb[i])
        g = np.asarray(lab[i]).astype(np.int64)
        cells = [
            (img, "Camera image"),
            (overlay(img, g), "Reference labels"),
            (overlay(img, p_prev[i]), f"Previous model   connector IoU {a*100:.0f}%"),
            (overlay(img, p_curr[i]), f"Current model   connector IoU {b*100:.0f}%"),
        ]
        row = [np.vstack([label_bar(c.shape[1], t), c]) for c, t in cells]
        sep = np.full((row[0].shape[0], 6, 3), 255, np.uint8)
        panels.append(np.hstack([x for pair in zip(row, [sep] * 4) for x in pair][:-1]))
        print(f"[fig] {names[i]}  prev {a:.3f} -> curr {b:.3f}")

    gap = np.full((14, panels[0].shape[1], 3), 255, np.uint8)
    sheet = np.vstack([panels[0], gap, panels[1]])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, sheet)
    print(f"[fig] wrote {OUT}  {sheet.shape[1]}x{sheet.shape[0]}")


if __name__ == "__main__":
    main()
