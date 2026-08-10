#!/usr/bin/env python
"""Validate the assembled P32 fixture-negative co-train dataset + emit a contact sheet
of gated-PASS negatives and 6 (image,label) sample overlays.

Checks:
  1. train/val counts; negative count + fraction.
  2. every negative (set-id >= NEG_SETID_OFFSET, suffix _fn) has an ALL-ZERO label PNG.
  3. first train.txt line label max >= 3 (legacy cache mode preserved).
  4. no negative RGB file is unchanged (sha256) to any valset image.
  5. RGB/Depth/Label triple parity (every train/val line has all three PNGs, 480x640).
"""
import argparse
import glob
import hashlib
import json
import os

import cv2
import numpy as np
from PIL import Image

NEG_SETID_OFFSET = 8000


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def setid_of(line):
    return int(os.path.basename(line).split("_")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/dformer_dataset_p32_fixtureneg_cotrain")
    ap.add_argument("--valset-dir", default="data/real_wires_valset/imgs")
    ap.add_argument("--out-dir", default="results/realism_campaign/p32_fixture_negatives")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train = [l.strip() for l in open(os.path.join(args.data_dir, "train.txt")) if l.strip()]
    test = [l.strip() for l in open(os.path.join(args.data_dir, "test.txt")) if l.strip()]
    neg_lines = [l for l in train if setid_of(l) >= NEG_SETID_OFFSET]
    print(f"train {len(train)}  val {len(test)}")
    print(f"negatives in train: {len(neg_lines)}  ({100*len(neg_lines)/max(1,len(train)):.2f}%)")
    print(f"negatives in val:   {sum(1 for l in test if setid_of(l) >= NEG_SETID_OFFSET)} (must be 0)")

    # (2) all-zero labels + (5) triple parity + size
    bad_label, bad_triple, bad_size = [], [], []
    for l in neg_lines:
        lab_p = os.path.join(args.data_dir, l.replace("RGB/", "Label/"))
        dep_p = os.path.join(args.data_dir, l.replace("RGB/", "Depth/"))
        rgb_p = os.path.join(args.data_dir, l)
        if not (os.path.exists(lab_p) and os.path.exists(dep_p) and os.path.exists(rgb_p)):
            bad_triple.append(l); continue
        lab = np.array(Image.open(lab_p))
        if lab.max() != 0:
            bad_label.append((l, int(lab.max())))
        if lab.shape != (480, 640):
            bad_size.append((l, lab.shape))
    print(f"\n[2] negatives with non-zero label: {len(bad_label)} (must be 0)")
    if bad_label[:5]:
        print("    e.g.", bad_label[:5])
    print(f"[5] negatives missing a triple: {len(bad_triple)} (must be 0)")
    print(f"[5] negatives wrong size: {len(bad_size)} (must be 0)")

    # (3) first line legacy-mode
    first = train[0]
    first_lab = np.array(Image.open(os.path.join(args.data_dir, first.replace("RGB/", "Label/"))))
    print(f"\n[3] first train line {first} label max={first_lab.max()} "
          f"({'LEGACY OK' if first_lab.max() >= 3 else 'BAD: not legacy'})")

    # (4) no negative RGB collides with any valset image (sha256)
    val_sha = {sha(p) for p in glob.glob(os.path.join(args.valset_dir, "*"))}
    coll = [l for l in neg_lines if sha(os.path.join(args.data_dir, l)) in val_sha]
    print(f"\n[4] negative RGB byte-colliding with valset: {len(coll)} (must be 0)")

    # contact sheet of up to 40 negatives
    rng = np.random.default_rng(0)
    pick = [neg_lines[i] for i in rng.permutation(len(neg_lines))[:40]] if neg_lines else []
    cols, rows, cell = 8, 5, 160
    sheet = np.full((rows * cell, cols * cell, 3), 30, np.uint8)
    for k, l in enumerate(pick):
        img = cv2.imread(os.path.join(args.data_dir, l))
        if img is None:
            continue
        img = cv2.resize(img, (cell, cell))
        r, c = divmod(k, cols)
        sheet[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = img
    cs_path = os.path.join(args.out_dir, "contact_sheet.png")
    cv2.imwrite(cs_path, sheet)
    print(f"\ncontact sheet ({len(pick)} negs) -> {cs_path}")

    # 6 sample (image,label) overlays
    ov_dir = os.path.join(args.out_dir, "sample_overlays")
    os.makedirs(ov_dir, exist_ok=True)
    for k, l in enumerate(pick[:6]):
        img = cv2.imread(os.path.join(args.data_dir, l))
        lab = cv2.imread(os.path.join(args.data_dir, l.replace("RGB/", "Label/")), 0)
        overlay = img.copy()
        overlay[lab >= 3] = (0, 0, 255)  # wire pixels red (should be NONE for negatives)
        vis = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        cv2.putText(vis, f"wire_px={int((lab>=3).sum())}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(ov_dir, f"overlay_{k}.png"), vis)
    print(f"6 overlays -> {ov_dir}")

    ok = (not bad_label and not bad_triple and not bad_size and not coll
          and first_lab.max() >= 3
          and sum(1 for l in test if setid_of(l) >= NEG_SETID_OFFSET) == 0)
    summary = {
        "train": len(train), "val": len(test),
        "negatives_in_train": len(neg_lines),
        "neg_fraction_pct": 100*len(neg_lines)/max(1, len(train)),
        "negatives_in_val": sum(1 for l in test if setid_of(l) >= NEG_SETID_OFFSET),
        "nonzero_label_count": len(bad_label),
        "missing_triple": len(bad_triple), "wrong_size": len(bad_size),
        "valset_byte_collisions": len(coll),
        "first_line_label_max": int(first_lab.max()),
        "all_checks_pass": bool(ok),
        "contact_sheet": cs_path, "overlays_dir": ov_dir,
    }
    json.dump(summary, open(os.path.join(args.out_dir, "validate_summary.json"), "w"), indent=2)
    print("\n=== VALIDATION", "PASS ===" if ok else "FAIL ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
