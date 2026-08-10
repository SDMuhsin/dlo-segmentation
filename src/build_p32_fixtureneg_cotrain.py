#!/usr/bin/env python
"""Build the P32 fixture-surface hard-negative co-train dataset =
   the current best dataset (data/dformer_dataset_p28_realtextured_cotrain)
   UNION gated-clean real electrical-fixture-SURFACE photos added as PURE
   all-background negatives to the TRAIN split only.

Mirrors src/build_p28_realtextured_cotrain.py conventions:
  * RGB/Depth/Label PNG dirs, train.txt/test.txt with "RGB/<base>.png" lines.
  * set-id = int(basename.split('_')[0]) (filter_indices_by_set grouping).
    synth < 1000, MovingCables 1000..2999, realtextured 3000..6999 ->
    fixture negatives use a fresh +8000 block (8000.. ), suffix "_fn",
    one set-id per negative image (no temporal duplication), so they never
    collide with any existing set.
  * Label encoding = legacy {0,4} (same as MC/RT). Negatives are ALL-ZERO
    (background), which is valid legacy bg. The base dataset's FIRST train.txt
    line stays a synth frame (Label max=5) -> build_cache keeps legacy mode;
    we APPEND negatives at the END of train.txt, never at the front.
  * Image format = letterbox 480x640 BGR (cv2 INTER_AREA), Depth = zeros uint16,
    identical to convert_realtextured_to_dformer.py.

Base frames are symlinked (mirroring the base build, which is itself symlinks).
Negative RGB/Depth/Label PNGs are MATERIALIZED (real files) under the out dir.
"""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np

BASE = "data/dformer_dataset_p28_realtextured_cotrain"
WIRE_VAL = 4
BG_VAL = 0
IMAGE_H, IMAGE_W = 480, 640
NEG_SETID_OFFSET = 8000  # fresh block, far above synth/MC(<3000)/RT(<7000)


def letterbox(img, target_h, target_w, interp, pad_value=0):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    if img.ndim == 3:
        canvas = np.full((target_h, target_w, img.shape[2]), pad_value, img.dtype)
    else:
        canvas = np.full((target_h, target_w), pad_value, img.dtype)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def read_lines(path):
    if not os.path.exists(path):
        return []
    return [l.strip() for l in open(path) if l.strip()]


def symlink_base(base_dir, out_dir):
    """Symlink every RGB/Depth/Label PNG from base_dir into out_dir (resolve through
    the base's own symlinks to the real source). Returns (train_lines, test_lines)."""
    for sub in ("RGB", "Depth", "Label"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    n = 0
    for sub in ("RGB", "Depth", "Label"):
        for sp in glob.glob(os.path.join(base_dir, sub, "*.png")):
            real = os.path.realpath(sp)  # follow base's symlink to the true source PNG
            link = os.path.join(out_dir, sub, os.path.basename(sp))
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(real, link)
            n += 1
    print(f"  symlinked {n} base PNGs (RGB+Depth+Label)")
    return read_lines(os.path.join(base_dir, "train.txt")), \
        read_lines(os.path.join(base_dir, "test.txt"))


REPLICA_STRIDE = 1000  # each oversample replica block offset by +1000 (distinct set-ids)


def add_negatives(neg_paths, out_dir, mult=1):
    """Materialize each gated-clean negative ONCE as RGB/Depth/Label PNG (all-zero label)
    at set-id +8000+i. For mult>1, add (mult-1) symlink replicas per image at distinct
    +1000-strided set-id blocks (sharing the underlying PNG), each referenced once more
    in train.txt — mirrors the realtextured/P29 oversample convention.
    Returns the list of 'RGB/<base>.png' train lines (len = mult * #materialized)."""
    zero_depth = np.zeros((IMAGE_H, IMAGE_W), np.uint16)
    zero_label = np.zeros((IMAGE_H, IMAGE_W), np.uint8)  # 100% background
    lines = []
    materialized = []  # (i, base) for the real copies
    for i, p in enumerate(neg_paths):
        rgb = cv2.imread(p, cv2.IMREAD_COLOR)
        if rgb is None:
            print(f"  WARN unreadable negative skipped: {p}")
            continue
        rgb_lb = letterbox(rgb, IMAGE_H, IMAGE_W, cv2.INTER_AREA, pad_value=0)
        setid = NEG_SETID_OFFSET + i
        base = f"{setid:05d}_{0:04d}_00_fn"
        cv2.imwrite(os.path.join(out_dir, "RGB", base + ".png"), rgb_lb)
        cv2.imwrite(os.path.join(out_dir, "Depth", base + ".png"), zero_depth)
        cv2.imwrite(os.path.join(out_dir, "Label", base + ".png"), zero_label)
        lines.append(f"RGB/{base}.png")
        materialized.append((i, base))
    # oversample replicas: symlink each materialized PNG into a fresh +1000-strided block
    for k in range(1, mult):
        for i, base in materialized:
            new_setid = NEG_SETID_OFFSET + k * REPLICA_STRIDE + i
            newbase = f"{new_setid:05d}_{0:04d}_00_fn"
            for sub in ("RGB", "Depth", "Label"):
                src = os.path.abspath(os.path.join(out_dir, sub, base + ".png"))
                link = os.path.join(out_dir, sub, newbase + ".png")
                if os.path.lexists(link):
                    os.remove(link)
                os.symlink(src, link)
            lines.append(f"RGB/{newbase}.png")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out-dir", default="data/dformer_dataset_p32_fixtureneg_cotrain")
    ap.add_argument("--gate-json", required=True,
                    help="gate_result.json from gate_fixture_negatives.py")
    ap.add_argument("--target-frac", type=float, default=0.065,
                    help="target negative fraction of the FINAL train split (5-8%)")
    ap.add_argument("--max-mult", type=int, default=2,
                    help="max oversample multiplier when unique negs < target (cap)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    gate = json.load(open(args.gate_json))
    kept = [r["path"] for r in gate["results"] if r["verdict"] == "KEEP"]
    print(f"gated-clean negatives available: {len(kept)}")

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    print("symlinking base dataset ...")
    base_train, base_test = symlink_base(args.base, out)
    n_base_train = len(base_train)
    print(f"  base train {n_base_train}  base val {len(base_test)}")

    # target negative COUNT (effective train references) to hit target_frac:
    #   neg / (n_base_train + neg) = frac  ->  neg = frac*n_base_train/(1-frac)
    target_neg = int(round(args.target_frac * n_base_train / (1.0 - args.target_frac)))
    rng = np.random.default_rng(args.seed)
    if len(kept) >= target_neg:
        # plenty of unique negatives: subsample to target, single pass (mult=1).
        idx = rng.permutation(len(kept))[:target_neg]
        chosen = [kept[i] for i in sorted(idx.tolist())]
        mult = 1
        print(f"  subsampling {len(kept)} -> {target_neg} unique negatives "
              f"(target {100*args.target_frac:.1f}% of train)")
    else:
        # fewer unique negatives than target: use ALL, oversample (capped) toward target.
        chosen = kept
        mult = max(1, min(args.max_mult, int(round(target_neg / max(1, len(kept))))))
        eff = mult * len(kept)
        print(f"  using ALL {len(kept)} unique negatives, oversample x{mult} "
              f"-> {eff} effective (target {target_neg}, cap x{args.max_mult})")

    print("materializing negatives (all-background labels) ...")
    neg_lines = add_negatives(chosen, out, mult=mult)

    train = base_train + neg_lines          # negatives APPENDED at the end
    test = base_test                        # NO negatives in val
    with open(os.path.join(out, "train.txt"), "w") as f:
        f.write("\n".join(train) + "\n")
    with open(os.path.join(out, "test.txt"), "w") as f:
        f.write("\n".join(test) + "\n")

    n_neg = len(neg_lines)
    total = len(train)
    frac = 100.0 * n_neg / max(1, total)
    meta = {
        "base": args.base, "out_dir": out,
        "base_train": n_base_train, "base_val": len(base_test),
        "gated_clean_available": len(kept),
        "unique_negatives_used": len(chosen),
        "oversample_mult": mult,
        "negatives_added": n_neg,
        "final_train": total, "final_val": len(test),
        "neg_fraction_of_train_pct": frac,
        "neg_setid_offset": NEG_SETID_OFFSET,
        "label_encoding": {"wire": WIRE_VAL, "background": BG_VAL,
                           "negative_label": "all-zero (100% background)"},
        "first_train_line": train[0],
    }
    json.dump(meta, open(os.path.join(out, "build_meta.json"), "w"), indent=2)
    print("\n=== P32 FIXTURE-NEG CO-TRAIN BUILT ===")
    print(f"out-dir: {out}")
    print(f"train: {total} = base {n_base_train} + fixture-neg {n_neg}  ({frac:.2f}% neg)")
    print(f"val:   {len(test)} (base only, NO negatives)")
    print(f"first train.txt line: {train[0]}")


if __name__ == "__main__":
    main()
