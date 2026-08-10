#!/usr/bin/env python3
"""Convert RMDLO TrackDLO/MultiDLO ROS1 bags into the binary wire-segmentation
training layout consumed by ``src/train_rgb_only_sota.py`` (P28, RMDLO source).

Pipeline per bag (all stages reuse already-validated modules):

  1. EXTRACT  — ``src/rmdlo_bag_to_pcl.read_bag_frames`` pulls synchronised
                (BGR 1280x720, aligned Z16 depth, intrinsics) frames at a chosen
                ``--frame-step``. RMDLO is a continuous ~30 fps recording, so
                consecutive frames are near-duplicates (measured: lag-1 wire-mask
                IoU ~0.96). The step is chosen to DE-CORRELATE frames (default 10
                → measured neighbour wire-mask IoU ~0.76, ~0.33 s apart).

  2. LABEL    — ``src/rmdlo_bag_to_pcl.hsv_wire_mask`` HSV-thresholds the blue/teal
                rope → the GT wire mask. The GT Label is derived ONLY from THIS
                mask, mapped ``wire → 4, bg → 0`` (legacy Encoding D). It is NEVER
                re-derived from the re-skinned RGB (the cable is no longer blue, so
                HSV re-segmentation of the reskin gives IoU ~0.15 — wrong).

  3. RE-SKIN  — ``src/rmdlo_wire_reskin.reskin_frame`` replaces the vivid-blue rope
                chroma/material with a realistic, VARIED cable appearance while
                preserving the real anti-aliased edge, the cylindrical shading, the
                depth and the mask. A per-frame coin flip (``--bg-mix``) chooses the
                REAL RMDLO white-table background or a VARIED-background composite
                (cable cut-out over a real movingcables backdrop) — RMDLO's own
                backdrop is near-uniform white, and our real-valset limiter is
                wire-vs-coplanar-surface PRECISION, so background variety matters.

  4. WRITE    — RGB/<base>.png (BGR, 640x480), Label/<base>.png ({0,4}, 640x480),
                Depth/<base>.png (uint16 aligned Z16, 640x480 nearest). The model is
                RGB-only so the depth VALUE is ignored by the trainer (build_cache
                reads it 8-bit), but the real metric uint16 depth is carried so the
                bundle is RGB-D-ready and depth is provably present + aligned.

Output layout (``--out-dir``, default data/dformer_dataset_rmdlo):
    RGB/<base>.png  Depth/<base>.png  Label/<base>.png  train.txt  test.txt

Basename: ``{setid:03d}_{frameidx:04d}_00_rm.png``. ``setid`` is a contiguous
per-BAG integer so ``filter_indices_by_set`` (int(basename.split('_')[0])) groups
frames by bag. The train/val split is BY BAG (no temporal leakage across the split).

Label encoding rationale (Encoding D, see convert_movingcables_to_dformer.py):
  build_cache auto-detects mode from the FIRST label's max: max<=2 => three_way
  (read as-is), else legacy (gt_transform v-1, 0->255). With wire=4 every frame has
  max>=3 => LEGACY forced, gt maps 4->3 and 0->255, binary collapse cache<=3 => wire.
  CORRECT and mode-stable. A {0,1} mask would mis-trip three_way and collapse all to fg.

Parameterised to scale UNCHANGED over the full data/rmdlo_raw/ download: point
``--bags`` / ``--bag-glob`` at the bag set and run.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from rmdlo_bag_to_pcl import (read_bag_frames, hsv_wire_mask,            # noqa: E402
                              HSV_LO_DEFAULT, HSV_HI_DEFAULT)
from rmdlo_wire_reskin import (reskin_frame, varied_background,          # noqa: E402
                               composite_on_bg)

WIRE_VAL = 4          # Encoding D: forces legacy mode + collapses to wire
BG_VAL = 0
IMAGE_H, IMAGE_W = 480, 640   # trainer cache shape (build_cache writes 480x640)


def discover_bags(args):
    """Return sorted list of bag paths from --bags and/or --bag-glob."""
    bags = list(args.bags or [])
    if args.bag_glob:
        bags += glob.glob(args.bag_glob, recursive=True)
    bags = sorted(set(os.path.abspath(b) for b in bags if b.endswith(".bag")))
    if not bags:
        raise SystemExit("no .bag files found (use --bags and/or --bag-glob)")
    return bags


def resize_to_cache(img, interp):
    if img.shape[:2] != (IMAGE_H, IMAGE_W):
        return cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation=interp)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bags", nargs="*", default=None,
                    help="explicit .bag paths")
    ap.add_argument("--bag-glob", default=None,
                    help="glob for .bag files, e.g. 'data/rmdlo_raw/**/*.bag'")
    ap.add_argument("--out-dir", default="data/dformer_dataset_rmdlo")
    ap.add_argument("--frame-step", type=int, default=10,
                    help="keep every Nth bag frame (de-correlation; default 10 → "
                         "measured neighbour wire-mask IoU ~0.76)")
    ap.add_argument("--max-frames-per-bag", type=int, default=100000,
                    help="cap kept frames per bag (read budget)")
    ap.add_argument("--bg-mix", type=float, default=0.6,
                    help="fraction of frames using the VARIED-background composite "
                         "(rest use the real RMDLO white-table bg); default 0.6")
    ap.add_argument("--n-seg", type=int, default=4,
                    help="along-wire colour segments per frame (reskin variety)")
    ap.add_argument("--hsv-lo", type=int, nargs=3, default=list(HSV_LO_DEFAULT))
    ap.add_argument("--hsv-hi", type=int, nargs=3, default=list(HSV_HI_DEFAULT))
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of BAGS held out for val (split BY BAG); "
                         "default 0.2 (>=1 bag)")
    ap.add_argument("--min-fg-px", type=int, default=300,
                    help="drop frames whose HSV wire mask has fewer fg px (no rope)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-panels-dir", default=None,
                    help="optional dir for a few [orig|mask|reskin|reskinbg] panels")
    args = ap.parse_args()

    bags = discover_bags(args)
    print(f"found {len(bags)} bag(s)")

    rgb_out = os.path.join(args.out_dir, "RGB")
    depth_out = os.path.join(args.out_dir, "Depth")
    label_out = os.path.join(args.out_dir, "Label")
    for d in (rgb_out, depth_out, label_out):
        os.makedirs(d, exist_ok=True)
    if args.save_panels_dir:
        os.makedirs(args.save_panels_dir, exist_ok=True)

    # Deterministic bag->set-id assignment and train/val split BY BAG.
    rng = np.random.default_rng(args.seed)
    n_bags = len(bags)
    perm = rng.permutation(n_bags)
    n_val = max(1, int(round(n_bags * args.val_frac))) if n_bags > 1 else 0
    val_bag_idx = set(perm[:n_val].tolist())

    train_lines, val_lines = [], []
    fg_fracs = []
    n_written = 0
    n_dropped_nowire = 0
    bg_variant_count = 0
    per_bag = []

    for setid, bag in enumerate(bags):
        # Pull a budget of frames; read_bag_frames samples every frame-step.
        frames, intr = read_bag_frames(bag, num_frames=args.max_frames_per_bag,
                                       stride=args.frame_step, start=0)
        is_val = setid in val_bag_idx
        bag_written = 0
        for fidx, fr in enumerate(frames):
            bgr = fr["bgr"]
            depth = fr["depth"]            # uint16 aligned Z16, native bag res

            # --- GT label from the ORIGINAL HSV mask (never from the reskin) ---
            hard = hsv_wire_mask(bgr, tuple(args.hsv_lo), tuple(args.hsv_hi))
            if int(hard.sum()) < args.min_fg_px:
                n_dropped_nowire += 1
                continue

            # --- re-skin (chroma/material swap; edge/shade/mask preserved) ---
            r = reskin_frame(bgr, rng, n_seg=args.n_seg,
                             hsv_lo=tuple(args.hsv_lo), hsv_hi=tuple(args.hsv_hi))
            # reskin_frame recomputes the same hard mask internally; use ITS hard
            # mask as the canonical label so RGB and Label are exactly consistent.
            hard = r["hard"]

            use_bg_swap = (rng.random() < args.bg_mix)
            if use_bg_swap:
                bg = varied_background(bgr.shape, rng)
                rgb_final = composite_on_bg(r["reskinned"], r["alpha"], bg)
                bg_variant_count += 1
            else:
                rgb_final = r["reskinned"]

            # --- resize ALL to the trainer cache shape (640x480) ---
            rgb_final = resize_to_cache(rgb_final, cv2.INTER_AREA)
            # Label: nearest so it stays {wire,bg}; map wire->4 bg->0 AFTER resize.
            hard_r = resize_to_cache(hard.astype(np.uint8), cv2.INTER_NEAREST) > 0
            lbl = np.where(hard_r, WIRE_VAL, BG_VAL).astype(np.uint8)
            depth_r = resize_to_cache(depth, cv2.INTER_NEAREST)   # keep uint16

            fg_fracs.append(float(hard_r.mean()))

            base = f"{setid:03d}_{fidx:04d}_00_rm"
            cv2.imwrite(os.path.join(rgb_out, base + ".png"), rgb_final)
            cv2.imwrite(os.path.join(label_out, base + ".png"), lbl)
            # uint16 depth carried through (PNG preserves 16-bit).
            cv2.imwrite(os.path.join(depth_out, base + ".png"),
                        depth_r.astype(np.uint16))
            (val_lines if is_val else train_lines).append(f"RGB/{base}.png")
            n_written += 1
            bag_written += 1

            if args.save_panels_dir and fidx < 4:
                mvis = cv2.cvtColor((hard_r * 255).astype(np.uint8),
                                    cv2.COLOR_GRAY2BGR)
                tiles = [resize_to_cache(bgr, cv2.INTER_AREA), mvis, rgb_final]
                panel = np.hstack([cv2.resize(t, (320, 240)) for t in tiles])
                cv2.imwrite(os.path.join(args.save_panels_dir,
                                         f"panel_{base}.png"), panel)

        per_bag.append({"setid": setid, "bag": os.path.basename(bag),
                        "frames_written": bag_written,
                        "split": "val" if is_val else "train"})
        print(f"  bag {setid:03d} [{'val' if is_val else 'train'}] "
              f"{os.path.basename(bag)}: {bag_written} frames "
              f"(from {len(frames)} pulled @ step {args.frame_step})")

    with open(os.path.join(args.out_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_lines) + ("\n" if train_lines else ""))
    with open(os.path.join(args.out_dir, "test.txt"), "w") as f:
        f.write("\n".join(val_lines) + ("\n" if val_lines else ""))

    fg = np.array(fg_fracs) if fg_fracs else np.zeros(1)
    meta = {
        "bags": n_bags, "val_bags": n_val, "frame_step": args.frame_step,
        "bg_mix": args.bg_mix, "frames_written": n_written,
        "train": len(train_lines), "val": len(val_lines),
        "dropped_no_wire": n_dropped_nowire,
        "bg_varied_composite_frames": bg_variant_count,
        "fg_fraction_mean": float(fg.mean()),
        "fg_fraction_median": float(np.median(fg)),
        "fg_fraction_p05": float(np.percentile(fg, 5)),
        "fg_fraction_p95": float(np.percentile(fg, 95)),
        "n_all_zero": int((fg == 0).sum()),
        "n_all_one": int((fg > 0.99).sum()),
        "per_bag": per_bag,
    }
    with open(os.path.join(args.out_dir, "convert_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== RMDLO CONVERSION DONE ===")
    print(f"bags: {n_bags}  (val bags: {n_val})")
    print(f"frames written: {n_written}  (train {len(train_lines)} / val {len(val_lines)})")
    print(f"dropped (no-wire frames): {n_dropped_nowire}")
    print(f"varied-bg composite frames: {bg_variant_count}/{n_written} "
          f"({100*bg_variant_count/max(1,n_written):.0f}%)")
    print(f"fg-fraction: mean={fg.mean():.4f} median={np.median(fg):.4f} "
          f"p05={np.percentile(fg,5):.4f} p95={np.percentile(fg,95):.4f}")
    print(f"all-zero: {(fg==0).sum()}  all-one(>0.99): {(fg>0.99).sum()}")
    print(f"out-dir: {args.out_dir}")


if __name__ == "__main__":
    main()
