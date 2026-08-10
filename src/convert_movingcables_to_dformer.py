#!/usr/bin/env python
"""Convert the MovingCables dataset (Holesovsky, Skoviera, Hlavac, 2024) into the
binary wire-segmentation training layout consumed by src/train_rgb_only_sota.py.

MovingCables source layout (after extracting MovingCables_small.tar):
    <root>/MovingCables/sampled_compositions_small/<split>/<image_type>/<clip>/<frame>.png
  image types:
    rgb_clips/              RGB photo (480x640, uint8 BGR) -- USE THIS
    flow_first_back/        3-channel 16-bit PNG. cv2(BGR)-channel-0 holds the
                            per-pixel CABLE INSTANCE label (0=bg, 1,2,3,..=cable
                            instances). The other two channels are optical flow
                            centred at 2**15. -- mask source
    normal_flow_first_back/ flow visualisation (unused)
    stick_masks/            poking-stick mask (unused)

Our trainer layout (--data-dir <out>):
    RGB/<base>.png      BGR colour, 480x640  (already native MovingCables res)
    Depth/<base>.png    grayscale (model is RGB-only; we write zeros -- the cache
                        builder requires the file to exist but the value is ignored)
    Label/<base>.png    grayscale. ENCODING D (bulletproof legacy mode):
                          background -> 0, wire (any cable instance) -> 4.
                        Rationale: train_rgbd_seg.build_cache auto-detects label
                        mode from the FIRST file's max value: max<=2 => "three_way"
                        (read as-is) else "legacy" (gt_transform: v-1, v==0 -> 255 bg).
                        The dataset then binary-collapses cache<=3 -> wire=1.
                        With wire=4: max>=3 forces LEGACY for every frame, gt
                        maps 4->3 and 0->255, collapse gives wire=1, bg=0. CORRECT
                        and mode-stable. (A binary {0,1} mask would mis-trigger
                        three_way and collapse EVERYTHING to foreground.)
    train.txt / test.txt   lines "RGB/<base>.png"

Basename convention: {setid:03d}_{frameidx:04d}_00_mc.png  where setid is a
contiguous per-clip integer (so train_rgb_only_sota's filter_indices_by_set, which
parses int(basename.split('_')[0]), groups frames by clip). The split is BY CLIP
(no temporal leakage between train and val).
"""
import os
import sys
import argparse
import glob

import cv2
import numpy as np

WIRE_VAL = 4          # Encoding D: forces legacy mode + collapses to wire
BG_VAL = 0
IMAGE_H, IMAGE_W = 480, 640


def find_split_root(raw_root):
    """Locate the .../sampled_compositions_small dir under raw_root."""
    cand = glob.glob(os.path.join(raw_root, "**", "sampled_compositions_small"),
                     recursive=True)
    if not cand:
        raise SystemExit(f"could not find sampled_compositions_small under {raw_root}")
    return cand[0]


def list_clips(comp_root):
    """Return [(split, clip, rgb_dir, flow_dir), ...] across all splits present."""
    out = []
    for split in sorted(os.listdir(comp_root)):
        rgb_base = os.path.join(comp_root, split, "rgb_clips")
        flow_base = os.path.join(comp_root, split, "flow_first_back")
        if not os.path.isdir(rgb_base):
            continue
        for clip in sorted(os.listdir(rgb_base)):
            rgb_dir = os.path.join(rgb_base, clip)
            flow_dir = os.path.join(flow_base, clip)
            if os.path.isdir(rgb_dir) and os.path.isdir(flow_dir):
                out.append((split, clip, rgb_dir, flow_dir))
    return out


def instance_label_from_flow(flow_path):
    """Read the cable-instance label channel from a flow_first_back PNG.

    The instance channel is the channel with small integer values (the other two
    are flow centred at 2**15). We pick the channel with the smallest max.
    """
    flow = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
    if flow is None or flow.ndim != 3 or flow.shape[2] != 3:
        raise ValueError(f"unexpected flow PNG {flow_path}: {None if flow is None else flow.shape}")
    maxes = [int(flow[:, :, c].max()) for c in range(3)]
    inst_c = int(np.argmin(maxes))
    return flow[:, :, inst_c]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True,
                    help="dir containing the extracted MovingCables/ tree")
    ap.add_argument("--out-dir", required=True, help="output dataset dir")
    ap.add_argument("--frame-step", type=int, default=2,
                    help="keep every Nth frame per clip (temporal subsample; default 2)")
    ap.add_argument("--val-frac", type=float, default=0.125,
                    help="fraction of CLIPS held out for val (default 0.125, matches phase15)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit-clips", type=int, default=None,
                    help="for smoke testing: only process the first K clips")
    args = ap.parse_args()

    comp_root = find_split_root(args.raw_root)
    clips = list_clips(comp_root)
    if args.limit_clips is not None:
        clips = clips[:args.limit_clips]
    print(f"found {len(clips)} clips under {comp_root}")

    rgb_out = os.path.join(args.out_dir, "RGB")
    depth_out = os.path.join(args.out_dir, "Depth")
    label_out = os.path.join(args.out_dir, "Label")
    for d in (rgb_out, depth_out, label_out):
        os.makedirs(d, exist_ok=True)

    # Deterministic clip->set-id assignment and train/val split BY CLIP.
    rng = np.random.RandomState(args.seed)
    n_clips = len(clips)
    perm = rng.permutation(n_clips)
    n_val = max(1, int(round(n_clips * args.val_frac)))
    val_clip_idx = set(perm[:n_val].tolist())

    zero_depth = np.zeros((IMAGE_H, IMAGE_W), np.uint8)
    train_lines, val_lines = [], []
    n_written = 0
    fg_fracs = []

    for setid, (split, clip, rgb_dir, flow_dir) in enumerate(clips):
        rgb_frames = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        kept = rgb_frames[::args.frame_step]
        is_val = setid in val_clip_idx
        for fidx, rgb_path in enumerate(kept):
            frame_name = os.path.basename(rgb_path)
            flow_path = os.path.join(flow_dir, frame_name)
            if not os.path.exists(flow_path):
                continue
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb is None or rgb.shape[:2] != (IMAGE_H, IMAGE_W):
                if rgb is not None:
                    rgb = cv2.resize(rgb, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
                else:
                    continue
            inst = instance_label_from_flow(flow_path)
            if inst.shape[:2] != (IMAGE_H, IMAGE_W):
                inst = cv2.resize(inst.astype(np.uint16), (IMAGE_W, IMAGE_H),
                                  interpolation=cv2.INTER_NEAREST)
            wire = (inst > 0)
            lbl = np.where(wire, WIRE_VAL, BG_VAL).astype(np.uint8)
            fg_fracs.append(float(wire.mean()))

            base = f"{setid:03d}_{fidx:04d}_00_mc"
            cv2.imwrite(os.path.join(rgb_out, base + ".png"), rgb)
            cv2.imwrite(os.path.join(depth_out, base + ".png"), zero_depth)
            cv2.imwrite(os.path.join(label_out, base + ".png"), lbl)
            line = f"RGB/{base}.png"
            (val_lines if is_val else train_lines).append(line)
            n_written += 1
        if (setid + 1) % 25 == 0:
            print(f"  {setid + 1}/{n_clips} clips, {n_written} frames written")

    with open(os.path.join(args.out_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(args.out_dir, "test.txt"), "w") as f:
        f.write("\n".join(val_lines) + "\n")

    fg = np.array(fg_fracs)
    print("\n=== CONVERSION DONE ===")
    print(f"clips: {n_clips}  (val clips: {n_val})")
    print(f"frames written: {n_written}  (train {len(train_lines)} / val {len(val_lines)})")
    print(f"frame-step: {args.frame_step}")
    print(f"fg-fraction: mean={fg.mean():.4f} median={np.median(fg):.4f} "
          f"min={fg.min():.4f} max={fg.max():.4f}")
    print(f"all-zero masks: {(fg == 0).sum()}  all-one(>0.99) masks: {(fg > 0.99).sum()}")
    print(f"out-dir: {args.out_dir}")


if __name__ == "__main__":
    main()
