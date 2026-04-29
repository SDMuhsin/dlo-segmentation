"""
Prepare the RGB-D video dataset in DFormer format for training.

Takes the rendered RGB-D dataset from data/rgbd_videos/ and creates a
DFormer-compatible dataset at data/dformer_dataset/ with:
  - RGB/    : symlinks to original (Phase 4 textured) RGB images
  - Depth/  : 8-bit depth images (converted from 16-bit originals)
  - Label/  : symlinks to per-pixel label PNGs from the renderer
  - train.txt, test.txt : file index lists

Subset strategy (to keep training < 6 hours):
  - All 6 views (front, back, left, right, top, bottom)
  - Every 5th source frame (0, 5, 10, ..., 295 → 60 per set)
  - Animation frame 0 only (static base pose, maximum variety)
  - Train: 21 sets × 60 frames × 6 views = 7,560 images
  - Val:    3 sets × 60 frames × 6 views = 1,080 images

Label encoding on disk (before gt_transform):
  0 = background → becomes 255 after gt_transform, ignored in loss
  1 = Wire
  2 = Endpoint
  3 = Bifurcation
  4 = Connector
  5 = Noise
"""

import argparse
import os
import shutil
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# ---------- CONFIG ----------
SRC_ROOT = Path("data/rgbd_videos")
DST_ROOT = Path("data/dformer_dataset")
VIEWS = ["front", "back", "left", "right", "top", "bottom"]
SRC_FRAME_STEP = 5      # every 5th source frame
ANIM_FRAME = 0           # only anim frame 0
DEPTH_MIN_MM = 500
DEPTH_MAX_MM = 1500


def convert_depth_16to8(depth_16: np.ndarray) -> np.ndarray:
    """Convert 16-bit depth (mm) to 8-bit grayscale.

    0 (no data) → 0
    500-1500 mm → 1-255 linearly
    """
    out = np.zeros(depth_16.shape, dtype=np.uint8)
    mask = depth_16 > 0
    # Clip to valid range and map linearly to 1-255
    clipped = np.clip(depth_16[mask].astype(np.float32), DEPTH_MIN_MM, DEPTH_MAX_MM)
    out[mask] = ((clipped - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 254 + 1).astype(np.uint8)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--clean", action="store_true",
                   help="Wipe RGB/, Depth/, Label/, cache/, train.txt, test.txt before staging. "
                        "Use this after re-rendering data/rgbd_videos/ so downstream cache is rebuilt fresh.")
    p.add_argument("--limit-sets", type=int, default=None,
                   help="Process only the first N sets per split (smoke testing).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.clean:
        for sub in ["RGB", "Depth", "Label", "cache"]:
            d = DST_ROOT / sub
            if d.is_dir():
                shutil.rmtree(d)
                print(f"  Wiped {d}")
        for f in ["train.txt", "test.txt"]:
            p = DST_ROOT / f
            if p.exists():
                p.unlink()
                print(f"  Removed {p}")

    # Load metadata
    with open(SRC_ROOT / "metadata.json") as f:
        meta = json.load(f)

    train_sets = meta["splits"]["train"]
    val_sets = meta["splits"]["val"]

    if args.limit_sets is not None:
        train_sets = train_sets[:args.limit_sets]
        val_sets = val_sets[:args.limit_sets]
        print(f"  --limit-sets {args.limit_sets}: train={train_sets}  val={val_sets}")

    # Create output directories
    for subdir in ["RGB", "Depth", "Label"]:
        (DST_ROOT / subdir).mkdir(parents=True, exist_ok=True)

    train_files = []
    val_files = []

    # Process each split
    for split_name, set_ids, file_list in [
        ("train", train_sets, train_files),
        ("val", val_sets, val_files),
    ]:
        print(f"\nProcessing {split_name} split ({len(set_ids)} sets)...")

        for set_id in tqdm(set_ids, desc=f"  Sets"):
            n_frames = meta["source_frames_per_set"][set_id]
            src_frames = range(0, n_frames, SRC_FRAME_STEP)

            for src_idx in src_frames:
                for view in VIEWS:
                    # Source file name pattern: {src:04d}_{anim:02d}_{view}.png
                    fname = f"{src_idx:04d}_{ANIM_FRAME:02d}_{view}"
                    src_rgb = SRC_ROOT / split_name / set_id / "rgb" / f"{fname}.png"
                    src_depth = SRC_ROOT / split_name / set_id / "depth" / f"{fname}.png"
                    src_label = SRC_ROOT / split_name / set_id / "label" / f"{fname}.png"

                    if not src_rgb.exists():
                        print(f"  WARNING: missing {src_rgb}")
                        continue
                    if not src_label.exists():
                        print(f"  WARNING: missing {src_label}")
                        continue

                    # Unique output name: {set_id}_{src:04d}_{anim:02d}_{view}
                    out_name = f"{set_id}_{fname}"

                    # RGB: symlink to original
                    dst_rgb = DST_ROOT / "RGB" / f"{out_name}.png"
                    if not os.path.lexists(dst_rgb):
                        os.symlink(src_rgb.resolve(), dst_rgb)

                    # Depth: convert 16-bit → 8-bit
                    dst_depth = DST_ROOT / "Depth" / f"{out_name}.png"
                    if not dst_depth.exists():
                        depth_16 = cv2.imread(str(src_depth), cv2.IMREAD_UNCHANGED)
                        depth_8 = convert_depth_16to8(depth_16)
                        cv2.imwrite(str(dst_depth), depth_8)

                    # Label: symlink to per-pixel label PNG from the renderer
                    dst_label = DST_ROOT / "Label" / f"{out_name}.png"
                    if not os.path.lexists(dst_label):
                        os.symlink(src_label.resolve(), dst_label)

                    file_list.append(f"RGB/{out_name}.png")

    # Write index files
    with open(DST_ROOT / "train.txt", "w") as f:
        f.write("\n".join(train_files) + "\n")
    with open(DST_ROOT / "test.txt", "w") as f:
        f.write("\n".join(val_files) + "\n")

    print(f"\nDataset prepared at {DST_ROOT}")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val:   {len(val_files)} images")

    # Verify a sample
    sample = train_files[0].split("/")[1].replace(".png", "")
    label = cv2.imread(str(DST_ROOT / "Label" / f"{sample}.png"), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(DST_ROOT / "Depth" / f"{sample}.png"), cv2.IMREAD_GRAYSCALE)
    print(f"\n  Sample '{sample}':")
    print(f"    Label unique values: {np.unique(label)}")
    print(f"    Label class distribution:")
    for val in np.unique(label):
        pct = (label == val).sum() / label.size * 100
        name = {0: "background", 1: "Wire", 2: "Endpoint", 3: "Bifurcation", 4: "Connector", 5: "Noise"}.get(val, "?")
        print(f"      {val} ({name}): {pct:.2f}%")
    print(f"    Depth range: {depth.min()}-{depth.max()}")


if __name__ == "__main__":
    main()
