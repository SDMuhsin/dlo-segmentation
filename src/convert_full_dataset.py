#!/usr/bin/env python3
"""Full CDLO → RGB-D Dataset Conversion.

Converts all available point cloud samples into RGB-D image pairs
for PyTorch data loading. Uses multiprocessing for parallelism.

Usage:
    python src/convert_full_dataset.py              # 8 workers (default)
    python src/convert_full_dataset.py --workers 4  # custom worker count
    python src/convert_full_dataset.py --dry-run    # show what would be done
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Import rasterizer and constants from existing pipeline
from pcl_to_rgbd import (
    CLASS_COLORS_RGB,
    CLASS_NAMES,
    DEPTH_FAR_MM,
    DEPTH_NEAR_MM,
    FRUSTUM_HALF,
    HALF_W,
    IMG_H,
    IMG_W,
    SCALE,
    VIEWS,
    make_view_matrix,
    rasterize_view,
)

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "set2"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "rgbd"

SAMPLES_PER_SET = 300
VIEW_NAMES = list(VIEWS.keys())


# ── Discovery ────────────────────────────────────────────────────────────────

def discover_sets():
    """Find all set directories that have the required subdirectories."""
    sets = []
    for entry in sorted(DATA_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        try:
            set_id = int(entry.name)
        except ValueError:
            continue
        pcl_dir = entry / "pointclouds_normed_4096"
        seg_dir = entry / "segmentation_normed_4096"
        if pcl_dir.is_dir() and seg_dir.is_dir():
            sets.append(set_id)
    return sets


def get_splits(available_sets):
    """Compute train/val/test splits intersected with available sets."""
    available = set(available_sets)
    return {
        "train": sorted(s for s in range(0, 32) if s in available),
        "val": sorted(s for s in range(32, 36) if s in available),
        "test": sorted(s for s in range(36, 40) if s in available),
    }


# ── Per-sample conversion ───────────────────────────────────────────────────

def sample_is_done(set_id, sample_id):
    """Check if all 12 output PNGs exist for this sample."""
    for view_name in VIEW_NAMES:
        fname = f"{sample_id:04d}_{view_name}.png"
        rgb_path = OUTPUT_ROOT / "rgb" / f"{set_id:03d}" / fname
        depth_path = OUTPUT_ROOT / "depth" / f"{set_id:03d}" / fname
        if not rgb_path.exists() or not depth_path.exists():
            return False
    return True


def convert_one_sample(args):
    """Convert a single sample. Designed for multiprocessing.Pool.map().

    Args:
        args: (set_id, sample_id) tuple

    Returns:
        (set_id, sample_id, status) where status is "ok", "skipped", or error string
    """
    set_id, sample_id = args

    # Check resumability
    if sample_is_done(set_id, sample_id):
        return (set_id, sample_id, "skipped")

    try:
        # Load data
        pcl_path = DATA_ROOT / f"{set_id:03d}" / "pointclouds_normed_4096" / f"pcl_{sample_id:04d}.npy"
        seg_path = DATA_ROOT / f"{set_id:03d}" / "segmentation_normed_4096" / f"seg_{sample_id:04d}.npy"
        points = np.load(str(pcl_path))
        labels = np.load(str(seg_path))

        rgb_dir = OUTPUT_ROOT / "rgb" / f"{set_id:03d}"
        depth_dir = OUTPUT_ROOT / "depth" / f"{set_id:03d}"

        for view_name, vdef in VIEWS.items():
            R = make_view_matrix(vdef["look"], vdef["up"])
            color_img, depth_img = rasterize_view(points, labels, R)

            fname = f"{sample_id:04d}_{view_name}.png"
            cv2.imwrite(str(rgb_dir / fname), color_img)
            cv2.imwrite(str(depth_dir / fname), depth_img)

        return (set_id, sample_id, "ok")
    except Exception as e:
        return (set_id, sample_id, f"error: {e}")


# ── Metadata ─────────────────────────────────────────────────────────────────

def build_metadata(available_sets, splits, stats):
    """Build the global metadata.json dict."""
    # Precompute view rotation matrices
    views = {}
    for vname, vdef in VIEWS.items():
        R = make_view_matrix(vdef["look"], vdef["up"])
        views[vname] = {
            "look_direction": vdef["look"].tolist(),
            "up_vector": vdef["up"].tolist(),
            "rotation_matrix": R.tolist(),
        }

    return {
        "projection": {
            "type": "orthographic",
            "image_width": IMG_W,
            "image_height": IMG_H,
            "scale_px_per_unit": float(SCALE),
            "frustum_half_vertical": float(FRUSTUM_HALF),
            "frustum_half_horizontal": float(HALF_W),
            "depth_near_mm": DEPTH_NEAR_MM,
            "depth_far_mm": DEPTH_FAR_MM,
            "color_format": "8-bit BGR PNG (class-colored)",
            "depth_format": "16-bit unsigned PNG (millimeters, 0=no data)",
        },
        "class_colors_rgb": {str(k): list(v) for k, v in CLASS_COLORS_RGB.items()},
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "views": views,
        "splits": {k: [f"{s:03d}" for s in v] for k, v in splits.items()},
        "stats": {
            "num_sets": len(available_sets),
            "samples_per_set": SAMPLES_PER_SET,
            "views_per_sample": len(VIEW_NAMES),
            "total_samples": stats["total_samples"],
            "total_image_pairs": stats["total_pairs"],
            "converted_ok": stats["ok"],
            "skipped_existing": stats["skipped"],
            "errors": stats["errors"],
            "sets": [f"{s:03d}" for s in available_sets],
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full CDLO → RGB-D conversion")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without converting")
    args = parser.parse_args()

    # Discover sets
    available_sets = discover_sets()
    if not available_sets:
        print("No valid sets found in", DATA_ROOT)
        sys.exit(1)

    splits = get_splits(available_sets)
    total_samples = len(available_sets) * SAMPLES_PER_SET

    print(f"Found {len(available_sets)} sets with {total_samples} total samples")
    print(f"  Train: {len(splits['train'])} sets ({splits['train'][0]}..{splits['train'][-1]})" if splits['train'] else "  Train: 0 sets")
    print(f"  Val:   {len(splits['val'])} sets ({splits['val'][0]}..{splits['val'][-1]})" if splits['val'] else "  Val:   0 sets")
    print(f"  Test:  {len(splits['test'])} sets ({splits['test'][0]}..{splits['test'][-1]})" if splits['test'] else "  Test:  0 sets")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"  Workers: {args.workers}")

    if args.dry_run:
        # Count how many would be skipped
        already_done = sum(
            1 for s in available_sets for i in range(SAMPLES_PER_SET)
            if sample_is_done(s, i)
        )
        print(f"\n  Already converted: {already_done}/{total_samples}")
        print(f"  Remaining: {total_samples - already_done}")
        return

    # Create output directories
    for set_id in available_sets:
        (OUTPUT_ROOT / "rgb" / f"{set_id:03d}").mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "depth" / f"{set_id:03d}").mkdir(parents=True, exist_ok=True)

    # Build work list
    work = [(set_id, sample_id)
            for set_id in available_sets
            for sample_id in range(SAMPLES_PER_SET)]

    # Convert with multiprocessing
    stats = {"ok": 0, "skipped": 0, "errors": 0,
             "total_samples": total_samples, "total_pairs": 0}
    error_list = []

    with Pool(processes=args.workers) as pool:
        for set_id, sample_id, status in tqdm(
            pool.imap_unordered(convert_one_sample, work),
            total=len(work),
            desc="Converting",
            unit="sample",
        ):
            if status == "ok":
                stats["ok"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            else:
                stats["errors"] += 1
                error_list.append(f"set={set_id:03d} sample={sample_id:04d}: {status}")

    stats["total_pairs"] = (stats["ok"] + stats["skipped"]) * len(VIEW_NAMES)

    # Report
    print(f"\nDone: {stats['ok']} converted, {stats['skipped']} skipped, "
          f"{stats['errors']} errors")
    print(f"Total image pairs: {stats['total_pairs']}")

    if error_list:
        print(f"\nErrors ({len(error_list)}):")
        for e in error_list[:20]:
            print(f"  {e}")
        if len(error_list) > 20:
            print(f"  ... and {len(error_list) - 20} more")

    # Write metadata
    metadata = build_metadata(available_sets, splits, stats)
    meta_path = OUTPUT_ROOT / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written to {meta_path}")


if __name__ == "__main__":
    main()
