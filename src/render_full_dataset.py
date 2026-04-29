#!/usr/bin/env python3
"""Render the full PointWire dataset to RGB-D videos (Phase 4 pipeline).

Production runner around ``convert_to_video_dataset.convert_one_video``. This
is what you run for the long full-dataset render (≈ 8 100 sources × 20 anim
× 6 views = 972 000 image triples). Adds, on top of the per-source worker:

* **Pre-flight**: verifies the texture / background-photo / 3D-object
  libraries before starting the multi-hour run, and refuses to start if any
  are missing.
* **File logging**: tees stdout to ``results/render_logs/<timestamp>.log``
  so you can detach (``tmux``, ``nohup``) and come back later.
* **Resumable**: each worker calls ``_video_is_done(...)`` and skips sources
  whose RGB / depth / label PNGs already exist — re-running this script after
  an interruption picks up where it left off.
* **Post-render validation**: random-sample sanity check on the produced
  PNGs (label values, depth range, file count vs. plan).
* **Optional DFormer cache rebuild** (``--rebuild-cache``): runs
  ``prepare_dformer_data.py`` after a successful render so the training
  cache is refreshed in one go.

Usage (from project root, with ``env`` activated):

    python src/render_full_dataset.py                    # 8 workers, 20 anim
    python src/render_full_dataset.py --workers 16
    python src/render_full_dataset.py --dry-run          # plan + estimate, no render
    python src/render_full_dataset.py --rebuild-cache    # also refresh DFormer cache
    python src/render_full_dataset.py --sets 0 1 2       # only render specific sets

Background runs (recommended for full dataset):

    nohup python src/render_full_dataset.py --workers 8 > /dev/null 2>&1 &
    tail -f results/render_logs/render_phase4_*.log
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from convert_to_video_dataset import (  # noqa: E402
    BG_N_POINTS, DATA_ROOT, OUTPUT_ROOT, VIEW_NAMES,
    _get_background_library, _get_object_library, _get_texture_library,
    build_work_list, convert_one_video, split_of, write_metadata,
)


# ── Tee logger ─────────────────────────────────────────────────────────────

class _Tee:
    """Mirror writes to stdout AND a log file (line-buffered)."""

    def __init__(self, log_path: Path):
        self._term = sys.stdout
        self._log = open(log_path, "a", buffering=1)
        self._log.write(f"\n{'=' * 70}\n")
        self._log.write(f"Run started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        self._log.write(f"argv: {' '.join(sys.argv)}\n")
        self._log.write(f"{'=' * 70}\n")

    def write(self, msg):
        self._term.write(msg)
        self._log.write(msg)

    def flush(self):
        self._term.flush()
        self._log.flush()


# ── Discovery ─────────────────────────────────────────────────────────────

def discover_sets(only_sets: set[int] | None) -> dict[int, int]:
    """Scan ``data/set2/`` and return ``{set_id: usable_source_frames}``."""
    sets_found: dict[int, int] = {}
    for entry in sorted(DATA_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        try:
            sid = int(entry.name)
        except ValueError:
            continue
        if only_sets is not None and sid not in only_sets:
            continue
        needed = ["pointclouds_normed_4096", "segmentation_normed_4096", "skeletons"]
        if not all((entry / d).is_dir() for d in needed):
            continue
        n_pcl = len(list((entry / "pointclouds_normed_4096").glob("pcl_*.npy")))
        n_seg = len(list((entry / "segmentation_normed_4096").glob("seg_*.npy")))
        n_skel = len(list((entry / "skeletons").glob("*.npz")))
        usable = min(n_pcl, n_seg, n_skel)
        if usable > 0:
            sets_found[sid] = usable
    return sets_found


# ── Pre-flight ────────────────────────────────────────────────────────────

def preflight() -> bool:
    """Check that all three asset libraries are loadable + non-trivial."""
    print("Pre-flight checks:")
    ok = True

    if not DATA_ROOT.is_dir():
        print(f"  FAIL: raw data dir missing: {DATA_ROOT}")
        return False

    tex = _get_texture_library()
    n_tex = sum(len(v) for v in tex.values())
    print(f"  [{'OK' if n_tex >= 20 else 'FAIL'}] wire/connector textures: "
          f"{n_tex} (expected ≥ 20)")
    ok &= n_tex >= 20

    photos = _get_background_library()
    print(f"  [{'OK' if len(photos) >= 8 else 'FAIL'}] background photos: "
          f"{len(photos)} (expected ≥ 8)")
    ok &= len(photos) >= 8

    objs = _get_object_library()
    print(f"  [{'OK' if len(objs) >= 5 else 'FAIL'}] 3D real-object PCLs: "
          f"{len(objs)} (expected ≥ 5)")
    ok &= len(objs) >= 5

    manifest = PROJECT_ROOT / "data" / "objects" / "manifest.json"
    if manifest.is_file():
        try:
            m = json.loads(manifest.read_text())
            print(f"  [OK] data/objects/manifest.json: {m.get('total', '?')} entries")
        except Exception as e:
            print(f"  [FAIL] data/objects/manifest.json invalid: {e}")
            ok = False
    else:
        print(f"  [FAIL] data/objects/manifest.json missing")
        ok = False

    return ok


# ── Post-render validation ────────────────────────────────────────────────

def post_render_validate(sets_found: dict[int, int],
                         num_frames: int) -> tuple[bool, dict]:
    """Random-sample sanity check on produced PNGs.

    Returns ``(all_ok, stats)``. Stats reports file counts, label-value
    spread, and depth-range spread across the sampled PNGs.
    """
    rng = random.Random(0)
    pop = [(s, f) for s, n in sets_found.items() for f in range(n)]
    rng.shuffle(pop)
    sample = pop[:50]  # 50 random sources is enough to flag systemic issues

    label_values: set[int] = set()
    depth_min = 1 << 30
    depth_max = 0
    bad: list[str] = []
    for sid, fid in sample:
        split = split_of(sid)
        base = OUTPUT_ROOT / split / f"{sid:03d}"
        ai = rng.randint(0, num_frames - 1)
        vn = rng.choice(VIEW_NAMES)
        rgb_path = base / "rgb" / f"{fid:04d}_{ai:02d}_{vn}.png"
        depth_path = base / "depth" / f"{fid:04d}_{ai:02d}_{vn}.png"
        lbl_path = base / "label" / f"{fid:04d}_{ai:02d}_{vn}.png"
        if not (rgb_path.exists() and depth_path.exists() and lbl_path.exists()):
            bad.append(f"missing: set{sid:03d}/{fid:04d}_{ai:02d}_{vn}")
            continue
        rgb = cv2.imread(str(rgb_path))
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        lbl = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or lbl is None:
            bad.append(f"unreadable: set{sid:03d}/{fid:04d}_{ai:02d}_{vn}")
            continue
        if rgb.shape != (480, 640, 3):
            bad.append(f"rgb shape {rgb.shape}: set{sid:03d}/{fid:04d}_{ai:02d}_{vn}")
        if depth.shape != (480, 640) or depth.dtype != np.uint16:
            bad.append(f"depth shape/dtype: set{sid:03d}/{fid:04d}_{ai:02d}_{vn}")
        if lbl.shape != (480, 640) or lbl.dtype != np.uint8:
            bad.append(f"label shape/dtype: set{sid:03d}/{fid:04d}_{ai:02d}_{vn}")
        if int(lbl.max()) > 5:
            bad.append(f"label > 5: set{sid:03d}/{fid:04d}_{ai:02d}_{vn} max={int(lbl.max())}")
        label_values.update(np.unique(lbl).tolist())
        valid_depth = depth[depth > 0]
        if valid_depth.size:
            depth_min = min(depth_min, int(valid_depth.min()))
            depth_max = max(depth_max, int(valid_depth.max()))

    stats = {
        "samples_checked": len(sample),
        "issues": len(bad),
        "label_values_seen": sorted(label_values),
        "depth_range_mm": [depth_min if depth_min < (1 << 30) else None, depth_max],
    }
    return len(bad) == 0, stats


def file_count_check(sets_found: dict[int, int], num_frames: int) -> dict:
    """Count rgb/depth/label PNGs and the per-source label .npy."""
    n_views = len(VIEW_NAMES)
    expected_rgb = sum(sets_found.values()) * num_frames * n_views
    actual_rgb = actual_depth = actual_label_png = actual_label_npy = 0
    for sid, n in sets_found.items():
        split = split_of(sid)
        base = OUTPUT_ROOT / split / f"{sid:03d}"
        if not base.is_dir():
            continue
        for sub, target in [("rgb", "actual_rgb"),
                            ("depth", "actual_depth"),
                            ("label", "actual_label_png")]:
            d = base / sub
            if d.is_dir():
                cnt = len(list(d.glob("*.png")))
                if target == "actual_rgb":
                    actual_rgb += cnt
                elif target == "actual_depth":
                    actual_depth += cnt
                else:
                    actual_label_png += cnt
        npy_dir = base / "labels"
        if npy_dir.is_dir():
            actual_label_npy += len(list(npy_dir.glob("*.npy")))
    return {
        "expected_per_channel": expected_rgb,
        "rgb": actual_rgb,
        "depth": actual_depth,
        "label_png": actual_label_png,
        "label_npy_per_source_expected": sum(sets_found.values()),
        "label_npy_per_source_actual": actual_label_npy,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the full PointWire dataset to RGB-D videos with the Phase 4 pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workers", type=int, default=8,
                        help="Worker processes for the multiprocess pool.")
    parser.add_argument("--num-frames", type=int, default=20,
                        help="Animation frames per source frame.")
    parser.add_argument("--max-angle", type=float, default=25.0,
                        help="Max joint rotation in degrees during animation.")
    parser.add_argument("--sets", type=int, nargs="+", default=None,
                        help="Restrict to specific set ids (default: all discovered).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run pre-flight + plan + estimate, write nothing.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="After a successful render, run prepare_dformer_data.py "
                             "to refresh the DFormer training cache.")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip the post-render random-sample validation.")
    args = parser.parse_args()

    log_dir = PROJECT_ROOT / "results" / "render_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"render_phase4_{datetime.now():%Y%m%d_%H%M%S}.log"
    sys.stdout = _Tee(log_path)

    print(f"=== PointWire → RGB-D video dataset (Phase 4) ===")
    print(f"Log file:   {log_path}")
    print(f"Workers:    {args.workers}")
    print(f"Anim/src:   {args.num_frames}")
    print(f"Max angle:  {args.max_angle}°")
    print(f"Sets:       {'ALL' if args.sets is None else args.sets}")
    print(f"Dry run:    {args.dry_run}")
    print()

    if not preflight():
        print("\nPre-flight FAILED. Fix the missing libraries before re-running.")
        return 1

    only_sets = set(args.sets) if args.sets is not None else None
    sets_found = discover_sets(only_sets)
    if not sets_found:
        print("\nNo source frames discovered. Is data/set2/ populated?")
        return 1

    total_src = sum(sets_found.values())
    n_train = sum(n for s, n in sets_found.items() if split_of(s) == "train")
    n_val = sum(n for s, n in sets_found.items() if split_of(s) == "val")
    n_test = sum(n for s, n in sets_found.items() if split_of(s) == "test")
    n_pngs = total_src * args.num_frames * len(VIEW_NAMES) * 3  # rgb + depth + label
    n_pcl = total_src * args.num_frames

    print()
    print("Plan:")
    print(f"  sets:               {len(sets_found)} "
          f"(train={n_train:,} val={n_val:,} test={n_test:,})")
    print(f"  source frames:      {total_src:,}")
    print(f"  animation frames:   {total_src * args.num_frames:,}")
    print(f"  PNG triples:        {total_src * args.num_frames * len(VIEW_NAMES):,}")
    print(f"  total PNG files:    {n_pngs:,}")
    print(f"  point clouds:       {n_pcl:,}")
    # Phase 4 measured: ≈ 25-27 s/source for 5 anim × 6 views (single proc),
    # i.e. ≈ 5 s per anim-frame (where one anim-frame = all 6 views).
    # Linear in num_frames per source, divided by worker count.
    per_src_s = 5.0 * args.num_frames  # 5 s × num_anim_frames × 6 views
    est_h = total_src * per_src_s / args.workers / 3600
    print(f"  est wall:           {est_h:.1f} h on {args.workers} workers")
    print()

    if args.dry_run:
        print("[--dry-run] No files written.")
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    work = build_work_list(sets_found, args.num_frames, args.max_angle, OUTPUT_ROOT)

    print(f"Rendering {len(work):,} source frames...")
    print()
    t0 = time.time()
    done = 0
    n_ok = 0
    n_skip = 0
    errors: list[tuple[int, int, str]] = []

    with Pool(processes=args.workers) as pool:
        for set_id, frame_id, status, _elapsed in pool.imap_unordered(
                convert_one_video, work, chunksize=4):
            done += 1
            if status == "ok":
                n_ok += 1
            elif status == "skipped":
                n_skip += 1
            else:
                errors.append((set_id, frame_id, status))

            if done % 25 == 0 or done == total_src or status.startswith("error"):
                wall = time.time() - t0
                rate = wall / max(done, 1)
                remaining = rate * (total_src - done)
                if remaining >= 3600:
                    eta = f"{remaining / 3600:.1f}h"
                elif remaining >= 60:
                    eta = f"{int(remaining // 60)}m{int(remaining % 60):02d}s"
                else:
                    eta = f"{int(remaining)}s"
                pct = 100 * done / total_src
                print(f"  [{done:5d}/{total_src}] {pct:5.1f}%  "
                      f"ok={n_ok} skip={n_skip} err={len(errors)}  ETA {eta}",
                      flush=True)

    wall_total = time.time() - t0
    h, m = divmod(wall_total, 3600)
    m, s = divmod(m, 60)

    print()
    print("Render complete.")
    print(f"  wall:          {int(h)}h {int(m)}m {int(s)}s")
    print(f"  converted:     {n_ok:,}")
    print(f"  skipped:       {n_skip:,}")
    print(f"  errors:        {len(errors)}")

    if errors:
        print()
        print("First 10 errors:")
        for sid, fid, msg in errors[:10]:
            print(f"  set {sid:03d} src {fid:04d}: {msg.splitlines()[0]}")

    stats = {
        "total_source_frames": total_src,
        "converted": n_ok,
        "skipped": n_skip,
        "errors": len(errors),
        "anim_frames_per_source": args.num_frames,
        "views": len(VIEW_NAMES),
        "total_rgb_depth_pairs": n_ok * args.num_frames * len(VIEW_NAMES),
        "wall_seconds": round(wall_total, 1),
        "bg_n_points": BG_N_POINTS,
        "phase": "Phase 4 (real-object 3D bg + 2D photo backdrop)",
    }
    meta_path = write_metadata(sets_found, args.num_frames, args.max_angle, stats)
    print(f"  metadata:      {meta_path}")

    print()
    print("File-count check:")
    counts = file_count_check(sets_found, args.num_frames)
    expected = counts["expected_per_channel"]
    for k in ("rgb", "depth", "label_png"):
        delta = counts[k] - expected
        flag = "OK" if delta == 0 else f"WARN ({delta:+d})"
        print(f"  {k:14s} {counts[k]:>10,}  vs expected {expected:>10,}  [{flag}]")
    print(f"  label_npy      {counts['label_npy_per_source_actual']:>10,}  "
          f"vs expected {counts['label_npy_per_source_expected']:>10,}")

    if not args.skip_validation:
        print()
        print("Random-sample validation (50 PNGs):")
        ok, vstats = post_render_validate(sets_found, args.num_frames)
        flag = "OK" if ok else "WARN"
        print(f"  [{flag}] issues: {vstats['issues']}")
        print(f"        label values seen: {vstats['label_values_seen']}")
        print(f"        depth range mm:    {vstats['depth_range_mm']}")

    if args.rebuild_cache:
        if errors:
            print()
            print("Skipping DFormer cache rebuild — render had errors.")
        else:
            print()
            print("Rebuilding DFormer training cache...")
            cache_script = PROJECT_ROOT / "src" / "prepare_dformer_data.py"
            if not cache_script.is_file():
                print(f"  WARN: {cache_script} not found, skipping.")
            else:
                rc = subprocess.call(
                    [sys.executable, str(cache_script)],
                    cwd=str(PROJECT_ROOT),
                )
                print(f"  prepare_dformer_data.py exit code: {rc}")

    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
