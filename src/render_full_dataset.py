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
    ABL_VARIANT, BG_N_POINTS, DATASET_MODE, DATA_ROOT, OUTPUT_ROOT, VIEW_NAMES,
    _get_background_library, _get_object_library, _get_texture_library,
    build_work_list, convert_one_video, split_of, write_metadata,
)
from texture_mapping import PHASE9_DROP_CATEGORIES  # noqa: E402


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
    """Check that all three asset libraries are loadable + non-trivial.

    Thresholds vary by ``DATASET_MODE``. Phase 9 explicitly drops the
    wire-shaped / hand asset families and so doesn't gate on those counts;
    instead it gates on the *retained* (non-dropped) library size.
    """
    print(f"Pre-flight checks (mode={DATASET_MODE}):")
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
    photo_min = 8
    print(f"  [{'OK' if len(photos) >= photo_min else 'FAIL'}] "
          f"background photos: {len(photos)} (expected ≥ {photo_min})")
    ok &= len(photos) >= photo_min

    objs = _get_object_library()
    n_obj_total = len(objs)

    if DATASET_MODE == "phase9":
        retained = [o for o in objs
                    if o.get("category") not in PHASE9_DROP_CATEGORIES]
        n_retained = len(retained)
        retained_min = 25
        print(f"  [{'OK' if n_retained >= retained_min else 'FAIL'}] "
              f"non-wire / non-hand object PCLs: {n_retained} "
              f"(expected ≥ {retained_min}; drop categories: "
              f"{sorted(PHASE9_DROP_CATEGORIES)})")
        ok &= n_retained >= retained_min
        from collections import Counter
        cats = Counter(o.get("category", "?") for o in retained)
        print(f"        retained categories: {dict(cats)}")
    elif DATASET_MODE == "phase12":
        from collections import Counter
        phase4_originals = [
            o for o in objs
            if o.get("category") not in {
                "hand", "gripper", "arm",
                "negative_wire_like", "rope", "clutter",
            }
        ]
        n_phase4 = len(phase4_originals)
        phase4_min = 15
        print(f"  [{'OK' if n_phase4 >= phase4_min else 'FAIL'}] "
              f"Phase 4 Polyhaven CC0 mesh-derived PCLs: {n_phase4} "
              f"(expected ≥ {phase4_min}; expected exactly 21 originals)")
        ok &= n_phase4 >= phase4_min
        cats = Counter(o.get("category", "?") for o in phase4_originals)
        print(f"        Phase 4 categories: {dict(cats)}")
        # Phase 12 enriches the 2D photographic backdrop. Phase 11 already
        # expanded the pool to 80; Phase 12 H2 targets ≥ 500.
        photo_min_phase12 = 11
        photo_target_phase12 = 50
        if len(photos) >= photo_target_phase12:
            flag = "OK"
        elif len(photos) >= photo_min_phase12:
            flag = "WARN"
        else:
            flag = "FAIL"
        print(f"  [{flag}] backdrop photos: {len(photos)} "
              f"(target ≥ {photo_target_phase12}; spec floor "
              f"{photo_min_phase12}; Phase 12 H2 aims for ≥ 500).")
        ok &= len(photos) >= photo_min_phase12
    elif DATASET_MODE == "phase13":
        from collections import Counter
        phase4_originals = [
            o for o in objs
            if o.get("category") not in {
                "hand", "gripper", "arm",
                "negative_wire_like", "rope", "clutter",
            }
        ]
        n_phase4 = len(phase4_originals)
        phase4_min = 15
        print(f"  [{'OK' if n_phase4 >= phase4_min else 'FAIL'}] "
              f"Phase 4 Polyhaven CC0 mesh-derived PCLs: {n_phase4} "
              f"(expected ≥ {phase4_min}; expected exactly 21 originals)")
        ok &= n_phase4 >= phase4_min
        cats = Counter(o.get("category", "?") for o in phase4_originals)
        print(f"        Phase 4 categories: {dict(cats)}")
        # Phase 13 keeps the ORIGINAL 11-photo backdrop (Phase 12 falsified
        # backdrop-pool enrichment); the levers are lighting + object colour.
        photo_exact_phase13 = 11
        flag = "OK" if len(photos) == photo_exact_phase13 else "WARN"
        print(f"  [{flag}] backdrop photos: {len(photos)} "
              f"(Phase 13 expects exactly {photo_exact_phase13} — the original "
              "Phase 4 pool via KIAT_BG_DIR=data/textures/backgrounds_p4orig11).")
        ok &= len(photos) >= photo_exact_phase13
    elif DATASET_MODE == "phase11":
        from collections import Counter
        clutter = [o for o in objs if o.get("category") == "clutter"]
        hands = [o for o in objs if o.get("category") == "hand"]
        n_clutter = len(clutter)
        n_hands = len(hands)
        clutter_min = 15
        hands_min = 20
        print(f"  [{'OK' if n_clutter >= clutter_min else 'FAIL'}] "
              f"Phase 4 clutter PCLs (for background): {n_clutter} "
              f"(expected ≥ {clutter_min})")
        print(f"  [{'OK' if n_hands >= hands_min else 'FAIL'}] "
              f"hand PCLs (for phase11 foreground): {n_hands} "
              f"(expected ≥ {hands_min})")
        ok &= n_clutter >= clutter_min
        ok &= n_hands >= hands_min
        cats = Counter(o.get("category", "?") for o in clutter)
        print(f"        clutter categories: {dict(cats)}")
        # Phase 11 spec asks for the original 11 Phase 4 photos plus
        # +40-80 new CC0 photos. The fetcher script
        # scripts/fetch_phase11_backdrops.py expands the library to ~80 by
        # default; the soft lower bound here is "Phase 4 baseline" (11).
        photo_min_phase11 = 11
        photo_target_phase11 = 50
        if len(photos) >= photo_target_phase11:
            flag = "OK"
        elif len(photos) >= photo_min_phase11:
            flag = "WARN"
        else:
            flag = "FAIL"
        print(f"  [{flag}] backdrop photos: {len(photos)} "
              f"(target ≥ {photo_target_phase11}; spec floor {photo_min_phase11}; "
              "run scripts/fetch_phase11_backdrops.py to grow the library)")
        ok &= len(photos) >= photo_min_phase11
    elif ABL_VARIANT is not None:
        # Phase 10 ablation: 'objects' variant needs the Phase 9 retained
        # library (~25+); other variants use only the 21 'clutter' entries.
        from collections import Counter
        clutter = [o for o in objs if o.get("category") == "clutter"]
        retained = [o for o in objs
                    if o.get("category") not in PHASE9_DROP_CATEGORIES]
        print(f"  [{'OK' if len(clutter) >= 15 else 'FAIL'}] "
              f"Phase 4 clutter PCLs: {len(clutter)} (expected ≥ 15)")
        ok &= len(clutter) >= 15
        if ABL_VARIANT == "objects":
            print(f"  [{'OK' if len(retained) >= 25 else 'FAIL'}] "
                  f"Phase 9 retained PCLs (for 'objects' variant): "
                  f"{len(retained)} (expected ≥ 25)")
            ok &= len(retained) >= 25
        cats = Counter(o.get("category", "?") for o in clutter)
        print(f"        clutter categories: {dict(cats)}")
        print(f"  ablation variant: {ABL_VARIANT}")
    else:
        print(f"  [{'OK' if n_obj_total >= 200 else 'FAIL'}] 3D real-object PCLs: "
              f"{n_obj_total} (expected ≥ 200 for v2)")
        ok &= n_obj_total >= 200
        hands = sum(1 for o in objs if o.get("category") == "hand")
        grippers = sum(1 for o in objs if o.get("category") == "gripper")
        arms = sum(1 for o in objs if o.get("category") == "arm")
        cables = sum(1 for o in objs
                     if o.get("category") == "negative_wire_like")
        graspables = sum(1 for o in objs if o.get("graspable_on_wire"))
        print(f"  [{'OK' if hands >= 30 else 'FAIL'}] hand variants: "
              f"{hands} (expected ≥ 30)")
        print(f"  [{'OK' if grippers + arms >= 18 else 'FAIL'}] "
              f"gripper+arm variants: {grippers}+{arms}={grippers + arms} (expected ≥ 18)")
        print(f"  [{'OK' if cables >= 30 else 'FAIL'}] negative-wire-like "
              f"clutter: {cables} (expected ≥ 30)")
        print(f"  [{'OK' if graspables >= 30 else 'FAIL'}] graspable-on-wire "
              f"objects: {graspables} (expected ≥ 30)")
        ok &= hands >= 30
        ok &= grippers + arms >= 18
        ok &= cables >= 30
        ok &= graspables >= 30

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
                         num_frames: int,
                         src_stride: int = 1,
                         only_splits: tuple[str, ...] | None = None,
                         ) -> tuple[bool, dict]:
    """Random-sample sanity check on produced PNGs.

    Returns ``(all_ok, stats)``. Stats reports file counts, label-value
    spread, and depth-range spread across the sampled PNGs.
    """
    rng = random.Random(0)
    pop = [
        (s, f) for s, n in sets_found.items()
        for f in range(0, n, max(1, src_stride))
        if only_splits is None or split_of(s) in only_splits
    ]
    rng.shuffle(pop)
    sample = pop[:50]  # 50 random sources is enough to flag systemic issues

    label_values: set[int] = set()
    label3_values: set[int] = set()
    depth_min = 1 << 30
    depth_max = 0
    bad: list[str] = []
    check_label3 = (DATASET_MODE == "phase11")
    for sid, fid in sample:
        split = split_of(sid)
        base = OUTPUT_ROOT / split / f"{sid:03d}"
        ai = rng.randint(0, num_frames - 1)
        vn = rng.choice(VIEW_NAMES)
        rgb_path = base / "rgb" / f"{fid:04d}_{ai:02d}_{vn}.png"
        depth_path = base / "depth" / f"{fid:04d}_{ai:02d}_{vn}.png"
        lbl_path = base / "label" / f"{fid:04d}_{ai:02d}_{vn}.png"
        lbl3_path = base / "label3" / f"{fid:04d}_{ai:02d}_{vn}.png"
        if not (rgb_path.exists() and depth_path.exists() and lbl_path.exists()):
            bad.append(f"missing: set{sid:03d}/{fid:04d}_{ai:02d}_{vn}")
            continue
        if check_label3 and not lbl3_path.exists():
            bad.append(f"missing label3: set{sid:03d}/{fid:04d}_{ai:02d}_{vn}")
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
        if check_label3:
            lbl3 = cv2.imread(str(lbl3_path), cv2.IMREAD_UNCHANGED)
            if lbl3 is None:
                bad.append(f"unreadable label3: set{sid:03d}/"
                           f"{fid:04d}_{ai:02d}_{vn}")
            else:
                if lbl3.shape != (480, 640) or lbl3.dtype != np.uint8:
                    bad.append(f"label3 shape/dtype: set{sid:03d}/"
                               f"{fid:04d}_{ai:02d}_{vn}")
                if int(lbl3.max()) > 2:
                    bad.append(f"label3 > 2: set{sid:03d}/"
                               f"{fid:04d}_{ai:02d}_{vn} max={int(lbl3.max())}")
                # Cross-check label3 against label: wherever label > 0 (any
                # harness pixel), label3 must equal 1 (wire). This catches
                # bugs in the post-hoc derivation.
                harness_mask = (lbl > 0)
                if harness_mask.any():
                    mismatch = int(np.sum(harness_mask & (lbl3 != 1)))
                    if mismatch > 0:
                        bad.append(f"label3 mismatches harness ({mismatch} "
                                   f"px): set{sid:03d}/"
                                   f"{fid:04d}_{ai:02d}_{vn}")
                label3_values.update(np.unique(lbl3).tolist())
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
    if check_label3:
        stats["label3_values_seen"] = sorted(label3_values)
    return len(bad) == 0, stats


def file_count_check(sets_found: dict[int, int], num_frames: int,
                     src_stride: int = 1,
                     only_splits: tuple[str, ...] | None = None) -> dict:
    """Count rgb/depth/label PNGs and the per-source label .npy.

    For Phase 11, also counts label3 PNGs (the 3-way per-pixel label).
    """
    n_views = len(VIEW_NAMES)
    rendered_sources = sum(
        len(range(0, n, max(1, src_stride)))
        for s, n in sets_found.items()
        if only_splits is None or split_of(s) in only_splits
    )
    expected_rgb = rendered_sources * num_frames * n_views
    actual_rgb = actual_depth = actual_label_png = actual_label_npy = 0
    actual_label3_png = 0
    for sid, n in sets_found.items():
        split = split_of(sid)
        base = OUTPUT_ROOT / split / f"{sid:03d}"
        if not base.is_dir():
            continue
        for sub, target in [("rgb", "actual_rgb"),
                            ("depth", "actual_depth"),
                            ("label", "actual_label_png"),
                            ("label3", "actual_label3_png")]:
            d = base / sub
            if d.is_dir():
                cnt = len(list(d.glob("*.png")))
                if target == "actual_rgb":
                    actual_rgb += cnt
                elif target == "actual_depth":
                    actual_depth += cnt
                elif target == "actual_label_png":
                    actual_label_png += cnt
                else:
                    actual_label3_png += cnt
        npy_dir = base / "labels"
        if npy_dir.is_dir():
            actual_label_npy += len(list(npy_dir.glob("*.npy")))
    out = {
        "expected_per_channel": expected_rgb,
        "rgb": actual_rgb,
        "depth": actual_depth,
        "label_png": actual_label_png,
        "label_npy_per_source_expected": rendered_sources,
        "label_npy_per_source_actual": actual_label_npy,
    }
    if DATASET_MODE == "phase11":
        out["label3_png"] = actual_label3_png
    return out


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the full PointWire dataset to RGB-D videos with the Phase 4 pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    def _positive_int(s: str) -> int:
        n = int(s)
        if n < 1:
            raise argparse.ArgumentTypeError(f"must be ≥ 1, got {n}")
        return n
    parser.add_argument("--workers", type=_positive_int, default=8,
                        help="Worker processes for the multiprocess pool (≥ 1).")
    parser.add_argument("--num-frames", type=int,
                        default=(1 if (DATASET_MODE in ("phase9", "phase11",
                                                        "phase12", "phase13")
                                       or ABL_VARIANT is not None) else 20),
                        help="Animation frames per source "
                             "(phase9 / phase11 / phase12 / phase13 / "
                             "ablation force 1).")
    parser.add_argument("--max-angle", type=float, default=25.0,
                        help="Max joint rotation in degrees during animation.")
    parser.add_argument("--sets", type=int, nargs="+", default=None,
                        help="Restrict to specific set ids (default: all discovered).")
    # Phase 10 ablation rendering targets ~1500 train + 200 val per variant.
    # 1500 train / 21 sets / 6 views ≈ 12 sources/set ⇒ src_stride ≈ 25.
    # 200 val / 3 sets / 6 views ≈ 12 sources/set ⇒ same stride.
    # Phase 11 matches Phase 7 / 9 / v3 scale: src_stride 5 = 21 × 60 × 6 =
    # 7,560 train + 3 × 60 × 6 = 1,080 val triples.
    if DATASET_MODE == "phase9":
        _default_stride = 5
    elif DATASET_MODE == "phase11":
        _default_stride = 5
    elif DATASET_MODE in ("phase12", "phase13"):
        _default_stride = 5
    elif ABL_VARIANT is not None:
        _default_stride = 25
    else:
        _default_stride = 1
    parser.add_argument("--src-stride", type=_positive_int,
                        default=_default_stride,
                        help="Render every Nth source per set "
                             "(phase9 / phase11 / phase12 default 5; "
                             "ablation default 25 → ≈12 sources/set "
                             "⇒ ~1,512 train + ~216 val triples).")
    parser.add_argument("--only-splits", type=str, nargs="+",
                        default=(["train", "val"]
                                 if (DATASET_MODE in ("phase9", "phase11",
                                                      "phase12", "phase13")
                                     or ABL_VARIANT is not None)
                                 else None),
                        choices=["train", "val", "test"],
                        help="Restrict rendering to specific splits "
                             "(phase9 / phase11 / phase12 / phase13 / "
                             "ablation default: train val).")
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
    log_path = log_dir / (
        f"render_{DATASET_MODE}_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    sys.stdout = _Tee(log_path)

    print(f"=== PointWire → RGB-D video dataset ({DATASET_MODE}) ===")
    print(f"Log file:    {log_path}")
    print(f"Workers:     {args.workers}")
    print(f"Anim/src:    {args.num_frames}")
    print(f"Max angle:   {args.max_angle}°")
    print(f"Sets:        {'ALL' if args.sets is None else args.sets}")
    print(f"Src stride:  {args.src_stride}")
    print(f"Only splits: {args.only_splits if args.only_splits else 'all'}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Dry run:     {args.dry_run}")
    print()

    if not preflight():
        print("\nPre-flight FAILED. Fix the missing libraries before re-running.")
        return 1

    only_sets = set(args.sets) if args.sets is not None else None
    sets_found = discover_sets(only_sets)
    if not sets_found:
        print("\nNo source frames discovered. Is data/set2/ populated?")
        return 1

    only_splits = tuple(args.only_splits) if args.only_splits is not None else None

    work = build_work_list(
        sets_found, args.num_frames, args.max_angle, OUTPUT_ROOT,
        src_stride=args.src_stride,
        only_splits=only_splits,
    )

    # Count work items by split.
    by_split = {"train": 0, "val": 0, "test": 0}
    for sid, _fid, _nf, _ma, _ot in work:
        by_split[split_of(sid)] += 1
    total_src = len(work)
    n_pngs = total_src * args.num_frames * len(VIEW_NAMES) * 3
    n_pcl = total_src * args.num_frames

    print()
    print("Plan:")
    print(f"  mode:                {DATASET_MODE}")
    print(f"  sets discovered:     {len(sets_found)}")
    print(f"  only_splits:         {only_splits if only_splits else 'all'}")
    print(f"  src_stride:          {args.src_stride}")
    print(f"  rendered sources:    {total_src:,} "
          f"(train={by_split['train']:,} "
          f"val={by_split['val']:,} test={by_split['test']:,})")
    print(f"  animation frames:    {total_src * args.num_frames:,}")
    print(f"  PNG triples:         {total_src * args.num_frames * len(VIEW_NAMES):,}")
    print(f"  total PNG files:     {n_pngs:,}")
    print(f"  point clouds:        {n_pcl:,}")
    # phase9: ~37 s/source single-process (smoke measured 188s for 5 srcs;
    # most time is rasterising the ~250k-point combined PCL across 6 views).
    # v2 / phase11: ~13 s per anim-frame * num_frames (Phase 11 has a much
    # smaller fg point budget than v2 but the same 60k bg, dominated by
    # the rasterisation pass).
    if DATASET_MODE == "phase9":
        per_src_s = 37.0 * args.num_frames
    elif DATASET_MODE == "phase11":
        per_src_s = 13.0 * max(1, args.num_frames)
    else:
        per_src_s = 13.0 * args.num_frames
    est_h = total_src * per_src_s / args.workers / 3600
    print(f"  est wall:            {est_h:.2f} h on {args.workers} workers")
    print()

    if args.dry_run:
        print("[--dry-run] No files written.")
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

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
        "src_stride": args.src_stride,
        "only_splits": list(only_splits) if only_splits else None,
        "phase": (
            "Phase 9 (5-wall enclosure + randomised camera + drop wire-shaped/hand negatives)"
            if DATASET_MODE == "phase9"
            else (
                "Phase 11 (Phase 4 baseline + at-most-one off-wire hand "
                "foreground, p_hand=KIAT_PHASE11_P_HAND)"
                if DATASET_MODE == "phase11"
                else "Phase 4 / v2 (real-object 3D bg + 2D photo backdrop)"
            )
        ),
    }
    meta_path = write_metadata(sets_found, args.num_frames, args.max_angle, stats)
    print(f"  metadata:      {meta_path}")

    print()
    print("File-count check:")
    counts = file_count_check(sets_found, args.num_frames,
                              src_stride=args.src_stride,
                              only_splits=only_splits)
    expected = counts["expected_per_channel"]
    channels = ["rgb", "depth", "label_png"]
    if "label3_png" in counts:
        channels.append("label3_png")
    for k in channels:
        delta = counts[k] - expected
        flag = "OK" if delta == 0 else f"WARN ({delta:+d})"
        print(f"  {k:14s} {counts[k]:>10,}  vs expected {expected:>10,}  [{flag}]")
    print(f"  label_npy      {counts['label_npy_per_source_actual']:>10,}  "
          f"vs expected {counts['label_npy_per_source_expected']:>10,}")

    if not args.skip_validation:
        print()
        print("Random-sample validation (50 PNGs):")
        ok, vstats = post_render_validate(sets_found, args.num_frames,
                                          src_stride=args.src_stride,
                                          only_splits=only_splits)
        flag = "OK" if ok else "WARN"
        print(f"  [{flag}] issues: {vstats['issues']}")
        print(f"        label values seen: {vstats['label_values_seen']}")
        if "label3_values_seen" in vstats:
            print(f"        label3 values seen:{vstats['label3_values_seen']}")
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
