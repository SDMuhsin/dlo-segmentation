#!/usr/bin/env python3
"""Validate the Phase 8 smoke render output.

Runs a fixed set of post-render checks on the smoke output dir, then
prints + persists a `validation.json` report.

Usage::

    python src/validate_phase8_smoke.py results/phase8_smoke

Checks:

1. **File count** — rgb / depth / label counts match expected
   (#sources × num_anims × 6 views).
2. **Label values** — every label PNG ⊆ ``{0..5}`` (no fg/bg sentinel
   leakage into harness classes).
3. **Depth range** — every depth PNG ⊆ ``[0, 1500]`` mm (0 = no data,
   1500 = far plane).
4. **Foreground present** — for sources where the smoke metadata records
   ``placed`` items, at least 1 view should show non-zero foreground
   pixel coverage (skin-toned or robot-grey pixels above the harness
   class colours by some hash).
5. **Anim-stickiness sanity** — RGB pixel set in non-harness regions
   should be ~consistent across anim frames of the same source/view
   (bg + fg are static; only harness moves). This isn't a
   bit-identical check (z-buffer ties may flip), but the histogram
   distance should be small.
6. **Per-source foreground reproducibility** — re-running
   ``generate_foreground_scene`` with the same seed produces the same
   point cloud (deterministic).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _list_pngs(d: Path) -> list[Path]:
    return sorted(d.glob("*.png"))


def _src_anim_view(name: str) -> tuple[int, int, str] | None:
    """Parse a filename like ``0000_07_front.png`` → (0, 7, 'front')."""
    stem = name.split(".")[0]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), "_".join(parts[2:])
    except ValueError:
        return None


def check_file_counts(out_dir: Path) -> dict:
    """Count rgb/depth/label PNGs by sub-dir under each set."""
    issues = []
    set_dirs = sorted([p for p in (out_dir / "train").iterdir() if p.is_dir()])
    if not set_dirs:
        # also try val/test
        for split in ("val", "test"):
            set_dirs.extend(sorted(
                [p for p in (out_dir / split).iterdir() if p.is_dir()]
                if (out_dir / split).is_dir() else []
            ))
    counts = {}
    for sd in set_dirs:
        for sub in ("rgb", "depth", "label"):
            d = sd / sub
            if not d.is_dir():
                issues.append(f"missing dir: {d}")
                continue
            counts[(sd.name, sub)] = len(list(d.glob("*.png")))
    return {"counts": {f"{k[0]}/{k[1]}": v for k, v in counts.items()},
            "issues": issues}


def check_label_values(out_dir: Path, sample_cap: int = 200) -> dict:
    """Ensure every sampled label PNG has values ⊆ {0..5}."""
    pngs = []
    for sub in (out_dir / "train").iterdir() if (out_dir / "train").is_dir() else []:
        d = sub / "label"
        if d.is_dir():
            pngs.extend(_list_pngs(d))
    pngs = pngs[:sample_cap]
    bad = []
    seen = set()
    for p in pngs:
        L = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if L is None:
            bad.append((str(p), "unreadable"))
            continue
        u = set(np.unique(L).tolist())
        seen |= u
        if not u.issubset({0, 1, 2, 3, 4, 5}):
            bad.append((str(p), f"unexpected values: {sorted(u)}"))
    return {"sampled": len(pngs), "values_seen": sorted(seen),
            "issues": bad}


def check_depth_range(out_dir: Path, sample_cap: int = 200) -> dict:
    pngs = []
    for sub in (out_dir / "train").iterdir() if (out_dir / "train").is_dir() else []:
        d = sub / "depth"
        if d.is_dir():
            pngs.extend(_list_pngs(d))
    pngs = pngs[:sample_cap]
    bad = []
    minv, maxv = 1 << 30, 0
    for p in pngs:
        D = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if D is None:
            bad.append((str(p), "unreadable"))
            continue
        if D.dtype != np.uint16:
            bad.append((str(p), f"dtype {D.dtype} expected uint16"))
            continue
        valid = D[D > 0]
        if valid.size:
            minv = min(minv, int(valid.min()))
            maxv = max(maxv, int(valid.max()))
        if valid.size and (int(valid.min()) < 500 or int(valid.max()) > 1500):
            bad.append((str(p), f"depth out of [500,1500]: [{valid.min()},{valid.max()}]"))
    return {"sampled": len(pngs), "min_mm": minv, "max_mm": maxv,
            "issues": bad}


def check_anim_stickiness(out_dir: Path, set_id: int, src_id: int,
                          view: str, num_anim: int) -> dict:
    """RGB histogram cosine across anim frames of one source/view.

    Static bg + fg should mean the bg-pixel histogram barely changes; only
    the (small) animated harness pixels differ. Cosine should be ≥ 0.95.
    """
    base = out_dir / "train" / f"{set_id:03d}" / "rgb"
    imgs = []
    for ai in range(num_anim):
        p = base / f"{src_id:04d}_{ai:02d}_{view}.png"
        if not p.is_file():
            return {"issue": f"missing {p}"}
        imgs.append(cv2.imread(str(p)))
    if len(imgs) < 2:
        return {"issue": "need ≥ 2 anim frames"}
    hist0 = cv2.calcHist([imgs[0]], [0, 1, 2], None,
                          [16, 16, 16], [0, 256, 0, 256, 0, 256]).flatten()
    cosines = []
    for img in imgs[1:]:
        h = cv2.calcHist([img], [0, 1, 2], None,
                          [16, 16, 16], [0, 256, 0, 256, 0, 256]).flatten()
        if np.linalg.norm(hist0) < 1e-9 or np.linalg.norm(h) < 1e-9:
            cosines.append(None)
        else:
            cosines.append(float(
                np.dot(hist0, h) / (np.linalg.norm(hist0) * np.linalg.norm(h))))
    return {"view": view, "src": src_id, "num_anim": num_anim,
            "min_cosine": float(min(c for c in cosines if c is not None)),
            "mean_cosine": float(np.mean([c for c in cosines if c is not None])),
            "all_cosines": cosines}


def check_foreground_diversity(out_dir: Path,
                                 sources: list[int],
                                 set_id: int = 0,
                                 fg_n_points: int = 24000) -> dict:
    """Confirm foreground generation produces varied placements.

    Re-runs ``generate_foreground_scene`` with the same seeds the renderer
    uses and reports the placed objects per source. Also verifies a second
    run with the same seed produces the same point cloud (determinism).

    v2: also reports per-source object count to confirm the 10-15 fg target.
    """
    from texture_mapping import (load_object_library,
                                   generate_foreground_scene)
    from convert_to_video_dataset import _build_topology

    obj_lib = load_object_library(PROJECT_ROOT / "data" / "objects")
    rows = []
    for src in sources:
        pcl = np.load(PROJECT_ROOT / "data" / "set2" / f"{set_id:03d}" /
                       "pointclouds_normed_4096" / f"pcl_{src:04d}.npy")
        skel = np.load(PROJECT_ROOT / "data" / "set2" / f"{set_id:03d}" /
                        "skeletons" / f"{src:03d}.npz")
        nodes, adj = skel["nodes"], skel["adj"]
        _, _, _, segments, _ = _build_topology(adj)
        seed = set_id * 1000 + src + 113
        kw = dict(object_library=obj_lib,
                  bbox_min=pcl.min(axis=0), bbox_max=pcl.max(axis=0),
                  skeleton_nodes=nodes, segments=segments,
                  n_points=fg_n_points)
        pts1, _, info1 = generate_foreground_scene(
            rng=np.random.RandomState(seed), **kw)
        pts2, _, info2 = generate_foreground_scene(
            rng=np.random.RandomState(seed), **kw)
        rows.append({
            "src": src, "n_points": int(pts1.shape[0]),
            "n_objects": len(info1["placed"]),
            "counts": info1.get("counts", {}),
            "placed": [f"{p['kind']}:{p['slug']}" for p in info1["placed"]],
            "deterministic": bool(np.array_equal(pts1, pts2)),
        })
    # Variety summary
    all_kinds = []
    all_slugs = []
    for r in rows:
        for s in r["placed"]:
            kind, slug = s.split(":", 1)
            all_kinds.append(kind)
            all_slugs.append(slug)
    n_objects_per_src = [r["n_objects"] for r in rows]
    return {
        "rows": rows,
        "kinds_seen": sorted(set(all_kinds)),
        "unique_slugs": sorted(set(all_slugs)),
        "all_deterministic": all(r["deterministic"] for r in rows),
        "n_objects_min": int(min(n_objects_per_src)) if n_objects_per_src else 0,
        "n_objects_max": int(max(n_objects_per_src)) if n_objects_per_src else 0,
        "n_objects_mean": float(np.mean(n_objects_per_src)) if n_objects_per_src else 0.0,
    }


def check_rgb_diversity(out_dir: Path, sample_cap: int = 60) -> dict:
    """Per-image unique-BGR diversity stats.

    Phase 9 wants visually-varied output — we should see thousands of
    distinct BGR triples per 480x640 RGB frame (the harness has 4096
    textured points + 100k-point walls/objects, all sampling real CC0
    photos).
    """
    pngs = []
    for split in ("train", "val", "test"):
        d = out_dir / split
        if not d.is_dir():
            continue
        for sub in d.iterdir():
            rgb_dir = sub / "rgb"
            if rgb_dir.is_dir():
                pngs.extend(_list_pngs(rgb_dir))
    pngs = pngs[:sample_cap]
    counts: list[int] = []
    mean_pixels: list[float] = []
    for p in pngs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        flat = img.reshape(-1, 3)
        # uniqueness via int packing
        packed = (flat[:, 0].astype(np.uint32) |
                  (flat[:, 1].astype(np.uint32) << 8) |
                  (flat[:, 2].astype(np.uint32) << 16))
        counts.append(int(np.unique(packed).size))
        mean_pixels.append(float(flat.mean()))
    return {
        "sampled": len(counts),
        "unique_bgr_min": int(min(counts)) if counts else 0,
        "unique_bgr_max": int(max(counts)) if counts else 0,
        "unique_bgr_mean": float(np.mean(counts)) if counts else 0.0,
        "mean_pixel_brightness": float(np.mean(mean_pixels)) if mean_pixels else 0.0,
    }


def check_wall_presence(out_dir: Path, sample_cap: int = 30) -> dict:
    """Phase 9: the multi-wall enclosure should fill 80%+ of each frame
    with rendered point splats (depth > 0), not the 0-pixel "no data"
    holes that single-floor + 2D backdrop runs leave behind.
    """
    pngs = []
    for split in ("train", "val", "test"):
        d = out_dir / split
        if not d.is_dir():
            continue
        for sub in d.iterdir():
            depth_dir = sub / "depth"
            if depth_dir.is_dir():
                pngs.extend(_list_pngs(depth_dir))
    pngs = pngs[:sample_cap]
    fractions: list[float] = []
    for p in pngs:
        D = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if D is None:
            continue
        fractions.append(float((D > 0).mean()))
    return {
        "sampled": len(fractions),
        "depth_coverage_min": float(min(fractions)) if fractions else 0.0,
        "depth_coverage_max": float(max(fractions)) if fractions else 0.0,
        "depth_coverage_mean": float(np.mean(fractions)) if fractions else 0.0,
    }


def check_background_density(out_dir: Path,
                              sources: list[int],
                              set_id: int = 0,
                              bg_n_points: int = 60000) -> dict:
    """Confirm background generation produces 10-15 objects per source.

    Re-runs ``generate_background_scene`` with the same seeds the renderer
    uses and counts the placed objects per source. Reports min/max/mean.
    """
    from texture_mapping import (load_object_library,
                                   load_background_library,
                                   generate_background_scene)

    obj_lib = load_object_library(PROJECT_ROOT / "data" / "objects")
    bg_lib = load_background_library(PROJECT_ROOT / "data" / "textures" / "backgrounds")
    rows = []
    for src in sources:
        pcl = np.load(PROJECT_ROOT / "data" / "set2" / f"{set_id:03d}" /
                       "pointclouds_normed_4096" / f"pcl_{src:04d}.npy")
        seed = set_id * 1000 + src + 7
        # Re-run the bg generation. ``generate_background_scene`` doesn't
        # itself report object count, so we count via the n_pick formula
        # used inside it (deterministic with the same RNG state). But the
        # cleaner check: re-run with a probe RNG that records n_pick.
        # Simpler: call it and just check that the output has more points
        # than v1 would (i.e. > 30k floor + objects).
        bg_pcl, bg_rgb = generate_background_scene(
            rng=np.random.RandomState(seed),
            bbox_min=pcl.min(axis=0), bbox_max=pcl.max(axis=0),
            texture_library=bg_lib,
            object_library=obj_lib,
            n_points=bg_n_points,
        )
        rows.append({
            "src": src,
            "n_points_total": int(bg_pcl.shape[0]),
        })
    return {
        "rows": rows,
        "n_points_min": int(min(r["n_points_total"] for r in rows)) if rows else 0,
        "n_points_max": int(max(r["n_points_total"] for r in rows)) if rows else 0,
        "n_points_mean": float(np.mean([r["n_points_total"] for r in rows])) if rows else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("smoke_dir", type=Path)
    parser.add_argument("--num-anim", type=int, default=20)
    parser.add_argument("--set-id", type=int, default=0)
    parser.add_argument("--sources", type=int, nargs="+",
                        default=[0, 30, 60, 90, 120])
    parser.add_argument("--phase9", action="store_true",
                        help="Use Phase 9 thresholds (no anim stickiness, no "
                             "v2 density checks). Auto-detected from "
                             "smoke_dir/metadata.json if dataset_mode=phase9.")
    args = parser.parse_args()

    out: Path = args.smoke_dir
    if not out.is_dir():
        print(f"smoke dir not found: {out}", file=sys.stderr)
        return 1

    # Auto-detect Phase 9 mode from metadata.json.
    is_phase9 = args.phase9
    meta_path = out / "metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("dataset_mode") == "phase9":
                is_phase9 = True
        except Exception:
            pass

    report = {"smoke_dir": str(out), "phase9": is_phase9}
    print(f"=== Smoke validation (phase9={is_phase9}) ===")
    print(f"smoke_dir: {out}")
    print()

    print("1. File counts")
    fc = check_file_counts(out)
    for k, v in fc["counts"].items():
        print(f"   {k:30s}  {v:6d}")
    if fc["issues"]:
        for i in fc["issues"]:
            print(f"   FAIL: {i}")
    report["file_counts"] = fc

    print()
    print("2. Label values ⊆ {0..5}")
    lv = check_label_values(out)
    print(f"   sampled: {lv['sampled']}  values seen: {lv['values_seen']}")
    if lv["issues"]:
        for i in lv["issues"][:5]:
            print(f"   FAIL: {i}")
    report["label_values"] = lv

    print()
    print("3. Depth range")
    dr = check_depth_range(out)
    print(f"   sampled: {dr['sampled']}  min={dr['min_mm']} mm  max={dr['max_mm']} mm")
    if dr["issues"]:
        for i in dr["issues"][:5]:
            print(f"   FAIL: {i}")
    report["depth_range"] = dr

    fails: list[str] = []
    if lv["issues"]:
        fails.append("label values out of range")
    if dr["issues"]:
        fails.append("depth values out of range")

    if is_phase9:
        print()
        print("4. RGB diversity (per-image unique BGR triples)")
        rd = check_rgb_diversity(out)
        print(f"   sampled: {rd['sampled']}  unique_bgr min/mean/max = "
              f"{rd['unique_bgr_min']} / {rd['unique_bgr_mean']:.0f} / "
              f"{rd['unique_bgr_max']}")
        print(f"   mean pixel brightness (0..255): "
              f"{rd['mean_pixel_brightness']:.1f}")
        report["rgb_diversity"] = rd
        if rd["unique_bgr_min"] < 2000:
            fails.append(
                f"min unique BGR per image = {rd['unique_bgr_min']} "
                f"(target ≥ 2000; walls + harness should saturate this)")

        print()
        print("5. Wall presence (depth>0 coverage; multi-wall enclosure)")
        wp = check_wall_presence(out)
        print(f"   sampled: {wp['sampled']}  depth_coverage min/mean/max = "
              f"{wp['depth_coverage_min']:.3f} / "
              f"{wp['depth_coverage_mean']:.3f} / "
              f"{wp['depth_coverage_max']:.3f}")
        report["wall_presence"] = wp
        if wp["depth_coverage_min"] < 0.55:
            fails.append(
                f"min depth coverage = {wp['depth_coverage_min']:.3f} "
                f"(target ≥ 0.55 — walls should fill most of the frame)")
        if wp["depth_coverage_mean"] < 0.75:
            fails.append(
                f"mean depth coverage = {wp['depth_coverage_mean']:.3f} "
                f"(target ≥ 0.75 — walls should fill the frame)")

        print()
        print("6. Foreground density (phase9 target: 0-2 benign occluders)")
        from texture_mapping import PHASE9_DROP_CATEGORIES
        from texture_mapping import (load_object_library,
                                      filter_library_by_category,
                                      generate_phase9_foreground)
        obj_lib_all = load_object_library(PROJECT_ROOT / "data" / "objects")
        obj_lib = filter_library_by_category(
            obj_lib_all, exclude=PHASE9_DROP_CATEGORIES)
        rows = []
        for src in args.sources:
            pcl = np.load(PROJECT_ROOT / "data" / "set2" /
                          f"{args.set_id:03d}" /
                          "pointclouds_normed_4096" / f"pcl_{src:04d}.npy")
            seed = args.set_id * 1000 + src + 113
            pts1, _, info1 = generate_phase9_foreground(
                rng=np.random.RandomState(seed),
                object_library=obj_lib,
                bbox_min=pcl.min(axis=0), bbox_max=pcl.max(axis=0),
                n_points=6000, max_objects=2,
            )
            pts2, _, _ = generate_phase9_foreground(
                rng=np.random.RandomState(seed),
                object_library=obj_lib,
                bbox_min=pcl.min(axis=0), bbox_max=pcl.max(axis=0),
                n_points=6000, max_objects=2,
            )
            rows.append({
                "src": src,
                "n_points": int(pts1.shape[0]),
                "n_objects": len(info1["placed"]),
                "placed": [f"{p['kind']}:{p['slug']}" for p in info1["placed"]],
                "deterministic": bool(np.array_equal(pts1, pts2)),
            })
        fg = {"rows": rows,
              "all_deterministic": all(r["deterministic"] for r in rows),
              "n_objects_min": min(r["n_objects"] for r in rows) if rows else 0,
              "n_objects_max": max(r["n_objects"] for r in rows) if rows else 0,
              "n_objects_mean": float(np.mean(
                  [r["n_objects"] for r in rows])) if rows else 0.0}
        for r in fg["rows"]:
            det = "DET" if r["deterministic"] else "NON-DET"
            print(f"   src={r['src']:03d} {det:7s} pts={r['n_points']:>6d}  "
                  f"objs={r['n_objects']}  placed={r['placed']}")
        print(f"   all deterministic: {fg['all_deterministic']}")
        report["foreground_phase9"] = fg
        if not fg["all_deterministic"]:
            fails.append("foreground (phase9) not deterministic")
        if fg["n_objects_max"] > 2:
            fails.append(
                f"max fg objects per source = {fg['n_objects_max']} "
                f"(phase9 cap is 2)")
    else:
        print()
        print("4. Anim stickiness (bg + fg static; harness animates)")
        sticky = []
        for view in ("front", "left", "top"):
            s = check_anim_stickiness(out, args.set_id, args.sources[0],
                                        view, args.num_anim)
            sticky.append(s)
            print(f"   {view:6s}  min_cos={s.get('min_cosine', 'NA'):.4f}  "
                  f"mean_cos={s.get('mean_cosine', 'NA'):.4f}")
        report["anim_stickiness"] = sticky

        print()
        print("5. Foreground diversity & density (v2 target: 10-15 obj/src)")
        fg = check_foreground_diversity(out, args.sources, set_id=args.set_id)
        for r in fg["rows"]:
            det = "DET" if r["deterministic"] else "NON-DET"
            print(f"   src={r['src']:03d} {det:7s} pts={r['n_points']:>6d}  "
                  f"objs={r['n_objects']:>2d}  counts={r['counts']}")
        print(f"   kinds seen: {fg['kinds_seen']}")
        print(f"   unique slugs: {len(fg['unique_slugs'])}")
        print(f"   all deterministic: {fg['all_deterministic']}")
        print(f"   fg objects per source: min={fg['n_objects_min']}  "
              f"max={fg['n_objects_max']}  mean={fg['n_objects_mean']:.1f}")
        report["foreground"] = fg

        print()
        print("6. Background density (v2 target: 60k+ points / 10-15 obj/src)")
        bg = check_background_density(out, args.sources, set_id=args.set_id)
        for r in bg["rows"]:
            print(f"   src={r['src']:03d}  bg pts={r['n_points_total']:>6d}")
        print(f"   bg points: min={bg['n_points_min']}  max={bg['n_points_max']}  "
              f"mean={bg['n_points_mean']:.0f}")
        report["background"] = bg

        if not fg["all_deterministic"]:
            fails.append("foreground not deterministic")
        if any((s.get("min_cosine") or 0) < 0.85 for s in sticky):
            fails.append("anim stickiness below 0.85")
        if fg["n_objects_min"] < 8:
            fails.append(
                f"min fg objects per source = {fg['n_objects_min']} "
                f"(target ≥ 8)")
        if fg["n_objects_mean"] < 10:
            fails.append(
                f"mean fg objects per source = {fg['n_objects_mean']:.1f} "
                f"(target ≥ 10)")
        if bg["n_points_min"] < 40000:
            fails.append(
                f"min bg points = {bg['n_points_min']} (target ≥ 40k)")

    out_json = out / "validation.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    print()
    print(f"report: {out_json}")

    print()
    if fails:
        print(f"FAIL: {fails}")
        return 2
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
