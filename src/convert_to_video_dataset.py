#!/usr/bin/env python3
"""Convert full CDLO point cloud dataset to animated RGB-D video dataset.

For EVERY source frame (28 sets × 300 frames = 8,100 samples), generates an
animated video clip via skeleton-based FK, then renders to RGB-D from 6 views.

Output structure (DL-ready):
    data/rgbd_videos/
    ├── metadata.json
    ├── train/{set_id}/
    │   ├── rgb/{src}_{anim}_{view}.png       8-bit BGR, 640×480
    │   ├── depth/{src}_{anim}_{view}.png     16-bit mm, 640×480
    │   ├── pointclouds/{src}_{anim}.npy      float32 (4096,3)
    │   └── labels/{src}.npy                  int8 (4096,)
    ├── val/{set_id}/...
    └── test/{set_id}/...

    Naming:  {src} = source frame (0000-0299)
             {anim} = animation frame (00-19)
             {view} = front|back|right|left|top|bottom

Usage:
    python src/convert_to_video_dataset.py
    python src/convert_to_video_dataset.py --workers 4 --num-frames 10
    python src/convert_to_video_dataset.py --dry-run
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "set2"
# Phase 8 default was data/rgbd_videos_v2/. Phase 9 ships a new default
# directory so the v2 dataset stays available for the post-mortem. Override
# via the KIAT_OUTPUT_ROOT env var.
_OUTPUT_ROOT_DEFAULT = PROJECT_ROOT / "data" / "rgbd_videos_phase9"
OUTPUT_ROOT = Path(os.environ.get("KIAT_OUTPUT_ROOT", str(_OUTPUT_ROOT_DEFAULT)))

# Dataset-mode switch. ``phase9`` flips to the multi-wall + randomised camera
# pipeline; ``phase11`` keeps the Phase 4 baseline (2D photo backdrop, 6
# canonical views, no HSV jitter) but adds at-most-one off-wire hand per
# sample; ``phase12`` reproduces the STRICT Phase 4 baseline (the 4-layer
# composition the Phase 7 teacher was trained on: textured harness + textured
# xz-floor + 3-6 CC0 mesh-derived clutter PCLs + 2D photographic backdrop, 6
# canonical views, no HSV jitter, no foreground hands or extra objects); the
# only lever Phase 12 exercises vs Phase 4 is enrichment of the 2D backdrop
# pool at ``data/textures/backgrounds/``. ``v2`` (or any other value) keeps
# the v2 high-density pipeline so the existing post-mortem references stay
# re-runnable. Auto-detected from the output dir name if not explicitly set.
if "phase13" in OUTPUT_ROOT.name:
    _DEFAULT_MODE = "phase13"
elif "phase12" in OUTPUT_ROOT.name:
    _DEFAULT_MODE = "phase12"
elif "phase11" in OUTPUT_ROOT.name:
    _DEFAULT_MODE = "phase11"
elif "phase9" in OUTPUT_ROOT.name:
    _DEFAULT_MODE = "phase9"
else:
    _DEFAULT_MODE = "v2"
DATASET_MODE = os.environ.get("KIAT_DATASET_MODE", _DEFAULT_MODE).lower()
if DATASET_MODE not in ("phase9", "v2", "phase11", "phase12", "phase13"):
    raise SystemExit(
        f"KIAT_DATASET_MODE={DATASET_MODE!r} not one of "
        "'phase9' / 'phase11' / 'phase12' / 'phase13' / 'v2'."
    )

# Phase 13 reproduces the STRICT Phase 4 base (identical to phase12 in every
# structural respect) and adds two independent, default-OFF appearance levers
# baked into the per-point BGR before rasterisation: (a) a 3D point-light
# scene-shading term and (b) a per-object colour gradient. Both leave the
# Phase-4 scene-composition rng stream untouched (separate rngs), so with both
# levers OFF a phase13 render is byte-identical to the phase12 / H0 control.
# ``_STRICT_P4_MODES`` lets every strict-Phase-4 branch below accept either
# mode without duplication.
_STRICT_P4_MODES = ("phase12", "phase13")

# Phase 13 lever toggles + strength overrides (only consulted in phase13).
P13_LIGHTING = os.environ.get("KIAT_P13_LIGHTING", "0").strip() in ("1", "true", "True")
P13_OBJGRAD = os.environ.get("KIAT_P13_OBJGRAD", "0").strip() in ("1", "true", "True")
# Stronger directional lighting per PI direction (2026-06-03): shadow side
# ~0.50x, lit side ~1.10x, steeper span ⇒ a clearly visible 3D light gradient.
# Lighting is a hue-preserving brightness multiply, so it owns luma and never
# overwrites the object-colour gradient's hue.
P13_LIGHT_AMBIENT = float(os.environ.get("KIAT_P13_LIGHT_AMBIENT", "0.50"))
P13_LIGHT_GAIN = float(os.environ.get("KIAT_P13_LIGHT_GAIN", "0.60"))
P13_LIGHT_SPAN = float(os.environ.get("KIAT_P13_LIGHT_SPAN", "1.4"))
# Object-colour-gradient strength (defaults = the "strong" setting). Tunable so
# a gentler envelope can be screened after the strong setting fails the gate.
P13_OBJGRAD_HUE = float(os.environ.get("KIAT_P13_OBJGRAD_HUE", "60.0"))
P13_OBJGRAD_SAT_LO = float(os.environ.get("KIAT_P13_OBJGRAD_SAT_LO", "0.60"))
P13_OBJGRAD_SAT_HI = float(os.environ.get("KIAT_P13_OBJGRAD_SAT_HI", "1.45"))
P13_OBJGRAD_VAL_LO = float(os.environ.get("KIAT_P13_OBJGRAD_VAL_LO", "0.85"))
P13_OBJGRAD_VAL_HI = float(os.environ.get("KIAT_P13_OBJGRAD_VAL_HI", "1.18"))
P13_OBJGRAD_KWARGS = {
    "max_hue_shift_deg": P13_OBJGRAD_HUE,
    "sat_scale_range": (P13_OBJGRAD_SAT_LO, P13_OBJGRAD_SAT_HI),
    "val_scale_range": (P13_OBJGRAD_VAL_LO, P13_OBJGRAD_VAL_HI),
}

# Phase 14 lever: targeted hard-negative confusers (dark cylinders / sharp
# edges / hands) appended to the strict-Phase-4 background clutter pool as
# labeled background. Default OFF ⇒ a phase13 render is unchanged. Addresses
# the real-world false positives on black cylindrical structures, black sharp
# edges, and hands. Uses a SEPARATE per-source rng so the base clutter stream
# is byte-identical when the lever is OFF.
P14_NEGATIVES = os.environ.get("KIAT_P14_NEGATIVES", "0").strip() in ("1", "true", "True")
# Floor confusers = small/scattered dark CYLINDERS + sharp EDGES only (n_hand=0):
# hands are NOT floor clutter — they are posed near/holding the wire (below),
# matching the real deployment false-positive.
P14_NEG_KWARGS = {
    "n_cyl": int(os.environ.get("KIAT_P14_NEG_NCYL", "6")),
    "n_edge": int(os.environ.get("KIAT_P14_NEG_NEDGE", "3")),
    "n_hand": 0,
    "dark_factor": float(os.environ.get("KIAT_P14_NEG_DARK", "0.16")),
}
# Near-wire hands (labeled BACKGROUND): how many to grip the harness, the
# up-sampled per-hand point budget (resolution; the hand assets cap at 8000
# points so we densify past that), and a size multiplier on the natural grasp
# scale (bigger hands). _place_hand_on_wire poses each on a random wire point.
P14_NEG_NHAND_WIRE = int(os.environ.get("KIAT_P14_NEG_NHAND", "2"))
P14_NEG_HAND_PTS = int(os.environ.get("KIAT_P14_NEG_HANDPTS", "32000"))
P14_NEG_HAND_SCALE = float(os.environ.get("KIAT_P14_NEG_HANDSCALE", "2.5"))
# Per-source probability that a wire-gripping hand is placed at all. ~half the
# source frames get a hand and half don't, so the model never learns "the wire
# always has a hand on it". Drawn from the SAME hand rng before any placement,
# so when no hand is emitted the foreground is empty (identical to lever-off
# strict Phase 4). Default 0.5.
P14_NEG_P_HAND = float(os.environ.get("KIAT_P14_NEG_PHAND", "0.5"))
if not 0.0 <= P14_NEG_P_HAND <= 1.0:
    raise SystemExit(
        f"KIAT_P14_NEG_PHAND={P14_NEG_P_HAND} must lie in [0, 1]."
    )
# Fidelity: scale the background point budget so the (small, scattered) floor
# confusers render as solid objects rather than sparse blobs.
P14_NEG_FIDELITY = float(os.environ.get("KIAT_P14_NEG_FIDELITY", "1.4"))

# Phase 15 lever: per-source probability that a SOURCE is rendered WIRE-FREE —
# the full scene (floor, clutter, dark confuser cylinders/edges, lighting, 2D
# backdrop) is composed exactly as normal using the harness bbox, but the
# harness itself is OMITTED from both the rendered points and the labels, so
# the frame shows the scene with NO wire and the label is ALL background. The
# near-wire hands are also skipped (no wire to grip). Decided once per
# (set_id, frame_id) with a SEPARATE rng, drawn ONLY when the prob > 0, so the
# lever-OFF (default 0.0) render is byte-identical to Phase 14 (the base
# composition rng streams are never touched). Default 0.0 ⇒ no wire-free
# frames. Only consulted in phase13 (the strict-Phase-4 base).
P15_WIREFREE_P = float(os.environ.get("KIAT_P15_WIREFREE_P", "0.0"))
if not 0.0 <= P15_WIREFREE_P <= 1.0:
    raise SystemExit(
        f"KIAT_P15_WIREFREE_P={P15_WIREFREE_P} must lie in [0, 1]."
    )

# Phase 16 lever: per-source probability that ONE OPEN / SPLAYED hand (palm /
# fingers visible, NOT gripping the wire) is dropped into the scene as labeled
# BACKGROUND foreground, so the model learns "open hand = not wire" — the last
# residual real-world false positive (the best Phase-15 model paints green on a
# splayed palm). Decided once per (set_id, frame_id) with a SEPARATE rng, drawn
# ONLY when the prob > 0, so the lever-OFF (default 0.0) render is byte-identical
# to Phase 15 (the P14/P15 rng streams are never touched). Applies to BOTH wired
# and wire-free frames (an open hand with no wire is also a valid negative).
# Only consulted in phase13 (the strict-Phase-4 base).
P16_OPENHAND_P = float(os.environ.get("KIAT_P16_OPENHAND_P", "0.0"))
if not 0.0 <= P16_OPENHAND_P <= 1.0:
    raise SystemExit(
        f"KIAT_P16_OPENHAND_P={P16_OPENHAND_P} must lie in [0, 1]."
    )
# The up-sampled per-hand point budget (resolution) and a multiplier on the
# natural scale — same rationale / defaults as the Phase-14 grip hands, so the
# splayed fingers render large enough to stay separated after the splat + close.
P16_OPENHAND_PTS = int(os.environ.get("KIAT_P16_OPENHAND_PTS", "32000"))
P16_OPENHAND_SCALE = float(os.environ.get("KIAT_P16_OPENHAND_SCALE", "2.0"))

# Phase 17 lever: scale the per-scene background clutter object count.
# Defaults reproduce the strict-Phase-4 (3,7) range EXACTLY, so with the
# knobs unset a render is byte-identical to Phase 15. Phase 17 sets 5/10
# (→ 5-9 objects, ~+50%). Only consulted in strict-Phase-4 modes.
P17_NOBJ_LO = int(os.environ.get("KIAT_P17_NOBJ_LO", "3"))
P17_NOBJ_HI = int(os.environ.get("KIAT_P17_NOBJ_HI", "7"))
if P17_NOBJ_HI <= P17_NOBJ_LO:
    raise ValueError(f"KIAT_P17_NOBJ_HI({P17_NOBJ_HI}) must be > KIAT_P17_NOBJ_LO({P17_NOBJ_LO})")

# Phase 18 levers: wire-appearance enrichment (both default OFF ⇒ a phase13
# render is identical to Phase 15). Addresses the real-world appearance
# gap measured on the partner GT valset: bright/saturated jacket colours
# (orange above all: 19.9 % of real wire px, recall 0.144) are absent from the
# 11-photo wire texture pool, and synth wires are far thinner than real ones
# (median ~6 px, 4.8 % >16 px vs 47 % >16 px real at 480x640).
#   KIAT_P18_WIRETEX_DIR — directory of EXTRA wire swatch textures (generate
#     with src/gen_wire_swatches.py). When set, each wire segment draws its
#     texture uniformly from (original pool ∪ swatches) using a SEPARATE rng
#     (offset +501); the original pool draws still occur (and are discarded)
#     so the master texture stream feeding the non-wire classes is unchanged.
#   KIAT_P18_THICK_LO / KIAT_P18_THICK_HI — per-harness radial thickness
#     multiplier k ~ U[lo, hi] (SEPARATE rng, offset +502) applied to the
#     skeleton-binding offsets (never the skeleton), fattening the harness
#     about its centerline. Set both together; unset ⇒ k ≡ 1.
# Only consulted in phase13 (the strict-Phase-4 base), like the P14-P17 levers.
P18_WIRETEX_DIR = os.environ.get("KIAT_P18_WIRETEX_DIR", "").strip()
if P18_WIRETEX_DIR:
    _p18_dir = Path(P18_WIRETEX_DIR)
    if not _p18_dir.is_absolute():
        _p18_dir = PROJECT_ROOT / _p18_dir
    if not _p18_dir.is_dir():
        raise SystemExit(f"KIAT_P18_WIRETEX_DIR={_p18_dir} is not a directory.")
_P18_TLO_RAW = os.environ.get("KIAT_P18_THICK_LO", "").strip()
_P18_THI_RAW = os.environ.get("KIAT_P18_THICK_HI", "").strip()
if bool(_P18_TLO_RAW) != bool(_P18_THI_RAW):
    raise SystemExit(
        "KIAT_P18_THICK_LO and KIAT_P18_THICK_HI must be set together.")
P18_THICK_RANGE = ((float(_P18_TLO_RAW), float(_P18_THI_RAW))
                   if _P18_TLO_RAW else None)
if P18_THICK_RANGE is not None and not (
        0.0 < P18_THICK_RANGE[0] <= P18_THICK_RANGE[1]):
    raise SystemExit(
        f"KIAT_P18_THICK_LO/HI={P18_THICK_RANGE} must satisfy 0 < lo <= hi.")

# Phase 19 levers: busy / painterly NEGATIVE textures (default OFF ⇒ a
# phase13 render is unchanged from Phase 18). Addresses the real-world
# false-positive forensics on the 62-frame GT valset: 69.6 % of FP pixels are
# texture blobs on BUSY surfaces (graffiti/mural strokes, striped terminal
# blocks, socket panels, shadow-traced granite) which the 11 calm photo
# backdrops never show. Textures come from src/gen_busy_negative_textures.py
# and are only ever composited as 2D backdrop / 3D floor texture, so every
# pixel they touch is labelled background by construction.
#   KIAT_P19_BUSYBG_DIR — directory of busy texture PNGs. When set, each
#     frame swaps its 2D photo BACKDROP for a busy texture with probability
#     KIAT_P19_BUSYBG_P using a SEPARATE rng (offset +601; decision then
#     uniform choice from the SAME stream). On a miss (or with the knob
#     unset) the deterministic (sid*1000+fid) % len(pool) selection is kept
#     the same.
#   KIAT_P19_BUSYBG_P — busy-backdrop probability (default 0.35 when the
#     DIR is set; ignored otherwise).
#   KIAT_P19_BUSYFLOOR_P — probability (default 0 = OFF) that the 3D FLOOR
#     texture is drawn from the busy pool (same DIR) instead of its usual
#     pool, via a SEPARATE rng (offset +602). The original-pool draw inside
#     generate_background_scene still occurs (and is discarded) so the
#     scene-composition rng stream is unchanged either way.
# Wire-present AND wire-free frames are both eligible (wires ON busy
# backgrounds train camouflage-robust recall; busy wire-free frames are pure
# negative pressure). Only consulted in phase13, like the P14-P18 levers.
P19_BUSYBG_DIR = os.environ.get("KIAT_P19_BUSYBG_DIR", "").strip()
if P19_BUSYBG_DIR:
    _p19_dir = Path(P19_BUSYBG_DIR)
    if not _p19_dir.is_absolute():
        _p19_dir = PROJECT_ROOT / _p19_dir
    if not _p19_dir.is_dir():
        raise SystemExit(f"KIAT_P19_BUSYBG_DIR={_p19_dir} is not a directory.")
P19_BUSYBG_P = float(os.environ.get("KIAT_P19_BUSYBG_P", "0.35"))
if not 0.0 <= P19_BUSYBG_P <= 1.0:
    raise SystemExit(
        f"KIAT_P19_BUSYBG_P={P19_BUSYBG_P} must lie in [0, 1].")
P19_BUSYFLOOR_P = float(os.environ.get("KIAT_P19_BUSYFLOOR_P", "0.0"))
if not 0.0 <= P19_BUSYFLOOR_P <= 1.0:
    raise SystemExit(
        f"KIAT_P19_BUSYFLOOR_P={P19_BUSYFLOOR_P} must lie in [0, 1].")
if P19_BUSYFLOOR_P > 0.0 and not P19_BUSYBG_DIR:
    raise SystemExit(
        "KIAT_P19_BUSYFLOOR_P > 0 requires KIAT_P19_BUSYBG_DIR (the busy "
        "floor draws from the same texture pool).")

# ── Phase 26.1: GEOMETRY-AWARE wire thickening (densify-then-thin-splat) ──────
# P26.0 thickened wires by splatting a fat (R=5px) SOLID disc per point onto the
# SPARSE 4096-pt harness skeleton. Big discs on sparse points BLOB: components
# became 1.56x rounder (circularity 0.081 -> 0.126), +68 % area, and thin
# parallel cables MERGED (comps/frame 9.3 -> 7.2) -> real-world FP/FN regression.
# The WIDTH target was right (real GT ~15px, P26.0 hit 12px) but the METHOD was
# the bug. P26.1 fixes the method: instead of fat-disc-on-sparse-points, it
# DENSIFIES the wire point cloud ALONG the skeleton first (a continuous dense
# tube of centerline samples x a small radial ring), then the soft compositor
# splats with a SMALL footprint. Dense points + small footprint => continuous,
# controllable width that stays a thin curvilinear DLO and does NOT round
# junctions or merge parallel strands.
#
# This lever is APPLIED in convert_one_video (where the skeleton topology +
# per-point bindings live) BEFORE the wire points are combined for rasterisation
# — the extra points are label-0 wire points so the existing P26 wire-split
# (pcl_to_rgbd.rasterize_view) picks them up automatically and the soft
# compositor renders them. The persisted per-source npys keep their (4096,)
# shape (extras are render-only, sliced off at [:n_pts_orig], exactly like the
# P18 thickening density extras). Only consulted in phase13, like P14-P19.
#
#   KIAT_P26_DENSIFY — master on/off for geometry-aware thickening. "0"/unset =>
#       no wire densification (identical to the pre-P26.1 path). "1" => on.
#       (Independent of KIAT_P26_SOFTEDGE: DENSIFY changes the wire POINT CLOUD;
#        SOFTEDGE changes how wire points are COMPOSITED. P26.1a/b use both.)
#   KIAT_P26_TUBE_RADIUS_PX — target wire tube RADIUS in NATIVE pixels (the
#       width lever). On-screen run-width ~= 2 * this (+ the small splat
#       footprint). Native wire radius ~4px (~6px run-width); ~7px => ~14px
#       run-width (real ~15px). Default 7.0. Converted to world units via SCALE.
#   KIAT_P26_AXIAL_STEP_PX — centerline sampling step along each wire edge, in
#       native px (default 0.7). Smaller => denser along-curve => no axial gaps.
#   KIAT_P26_RING_N — points per cross-section ring (default 14). More => no
#       circumferential gaps at the target radius. Ring is placed in the plane
#       perpendicular to the local edge tangent.
P26_DENSIFY = os.environ.get("KIAT_P26_DENSIFY", "0").strip() in (
    "1", "true", "True")
P26_TUBE_RADIUS_PX = float(os.environ.get("KIAT_P26_TUBE_RADIUS_PX", "7.0"))
P26_AXIAL_STEP_PX = float(os.environ.get("KIAT_P26_AXIAL_STEP_PX", "0.7"))
P26_RING_N = int(os.environ.get("KIAT_P26_RING_N", "14"))
if P26_DENSIFY:
    if not (0.0 < P26_TUBE_RADIUS_PX <= 30.0):
        raise SystemExit(
            f"KIAT_P26_TUBE_RADIUS_PX={P26_TUBE_RADIUS_PX} must be in (0, 30].")
    if not (0.05 <= P26_AXIAL_STEP_PX <= 5.0):
        raise SystemExit(
            f"KIAT_P26_AXIAL_STEP_PX={P26_AXIAL_STEP_PX} must be in [0.05, 5].")
    if not (3 <= P26_RING_N <= 64):
        raise SystemExit(f"KIAT_P26_RING_N={P26_RING_N} must be in [3, 64].")

# Phase 10 ablation overlay. When set, forces DATASET_MODE=v2 (the Phase 4
# baseline path), num_frames=1, and the object library to the original
# 21-entry "clutter" set; THEN applies exactly ONE Phase 9 change atop that
# baseline. Used by the Phase 10 ablation runner to isolate which of the
# 4 Phase 9 levers (wall / camera / harness_hsv / objects) is responsible
# for the real-world coverage collapse. Default (None) leaves the
# (DATASET_MODE x library) behaviour untouched so existing render pipelines
# are unaffected.
_ABL_RAW = os.environ.get("KIAT_ABL_VARIANT", "").strip().lower()
ABL_LEVERS = set()
_ABL_VALID = {"wall", "camera", "harness_hsv", "objects"}
if _ABL_RAW and _ABL_RAW != "none":
    for _piece in _ABL_RAW.split(","):
        _piece = _piece.strip()
        if _piece and _piece != "none":
            if _piece not in _ABL_VALID:
                raise SystemExit(
                    f"KIAT_ABL_VARIANT piece {_piece!r} not one of "
                    f"{sorted(_ABL_VALID)} / 'none'."
                )
            ABL_LEVERS.add(_piece)
ABL_VARIANT = "+".join(sorted(ABL_LEVERS)) if ABL_LEVERS else (
    "none" if _ABL_RAW == "none" else None
)
if ABL_VARIANT is not None:
    # The ablation always runs on the Phase-4-style v2 code path; the
    # ABL_LEVERS set then overlays the named axes. ABL_VARIANT='none'
    # means the strict-Phase-4 baseline (no overlay), useful for the
    # sanity-check that the pipeline reproduces canonical Phase 4 input.
    DATASET_MODE = "v2"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pcl_to_rgbd import (
    VIEWS, make_view_matrix, rasterize_view, random_view_dirs,
    IMG_W, IMG_H, FRUSTUM_HALF, SCALE, HALF_W,
    DEPTH_NEAR_MM, DEPTH_FAR_MM,
    CLASS_COLORS_RGB, CLASS_NAMES,
)
from texture_mapping import (
    BG_LABEL,
    PHASE9_DROP_CATEGORIES,
    apply_scene_lighting,
    build_confuser_negatives,
    compute_per_point_rgb,
    densify_point_cloud,
    _place_hand_on_wire,
    _place_open_hand,
    _OPEN_HAND_POSE_FAMILIES,
    filter_library_by_category,
    generate_background_scene,
    generate_foreground_scene,
    generate_phase9_foreground,
    generate_phase9_room_scene,
    generate_phase11_foreground,
    jitter_per_point_rgb,
    load_background_library,
    load_object_library,
    load_texture_library,
    sample_light_direction,
)

# Phase 9 uses random-per-source camera viewpoints; the 6 fixed names below
# label the OUTPUT slots only, not specific axis-aligned orientations.
# The Phase 10 ablation 'camera' variant also uses random views (view0..view5
# slot names) so the downstream stager picks them up; all other ablation
# variants keep Phase 4's canonical 6-view names.
N_VIEWS_PER_SOURCE = 6
# Phase 11 explicitly keeps Phase 4's 6 canonical views — `_USE_RANDOM_VIEWS`
# stays False even when phase11 is selected. Only phase9 / the 'camera'
# ablation lever flip to random per-source views.
_USE_RANDOM_VIEWS = (DATASET_MODE == "phase9") or ("camera" in ABL_LEVERS)
VIEW_NAMES = ([f"view{i}" for i in range(N_VIEWS_PER_SOURCE)]
              if _USE_RANDOM_VIEWS else
              list(VIEWS.keys()))
VIEW_ROTATIONS = ({}  # built per-source in convert_one_video
                  if _USE_RANDOM_VIEWS else
                  {vn: make_view_matrix(VIEWS[vn]["look"], VIEWS[vn]["up"])
                   for vn in VIEW_NAMES})

# Texture library: loaded lazily on first use per worker process.
_TEX_LIBRARY_CACHE = None
def _get_texture_library():
    global _TEX_LIBRARY_CACHE
    if _TEX_LIBRARY_CACHE is None:
        _TEX_LIBRARY_CACHE = load_texture_library(PROJECT_ROOT / "data" / "textures")
    return _TEX_LIBRARY_CACHE

# Phase 18 extra wire-swatch textures (KIAT_P18_WIRETEX_DIR). Lazy per-worker
# cache like the main library; empty list when the knob is unset.
_WIRE_EXT_CACHE = None
def _get_wire_ext_library():
    global _WIRE_EXT_CACHE
    if _WIRE_EXT_CACHE is None:
        _WIRE_EXT_CACHE = []
        if P18_WIRETEX_DIR:
            ext_dir = Path(P18_WIRETEX_DIR)
            if not ext_dir.is_absolute():
                ext_dir = PROJECT_ROOT / ext_dir
            for fp in sorted(ext_dir.iterdir()):
                if fp.is_file() and fp.suffix.lower() in (
                        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                    img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
                    if img is not None:
                        _WIRE_EXT_CACHE.append(img)
            if not _WIRE_EXT_CACHE:
                raise RuntimeError(
                    f"KIAT_P18_WIRETEX_DIR={ext_dir} has no readable textures.")
    return _WIRE_EXT_CACHE

# Phase 19 busy negative textures (KIAT_P19_BUSYBG_DIR). Two lazy per-worker
# caches over the SAME directory: the raw textures feed the 3D floor's
# per-point UV sampling (any size works), while the 2D backdrop path needs
# images resized once to the exact (IMG_H, IMG_W) frame that rasterize_view
# enforces. Both stay empty/unused when the knob is unset.
_BUSY_RAW_CACHE = None
_BUSY_BG_CACHE = None
def _get_busy_library_raw():
    global _BUSY_RAW_CACHE
    if _BUSY_RAW_CACHE is None:
        _BUSY_RAW_CACHE = []
        if P19_BUSYBG_DIR:
            busy_dir = Path(P19_BUSYBG_DIR)
            if not busy_dir.is_absolute():
                busy_dir = PROJECT_ROOT / busy_dir
            for fp in sorted(busy_dir.iterdir()):
                if fp.is_file() and fp.suffix.lower() in (
                        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                    img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
                    if img is not None:
                        _BUSY_RAW_CACHE.append(img)
            if not _BUSY_RAW_CACHE:
                raise RuntimeError(
                    f"KIAT_P19_BUSYBG_DIR={busy_dir} has no readable textures.")
    return _BUSY_RAW_CACHE

def _get_busy_backdrop_library():
    global _BUSY_BG_CACHE
    if _BUSY_BG_CACHE is None:
        _BUSY_BG_CACHE = [
            (img if img.shape[:2] == (IMG_H, IMG_W)
             else cv2.resize(img, (IMG_W, IMG_H),
                             interpolation=cv2.INTER_LINEAR))
            for img in _get_busy_library_raw()
        ]
    return _BUSY_BG_CACHE

# Background-photo library: used (a) as the texture source for the 3D floor
# in the bg scene and (b) as the 2D backdrop filling pixels with no 3D point.
# Lazy per-worker cache. The directory defaults to data/textures/backgrounds
# but can be overridden with KIAT_BG_DIR — used by the Phase 12 backdrop-pool
# experiments to point at an 11-photo control dir (reproduce Phase 4 exactly)
# or a 500+-photo enriched dir (H2) without disturbing the canonical library.
_BG_LIBRARY_CACHE = None
_BG_DIR = Path(os.environ.get(
    "KIAT_BG_DIR", str(PROJECT_ROOT / "data" / "textures" / "backgrounds")))
def _get_background_library():
    global _BG_LIBRARY_CACHE
    if _BG_LIBRARY_CACHE is None:
        _BG_LIBRARY_CACHE = load_background_library(_BG_DIR)
    return _BG_LIBRARY_CACHE

# Real-object library (Phase 4): CC0 mesh-derived point clouds dropped on the
# floor as workshop clutter. Lazy per-worker cache (loaded once per process).
_OBJ_LIBRARY_CACHE = None
def _get_object_library():
    global _OBJ_LIBRARY_CACHE
    if _OBJ_LIBRARY_CACHE is None:
        _OBJ_LIBRARY_CACHE = load_object_library(
            PROJECT_ROOT / "data" / "objects"
        )
    return _OBJ_LIBRARY_CACHE

# Number of background points sampled per source frame's 3D scene.
#   v2:       60000 (single floor + photo backdrop, 10-15 floor objects)
#   phase9:  240000 (5-wall enclosure + 3-6 floor clutter; walls each need
#                    ~40k points to splat to full-image coverage at
#                    HALF_W * 2 * FRUSTUM_H * 2 = ~6.4 wu^2 with splat
#                    radius 1).
#   phase11: 60000 (Phase 4 backbone — single floor + photo backdrop +
#                    3-15 floor clutter, same as v2 bg).
#   phase12: 30000 (Phase 4 original — single floor + photo backdrop +
#                   3-6 floor clutter, the sparser baseline that the Phase 7
#                   teacher was trained on; cf. §4.20 #5 in CONTEXT.md).
if DATASET_MODE == "phase9":
    BG_N_POINTS = 240000
elif DATASET_MODE in _STRICT_P4_MODES:
    BG_N_POINTS = 30000
else:
    BG_N_POINTS = 60000

# Foreground budget.
#   v2:      24000 (8-17 fg objects, dense skin coverage)
#   phase9:   6000 (0-2 benign occluders, no hands / wire-shaped negatives)
#   phase11:  6000 (≤1 off-wire hand; only one object's worth needed)
#   phase12:  8000 (unused — Phase 4 has no foreground objects; kept at the
#                   Phase 4 original value so any accidental fg path still
#                   matches the baseline budget).
if DATASET_MODE == "phase9":
    FG_N_POINTS = 6000
elif DATASET_MODE == "phase11":
    FG_N_POINTS = 6000
elif DATASET_MODE in _STRICT_P4_MODES:
    FG_N_POINTS = 8000
else:
    FG_N_POINTS = 24000

# Phase 11: probability that any given sample gets a hand foreground occluder
# (vs. no foreground at all). 0.30 = 30 % of samples have a hand, 70 % look
# identical to a vanilla Phase 4 sample. Override via KIAT_PHASE11_P_HAND.
PHASE11_P_HAND = float(os.environ.get("KIAT_PHASE11_P_HAND", "0.30"))
if not 0.0 <= PHASE11_P_HAND <= 1.0:
    raise SystemExit(
        f"KIAT_PHASE11_P_HAND={PHASE11_P_HAND} must lie in [0, 1]."
    )

TRAIN_SETS = set(range(0, 32))
VAL_SETS   = set(range(32, 36))
TEST_SETS  = set(range(36, 40))
FRAMES_PER_SET = 300


def split_of(set_id):
    if set_id in TRAIN_SETS:   return "train"
    if set_id in VAL_SETS:     return "val"
    return "test"


# ── Geometry helpers (vectorised, no per-point loops) ──────────────────────

def _rotation_matrix(axis, angle):
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _build_topology(adj):
    """Build kinematic-tree topology from adjacency (constant within a set).

    Returns root, children-list, degrees, structural-node list,
    wire-segment lists, selected animation joints, and the edge index array.
    """
    n = adj.shape[0]
    degrees = np.sum(adj > 0, axis=1).astype(int)
    structural = [i for i in range(n) if degrees[i] != 2]

    # Edge index (upper triangle)
    rows, cols = np.where(np.triu(adj > 0))
    edges = np.column_stack([rows, cols])  # (E, 2)

    # Wire segments between structural nodes
    struct_set = set(structural)
    segments, visited_edges = [], set()
    for start in structural:
        for nxt in np.where(adj[start] > 0)[0]:
            key = (min(start, nxt), max(start, nxt))
            if key in visited_edges:
                continue
            seg = [start, nxt]; visited_edges.add(key)
            cur, prev = nxt, start
            while degrees[cur] == 2:
                nbrs = np.where(adj[cur] > 0)[0]
                others = [x for x in nbrs if x != prev]
                if not others:
                    break
                nx = others[0]
                visited_edges.add((min(cur, nx), max(cur, nx)))
                seg.append(nx); prev, cur = cur, nx
            segments.append(seg)

    # Animation joints: structural + every-3rd interior
    joints = set(structural)
    for seg in segments:
        for i in range(0, len(seg), 3):
            joints.add(seg[i])
    joints = sorted(joints)

    return degrees, structural, edges, segments, joints


def _pick_root(nodes, structural):
    centroid = nodes.mean(axis=0)
    return structural[int(np.argmin(
        np.linalg.norm(nodes[structural] - centroid, axis=1)))]


def _build_children(n, root, adj):
    children = [[] for _ in range(n)]
    visited = set([root])
    queue = deque([root])
    while queue:
        nd = queue.popleft()
        for nb in np.where(adj[nd] > 0)[0]:
            if nb not in visited:
                visited.add(nb)
                children[nd].append(nb)
                queue.append(nb)
    return children


def _bind_points(points, nodes, edges):
    """Bind surface points to nearest skeleton edge (vectorised)."""
    N = len(points)
    ea, eb = nodes[edges[:, 0]], nodes[edges[:, 1]]
    ev = eb - ea
    el2 = np.maximum(np.sum(ev ** 2, axis=1), 1e-12)

    node_a = np.empty(N, dtype=np.int32)
    node_b = np.empty(N, dtype=np.int32)
    wa = np.empty(N, dtype=np.float64)
    offsets = np.empty((N, 3), dtype=np.float64)

    BS = 512
    for s in range(0, N, BS):
        e = min(s + BS, N)
        pv = points[s:e, None, :] - ea[None, :, :]
        t = np.clip(np.sum(pv * ev[None], axis=2) / el2, 0, 1)
        closest = ea[None] + t[:, :, None] * ev[None]
        d = np.linalg.norm(points[s:e, None, :] - closest, axis=2)
        idx = np.argmin(d, axis=1)
        b = np.arange(e - s)
        t_val = t[b, idx]
        node_a[s:e] = edges[idx, 0]
        node_b[s:e] = edges[idx, 1]
        wa[s:e] = 1.0 - t_val
        skel_pos = nodes[edges[idx, 0]] * (1 - t_val[:, None]) \
                   + nodes[edges[idx, 1]] * t_val[:, None]
        offsets[s:e] = points[s:e] - skel_pos

    return node_a, node_b, wa, 1.0 - wa, offsets


def _joint_axes(nodes, adj, joints):
    axes = {}
    for j in joints:
        nbrs = np.where(adj[j] > 0)[0]
        if len(nbrs) == 0:
            axes[j] = np.array([0, 0, 1.0]); continue
        dirs = nodes[nbrs] - nodes[j]
        avg = dirs.mean(axis=0)
        n = np.linalg.norm(avg)
        if n < 1e-8:
            avg = dirs[0]; n = np.linalg.norm(avg)
        tang = avg / (n + 1e-12)
        up = np.array([0, 0, 1.0])
        perp = np.cross(tang, up)
        if np.linalg.norm(perp) < 0.1:
            perp = np.cross(tang, np.array([0, 1, 0.0]))
        axes[j] = perp / (np.linalg.norm(perp) + 1e-12)
    return axes


def _densify_wire_along_skeleton(pcl, seg, point_rgb, na, nb, offsets,
                                 nodes, edges, tube_radius_world,
                                 axial_step_world, ring_n):
    """Phase 26.1 geometry-aware wire thickening.

    Build a DENSE, CONTINUOUS tube of wire points along the skeleton so the soft
    compositor can splat them with a SMALL footprint and reach a controllable
    on-screen width WITHOUT blobbing (the P26.0 failure mode of fat discs on the
    sparse 4096-pt cloud). For every skeleton edge that carries at least one
    wire (class-0) surface point, the edge centerline is resampled at
    ``axial_step_world`` and a ring of ``ring_n`` points at ``tube_radius_world``
    is placed in the plane perpendicular to the edge tangent. The cross-section
    is a SOLID disc (centerline + concentric rings) so the projected tube is a
    solid band of width ≈ 2·radius at any view angle — a hollow shell ring would
    project side-on to two parallel lines and render at native (not target)
    width. Disc points inherit the BGR of the nearest original wire surface point
    bound to that edge, and label 0 (wire). Returns the EXTRA (points, seg, rgb)
    to APPEND to the rest-
    pose harness arrays (the original 4096 are left untouched, so the persisted
    npys and every other consumer are identical). The extras are render-only
    and are sliced off via ``[:n_pts_orig]`` exactly like the P18 density extras.

    Geometry-only: depends solely on the rest-pose skeleton + bindings, so it is
    deterministic (no rng) and adds NOTHING to any rng stream — every other
    lever stays unchanged regardless of this knob.
    """
    seg = np.asarray(seg)
    wire = seg == 0
    if not np.any(wire) or ring_n < 3:
        return (np.empty((0, 3), pcl.dtype),
                np.empty((0,), seg.dtype),
                np.empty((0, 3), np.asarray(point_rgb).dtype))

    # Edges that carry wire surface points (keyed canonically). Each wire point
    # binds to one edge via (na, nb); collect the set of such edges and, for
    # colour transfer, the per-edge list of original wire surface points.
    wa_idx = na[wire].astype(np.int64)
    wb_idx = nb[wire].astype(np.int64)
    wire_pts = pcl[wire]
    wire_rgb = np.asarray(point_rgb)[wire]

    edge_pts: dict = {}      # (a,b) -> list of (worldpt, rgb)
    for k in range(wire_pts.shape[0]):
        a, b = int(wa_idx[k]), int(wb_idx[k])
        key = (a, b) if a <= b else (b, a)
        edge_pts.setdefault(key, []).append((wire_pts[k], wire_rgb[k]))

    out_pts, out_rgb = [], []
    for (a, b), members in edge_pts.items():
        pa, pb = nodes[a], nodes[b]
        ev = pb - pa
        L = float(np.linalg.norm(ev))
        if L < 1e-9:
            continue
        tang = ev / L
        # Orthonormal basis (u, w) spanning the cross-section plane.
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(tang, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(tang, ref)
        u /= (np.linalg.norm(u) + 1e-12)
        w = np.cross(tang, u)
        w /= (np.linalg.norm(w) + 1e-12)

        # Centerline samples along the edge (inclusive of both ends).
        n_ax = max(int(np.ceil(L / max(axial_step_world, 1e-9))) + 1, 2)
        ts = np.linspace(0.0, 1.0, n_ax)
        centers = pa[None, :] + ts[:, None] * ev[None, :]

        # Colour-per-center: nearest original wire surface point on this edge,
        # so the tube keeps the cable's real texture variation along its run.
        mem_pos = np.array([m[0] for m in members])
        mem_rgb = np.array([m[1] for m in members])
        # Project members onto the edge param to pick the nearest by t.
        mt = np.clip(((mem_pos - pa[None, :]) @ ev) / (L * L), 0.0, 1.0)
        nearest = np.argmin(np.abs(ts[:, None] - mt[None, :]), axis=1)
        center_rgb = mem_rgb[nearest]

        # SOLID disc cross-section per center (centerline + concentric rings out
        # to the target radius). A HOLLOW shell ring projects side-on to two
        # parallel lines with a hollow gap between them — the tube then renders
        # at native (not target) width. Filling the disc makes the projected
        # tube a SOLID band of width ≈ 2·tube_radius at any view angle, fully
        # covered by the small splat footprint. ``ring_n`` sets the angular
        # density on the OUTER ring; inner rings carry proportionally fewer
        # points (≈ uniform areal density) so we don't over-generate near the
        # axis. ``cross`` accumulates the (radius-scaled) ring offsets.
        radii = np.arange(1, ring_n + 1) / float(ring_n) * tube_radius_world
        cross = [np.zeros((1, 3))]  # the centerline point itself
        for ri in radii:
            n_a = max(int(round(ring_n * ri / tube_radius_world)), 1)
            angs = np.linspace(0.0, 2.0 * np.pi, n_a, endpoint=False)
            ring = (np.cos(angs)[:, None] * u[None, :]
                    + np.sin(angs)[:, None] * w[None, :]) * ri
            cross.append(ring)
        cross = np.concatenate(cross, axis=0)            # (M, 3) disc offsets
        # (n_ax, M, 3): every center gets the full disc cross-section.
        pts = centers[:, None, :] + cross[None, :, :]
        out_pts.append(pts.reshape(-1, 3))
        out_rgb.append(np.repeat(center_rgb, cross.shape[0], axis=0))

    if not out_pts:
        return (np.empty((0, 3), pcl.dtype),
                np.empty((0,), seg.dtype),
                np.empty((0, 3), np.asarray(point_rgb).dtype))

    add_pts = np.concatenate(out_pts, axis=0).astype(pcl.dtype)
    add_rgb = np.concatenate(out_rgb, axis=0).astype(np.asarray(point_rgb).dtype)
    add_seg = np.zeros(add_pts.shape[0], dtype=seg.dtype)  # class 0 = wire
    return add_pts, add_seg, add_rgb


def _forward_kinematics(nodes, root, children, joint_rotations):
    n = len(nodes)
    node_R = np.tile(np.eye(3), (n, 1, 1))
    new_pos = nodes.copy()
    queue = deque([root])
    visited = set([root])
    while queue:
        nd = queue.popleft()
        for ch in children[nd]:
            if ch in visited:
                continue
            visited.add(ch)
            pR = node_R[nd]
            if nd in joint_rotations:
                axis, angle = joint_rotations[nd]
                cR = _rotation_matrix(axis, angle) @ pR
            else:
                cR = pR
            node_R[ch] = cR
            new_pos[ch] = new_pos[nd] + cR @ (nodes[ch] - nodes[nd])
            queue.append(ch)
    return new_pos, node_R


def _animate_points_fast(wa, wb, na, nb, offsets, new_nodes, node_R):
    new_skel = new_nodes[na] * wa[:, None] + new_nodes[nb] * wb[:, None]
    R_blend = wa[:, None, None] * node_R[na] + wb[:, None, None] * node_R[nb]
    U, _, Vt = np.linalg.svd(R_blend)
    det = np.linalg.det(U @ Vt)
    flip = det < 0
    if np.any(flip):
        U[flip, :, -1] *= -1
    R_ortho = U @ Vt
    new_off = np.einsum("nij,nj->ni", R_ortho, offsets)
    return new_skel + new_off


# ── Per-sample worker ──────────────────────────────────────────────────────

def _video_is_done(out_base, src_frame, num_anim, view_names):
    """Check if all output files for this source frame exist."""
    rgb_dir = out_base / "rgb"
    depth_dir = out_base / "depth"
    label_dir = out_base / "label"
    label3_dir = out_base / "label3"
    require_label3 = (DATASET_MODE == "phase11")
    for ai in range(num_anim):
        for vn in view_names:
            fname = f"{src_frame:04d}_{ai:02d}_{vn}.png"
            if not (rgb_dir / fname).exists():
                return False
            if not (depth_dir / fname).exists():
                return False
            if not (label_dir / fname).exists():
                return False
            if require_label3 and not (label3_dir / fname).exists():
                return False
    return True


def convert_one_video(args):
    """Convert one source frame into one set of rendered RGB-D images.

    For ``DATASET_MODE == "v2"`` this is an animated 20-anim × 6-view clip on
    a Phase 4 single-floor backdrop (the v2 behaviour, preserved verbatim).
    For ``DATASET_MODE == "phase9"`` this is a *single* anim frame × 6
    randomised-camera views on a multi-wall Phase 9 backdrop, with the
    harness texture per-source HSV-jittered to widen the DLO colour
    manifold.

    Work unit = (set_id, source_frame_id).
    Returns (set_id, frame_id, status, elapsed).
    """
    set_id, frame_id, num_frames, max_angle_deg, output_root = args
    t0 = time.time()
    set_str = f"{set_id:03d}"
    split = split_of(set_id)

    # Phase 9, Phase 11, Phase 12, and the Phase 10 ablation paths all render
    # a single anim frame per source. Phase 11 and Phase 12 follow Phase 4's
    # training convention (anim 0 only is consumed by the DFormer stager), so
    # rendering further anim frames is wasted work; phase9 + ablation focus
    # on scene composition rather than within-source temporal augmentation.
    if (DATASET_MODE in ("phase9", "phase11", "phase12", "phase13")
            or ABL_VARIANT is not None):
        num_frames = 1

    try:
        out_base = Path(output_root) / split / set_str
        if _video_is_done(out_base, frame_id, num_frames, VIEW_NAMES):
            return (set_id, frame_id, "skipped", 0.0)

        base = DATA_ROOT / set_str
        pcl = np.load(str(base / "pointclouds_normed_4096" / f"pcl_{frame_id:04d}.npy"))
        seg = np.load(str(base / "segmentation_normed_4096" / f"seg_{frame_id:04d}.npy"))
        skel = np.load(str(base / "skeletons" / f"{frame_id:03d}.npz"))
        nodes, adj = skel["nodes"], skel["adj"]

        # Topology (cheap to recompute, ~10 ms)
        degrees, structural, edges, segments, joints = _build_topology(adj)
        root = _pick_root(nodes, structural)
        children = _build_children(len(nodes), root, adj)

        # Bind + axes
        na, nb, wa, wb, offsets = _bind_points(pcl, nodes, edges)
        axes = _joint_axes(nodes, adj, joints)

        # Phase 18 lever (thickness): fatten the harness cross-section about
        # its centerline by k ~ U[lo, hi], drawn once per source from a
        # SEPARATE rng (offset +502) ONLY when the knob is set, so lever-OFF
        # renders are the same as Phase 15. Scaling the binding offsets
        # (never the skeleton) and rebuilding the surface points keeps every
        # downstream consumer consistent automatically: texture UVs (v is the
        # offset ANGLE — scale-invariant), animation (LBS re-applies the
        # scaled offsets), labels/depth (same points, same seg array). The
        # scene bbox below sees the fattened pcl, so floor/backdrop placement
        # still clears the wire surface.
        n_pts_orig = len(pcl)   # extras below are render-only; npys stay 4096
        if DATASET_MODE == "phase13" and P18_THICK_RANGE is not None:
            k_rng = np.random.RandomState(set_id * 1000 + frame_id + 502)
            thick_k = float(k_rng.uniform(*P18_THICK_RANGE))
            skel_pos = pcl - offsets   # _bind_points: offsets = pcl - skel_pos
            offsets = offsets * thick_k
            pcl = skel_pos + offsets
            # Restore surface density: the tube circumference grew by k but
            # the point count didn't, so a fat tube splats with pinholes
            # (measured ~19 % interior hole px at k=2.6; uniform-RANDOM
            # refill still left 8 % — Poisson gaps). Deterministic fix: give
            # every surface point ceil(k)-1 copies rotated about its bound
            # edge's tangent at evenly spaced angles, splitting each ring gap
            # into ceil(k) parts ⇒ max circumferential gap = k/ceil(k) ≤ 1x
            # the original spacing, which the splat+close already renders
            # solid. Copies inherit their base point's label + binding so
            # texturing / labels / animation handle them like any point.
            # Noise (class 4) points are off-surface scatter — not copied.
            n_copies = max(int(np.ceil(thick_k)) - 1, 0)
            surf = np.where(seg.astype(int) != 4)[0]
            if n_copies > 0 and surf.size:
                tang = nodes[nb[surf]] - nodes[na[surf]]
                tang /= np.maximum(
                    np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
                off_b = offsets[surf]
                skel_b = pcl[surf] - off_b
                t_dot = np.sum(tang * off_b, axis=1, keepdims=True)
                t_cross = np.cross(tang, off_b)
                new_pts, new_off = [], []
                for j in range(1, n_copies + 1):
                    ang = 2.0 * np.pi * j / (n_copies + 1)
                    # Rodrigues rotation of off_b about tang by ang.
                    off_new = (off_b * np.cos(ang) + t_cross * np.sin(ang)
                               + tang * t_dot * (1.0 - np.cos(ang)))
                    new_pts.append(skel_b + off_new)
                    new_off.append(off_new)
                rep = np.tile(surf, n_copies)
                pcl = np.concatenate([pcl] + new_pts)
                offsets = np.concatenate([offsets] + new_off)
                seg = np.concatenate([seg, seg[rep]])
                na = np.concatenate([na, na[rep]])
                nb = np.concatenate([nb, nb[rep]])
                wa = np.concatenate([wa, wa[rep]])
                wb = np.concatenate([wb, wb[rep]])

        # Texture mapping: compute per-point BGR ONCE at rest pose. Reusable
        # across all animation frames because LBS deforms positions but never
        # reorders points, so each point's identity (and its texture sample)
        # remains stable.
        texture_library = _get_texture_library()
        # Phase 18 lever (wire-colour pool): extra swatch textures for the
        # wire class, drawn from a SEPARATE rng (offset +501) inside
        # compute_per_point_rgb. None when the knob is unset ⇒ identical.
        wire_ext = (_get_wire_ext_library()
                    if (DATASET_MODE == "phase13" and P18_WIRETEX_DIR)
                    else None)
        point_rgb = compute_per_point_rgb(
            pcl=pcl, labels=seg, nodes=nodes, edges=edges, segments=segments,
            na=na, nb=nb, wa=wa, wb=wb, offsets=offsets,
            texture_library=texture_library,
            seed=set_id * 1000 + frame_id,
            wire_ext_library=wire_ext,
            wire_ext_seed=set_id * 1000 + frame_id + 501,
        )

        # Phase 9: per-source HSV jitter on the harness BGR. Breaks the
        # "harness texture statistics = DLO" shortcut that v2 over-fit to.
        # Random shift in hue [-25°, 25°], sat scale [0.75, 1.25], val
        # scale [0.80, 1.20]. Stays inside §0.5: colour mutation happens in
        # the PCL BGR array BEFORE rasterisation, never as image post-fx.
        # Phase 10 ablation 'harness_hsv' variant applies the SAME jitter on
        # top of the Phase 4 baseline to isolate this single lever.
        if DATASET_MODE == "phase9" or "harness_hsv" in ABL_LEVERS:
            point_rgb = jitter_per_point_rgb(
                point_rgb,
                rng=np.random.RandomState(set_id * 1000 + frame_id + 233),
            )

        # Phase 26.1 lever: GEOMETRY-AWARE wire thickening (densify-then-thin-
        # splat). Build a dense, continuous tube of class-0 wire points along the
        # skeleton (centerline resample x small SOLID radial disc) so the soft
        # compositor reaches the real ~14px width with a SMALL footprint and
        # WITHOUT the P26.0 blobbing. The extras are kept in a SEPARATE buffer
        # (NOT concatenated into ``pcl``) and merged into the combined render
        # cloud only at the combine step below — so the harness bbox (which
        # drives floor/backdrop placement + the lighting centre) and every
        # persisted (4096,) npy stay the same, and ONLY the wire pixels
        # change. Deterministic (no rng) ⇒ every other lever's rng stream is
        # untouched. Only on the strict-Phase-4 (phase13) base.
        wire_extra_pts = np.empty((0, 3), dtype=pcl.dtype)
        wire_extra_rgb = np.empty((0, 3), dtype=np.uint8)
        if DATASET_MODE == "phase13" and P26_DENSIFY:
            wire_extra_pts, _wx_seg, wire_extra_rgb = \
                _densify_wire_along_skeleton(
                    pcl=pcl, seg=seg, point_rgb=point_rgb, na=na, nb=nb,
                    offsets=offsets, nodes=nodes, edges=edges,
                    tube_radius_world=P26_TUBE_RADIUS_PX / SCALE,
                    axial_step_world=P26_AXIAL_STEP_PX / SCALE,
                    ring_n=P26_RING_N,
                )

        # Phase 15 lever: decide (once per source) whether this source is
        # WIRE-FREE. SEPARATE rng (offset +401) drawn ONLY when the prob is
        # > 0, so with the lever OFF (default 0.0) no rng is consumed and the
        # render is byte-identical to Phase 14. Wire-free only applies on the
        # strict-Phase-4 (phase13) base. The scene below is still composed
        # using the harness bbox exactly as normal; the harness points/labels
        # are dropped at the combine step and the near-wire hands are skipped.
        wirefree = False
        if (DATASET_MODE == "phase13" and P15_WIREFREE_P > 0.0):
            wf_rng = np.random.RandomState(set_id * 1000 + frame_id + 401)
            wirefree = bool(wf_rng.uniform(0.0, 1.0) < P15_WIREFREE_P)

        # ── Background scene ─────────────────────────────────────────────
        bg_library = _get_background_library()
        obj_library = _get_object_library()
        # Resolve the object library according to mode / ablation variant.
        # Phase 9 and the 'objects' ablation variant use the Phase-9-retained
        # (42-entry) library; Phase 11 and the other ablation variants use
        # the procedural "clutter" library (keyboard/monitor/etc.); Phase 12
        # restores the STRICT Phase 4 baseline of the 21 Polyhaven CC0
        # mesh-derived originals (categories: tool / container / lighting /
        # electronics / decor / kitchenware / signage / appliance), which is
        # what the Phase 7 teacher was trained on (CONTEXT.md §1.5). Default
        # v2 uses the full 264-entry library.
        if DATASET_MODE == "phase9" or "objects" in ABL_LEVERS:
            obj_lib_for_scene = filter_library_by_category(
                obj_library or [], exclude=PHASE9_DROP_CATEGORIES,
            )
        elif DATASET_MODE in _STRICT_P4_MODES:
            obj_lib_for_scene = filter_library_by_category(
                obj_library or [],
                exclude={"hand", "gripper", "arm",
                         "negative_wire_like", "rope", "clutter"},
            )
            # Phase 14 lever: append dark/black confuser negatives (cylinders /
            # sharp edges / hands) to the background clutter pool. SEPARATE rng
            # ⇒ base composition unchanged when the lever is OFF.
            if DATASET_MODE == "phase13" and P14_NEGATIVES:
                obj_lib_for_scene = obj_lib_for_scene + build_confuser_negatives(
                    obj_library or [],
                    rng=np.random.RandomState(set_id * 1000 + frame_id + 877),
                    **P14_NEG_KWARGS,
                )
        elif DATASET_MODE == "phase11" or ABL_VARIANT is not None:
            obj_lib_for_scene = filter_library_by_category(
                obj_library or [], categories={"clutter"},
            )
        else:
            obj_lib_for_scene = obj_library

        # Choose room scene: Phase 9 multi-wall enclosure (also for the
        # 'wall' ablation variant), or the v2 single-floor + photo backdrop.
        use_phase9_room = (DATASET_MODE == "phase9") or ("wall" in ABL_LEVERS)
        if use_phase9_room:
            if bg_library:
                bg_pcl, bg_rgb = generate_phase9_room_scene(
                    rng=np.random.RandomState(set_id * 1000 + frame_id + 7),
                    bbox_min=pcl.min(axis=0),
                    bbox_max=pcl.max(axis=0),
                    texture_library=bg_library,
                    object_library=obj_lib_for_scene,
                    n_points=(BG_N_POINTS
                              if DATASET_MODE == "phase9"
                              else 240000),  # phase9 budget for wall coverage
                )
            else:
                bg_pcl = np.empty((0, 3))
                bg_rgb = np.empty((0, 3), dtype=np.uint8)
        else:
            if bg_library:
                # Phase 12 / Phase 13 revert to Phase 4's sparser clutter count
                # (3-6 objects per source, half of v2's 10-15) — see §4.20 #1.
                bg_n_objects_range = (
                    (P17_NOBJ_LO, P17_NOBJ_HI) if DATASET_MODE in _STRICT_P4_MODES else (10, 16)
                )
                # Phase 13 lever (b): per-object colour gradient, driven by a
                # SEPARATE rng so the scene-composition rng stream (object
                # choice / placement / floor) is byte-identical to the control.
                p13_objgrad = (DATASET_MODE == "phase13" and P13_OBJGRAD)
                # Phase 19 lever (busy floor): with prob P19_BUSYFLOOR_P the
                # floor TEXTURE is drawn from the busy pool. SEPARATE rng
                # (offset +602; decision first, pool choice second) created
                # ONLY when the prob > 0, so the lever-OFF path is
                # identical to Phase 18. The override swaps the texture
                # AFTER the scene rng's own (discarded) floor-texture draw
                # inside generate_background_scene — point positions, object
                # choice/placement and labels are untouched.
                p19_floor_tex = None
                if (DATASET_MODE == "phase13" and P19_BUSYBG_DIR
                        and P19_BUSYFLOOR_P > 0.0):
                    bf_rng = np.random.RandomState(
                        set_id * 1000 + frame_id + 602)
                    if bf_rng.uniform(0.0, 1.0) < P19_BUSYFLOOR_P:
                        busy_raw = _get_busy_library_raw()
                        p19_floor_tex = busy_raw[
                            bf_rng.randint(0, len(busy_raw))]
                bg_pcl, bg_rgb = generate_background_scene(
                    rng=np.random.RandomState(set_id * 1000 + frame_id + 7),
                    bbox_min=pcl.min(axis=0),
                    bbox_max=pcl.max(axis=0),
                    texture_library=bg_library,
                    object_library=obj_lib_for_scene,
                    n_points=(int(BG_N_POINTS * P14_NEG_FIDELITY)
                              if (DATASET_MODE == "phase13" and P14_NEGATIVES)
                              else BG_N_POINTS),
                    n_objects_range=bg_n_objects_range,
                    object_color_gradient=p13_objgrad,
                    grad_rng=(np.random.RandomState(set_id * 1000 + frame_id + 619)
                              if p13_objgrad else None),
                    grad_kwargs=(P13_OBJGRAD_KWARGS if p13_objgrad else None),
                    floor_tex_override=p19_floor_tex,
                )
            else:
                bg_pcl = np.empty((0, 3))
                bg_rgb = np.empty((0, 3), dtype=np.uint8)

        # ── Foreground scene ─────────────────────────────────────────────
        # Phase 9 and the 'objects' ablation variant use phase9_foreground
        # (0-2 benign occluders). Phase 11 uses phase11_foreground (at most
        # 1 off-wire hand). Phase 12 and the other ablation variants emit
        # NO foreground (Phase 4 baseline has no fg occluders by spec); the
        # v2 default keeps the v2 high-density hand/cable foreground.
        if DATASET_MODE == "phase9" or "objects" in ABL_LEVERS:
            fg_pcl, fg_rgb, fg_info = generate_phase9_foreground(
                rng=np.random.RandomState(set_id * 1000 + frame_id + 113),
                object_library=obj_lib_for_scene,
                bbox_min=pcl.min(axis=0),
                bbox_max=pcl.max(axis=0),
                n_points=(FG_N_POINTS if DATASET_MODE == "phase9" else 6000),
                max_objects=2,
            )
        elif DATASET_MODE == "phase11":
            # Phase 11: filter the full library to "hand" entries so the
            # phase11 foreground placer has the 120 hand variants to draw
            # from regardless of which (clutter-only) library was used for
            # the background.
            hand_lib = filter_library_by_category(
                obj_library or [], categories={"hand"},
            )
            fg_pcl, fg_rgb, fg_info = generate_phase11_foreground(
                rng=np.random.RandomState(set_id * 1000 + frame_id + 113),
                object_library=hand_lib,
                bbox_min=pcl.min(axis=0),
                bbox_max=pcl.max(axis=0),
                n_points=FG_N_POINTS,
                p_hand=PHASE11_P_HAND,
            )
        elif DATASET_MODE in _STRICT_P4_MODES or ABL_VARIANT is not None:
            # Phase 14 lever: pose hands gripping the wire as labeled-BACKGROUND
            # foreground (both bg_pcl and fg_pcl get BG_LABEL below), so the
            # model learns "hand near/holding the wire = NOT wire" — the real
            # deployment false positive. SEPARATE rng keeps the lever-OFF path
            # byte-identical. Strict Phase 4 otherwise has no foreground.
            if (DATASET_MODE == "phase13" and P14_NEGATIVES
                    and P14_NEG_NHAND_WIRE > 0 and not wirefree):
                hand_lib = filter_library_by_category(
                    obj_library or [], categories={"hand"})
                hrng = np.random.RandomState(set_id * 1000 + frame_id + 911)
                # Half the source frames get a wire-gripping hand, half don't
                # (so "wire ⇒ hand" is never a learnable shortcut). The draw is
                # the FIRST consumption of hrng; on a miss we emit empty fg.
                place_hand = hrng.uniform(0.0, 1.0) < P14_NEG_P_HAND
                h_pts, h_rgb = [], []
                for _ in range(P14_NEG_NHAND_WIRE if place_hand else 0):
                    if not hand_lib:
                        break
                    base = hand_lib[hrng.randint(0, len(hand_lib))]
                    ho = dict(base)
                    # Up-sample for resolution and enlarge the grasp scale so
                    # the near-wire hand reads as a big, crisp hand.
                    ho["points"], ho["colors"] = densify_point_cloud(
                        base["points"], base["colors"], P14_NEG_HAND_PTS, hrng)
                    lo, hi = ho.get("natural_scale_range", (0.18, 0.22))
                    ho["natural_scale_range"] = (lo * P14_NEG_HAND_SCALE,
                                                 hi * P14_NEG_HAND_SCALE)
                    res = _place_hand_on_wire(
                        ho, hrng, nodes, segments, n_keep=None)
                    if res is not None:
                        h_pts.append(res[0])
                        h_rgb.append(res[1])
                if h_pts:
                    fg_pcl = np.concatenate(h_pts, axis=0)
                    fg_rgb = np.concatenate(h_rgb, axis=0)
                    fg_info = {"placed": [], "n_points": int(len(fg_pcl)),
                               "counts": {"p14_neg_wire_hands": len(h_pts)}}
                else:
                    fg_pcl = np.empty((0, 3))
                    fg_rgb = np.empty((0, 3), dtype=np.uint8)
                    fg_info = {"placed": [], "n_points": 0,
                               "counts": {"p14_neg_wire_hands": 0}}
            else:
                fg_pcl = np.empty((0, 3))
                fg_rgb = np.empty((0, 3), dtype=np.uint8)
                fg_info = {"placed": [], "n_points": 0,
                           "counts": {"phase4_strict_no_fg": 0}}
        else:
            fg_pcl, fg_rgb, fg_info = generate_foreground_scene(
                rng=np.random.RandomState(set_id * 1000 + frame_id + 113),
                object_library=obj_library or [],
                bbox_min=pcl.min(axis=0),
                bbox_max=pcl.max(axis=0),
                skeleton_nodes=nodes,
                segments=segments,
                n_points=FG_N_POINTS,
            )

        # Phase 16 lever: with probability P16_OPENHAND_P, append ONE OPEN /
        # SPLAYED hand to the foreground PCL (labeled BACKGROUND below, like the
        # Phase-14 grip hands). SEPARATE rng (offset +1217) drawn ONLY when the
        # prob > 0, and ONLY after the existing fg block has fully consumed its
        # own rng streams, so the lever-OFF path is byte-identical to Phase 15.
        # Applies to BOTH wired and wire-free frames (an open hand with no wire
        # is also a valid negative). Only on the strict-Phase-4 (phase13) base.
        placed_open_hand = False
        if DATASET_MODE == "phase13" and P16_OPENHAND_P > 0.0:
            oh_rng = np.random.RandomState(set_id * 1000 + frame_id + 1217)
            # The draw is the FIRST consumption of oh_rng; on a miss nothing is
            # placed and no further rng is consumed.
            if oh_rng.uniform(0.0, 1.0) < P16_OPENHAND_P:
                open_lib = [
                    o for o in (obj_library or [])
                    if o.get("category") == "hand"
                    and o.get("pose_family") in _OPEN_HAND_POSE_FAMILIES
                ]
                # Bias toward the two clearest broadside splays (palm facing
                # the camera, fingers fanned): flat_palm_down / spread_fan —
                # the exact splayed appearance the model misfires on. The
                # fronto-parallel ``open_palm_up`` fingers point down and read
                # less crisply, so it is the minority pose.
                hero = [o for o in open_lib if o.get("pose_family") in
                        ("flat_palm_down", "spread_fan")]
                pool = hero if (hero and oh_rng.uniform(0.0, 1.0) < 0.8) \
                    else open_lib
                if pool:
                    base = pool[oh_rng.randint(0, len(pool))]
                    ho = dict(base)
                    # Up-sample for resolution + widen the natural scale so the
                    # splayed fingers read large/crisp (cf. Phase-14 grip hands).
                    ho["points"], ho["colors"] = densify_point_cloud(
                        base["points"], base["colors"], P16_OPENHAND_PTS, oh_rng)
                    lo, hi = ho.get("natural_scale_range", (0.27, 0.33))
                    ho["natural_scale_range"] = (lo * P16_OPENHAND_SCALE,
                                                 hi * P16_OPENHAND_SCALE)
                    oh_pts, oh_rgb = _place_open_hand(
                        ho, oh_rng,
                        bbox_min=pcl.min(axis=0), bbox_max=pcl.max(axis=0))
                    fg_pcl = np.concatenate([fg_pcl, oh_pts], axis=0)
                    fg_rgb = np.concatenate(
                        [fg_rgb, oh_rgb.astype(np.uint8)], axis=0)
                    placed_open_hand = True
                    fg_info.setdefault("counts", {})["p16_open_hand"] = 1
                    fg_info["n_points"] = int(len(fg_pcl))

        # 2D photographic backdrop. v2 layers it under the rendered scene.
        # Phase 9 instead encloses the scene with 4-5 wall planes (the
        # walls fill the frustum from every randomised viewpoint) → no
        # 2D backdrop needed; we pass None so the rasteriser keeps any
        # un-splatted pixels as 0 (which gets filled by the morphological
        # closing if the closest neighbour is a wall point).
        # The Phase 10 ablation 'wall' variant matches Phase 9 here: when the
        # multi-wall room is generated, the 2D backdrop is dropped so the
        # wall lever is isolated.
        if DATASET_MODE == "phase9" or "wall" in ABL_LEVERS:
            photo_background = None
        elif bg_library:
            photo_background = bg_library[
                (set_id * 1000 + frame_id) % len(bg_library)
            ]
            # Phase 19 lever (busy backdrop): with prob P19_BUSYBG_P the 2D
            # backdrop is drawn uniformly from the busy pool instead of the
            # deterministic selection above. SEPARATE rng (offset +601;
            # decision is the FIRST draw, the pool choice the second) created
            # ONLY when the knob is set, so the lever-OFF path is
            # identical to Phase 18. Colours only — depth/labels never
            # see the backdrop.
            if DATASET_MODE == "phase13" and P19_BUSYBG_DIR:
                bb_rng = np.random.RandomState(
                    set_id * 1000 + frame_id + 601)
                if bb_rng.uniform(0.0, 1.0) < P19_BUSYBG_P:
                    busy_lib = _get_busy_backdrop_library()
                    photo_background = busy_lib[
                        bb_rng.randint(0, len(busy_lib))]
        else:
            photo_background = None

        # Phase 15 wire-free: drop the harness BGR + harness labels from the
        # combine so the rendered frame shows the scene with NO wire and the
        # label PNG is ALL background. The harness POSITIONS (new_pcl) are
        # likewise dropped from combined_pcl in the anim loop below. bg/fg are
        # composed exactly as normal (scene looks identical minus the wire).
        if wirefree:
            combined_rgb = np.concatenate(
                [bg_rgb, fg_rgb], axis=0).astype(np.uint8)
            combined_labels = np.concatenate([
                np.full(len(bg_pcl), BG_LABEL, dtype=np.int64),
                np.full(len(fg_pcl), BG_LABEL, dtype=np.int64),
            ])
        else:
            combined_rgb = np.concatenate(
                [point_rgb, bg_rgb, fg_rgb], axis=0).astype(np.uint8)
            combined_labels = np.concatenate([
                seg.astype(np.int64),
                np.full(len(bg_pcl), BG_LABEL, dtype=np.int64),
                np.full(len(fg_pcl), BG_LABEL, dtype=np.int64),
            ])
            # Phase 26.1: append the geometry-aware densified wire tube (class 0
            # = wire) AFTER fg, so the render cloud carries it but the harness
            # bbox / lighting centre / scene composition (all from the pristine
            # ``pcl``) are untouched. Skipped on wire-free frames (no wire).
            if wire_extra_pts.shape[0]:
                combined_rgb = np.concatenate(
                    [combined_rgb, wire_extra_rgb], axis=0).astype(np.uint8)
                combined_labels = np.concatenate([
                    combined_labels,
                    np.zeros(wire_extra_pts.shape[0], dtype=np.int64),
                ])

        # Phase 13 lever (a): pick one distant-light direction per source
        # (separate rng). Applied per-anim-frame below once ``combined_pcl``
        # exists; the 2D backdrop is not a point and is therefore never lit.
        p13_lighting = (DATASET_MODE == "phase13" and P13_LIGHTING)
        if p13_lighting:
            light_dir = sample_light_direction(
                np.random.RandomState(set_id * 1000 + frame_id + 517))
            light_centre = 0.5 * (pcl.min(axis=0) + pcl.max(axis=0))
        else:
            light_dir = None
            light_centre = None

        # Pre-build the per-source view rotations. v2 reuses the fixed
        # 6 canonical orientations; phase9 picks 6 random (az, el) per
        # source. The Phase 10 'camera' ablation variant matches Phase 9
        # here (random per-source views) so the camera lever is isolated.
        if _USE_RANDOM_VIEWS:
            cam_rng = np.random.RandomState(set_id * 1000 + frame_id + 911)
            view_rotations: dict[str, np.ndarray] = {}
            view_dirs: dict[str, dict] = {}
            for vn in VIEW_NAMES:
                look, up = random_view_dirs(cam_rng)
                R = make_view_matrix(look, up)
                view_rotations[vn] = R
                view_dirs[vn] = {"look": look.tolist(), "up": up.tolist()}
        else:
            view_rotations = VIEW_ROTATIONS
            view_dirs = None

        rng = np.random.RandomState(set_id * 1000 + frame_id)
        phase = {j: rng.uniform(0, 2 * np.pi) for j in joints}
        freq  = {j: rng.uniform(0.5, 2.0)     for j in joints}
        amp   = {j: rng.uniform(0.3, 1.0)     for j in joints}
        max_ang = np.radians(max_angle_deg)

        rgb_dir     = out_base / "rgb"
        depth_dir   = out_base / "depth"
        pcl_dir     = out_base / "pointclouds"
        lbl_dir     = out_base / "labels"   # per-source per-point .npy (existing)
        lbl_img_dir = out_base / "label"    # per-anim per-view PNG (binary-collapsible: 0=bg, 1..5=harness classes)
        # Phase 11 also writes a parallel per-anim per-view 3-way label PNG:
        # 0 = backdrop (2D photo), 1 = wire (any harness pixel), 2 = objects
        # (floor clutter + hand). Derived from (label_img, depth_img) after
        # rasterisation; no change to the rasteriser itself.
        write_label3 = (DATASET_MODE == "phase11")
        lbl3_img_dir = out_base / "label3" if write_label3 else None

        dirs_to_make = [rgb_dir, depth_dir, pcl_dir, lbl_dir, lbl_img_dir]
        if write_label3:
            dirs_to_make.append(lbl3_img_dir)
        for d in dirs_to_make:
            d.mkdir(parents=True, exist_ok=True)

        lbl_path = lbl_dir / f"{frame_id:04d}.npy"
        if not lbl_path.exists():
            # [:n_pts_orig] drops the Phase 18 render-only density extras
            # (no-op when the lever is off) — the npy keeps the (4096,) shape.
            np.save(str(lbl_path), seg[:n_pts_orig].astype(np.int8))

        # Phase 9: persist the per-source random camera spec so downstream
        # tools (and humans) can reconstruct what view each PNG was rendered
        # from. v2 doesn't need this because views are fixed at module load.
        # Phase 10 'camera' ablation also persists views.
        if _USE_RANDOM_VIEWS:
            view_json = out_base / "views" / f"{frame_id:04d}.json"
            view_json.parent.mkdir(parents=True, exist_ok=True)
            view_json.write_text(json.dumps(view_dirs, indent=2))

        for ai in range(num_frames):
            if num_frames == 1:
                new_pcl = pcl  # rest pose; no animation in phase 9
            else:
                t = ai / max(num_frames - 1, 1)
                jrot = {}
                for j in joints:
                    angle = max_ang * amp[j] * np.sin(
                        2 * np.pi * freq[j] * t + phase[j])
                    jrot[j] = (axes[j], angle)
                new_nodes, node_R = _forward_kinematics(
                    nodes, root, children, jrot)
                new_pcl = _animate_points_fast(
                    wa, wb, na, nb, offsets, new_nodes, node_R)

            np.save(str(pcl_dir / f"{frame_id:04d}_{ai:02d}.npy"),
                    new_pcl[:n_pts_orig].astype(np.float32))

            # Phase 15 wire-free: omit the harness positions so no wire points
            # are rasterised (label PNG already excludes the harness above).
            if wirefree:
                combined_pcl = np.concatenate([bg_pcl, fg_pcl], axis=0)
            else:
                combined_pcl = np.concatenate([new_pcl, bg_pcl, fg_pcl], axis=0)
                # Phase 26.1: append the rest-pose densified wire tube so it
                # aligns with combined_rgb/combined_labels (built above). phase13
                # renders a single rest-pose anim frame (num_frames==1 ⇒
                # new_pcl is the rest pose), so the rest-pose tube matches; the
                # animated path is never taken in phase13 (the only mode this
                # lever runs in).
                if wire_extra_pts.shape[0]:
                    combined_pcl = np.concatenate(
                        [combined_pcl, wire_extra_pts], axis=0)

            # Phase 13 lever (a): bake the directional shading into the
            # per-point BGR for this frame's world positions. View-independent
            # (depends only on world coords) so it is shared across all views.
            view_rgb = combined_rgb
            if light_dir is not None:
                view_rgb = apply_scene_lighting(
                    combined_pcl, combined_rgb, light_dir, light_centre,
                    ambient=P13_LIGHT_AMBIENT, gain=P13_LIGHT_GAIN,
                    span=P13_LIGHT_SPAN,
                )

            for vn in VIEW_NAMES:
                fname = f"{frame_id:04d}_{ai:02d}_{vn}.png"
                color, depth, label_img = rasterize_view(
                    combined_pcl, combined_labels, view_rotations[vn],
                    point_rgb=view_rgb,
                    background=photo_background,
                )
                cv2.imwrite(str(rgb_dir / fname), color)
                cv2.imwrite(str(depth_dir / fname), depth)
                cv2.imwrite(str(lbl_img_dir / fname), label_img)
                if write_label3:
                    # 0=backdrop, 1=wire (any harness class), 2=objects
                    # (3D points whose source label was BG_LABEL).
                    label3_img = np.zeros_like(label_img, dtype=np.uint8)
                    label3_img[(label_img == 0) & (depth > 0)] = 2
                    label3_img[label_img > 0] = 1
                    cv2.imwrite(str(lbl3_img_dir / fname), label3_img)

        return (set_id, frame_id, "ok", time.time() - t0)

    except Exception as e:
        import traceback
        return (set_id, frame_id, f"error: {e}\n{traceback.format_exc()}", time.time() - t0)


# ── Discovery ──────────────────────────────────────────────────────────────

def discover_work(num_frames):
    """Find all valid (set_id, frame_id) pairs."""
    work = []
    sets_found = {}
    for entry in sorted(DATA_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        try:
            sid = int(entry.name)
        except ValueError:
            continue
        needed = ["pointclouds_normed_4096", "segmentation_normed_4096", "skeletons"]
        if not all((entry / d).is_dir() for d in needed):
            continue

        n_pcl  = len(list((entry / "pointclouds_normed_4096").glob("pcl_*.npy")))
        n_seg  = len(list((entry / "segmentation_normed_4096").glob("seg_*.npy")))
        n_skel = len(list((entry / "skeletons").glob("*.npz")))
        usable = min(n_pcl, n_seg, n_skel)
        if usable == 0:
            continue

        sets_found[sid] = usable
        for fid in range(usable):
            work.append(sid, )  # placeholder — filled below

    return sets_found


def build_work_list(sets_found, num_frames, max_angle, output_root,
                    src_stride: int = 1,
                    only_splits: tuple[str, ...] | None = None):
    """Build the multiprocess work list.

    Parameters
    ----------
    sets_found :
        ``{set_id: n_source_frames}`` from :func:`discover_work` /
        :func:`render_full_dataset.discover_sets`.
    num_frames :
        Anim frames per source (Phase 9 forces 1 in the worker).
    max_angle :
        Maximum joint rotation in degrees (ignored when num_frames == 1).
    output_root :
        Render output root.
    src_stride :
        Render every Nth source frame. ``1`` = render every source. Phase 9
        default at the runner level is 5 so the rendered set matches the
        Phase 7 / Phase 4 7,560-train target without any subset step later.
    only_splits :
        If not None, restrict to splits in this tuple (e.g.
        ``("train", "val")`` skips the test split for Phase 9).
    """
    items = []
    for sid in sorted(sets_found):
        if only_splits is not None and split_of(sid) not in only_splits:
            continue
        for fid in range(0, sets_found[sid], max(1, int(src_stride))):
            items.append((sid, fid, num_frames, max_angle, str(output_root)))
    return items


# ── Metadata ───────────────────────────────────────────────────────────────

def write_metadata(sets_found, num_frames, max_angle_deg, stats):
    if _USE_RANDOM_VIEWS:
        views_meta = {
            "scheme": "per-source random orthographic (6 views per source)",
            "azimuth_deg_range": [0.0, 360.0],
            "elevation_deg_range": [-30.0, 75.0],
            "view_slot_names": VIEW_NAMES,
            "per_source_dirs": "see <split>/<set>/views/<src>.json for "
                              "(look, up) per (set, src, view_slot).",
        }
    else:
        views_meta = {}
        for vn, vdef in VIEWS.items():
            R = make_view_matrix(vdef["look"], vdef["up"])
            views_meta[vn] = {
                "look_direction": vdef["look"].tolist(),
                "up_vector": vdef["up"].tolist(),
                "rotation_matrix": R.tolist(),
            }

    splits = {"train": [], "val": [], "test": []}
    for s in sorted(sets_found):
        splits[split_of(s)].append(f"{s:03d}")

    if DATASET_MODE == "phase9":
        meta_description = "Phase 9 multi-wall, randomised-camera RGB-D dataset"
    elif DATASET_MODE == "phase11":
        meta_description = (
            "Phase 11 RGB-D dataset (Phase 4 baseline + light off-wire hand "
            "foreground; expanded CC0 photo backdrop library)"
        )
    elif DATASET_MODE == "phase12":
        meta_description = (
            "Phase 12 RGB-D dataset (STRICT Phase 4 baseline — textured "
            "harness + textured xz-floor + 3-6 Polyhaven CC0 mesh-derived "
            "clutter PCLs + 2D photographic backdrop; the only lever vs "
            "Phase 4 is the size and diversity of the 2D backdrop pool)"
        )
    elif DATASET_MODE == "phase13":
        meta_description = (
            "Phase 13 RGB-D dataset (STRICT Phase 4 baseline + per-point "
            f"appearance levers: 3D scene lighting={P13_LIGHTING}, object "
            f"colour gradients={P13_OBJGRAD}; both baked into per-point BGR "
            "before rasterisation, 2D backdrop left pristine)"
        )
    elif ABL_VARIANT is not None:
        meta_description = (
            f"Phase 10 ablation variant '{ABL_VARIANT}' "
            "(Phase 4 baseline + 1 Phase 9 lever)"
        )
    else:
        meta_description = "Animated RGB-D video dataset from CDLO point clouds"

    if DATASET_MODE == "phase9":
        _bg_meta_block = {
            "method": (
                "Phase 9: 5-wall enclosure (floor, back, front, left, right) "
                "with optional ceiling. Each wall is independently sampled "
                "from data/textures/backgrounds/ AND given its own random "
                "HSV tint, yielding 4-5 visibly distinct shades per source. "
                "3-6 non-wire-shaped floor-clutter PCLs from data/objects/ "
                "(categories filtered by PHASE9_DROP_CATEGORIES) populate "
                "the room. The 2D photographic backdrop kwarg is NOT used."
            ),
            "n_points_per_scene": BG_N_POINTS,
            "library_dir_3d_objects": "data/objects/",
            "library_dir_photos": "data/textures/backgrounds/",
            "phase9_drop_categories": sorted(PHASE9_DROP_CATEGORIES),
            "license": "CC0",
            "seed_per_source": "set_id * 1000 + frame_id + 7",
        }
        _fg_meta_block = {
            "method": (
                "Phase 9: 0-2 benign non-wire occluders placed near the "
                "harness bbox. No hands, no grippers, no wire-shaped "
                "negatives, no grasp-on-wire poses. All foreground points "
                "carry BG_LABEL=255."
            ),
            "n_points_per_scene": FG_N_POINTS,
            "max_objects_per_source": 2,
            "categories_used": "complement of PHASE9_DROP_CATEGORIES",
            "seed_per_source": "set_id * 1000 + frame_id + 113",
        }
    elif DATASET_MODE in _STRICT_P4_MODES:
        _bg_meta_block = {
            "method": (
                f"{DATASET_MODE} (= strict Phase 4): textured xz-floor + 3-6 "
                "Polyhaven CC0 mesh-derived clutter PCLs (categories: tool, "
                "container, lighting, electronics, decor, kitchenware, "
                "signage, appliance — the 21 originals from the Phase 4 "
                "render of 2026-04-27) + 2D photographic backdrop. Phase 13 "
                "additionally bakes per-object colour gradients (lever b) "
                "into the clutter BGR via a separate rng when enabled."
            ),
            "n_points_per_scene": BG_N_POINTS,
            "n_objects_per_source_range": [P17_NOBJ_LO, P17_NOBJ_HI - 1],
            "library_dir_3d_objects": "data/objects/",
            "library_dir_photos": "data/textures/backgrounds/",
            "license": "CC0",
            "seed_per_source": "set_id * 1000 + frame_id + 7",
            "object_color_gradient": (DATASET_MODE == "phase13" and P13_OBJGRAD),
            "object_color_gradient_seed": "set_id * 1000 + frame_id + 619",
        }
        _fg_meta_block = {
            "method": (
                f"{DATASET_MODE}: NO foreground objects (Phase 4 baseline)."
            ),
            "n_points_per_scene": 0,
        }
    elif DATASET_MODE == "phase11":
        _bg_meta_block = {
            "method": (
                "Phase 11: textured xz-floor + Phase 4 'clutter' real-object "
                "PCLs + 2D photographic backdrop. Same as Phase 4."
            ),
            "n_points_per_scene": BG_N_POINTS,
            "library_dir_3d_objects": "data/objects/",
            "library_dir_photos": "data/textures/backgrounds/",
            "license": "CC0",
            "seed_per_source": "set_id * 1000 + frame_id + 7",
        }
        _fg_meta_block = {
            "method": (
                "Phase 11: at most ONE hand from the 120-entry CC0 hand "
                "library, placed in the foreground such that its 3D AABB "
                "does NOT intersect the harness AABB (safety_margin "
                "0.08 wu). p_hand = KIAT_PHASE11_P_HAND (default 0.30) of "
                "samples get a hand; the rest look identical to Phase 4. "
                "Hand poses biased toward open / non-grasp (palm up/down, "
                "half-open, relaxed, flat) with bias 0.7."
            ),
            "n_points_per_scene": FG_N_POINTS,
            "p_hand": PHASE11_P_HAND,
            "max_objects_per_source": 1,
            "categories_used": ["hand"],
            "grasp_on_wire": False,
            "safety_margin_wu": 0.08,
            "open_pose_bias": 0.7,
            "seed_per_source": "set_id * 1000 + frame_id + 113",
        }
    else:
        _bg_meta_block = {
            "method": (
                "Phase 4: textured xz-floor + 3-15 real-object CC0 point "
                "clouds + 2D photographic backdrop."
            ),
            "n_points_per_scene": BG_N_POINTS,
            "library_dir_3d_objects": "data/objects/",
            "library_dir_photos": "data/textures/backgrounds/",
            "license": "CC0",
            "seed_per_source": "set_id * 1000 + frame_id + 7",
        }
        _fg_meta_block = {
            "method": (
                "Phase 8 v2: 8-17 fg objects per source (2-4 grasping hand/"
                "gripper, 2-4 free hands, 3-6 cables, 1-3 other) — see "
                "generate_foreground_scene."
            ),
            "n_points_per_scene": FG_N_POINTS,
            "grasp_probability": 0.7,
            "free_floating_probability": 0.55,
            "categories_used": ["hand", "gripper", "arm",
                                "negative_wire_like", "rope"],
            "seed_per_source": "set_id * 1000 + frame_id + 113",
        }

    meta = {
        "description": meta_description,
        "dataset_mode": DATASET_MODE,
        "abl_variant": ABL_VARIANT,
        "phase13_levers": (
            {"lighting": P13_LIGHTING,
             "object_color_gradient": P13_OBJGRAD,
             "light_ambient": P13_LIGHT_AMBIENT,
             "light_gain": P13_LIGHT_GAIN,
             "light_span": P13_LIGHT_SPAN,
             "light_model": "directional ramp (distant light)",
             "light_dir_seed": "set_id * 1000 + frame_id + 517"}
            if DATASET_MODE == "phase13" else None
        ),
        "animation": (
            {"method": {
                 "phase9":  "static (Phase 9: 1 anim frame per source)",
                 "phase11": "static (Phase 11: 1 anim frame per source)",
                 "phase12": "static (Phase 12: 1 anim frame per source)",
                 "phase13": "static (Phase 13: 1 anim frame per source)",
             }[DATASET_MODE],
             "anim_frames_per_source": num_frames}
            if DATASET_MODE in ("phase9", "phase11", "phase12", "phase13")
            else {"method": "Skeleton-based FK with LBS point binding",
                  "anim_frames_per_source": num_frames,
                  "max_angle_deg": max_angle_deg,
                  "joint_selection": "structural + every 3rd interior skeleton node"}
        ),
        "projection": {
            "type": "orthographic",
            "image_width": IMG_W,
            "image_height": IMG_H,
            "scale_px_per_unit": float(SCALE),
            "frustum_half_vertical": float(FRUSTUM_HALF),
            "frustum_half_horizontal": float(HALF_W),
            "depth_near_mm": DEPTH_NEAR_MM,
            "depth_far_mm": DEPTH_FAR_MM,
        },
        "formats": {
            "rgb": "8-bit BGR PNG 640×480 (textured from real CC0 textures, see texture_mapping)",
            "depth": "16-bit unsigned PNG (mm, 0=no data) 640×480",
            "label": "8-bit grayscale PNG 640×480 (0=bg, 1=Wire, 2=Endpoint, 3=Bifurcation, 4=Connector, 5=Noise)",
            **({"label3": (
                "8-bit grayscale PNG 640×480 (Phase 11 only): "
                "0=backdrop (2D photo), 1=wire (any harness pixel), "
                "2=objects (floor clutter, hand). Derived post-hoc from "
                "(label, depth)."
            )} if DATASET_MODE == "phase11" else {}),
            "pointclouds": "float32 .npy (4096, 3)",
            "labels": "int8 .npy (4096,) per source frame (per-point class labels, 0..4)",
        },
        "naming": {
            "rgb_depth_label": "{src_frame:04d}_{anim_frame:02d}_{view}.png",
            **({"label3": "{src_frame:04d}_{anim_frame:02d}_{view}.png "
                "(same naming as label, written under label3/)"}
               if DATASET_MODE == "phase11" else {}),
            "pointcloud": "{src_frame:04d}_{anim_frame:02d}.npy",
            "labels": "{src_frame:04d}.npy",
        },
        "texture_mapping": {
            "method": "Per-point UV sampling at rest pose. Wire: cylindrical UV using rotation-minimising frame along skeleton segments. Other classes: cluster-based PCA-planar UV.",
            "textures_root": "data/textures/",
            "license": "CC0 (ambientCG)",
            "anti_aliasing": "Gaussian-blurred (sigma=2px) source textures + bilinear sampling",
        },
        "background": _bg_meta_block,
        "foreground": _fg_meta_block,
        "color_jitter": (
            {"per_source_hsv_jitter": True,
             "hue_shift_deg": 25.0,
             "sat_scale": 0.25,
             "val_scale": 0.20,
             "seed_per_source": "set_id * 1000 + frame_id + 233"}
            if DATASET_MODE == "phase9" else None
        ),
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "class_colors_rgb": {str(k): list(v) for k, v in CLASS_COLORS_RGB.items()},
        "views": views_meta,
        "splits": splits,
        "source_frames_per_set": {f"{s:03d}": n for s, n in sorted(sets_found.items())},
        "stats": stats,
    }
    path = OUTPUT_ROOT / "metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert full CDLO dataset to animated RGB-D videos")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=20,
                        help="Animation frames per source frame (default: 20)")
    parser.add_argument("--max-angle", type=float, default=25.0,
                        help="Max joint rotation degrees (default: 25)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Discover
    sets_found = {}
    for entry in sorted(DATA_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        try:
            sid = int(entry.name)
        except ValueError:
            continue
        needed = ["pointclouds_normed_4096", "segmentation_normed_4096", "skeletons"]
        if not all((entry / d).is_dir() for d in needed):
            continue
        n_pcl  = len(list((entry / "pointclouds_normed_4096").glob("pcl_*.npy")))
        n_seg  = len(list((entry / "segmentation_normed_4096").glob("seg_*.npy")))
        n_skel = len(list((entry / "skeletons").glob("*.npz")))
        usable = min(n_pcl, n_seg, n_skel)
        if usable > 0:
            sets_found[sid] = usable

    total_src = sum(sets_found.values())
    n_train = sum(n for s, n in sets_found.items() if split_of(s) == "train")
    n_val   = sum(n for s, n in sets_found.items() if split_of(s) == "val")
    n_test  = sum(n for s, n in sets_found.items() if split_of(s) == "test")
    imgs_total = total_src * args.num_frames * len(VIEW_NAMES) * 2

    print("CDLO → Animated RGB-D Video Dataset (FULL)")
    print(f"  Sets:          {len(sets_found)}")
    print(f"  Source frames: {total_src:,} (train={n_train:,} val={n_val:,} test={n_test:,})")
    print(f"  Anim frames:   {args.num_frames} per source")
    print(f"  Videos total:  {total_src:,}")
    print(f"  Total images:  {imgs_total:,}")
    print(f"  Max angle:     {args.max_angle}°")
    print(f"  Output:        {OUTPUT_ROOT}")
    print(f"  Workers:       {args.workers}")

    if args.dry_run:
        # Estimate time
        per_src_s = 0.21 + args.num_frames * 0.70  # bind + N*(FK+raster)
        est_h = total_src * per_src_s / args.workers / 3600
        est_gb = total_src * args.num_frames * len(VIEW_NAMES) * (6515 + 10387) / 1e9 \
                 + total_src * args.num_frames * 4096 * 3 * 4 / 1e9
        print(f"\n  Estimated time: {est_h:.1f} h")
        print(f"  Estimated size: {est_gb:.0f} GB")
        print("  [dry-run] No files written.")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    work = build_work_list(sets_found, args.num_frames, args.max_angle, OUTPUT_ROOT)

    t_start = time.time()
    done, ok, skipped, errors = 0, 0, 0, []

    print(f"\nConverting {total_src:,} source frames ...\n")

    with Pool(processes=args.workers) as pool:
        for set_id, frame_id, status, elapsed in pool.imap_unordered(
                convert_one_video, work, chunksize=4):
            done += 1
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors.append((set_id, frame_id, status))

            # ETA
            wall = time.time() - t_start
            rate = wall / done
            remaining = rate * (total_src - done)

            if remaining >= 3600:
                eta = f"{remaining/3600:.1f}h"
            elif remaining >= 60:
                eta = f"{int(remaining//60)}m{int(remaining%60):02d}s"
            else:
                eta = f"{int(remaining)}s"

            # Print progress every 10 items or on error
            if done % 10 == 0 or done == total_src or "error" in status:
                pct = 100 * done / total_src
                print(f"  [{done:5d}/{total_src}] {pct:5.1f}%  "
                      f"ok={ok} skip={skipped} err={len(errors)}  "
                      f"ETA {eta}", flush=True)

    wall_total = time.time() - t_start
    stats = {
        "total_source_frames": total_src,
        "converted": ok,
        "skipped": skipped,
        "errors": len(errors),
        "anim_frames_per_source": args.num_frames,
        "views": len(VIEW_NAMES),
        "total_rgb_depth_pairs": ok * args.num_frames * len(VIEW_NAMES),
        "wall_seconds": round(wall_total, 1),
    }

    meta_path = write_metadata(sets_found, args.num_frames, args.max_angle, stats)

    h, m = divmod(wall_total, 3600)
    m, s = divmod(m, 60)
    print(f"\nDone in {int(h)}h {int(m)}m {int(s)}s")
    print(f"  Converted:    {ok:,}")
    print(f"  Skipped:      {skipped:,}")
    print(f"  Errors:       {len(errors)}")
    print(f"  RGB-D pairs:  {stats['total_rgb_depth_pairs']:,}")
    print(f"  Metadata:     {meta_path}")

    if errors:
        print(f"\nFirst 10 errors:")
        for sid, fid, msg in errors[:10]:
            print(f"  set {sid:03d} frame {fid:04d}: {msg.splitlines()[0]}")


if __name__ == "__main__":
    main()
