#!/usr/bin/env python3
"""Point Cloud → RGB-D Conversion.

Converts CDLO point cloud data (XYZ + segmentation labels) into RGB-D image
pairs using orthographic projection across 6 canonical views.

Usage:
    python src/pcl_to_rgbd.py --convert            # Generate RGB-D pairs
    python src/pcl_to_rgbd.py --validate            # Run validation suite
    python src/pcl_to_rgbd.py --convert --validate  # Both
    python src/pcl_to_rgbd.py --ply                 # Export colored PLY files
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import KDTree

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "set2"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "rgbd"

# Image dimensions
IMG_W, IMG_H = 640, 480

# Frustum: fit [-1.1, 1.1] to shorter axis (height), derive width from that
FRUSTUM_HALF = 1.1
SCALE = IMG_H / (2 * FRUSTUM_HALF)  # 218.18 px/unit
HALF_W = IMG_W / (2 * SCALE)        # ~1.467 units

# Depth mapping: camera Z in [-1.1, 1.1] → [500, 1500] mm
DEPTH_NEAR_MM = 500
DEPTH_FAR_MM = 1500
DEPTH_RANGE_MM = DEPTH_FAR_MM - DEPTH_NEAR_MM

# Splat radius in pixels (R=1 gives 5-pixel cross; keeps reprojection error
# under 0.005 world units since max pixel offset = 1px ≈ 0.0046 units)
SPLAT_RADIUS = 1

# ── Phase 26.0: soft anti-aliased + realistically-thick WIRE rendering ───────
# Two measured synth→real gaps this lever closes (and NOTHING else):
#   (1) EDGE SOFTNESS — the hard binary cross-splat gives wires a step-function
#       boundary (measured boundary Sobel ≈ 211-259 vs image-mean ≈ 33; real
#       cables have a 1.4-2.5 px anti-aliased penumbra).
#   (2) THICKNESS — the 4096-pt harness PCL splats as ~6-7 px median run-width
#       ribbons; real cables are ≫16 px at 640×480.
# The fix is isolated to the WIRE class (label class 0 → label_out == 1) and
# is applied ENTIRELY inside ``rasterize_view``: the non-wire layer (2D photo
# backdrop, floor, clutter, hands) is rendered by the EXACT pre-P26 code path,
# so every non-wire pixel is the same as the old render except where a
# wire now overlaps it. Wire points are lifted out, rasterised at SS× into a
# soft Gaussian-alpha super-buffer, then area-downsampled (colour + coverage)
# to native res with WINNER-TAKE-ALL (crisp) depth/label, z-buffered against
# the non-wire depth so occluded wires stay hidden.
#
#   KIAT_P26_SOFTEDGE — master on/off. "0"/unset ⇒ render identical to the
#       pre-P26 rasteriser (the old single-loop hard splat). "1" ⇒ soft wires.
#   KIAT_P26_SS — super-sample factor (default 3 ⇒ 1920×1440 wire buffer).
#   KIAT_P26_WIRE_RADIUS — SOLID-disc footprint radius in NATIVE pixels
#       (default 5.0). This is the lever that sets on-screen THICKNESS: a wire
#       built from radius-R discs has an on-screen run-width ≈ 2·R px (R=5.0 ⇒
#       ~14 px median, mid-range of the real 12-16 px target).
#   KIAT_P26_RIM — width (NATIVE px) of the thin SOFT rim on the disc edge
#       (default 1.0). This is the lever that sets edge SOFTNESS independently
#       of thickness; the rendered boundary transition ≈ this width. Tune into
#       the measured real 1.4-2.5 px range.
#   KIAT_P26_COV_THRESH — coverage alpha at which a downsampled pixel becomes a
#       (crisp, binary) wire LABEL/DEPTH pixel (default 0.5).
def _p26_flag(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


P26_SOFTEDGE = _p26_flag("KIAT_P26_SOFTEDGE", "0") in ("1", "true", "True")
P26_SS = max(1, int(_p26_flag("KIAT_P26_SS", "3")))
P26_WIRE_RADIUS = float(_p26_flag("KIAT_P26_WIRE_RADIUS", "5.0"))
P26_RIM = float(_p26_flag("KIAT_P26_RIM", "1.0"))
P26_COV_THRESH = float(_p26_flag("KIAT_P26_COV_THRESH", "0.5"))
#   KIAT_P26_COVBLUR — extra native-res Gaussian on the area-averaged coverage
#       to smooth residual SS stair-steps (default 0.0; the SS area-average +
#       soft rim already give the penumbra, so no extra blur is needed).
P26_COVBLUR = float(_p26_flag("KIAT_P26_COVBLUR", "0.0"))

# ── Phase 26.2: SURFACE realism for the flat floor + 2D backdrop ─────────────
# The dominant real-world FALSE-POSITIVE mode (real valset, b2_precision_fp) is
# the model firing on "blobby busy-textured regions on real everyday surfaces":
# standalone, desaturated, ≥1000-px blobs, NOT thin/wire-shaped (camera c4 —
# parquet/desk — holds ~39% of the winnable FP mass). ROOT CAUSE: the synthetic
# FLOOR is a FLAT photo tile (``_make_textured_plane`` → raw per-point UV texel,
# NO shading/normal/AO) and the BACKDROP is an UNLIT 2D billboard
# (``color_img = background.copy()``). The model never sees a realistic 3D
# textured surface with relief shading-gradients / a contact shadow / ambient
# occlusion, so at test time any real surface with shading variation reads as
# "not-the-flat-bg-I-trained-on → maybe wire".
#
# This lever re-renders the EXISTING surface CONTENT (no photo swap — cf. the
# Phase-12 backdrop-pool catastrophe) with a LOCAL, surface-only RGB modulation
# applied as a POST-PROCESS to the finished frame, gated to BACKGROUND pixels
# (``label_img == 0`` — every floor/backdrop/clutter/hand pixel; wire pixels,
# label and depth are NEVER touched). Three SOFT, LOW-FREQUENCY components:
#   (a) RELIEF — derive a micro-relief shading field from the surface photo's
#       OWN luminance (blurred Sobel emboss) and modulate brightness ±, so the
#       flat tile gains gentle highlight/shadow gradients. Pre-blurred so the
#       relief is broad (edge transitions ≫ the ~1.1 px wire splat): the
#       Phase-19 edge-width-shortcut catastrophe taught us the floor must carry
#       NO thin sharp dark lines that could mimic a wire silhouette.
#   (b) SHADOW — a soft, blurred, darkened "grounding/contact" shadow under the
#       harness silhouette (its splatted footprint dilated + heavily blurred),
#       confined to background pixels. RGB darkening only — depth/label untouched.
#   (c) AO — a gentle large-scale radial ambient-occlusion / vignette so the
#       surface is not uniformly flat-lit.
# This is NOT a global relight (cf. the Phase-13 lighting catastrophe): it is a
# per-pixel multiplicative shade confined to the surface class, low-frequency,
# and subtle by default. ``KIAT_P26_SURFACE`` master flag (default OFF ⇒ output
# unchanged from the pre-P26.2 pipeline).
#   KIAT_P26_SURFACE        — master on/off ("0"/unset ⇒ no-op).
#   KIAT_P26_SURFACE_RELIEF — relief shading amplitude (default 0.10; the peak
#       fractional brightness swing from the embossed luminance field).
#   KIAT_P26_SURFACE_SHADOW — contact-shadow strength (default 0.30; peak
#       fractional darkening directly under the harness footprint).
#   KIAT_P26_SURFACE_AO     — ambient-occlusion / vignette strength (default
#       0.08; peak fractional darkening at the frame corners).
#   KIAT_P26_SURFACE_RELIEF_BLUR — Gaussian sigma (native px) the relief field
#       is pre-blurred with (default 2.5; keeps relief broad/soft, ≫ wire width).
#   KIAT_P26_SURFACE_SHADOW_BLUR — Gaussian sigma (native px) for the contact
#       shadow (default 9.0; a broad soft penumbra, never a hard line).
P26_SURFACE = _p26_flag("KIAT_P26_SURFACE", "0") in ("1", "true", "True")
P26_SURFACE_RELIEF = float(_p26_flag("KIAT_P26_SURFACE_RELIEF", "0.10"))
P26_SURFACE_SHADOW = float(_p26_flag("KIAT_P26_SURFACE_SHADOW", "0.30"))
P26_SURFACE_AO = float(_p26_flag("KIAT_P26_SURFACE_AO", "0.08"))
P26_SURFACE_RELIEF_BLUR = float(_p26_flag("KIAT_P26_SURFACE_RELIEF_BLUR", "2.5"))
P26_SURFACE_SHADOW_BLUR = float(_p26_flag("KIAT_P26_SURFACE_SHADOW_BLUR", "9.0"))

# ── 3-way connector-scale lever: render connectors as the BULKIER objects ────
# In the 3-class {bg, wire, connector} task with ``KIAT_DLO_UNICOLOR`` (the
# whole DLO recoloured to ONE wire colour so the colour cheat is removed), a
# connector carries the SAME thin point-splat geometry AND the same colour as
# the cable body, so there is NO cue separating them and the model paints the
# wire↔connector transition as wire (≈97% of missed connector px → wire). This
# lever gives SOURCE class-3 (connector) points a LARGER splat footprint so a
# connector renders as a coherent blob that is visibly WIDER than the thin
# cable — a width/scale discontinuity at the cable ends, which is the
# discriminative cue the colour-matched task lacks.
#
# It is purely GEOMETRIC and orthogonal to ``KIAT_DLO_UNICOLOR`` (which owns
# COLOUR): the connector COLOUR is never changed here. Wire (class 0) and every
# other class are untouched. Connector LABEL / COLOUR / DEPTH remain mutually
# pixel-aligned because every pixel of the enlarged disc is written by the SAME
# z-buffered splat that writes that connector point's label and depth.
#
#   KIAT_CONNECTOR_SCALE — float, default "1.0" = OFF / no-op (the entire effect
#       is gated behind ``scale > 1.0``, so the render is identical to the
#       pre-flag pipeline at the default). When > 1.0, a connector point splats
#       with a disc whose on-screen run-WIDTH is ≈ ``scale`` × the wire body
#       run-width (disc radius derived from the base ``SPLAT_RADIUS`` cross).
CONNECTOR_SCALE = float(_p26_flag("KIAT_CONNECTOR_SCALE", "1.0"))

# Class colors (BGR for OpenCV)
CLASS_COLORS_RGB = {
    0: (180, 180, 180),  # Wire - gray
    1: (255, 0, 0),      # Endpoint - red
    2: (0, 0, 255),      # Bifurcation - blue
    3: (0, 255, 0),      # Connector - green
    4: (255, 255, 0),    # Noise - yellow
}
CLASS_COLORS_BGR = {k: (b, g, r) for k, (r, g, b) in CLASS_COLORS_RGB.items()}
CLASS_NAMES = {0: "Wire", 1: "Endpoint", 2: "Bifurcation", 3: "Connector", 4: "Noise"}

# 10 sample definitions: (set_id, sample_id)
SAMPLES = [
    (0, 0), (3, 0), (6, 0), (9, 0), (12, 0),
    (16, 0), (22, 0), (25, 0), (30, 0), (35, 0),
]

# 6 canonical views: (name, look_direction, up_vector)
# Rotation matrix R transforms world coords to camera coords where:
#   camera X = right, camera Y = down, camera Z = into screen (depth)
VIEWS = {
    "front":  {"look": np.array([0, 0, -1.0]), "up": np.array([0, 1, 0.0])},
    "back":   {"look": np.array([0, 0, 1.0]),  "up": np.array([0, 1, 0.0])},
    "right":  {"look": np.array([1, 0, 0.0]),  "up": np.array([0, 1, 0.0])},
    "left":   {"look": np.array([-1, 0, 0.0]), "up": np.array([0, 1, 0.0])},
    "top":    {"look": np.array([0, -1, 0.0]), "up": np.array([0, 0, -1.0])},
    "bottom": {"look": np.array([0, 1, 0.0]),  "up": np.array([0, 0, 1.0])},
}


def random_view_dirs(
    rng,
    azimuth_range_deg: tuple[float, float] = (0.0, 360.0),
    elevation_range_deg: tuple[float, float] = (-30.0, 75.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 9: pick a random orthographic look + up direction.

    The "camera" sits on a sphere at unit distance from the origin (the
    harness PCL is normalised to roughly ``[-1, 1]^3``). Look direction
    points from the camera position back toward the origin. Up vector is
    derived by removing the look-component from world-up, so the camera
    stays roughly upright unless elevation is near ±90°.

    Returns ``(look, up)`` — both ``(3,) float64``, normalised.
    """
    az = float(rng.uniform(*azimuth_range_deg))
    el = float(rng.uniform(*elevation_range_deg))
    az_rad = np.deg2rad(az)
    el_rad = np.deg2rad(el)
    # Camera position on unit sphere
    cx = np.cos(el_rad) * np.sin(az_rad)
    cy = np.sin(el_rad)
    cz = np.cos(el_rad) * np.cos(az_rad)
    cam_pos = np.array([cx, cy, cz], dtype=np.float64)
    look = -cam_pos                # look points toward origin
    look_n = look / (np.linalg.norm(look) + 1e-12)
    # World up minus its component along look gives an in-plane up vector.
    world_up = np.array([0.0, 1.0, 0.0])
    up = world_up - look_n * float(np.dot(world_up, look_n))
    if np.linalg.norm(up) < 1e-6:
        # Looking straight up or down → fall back to world Z as up reference.
        world_up = np.array([0.0, 0.0, 1.0])
        up = world_up - look_n * float(np.dot(world_up, look_n))
    up = up / (np.linalg.norm(up) + 1e-12)
    return look_n, up


# ── Camera Math ──────────────────────────────────────────────────────────────

def make_view_matrix(look_dir, up_vec):
    """Build a 3×3 rotation matrix for orthographic view.

    Camera convention: X=right, Y=down, Z=into screen (depth direction).
    Returns R such that cam_coords = R @ world_coords.
    """
    forward = look_dir / np.linalg.norm(look_dir)
    right = np.cross(forward, up_vec)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    # R rows = camera axes expressed in world coords
    R = np.array([right, down, forward], dtype=np.float64)
    return R


def project_ortho(points, R):
    """Orthographic projection: world → camera (u_world, v_world, depth).

    Returns:
        cam_coords: (N, 3) where [:, 0]=right, [:, 1]=down, [:, 2]=depth
    """
    return points @ R.T


def cam_to_pixel(cam_xy):
    """Convert camera XY (world units) to pixel coordinates.

    Camera center maps to image center. Y is already "down" in camera coords.
    """
    u = cam_xy[:, 0] * SCALE + IMG_W / 2.0
    v = cam_xy[:, 1] * SCALE + IMG_H / 2.0
    return u, v


def depth_to_uint16(cam_z):
    """Map camera-space depth from [-1.1, 1.1] → [500, 1500] mm as uint16."""
    normalized = (cam_z + FRUSTUM_HALF) / (2 * FRUSTUM_HALF)  # [0, 1]
    mm = normalized * DEPTH_RANGE_MM + DEPTH_NEAR_MM
    return np.clip(mm, 0, 65535).astype(np.uint16)


def uint16_to_cam_z(depth_mm):
    """Inverse of depth_to_uint16: uint16 mm → camera Z in [-1.1, 1.1]."""
    normalized = (depth_mm.astype(np.float64) - DEPTH_NEAR_MM) / DEPTH_RANGE_MM
    return normalized * (2 * FRUSTUM_HALF) - FRUSTUM_HALF


def pixel_to_cam_xy(u, v):
    """Inverse of cam_to_pixel: pixel coords → camera XY in world units."""
    cx = (u - IMG_W / 2.0) / SCALE
    cy = (v - IMG_H / 2.0) / SCALE
    return cx, cy


# ── Rasterizer ───────────────────────────────────────────────────────────────

def rasterize_view(points, labels, R, point_rgb=None, background=None):
    """Render one orthographic view with z-buffered point splatting.

    Args:
        points: (N, 3) world coordinates
        labels: (N,) integer class labels
        R: 3×3 view rotation matrix
        point_rgb: optional (N, 3) uint8 BGR per-point colors. When given,
            used instead of class colour from CLASS_COLORS_BGR.
        background: optional (H, W, 3) uint8 BGR image. When given, color_img
            is initialised to background.copy() instead of zeros. Splatted
            points then overwrite the background pixels.
            Note: depth_img and label_img are still bg-zeroed (background has
            no depth and is not part of any class). The morphological-closing
            fill is unaffected.

    Returns:
        color_img: (H, W, 3) uint8 BGR image
        depth_img: (H, W) uint16 depth in mm (0 = no data)
        label_img: (H, W) uint8 label image; 0=bg, 1..5 for classes 0..4.
            Any point label outside ``{0..4}`` (e.g., the background sentinel
            ``BG_LABEL=255`` written by ``texture_mapping.generate_background_scene``)
            yields ``label_img = 0``, so the per-pixel label PNG cleanly
            separates harness from environment.
    """
    cam = project_ortho(points, R)
    u_f, v_f = cam_to_pixel(cam[:, :2])
    depth_vals = depth_to_uint16(cam[:, 2])

    # Frustum cull: points whose camera-space Z falls outside the depth range
    # would otherwise be drawn with clamped/aliased depth (e.g., a floor that
    # extends past ``FRUSTUM_HALF`` in world Z would render as 38 mm). Drop
    # them before any z-buffer write so the depth image stays well-defined.
    in_frustum = (cam[:, 2] >= -FRUSTUM_HALF) & (cam[:, 2] <= FRUSTUM_HALF)

    # Pre-compute per-point output labels: harness classes 0..4 → 1..5,
    # everything else (background sentinel, unknown) → 0.
    labels_arr = np.asarray(labels)
    label_out = np.zeros(labels_arr.shape[0], dtype=np.uint8)
    valid = (labels_arr >= 0) & (labels_arr <= 4)
    label_out[valid] = labels_arr[valid].astype(np.uint8) + np.uint8(1)

    # ── Phase 26.0: split off the WIRE layer (harness class 0 → label_out==1).
    # When the lever is OFF, ``wire_pt`` is the all-False default so the rest of
    # this function is the same as the pre-P26 single-loop rasteriser.
    wire_pt = np.zeros(labels_arr.shape[0], dtype=bool)
    if P26_SOFTEDGE:
        wire_pt = (labels_arr == 0)

    # Initialize outputs
    if background is not None:
        if background.shape != (IMG_H, IMG_W, 3):
            raise ValueError(
                f"background shape {background.shape}, "
                f"expected ({IMG_H}, {IMG_W}, 3)")
        if background.dtype != np.uint8:
            raise ValueError(
                f"background dtype {background.dtype}, expected uint8")
        color_img = background.copy()
    else:
        color_img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    depth_img = np.zeros((IMG_H, IMG_W), dtype=np.uint16)
    label_img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)  # bg=0
    # Z-buffer: initialize to max uint16
    zbuf = np.full((IMG_H, IMG_W), 65535, dtype=np.uint16)

    # Sort points front-to-back (smallest depth first) for first-writer-wins,
    # then drop the out-of-frustum tail so they never reach the splat loop.
    if P26_SOFTEDGE:
        # Phase 26.0/26.1: wire points are handled by the soft super-sampled
        # pass below, so exclude them from the hard splat loop. CRITICAL: drop
        # the wire points BEFORE the argsort, not after. np.argsort's default
        # quicksort is NOT stable, so the number of (equal-depth) wire points
        # interleaved into the global sort perturbs the tie-break ORDER of the
        # NON-wire points — which, under first-writer-wins, silently changes
        # which non-wire surface wins a pixel and thus its colour, scene-wide
        # (the P26.1 dense-tube would otherwise tint background far from any
        # wire). Sorting the non-wire subset on its own makes the non-wire
        # layer independent of the wire point count, so it is identical
        # across every wire-thickening setting. No-op when the lever is OFF.
        nonwire = np.where(~wire_pt)[0]
        order = nonwire[np.argsort(depth_vals[nonwire], kind="stable")]
        order = order[in_frustum[order]]
    else:
        order = np.argsort(depth_vals)
        order = order[in_frustum[order]]

    # Precompute splat offsets
    r = SPLAT_RADIUS
    offsets = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                offsets.append((dx, dy))
    offsets = np.array(offsets, dtype=np.int32)

    # ── KIAT_CONNECTOR_SCALE: enlarged disc for SOURCE class-3 (connector) ────
    # When OFF (scale == 1.0), ``conn_offsets is offsets`` and the per-point
    # selection in the splat loop ALWAYS returns ``offsets`` ⇒ the loop is
    # identical to the pre-flag pipeline. When > 1.0, connector points use
    # a SOLID disc whose on-screen run-width ≈ scale × the wire run-width: the
    # base footprint spans ``2*r+1`` px, so a disc of radius
    # ``rc = round(((2*r+1)*scale - 1) / 2)`` spans ``2*rc+1 ≈ scale*(2*r+1)``.
    # Clamped to at least ``r+1`` so any scale > 1 produces a real step.
    conn_offsets = offsets
    if CONNECTOR_SCALE > 1.0:
        rc = max(r + 1, int(round(((2 * r + 1) * CONNECTOR_SCALE - 1) / 2.0)))
        co = [(dx, dy)
              for dy in range(-rc, rc + 1)
              for dx in range(-rc, rc + 1)
              if dx * dx + dy * dy <= rc * rc]
        conn_offsets = np.array(co, dtype=np.int32)

    # Vectorized splatting with z-buffer
    u_int = np.round(u_f).astype(np.int32)
    v_int = np.round(v_f).astype(np.int32)

    for idx in order:
        cu, cv = u_int[idx], v_int[idx]
        d = depth_vals[idx]
        lbl = labels[idx]
        if point_rgb is not None:
            bgr = tuple(int(x) for x in point_rgb[idx])
        else:
            bgr = CLASS_COLORS_BGR.get(int(lbl), (128, 128, 128))

        # Compute splat pixel positions. SOURCE class-3 (connector) points use
        # the enlarged disc when KIAT_CONNECTOR_SCALE > 1.0; every other point
        # — and the entire OFF path — uses the unchanged base ``offsets``, so
        # the connector blob's added pixels carry the connector label, colour,
        # and depth via the very same z-buffered write below (stay aligned).
        off = (conn_offsets if (CONNECTOR_SCALE > 1.0 and int(lbl) == 3)
               else offsets)
        us = cu + off[:, 0]
        vs = cv + off[:, 1]

        # Bounds check
        mask = (us >= 0) & (us < IMG_W) & (vs >= 0) & (vs < IMG_H)
        us = us[mask]
        vs = vs[mask]

        # Z-buffer test: only write where this point is closer
        zmask = d < zbuf[vs, us]
        us = us[zmask]
        vs = vs[zmask]

        # Write
        zbuf[vs, us] = d
        depth_img[vs, us] = d
        color_img[vs, us] = bgr
        label_img[vs, us] = label_out[idx]

    # Morphological closing to fill 1-pixel interior gaps
    from scipy.ndimage import distance_transform_edt
    kernel = np.ones((3, 3), dtype=np.uint8)
    valid_mask = (depth_img > 0).astype(np.uint8)
    closed_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel)
    fill_mask = (closed_mask > 0) & (depth_img == 0)
    if np.any(fill_mask):
        # Use distance transform for efficient nearest-valid-pixel lookup
        empty = (depth_img == 0).astype(np.float64)
        _, nn_indices = distance_transform_edt(empty, return_distances=True,
                                               return_indices=True)
        fill_ys, fill_xs = np.where(fill_mask)
        src_y = nn_indices[0, fill_ys, fill_xs]
        src_x = nn_indices[1, fill_ys, fill_xs]
        depth_img[fill_ys, fill_xs] = depth_img[src_y, src_x]
        color_img[fill_ys, fill_xs] = color_img[src_y, src_x]
        label_img[fill_ys, fill_xs] = label_img[src_y, src_x]

    # ── Phase 26.2: SURFACE realism (relief + contact shadow + AO) ────────────
    # Re-shades the FLAT floor + UNLIT 2D backdrop (and other background pixels)
    # with a soft, low-frequency, surface-only RGB modulation so the model
    # learns "real-looking 3D textured surface = background, still not a wire".
    # Runs on the NON-WIRE layer (BEFORE the soft-wire pass, so soft wires
    # composite on top), gated to ``label_img == 0`` — wire/harness pixels,
    # the LABEL image and the DEPTH image are never touched. No-op when OFF.
    if P26_SURFACE:
        # Harness silhouette (for the grounding/contact shadow): the projected,
        # in-frustum harness points (source label 0..4 → label_out 1..5). On the
        # soft-edge path the wires are not splatted yet, so derive the footprint
        # directly from the harness point projection (mode-independent).
        harness_pt = (label_out >= 1) & in_frustum
        color_img = _apply_surface_realism(
            color_img, label_img,
            harness_u=u_f[harness_pt], harness_v=v_f[harness_pt],
        )

    # ── Phase 26.0: soft, anti-aliased, realistically-thick WIRE pass ────────
    # At this point color/depth/label/zbuf hold the fully-rendered NON-WIRE
    # layer (identical to pre-P26 minus the wire points). Now composite the
    # wire on top with sub-pixel-soft edges and real cable thickness. No-op
    # when the lever is OFF (wire_pt all-False).
    if P26_SOFTEDGE and np.any(wire_pt):
        color_img, depth_img, label_img = _composite_soft_wire(
            color_img, depth_img, label_img, zbuf,
            u_f=u_f[wire_pt], v_f=v_f[wire_pt],
            depth_vals=depth_vals[wire_pt],
            wire_rgb=(point_rgb[wire_pt] if point_rgb is not None
                      else np.broadcast_to(
                          np.array(CLASS_COLORS_BGR[0], dtype=np.uint8),
                          (int(wire_pt.sum()), 3))),
            in_frustum=in_frustum[wire_pt],
        )

    return color_img, depth_img, label_img


def _apply_surface_realism(color_img, label_img, harness_u, harness_v):
    """Phase 26.2 surface-realism post-process (RGB-only, background-only).

    Modulates the brightness of BACKGROUND pixels (``label_img == 0`` — the flat
    floor, the unlit 2D backdrop, and any clutter/hand) with three SOFT,
    LOW-FREQUENCY shading fields so the flat synthetic surfaces gain 3D realism:

      (a) RELIEF — a micro-relief shading field derived from the surface photo's
          OWN luminance (pre-blurred Sobel emboss). Pre-blurring with
          ``P26_SURFACE_RELIEF_BLUR`` keeps the relief BROAD: the brightness
          transitions are far wider than the ~1.1 px wire splat, so this never
          paints a thin sharp dark line that could mimic a wire (the Phase-19
          edge-width-shortcut guard).
      (b) SHADOW — a soft grounding/contact shadow under the harness footprint
          (silhouette dilated + heavily blurred), a pure RGB darkening.
      (c) AO — a gentle radial ambient-occlusion / vignette so the surface is
          not uniformly flat-lit.

    Everything is a single per-pixel multiplicative ``shade`` map in (0, ~1.1],
    applied ONLY where ``label_img == 0``. The LABEL and DEPTH images are not
    arguments and are never written; wire/harness pixels (label >= 1) are
    untouched by construction. Returns the modified ``color_img``.
    """
    from scipy.ndimage import gaussian_filter, binary_dilation

    bg = (label_img == 0)
    if not np.any(bg):
        return color_img

    H, W = label_img.shape
    # Per-pixel multiplicative shade, starts at 1.0 (no change).
    shade = np.ones((H, W), dtype=np.float32)

    # ── (a) RELIEF: emboss the surface's own luminance into a soft shade ──────
    if P26_SURFACE_RELIEF > 0.0:
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Pre-blur so the relief is LOW-FREQUENCY (broad transitions, never a
        # sharp wire-like line). Sobel of the blurred luminance = a directional
        # emboss; normalise to a symmetric ±1 field and post-blur once more.
        sigma = max(0.3, P26_SURFACE_RELIEF_BLUR)
        g = gaussian_filter(gray, sigma=sigma)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        # Diagonal "light from upper-left" emboss; mean-centred.
        emb = gx + gy
        emb = gaussian_filter(emb, sigma=sigma)
        emb -= float(emb[bg].mean()) if np.any(bg) else 0.0
        # Robust scale by the 95th percentile magnitude over background pixels
        # so amplitude is photo-independent; clamp the field to [-1, 1].
        mag = float(np.percentile(np.abs(emb[bg]), 95)) if np.any(bg) else 0.0
        if mag > 1e-6:
            emb = np.clip(emb / mag, -1.0, 1.0)
            shade *= (1.0 + P26_SURFACE_RELIEF * emb).astype(np.float32)

    # ── (c) AO / vignette: gentle large-scale corner darkening ────────────────
    if P26_SURFACE_AO > 0.0:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cy, cx = (H - 1) * 0.5, (W - 1) * 0.5
        rr = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        rr /= max(rr.max(), 1e-6)            # 0 at centre → 1 at the far corner
        vign = 1.0 - P26_SURFACE_AO * (rr ** 2)
        shade *= vign.astype(np.float32)

    # ── (b) Contact / grounding shadow under the harness footprint ────────────
    if P26_SURFACE_SHADOW > 0.0 and harness_u is not None and len(harness_u):
        foot = np.zeros((H, W), dtype=np.uint8)
        ui = np.round(harness_u).astype(np.int64)
        vi = np.round(harness_v).astype(np.int64)
        m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if np.any(m):
            foot[vi[m], ui[m]] = 1
            # Dilate the silhouette a little, then drop the shadow slightly
            # "down" the image (toward +v) to read as a cast/contact shadow,
            # and BLUR it broadly into a soft penumbra (never a hard edge).
            foot = binary_dilation(foot, iterations=4).astype(np.float32)
            drop = max(0, int(round(P26_SURFACE_SHADOW_BLUR * 0.6)))
            if drop:
                foot = np.roll(foot, drop, axis=0)
                foot[:drop, :] = 0.0
            foot = gaussian_filter(foot, sigma=max(1.0, P26_SURFACE_SHADOW_BLUR))
            fmax = float(foot.max())
            if fmax > 1e-6:
                foot /= fmax
                shade *= (1.0 - P26_SURFACE_SHADOW * foot).astype(np.float32)

    # Apply the combined shade to BACKGROUND pixels only.
    out = color_img.astype(np.float32)
    s = np.clip(shade, 0.0, None)[..., None]
    shaded = np.clip(np.round(out * s), 0, 255).astype(np.uint8)
    color_img[bg] = shaded[bg]
    return color_img


# Debug side-channel populated by _composite_soft_wire when KIAT_P26_DEBUG_COV=1
# (off by default ⇒ no effect on any render). Lets the validation harness read
# the TRUE rendered coverage-alpha to measure the penumbra ramp width.
_P26_DEBUG: dict = {}


def _composite_soft_wire(color_img, depth_img, label_img, zbuf,
                         u_f, v_f, depth_vals, wire_rgb, in_frustum):
    """Phase 26.0 soft wire compositor.

    Super-samples the WIRE points (only) into an SS× buffer with a SOLID-disc
    footprint of native radius ``P26_WIRE_RADIUS`` (the THICKNESS lever) carrying
    a thin soft rim of native width ``P26_RIM`` (the independent SOFTNESS lever),
    z-buffers them against each other, then:

      * COLOUR  — area-downsamples the SS wire colour×alpha → a native-res
        penumbra; alpha-blends wire-over-base so boundaries get a real
        ~1.4-2.5 px transition instead of a hard step.
      * DEPTH / LABEL — winner-take-all (min-depth) from the SS z-buffer,
        thresholded at ``P26_COV_THRESH`` coverage so the GT mask stays a
        CRISP binary (no grey label edges) and depth stays pixel-exact.
      * OCCLUSION — a wire pixel is only written where the wire's min depth is
        in front of (<) the existing non-wire depth (or the non-wire pixel is
        empty), so wires behind hands / clutter stay hidden.

    The non-wire ``color_img`` / ``depth_img`` / ``label_img`` are modified in
    place only where a wire wins, and returned.
    """
    from scipy.ndimage import gaussian_filter

    ss = P26_SS
    H, W = IMG_H, IMG_W
    Hs, Ws = H * ss, W * ss

    # Drop out-of-frustum wire points (consistent with the hard path).
    keep = in_frustum
    u_f = u_f[keep]
    v_f = v_f[keep]
    depth_vals = depth_vals[keep]
    wire_rgb = np.asarray(wire_rgb)[keep]
    if u_f.shape[0] == 0:
        return color_img, depth_img, label_img

    # Super-sampled pixel centres of each wire point. (+0.5 maps a native pixel
    # centre to the SS-grid centre of its ss×ss block.)
    us_c = u_f * ss + (ss - 1) * 0.5
    vs_c = v_f * ss + (ss - 1) * 0.5

    # Footprint: a SOLID disc of native radius R = P26_WIRE_RADIUS (this sets
    # THICKNESS) with a THIN soft rim of native width P26_RIM (this sets edge
    # SOFTNESS). Decoupling the two means a thick wire can still carry a real
    # but narrow ~1.4-2.5 px anti-aliased penumbra instead of being soft all
    # the way through. Alpha(r) = 1 for r ≤ R-rim, then linearly ramps 1→0
    # over the rim, in SS units. Working in the SS grid, a ½-SS-pixel offset is
    # added so the painted disc radius rounds correctly after downsampling.
    R_ss = P26_WIRE_RADIUS * ss
    rim_ss = max(P26_RIM * ss, 1e-3)
    rad = max(1, int(np.ceil(R_ss)) + 1)
    dy, dx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    rr = np.sqrt((dx * dx + dy * dy).astype(np.float64))
    # 1 inside the solid core, linear ramp across the rim, 0 beyond R.
    foot_alpha = np.clip((R_ss - rr) / rim_ss, 0.0, 1.0)
    off_dx = dx.ravel()
    off_dy = dy.ravel()
    off_a = foot_alpha.ravel()
    keep_off = off_a > 1e-4
    off_dx, off_dy, off_a = off_dx[keep_off], off_dy[keep_off], off_a[keep_off]

    # SS accumulators: a z-buffer (front-most wire depth per SS pixel) plus the
    # winning wire's colour, and a separate alpha-coverage buffer.
    zbuf_s = np.full((Hs, Ws), 65535, dtype=np.uint16)
    color_s = np.zeros((Hs, Ws, 3), dtype=np.uint8)
    alpha_s = np.zeros((Hs, Ws), dtype=np.float64)

    cu = np.round(us_c).astype(np.int64)
    cv = np.round(vs_c).astype(np.int64)
    d_all = depth_vals
    # Front-to-back so the nearest wire wins the SS z-buffer (first-writer-wins
    # on depth; alpha records max coverage seen so thin penumbra tails persist).
    order = np.argsort(d_all)
    for idx in order:
        us = cu[idx] + off_dx
        vs = cv[idx] + off_dy
        m = (us >= 0) & (us < Ws) & (vs >= 0) & (vs < Hs)
        us, vs, a = us[m], vs[m], off_a[m]
        if us.size == 0:
            continue
        d = d_all[idx]
        # Alpha coverage: keep the strongest footprint weight that lands here.
        cur_a = alpha_s[vs, us]
        amask = a > cur_a
        if np.any(amask):
            alpha_s[vs[amask], us[amask]] = a[amask]
        # Colour / depth: only the front-most wire writes (z-buffer test).
        zmask = d < zbuf_s[vs, us]
        if np.any(zmask):
            zu, zv = us[zmask], vs[zmask]
            zbuf_s[zv, zu] = d
            color_s[zv, zu] = wire_rgb[idx]

    # ── Downsample to native res ─────────────────────────────────────────────
    # COVERAGE (soft): area-average the SS alpha → native penumbra in [0, 1].
    cov = alpha_s.reshape(H, ss, W, ss).mean(axis=(1, 3))
    # Optional light blur of the (already area-averaged) coverage smooths
    # residual SS-grid stair-steps at the boundary. Default 0.0 (the SS
    # area-average over the soft Gaussian footprint already yields the
    # penumbra) — the crisp 0.5 contour is unaffected either way.
    if P26_COVBLUR > 0.0:
        cov = gaussian_filter(cov, sigma=P26_COVBLUR)

    # COLOUR (soft): area-average the SS wire colour, but only over SS pixels
    # that actually carry wire (alpha>0) so the penumbra colour is the true
    # wire colour, not wire-diluted-with-black. Pixels with no SS wire get 0.
    wire_ss = (alpha_s > 0).astype(np.float64)
    denom = wire_ss.reshape(H, ss, W, ss).sum(axis=(1, 3))
    col_sum = (color_s.astype(np.float64) * wire_ss[..., None]).reshape(
        H, ss, W, ss, 3).sum(axis=(1, 3))
    safe = denom > 0
    wire_color = np.zeros((H, W, 3), dtype=np.float64)
    wire_color[safe] = col_sum[safe] / denom[safe, None]

    # DEPTH / LABEL (crisp): winner-take-all from the SS z-buffer. The native
    # wire depth is the min SS depth in each block (front-most surface).
    zb = zbuf_s.reshape(H, ss, W, ss)
    wire_depth = zb.min(axis=(1, 3)).astype(np.uint16)
    has_wire_depth = wire_depth < 65535

    # ── Occlusion + compositing ──────────────────────────────────────────────
    # A "core" pixel = SS block has a z-buffered wire surface (has_wire_depth).
    # It is VISIBLE where the bg is empty OR the wire depth is in front of the
    # bg depth. Occluded cores (wire behind a hand/clutter) are not painted.
    bg_empty = depth_img == 0
    visible_core = has_wire_depth & (bg_empty | (wire_depth < depth_img))

    # CRISP wire core = coverage ≥ threshold AND a visible z-buffered surface.
    # Computed BEFORE the colour blend so the soft penumbra can be tied to it.
    own = (cov >= P26_COV_THRESH) & visible_core

    # Penumbra tail = coverage>0 with no own z-sample (the soft footprint edge
    # that fell between wire surface points). Paint it only within a few-pixel
    # morphological reach of a CLAIMED wire core (``own``), so soft tails never
    # bleed over an occluding foreground far from the wire AND a stray isolated
    # sub-threshold wire sample far from any real cable can never tint the
    # background (the P26.1 dense-tube isolation fix: reach the CLAIMED core,
    # not every ``visible_core`` SS sample). The reach radius covers the native
    # footprint penumbra (≈ a couple of native px).
    reach = cv2.dilate(
        own.astype(np.uint8),
        np.ones((2 * P26_SS + 1, 2 * P26_SS + 1), np.uint8)) > 0
    cov_pos = cov > 1e-3
    paint = cov_pos & (own | reach)

    # Alpha-blend wire colour over the base where painted and the SS block had
    # real wire colour (safe = denom>0). Clamp alpha to [0, 1].
    a = np.clip(cov, 0.0, 1.0)[..., None]
    blendable = paint & safe
    base = color_img.astype(np.float64)
    blended = a * wire_color + (1.0 - a) * base
    color_img[blendable] = np.clip(
        np.round(blended[blendable]), 0, 255).astype(np.uint8)

    # CRISP label / depth: wire OWNS the pixel where coverage ≥ threshold AND
    # the SS block has a z-buffered (visible) wire surface (``own``, above).
    # This yields a hard binary mask contour and pixel-exact depth — no soft/
    # grey label edges.
    label_img[own] = np.uint8(1)            # wire class 0 → label 1
    depth_img[own] = wire_depth[own]

    # Optional debug side-channel (off by default): stash the true native
    # coverage-alpha + visible mask so the validation harness can measure the
    # rendered penumbra ramp directly (ground truth, not RGB-reconstructed).
    if os.environ.get("KIAT_P26_DEBUG_COV", "0").strip() in ("1", "true", "True"):
        _P26_DEBUG["cov"] = cov.copy()
        _P26_DEBUG["own"] = own.copy()

    return color_img, depth_img, label_img


# ── Converter Pipeline ───────────────────────────────────────────────────────

def load_sample(set_id, sample_id):
    """Load point cloud and segmentation for a sample."""
    pcl_path = DATA_ROOT / f"{set_id:03d}" / "pointclouds_normed_4096" / f"pcl_{sample_id:04d}.npy"
    seg_path = DATA_ROOT / f"{set_id:03d}" / "segmentation_normed_4096" / f"seg_{sample_id:04d}.npy"
    points = np.load(str(pcl_path))
    labels = np.load(str(seg_path))
    return points, labels


def build_metadata(set_id, sample_id, points):
    """Build metadata dict with camera parameters for all views."""
    meta = {
        "source": {
            "set_id": set_id,
            "sample_id": sample_id,
            "num_points": int(points.shape[0]),
            "point_range": {
                "x": [float(points[:, 0].min()), float(points[:, 0].max())],
                "y": [float(points[:, 1].min()), float(points[:, 1].max())],
                "z": [float(points[:, 2].min()), float(points[:, 2].max())],
            },
        },
        "image": {
            "width": IMG_W,
            "height": IMG_H,
            "color_format": "8-bit RGB PNG (class-colored)",
            "depth_format": "16-bit unsigned PNG (millimeters, 0=no data)",
        },
        "projection": {
            "type": "orthographic",
            "scale_px_per_unit": float(SCALE),
            "frustum_half_vertical": float(FRUSTUM_HALF),
            "frustum_half_horizontal": float(HALF_W),
            "depth_near_mm": DEPTH_NEAR_MM,
            "depth_far_mm": DEPTH_FAR_MM,
        },
        "class_colors_rgb": {CLASS_NAMES[k]: list(v) for k, v in CLASS_COLORS_RGB.items()},
        "views": {},
    }
    for vname, vdef in VIEWS.items():
        R = make_view_matrix(vdef["look"], vdef["up"])
        meta["views"][vname] = {
            "look_direction": vdef["look"].tolist(),
            "up_vector": vdef["up"].tolist(),
            "rotation_matrix": R.tolist(),
        }
    return meta


def convert_sample(sample_idx, set_id, sample_id):
    """Convert one sample: load, render 6 views, save PNGs + metadata."""
    out_dir = OUTPUT_ROOT / f"sample_{sample_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    label_dir = out_dir / "label"
    label_dir.mkdir(parents=True, exist_ok=True)

    points, labels = load_sample(set_id, sample_id)
    print(f"  Sample {sample_idx:02d} (set={set_id:03d}): {points.shape[0]} points, "
          f"labels {np.unique(labels)}")

    for vname, vdef in VIEWS.items():
        R = make_view_matrix(vdef["look"], vdef["up"])
        color_img, depth_img, label_img = rasterize_view(points, labels, R)

        # Save color as RGB PNG (OpenCV uses BGR, convert)
        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(out_dir / f"color_{vname}.png"), color_img)
        cv2.imwrite(str(out_dir / f"depth_{vname}.png"), depth_img)
        cv2.imwrite(str(label_dir / f"label_{vname}.png"), label_img)

        valid_px = np.count_nonzero(depth_img)
        print(f"    {vname:8s}: {valid_px:6d} valid pixels "
              f"({100*valid_px/(IMG_W*IMG_H):.1f}%)")

    metadata = build_metadata(set_id, sample_id, points)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return out_dir


def run_conversion():
    """Convert all 10 samples."""
    print("=" * 60)
    print("Point Cloud → RGB-D Conversion")
    print("=" * 60)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for i, (set_id, sample_id) in enumerate(SAMPLES):
        convert_sample(i, set_id, sample_id)
    print(f"\nConversion complete. Output: {OUTPUT_ROOT}")


# ── Validation Suite ─────────────────────────────────────────────────────────

def validate_v1_reprojection(sample_idx, set_id, sample_id):
    """V1: Reprojection roundtrip — project depth back to 3D, compare to original."""
    points, labels = load_sample(set_id, sample_id)
    tree = KDTree(points)
    sample_dir = OUTPUT_ROOT / f"sample_{sample_idx:02d}"
    results = {}

    for vname, vdef in VIEWS.items():
        R = make_view_matrix(vdef["look"], vdef["up"])
        R_inv = R.T  # Orthogonal matrix: inverse = transpose

        depth_path = sample_dir / f"depth_{vname}.png"
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        # Find valid pixels
        vs, us = np.where(depth_img > 0)
        if len(us) == 0:
            results[vname] = {"status": "skip", "reason": "no valid pixels"}
            continue

        # Inverse project: pixel → camera XY, depth → camera Z
        cam_x, cam_y = pixel_to_cam_xy(us.astype(np.float64), vs.astype(np.float64))
        cam_z = uint16_to_cam_z(depth_img[vs, us])
        cam_pts = np.column_stack([cam_x, cam_y, cam_z])

        # Camera → world
        world_pts = cam_pts @ R_inv.T

        # Find nearest original point
        dists, _ = tree.query(world_pts)
        results[vname] = {
            "mean_error": float(np.mean(dists)),
            "p95_error": float(np.percentile(dists, 95)),
            "max_error": float(np.max(dists)),
            "num_pixels": int(len(us)),
        }

    # Aggregate across views
    all_means = [v["mean_error"] for v in results.values() if "mean_error" in v]
    all_p95 = [v["p95_error"] for v in results.values() if "p95_error" in v]
    all_max = [v["max_error"] for v in results.values() if "max_error" in v]

    agg = {
        "mean_error": float(np.mean(all_means)) if all_means else float("inf"),
        "p95_error": float(np.max(all_p95)) if all_p95 else float("inf"),
        "max_error": float(np.max(all_max)) if all_max else float("inf"),
    }
    passed = (agg["mean_error"] < 0.005 and agg["p95_error"] < 0.01 and
              agg["max_error"] < 0.02)

    return {
        "level": "V1",
        "name": "Reprojection Roundtrip",
        "passed": passed,
        "aggregate": agg,
        "per_view": results,
    }


def validate_v2_coverage(sample_idx, set_id, sample_id):
    """V2: Multi-view coverage — every point must appear in at least one view."""
    points, _ = load_sample(set_id, sample_id)
    N = points.shape[0]
    covered = np.zeros(N, dtype=bool)

    for vname, vdef in VIEWS.items():
        R = make_view_matrix(vdef["look"], vdef["up"])
        cam = project_ortho(points, R)
        u, v = cam_to_pixel(cam[:, :2])
        in_bounds = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        in_depth = (cam[:, 2] >= -FRUSTUM_HALF) & (cam[:, 2] <= FRUSTUM_HALF)
        covered |= (in_bounds & in_depth)

    coverage = float(np.sum(covered)) / N
    return {
        "level": "V2",
        "name": "Multi-View Coverage",
        "passed": coverage >= 1.0,
        "coverage": coverage,
        "total_points": int(N),
        "covered_points": int(np.sum(covered)),
        "uncovered_points": int(N - np.sum(covered)),
    }


def validate_v3_label_consistency(sample_idx, set_id, sample_id):
    """V3: Label consistency — reproject pixels to 3D, compare labels."""
    points, labels = load_sample(set_id, sample_id)
    tree = KDTree(points)
    sample_dir = OUTPUT_ROOT / f"sample_{sample_idx:02d}"

    total_correct = 0
    total_pixels = 0
    per_class_correct = {}
    per_class_total = {}

    for vname, vdef in VIEWS.items():
        R = make_view_matrix(vdef["look"], vdef["up"])
        R_inv = R.T

        color_path = sample_dir / f"color_{vname}.png"
        depth_path = sample_dir / f"depth_{vname}.png"
        color_img = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        vs, us = np.where(depth_img > 0)
        if len(us) == 0:
            continue

        # Get pixel colors (BGR) and map to labels
        pixel_bgr = color_img[vs, us]  # (N, 3)
        pixel_labels = np.full(len(us), -1, dtype=np.int64)
        for cls_id, bgr in CLASS_COLORS_BGR.items():
            match = np.all(pixel_bgr == np.array(bgr, dtype=np.uint8), axis=1)
            pixel_labels[match] = cls_id

        # Reproject to 3D
        cam_x, cam_y = pixel_to_cam_xy(us.astype(np.float64), vs.astype(np.float64))
        cam_z = uint16_to_cam_z(depth_img[vs, us])
        cam_pts = np.column_stack([cam_x, cam_y, cam_z])
        world_pts = cam_pts @ R_inv.T

        # Find nearest original point
        _, nn_idx = tree.query(world_pts)
        orig_labels = labels[nn_idx]

        # Only count pixels with valid label assignment
        valid = pixel_labels >= 0
        correct = (pixel_labels[valid] == orig_labels[valid])
        total_correct += int(np.sum(correct))
        total_pixels += int(np.sum(valid))

        for cls_id in CLASS_NAMES:
            cls_mask = orig_labels[valid] == cls_id
            per_class_total[cls_id] = per_class_total.get(cls_id, 0) + int(np.sum(cls_mask))
            per_class_correct[cls_id] = per_class_correct.get(cls_id, 0) + int(
                np.sum(correct[cls_mask]))

    overall_acc = total_correct / max(total_pixels, 1)
    per_class_acc = {}
    for cls_id in CLASS_NAMES:
        t = per_class_total.get(cls_id, 0)
        c = per_class_correct.get(cls_id, 0)
        per_class_acc[CLASS_NAMES[cls_id]] = float(c / t) if t > 0 else None

    # Pass criteria: overall > 99%, each class > 95%
    class_pass = all(
        v is None or v > 0.95 for v in per_class_acc.values()
    )
    passed = overall_acc > 0.99 and class_pass

    return {
        "level": "V3",
        "name": "Label Consistency",
        "passed": passed,
        "overall_accuracy": float(overall_acc),
        "per_class_accuracy": per_class_acc,
        "total_pixels": total_pixels,
    }


def validate_v4_cross_view(sample_idx, set_id, sample_id):
    """V4: Cross-view geometric consistency — reproject from two views, compare."""
    sample_dir = OUTPUT_ROOT / f"sample_{sample_idx:02d}"
    view_names = list(VIEWS.keys())
    pair_results = []

    # Test pairs: front-back, right-left, top-bottom (opposing views)
    pairs = [("front", "back"), ("right", "left"), ("top", "bottom")]

    for v1_name, v2_name in pairs:
        R1 = make_view_matrix(VIEWS[v1_name]["look"], VIEWS[v1_name]["up"])
        R2 = make_view_matrix(VIEWS[v2_name]["look"], VIEWS[v2_name]["up"])

        depth1 = cv2.imread(str(sample_dir / f"depth_{v1_name}.png"), cv2.IMREAD_UNCHANGED)
        depth2 = cv2.imread(str(sample_dir / f"depth_{v2_name}.png"), cv2.IMREAD_UNCHANGED)

        # Reproject view 1 to 3D
        vs1, us1 = np.where(depth1 > 0)
        if len(us1) == 0:
            continue
        cx1, cy1 = pixel_to_cam_xy(us1.astype(np.float64), vs1.astype(np.float64))
        cz1 = uint16_to_cam_z(depth1[vs1, us1])
        world1 = np.column_stack([cx1, cy1, cz1]) @ R1  # R1.T.T = R1

        # Reproject view 2 to 3D
        vs2, us2 = np.where(depth2 > 0)
        if len(us2) == 0:
            continue
        cx2, cy2 = pixel_to_cam_xy(us2.astype(np.float64), vs2.astype(np.float64))
        cz2 = uint16_to_cam_z(depth2[vs2, us2])
        world2 = np.column_stack([cx2, cy2, cz2]) @ R2

        # Find correspondences: for each point in view1, find nearest in view2
        tree2 = KDTree(world2)
        dists, _ = tree2.query(world1)

        # Only consider close matches as correspondences; tight threshold
        # ensures we compare truly same-surface points (not opposite faces
        # of wires which differ by wire thickness ~0.02-0.15 units)
        close_mask = dists < 0.008
        if np.sum(close_mask) == 0:
            pair_results.append({
                "pair": f"{v1_name}-{v2_name}",
                "status": "no_correspondences",
            })
            continue

        close_dists = dists[close_mask]
        pair_results.append({
            "pair": f"{v1_name}-{v2_name}",
            "mean_error": float(np.mean(close_dists)),
            "num_correspondences": int(np.sum(close_mask)),
        })

    all_means = [p["mean_error"] for p in pair_results if "mean_error" in p]
    agg_mean = float(np.mean(all_means)) if all_means else float("inf")
    passed = agg_mean < 0.005

    return {
        "level": "V4",
        "name": "Cross-View Geometric Consistency",
        "passed": passed,
        "aggregate_mean_error": agg_mean,
        "pairs": pair_results,
    }


def validate_v5_depth_distribution(sample_idx, set_id, sample_id):
    """V5: Depth distribution sanity — check depth values are in expected range."""
    points, _ = load_sample(set_id, sample_id)
    sample_dir = OUTPUT_ROOT / f"sample_{sample_idx:02d}"
    results = {}

    for vname, vdef in VIEWS.items():
        R = make_view_matrix(vdef["look"], vdef["up"])
        cam = project_ortho(points, R)

        # Expected depth range from projected point cloud
        expected_z = cam[:, 2]
        expected_mm = (expected_z + FRUSTUM_HALF) / (2 * FRUSTUM_HALF) * DEPTH_RANGE_MM + DEPTH_NEAR_MM
        expected_min_mm = float(np.min(expected_mm))
        expected_max_mm = float(np.max(expected_mm))
        expected_mean_mm = float(np.mean(expected_mm))

        # Actual depth from image
        depth_img = cv2.imread(str(sample_dir / f"depth_{vname}.png"), cv2.IMREAD_UNCHANGED)
        valid = depth_img[depth_img > 0].astype(np.float64)

        if len(valid) == 0:
            results[vname] = {"status": "skip", "reason": "no valid pixels"}
            continue

        results[vname] = {
            "actual_min_mm": float(np.min(valid)),
            "actual_max_mm": float(np.max(valid)),
            "actual_mean_mm": float(np.mean(valid)),
            "expected_min_mm": expected_min_mm,
            "expected_max_mm": expected_max_mm,
            "expected_mean_mm": expected_mean_mm,
        }

    # Pass criteria: depth values must be in valid range and within the
    # projected depth envelope (z-buffer picks front surface, so actual mean
    # can differ from all-point mean; we check range containment instead)
    all_ok = True
    for vname, r in results.items():
        if "actual_min_mm" not in r:
            continue
        # All depth values must be in [500, 1500] mm
        if r["actual_min_mm"] < DEPTH_NEAR_MM:
            all_ok = False
        if r["actual_max_mm"] > DEPTH_FAR_MM:
            all_ok = False
        # Actual range must be within the projected depth envelope (±2mm for quantization)
        if r["actual_min_mm"] < r["expected_min_mm"] - 2:
            all_ok = False
        if r["actual_max_mm"] > r["expected_max_mm"] + 2:
            all_ok = False
        # Actual mean must be between the expected min and max
        if r["actual_mean_mm"] < r["expected_min_mm"] - 2:
            all_ok = False
        if r["actual_mean_mm"] > r["expected_max_mm"] + 2:
            all_ok = False

    return {
        "level": "V5",
        "name": "Depth Distribution Sanity",
        "passed": all_ok,
        "per_view": results,
    }


def run_validation():
    """Run full validation suite on all samples."""
    print("=" * 60)
    print("Validation Suite")
    print("=" * 60)

    all_results = {}
    summary_lines = []
    all_passed = True

    for i, (set_id, sample_id) in enumerate(SAMPLES):
        sample_key = f"sample_{i:02d}"
        print(f"\n  Validating {sample_key} (set={set_id:03d})...")

        sample_results = {}
        for vfunc, label in [
            (validate_v1_reprojection, "V1: Reprojection"),
            (validate_v2_coverage, "V2: Coverage"),
            (validate_v3_label_consistency, "V3: Labels"),
            (validate_v4_cross_view, "V4: Cross-View"),
            (validate_v5_depth_distribution, "V5: Depth"),
        ]:
            result = vfunc(i, set_id, sample_id)
            level = result["level"]
            sample_results[level] = result
            status = "PASS" if result["passed"] else "FAIL"
            if not result["passed"]:
                all_passed = False
            print(f"    {label}: {status}")

            # Build summary detail
            if level == "V1":
                agg = result.get("aggregate", {})
                summary_lines.append(
                    f"{sample_key} {level}: {status} "
                    f"(mean={agg.get('mean_error', 'N/A'):.6f}, "
                    f"p95={agg.get('p95_error', 'N/A'):.6f}, "
                    f"max={agg.get('max_error', 'N/A'):.6f})")
            elif level == "V2":
                summary_lines.append(
                    f"{sample_key} {level}: {status} "
                    f"(coverage={result.get('coverage', 0):.4f})")
            elif level == "V3":
                summary_lines.append(
                    f"{sample_key} {level}: {status} "
                    f"(accuracy={result.get('overall_accuracy', 0):.4f})")
            elif level == "V4":
                summary_lines.append(
                    f"{sample_key} {level}: {status} "
                    f"(mean_err={result.get('aggregate_mean_error', 'N/A')})")
            elif level == "V5":
                summary_lines.append(
                    f"{sample_key} {level}: {status}")

        all_results[sample_key] = sample_results

    # Write validation report
    report = {
        "all_passed": all_passed,
        "samples": all_results,
    }
    report_path = OUTPUT_ROOT / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Write human-readable summary
    summary_path = OUTPUT_ROOT / "validation_summary.txt"
    header = "VALIDATION SUMMARY\n" + "=" * 40 + "\n"
    header += f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n\n"
    with open(summary_path, "w") as f:
        f.write(header)
        f.write("\n".join(summary_lines))
        f.write("\n")

    print(f"\n{'=' * 60}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")

    return all_passed


# ── PLY Export ───────────────────────────────────────────────────────────────

def write_ply(path, points, rgb):
    """Write an ASCII PLY file with per-vertex colors.

    Args:
        path: Output file path.
        points: (N, 3) float array of XYZ coordinates.
        rgb: (N, 3) uint8 array of RGB colors.
    """
    n = points.shape[0]
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                    f"{rgb[i, 0]} {rgb[i, 1]} {rgb[i, 2]}\n")


def labels_to_rgb(labels):
    """Convert class labels to an (N, 3) uint8 RGB array."""
    return np.array([CLASS_COLORS_RGB.get(int(l), (128, 128, 128)) for l in labels],
                    dtype=np.uint8)


def reproject_views_to_pointcloud(sample_dir):
    """Reproject all 6 RGB-D views back to a merged 3D point cloud.

    Returns:
        points: (M, 3) float64 world coordinates.
        rgb: (M, 3) uint8 RGB colors.
    """
    all_points = []
    all_rgb = []

    for vname, vdef in VIEWS.items():
        R = make_view_matrix(vdef["look"], vdef["up"])
        R_inv = R.T  # orthogonal → inverse is transpose

        depth_img = cv2.imread(str(sample_dir / f"depth_{vname}.png"),
                               cv2.IMREAD_UNCHANGED)
        color_img = cv2.imread(str(sample_dir / f"color_{vname}.png"),
                               cv2.IMREAD_COLOR)

        vs, us = np.where(depth_img > 0)
        if len(us) == 0:
            continue

        # Pixel → camera XY, depth → camera Z
        cam_x, cam_y = pixel_to_cam_xy(us.astype(np.float64),
                                        vs.astype(np.float64))
        cam_z = uint16_to_cam_z(depth_img[vs, us])
        cam_pts = np.column_stack([cam_x, cam_y, cam_z])

        # Camera → world
        world_pts = cam_pts @ R_inv.T
        all_points.append(world_pts)

        # BGR → RGB
        bgr = color_img[vs, us]
        all_rgb.append(bgr[:, ::-1])

    return np.vstack(all_points), np.vstack(all_rgb).astype(np.uint8)


def reproject_single_view(sample_dir, vname):
    """Reproject a single RGB-D view back to 3D.

    Returns:
        points: (M, 3) float64 world coordinates.
        rgb: (M, 3) uint8 RGB colors.
    """
    vdef = VIEWS[vname]
    R = make_view_matrix(vdef["look"], vdef["up"])
    R_inv = R.T

    depth_img = cv2.imread(str(sample_dir / f"depth_{vname}.png"),
                           cv2.IMREAD_UNCHANGED)
    color_img = cv2.imread(str(sample_dir / f"color_{vname}.png"),
                           cv2.IMREAD_COLOR)

    vs, us = np.where(depth_img > 0)
    if len(us) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    cam_x, cam_y = pixel_to_cam_xy(us.astype(np.float64),
                                    vs.astype(np.float64))
    cam_z = uint16_to_cam_z(depth_img[vs, us])
    cam_pts = np.column_stack([cam_x, cam_y, cam_z])
    world_pts = cam_pts @ R_inv.T
    bgr = color_img[vs, us]
    return world_pts, bgr[:, ::-1].astype(np.uint8)


def run_ply_export():
    """Export original + per-view roundtrip PLY point clouds for all 10 samples."""
    print("=" * 60)
    print("PLY Point Cloud Export")
    print("=" * 60)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for i, (set_id, sample_id) in enumerate(SAMPLES):
        out_dir = OUTPUT_ROOT / f"sample_{i:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Original ground-truth point cloud
        points, labels = load_sample(set_id, sample_id)
        orig_path = out_dir / "pointcloud.ply"
        write_ply(orig_path, points, labels_to_rgb(labels))

        # Per-view roundtrip PLYs
        view_counts = []
        for vname in VIEWS:
            rt_pts, rt_rgb = reproject_single_view(out_dir, vname)
            rt_path = out_dir / f"pointcloud_roundtrip_{vname}.ply"
            write_ply(rt_path, rt_pts, rt_rgb)
            view_counts.append(f"{vname}={rt_pts.shape[0]}")

        print(f"  sample_{i:02d}: original {points.shape[0]} pts | "
              f"{', '.join(view_counts)}")
    print(f"\nPLY export complete. Output: {OUTPUT_ROOT}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Point Cloud → RGB-D Conversion")
    parser.add_argument("--convert", action="store_true", help="Run conversion")
    parser.add_argument("--validate", action="store_true", help="Run validation suite")
    parser.add_argument("--ply", action="store_true", help="Export colored PLY point clouds")
    args = parser.parse_args()

    if not args.convert and not args.validate and not args.ply:
        parser.print_help()
        sys.exit(1)

    if args.convert:
        run_conversion()
    if args.validate:
        run_validation()
    if args.ply:
        run_ply_export()


if __name__ == "__main__":
    main()
