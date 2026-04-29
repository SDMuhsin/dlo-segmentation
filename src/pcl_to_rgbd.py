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

        # Compute splat pixel positions
        us = cu + offsets[:, 0]
        vs = cv + offsets[:, 1]

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
