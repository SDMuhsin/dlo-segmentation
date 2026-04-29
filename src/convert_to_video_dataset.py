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
OUTPUT_ROOT = PROJECT_ROOT / "data" / "rgbd_videos"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pcl_to_rgbd import (
    VIEWS, make_view_matrix, rasterize_view,
    IMG_W, IMG_H, FRUSTUM_HALF, SCALE, HALF_W,
    DEPTH_NEAR_MM, DEPTH_FAR_MM,
    CLASS_COLORS_RGB, CLASS_NAMES,
)
from texture_mapping import (
    BG_LABEL,
    compute_per_point_rgb,
    generate_background_scene,
    load_background_library,
    load_object_library,
    load_texture_library,
)

VIEW_NAMES = list(VIEWS.keys())
VIEW_ROTATIONS = {vn: make_view_matrix(VIEWS[vn]["look"], VIEWS[vn]["up"])
                  for vn in VIEW_NAMES}

# Texture library: loaded lazily on first use per worker process.
_TEX_LIBRARY_CACHE = None
def _get_texture_library():
    global _TEX_LIBRARY_CACHE
    if _TEX_LIBRARY_CACHE is None:
        _TEX_LIBRARY_CACHE = load_texture_library(PROJECT_ROOT / "data" / "textures")
    return _TEX_LIBRARY_CACHE

# Background-photo library: used (a) as the texture source for the 3D floor
# in the bg scene and (b) as the 2D backdrop filling pixels with no 3D point.
# Lazy per-worker cache.
_BG_LIBRARY_CACHE = None
def _get_background_library():
    global _BG_LIBRARY_CACHE
    if _BG_LIBRARY_CACHE is None:
        _BG_LIBRARY_CACHE = load_background_library(
            PROJECT_ROOT / "data" / "textures" / "backgrounds"
        )
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

# Number of background points sampled per source frame's 3D scene. Held at the
# module level so a worker process pays a single allocation per call.
BG_N_POINTS = 30000

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
    for ai in range(num_anim):
        for vn in view_names:
            fname = f"{src_frame:04d}_{ai:02d}_{vn}.png"
            if not (rgb_dir / fname).exists():
                return False
            if not (depth_dir / fname).exists():
                return False
            if not (label_dir / fname).exists():
                return False
    return True


def convert_one_video(args):
    """Convert one source frame into an animated video clip.

    Work unit = (set_id, source_frame_id).
    Returns (set_id, frame_id, status, elapsed).
    """
    set_id, frame_id, num_frames, max_angle_deg, output_root = args
    t0 = time.time()
    set_str = f"{set_id:03d}"
    split = split_of(set_id)

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

        # Texture mapping: compute per-point BGR ONCE at rest pose. Reusable
        # across all animation frames because LBS deforms positions but never
        # reorders points, so each point's identity (and its texture sample)
        # remains stable.
        texture_library = _get_texture_library()
        point_rgb = compute_per_point_rgb(
            pcl=pcl, labels=seg, nodes=nodes, edges=edges, segments=segments,
            na=na, nb=nb, wa=wa, wb=wb, offsets=offsets,
            texture_library=texture_library,
            seed=set_id * 1000 + frame_id,
        )

        # 3D background scene (Phase 4): textured floor + 3-5 real-object
        # PCLs from the CC0 object library scattered around the harness.
        # Built once per source frame and held static across all 20 anim
        # frames + 6 views — the harness animates against a fixed
        # environment. Seed offset by +7 so the bg RNG stream is independent
        # of the harness texture RNG.
        bg_library = _get_background_library()
        obj_library = _get_object_library()
        if bg_library:
            bg_pcl, bg_rgb = generate_background_scene(
                rng=np.random.RandomState(set_id * 1000 + frame_id + 7),
                bbox_min=pcl.min(axis=0),
                bbox_max=pcl.max(axis=0),
                texture_library=bg_library,
                object_library=obj_library,
                n_points=BG_N_POINTS,
            )
        else:
            bg_pcl = np.empty((0, 3))
            bg_rgb = np.empty((0, 3), dtype=np.uint8)

        # 2D photographic backdrop (Phase 2 → restored in Phase 4).
        # Deterministically picked per source frame; same photo across all
        # 20 anim frames + 6 views so the backdrop is part of one coherent
        # "shot". The 3D objects supply geometry/depth; the photo fills
        # pixels where no 3D point projects.
        if bg_library:
            photo_background = bg_library[
                (set_id * 1000 + frame_id) % len(bg_library)
            ]
        else:
            photo_background = None

        # Concatenate harness + bg colour and labels once. Positions for the
        # harness are concatenated PER ANIMATION FRAME below (only the harness
        # animates; bg is static).
        combined_rgb = np.concatenate([point_rgb, bg_rgb], axis=0).astype(np.uint8)
        combined_labels = np.concatenate(
            [seg.astype(np.int64), np.full(len(bg_pcl), BG_LABEL, dtype=np.int64)]
        )

        # Per-joint random params (deterministic: seed = set_id * 1000 + frame_id)
        rng = np.random.RandomState(set_id * 1000 + frame_id)
        phase = {j: rng.uniform(0, 2 * np.pi) for j in joints}
        freq  = {j: rng.uniform(0.5, 2.0)     for j in joints}
        amp   = {j: rng.uniform(0.3, 1.0)     for j in joints}
        max_ang = np.radians(max_angle_deg)

        labels_int = seg.astype(int)
        rgb_dir     = out_base / "rgb"
        depth_dir   = out_base / "depth"
        pcl_dir     = out_base / "pointclouds"
        lbl_dir     = out_base / "labels"   # per-source per-point .npy (existing)
        lbl_img_dir = out_base / "label"    # per-anim per-view PNG (new)

        for d in (rgb_dir, depth_dir, pcl_dir, lbl_dir, lbl_img_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Save labels (one file per source frame, constant across animation)
        lbl_path = lbl_dir / f"{frame_id:04d}.npy"
        if not lbl_path.exists():
            np.save(str(lbl_path), seg.astype(np.int8))

        for ai in range(num_frames):
            t = ai / max(num_frames - 1, 1)
            jrot = {}
            for j in joints:
                angle = max_ang * amp[j] * np.sin(
                    2 * np.pi * freq[j] * t + phase[j])
                jrot[j] = (axes[j], angle)

            new_nodes, node_R = _forward_kinematics(nodes, root, children, jrot)
            new_pcl = _animate_points_fast(wa, wb, na, nb, offsets, new_nodes, node_R)

            # Per-point .npy output remains harness-only (per Phase 3 plan
            # decision (a): bg points exist only in the rendered images).
            np.save(str(pcl_dir / f"{frame_id:04d}_{ai:02d}.npy"),
                    new_pcl.astype(np.float32))

            # Stitch the (animated) harness with the (static) bg for rendering.
            combined_pcl = np.concatenate([new_pcl, bg_pcl], axis=0)

            for vn in VIEW_NAMES:
                fname = f"{frame_id:04d}_{ai:02d}_{vn}.png"
                color, depth, label_img = rasterize_view(
                    combined_pcl, combined_labels, VIEW_ROTATIONS[vn],
                    point_rgb=combined_rgb,
                    background=photo_background,
                )
                cv2.imwrite(str(rgb_dir / fname), color)
                cv2.imwrite(str(depth_dir / fname), depth)
                cv2.imwrite(str(lbl_img_dir / fname), label_img)

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


def build_work_list(sets_found, num_frames, max_angle, output_root):
    items = []
    for sid in sorted(sets_found):
        for fid in range(sets_found[sid]):
            items.append((sid, fid, num_frames, max_angle, str(output_root)))
    return items


# ── Metadata ───────────────────────────────────────────────────────────────

def write_metadata(sets_found, num_frames, max_angle_deg, stats):
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

    meta = {
        "description": "Animated RGB-D video dataset from CDLO point clouds",
        "animation": {
            "method": "Skeleton-based FK with LBS point binding",
            "anim_frames_per_source": num_frames,
            "max_angle_deg": max_angle_deg,
            "joint_selection": "structural + every 3rd interior skeleton node",
        },
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
            "pointclouds": "float32 .npy (4096, 3)",
            "labels": "int8 .npy (4096,) per source frame (per-point class labels, 0..4)",
        },
        "naming": {
            "rgb_depth_label": "{src_frame:04d}_{anim_frame:02d}_{view}.png",
            "pointcloud": "{src_frame:04d}_{anim_frame:02d}.npy",
            "labels": "{src_frame:04d}.npy",
        },
        "texture_mapping": {
            "method": "Per-point UV sampling at rest pose. Wire: cylindrical UV using rotation-minimising frame along skeleton segments. Other classes: cluster-based PCA-planar UV.",
            "textures_root": "data/textures/",
            "license": "CC0 (ambientCG)",
            "anti_aliasing": "Gaussian-blurred (sigma=2px) source textures + bilinear sampling",
        },
        "background": {
            "method": "Phase 4: textured xz-floor + 3-5 real-object CC0 point clouds (sampled from Poly Haven mesh assets in data/objects/) placed on the floor outside the harness footprint, COMPOSITED OVER a 2D photographic backdrop. The 3D objects/floor supply geometry/depth; the 2D photo fills pixels with no 3D point. Background points carry a sentinel label (BG_LABEL=255) so the per-pixel label PNG keeps bg=0. The scene is static across all 20 anim frames + 6 views of one source for video coherence (only the harness animates).",
            "n_points_per_scene": BG_N_POINTS,
            "library_dir_3d_objects": "data/objects/",
            "library_dir_photos": "data/textures/backgrounds/",
            "license": "CC0",
            "seed_per_source": "set_id * 1000 + frame_id + 7 for the 3D scene; (set_id * 1000 + frame_id) % len(bg_library) selects the 2D backdrop photo.",
        },
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
