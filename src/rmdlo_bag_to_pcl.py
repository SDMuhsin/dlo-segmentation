#!/usr/bin/env python3
"""RMDLO ROS1 bag  →  wire-labeled point cloud (pcl_to_rgbd.py input format).

The RMDLO TrackDLO data release (Illinois Data Bank IDB-2916472) ships ROS1
``.bag`` files recorded from a STATIC Intel RealSense D435.  Each bag contains:

  * ``/camera/color/image_raw``                       RGB  (rgb8, 1280x720)
  * ``/camera/aligned_depth_to_color/image_raw``      DEPTH (16UC1, aligned-to-
                                                      color metric Z, 1280x720)
  * ``/camera/aligned_depth_to_color/camera_info``    CameraInfo (RealSense D435
                                                      intrinsics: fx,fy,cx,cy)

There are NO wire masks.  The DLO is a bright blue/teal rope or rubber tube that
the authors segment with a fixed HSV threshold.  This module:

  1. reads the bag with the pure-python ``rosbags`` lib (no ROS install),
  2. extracts time-synchronised (RGB, aligned-Z16 depth, intrinsics) frames,
  3. HSV-thresholds the blue/teal DLO in RGB  ->  2D wire mask,
  4. back-projects ALL valid-depth pixels with the pinhole intrinsics to metric
     3D points (meters), tagging wire pixels label 0 (== ``pcl_to_rgbd`` Wire)
     and the rest of the scene label 4 (== Noise; kept so the scene context is
     available, dropped by default to keep clouds small),
  5. writes the cloud in the EXACT on-disk format ``src/pcl_to_rgbd.py`` reads:
        pcl_NNNN.npy   (M, 3) float64  XYZ
        seg_NNNN.npy   (M,)   int64    class label   (0 = Wire)

  ── FORMAT BRIDGE ────────────────────────────────────────────────────────────
  ``pcl_to_rgbd.load_sample`` expects coordinates NORMALISED into roughly the
  orthographic frustum cube ``[-1.1, 1.1]^3`` (the CDLO clouds are unit-scaled).
  Our back-projection produces METRIC meters in CAMERA coordinates (Z away from
  the camera, ~0.4-0.6 m).  ``--normed`` rigidly recenters+rescales the cloud so
  its longest extent maps to the frustum, WITHOUT distorting geometry, and flips
  to the renderer's Y-down convention, so the cloud renders correctly through
  ``rasterize_view``.  The UN-normalised metric cloud (``pcl_metric_NNNN.npy``)
  is ALSO written so geometry measurements stay in real millimetres.

Usage:
    python src/rmdlo_bag_to_pcl.py --bag data/rmdlo_probe/foo.bag \
        --out data/rmdlo_probe/extracted --num-frames 5 --stride 80 \
        --save-overlays --normed

The HSV range + depth scale are parameters so this scales to the full dataset.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ── Defaults ──────────────────────────────────────────────────────────────────

# RMDLO topic names (RealSense D435, aligned depth-to-color).
TOPIC_COLOR = "/camera/color/image_raw"
TOPIC_DEPTH = "/camera/aligned_depth_to_color/image_raw"
# CameraInfo topic differs across the RMDLO families: TrackDLO bags publish the
# aligned-depth CameraInfo, MultiDLO bags publish only the color CameraInfo. Both
# carry the SAME RealSense intrinsics for the aligned-depth-to-color stream (the
# aligned depth lives in the color optical frame). Try them in order.
TOPIC_CINFO_CANDIDATES = (
    "/camera/aligned_depth_to_color/camera_info",
    "/camera/color/camera_info",
)
TOPIC_CINFO = TOPIC_CINFO_CANDIDATES[0]  # back-compat alias

# Author's blue/teal rope HSV range (OpenCV H in [0,179], S,V in [0,255]).
# RMDLO uses H[90,130] in 0-179 space for the cyan/blue DLO.
HSV_LO_DEFAULT = (90, 90, 30)
HSV_HI_DEFAULT = (130, 255, 255)

# RealSense aligned-depth is 16UC1.  D435 default depth unit = 1 mm/count
# (so a count of 510 == 0.510 m).  RMDLO bags follow this; auto-verified below.
DEPTH_UNIT_M_DEFAULT = 1.0e-3   # meters per depth count

# pcl_to_rgbd orthographic frustum half-extent (see FRUSTUM_HALF there).
FRUSTUM_HALF = 1.1


# ── Bag reading ───────────────────────────────────────────────────────────────

def _imgmsg_to_array(msg):
    """Convert a sensor_msgs/Image into an HxW(xC) numpy array (no cv_bridge)."""
    enc = msg.encoding
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        arr = buf.reshape(h, w, 3)
        if enc == "rgb8":               # we want BGR for OpenCV
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
    if enc in ("16UC1", "mono16"):
        return np.frombuffer(msg.data, dtype="<u2").reshape(h, w).copy()
    if enc in ("mono8", "8UC1"):
        return buf.reshape(h, w).copy()
    raise ValueError(f"unsupported image encoding: {enc}")


def read_bag_frames(bag_path, num_frames, stride, start=0):
    """Pull synchronised (bgr, depth_u16, intrinsics) frames from a bag.

    The RGB and aligned-depth streams are recorded at the same rate with matching
    header stamps; we index both by acquisition order and pair the i-th of each,
    sampling every ``stride`` frames starting at ``start`` until ``num_frames``.

    Returns:
        frames: list of dicts  {idx, bgr (HxWx3 uint8), depth (HxW uint16)}
        intr:   dict  {fx, fy, cx, cy, width, height}
    """
    from rosbags.highlevel import AnyReader

    bag_path = Path(bag_path)
    intr = None
    colors, depths = [], []
    with AnyReader([bag_path]) as reader:
        # CameraInfo (first message is enough; intrinsics are static). The topic
        # name differs across RMDLO families, so try the candidates in order and
        # only iterate a NON-EMPTY connection list (rosbags treats an empty
        # connections list as "all topics", which would deserialize a /tf_static
        # TFMessage as CameraInfo and crash on the missing .K field).
        ci_conns = []
        for cand in TOPIC_CINFO_CANDIDATES:
            ci_conns = [c for c in reader.connections if c.topic == cand]
            if ci_conns:
                break
        if not ci_conns:
            raise RuntimeError(
                f"no CameraInfo topic found in {bag_path.name}; "
                f"tried {TOPIC_CINFO_CANDIDATES}")
        for conn, _, raw in reader.messages(connections=ci_conns):
            m = reader.deserialize(raw, conn.msgtype)
            K = list(m.k) if hasattr(m, "k") else list(m.K)
            intr = {"fx": float(K[0]), "fy": float(K[4]),
                    "cx": float(K[2]), "cy": float(K[5]),
                    "width": int(m.width), "height": int(m.height)}
            break

        col_conns = [c for c in reader.connections if c.topic == TOPIC_COLOR]
        dep_conns = [c for c in reader.connections if c.topic == TOPIC_DEPTH]
        for conn, _, raw in reader.messages(connections=col_conns):
            colors.append(_imgmsg_to_array(reader.deserialize(raw, conn.msgtype)))
        for conn, _, raw in reader.messages(connections=dep_conns):
            depths.append(_imgmsg_to_array(reader.deserialize(raw, conn.msgtype)))

    n = min(len(colors), len(depths))
    if n == 0:
        raise RuntimeError("no color/depth frames found in bag")
    frames = []
    idx = start
    while idx < n and len(frames) < num_frames:
        frames.append({"idx": idx, "bgr": colors[idx], "depth": depths[idx]})
        idx += stride
    return frames, intr


# ── Wire labelling + back-projection ──────────────────────────────────────────

def hsv_wire_mask(bgr, hsv_lo=HSV_LO_DEFAULT, hsv_hi=HSV_HI_DEFAULT,
                  min_area=200, open_ksize=3):
    """HSV-threshold the blue/teal DLO -> cleaned binary wire mask (HxW bool)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo, np.uint8),
                       np.array(hsv_hi, np.uint8))
    if open_ksize > 0:
        k = np.ones((open_ksize, open_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    # Drop tiny speckle components below ``min_area`` px.
    if min_area > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
        keep = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[lab == i] = 255
        mask = keep
    return mask > 0


def backproject(depth_u16, intr, depth_unit_m=DEPTH_UNIT_M_DEFAULT,
                z_min_m=0.1, z_max_m=3.0):
    """Pinhole back-projection of every valid-depth pixel to camera-frame meters.

    Returns:
        pts:  (M, 3) float64  camera-frame XYZ in meters  (Z = away from camera)
        vs:   (M,)   int       pixel row of each point
        us:   (M,)   int       pixel col of each point
    """
    h, w = depth_u16.shape
    z = depth_u16.astype(np.float64) * depth_unit_m
    valid = (depth_u16 > 0) & (z >= z_min_m) & (z <= z_max_m)
    vs, us = np.where(valid)
    zz = z[vs, us]
    xx = (us - intr["cx"]) / intr["fx"] * zz
    yy = (vs - intr["cy"]) / intr["fy"] * zz
    pts = np.column_stack([xx, yy, zz])
    return pts, vs, us


def normalize_to_frustum(pts_cam):
    """FORMAT BRIDGE: metric camera meters -> pcl_to_rgbd frustum cube.

    Rigidly recenters on the cloud centroid and uniformly scales so the longest
    half-extent maps to ``0.9 * FRUSTUM_HALF`` (a small margin keeps the splats
    inside the frame).  Camera Y already points DOWN in image space, matching the
    renderer's Y-down convention, so no axis flip is needed.  No shearing /
    anisotropic scaling: geometry (relief ratios, straightness) is preserved.
    """
    c = pts_cam.mean(axis=0)
    centered = pts_cam - c
    half = np.abs(centered).max()
    scale = (0.9 * FRUSTUM_HALF) / max(half, 1e-9)
    return centered * scale


def bag_to_pointcloud(bgr, depth_u16, intr, hsv_lo, hsv_hi, depth_unit_m,
                      include_scene=False, scene_subsample=8):
    """One frame -> (points_metric, labels, mask, dbg).

    label 0 = Wire (HSV+valid-depth), label 4 = Noise/scene context (optional).
    """
    mask = hsv_wire_mask(bgr, hsv_lo, hsv_hi)
    pts, vs, us = backproject(depth_u16, intr, depth_unit_m)

    is_wire = mask[vs, us]
    wire_pts = pts[is_wire]
    out_pts = [wire_pts]
    out_lbl = [np.zeros(len(wire_pts), dtype=np.int64)]        # 0 = Wire

    if include_scene:
        scene_pts = pts[~is_wire][::scene_subsample]
        out_pts.append(scene_pts)
        out_lbl.append(np.full(len(scene_pts), 4, dtype=np.int64))  # 4 = Noise

    points = np.vstack(out_pts) if any(len(p) for p in out_pts) else np.zeros((0, 3))
    labels = np.concatenate(out_lbl) if out_lbl else np.zeros(0, np.int64)
    dbg = {"n_wire_px": int(is_wire.sum()),
           "n_valid_depth_px": int(len(vs)),
           "fg_fraction": float(mask.mean())}
    return points, labels, mask, dbg


# ── Driver ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--num-frames", type=int, default=5)
    ap.add_argument("--stride", type=int, default=80)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--hsv-lo", type=int, nargs=3, default=list(HSV_LO_DEFAULT))
    ap.add_argument("--hsv-hi", type=int, nargs=3, default=list(HSV_HI_DEFAULT))
    ap.add_argument("--depth-unit-m", type=float, default=DEPTH_UNIT_M_DEFAULT,
                    help="meters per depth count (RealSense D435 = 1e-3)")
    ap.add_argument("--include-scene", action="store_true",
                    help="also emit subsampled non-wire scene points (label 4)")
    ap.add_argument("--save-overlays", action="store_true")
    ap.add_argument("--save-frames", action="store_true",
                    help="dump raw RGB+depth PNGs for inspection")
    ap.add_argument("--normed", action="store_true",
                    help="also write frustum-normalised pcl_/seg_ for pcl_to_rgbd")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames, intr = read_bag_frames(args.bag, args.num_frames, args.stride, args.start)
    print(f"intrinsics: {intr}")
    print(f"extracted {len(frames)} frames (stride {args.stride})")
    with open(out / "intrinsics.json", "w") as f:
        json.dump(intr, f, indent=2)

    manifest = []
    for j, fr in enumerate(frames):
        bgr, depth = fr["bgr"], fr["depth"]
        pts_m, labels, mask, dbg = bag_to_pointcloud(
            bgr, depth, intr, tuple(args.hsv_lo), tuple(args.hsv_hi),
            args.depth_unit_m, include_scene=args.include_scene)

        # Metric (un-normalised) cloud for geometry measurement.
        np.save(out / f"pcl_metric_{j:04d}.npy", pts_m)
        np.save(out / f"seg_{j:04d}.npy", labels)

        # Frustum-normalised cloud for the pcl_to_rgbd renderer (format bridge).
        if args.normed and len(pts_m):
            np.save(out / f"pcl_{j:04d}.npy", normalize_to_frustum(pts_m))

        if args.save_frames:
            cv2.imwrite(str(out / f"rgb_{j:04d}.png"), bgr)
            # depth visualised (scaled) for human inspection
            dv = np.clip(depth.astype(np.float32) / max(depth.max(), 1) * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out / f"depth_{j:04d}.png"), depth)            # raw 16-bit
            cv2.imwrite(str(out / f"depthvis_{j:04d}.png"),
                        cv2.applyColorMap(dv, cv2.COLORMAP_JET))

        if args.save_overlays:
            ov = bgr.copy()
            ov[mask] = (0, 0, 255)
            blend = cv2.addWeighted(bgr, 0.55, ov, 0.45, 0)
            cv2.imwrite(str(out / f"overlay_{j:04d}.png"), blend)

        print(f"  frame {j} (bag idx {fr['idx']}): "
              f"wire_px={dbg['n_wire_px']}  fg_frac={dbg['fg_fraction']*100:.3f}%  "
              f"wire_pts={int((labels==0).sum())}  total_pts={len(pts_m)}")
        manifest.append({"frame": j, "bag_idx": fr["idx"], **dbg,
                         "wire_pts": int((labels == 0).sum()),
                         "total_pts": int(len(pts_m))})

    with open(out / "manifest.json", "w") as f:
        json.dump({"intrinsics": intr, "frames": manifest}, f, indent=2)
    print(f"wrote outputs to {out}")


if __name__ == "__main__":
    main()
