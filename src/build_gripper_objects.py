#!/usr/bin/env python3
"""Procedurally generate robot gripper / end-effector / arm point clouds.

Phase 8 v2: each base shape is multiplied across 3 robot colour palettes for
~24 gripper/arm variants (up from v1's 8). Scale-jitter is wider so per-sample
randomness covers both small precision grippers and bulky industrial arms.

Same schema and conventions as :mod:`build_hand_objects`. Produces
``data/objects/gripper_*.npz`` and ``data/objects/arm_*.npz`` and updates the
manifest with category ``gripper`` (or ``arm`` for arm tubes).

Each gripper carries ``grasp_axis_local`` so the foreground placement code
can pose it gripping the wire (mirrors the hand-on-wire case).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
import trimesh.creation


# Three robot colour palettes (BGR uint8). v2 cycles each base shape through
# all three so the dataset has visual diversity in robot appearance, not just
# silhouette diversity.
PALETTES: list[dict] = [
    {  # 0 — Franka-ish: light/grey/black with blue accent.
        "name": "franka",
        "BLACK":   (32, 32, 36),
        "GREY":    (135, 138, 140),
        "LIGHT":   (200, 202, 205),
        "ACCENT":  (180, 110, 60),    # blue
        "RUBBER":  (40, 40, 40),
        "BRASS":   (90, 130, 165),
        "RED":     (60, 60, 200),
    },
    {  # 1 — Industrial: dark grey/orange (KUKA / ABB-ish).
        "name": "industrial",
        "BLACK":   (40, 40, 50),
        "GREY":    (90, 90, 100),
        "LIGHT":   (45, 95, 215),     # KUKA orange
        "ACCENT":  (180, 90, 40),
        "RUBBER":  (35, 35, 38),
        "BRASS":   (60, 80, 110),
        "RED":     (50, 60, 220),
    },
    {  # 2 — Yumi/sterile: white with red/grey accents.
        "name": "yumi",
        "BLACK":   (60, 60, 60),
        "GREY":    (200, 200, 205),
        "LIGHT":   (245, 245, 240),
        "ACCENT":  (60, 60, 200),     # red
        "RUBBER":  (45, 45, 50),
        "BRASS":   (130, 150, 165),
        "RED":     (50, 50, 215),
    },
]


def _mesh_with_color(mesh: trimesh.Trimesh,
                     bgr: tuple[int, int, int]) -> tuple[trimesh.Trimesh, tuple]:
    return mesh, bgr


def _translate(x: float, y: float, z: float) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3] = (x, y, z)
    return M


def _rot(angle: float, axis: list[float]) -> np.ndarray:
    return trimesh.transformations.rotation_matrix(angle, axis)


def _sample_coloured(parts: list[tuple[trimesh.Trimesh, tuple]],
                     n_total: int,
                     rng: np.random.RandomState,
                     ) -> tuple[np.ndarray, np.ndarray]:
    meshes = [m for m, _ in parts]
    cols = [c for _, c in parts]
    areas = np.array([float(m.area) for m in meshes])
    weights = areas / max(areas.sum(), 1e-12)
    counts = (weights * n_total).astype(np.int64)
    deficit = n_total - int(counts.sum())
    if deficit != 0:
        order = np.argsort(weights * n_total - counts)[::-1]
        for i in range(abs(deficit)):
            counts[order[i % len(order)]] += int(np.sign(deficit))

    pts_acc, col_acc = [], []
    for mesh, base_bgr, n in zip(meshes, cols, counts):
        if n <= 0 or mesh.area <= 0:
            continue
        pts, _ = trimesh.sample.sample_surface(
            mesh, int(n), seed=int(rng.randint(0, 2**32 - 1)))
        pts = np.asarray(pts, dtype=np.float64)
        base = np.array(base_bgr, dtype=np.int32)
        noise = rng.normal(0, 6.0, size=(pts.shape[0], 3))
        cols_out = np.clip(base[None, :] + noise, 0, 255).astype(np.uint8)
        pts_acc.append(pts)
        col_acc.append(cols_out)
    if not pts_acc:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    pts = np.concatenate(pts_acc, axis=0)
    cols = np.concatenate(col_acc, axis=0)
    if pts.shape[0] < n_total:
        pad_n = n_total - pts.shape[0]
        idx = rng.randint(0, pts.shape[0], size=pad_n)
        pts = np.concatenate([pts, pts[idx]], axis=0)
        cols = np.concatenate([cols, cols[idx]], axis=0)
    elif pts.shape[0] > n_total:
        idx = rng.choice(pts.shape[0], size=n_total, replace=False)
        pts = pts[idx]
        cols = cols[idx]
    return pts, cols


def _normalise_local_frame(points: np.ndarray) -> tuple[np.ndarray, dict]:
    pts = points.astype(np.float64).copy()
    y_min = pts[:, 1].min()
    pts[:, 1] -= y_min
    bbox_mid_x = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
    bbox_mid_z = 0.5 * (pts[:, 2].min() + pts[:, 2].max())
    pts[:, 0] -= bbox_mid_x
    pts[:, 2] -= bbox_mid_z
    extent = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2]))
    scale = 1.0 / extent if extent > 1e-9 else 1.0
    pts *= scale
    return pts, {"natural_size_units": float(extent)}


# Gripper builders — each takes a palette dict and returns coloured parts.

def _build_parallel_jaw(p: dict, open_distance: float = 0.085,
                         finger_length: float = 0.080,
                         ) -> list[tuple[trimesh.Trimesh, tuple]]:
    parts: list[tuple[trimesh.Trimesh, tuple]] = []

    flange = trimesh.creation.cylinder(radius=0.045, height=0.012, sections=24)
    flange.apply_translation((0, 0, -0.006))
    parts.append(_mesh_with_color(flange, p["BRASS"]))

    body = trimesh.creation.box(extents=(0.090, 0.080, 0.090))
    body.apply_translation((0, 0.040, 0.045))
    parts.append(_mesh_with_color(body, p["LIGHT"]))

    for sign in (+1, -1):
        rail = trimesh.creation.box(extents=(0.012, 0.020, 0.080))
        rail.apply_translation((sign * 0.045, 0.040, 0.090))
        parts.append(_mesh_with_color(rail, p["BLACK"]))

    half_open = open_distance / 2.0
    for sign in (+1, -1):
        knuckle = trimesh.creation.box(extents=(0.018, 0.030, 0.018))
        knuckle.apply_translation((sign * half_open, 0.040, 0.105))
        parts.append(_mesh_with_color(knuckle, p["BLACK"]))

        finger = trimesh.creation.box(extents=(0.012, 0.040, finger_length))
        finger.apply_translation(
            (sign * half_open, 0.040, 0.115 + finger_length / 2.0))
        parts.append(_mesh_with_color(finger, p["GREY"]))

        pad = trimesh.creation.box(extents=(0.005, 0.034, finger_length * 0.85))
        pad.apply_translation(
            (sign * (half_open - 0.0085), 0.040,
             0.115 + finger_length / 2.0))
        parts.append(_mesh_with_color(pad, p["RUBBER"]))

    return parts


def _build_industrial_jaw(p: dict) -> list[tuple[trimesh.Trimesh, tuple]]:
    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    flange = trimesh.creation.cylinder(radius=0.060, height=0.020)
    flange.apply_translation((0, 0, -0.010))
    parts.append(_mesh_with_color(flange, p["BRASS"]))

    body = trimesh.creation.box(extents=(0.140, 0.110, 0.120))
    body.apply_translation((0, 0.055, 0.060))
    parts.append(_mesh_with_color(body, p["GREY"]))

    body_top = trimesh.creation.box(extents=(0.110, 0.020, 0.140))
    body_top.apply_translation((0, 0.120, 0.070))
    parts.append(_mesh_with_color(body_top, p["BLACK"]))

    for sign in (+1, -1):
        finger = trimesh.creation.box(extents=(0.020, 0.090, 0.110))
        finger.apply_translation((sign * 0.040, 0.055, 0.180))
        parts.append(_mesh_with_color(finger, p["BLACK"]))
        tip = trimesh.creation.box(extents=(0.025, 0.025, 0.012))
        tip.apply_translation((sign * 0.040, 0.055, 0.241))
        parts.append(_mesh_with_color(tip, p["RUBBER"]))

    return parts


def _build_three_finger_gripper(p: dict) -> list[tuple[trimesh.Trimesh, tuple]]:
    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    flange = trimesh.creation.cylinder(radius=0.050, height=0.014)
    flange.apply_translation((0, 0, -0.007))
    parts.append(_mesh_with_color(flange, p["BRASS"]))

    body = trimesh.creation.cylinder(radius=0.060, height=0.080)
    body.apply_translation((0, 0, 0.040))
    parts.append(_mesh_with_color(body, p["LIGHT"]))

    palm = trimesh.creation.cylinder(radius=0.075, height=0.020)
    palm.apply_translation((0, 0, 0.080))
    parts.append(_mesh_with_color(palm, p["BLACK"]))

    for k in range(3):
        ang = k * 2.0 * np.pi / 3.0
        T0 = (
            _translate(0.05 * np.cos(ang), 0.05 * np.sin(ang), 0.090)
            @ _rot(ang, [0, 0, 1])
            @ _rot(np.deg2rad(20), [0, 1, 0])
        )
        ph0 = trimesh.creation.box(extents=(0.018, 0.025, 0.070))
        ph0.apply_translation((0, 0, 0.035))
        ph0.apply_transform(T0)
        parts.append(_mesh_with_color(ph0, p["GREY"]))

        T1 = T0 @ _translate(0, 0, 0.070) @ _rot(np.deg2rad(40), [0, 1, 0])
        ph1 = trimesh.creation.box(extents=(0.014, 0.020, 0.055))
        ph1.apply_translation((0, 0, 0.027))
        ph1.apply_transform(T1)
        parts.append(_mesh_with_color(ph1, p["BLACK"]))

    return parts


def _build_suction_cup(p: dict) -> list[tuple[trimesh.Trimesh, tuple]]:
    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    flange = trimesh.creation.cylinder(radius=0.040, height=0.010)
    flange.apply_translation((0, 0, -0.005))
    parts.append(_mesh_with_color(flange, p["BRASS"]))

    body = trimesh.creation.cylinder(radius=0.025, height=0.070)
    body.apply_translation((0, 0, 0.035))
    parts.append(_mesh_with_color(body, p["GREY"]))

    cup_top = trimesh.creation.cylinder(radius=0.020, height=0.005)
    cup_top.apply_translation((0, 0, 0.073))
    parts.append(_mesh_with_color(cup_top, p["BLACK"]))
    cup_skirt = trimesh.creation.cone(radius=0.045, height=0.040, sections=20)
    cup_skirt.apply_transform(_rot(np.pi, [1, 0, 0]))
    cup_skirt.apply_translation((0, 0, 0.115))
    parts.append(_mesh_with_color(cup_skirt, p["RED"]))

    n = 18
    hose_pts = np.zeros((n, 3))
    s = np.linspace(0, 1, n)
    hose_pts[:, 0] = 0.030 * np.sin(np.pi * s) ** 1.5
    hose_pts[:, 1] = 0.040 * s + 0.020
    hose_pts[:, 2] = -0.005 + (-0.005) * s
    hose = _curving_tube(hose_pts, radius=0.012)
    parts.append(_mesh_with_color(hose, p["BLACK"]))
    return parts


def _build_soft_four_finger(p: dict) -> list[tuple[trimesh.Trimesh, tuple]]:
    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    flange = trimesh.creation.cylinder(radius=0.040, height=0.012)
    flange.apply_translation((0, 0, -0.006))
    parts.append(_mesh_with_color(flange, p["BRASS"]))

    body = trimesh.creation.box(extents=(0.080, 0.060, 0.060))
    body.apply_translation((0, 0.030, 0.030))
    parts.append(_mesh_with_color(body, p["LIGHT"]))

    palm = trimesh.creation.box(extents=(0.110, 0.020, 0.080))
    palm.apply_translation((0, 0.060, 0.040))
    parts.append(_mesh_with_color(palm, p["BLACK"]))

    finger_x = [-0.040, -0.014, +0.014, +0.040]
    for k, fx in enumerate(finger_x):
        T = _translate(fx, 0.050, 0.080)
        for seg in range(3):
            T = T @ _rot(np.deg2rad(22), [1, 0, 0])
            ph = trimesh.creation.cylinder(radius=0.010 - seg * 0.001,
                                            height=0.030)
            ph.apply_translation((0, 0, 0.015))
            ph.apply_transform(T)
            parts.append(_mesh_with_color(
                ph, p["ACCENT"] if seg < 2 else p["RUBBER"]))
            T = T @ _translate(0, 0, 0.030)
    return parts


def _build_panda_arm_link(p: dict) -> list[tuple[trimesh.Trimesh, tuple]]:
    parts: list[tuple[trimesh.Trimesh, tuple]] = []

    flange = trimesh.creation.cylinder(radius=0.045, height=0.014)
    flange.apply_translation((0, 0, -0.007))
    parts.append(_mesh_with_color(flange, p["BRASS"]))

    upper = trimesh.creation.cylinder(radius=0.045, height=0.180)
    upper.apply_translation((0, 0, 0.090))
    parts.append(_mesh_with_color(upper, p["LIGHT"]))

    elbow = trimesh.creation.uv_sphere(radius=0.060)
    elbow.apply_translation((0, 0, 0.180))
    parts.append(_mesh_with_color(elbow, p["LIGHT"]))

    T = _translate(0, 0, 0.180) @ _rot(np.deg2rad(60), [1, 0, 0])
    forearm = trimesh.creation.cylinder(radius=0.038, height=0.180)
    forearm.apply_translation((0, 0, 0.090))
    forearm.apply_transform(T)
    parts.append(_mesh_with_color(forearm, p["LIGHT"]))

    for h in (0.018, 0.075, 0.165):
        ring = trimesh.creation.cylinder(radius=0.046, height=0.005)
        ring.apply_translation((0, 0, h))
        parts.append(_mesh_with_color(ring, p["BLACK"]))

    stripe = trimesh.creation.cylinder(radius=0.046, height=0.020)
    stripe.apply_translation((0, 0, 0.040))
    parts.append(_mesh_with_color(stripe, p["ACCENT"]))
    return parts


def _build_industrial_arm_link(p: dict) -> list[tuple[trimesh.Trimesh, tuple]]:
    parts: list[tuple[trimesh.Trimesh, tuple]] = []

    base = trimesh.creation.cylinder(radius=0.080, height=0.060)
    base.apply_translation((0, 0, 0.030))
    parts.append(_mesh_with_color(base, p["GREY"]))

    housing = trimesh.creation.box(extents=(0.150, 0.180, 0.140))
    housing.apply_translation((0, 0, 0.130))
    parts.append(_mesh_with_color(housing, p["LIGHT"]))

    T = _translate(0, 0, 0.200) @ _rot(np.deg2rad(45), [1, 0, 0])
    forearm = trimesh.creation.box(extents=(0.110, 0.110, 0.260))
    forearm.apply_translation((0, 0, 0.130))
    forearm.apply_transform(T)
    parts.append(_mesh_with_color(forearm, p["GREY"]))

    T2 = T @ _translate(0, 0, 0.260)
    wrist = trimesh.creation.cylinder(radius=0.055, height=0.060)
    wrist.apply_translation((0, 0, 0.030))
    wrist.apply_transform(T2)
    parts.append(_mesh_with_color(wrist, p["BLACK"]))
    return parts


def _curving_tube(centerline: np.ndarray, radius: float, sections: int = 12,
                   ) -> trimesh.Trimesh:
    parts = []
    for i in range(centerline.shape[0] - 1):
        a, b = centerline[i], centerline[i + 1]
        v = b - a
        length = float(np.linalg.norm(v))
        if length < 1e-7:
            continue
        c = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
        z = np.array([0.0, 0.0, 1.0])
        v_unit = v / length
        if np.allclose(v_unit, z):
            R = np.eye(4)
        elif np.allclose(v_unit, -z):
            R = _rot(np.pi, [1, 0, 0])
        else:
            axis = np.cross(z, v_unit)
            axis_n = axis / np.linalg.norm(axis)
            angle = float(np.arccos(np.clip(np.dot(z, v_unit), -1.0, 1.0)))
            R = _rot(angle, axis_n.tolist())
        c.apply_transform(R)
        c.apply_translation((a + b) / 2.0)
        parts.append(c)
    return trimesh.util.concatenate(parts) if parts else trimesh.Trimesh()


@dataclass
class GripperVariant:
    name: str
    description: str
    category: str  # "gripper" or "arm"
    builder: callable
    natural_scale: float
    grasp_axis_local: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    graspable_on_wire: bool = False
    builder_kwargs: dict = field(default_factory=dict)


GRIPPER_VARIANTS: list[GripperVariant] = [
    GripperVariant(
        name="parallel_jaw_closed",
        description="Robotiq 2F-85-style parallel-jaw gripper, jaws nearly closed.",
        category="gripper",
        builder=_build_parallel_jaw,
        natural_scale=0.18,
        grasp_axis_local=[1.0, 0.0, 0.0],
        graspable_on_wire=True,
        builder_kwargs={"open_distance": 0.020, "finger_length": 0.080},
    ),
    GripperVariant(
        name="parallel_jaw_open",
        description="Robotiq 2F-85-style parallel-jaw gripper, jaws fully open.",
        category="gripper",
        builder=_build_parallel_jaw,
        natural_scale=0.20,
        grasp_axis_local=[1.0, 0.0, 0.0],
        graspable_on_wire=True,
        builder_kwargs={"open_distance": 0.085, "finger_length": 0.080},
    ),
    GripperVariant(
        name="industrial_jaw",
        description="Bulky industrial parallel-jaw gripper.",
        category="gripper",
        builder=_build_industrial_jaw,
        natural_scale=0.30,
        grasp_axis_local=[1.0, 0.0, 0.0],
        graspable_on_wire=True,
    ),
    GripperVariant(
        name="three_finger",
        description="3-finger Barrett-style splay gripper.",
        category="gripper",
        builder=_build_three_finger_gripper,
        natural_scale=0.22,
        grasp_axis_local=[0.0, 0.0, 1.0],
        graspable_on_wire=False,
    ),
    GripperVariant(
        name="suction_cup",
        description="Pneumatic suction-cup end effector with hose.",
        category="gripper",
        builder=_build_suction_cup,
        natural_scale=0.18,
        grasp_axis_local=[0.0, 0.0, 1.0],
        graspable_on_wire=False,
    ),
    GripperVariant(
        name="soft_four_finger",
        description="Soft-pneumatic 4-finger gripper, partially closed.",
        category="gripper",
        builder=_build_soft_four_finger,
        natural_scale=0.22,
        grasp_axis_local=[1.0, 0.0, 0.0],
        graspable_on_wire=True,
    ),
    GripperVariant(
        name="arm_panda",
        description="Franka Panda-style arm link with elbow.",
        category="arm",
        builder=_build_panda_arm_link,
        natural_scale=0.45,
        grasp_axis_local=[0.0, 0.0, 1.0],
        graspable_on_wire=False,
    ),
    GripperVariant(
        name="arm_industrial",
        description="Bulky industrial 6-axis arm link.",
        category="arm",
        builder=_build_industrial_arm_link,
        natural_scale=0.55,
        grasp_axis_local=[0.0, 0.0, 1.0],
        graspable_on_wire=False,
    ),
]


def _sample_variant(v: GripperVariant, palette: dict, n_points: int, seed: int,
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.RandomState(seed)
    parts = v.builder(palette, **v.builder_kwargs)
    pts, cols = _sample_coloured(parts, n_points, rng)
    pts_norm, calib = _normalise_local_frame(pts)
    calib["grasp_axis_local"] = list(v.grasp_axis_local)
    return pts_norm, cols, calib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/objects"))
    parser.add_argument("--n-points", type=int, default=8000)
    parser.add_argument("--seed-base", type=int, default=2027)
    parser.add_argument("--scale-jitter", type=float, default=0.20)
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Drop existing gripper / arm entries; we rewrite them.
    manifest["objects"] = [o for o in manifest.get("objects", [])
                           if o.get("category") not in ("gripper", "arm")]
    # Wipe stale gripper_*.npz / arm_*.npz files (v1 schema had different naming).
    for stale in out_dir.glob("gripper_*.npz"):
        stale.unlink()
    for stale in out_dir.glob("arm_*.npz"):
        stale.unlink()

    new_entries: list[dict] = []
    written = 0
    for i, v in enumerate(GRIPPER_VARIANTS):
        for pi, palette in enumerate(PALETTES):
            slug_prefix = "gripper" if v.category == "gripper" else "arm"
            # arm_panda → arm_panda_franka, gripper_parallel_jaw_closed → gripper_parallel_jaw_closed_franka
            stem = v.name if v.name.startswith(slug_prefix) else f"{slug_prefix}_{v.name}"
            slug = f"{stem}_{palette['name']}"
            seed = args.seed_base + 31 * i + 97 * pi
            pts, cols, calib = _sample_variant(v, palette, args.n_points, seed)

            out_path = out_dir / f"{slug}.npz"
            np.savez_compressed(out_path,
                                points=pts.astype(np.float32),
                                colors=cols.astype(np.uint8))
            scale_lo = v.natural_scale * (1.0 - args.scale_jitter)
            scale_hi = v.natural_scale * (1.0 + args.scale_jitter)
            entry = {
                "slug": slug,
                "file": f"{slug}.npz",
                "category": v.category,
                "description": f"{v.description} ({palette['name']} colour palette).",
                "source_url": "procedurally generated (src/build_gripper_objects.py)",
                "license": "internal",
                "source": "procedural",
                "n_points": int(pts.shape[0]),
                "natural_scale_units": v.natural_scale,
                "natural_scale_range": [round(scale_lo, 4), round(scale_hi, 4)],
                "palette": palette["name"],
                "shape_family": v.name,
                "grasp_axis_local": calib["grasp_axis_local"],
                "graspable_on_wire": bool(v.graspable_on_wire),
            }
            new_entries.append(entry)
            written += 1

    manifest["objects"].extend(new_entries)
    manifest["total"] = len(manifest["objects"])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {written} gripper/arm variants  "
          f"({len(GRIPPER_VARIANTS)} shapes × {len(PALETTES)} palettes)")
    print(f"  manifest: {manifest_path}  total now: {manifest['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
