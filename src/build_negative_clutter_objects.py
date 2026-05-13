#!/usr/bin/env python3
"""Procedurally generate "negative similarity" clutter objects.

Phase 8 v2: this module produces ~100 negative-clutter / clutter PCLs (up from
v1's 15). Cables specifically multiply 12 insulation colours × 5 shape seeds
to give 60 random-curve cables — by far the most important wire-shaped
negatives the model has never seen in Phase 4 / Phase 5 / Phase 7 training.

These are the *critical* additions for Phase 8 — visually wire-like things
that should NOT be classified as DLO. The model has zero such negatives in
the Phase 4 data, which is why it false-positives skin/fingers as wire
(Phase 7.5 finding).

Categories produced:
    * **negative_wire_like** — power cables, USB cables, headphone cords,
      extension cords, garden hoses. Curving cylindrical shape (≈ wire-shaped)
      with non-wire colours and bulky end connectors.
    * **rope** — coiled or curving thick rope. Wider than a wire, fibre
      texture cue.
    * **clutter** — chairs, monitors, keyboards, plants, bottles, mugs,
      book stacks. Different shapes altogether to broaden the workshop scene
      variety. v2 multiplies each by colour variants so 4 base shapes become
      ~21 entries.

All objects follow the same .npz schema as data/objects/*.npz.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
import trimesh.creation


# 12 cable colours (BGR uint8). v1 had 6; v2 expands the palette so the model
# sees iridescent/lime/coral/beige cables, not just the typical black/white.
CABLE_INSULATION_BGR: list[tuple[int, int, int]] = [
    (24, 24, 28),       # 0 black power cable
    (200, 195, 190),    # 1 white phone charger
    (45, 70, 120),      # 2 navy fabric-braided cable
    (140, 50, 30),      # 3 red headphone cord
    (60, 90, 50),       # 4 green ethernet sheath
    (175, 175, 30),     # 5 yellow coiled cable
    (150, 165, 185),    # 6 beige (skin-toned cable; intentionally hand-coloured)
    (40, 65, 95),       # 7 brown leather-look cable
    (45, 200, 55),      # 8 lime high-vis power cable
    (160, 80, 200),     # 9 hot pink decorative cable
    (165, 165, 170),    # 10 silver iridescent cable
    (60, 120, 210),     # 11 coral/orange extension cord
]

CABLE_CONNECTOR_BGR: tuple[int, int, int] = (40, 40, 50)

# 4 rope colours
ROPE_BGR: list[tuple[int, int, int]] = [
    (95, 130, 165),     # tan jute
    (60, 90, 130),      # darker brown
    (180, 180, 180),    # white nylon
    (50, 65, 90),       # deep brown
]

# Other clutter palettes (small, tasteful palettes per object family).
KEYBOARD_PALETTES: list[dict] = [
    {"name": "graphite", "BASE": (40, 40, 45), "KEY": (70, 70, 75)},
    {"name": "white",    "BASE": (210, 210, 215), "KEY": (170, 170, 175)},
    {"name": "rose",     "BASE": (110, 100, 175), "KEY": (140, 130, 195)},
]
MONITOR_PALETTES: list[dict] = [
    {"name": "black",  "BASE": (28, 28, 32), "BEZEL": (15, 15, 18), "SCREEN": (110, 110, 110)},
    {"name": "silver", "BASE": (130, 130, 140), "BEZEL": (40, 40, 50), "SCREEN": (90, 90, 100)},
    {"name": "white",  "BASE": (220, 220, 225), "BEZEL": (40, 40, 50), "SCREEN": (130, 130, 130)},
]
CHAIR_PALETTES: list[dict] = [
    {"name": "black",  "FRAME": (32, 32, 36), "FABRIC": (90, 100, 110)},
    {"name": "navy",   "FRAME": (40, 40, 45), "FABRIC": (70, 50, 35)},
    {"name": "ivory",  "FRAME": (140, 140, 145), "FABRIC": (215, 220, 225)},
]
PLANT_PALETTES: list[dict] = [
    {"name": "monstera",   "POT": (75, 95, 145),  "SOIL": (40, 55, 75),  "LEAF": (60, 130, 60),  "STEM": (45, 90, 60)},
    {"name": "succulent",  "POT": (140, 170, 200), "SOIL": (35, 50, 70),  "LEAF": (80, 160, 100), "STEM": (60, 110, 80)},
    {"name": "fern",       "POT": (95, 110, 150),  "SOIL": (30, 45, 65),  "LEAF": (40, 110, 50),  "STEM": (35, 70, 50)},
]
BOTTLE_PALETTES: list[dict] = [
    {"name": "water",  "BODY": (180, 200, 215), "CAP": (50, 80, 130)},
    {"name": "soda",   "BODY": (80, 60, 200), "CAP": (210, 210, 215)},
    {"name": "amber",  "BODY": (40, 90, 145), "CAP": (35, 35, 40)},
]
MUG_PALETTES: list[dict] = [
    {"name": "white",  "BODY": (235, 235, 240), "RIM": (170, 170, 175)},
    {"name": "blue",   "BODY": (180, 110, 50),  "RIM": (130, 80, 35)},
    {"name": "ceramic","BODY": (170, 195, 220), "RIM": (130, 150, 175)},
]
BOOK_PALETTES: list[dict] = [
    {"name": "library", "COVERS": [(30, 40, 80), (60, 90, 60), (80, 60, 120)]},
    {"name": "office",  "COVERS": [(40, 40, 45), (215, 215, 220), (35, 90, 145)]},
    {"name": "warm",    "COVERS": [(60, 60, 165), (45, 110, 175), (175, 130, 90)]},
]


def _rot(angle: float, axis: list[float]) -> np.ndarray:
    return trimesh.transformations.rotation_matrix(angle, axis)


def _translate(x: float, y: float, z: float) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3] = (x, y, z)
    return M


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


def _sample_coloured(parts: list[tuple[trimesh.Trimesh, tuple]],
                     n_total: int, rng: np.random.RandomState,
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


def _curving_tube(centerline: np.ndarray, radius: float, sections: int = 10,
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


# CABLE — the critical negative sample

def _build_cable_random_curve(seed: int,
                               length: float = 0.55,
                               radius: float = 0.005,
                               n_segs: int = 28,
                               curl: float = 0.12,
                               with_connectors: bool = True,
                               insulation_bgr: tuple[int, int, int] = (24, 24, 28),
                               ) -> list[tuple[trimesh.Trimesh, tuple]]:
    rng = np.random.RandomState(seed)
    s = np.linspace(0, 1, n_segs)
    z = (s - 0.5) * length

    def wander(amp: float, freq_low: int = 1, freq_high: int = 4) -> np.ndarray:
        out = np.zeros_like(s)
        for k in range(rng.randint(2, 5)):
            f = rng.randint(freq_low, freq_high + 1)
            phi = rng.uniform(0, 2 * np.pi)
            a = rng.uniform(0.4, 1.0) * amp
            out += a * np.sin(2 * np.pi * f * s + phi)
        return out

    x = wander(curl)
    y = wander(curl * 0.6)
    centerline = np.stack([x, y, z], axis=1)

    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    tube = _curving_tube(centerline, radius=radius)
    parts.append((tube, insulation_bgr))

    if with_connectors:
        for end_idx in (0, n_segs - 1):
            tip = centerline[end_idx]
            plug = trimesh.creation.box(
                extents=(radius * 6, radius * 6, radius * 8))
            plug.apply_translation(tip)
            parts.append((plug, CABLE_CONNECTOR_BGR))
            pin = trimesh.creation.box(
                extents=(radius * 1.2, radius * 1.2, radius * 5))
            offset = np.array([0.0, 0.0, radius * 4]) * (1 if end_idx else -1)
            pin.apply_translation(tip + offset)
            parts.append((pin, (135, 135, 130)))
    return parts


def _build_cable_coiled(seed: int,
                         insulation_bgr: tuple[int, int, int] = (24, 24, 28),
                         ) -> list[tuple[trimesh.Trimesh, tuple]]:
    rng = np.random.RandomState(seed)
    n_loops = rng.randint(2, 4)
    coil_radius = rng.uniform(0.10, 0.16)
    cable_radius = 0.006
    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    n_per_loop = 36
    s = np.linspace(0, 1, n_loops * n_per_loop)
    theta = 2 * np.pi * (s * n_loops)
    radius_per_loop = coil_radius - 0.018 * np.floor(s * n_loops)
    cx = radius_per_loop * np.cos(theta)
    cy = 0.005 + 0.014 * np.floor(s * n_loops)
    cz = radius_per_loop * np.sin(theta)
    centerline = np.stack([cx, cy, cz], axis=1)
    tube = _curving_tube(centerline, radius=cable_radius, sections=8)
    parts.append((tube, insulation_bgr))

    tail_n = 12
    s2 = np.linspace(0, 1, tail_n)
    tail = np.stack([
        radius_per_loop[0] * np.cos(theta[0]) + 0.05 * s2,
        np.zeros(tail_n) + 0.005,
        radius_per_loop[0] * np.sin(theta[0]) + 0.10 * s2 ** 1.2,
    ], axis=1)
    parts.append((_curving_tube(tail, radius=cable_radius, sections=8),
                   insulation_bgr))
    plug = trimesh.creation.box(extents=(0.030, 0.018, 0.040))
    plug.apply_translation(tail[-1])
    parts.append((plug, CABLE_CONNECTOR_BGR))
    return parts


def _build_extension_cord_with_outlet(seed: int,
                                       insulation_bgr: tuple[int, int, int] = (24, 24, 28),
                                       strip_bgr: tuple[int, int, int] = (200, 200, 200),
                                       ) -> list[tuple[trimesh.Trimesh, tuple]]:
    rng = np.random.RandomState(seed)
    parts: list[tuple[trimesh.Trimesh, tuple]] = []

    strip = trimesh.creation.box(extents=(0.260, 0.025, 0.060))
    strip.apply_translation((0.130, 0.0125, 0))
    parts.append((strip, strip_bgr))

    for i in range(4):
        sx = 0.040 + i * 0.060
        sock = trimesh.creation.box(extents=(0.038, 0.005, 0.038))
        sock.apply_translation((sx, 0.025 + 0.0025, 0))
        parts.append((sock, (15, 15, 15)))
        for s in (-1, 1):
            slot = trimesh.creation.box(extents=(0.002, 0.003, 0.014))
            slot.apply_translation((sx + s * 0.008, 0.025 + 0.004, 0))
            parts.append((slot, (60, 60, 60)))

    s = np.linspace(0, 1, 26)
    cz = 0.030 + 0.55 * s
    cx = 0.260 + 0.10 * np.sin(np.pi * s)
    cy = 0.012 + 0.05 * (1 - s) ** 1.5
    centerline = np.stack([cx, cy, cz], axis=1)
    tube = _curving_tube(centerline, radius=0.006)
    parts.append((tube, insulation_bgr))

    plug = trimesh.creation.box(extents=(0.030, 0.030, 0.040))
    plug.apply_translation(centerline[-1])
    parts.append((plug, CABLE_CONNECTOR_BGR))
    pin = trimesh.creation.box(extents=(0.005, 0.005, 0.020))
    pin.apply_translation(centerline[-1] + np.array([0, 0, 0.025]))
    parts.append((pin, (135, 135, 130)))
    return parts


def _build_garden_hose(seed: int,
                        insulation_bgr: tuple[int, int, int] = (40, 130, 80),
                        ) -> list[tuple[trimesh.Trimesh, tuple]]:
    """Thicker, more aggressively coiled garden hose with brass nozzle."""
    rng = np.random.RandomState(seed)
    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    n_loops = rng.randint(3, 5)
    coil_radius = rng.uniform(0.16, 0.22)
    hose_radius = rng.uniform(0.015, 0.022)

    n_per_loop = 32
    s = np.linspace(0, 1, n_loops * n_per_loop)
    theta = 2 * np.pi * (s * n_loops)
    radius_per_loop = coil_radius - 0.012 * np.floor(s * n_loops) * 0.5
    cx = radius_per_loop * np.cos(theta)
    cy = hose_radius * 1.2 + 0.020 * np.floor(s * n_loops)
    cz = radius_per_loop * np.sin(theta)
    centerline = np.stack([cx, cy, cz], axis=1)
    parts.append((_curving_tube(centerline, radius=hose_radius, sections=12),
                   insulation_bgr))

    # Brass nozzle at one end
    nozzle = trimesh.creation.cylinder(radius=hose_radius * 1.4, height=0.060)
    nozzle.apply_translation(centerline[-1] + np.array([0.030, 0, 0]))
    parts.append((nozzle, (90, 130, 165)))
    return parts


# ROPE

def _build_rope(seed: int,
                 base_bgr: tuple[int, int, int] = (95, 130, 165),
                 ) -> list[tuple[trimesh.Trimesh, tuple]]:
    rng = np.random.RandomState(seed)
    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    radius = rng.uniform(0.014, 0.022)

    def s_curve(seed2):
        rng2 = np.random.RandomState(seed2)
        n = 18
        s = np.linspace(0, 1, n)
        cx = 0.20 * np.sin(2.5 * np.pi * s + rng2.uniform(0, np.pi))
        cy = radius + 0.005
        cz = (s - 0.5) * 0.50 + rng2.uniform(-0.05, 0.05)
        return np.stack([cx, np.full(n, cy), cz], axis=1)
    parts.append((_curving_tube(s_curve(seed), radius=radius, sections=10),
                  base_bgr))
    parts.append((_curving_tube(s_curve(seed + 7), radius=radius, sections=10),
                  base_bgr))
    return parts


# CLUTTER — broader workshop variety

def _build_keyboard(p: dict) -> list[tuple[trimesh.Trimesh, tuple]]:
    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    base = trimesh.creation.box(extents=(0.450, 0.025, 0.150))
    base.apply_translation((0, 0.0125, 0))
    parts.append((base, p["BASE"]))
    for r in range(5):
        for c in range(16):
            key = trimesh.creation.box(extents=(0.022, 0.006, 0.022))
            x = -0.205 + c * 0.027
            z = -0.060 + r * 0.030
            key.apply_translation((x, 0.025 + 0.003, z))
            parts.append((key, p["KEY"]))
    return parts


def _build_monitor(p: dict) -> list[tuple[trimesh.Trimesh, tuple]]:
    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    base = trimesh.creation.cylinder(radius=0.085, height=0.020)
    base.apply_translation((0, 0.010, 0))
    parts.append((base, p["BASE"]))
    pole = trimesh.creation.box(extents=(0.030, 0.180, 0.030))
    pole.apply_translation((0, 0.110, 0))
    parts.append((pole, p["BASE"]))
    screen_w, screen_h, screen_d = 0.420, 0.250, 0.020
    screen = trimesh.creation.box(extents=(screen_w, screen_h, screen_d))
    screen.apply_translation((0, 0.300, -0.010))
    parts.append((screen, p["BEZEL"]))
    display = trimesh.creation.box(extents=(screen_w * 0.94, screen_h * 0.94, 0.005))
    display.apply_translation((0, 0.300, 0.002))
    parts.append((display, p["SCREEN"]))
    return parts


def _build_chair(p: dict) -> list[tuple[trimesh.Trimesh, tuple]]:
    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    for k in range(5):
        ang = k * 2 * np.pi / 5
        leg = trimesh.creation.box(extents=(0.250, 0.020, 0.040))
        leg.apply_translation((0.125, 0.010, 0))
        leg.apply_transform(_rot(ang, [0, 1, 0]))
        parts.append((leg, p["FRAME"]))
    pole = trimesh.creation.cylinder(radius=0.025, height=0.420)
    pole.apply_translation((0, 0.220, 0))
    parts.append((pole, p["FRAME"]))
    seat = trimesh.creation.box(extents=(0.480, 0.060, 0.450))
    seat.apply_translation((0, 0.460, 0))
    parts.append((seat, p["FABRIC"]))
    back = trimesh.creation.box(extents=(0.460, 0.500, 0.040))
    back.apply_translation((0, 0.770, -0.220))
    parts.append((back, p["FABRIC"]))
    return parts


def _build_potted_plant(p: dict, seed: int = 0) -> list[tuple[trimesh.Trimesh, tuple]]:
    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    pot = trimesh.creation.cylinder(radius=0.090, height=0.110)
    pot.apply_translation((0, 0.055, 0))
    parts.append((pot, p["POT"]))
    soil = trimesh.creation.cylinder(radius=0.085, height=0.012)
    soil.apply_translation((0, 0.110, 0))
    parts.append((soil, p["SOIL"]))
    rng = np.random.RandomState(seed)
    n_leaf = 18
    for k in range(n_leaf):
        r = rng.uniform(0.030, 0.060)
        x = rng.uniform(-0.06, 0.06)
        z = rng.uniform(-0.06, 0.06)
        y = 0.140 + rng.uniform(0.0, 0.180)
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=r)
        sphere.apply_translation((x, y, z))
        S = np.diag([1.0, rng.uniform(0.6, 0.9), 1.0, 1.0])
        sphere.apply_transform(S)
        green_var = (max(0, p["LEAF"][0] + rng.randint(-15, 15)),
                     max(0, p["LEAF"][1] + rng.randint(-15, 15)),
                     max(0, p["LEAF"][2] + rng.randint(-15, 15)))
        parts.append((sphere, green_var))
    stem = trimesh.creation.cylinder(radius=0.012, height=0.140)
    stem.apply_translation((0, 0.170, 0))
    parts.append((stem, p["STEM"]))
    return parts


def _build_headphones(seed: int,
                       earcup_bgr: tuple[int, int, int] = (32, 32, 36),
                       cable_bgr: tuple[int, int, int] = (40, 40, 44),
                       ) -> list[tuple[trimesh.Trimesh, tuple]]:
    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    n = 18
    s = np.linspace(0, np.pi, n)
    radius = 0.090
    centre = np.stack([
        radius * np.sin(s),
        0.110 - radius * (1.0 - np.cos(s)) * 0.5,
        np.zeros(n),
    ], axis=1)
    centre[:, 0] -= radius * np.sin(s).mean()
    headband = _curving_tube(centre, radius=0.012, sections=10)
    parts.append((headband, earcup_bgr))
    for sign in (-1, +1):
        cup = trimesh.creation.cylinder(radius=0.040, height=0.030)
        cup.apply_transform(_rot(np.pi / 2, [0, 0, 1]))
        cup.apply_translation((sign * radius, 0.055, 0))
        parts.append((cup, earcup_bgr))
        pad = trimesh.creation.cylinder(radius=0.044, height=0.018)
        pad.apply_transform(_rot(np.pi / 2, [0, 0, 1]))
        pad.apply_translation((sign * radius - sign * 0.020, 0.055, 0))
        parts.append((pad, (16, 16, 18)))
    s2 = np.linspace(0, 1, 22)
    cable = np.stack([
        -radius - 0.020 + 0.05 * s2,
        0.050 - 0.090 * s2,
        0.000 + 0.080 * s2 ** 1.2,
    ], axis=1)
    parts.append((_curving_tube(cable, radius=0.0035, sections=8), cable_bgr))
    return parts


def _build_bottle(p: dict) -> list[tuple[trimesh.Trimesh, tuple]]:
    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    body = trimesh.creation.cylinder(radius=0.040, height=0.180, sections=24)
    body.apply_translation((0, 0.090, 0))
    parts.append((body, p["BODY"]))
    neck = trimesh.creation.cylinder(radius=0.015, height=0.030)
    neck.apply_translation((0, 0.195, 0))
    parts.append((neck, p["BODY"]))
    cap = trimesh.creation.cylinder(radius=0.018, height=0.025)
    cap.apply_translation((0, 0.222, 0))
    parts.append((cap, p["CAP"]))
    return parts


def _build_mug(p: dict) -> list[tuple[trimesh.Trimesh, tuple]]:
    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    body = trimesh.creation.cylinder(radius=0.045, height=0.105, sections=24)
    body.apply_translation((0, 0.0525, 0))
    parts.append((body, p["BODY"]))
    rim = trimesh.creation.cylinder(radius=0.046, height=0.008, sections=24)
    rim.apply_translation((0, 0.108, 0))
    parts.append((rim, p["RIM"]))
    # Handle: torus segment via curving tube.
    n = 12
    s = np.linspace(-np.pi / 2, np.pi / 2, n)
    handle_r = 0.030
    centre = np.stack([
        0.045 + handle_r * np.cos(s),
        0.060 + handle_r * np.sin(s),
        np.zeros(n),
    ], axis=1)
    parts.append((_curving_tube(centre, radius=0.008, sections=8), p["BODY"]))
    return parts


def _build_book_stack(p: dict, seed: int = 0) -> list[tuple[trimesh.Trimesh, tuple]]:
    rng = np.random.RandomState(seed)
    parts: list[tuple[trimesh.Trimesh, tuple]] = []
    covers = p["COVERS"]
    y = 0.0
    n_books = 5
    for k in range(n_books):
        w = rng.uniform(0.15, 0.22)
        h = rng.uniform(0.025, 0.045)
        d = rng.uniform(0.20, 0.28)
        offset_x = rng.uniform(-0.015, 0.015)
        offset_z = rng.uniform(-0.015, 0.015)
        rot_y = rng.uniform(-0.12, 0.12)
        book = trimesh.creation.box(extents=(w, h, d))
        book.apply_transform(_rot(rot_y, [0, 1, 0]))
        book.apply_translation((offset_x, y + h / 2.0, offset_z))
        parts.append((book, covers[k % len(covers)]))
        y += h
    return parts


# Variant table

@dataclass
class ClutterVariant:
    name: str
    description: str
    category: str
    builder: callable
    natural_scale: float
    grasp_axis_local: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 1.0])
    graspable_on_wire: bool = False
    builder_kwargs: dict = field(default_factory=dict)
    seed_offset: int = 0


def _build_clutter_variants() -> list[ClutterVariant]:
    """Construct the full variant list — 60 random-curve cables + others."""
    variants: list[ClutterVariant] = []

    # 12 colours × 5 shape seeds = 60 random-curve cables
    for tone_idx, bgr in enumerate(CABLE_INSULATION_BGR):
        for shape_seed in range(5):
            length = 0.42 + 0.06 * shape_seed                # 0.42..0.66
            curl = 0.08 + 0.025 * shape_seed                  # 0.08..0.18
            radius = 0.0045 + 0.0008 * (shape_seed % 3)
            slug = f"cable_t{tone_idx:02d}_s{shape_seed}"
            variants.append(ClutterVariant(
                name=slug,
                description=(f"Random-curve cable, length {length:.2f} m, "
                             f"curl {curl:.2f}, colour idx {tone_idx}."),
                category="negative_wire_like",
                builder=_build_cable_random_curve,
                natural_scale=length,
                grasp_axis_local=[0.0, 0.0, 1.0],
                graspable_on_wire=False,
                seed_offset=tone_idx * 17 + shape_seed,
                builder_kwargs={"length": length, "radius": radius, "curl": curl,
                                 "n_segs": int(28 + length * 30),
                                 "with_connectors": True,
                                 "insulation_bgr": bgr},
            ))

    # Coiled cables × 4 colours
    for tone_idx in (0, 2, 5, 11):
        slug = f"cable_coiled_t{tone_idx:02d}"
        variants.append(ClutterVariant(
            name=slug,
            description=f"Coiled cable lying flat with tail and plug (colour {tone_idx}).",
            category="negative_wire_like",
            builder=_build_cable_coiled,
            natural_scale=0.30,
            seed_offset=11 + tone_idx * 7,
            builder_kwargs={"insulation_bgr": CABLE_INSULATION_BGR[tone_idx]},
        ))

    # Extension cords × 3 (cable colour, strip colour pairs)
    for tone_idx, strip_bgr in [(0, (200, 200, 200)),
                                  (8, (220, 220, 220)),
                                  (11, (60, 60, 60))]:
        slug = f"extension_cord_t{tone_idx:02d}"
        variants.append(ClutterVariant(
            name=slug,
            description=f"Power strip with 4 outlets and extension cord (colour {tone_idx}).",
            category="negative_wire_like",
            builder=_build_extension_cord_with_outlet,
            natural_scale=0.65,
            seed_offset=13 + tone_idx * 11,
            builder_kwargs={
                "insulation_bgr": CABLE_INSULATION_BGR[tone_idx],
                "strip_bgr": strip_bgr,
            },
        ))

    # Garden hoses × 3 colours
    for slug, bgr in [("garden_hose_green", (40, 130, 80)),
                       ("garden_hose_yellow", (40, 165, 200)),
                       ("garden_hose_grey",   (110, 110, 115))]:
        variants.append(ClutterVariant(
            name=slug,
            description=f"Coiled garden hose ({slug.split('_')[-1]}).",
            category="negative_wire_like",
            builder=_build_garden_hose,
            natural_scale=0.55,
            seed_offset=29 + hash(slug) % 100,
            builder_kwargs={"insulation_bgr": bgr},
        ))

    # Headphones × 4 colours
    headphone_palettes = [
        ("headphones_black",  (32, 32, 36),  (40, 40, 44)),
        ("headphones_white",  (210, 210, 215), (165, 165, 170)),
        ("headphones_red",    (45, 45, 175),  (55, 55, 165)),
        ("headphones_navy",   (95, 50, 35),   (100, 60, 45)),
    ]
    for i, (slug, ear, cab) in enumerate(headphone_palettes):
        variants.append(ClutterVariant(
            name=slug,
            description=f"Over-ear headphones with curving aux cable ({slug.split('_')[-1]}).",
            category="negative_wire_like",
            builder=_build_headphones,
            natural_scale=0.32,
            seed_offset=37 + i * 13,
            builder_kwargs={"earcup_bgr": ear, "cable_bgr": cab},
        ))

    # Ropes × 4 colours
    for ti, bgr in enumerate(ROPE_BGR):
        slug = f"rope_t{ti:02d}"
        variants.append(ClutterVariant(
            name=slug,
            description=f"Coiled rope ({['tan','brown','white','deep brown'][ti]}).",
            category="rope",
            builder=_build_rope,
            natural_scale=0.55 if ti < 2 else 0.50,
            seed_offset=43 + ti * 7,
            builder_kwargs={"base_bgr": bgr},
        ))

    # Keyboards × 3
    for p in KEYBOARD_PALETTES:
        slug = f"keyboard_{p['name']}"
        variants.append(ClutterVariant(
            name=slug,
            description=f"Standard PC keyboard ({p['name']}).",
            category="clutter",
            builder=_build_keyboard,
            natural_scale=0.50,
            builder_kwargs={"p": p},
        ))

    # Monitors × 3
    for p in MONITOR_PALETTES:
        slug = f"monitor_{p['name']}"
        variants.append(ClutterVariant(
            name=slug,
            description=f"Computer monitor on stand ({p['name']}).",
            category="clutter",
            builder=_build_monitor,
            natural_scale=0.65,
            builder_kwargs={"p": p},
        ))

    # Chairs × 3
    for p in CHAIR_PALETTES:
        slug = f"office_chair_{p['name']}"
        variants.append(ClutterVariant(
            name=slug,
            description=f"Office chair with star base ({p['name']}).",
            category="clutter",
            builder=_build_chair,
            natural_scale=1.10,
            builder_kwargs={"p": p},
        ))

    # Potted plants × 3
    for i, p in enumerate(PLANT_PALETTES):
        slug = f"potted_plant_{p['name']}"
        variants.append(ClutterVariant(
            name=slug,
            description=f"Potted plant ({p['name']}).",
            category="clutter",
            builder=_build_potted_plant,
            natural_scale=0.55,
            seed_offset=53 + i * 7,
            builder_kwargs={"p": p, "seed": 100 + i},
        ))

    # Bottles × 3
    for p in BOTTLE_PALETTES:
        slug = f"bottle_{p['name']}"
        variants.append(ClutterVariant(
            name=slug,
            description=f"Plastic bottle ({p['name']}).",
            category="clutter",
            builder=_build_bottle,
            natural_scale=0.25,
            builder_kwargs={"p": p},
        ))

    # Mugs × 3
    for p in MUG_PALETTES:
        slug = f"mug_{p['name']}"
        variants.append(ClutterVariant(
            name=slug,
            description=f"Coffee mug ({p['name']}).",
            category="clutter",
            builder=_build_mug,
            natural_scale=0.13,
            builder_kwargs={"p": p},
        ))

    # Book stacks × 3
    for i, p in enumerate(BOOK_PALETTES):
        slug = f"book_stack_{p['name']}"
        variants.append(ClutterVariant(
            name=slug,
            description=f"Stack of 5 books ({p['name']}).",
            category="clutter",
            builder=_build_book_stack,
            natural_scale=0.30,
            seed_offset=61 + i * 11,
            builder_kwargs={"p": p, "seed": 200 + i},
        ))

    return variants


def _sample_variant(v: ClutterVariant, n_points: int, seed: int,
                     ) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.RandomState(seed + v.seed_offset)
    builder_kwargs = dict(v.builder_kwargs)
    if v.builder is _build_cable_random_curve:
        builder_kwargs.setdefault("seed", seed + v.seed_offset)
        parts = v.builder(**builder_kwargs)
    elif v.builder in (_build_cable_coiled, _build_extension_cord_with_outlet,
                        _build_garden_hose, _build_rope, _build_headphones):
        builder_kwargs.setdefault("seed", seed + v.seed_offset)
        parts = v.builder(**builder_kwargs)
    else:
        parts = v.builder(**builder_kwargs)
    pts, cols = _sample_coloured(parts, n_points, rng)
    pts_norm, calib = _normalise_local_frame(pts)
    calib["grasp_axis_local"] = list(v.grasp_axis_local)
    return pts_norm, cols, calib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/objects"))
    parser.add_argument("--n-points", type=int, default=8000)
    parser.add_argument("--seed-base", type=int, default=2028)
    parser.add_argument("--scale-jitter", type=float, default=0.20)
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Drop existing entries in the categories we own.
    manifest["objects"] = [o for o in manifest.get("objects", [])
                           if o.get("category") not in
                           ("negative_wire_like", "rope", "clutter")]
    # Wipe stale procedural files for these categories.
    for prefix in ("cable_", "rope_", "keyboard", "monitor", "office_chair",
                    "potted_plant", "extension_cord", "headphones",
                    "garden_hose", "bottle_", "mug_", "book_stack_"):
        for stale in out_dir.glob(f"{prefix}*.npz"):
            stale.unlink()

    variants = _build_clutter_variants()

    new_entries: list[dict] = []
    for i, v in enumerate(variants):
        pts, cols, calib = _sample_variant(v, args.n_points,
                                            args.seed_base + 11 * i)
        out_path = out_dir / f"{v.name}.npz"
        np.savez_compressed(out_path,
                            points=pts.astype(np.float32),
                            colors=cols.astype(np.uint8))
        scale_lo = v.natural_scale * (1.0 - args.scale_jitter)
        scale_hi = v.natural_scale * (1.0 + args.scale_jitter)
        entry = {
            "slug": v.name,
            "file": f"{v.name}.npz",
            "category": v.category,
            "description": v.description,
            "source_url": "procedurally generated (src/build_negative_clutter_objects.py)",
            "license": "internal",
            "source": "procedural",
            "n_points": int(pts.shape[0]),
            "natural_scale_units": v.natural_scale,
            "natural_scale_range": [round(scale_lo, 4), round(scale_hi, 4)],
            "grasp_axis_local": calib["grasp_axis_local"],
            "graspable_on_wire": bool(v.graspable_on_wire),
        }
        new_entries.append(entry)

    manifest["objects"].extend(new_entries)
    manifest["total"] = len(manifest["objects"])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {len(variants)} clutter/cable variants")
    print(f"  manifest: {manifest_path}  total now: {manifest['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
