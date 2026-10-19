"""Mesh to canonical coloured point cloud (build-time only, needs trimesh).

Pipeline for one model file:

1. :func:`load_parts`          load .glb/.gltf/.obj/.stl/.ply, bake scene transforms.
2. :func:`sample_coloured_surface`  area-uniform samples with per-sample colour
   (baseColorTexture via UV barycentric, else baseColorFactor, else vertex/grey).
   trimesh's own ``sample_color=True`` fails on PBR materials, hence our own.
3. :func:`canonicalise`        PCA long axis -> +X, unit length, centred,
   cable end decided and put at -X.
4. :func:`outer_shell_mask`    drop hidden interior samples (full assemblies such
   as the XLR contain internal parts that must never show through gaps).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh

MESH_SUFFIXES = (".glb", ".gltf", ".obj", ".stl", ".ply")
MAX_TEXTURE_SIDE = 1024   # 4096^2 XLR textures are downscaled before sampling

_AXIS_VECS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0),
              "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def load_parts(path: str | Path) -> list[trimesh.Trimesh]:
    """Load a model as a list of meshes with scene-graph transforms applied."""
    scene = trimesh.load(str(path), force="scene")
    parts = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not parts:
        raise ValueError(f"{path}: no triangle meshes found")
    return parts


def _texture_array(material) -> np.ndarray | None:
    """baseColorTexture as an (H, W, 3) uint8 RGB array, downscaled, or None."""
    img = getattr(material, "baseColorTexture", None)
    if img is None:
        img = getattr(material, "image", None)
    if img is None:
        return None
    img = img.convert("RGB")
    if max(img.size) > MAX_TEXTURE_SIDE:
        s = MAX_TEXTURE_SIDE / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))))
    return np.asarray(img)


def _factor_rgb(material) -> np.ndarray:
    f = getattr(material, "baseColorFactor", None)
    if f is None:
        f = getattr(material, "main_color", None)
    return np.array([200, 200, 200] if f is None else np.asarray(f)[:3], dtype=np.float64)


def sample_coloured_surface(parts: Sequence[trimesh.Trimesh], n_total: int,
                            seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-uniform surface samples. Returns (points, RGB uint8, unit normals)."""
    areas = np.array([m.area for m in parts])
    P, C, N = [], [], []
    for j, (m, a) in enumerate(zip(parts, areas)):
        k = max(200, int(n_total * a / areas.sum()))
        p, fi = trimesh.sample.sample_surface(m, k, seed=seed * 100003 + j)[:2]
        mat = getattr(m.visual, "material", None)
        tex = _texture_array(mat) if mat is not None else None
        uv = getattr(m.visual, "uv", None)
        fac = _factor_rgb(mat) if mat is not None else np.array([200.0] * 3)
        if tex is not None and uv is not None:
            bary = trimesh.triangles.points_to_barycentric(m.triangles[fi], p)
            q = (uv[m.faces[fi]] * bary[:, :, None]).sum(1)
            h, w = tex.shape[:2]
            c = tex[((1 - q[:, 1]) % 1.0 * (h - 1)).astype(int),
                    (q[:, 0] % 1.0 * (w - 1)).astype(int)].astype(np.float64)
            c = c * (fac / 255.0)        # glTF: texture x factor
        elif m.visual.kind == "vertex":
            bary = trimesh.triangles.points_to_barycentric(m.triangles[fi], p)
            vc = np.asarray(m.visual.vertex_colors)[:, :3].astype(np.float64)
            c = (vc[m.faces[fi]] * bary[:, :, None]).sum(1)
        elif m.visual.kind == "face":
            c = np.asarray(m.visual.face_colors)[fi, :3].astype(np.float64)
        else:
            c = np.tile(fac, (len(p), 1))
        P.append(p); C.append(np.clip(c, 0, 255)); N.append(m.face_normals[fi])
    return (np.vstack(P).astype(np.float64), np.vstack(C).astype(np.uint8),
            np.vstack(N).astype(np.float64))


def _cross_section_radius(pts: np.ndarray, lo: float, hi: float) -> float:
    """90th-percentile radial distance of points with x in [lo, hi]."""
    sel = pts[(pts[:, 0] >= lo) & (pts[:, 0] <= hi)]
    if len(sel) < 10:
        return 0.0
    return float(np.percentile(np.hypot(sel[:, 1], sel[:, 2]), 90))


def canonicalise(points: np.ndarray, normals: np.ndarray, cable_end: str = "auto",
                 end_frac: float = 0.25):
    """Rotate/scale/centre into the canonical frame.

    Cable-end decision (documented heuristic): the two end slabs (outer
    ``end_frac`` of the length) are compared by 90th-percentile radial extent;
    the NARROWER end is the cable end (strain-relief boot, or the plain cable
    stub). A manifest ``cable_end`` of ``+x/-x/+y/-y/+z/-z`` (a direction in the
    SOURCE model's own coordinates pointing toward the cable) overrides it.
    Returns ``(points, normals, info)``.
    """
    ctr = points.mean(0)
    _, _, vt = np.linalg.svd(points - ctr, full_matrices=False)
    R = vt.copy()                              # rows: long, wide, thin axes
    if np.linalg.det(R) < 0:
        R[2] *= -1
    p = (points - ctr) @ R.T
    n = normals @ R.T
    # decide cable end BEFORE choosing the sign of X
    lo, hi = np.percentile(p[:, 0], [0.1, 99.9])
    L = hi - lo
    r_neg = _cross_section_radius(p, lo, lo + end_frac * L)
    r_pos = _cross_section_radius(p, hi - end_frac * L, hi)
    if cable_end != "auto":
        d = np.array(_AXIS_VECS[cable_end], float)
        cable_sign = -1.0 if float(R[0] @ d) > 0 else 1.0   # cable toward -X
        method = f"manifest:{cable_end}"
    else:
        cable_sign = 1.0 if r_neg <= r_pos else -1.0   # +1: cable already at -X
        method = "auto:narrower_end"
    if cable_sign < 0:                          # flip X (and Y to stay right-handed)
        p[:, 0] *= -1; p[:, 1] *= -1; n[:, 0] *= -1; n[:, 1] *= -1
        lo, hi = -hi, -lo
        r_neg, r_pos = r_pos, r_neg
    p[:, 0] -= (lo + hi) / 2
    # centre Y/Z on the bounding box too, then scale so long extent = 1
    for ax in (1, 2):
        a, b = np.percentile(p[:, ax], [0.1, 99.9])
        p[:, ax] -= (a + b) / 2
    s = 1.0 / L
    p *= s
    ext = np.percentile(p, 99, axis=0) - np.percentile(p, 1, axis=0)
    info = {"cable_end_method": method, "length_source_units": float(L),
            "end_radius_cable": float(r_neg * s), "end_radius_mating": float(r_pos * s),
            "extent_xyz": [float(x) for x in ext], "width": float(max(ext[1], ext[2]))}
    return p, n, info


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def outer_shell_mask(points: np.ndarray, n_views: int = 32, grid: int = 192,
                     tol_cells: float = 2.0) -> np.ndarray:
    """True for samples visible from at least one of ``n_views`` directions.

    Orthographic depth test on the point set itself: per view, points are
    binned into a ``grid`` x ``grid`` image, the nearest depth per bin is kept,
    and a point survives if it lies within ``tol_cells`` bin pitches of it.
    Interior parts hidden behind the shell from EVERY direction are dropped,
    while interiors seen through real openings (pin sockets) are kept.
    """
    half = float(np.abs(points).max()) * 1.02
    pitch = 2 * half / grid
    vis = np.zeros(len(points), bool)
    for d in fibonacci_sphere(n_views):
        ref = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
        u = np.cross(d, ref); u /= np.linalg.norm(u)
        v = np.cross(d, u)
        ix = np.clip(((points @ u + half) / pitch).astype(int), 0, grid - 1)
        iy = np.clip(((points @ v + half) / pitch).astype(int), 0, grid - 1)
        cell = ix * grid + iy
        depth = -(points @ d)                  # viewer at +d: smaller = nearer
        zmin = np.full(grid * grid, np.inf)
        np.minimum.at(zmin, cell, depth)
        vis |= depth <= zmin[cell] + tol_cells * pitch
    return vis
