"""Texture mapping for wire-harness point clouds at REST POSE.

Public API:
    - :func:`load_texture_library` -> ``dict[int, list[np.ndarray]]``
    - :func:`compute_per_point_rgb` -> ``(N, 3) uint8`` BGR array, suitable
      for the ``rasterize_view(..., point_rgb=...)`` arg of ``pcl_to_rgbd``.

Algorithm overview:
    - Wire points (label 0) get a cylindrical UV around their bound skeleton
      edge. A per-segment rotation-minimising frame (Wang 2008 double
      reflection) fixes the radial reference so the texture doesn't twist.
    - All other classes use a cluster-based UV: ``DBSCAN`` groups points of
      a class spatially, then each cluster is parameterised in the plane of
      its two leading principal components.
    - All randomness flows through a single ``np.random.RandomState(seed)``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: BGR fallback colour per class (mirrors ``CLASS_COLORS_BGR`` in
#: ``src/pcl_to_rgbd.py``).  Local copy so this module has no dependency
#: on the rasteriser.
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (180, 180, 180),  # Wire        -> gray
    1: (0, 0, 255),      # Endpoint    -> red  (BGR of RGB (255, 0, 0))
    2: (255, 0, 0),      # Bifurcation -> blue (BGR of RGB (0, 0, 255))
    3: (0, 255, 0),      # Connector   -> green
    4: (0, 255, 255),    # Noise       -> yellow (BGR of RGB (255, 255, 0))
}

#: Folder name on disk for each class index.
_CLASS_FOLDERS: dict[int, str] = {
    0: "wire",
    1: "endpoint",
    2: "bifurcation",
    3: "connector",
    4: "noise",
}

#: DBSCAN ``eps`` per non-wire class.
_CLUSTER_EPS: dict[int, float] = {1: 0.05, 2: 0.10, 3: 0.10, 4: 0.20}

#: Image suffixes accepted as textures.
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Texture loading
# ---------------------------------------------------------------------------

def load_texture_library(
    textures_root: Path = Path("data/textures"),
) -> dict[int, list[np.ndarray]]:
    """Load every texture image grouped by class index.

    Each texture is loaded with ``cv2.imread`` (BGR ``uint8``) at native
    resolution. If a class folder is missing or empty, that entry is an
    empty list -- callers fall back to the flat colour in
    :data:`CLASS_COLORS_BGR`.
    """
    textures_root = Path(textures_root)
    library: dict[int, list[np.ndarray]] = {cls: [] for cls in _CLASS_FOLDERS}

    for cls, folder in _CLASS_FOLDERS.items():
        folder_path = textures_root / folder
        if not folder_path.is_dir():
            log.debug("texture folder missing: %s", folder_path)
            continue
        files = sorted(
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        for fp in files:
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                log.warning("failed to read texture %s", fp)
                continue
            library[cls].append(img)
        log.debug("loaded %d textures for class %d (%s)",
                  len(library[cls]), cls, folder)
    return library


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_texture_uv(
    tex: np.ndarray, u: np.ndarray, v: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample a BGR texture at parametric UV coordinates.

    ``u`` indexes the texture *width* (column / X) axis, ``v`` indexes the
    *height* (row / Y) axis. Both wrap with period 1 thanks to
    ``mode='wrap'``.
    """
    H, W = tex.shape[:2]
    u_mod = np.mod(u, 1.0) * W
    v_mod = np.mod(v, 1.0) * H
    coords = np.stack([v_mod, u_mod], axis=0)  # (2, K) for map_coordinates

    out = np.empty((u.shape[0], 3), dtype=np.float64)
    for c in range(3):
        out[:, c] = map_coordinates(tex[..., c], coords, order=1, mode="wrap")
    return np.clip(out, 0, 255).astype(np.uint8)


def _perpendicular_unit(t: np.ndarray) -> np.ndarray:
    """Unit vector orthogonal to ``t`` (a unit tangent).

    Crosses ``t`` with whichever world axis is least parallel to it; this
    keeps the result numerically stable.
    """
    axes = np.eye(3)
    axis = axes[int(np.argmin(np.abs(axes @ t)))]
    r = np.cross(t, axis)
    n = np.linalg.norm(r)
    if n < 1e-12:
        r = np.cross(t, np.array([0.0, 1.0, 0.0]))
        n = np.linalg.norm(r) + 1e-12
    return r / n


def _rotation_minimising_frame(seg_nodes: np.ndarray) -> np.ndarray:
    """Per-node rotation-minimising frame along a polyline.

    Implements Wang et al. (2008) **double reflection**, second-order
    accurate and numerically robust. ``R[p]`` is perpendicular to the
    *outgoing* tangent of edge ``(p, p+1)``. The last node inherits
    ``R[-2]`` since it has no outgoing edge.
    """
    k1 = seg_nodes.shape[0]
    if k1 < 2:
        return np.tile(np.array([1.0, 0.0, 0.0]), (k1, 1))

    edge_vecs = seg_nodes[1:] - seg_nodes[:-1]
    edge_lens = np.linalg.norm(edge_vecs, axis=1)
    safe_lens = np.where(edge_lens > 1e-12, edge_lens, 1.0)
    edge_tangents = edge_vecs / safe_lens[:, None]

    R = np.empty_like(seg_nodes)
    R[0] = _perpendicular_unit(edge_tangents[0])

    for i in range(edge_tangents.shape[0] - 1):
        x_i, x_ip1 = seg_nodes[i], seg_nodes[i + 1]
        t_i, t_ip1 = edge_tangents[i], edge_tangents[i + 1]
        r_i = R[i]

        # Reflection 1: bisecting plane between x_i and x_{i+1}.
        v1 = x_ip1 - x_i
        c1 = float(np.dot(v1, v1))
        if c1 < 1e-24:
            R[i + 1] = r_i
            continue
        rL = r_i - (2.0 / c1) * float(np.dot(v1, r_i)) * v1
        tL = t_i - (2.0 / c1) * float(np.dot(v1, t_i)) * v1

        # Reflection 2: bisects ``tL`` and ``t_ip1``.
        v2 = t_ip1 - tL
        c2 = float(np.dot(v2, v2))
        r_ip1 = rL if c2 < 1e-24 else (
            rL - (2.0 / c2) * float(np.dot(v2, rL)) * v2
        )

        # Cheap re-orthogonalisation against the new tangent.
        r_ip1 = r_ip1 - float(np.dot(r_ip1, t_ip1)) * t_ip1
        n = np.linalg.norm(r_ip1)
        R[i + 1] = r_ip1 / n if n > 1e-12 else _perpendicular_unit(t_ip1)

    if k1 > edge_tangents.shape[0]:
        R[-1] = R[-2]
    return R


# ---------------------------------------------------------------------------
# Wire texturing helpers
# ---------------------------------------------------------------------------

def _build_segment_lookup(
    segments: list[list[int]],
) -> dict[frozenset, tuple[int, int]]:
    """Map each unordered edge ``frozenset({va, vb})`` to ``(seg_id, p)``.

    ``p`` is the index of the edge inside its segment, i.e. the edge runs
    from ``segments[seg_id][p]`` to ``segments[seg_id][p + 1]``.
    """
    lookup: dict[frozenset, tuple[int, int]] = {}
    for seg_id, seg in enumerate(segments):
        for p in range(len(seg) - 1):
            lookup.setdefault(frozenset((seg[p], seg[p + 1])), (seg_id, p))
    return lookup


def _segment_arclengths(
    segments: list[list[int]], nodes: np.ndarray,
) -> list[np.ndarray]:
    """Cumulative arc-length along each segment (``arc[seg][p]``)."""
    arcs: list[np.ndarray] = []
    for seg in segments:
        if len(seg) < 2:
            arcs.append(np.zeros(len(seg), dtype=np.float64))
            continue
        diffs = np.linalg.norm(np.diff(nodes[seg], axis=0), axis=1)
        arcs.append(np.concatenate([[0.0], np.cumsum(diffs)]))
    return arcs


def _segment_frames(
    segments: list[list[int]], nodes: np.ndarray,
) -> list[np.ndarray]:
    """Per-node rotation-minimising frame ``R`` for every segment."""
    return [
        _rotation_minimising_frame(nodes[seg] if len(seg) > 0 else np.zeros((0, 3)))
        for seg in segments
    ]


def _wire_uv_for_point(
    p: int,
    wb_i: float,
    offset: np.ndarray,
    seg_arc: np.ndarray,
    seg_R: np.ndarray,
    seg_nodes_xyz: np.ndarray,
    n_tile: float,
    radial_scale: float,
) -> tuple[float, float]:
    """Compute (u, v) for one Wire point bound to edge ``(p, p+1)``."""
    total_len = float(seg_arc[-1])
    if total_len <= 1e-12:
        u = 0.0
    else:
        arclen = seg_arc[p] + wb_i * (seg_arc[p + 1] - seg_arc[p])
        u = (arclen / total_len) * n_tile

    edge_vec = seg_nodes_xyz[p + 1] - seg_nodes_xyz[p]
    el = float(np.linalg.norm(edge_vec))
    if el < 1e-12:
        return u, 0.5 * radial_scale
    T = edge_vec / el
    R = seg_R[p]
    R = R - float(np.dot(R, T)) * T
    n = np.linalg.norm(R)
    R = R / n if n > 1e-12 else _perpendicular_unit(T)
    B = np.cross(T, R)

    radial = offset - float(np.dot(offset, T)) * T
    angle = np.arctan2(float(np.dot(radial, B)), float(np.dot(radial, R)))
    v = (angle / (2.0 * np.pi) + 0.5) * radial_scale
    return u, v


def _color_wire_points(
    pcl: np.ndarray,
    labels: np.ndarray,
    nodes: np.ndarray,
    segments: list[list[int]],
    na: np.ndarray,
    nb: np.ndarray,
    wa: np.ndarray,
    wb: np.ndarray,
    offsets: np.ndarray,
    library: list[np.ndarray],
    rng: np.random.RandomState,
    n_tile: float,
    radial_scale: float,
    out_rgb: np.ndarray,
) -> None:
    """Texture all Wire (label 0) points in place."""
    wire_idx = np.where(labels == 0)[0]
    if wire_idx.size == 0:
        return

    if not library:
        out_rgb[wire_idx] = CLASS_COLORS_BGR[0]
        return

    seg_lookup = _build_segment_lookup(segments)
    seg_arc = _segment_arclengths(segments, nodes)
    seg_R = _segment_frames(segments, nodes)
    seg_textures = [library[rng.randint(0, len(library))]
                    for _ in range(len(segments))]
    seg_nodes_xyz = [nodes[seg] for seg in segments]

    # Bucket points by segment so each bucket is sampled with one map_coordinates
    # call -- the only meaningful bottleneck.
    groups: dict[int, list[int]] = {}
    point_p = np.empty(wire_idx.size, dtype=np.int32)
    point_wb = np.empty(wire_idx.size, dtype=np.float64)
    fallback: list[int] = []

    for k, i in enumerate(wire_idx):
        a, b = int(na[i]), int(nb[i])
        info = seg_lookup.get(frozenset((a, b)))
        if info is None:
            fallback.append(int(i))
            continue
        seg_id, p = info
        # ``wb`` is the weight on ``nb`` from the binding step. Map to the
        # weight toward segment node p+1 (which may be ``a`` or ``b``).
        wb_local = float(wb[i]) if b == segments[seg_id][p + 1] else float(wa[i])
        groups.setdefault(seg_id, []).append(k)
        point_p[k] = p
        point_wb[k] = wb_local

    for seg_id, ks in groups.items():
        ks_arr = np.asarray(ks, dtype=np.int64)
        u_vals = np.empty(ks_arr.size, dtype=np.float64)
        v_vals = np.empty(ks_arr.size, dtype=np.float64)
        seg_xyz = seg_nodes_xyz[seg_id]
        sR = seg_R[seg_id]
        sarc = seg_arc[seg_id]
        for j, k in enumerate(ks_arr):
            i = wire_idx[k]
            u_vals[j], v_vals[j] = _wire_uv_for_point(
                p=int(point_p[k]),
                wb_i=float(point_wb[k]),
                offset=offsets[i],
                seg_arc=sarc,
                seg_R=sR,
                seg_nodes_xyz=seg_xyz,
                n_tile=n_tile,
                radial_scale=radial_scale,
            )
        out_rgb[wire_idx[ks_arr]] = _sample_texture_uv(
            seg_textures[seg_id], u_vals, v_vals
        )

    if fallback:
        _color_wire_fallback(
            fallback, segments, offsets, seg_textures, radial_scale,
            int_na=na, int_nb=nb, out_rgb=out_rgb,
        )


def _color_wire_fallback(
    fallback: list[int],
    segments: list[list[int]],
    offsets: np.ndarray,
    seg_textures: list[np.ndarray],
    radial_scale: float,
    int_na: np.ndarray,
    int_nb: np.ndarray,
    out_rgb: np.ndarray,
) -> None:
    """Texture wire points whose bound edge is not on any segment.

    Their u defaults to 0 (segment start) and v is computed in a default
    world frame -- reasonable since this branch is rare and the textures
    repeat anyway.
    """
    node_to_seg: dict[int, int] = {}
    for seg_id, seg in enumerate(segments):
        for v_idx in seg:
            node_to_seg.setdefault(v_idx, seg_id)
    fb = np.asarray(fallback, dtype=np.int64)
    seg_assignment = np.empty(fb.size, dtype=np.int64)
    u_vals = np.zeros(fb.size, dtype=np.float64)
    v_vals = np.empty(fb.size, dtype=np.float64)
    T_default = np.array([0.0, 0.0, 1.0])
    R_default = _perpendicular_unit(T_default)
    B_default = np.cross(T_default, R_default)
    for j, i in enumerate(fb):
        sa = node_to_seg.get(int(int_na[i]))
        sb = node_to_seg.get(int(int_nb[i]))
        seg_assignment[j] = sa if sa is not None else (sb if sb is not None else 0)
        radial = offsets[i] - float(np.dot(offsets[i], T_default)) * T_default
        angle = np.arctan2(
            float(np.dot(radial, B_default)),
            float(np.dot(radial, R_default)),
        )
        v_vals[j] = (angle / (2.0 * np.pi) + 0.5) * radial_scale
    for seg_id in np.unique(seg_assignment):
        mask = seg_assignment == seg_id
        bgr = _sample_texture_uv(seg_textures[int(seg_id)], u_vals[mask], v_vals[mask])
        out_rgb[fb[mask]] = bgr


# ---------------------------------------------------------------------------
# Cluster-based texturing (non-wire classes)
# ---------------------------------------------------------------------------

def _project_pca_uv(cluster_pts: np.ndarray) -> np.ndarray:
    """Project points to ``[0, 1]^2`` using the cluster's top-2 PCA axes."""
    centred = cluster_pts - cluster_pts.mean(axis=0, keepdims=True)
    if centred.shape[0] == 1:
        return np.full((1, 2), 0.5, dtype=np.float64)
    try:
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.full((centred.shape[0], 2), 0.5, dtype=np.float64)
    p1 = Vt[0]
    p2 = Vt[1] if Vt.shape[0] > 1 else _perpendicular_unit(p1)
    coords = np.stack([centred @ p1, centred @ p2], axis=1)
    mn = coords.min(axis=0, keepdims=True)
    mx = coords.max(axis=0, keepdims=True)
    span = np.where(mx - mn > 1e-12, mx - mn, 1.0)
    return (coords - mn) / span


def _color_class_clusters(
    cls: int,
    pcl: np.ndarray,
    labels: np.ndarray,
    library: list[np.ndarray],
    rng: np.random.RandomState,
    out_rgb: np.ndarray,
) -> None:
    """Texture all points of a non-wire class by spatial DBSCAN clusters."""
    cls_idx = np.where(labels == cls)[0]
    if cls_idx.size == 0:
        return
    if not library:
        out_rgb[cls_idx] = CLASS_COLORS_BGR[cls]
        return

    pts = pcl[cls_idx]
    cluster_labels = DBSCAN(eps=_CLUSTER_EPS[cls], min_samples=3).fit_predict(pts)

    for lab in np.unique(cluster_labels):
        in_cluster = cluster_labels == lab
        local_idx = np.where(in_cluster)[0]
        cluster_pts = pts[local_idx]
        if lab == -1:
            # Outliers: each is its own 1-point "cluster".
            tex = library[rng.randint(0, len(library))]
            for li in local_idx:
                gi = int(cls_idx[li])
                # Deterministic UV per global point index via golden-ratio hashing.
                u = (gi * 0.6180339887498949) % 1.0
                v = ((gi + 0.5) * 0.7548776662466927) % 1.0
                out_rgb[gi] = _sample_texture_uv(
                    tex, np.array([u]), np.array([v])
                )[0]
            continue

        tex = library[rng.randint(0, len(library))]
        uvs = _project_pca_uv(cluster_pts)
        out_rgb[cls_idx[local_idx]] = _sample_texture_uv(
            tex, uvs[:, 0], uvs[:, 1]
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_per_point_rgb(
    pcl: np.ndarray,
    labels: np.ndarray,
    nodes: np.ndarray,
    edges: np.ndarray,
    segments: list[list[int]],
    na: np.ndarray,
    nb: np.ndarray,
    wa: np.ndarray,
    wb: np.ndarray,
    offsets: np.ndarray,
    texture_library: dict[int, list[np.ndarray]],
    seed: int = 0,
    n_tile: float = 6.0,
    radial_scale: float = 4.0,
) -> np.ndarray:
    """Compute a per-point BGR colour by mapping textures onto a point cloud.

    Parameters
    ----------
    pcl, labels :
        Rest-pose point cloud and per-point class labels.
    nodes, edges, segments :
        Skeleton geometry. ``segments`` is the per-branch decomposition
        produced by ``_build_topology`` in
        ``src/convert_to_video_dataset.py``.
    na, nb, wa, wb, offsets :
        Per-point binding to the skeleton (see module docstring).
    texture_library :
        Output of :func:`load_texture_library`.
    seed :
        Master RNG seed -- all randomness flows from a single
        ``np.random.RandomState(seed)``.
    n_tile :
        Number of texture repeats *along* a wire segment.
    radial_scale :
        Number of texture repeats *around* a wire segment.

    Returns
    -------
    (N, 3) uint8
        BGR colour per point.
    """
    pcl = np.asarray(pcl)
    labels = np.asarray(labels).astype(int)
    N = pcl.shape[0]

    out_rgb = np.zeros((N, 3), dtype=np.uint8)
    rng = np.random.RandomState(seed)

    # Wire first so the seed-consume order is stable.
    _color_wire_points(
        pcl=pcl, labels=labels, nodes=nodes, segments=segments,
        na=na, nb=nb, wa=wa, wb=wb, offsets=offsets,
        library=texture_library.get(0, []),
        rng=rng, n_tile=n_tile, radial_scale=radial_scale,
        out_rgb=out_rgb,
    )

    for cls in (1, 2, 3, 4):
        _color_class_clusters(
            cls=cls, pcl=pcl, labels=labels,
            library=texture_library.get(cls, []),
            rng=rng, out_rgb=out_rgb,
        )

    # Any unexpected labels stay at the safe mid-gray.
    unknown = ~np.isin(labels, np.array(list(_CLASS_FOLDERS.keys())))
    if np.any(unknown):
        out_rgb[unknown] = (128, 128, 128)

    return out_rgb


# ---------------------------------------------------------------------------
# Background scene generation (Phase 3)
# ---------------------------------------------------------------------------

#: Sentinel label written to background points. ``rasterize_view`` maps any
#: label outside ``{0..4}`` to ``label_img = 0`` so background points carry
#: no foreground class through the rendering pipeline.
BG_LABEL: int = 255


def load_background_library(
    bg_dir: Path = Path("data/textures/backgrounds"),
) -> list[np.ndarray]:
    """Load every background photo (BGR ``uint8``) into a flat list."""
    bg_dir = Path(bg_dir)
    if not bg_dir.is_dir():
        return []
    images: list[np.ndarray] = []
    for p in sorted(bg_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
    return images


def _make_textured_plane(
    centre: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    half_u: float,
    half_v: float,
    texture: np.ndarray,
    rng: np.random.RandomState,
    n_points: int,
    u_tile: float = 3.0,
    v_tile: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample uniform points on a textured rectangle in 3D.

    The rectangle is centred at ``centre`` and spans
    ``[-half_u, half_u] * u_axis + [-half_v, half_v] * v_axis``. Texture UVs
    are tiled by ``u_tile`` / ``v_tile`` to avoid stretching a small photo
    over a large surface.
    """
    if n_points <= 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    us = rng.uniform(-half_u, half_u, n_points)
    vs = rng.uniform(-half_v, half_v, n_points)
    pts = centre[None, :] + us[:, None] * u_axis[None, :] + vs[:, None] * v_axis[None, :]
    u_param = (us / max(2.0 * half_u, 1e-12) + 0.5) * u_tile
    v_param = (vs / max(2.0 * half_v, 1e-12) + 0.5) * v_tile
    rgb = _sample_texture_uv(texture, u_param, v_param)
    return pts, rgb


def load_object_library(
    objects_dir: Path = Path("data/objects"),
) -> list[dict]:
    """Load real-object point clouds + colours from ``data/objects/*.npz``.

    Returns a list of dicts with keys:

    * ``slug``  — object slug (file stem in the manifest)
    * ``points`` — ``(N, 3) float64`` in the object-local frame: centred at
      origin in X and Z, lowest Y at 0, longest XYZ extent normalised to 1.
    * ``colors`` — ``(N, 3) uint8`` BGR.
    * ``natural_scale_range`` — ``(lo, hi)`` allowed scale multiplier in the
      harness world frame (per-object calibration in ``manifest.json``).

    Each entry corresponds to a single CC0 model (see ``manifest.json`` for
    provenance). Returns ``[]`` if the directory or manifest is missing.
    """
    objects_dir = Path(objects_dir)
    manifest_path = objects_dir / "manifest.json"
    if not manifest_path.is_file():
        log.debug("object manifest missing: %s", manifest_path)
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("failed to read %s: %s", manifest_path, exc)
        return []

    library: list[dict] = []
    for entry in manifest.get("objects", []):
        npz_path = objects_dir / entry["file"]
        if not npz_path.is_file():
            log.warning("object .npz missing: %s", npz_path)
            continue
        with np.load(npz_path) as z:
            pts = z["points"].astype(np.float64)
            cols = z["colors"]
        if cols.dtype != np.uint8 or cols.shape != pts.shape:
            log.warning("bad colors in %s: dtype=%s shape=%s",
                        npz_path, cols.dtype, cols.shape)
            continue
        scale_range = entry.get("natural_scale_range")
        if (not isinstance(scale_range, (list, tuple))
                or len(scale_range) != 2):
            scale_range = (0.20, 0.40)
        library.append({
            "slug": entry["slug"],
            "points": pts,
            "colors": cols,
            "natural_scale_range": (float(scale_range[0]),
                                    float(scale_range[1])),
        })
    return library


def _pick_clutter_position(
    rng: np.random.RandomState,
    bbox_min: np.ndarray, bbox_max: np.ndarray,
    floor_extent: float, margin: float,
) -> tuple[float, float]:
    """Pick an (x, z) on the floor outside the harness footprint."""
    for _ in range(200):
        x = float(rng.uniform(-floor_extent, floor_extent))
        z = float(rng.uniform(-floor_extent, floor_extent))
        in_x = bbox_min[0] - margin <= x <= bbox_max[0] + margin
        in_z = bbox_min[2] - margin <= z <= bbox_max[2] + margin
        if not (in_x and in_z):
            return x, z
    # Fallback: corner of the floor.
    sign_x = 1.0 if rng.uniform(0.0, 1.0) > 0.5 else -1.0
    sign_z = 1.0 if rng.uniform(0.0, 1.0) > 0.5 else -1.0
    return sign_x * (floor_extent - 0.2), sign_z * (floor_extent - 0.2)


def _place_object_on_floor(
    obj: dict,
    rng: np.random.RandomState,
    floor_y: float,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    floor_extent: float,
    placed: list[tuple[float, float, float]],
    n_keep: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate, scale, and translate one object onto the floor.

    The object's points already live in a normalised local frame (centred X/Z,
    lowest Y at 0, longest axis = 1). This helper picks a random scale within
    the object's calibrated range, a random Y-axis rotation, and an (x, z)
    position outside the harness footprint that doesn't overlap previously
    placed objects in ``placed`` (list of ``(x, z, half_extent)`` tuples,
    mutated in place).

    ``n_keep`` optionally subsamples the object to fit a global point budget.
    """
    base_pts = obj["points"]
    base_col = obj["colors"]
    if n_keep is not None and n_keep < base_pts.shape[0]:
        idx = rng.choice(base_pts.shape[0], size=n_keep, replace=False)
        base_pts = base_pts[idx]
        base_col = base_col[idx]

    lo, hi = obj["natural_scale_range"]
    scale = float(rng.uniform(lo, hi))
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    pts_local = (base_pts @ R_y.T) * scale  # (N, 3)
    half_extent = float(np.max(np.abs(pts_local[:, [0, 2]])))

    # Position search: stay outside the harness footprint AND non-overlapping
    # with previously-placed objects, while staying close to the camera
    # frustum (FRUSTUM_HALF=1.1 in pcl_to_rgbd; duplicated here to avoid a
    # circular import). Objects placed near the edge may overflow the
    # frustum — that's fine, the rasteriser's frustum-cull just drops the
    # overflow points and the visible portion still reads as a real object
    # peeking into frame.
    cam_frustum_half = 1.1
    placement_extent = max(0.5, min(floor_extent - 0.05,
                                    cam_frustum_half - 0.1))
    margin = max(0.05, half_extent * 0.8 + 0.05)
    chosen = None
    for _ in range(50):
        pos_x, pos_z = _pick_clutter_position(
            rng, bbox_min, bbox_max,
            floor_extent=placement_extent,
            margin=margin,
        )
        ok = True
        for (px, pz, pe) in placed:
            min_sep = half_extent + pe + 0.04
            if (pos_x - px) ** 2 + (pos_z - pz) ** 2 < min_sep * min_sep:
                ok = False
                break
        if ok:
            chosen = (pos_x, pos_z)
            break
    if chosen is None:
        # No non-overlapping spot found in 50 tries — accept the last pick.
        chosen = (pos_x, pos_z)

    pos_x, pos_z = chosen
    placed.append((pos_x, pos_z, half_extent))

    pts_world = pts_local
    pts_world[:, 0] += pos_x
    pts_world[:, 1] += floor_y      # lowest Y of object → floor surface
    pts_world[:, 2] += pos_z
    return pts_world, base_col


def generate_background_scene(
    rng: np.random.RandomState,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    texture_library: list[np.ndarray],
    object_library: list[dict] | None = None,
    n_points: int = 30000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 3D point-cloud background scene around a harness bbox.

    Composition (deterministic for a given ``rng`` state):

    * One textured floor xz-plane just below the harness, sampled from a
      random CC0 photo in ``texture_library``.
    * 3-5 real-object point clouds drawn at random from ``object_library``
      (output of :func:`load_object_library`), each randomly Y-rotated,
      randomly scaled within its calibrated ``natural_scale_range``, and
      positioned on the floor outside the harness footprint with no
      overlap with previously placed objects.

    Parameters
    ----------
    rng :
        Source of randomness; same seed → same scene.
    bbox_min, bbox_max :
        Per-axis (3,) extent of the harness at rest pose. Used to place the
        floor just below the harness and pick object positions outside the
        harness footprint.
    texture_library :
        Flat list of BGR ``uint8`` images for the floor (typically the
        output of :func:`load_background_library`).
    object_library :
        List of dicts as returned by :func:`load_object_library`. When empty
        or ``None``, only the floor is generated and the rest of the budget
        is dropped.
    n_points :
        Soft point budget. Objects use their full per-mesh density when
        feasible; the floor takes the remainder. Total returned point count
        is ≤ ``n_points`` (may be slightly smaller if objects are sparse).

    Returns
    -------
    bg_pcl : (M, 3) float64
    bg_rgb : (M, 3) uint8 BGR
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=np.float64).reshape(3)

    if not texture_library or n_points <= 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    floor_y = float(bbox_min[1] - 0.05)
    # The floor needs to fill the bottom view's image plane (HALF_W=1.467,
    # FRUSTUM_HALF=1.1) and accommodate object placement just inside the
    # depth frustum on side views (~1.05). 1.5 is a safe square half-extent.
    floor_extent = float(max(
        1.5,
        max(bbox_max[0] - bbox_min[0], bbox_max[2] - bbox_min[2]) * 0.5 + 0.6,
    ))

    obj_lib = object_library or []

    # Choose 3-5 objects (with replacement is fine — repeats look like real
    # clutter where a workshop has multiple copies of similar items).
    chosen: list[dict] = []
    if obj_lib:
        n_pick = int(rng.randint(3, 6))  # 3, 4, or 5
        chosen = [obj_lib[rng.randint(0, len(obj_lib))] for _ in range(n_pick)]

    total_obj_available = sum(o["points"].shape[0] for o in chosen)

    # Budget: when objects exist, give them up to 65 % of the budget; the
    # floor gets the rest. A sparse floor is fine because the 2D photographic
    # backdrop fills empty pixels behind it.
    if chosen:
        target_obj_total = min(total_obj_available, int(0.65 * n_points))
        n_floor = max(0, n_points - target_obj_total)
        keep_ratio = (target_obj_total
                      / max(total_obj_available, 1))
    else:
        n_floor = n_points
        keep_ratio = 0.0

    # ── Floor ─────────────────────────────────────────────────────────
    floor_tex = texture_library[rng.randint(0, len(texture_library))]
    floor_pts, floor_rgb = _make_textured_plane(
        centre=np.array([0.0, floor_y, 0.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 0.0, 1.0]),
        half_u=floor_extent,
        half_v=floor_extent,
        texture=floor_tex,
        rng=rng, n_points=n_floor,
        u_tile=4.0, v_tile=4.0,
    )

    # ── Real-object clutter ───────────────────────────────────────────
    obj_pts_list: list[np.ndarray] = []
    obj_rgb_list: list[np.ndarray] = []
    placed: list[tuple[float, float, float]] = []
    for obj in chosen:
        avail = obj["points"].shape[0]
        n_keep = max(64, int(round(avail * keep_ratio))) if avail > 0 else 0
        n_keep = min(n_keep, avail)
        pts, rgb = _place_object_on_floor(
            obj=obj, rng=rng, floor_y=floor_y,
            bbox_min=bbox_min, bbox_max=bbox_max,
            floor_extent=floor_extent,
            placed=placed,
            n_keep=n_keep,
        )
        obj_pts_list.append(pts)
        obj_rgb_list.append(rgb)

    pieces_pts = [floor_pts] + obj_pts_list
    pieces_rgb = [floor_rgb] + obj_rgb_list
    nonempty_pts = [p for p in pieces_pts if p.size]
    nonempty_rgb = [r for r in pieces_rgb if r.size]
    if not nonempty_pts:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    bg_pcl = np.concatenate(nonempty_pts, axis=0)
    bg_rgb = np.concatenate(nonempty_rgb, axis=0)
    return bg_pcl, bg_rgb


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """Smoke test using a real source frame."""
    import sys
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from convert_to_video_dataset import _build_topology, _bind_points

    project_root = here.parent
    pcl = np.load(project_root / "data" / "set2" / "000"
                  / "pointclouds_normed_4096" / "pcl_0000.npy")
    seg = np.load(project_root / "data" / "set2" / "000"
                  / "segmentation_normed_4096" / "seg_0000.npy")
    skel = np.load(project_root / "data" / "set2" / "000"
                   / "skeletons" / "000.npz")
    nodes, adj = skel["nodes"], skel["adj"]
    _, _, edges, segments, _ = _build_topology(adj)
    na, nb, wa, wb, offsets = _bind_points(pcl, nodes, edges)

    library = load_texture_library(project_root / "data" / "textures")
    n_textures = sum(len(v) for v in library.values())
    fallback_only = n_textures == 0

    point_rgb = compute_per_point_rgb(
        pcl=pcl, labels=seg, nodes=nodes, edges=edges, segments=segments,
        na=na, nb=nb, wa=wa, wb=wb, offsets=offsets,
        texture_library=library, seed=0,
    )

    assert point_rgb.shape == (4096, 3), f"shape: {point_rgb.shape}"
    assert point_rgb.dtype == np.uint8, f"dtype: {point_rgb.dtype}"
    assert not np.any(np.isnan(point_rgb.astype(np.float64))), "NaN found"
    assert point_rgb.min() >= 0 and point_rgb.max() <= 255

    for cls in range(5):
        idx = np.where(seg == cls)[0]
        if idx.size == 0:
            continue
        pts_rgb = point_rgb[idx]
        if library.get(cls):
            std = pts_rgb.std(axis=0).sum()
            assert std > 0.0, (
                f"class {cls} has textures but produced uniform colour "
                f"(std={std})"
            )
        else:
            expected = np.array(CLASS_COLORS_BGR[cls], dtype=np.uint8)
            assert np.all(pts_rgb == expected), (
                f"class {cls} fallback should match flat colour {expected!r}"
            )

    msg = (
        "OK (fallback path: no textures yet)"
        if fallback_only
        else f"OK (textured: {n_textures} texture images loaded)"
    )
    print(msg)

    # Background scene smoke test (Phase 4: real objects + floor).
    bg_lib = load_background_library(project_root / "data" / "textures" / "backgrounds")
    obj_lib = load_object_library(project_root / "data" / "objects")
    if bg_lib:
        kw = dict(
            bbox_min=pcl.min(axis=0),
            bbox_max=pcl.max(axis=0),
            texture_library=bg_lib,
            object_library=obj_lib,
            n_points=20000,
        )
        bg_pcl, bg_rgb = generate_background_scene(rng=np.random.RandomState(0), **kw)
        assert bg_pcl.ndim == 2 and bg_pcl.shape[1] == 3, f"bg_pcl shape: {bg_pcl.shape}"
        assert bg_rgb.shape == bg_pcl.shape, f"bg_rgb shape: {bg_rgb.shape}"
        assert bg_rgb.dtype == np.uint8
        # Reproducibility: same seed → identical scene.
        bg_pcl2, bg_rgb2 = generate_background_scene(rng=np.random.RandomState(0), **kw)
        assert np.array_equal(bg_pcl, bg_pcl2), "bg_pcl not reproducible"
        assert np.array_equal(bg_rgb, bg_rgb2), "bg_rgb not reproducible"
        # When the object library is non-empty, lots of points should sit
        # ABOVE the floor surface (real objects have height).
        floor_y = float(pcl[:, 1].min() - 0.05)
        above_floor_frac = float(np.mean(bg_pcl[:, 1] > floor_y + 1e-3))
        msg = (
            f"BG OK: {bg_pcl.shape[0]} pts, "
            f"BGR std={tuple(round(s, 1) for s in bg_rgb.std(axis=0))}, "
            f"objects={'yes' if obj_lib else 'NO'} ({len(obj_lib)} in lib), "
            f"above-floor frac={above_floor_frac:.3f}"
        )
        if obj_lib:
            assert above_floor_frac > 0.05, (
                f"object library is non-empty but only {above_floor_frac:.3%}"
                " of bg points lie above the floor — objects probably aren't"
                " landing where expected."
            )
        print(msg)
    else:
        print("BG SKIP: no background photos available")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
