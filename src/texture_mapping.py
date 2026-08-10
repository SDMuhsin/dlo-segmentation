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
import os
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
    ext_library: list[np.ndarray] | None = None,
    ext_rng: np.random.RandomState | None = None,
) -> None:
    """Texture all Wire (label 0) points in place."""
    wire_idx = np.where(labels == 0)[0]
    if wire_idx.size == 0:
        return

    if not library and not ext_library:
        out_rgb[wire_idx] = CLASS_COLORS_BGR[0]
        return

    seg_lookup = _build_segment_lookup(segments)
    seg_arc = _segment_arclengths(segments, nodes)
    seg_R = _segment_frames(segments, nodes)
    seg_textures = ([library[rng.randint(0, len(library))]
                     for _ in range(len(segments))]
                    if library else [])
    # Phase 18 wire-pool extension: redraw each segment's texture uniformly
    # over (original ∪ extension) from the extension's OWN rng. The original
    # per-segment draws above still happen (and are discarded) so the main
    # rng stream consumed afterwards by the non-wire classes is identical
    # whether the lever is on or off.
    if ext_library and ext_rng is not None:
        combined = list(library) + list(ext_library)
        seg_textures = [combined[ext_rng.randint(0, len(combined))]
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
    wire_ext_library: list[np.ndarray] | None = None,
    wire_ext_seed: int = 0,
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
    wire_ext_library, wire_ext_seed :
        Phase 18 lever: extra wire-only textures appended to the class-0
        pool. When non-empty, each wire segment's texture is drawn uniformly
        from (class-0 pool ∪ extension) using a SEPARATE
        ``RandomState(wire_ext_seed)`` so the master stream is consumed
        exactly as before (lever OFF ⇒ identical output).

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

    # ── KIAT_DLO_UNICOLOR "de-cheat" lever (default OFF) ────────────────────
    # The per-class colour cheat: connectors/endpoints/bifurcations (source
    # classes 3/1/2) are normally textured from their OWN pools (or fall back
    # to green/red/blue), so they are trivially separable from the wire body by
    # COLOUR alone. When this lever is enabled, the WHOLE DLO — wire body AND
    # endpoint/bifurcation/connector (source classes {0,1,2,3}) — is coloured by
    # the SAME wire-texturing mechanism off the SAME wire texture pool, so those
    # sub-parts are visually indistinguishable from the wire body (they carry
    # the adjacent segment's wire texture, NOT a flat colour). Only the Label
    # PNG (untouched here) then distinguishes wire vs connector. Source class 4
    # (noise → background) is left coloured exactly as before.
    #
    # The entire effect is confined to this branch: when the flag is OFF,
    # ``wire_labels is labels`` and ``nonwire_classes == (1, 2, 3, 4)``, so the
    # code path — and therefore the output — is identical to before.
    wire_pool = texture_library.get(0, [])
    if os.environ.get("KIAT_DLO_UNICOLOR", "0").strip() in ("1", "true", "True"):
        wire_labels = labels.copy()
        wire_labels[np.isin(labels, (1, 2, 3))] = 0   # fold sub-parts into wire
        nonwire_classes: tuple[int, ...] = (4,)       # only noise keeps its own
        # ONE wire texture for the WHOLE DLO so body + endpoint/bifurcation/
        # connector share a single continuous wire appearance (a single-element
        # library makes every segment's per-segment draw collapse to index 0).
        # When the pool is empty the list stays empty and _color_wire_points
        # falls back to the wire colour, exactly as on the OFF path.
        wire_library = [wire_pool[rng.randint(0, len(wire_pool))]] if wire_pool else wire_pool
        # The de-cheat is mutually exclusive with the Phase-18 wire-pool lever
        # (which would re-introduce per-segment variety), so ignore the ext pool.
        unicolor_ext_library = None
    else:
        wire_labels = labels
        nonwire_classes = (1, 2, 3, 4)
        wire_library = wire_pool
        unicolor_ext_library = wire_ext_library

    # Wire first so the seed-consume order is stable.
    _color_wire_points(
        pcl=pcl, labels=wire_labels, nodes=nodes, segments=segments,
        na=na, nb=nb, wa=wa, wb=wb, offsets=offsets,
        library=wire_library,
        rng=rng, n_tile=n_tile, radial_scale=radial_scale,
        out_rgb=out_rgb,
        ext_library=unicolor_ext_library,
        ext_rng=(np.random.RandomState(wire_ext_seed)
                 if unicolor_ext_library else None),
    )

    for cls in nonwire_classes:
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
    * ``category`` — manifest category (e.g. ``"hand"``, ``"gripper"``,
      ``"arm"``, ``"negative_wire_like"``, ``"clutter"``, original Phase 4
      categories like ``"tool"``/``"container"``/etc.). Defaults to
      ``"unknown"`` if the manifest entry omits it.
    * ``grasp_axis_local`` — optional ``(3,) float64`` unit vector in the
      object's local frame indicating the line a wire would lay across when
      gripped. ``None`` if not provided.
    * ``graspable_on_wire`` — bool. Marks objects that the foreground placer
      can pose specifically grasping a wire skeleton point.

    Each entry corresponds to a single mesh/procedural model (see
    ``manifest.json`` for provenance). Returns ``[]`` if the directory or
    manifest is missing.
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
        grasp_axis_raw = entry.get("grasp_axis_local")
        grasp_axis = None
        if isinstance(grasp_axis_raw, (list, tuple)) and len(grasp_axis_raw) == 3:
            v = np.asarray(grasp_axis_raw, dtype=np.float64)
            n = float(np.linalg.norm(v))
            if n > 1e-9:
                grasp_axis = v / n
        library.append({
            "slug": entry["slug"],
            "points": pts,
            "colors": cols,
            "natural_scale_range": (float(scale_range[0]),
                                    float(scale_range[1])),
            "category": entry.get("category", "unknown"),
            "grasp_axis_local": grasp_axis,
            "graspable_on_wire": bool(entry.get("graspable_on_wire", False)),
            # Pose roster for hands (e.g. "flat_palm_down", "spread_fan",
            # "open_palm_up", "fist", ...); None for non-hand objects. Used by
            # the Phase-16 open-hand lever to pick OPEN/SPLAYED poses. Purely
            # additive metadata — never consulted on the lever-OFF path.
            "pose_family": entry.get("pose_family"),
        })
    return library


def filter_library_by_category(
    library: list[dict],
    categories: set[str] | None = None,
    exclude: set[str] | None = None,
) -> list[dict]:
    """Return a sub-list filtered to ``categories`` (or excluding ``exclude``).

    If ``categories`` is provided, only entries whose ``category`` is in that
    set are kept. Otherwise all entries are kept. ``exclude`` runs after the
    inclusion filter and removes any entry whose category is in the set.
    """
    out = library
    if categories is not None:
        out = [o for o in out if o.get("category") in categories]
    if exclude:
        out = [o for o in out if o.get("category") not in exclude]
    return out


# ---------------------------------------------------------------------------
# Phase 14: targeted hard-negative "confusers" (dark cylinders / sharp edges /
# hands) placed as labeled-BACKGROUND floor clutter.
#
# Motivation (real-world inspection of the Phase 7 / Phase 13-lighting model):
# it segments wires well but ALSO fires on (a) black cylindrical structures
# (a black stand), (b) black sharp edges, and (c) hands. The strict Phase 4
# clutter is the 21 mesh reals — whose cylinders/boxes are BRIGHT (thermos
# meanV~107, pot~139, vase~194) and which contain NO hands — so "black +
# elongated/edge" stayed a wire-only cue. This lever shows the model the exact
# confusers as background: dark/black cylinders, darkened sharp-edged boxes,
# and hands. All colour is baked per-point in BGR BEFORE rasterisation (§0.5),
# never a screen-space pass.
#
# Deliberately EXCLUDES thin ``negative_wire_like`` / ``rope`` — those are too
# close to the wire and collapsed recall in v2 (§0.13). The confusers here are
# THICK (cylinders) or BOXY (edges), unambiguously not-wire by shape, so the
# model learns a finer wire cue (thinness + continuity) rather than a tighter
# one.
# ---------------------------------------------------------------------------
_CONFUSER_CYLINDRICAL: tuple[str, ...] = (
    "plastic_thermos", "metal_jerrycan", "plastic_jerrycan", "can_rusted",
    "brass_pot_01", "ceramic_vase_02", "brass_blowtorch", "Lantern_01",
    "desk_lamp_arm_01",
)
_CONFUSER_EDGED: tuple[str, ...] = (
    "cardboard_box_01", "metal_toolbox", "wooden_crate_02", "Television_01",
    "Camera_01", "cassette_player", "CashRegister_01",
)
# Category that survives ``generate_background_scene``'s foreground-only filter
# (so re-tagged hands place as floor clutter, not foreground occluders).
_NEG_CLUTTER_CATEGORY = "container"


def _darken_object_copy(obj: dict, factor: float, slug_prefix: str) -> dict:
    """Copy ``obj`` with per-point colours scaled toward black (``factor`` in
    [0,1]) and a background-safe category, so it renders as a dark/black
    version of the same shape placed on the floor as labeled background."""
    out = dict(obj)
    cols = np.asarray(obj["colors"])
    out["colors"] = np.clip(cols.astype(np.float32) * float(factor),
                            0.0, 255.0).astype(cols.dtype)
    out["slug"] = f"{slug_prefix}{obj.get('slug', 'obj')}"
    out["category"] = _NEG_CLUTTER_CATEGORY
    return out


def build_confuser_negatives(
    library: list[dict],
    rng: np.random.RandomState,
    n_cyl: int = 6,
    n_edge: int = 3,
    n_hand: int = 3,
    dark_factor: float = 0.16,
    edge_dark_frac: float = 0.6,
) -> list[dict]:
    """Return a pool of dark/black confuser clutter dicts to APPEND to the
    strict-Phase-4 background clutter library.

    Each dict has the same schema as :func:`load_object_library` entries
    (``points`` / ``colors`` / scale fields) with a background-safe
    ``category`` so :func:`generate_background_scene` places it on the floor as
    labeled-BACKGROUND clutter. Cylinders are always darkened (the "black
    stand" confuser); a fraction ``edge_dark_frac`` of edged objects are
    darkened (the "black sharp edge" confuser); hands keep their natural colour
    (skin) and are re-tagged so they survive the background-clutter filter.

    A SEPARATE ``rng`` should be passed (as for the Phase 13 levers) so the
    base scene-composition stream is unchanged when the lever is OFF.
    """
    by = {o.get("slug"): o for o in (library or [])}
    out: list[dict] = []

    cyl = [by[s] for s in _CONFUSER_CYLINDRICAL if s in by]
    for _ in range(max(0, int(n_cyl))):
        if not cyl:
            break
        out.append(_darken_object_copy(
            cyl[rng.randint(0, len(cyl))], dark_factor, "darkcyl_"))

    edg = [by[s] for s in _CONFUSER_EDGED if s in by]
    for _ in range(max(0, int(n_edge))):
        if not edg:
            break
        o = edg[rng.randint(0, len(edg))]
        if rng.uniform(0.0, 1.0) < edge_dark_frac:
            out.append(_darken_object_copy(o, dark_factor, "darkedge_"))
        else:
            c = dict(o)
            c["slug"] = f"edge_{o.get('slug', 'obj')}"
            c["category"] = _NEG_CLUTTER_CATEGORY
            out.append(c)

    hands = [o for o in (library or []) if o.get("category") == "hand"]
    for _ in range(max(0, int(n_hand))):
        if not hands:
            break
        h = dict(hands[rng.randint(0, len(hands))])
        h["slug"] = f"neghand_{h.get('slug', 'hand')}"
        h["category"] = _NEG_CLUTTER_CATEGORY  # survive the bg-clutter filter
        out.append(h)

    return out


def densify_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    target_n: int,
    rng: np.random.RandomState,
    jitter_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Up-sample a surface point cloud to ``target_n`` points.

    Replicates points with small surface-local Gaussian jitter (std ≈
    ``jitter_frac`` × the estimated inter-point spacing) so the shape is
    preserved but the splat cloud is denser — a scaled-up object then renders
    solid / high-resolution instead of sparse. Returns the input unchanged when
    ``target_n`` <= the current count. Asset meshes here cap at 8000 points, so
    this is the only way to make a large near-wire hand read crisply.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if target_n <= n or n == 0:
        return points, colors
    extent = float(np.mean(pts.max(axis=0) - pts.min(axis=0)))
    spacing = extent / max(n ** (1.0 / 3.0), 1.0)
    std = float(jitter_frac) * spacing
    reps = int(np.ceil(target_n / n))
    out_pts = [pts]
    out_col = [np.asarray(colors)]
    for _ in range(reps - 1):
        out_pts.append(pts + rng.normal(0.0, std, size=pts.shape))
        out_col.append(np.asarray(colors))
    P = np.concatenate(out_pts, axis=0)[:target_n]
    C = np.concatenate(out_col, axis=0)[:target_n]
    return P, C


# ---------------------------------------------------------------------------
# Phase 8: rotation helpers for foreground placement
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix from a 3-vector (Rodrigues helper)."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation around a unit ``axis`` by ``angle`` (radians)."""
    axis = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.eye(3)
    a = axis / n
    K = _skew(a)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _rotation_align(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Rotation matrix that takes unit ``v_from`` to unit ``v_to``."""
    v_from = np.asarray(v_from, dtype=np.float64)
    v_to = np.asarray(v_to, dtype=np.float64)
    n_from = np.linalg.norm(v_from)
    n_to = np.linalg.norm(v_to)
    if n_from < 1e-12 or n_to < 1e-12:
        return np.eye(3)
    f = v_from / n_from
    t = v_to / n_to
    c = float(np.dot(f, t))
    if c > 1.0 - 1e-9:
        return np.eye(3)
    if c < -1.0 + 1e-9:
        # Antiparallel — pick any axis perpendicular to f.
        ortho = _perpendicular_unit(f)
        return _rotation_about_axis(ortho, np.pi)
    axis = np.cross(f, t)
    s = float(np.linalg.norm(axis))
    if s < 1e-12:
        return np.eye(3)
    axis_n = axis / s
    K = _skew(axis_n)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


# ---------------------------------------------------------------------------
# Phase 8: wire grasp picker
# ---------------------------------------------------------------------------

def _pick_wire_grasp_point(
    rng: np.random.RandomState,
    nodes: np.ndarray,
    segments: list[list[int]],
    interior_frac: tuple[float, float] = (0.2, 0.8),
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pick a random point along a wire segment and return ``(point, tangent)``.

    Iterates segments preferring longer ones (sampled proportional to total
    arc length), picks a random interior edge, then a random ``t ∈
    interior_frac`` along that edge so the hand isn't stuck at endpoints.

    Returns ``None`` if there are no usable segments (degenerate skeleton).
    """
    if not segments:
        return None
    # Build arc-length-weighted segment selection.
    seg_lengths = []
    for seg in segments:
        if len(seg) < 2:
            seg_lengths.append(0.0)
            continue
        diffs = np.linalg.norm(np.diff(nodes[seg], axis=0), axis=1)
        seg_lengths.append(float(np.sum(diffs)))
    total = float(np.sum(seg_lengths))
    if total <= 1e-9:
        return None
    weights = np.array(seg_lengths) / total
    # Discrete sample by weight.
    r = float(rng.uniform(0.0, 1.0))
    cum = 0.0
    seg_id = 0
    for i, w in enumerate(weights):
        cum += float(w)
        if r <= cum:
            seg_id = i
            break
    seg = segments[seg_id]
    if len(seg) < 2:
        return None
    # Pick edge: weighted by edge length within the segment.
    edges = np.diff(nodes[seg], axis=0)
    elens = np.linalg.norm(edges, axis=1)
    if float(np.sum(elens)) < 1e-9:
        return None
    eweights = elens / float(np.sum(elens))
    r2 = float(rng.uniform(0.0, 1.0))
    cum = 0.0
    edge_id = 0
    for i, w in enumerate(eweights):
        cum += float(w)
        if r2 <= cum:
            edge_id = i
            break
    a = seg[edge_id]
    b = seg[edge_id + 1]
    t = float(rng.uniform(interior_frac[0], interior_frac[1]))
    p = nodes[a] * (1.0 - t) + nodes[b] * t
    tangent = nodes[b] - nodes[a]
    n = float(np.linalg.norm(tangent))
    if n < 1e-12:
        return None
    return p.astype(np.float64), (tangent / n).astype(np.float64)


# ---------------------------------------------------------------------------
# Phase 8: foreground placement (hand-on-wire, gripper-on-wire, free-floating)
# ---------------------------------------------------------------------------

def _place_hand_on_wire(
    obj: dict,
    rng: np.random.RandomState,
    skeleton_nodes: np.ndarray,
    segments: list[list[int]],
    n_keep: int | None = None,
    extra_offset_max: float = 0.020,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pose a graspable object (hand/gripper) so it grips a random wire point.

    Returns ``None`` if the wire grasp picker fails (e.g. degenerate skeleton).

    Procedure
    ---------
    1. Pick wire grasp point ``P`` and tangent ``T`` (random).
    2. Apply per-object scale within ``natural_scale_range``.
    3. Rotate so the object's ``grasp_axis_local`` aligns with ``T``.
    4. Add a free random rotation about ``T`` (orientation around the wire).
    5. Translate so the rotated centroid lands at ``P + small_offset``.
       The small offset (uniform in a sphere of radius ``extra_offset_max``)
       prevents perfect alignment so the hand reads as "grasping near" the
       wire, not surgically tangent to it.
    """
    pick = _pick_wire_grasp_point(rng, skeleton_nodes, segments)
    if pick is None:
        return None
    P, T = pick

    base_pts = obj["points"].astype(np.float64).copy()
    base_col = obj["colors"]
    if n_keep is not None and n_keep < base_pts.shape[0]:
        idx = rng.choice(base_pts.shape[0], size=n_keep, replace=False)
        base_pts = base_pts[idx]
        base_col = base_col[idx]

    lo, hi = obj["natural_scale_range"]
    scale = float(rng.uniform(lo, hi))
    pts = base_pts * scale

    grasp_axis = obj.get("grasp_axis_local")
    if grasp_axis is None or np.allclose(grasp_axis, 0.0):
        grasp_axis = np.array([1.0, 0.0, 0.0])
    grasp_axis = np.asarray(grasp_axis, dtype=np.float64)

    R_align = _rotation_align(grasp_axis, T)
    extra_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    R_about = _rotation_about_axis(T, extra_angle)

    pts_rot = pts @ R_align.T @ R_about.T

    # Centroid of the rotated point cloud — the wire grasp point should pass
    # through here.
    grasp_centre = pts_rot.mean(axis=0)

    # Tiny jitter so the hand isn't surgically perfect.
    jitter = rng.normal(0.0, extra_offset_max / 3.0, size=3)
    jitter_norm = float(np.linalg.norm(jitter))
    if jitter_norm > extra_offset_max:
        jitter = jitter * (extra_offset_max / jitter_norm)

    pts_world = pts_rot - grasp_centre[None, :] + (P + jitter)[None, :]
    return pts_world, base_col


# ---------------------------------------------------------------------------
# Phase 16: OPEN / SPLAYED hand placed as labeled-BACKGROUND.
#
# Motivation (real-world inspection of the best Phase-15 model): it still paints
# GREEN (predicted wire) on an OPEN / SPLAYED hand — a palm facing the camera
# with the fingers spread. Phase 14 added hands GRIPPING the wire as negatives
# (helped grasping hands), but open hands away-from-grip are under-represented,
# so spread fingers still read as wire. This lever drops ONE open hand into the
# scene as labeled BACKGROUND so the model learns "open hand = not wire".
#
# Pose families whose normalised local frame already presents the palm broadside
# along local +Z (the thin axis ≈ [0,0,1]; cf. build_hand_objects: these three
# carry a wrist_tilt that lays the flat hand in the local XY plane). Restricting
# to these lets us align the broad face to the camera with a near-identity
# rotation, so the splayed fingers stay un-foreshortened. ``flat_palm_down`` is
# the clearest splay; all three render as a recognisable open hand broadside.
# ---------------------------------------------------------------------------
_OPEN_HAND_POSE_FAMILIES: tuple[str, ...] = (
    "flat_palm_down", "spread_fan", "open_palm_up",
)


def _place_open_hand(
    obj: dict,
    rng: np.random.RandomState,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    pad: float = 0.10,
    tilt_deg: float = 22.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose an OPEN / SPLAYED hand broadside to the front/back camera.

    The hand assets in ``_OPEN_HAND_POSE_FAMILIES`` have their palm normal along
    local +Z (the flattest PCA axis). We:

    1. Scale per ``natural_scale_range`` (the caller is expected to have already
       widened that range, as for the Phase-14 grip hands, so the open hand
       reads large on screen).
    2. Rotate the palm normal (local +Z) onto a target axis near world ±Z — i.e.
       toward the front (``-Z``) or back (``+Z``) camera, chosen at random — so
       the broad open palm/back faces a canonical view broadside and the spread
       fingers are NOT foreshortened. A small random tilt (``tilt_deg``) plus a
       free in-plane spin about the palm normal add pose variety without ever
       turning the hand edge-on.
    3. Translate the centroid to a random point inside the (slightly padded)
       harness bbox, biased in Z toward the wire plane (the harness is thin in Z)
       so the hand sits in the scene across views. It may overlap the wire but is
       an OPEN hand, never wrapped around it.

    Returns ``(points_world (N,3) float64, colors (N,3) uint8)``. The caller
    appends these to the foreground PCL, where they are labeled BG_LABEL.
    """
    base_pts = obj["points"].astype(np.float64).copy()
    base_col = obj["colors"]

    lo, hi = obj["natural_scale_range"]
    scale = float(rng.uniform(lo, hi))
    pts = base_pts * scale

    # Local palm normal ≈ +Z for these poses. Target: world ±Z (toward the
    # front/back camera) so the open palm is broadside; small tilt for realism.
    facing = +1.0 if rng.uniform(0.0, 1.0) < 0.5 else -1.0
    target = np.array([0.0, 0.0, facing], dtype=np.float64)
    R_face = _rotation_align(np.array([0.0, 0.0, 1.0]), target)

    # Free spin about the palm normal keeps the hand broadside but rotates the
    # splay in-plane (wrist up / sideways / down).
    spin = float(rng.uniform(0.0, 2.0 * np.pi))
    R_spin = _rotation_about_axis(target, spin)
    # Small off-broadside tilt so it isn't perfectly fronto-parallel.
    tx = float(rng.uniform(-np.deg2rad(tilt_deg), np.deg2rad(tilt_deg)))
    ty = float(rng.uniform(-np.deg2rad(tilt_deg), np.deg2rad(tilt_deg)))
    R_tilt = (_rotation_about_axis(np.array([1.0, 0.0, 0.0]), tx)
              @ _rotation_about_axis(np.array([0.0, 1.0, 0.0]), ty))

    pts_rot = pts @ R_face.T @ R_spin.T @ R_tilt.T

    # Place the centroid inside the harness bbox. The harness is essentially
    # planar in Z (extent ~0.16 wu), so keep the hand's centroid near that plane
    # (within ±half-Z-extent + a small pad) so it lands in the scene, not far in
    # front of / behind the wire.
    z_mid = 0.5 * (bbox_min[2] + bbox_max[2])
    z_half = 0.5 * (bbox_max[2] - bbox_min[2]) + 0.05
    pos = np.array([
        rng.uniform(bbox_min[0] - pad, bbox_max[0] + pad),
        rng.uniform(bbox_min[1] - pad, bbox_max[1] + pad),
        z_mid + rng.uniform(-z_half, z_half),
    ])
    centroid = pts_rot.mean(axis=0)
    pts_world = pts_rot - centroid[None, :] + pos[None, :]
    return pts_world, base_col


def _place_object_in_foreground(
    obj: dict,
    rng: np.random.RandomState,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    n_keep: int | None = None,
    pad: float = 0.20,
    tilt_deg: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Place an object randomly in/near the harness bbox with random pose.

    Foreground here is intentionally not view-dependent: the object is just
    placed somewhere in the harness's spatial extent so that, depending on
    view, it may occlude part of the harness, sit beside it, or peek into
    frame from an edge.

    The position is sampled from ``[bbox_min - pad, bbox_max + pad]`` along
    each axis. ``tilt_deg`` bounds the random pitch / roll added to the
    Y-axis spin (so the object isn't always axis-aligned).
    """
    base_pts = obj["points"].astype(np.float64).copy()
    base_col = obj["colors"]
    if n_keep is not None and n_keep < base_pts.shape[0]:
        idx = rng.choice(base_pts.shape[0], size=n_keep, replace=False)
        base_pts = base_pts[idx]
        base_col = base_col[idx]

    lo, hi = obj["natural_scale_range"]
    scale = float(rng.uniform(lo, hi))
    pts = base_pts * scale

    theta_y = float(rng.uniform(0.0, 2.0 * np.pi))
    theta_x = float(rng.uniform(-np.deg2rad(tilt_deg), np.deg2rad(tilt_deg)))
    theta_z = float(rng.uniform(-np.deg2rad(tilt_deg), np.deg2rad(tilt_deg)))
    R = (_rotation_about_axis(np.array([0.0, 1.0, 0.0]), theta_y)
         @ _rotation_about_axis(np.array([1.0, 0.0, 0.0]), theta_x)
         @ _rotation_about_axis(np.array([0.0, 0.0, 1.0]), theta_z))
    pts_rot = pts @ R.T

    pos = np.array([
        rng.uniform(bbox_min[0] - pad, bbox_max[0] + pad),
        rng.uniform(bbox_min[1] - pad, bbox_max[1] + pad),
        rng.uniform(bbox_min[2] - pad, bbox_max[2] + pad),
    ])
    centroid = pts_rot.mean(axis=0)
    pts_world = pts_rot - centroid[None, :] + pos[None, :]
    return pts_world, base_col


def generate_foreground_scene(
    rng: np.random.RandomState,
    object_library: list[dict],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    skeleton_nodes: np.ndarray,
    segments: list[list[int]],
    n_points: int = 24000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate foreground (occluding/near-harness) objects for one source.

    Phase 8 v2: deterministic high-density placement, target 10-15 fg objects
    per source (up from v1's 0-4). Composition mixes:

    * **2-4 grasping objects** (hand or gripper gripping a random wire point).
      Biased toward hands (70%) so skin is impossible-to-miss.
    * **2-4 free-floating hands** (not grasping the wire). v2 spec requires
      skin presence in EVERY source.
    * **3-6 free-floating cables** (negative_wire_like). The wire-shaped
      negatives the model has never seen.
    * **1-3 free-floating grippers / arms / ropes** (other clutter).

    Total typical: 8-17 objects, mean ~12. Ensures the user-required 10-15
    foreground objects per sample.

    Parameters
    ----------
    rng :
        Source of randomness; same state → same scene.
    object_library :
        Output of :func:`load_object_library`. Empty → no foreground.
    bbox_min, bbox_max :
        Per-axis (3,) extent of the harness at rest pose.
    skeleton_nodes, segments :
        Wire skeleton at rest pose (or any chosen anim frame); used to pick
        the grasp point + tangent for hand/gripper-on-wire poses.
    n_points :
        Soft total point budget for foreground (split across placed objects).

    Returns
    -------
    fg_pcl : (M, 3) float64
    fg_rgb : (M, 3) uint8 BGR
    info : dict
        Bookkeeping for the smoke test / log: ``placed`` is a list of
        ``{"slug", "kind"}`` per object, ``n_points`` is the actual total
        point count returned, ``counts`` summarises per-kind counts.
    """
    if not object_library or n_points <= 0:
        return (np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8),
                {"placed": [], "n_points": 0,
                 "counts": {"grasping": 0, "hand_free": 0,
                             "cable": 0, "other": 0}})

    hand_graspable = [o for o in object_library
                       if o.get("category") == "hand"
                       and o.get("graspable_on_wire")]
    gripper_graspable = [o for o in object_library
                          if o.get("category") in ("gripper", "arm")
                          and o.get("graspable_on_wire")]
    hand_pool = [o for o in object_library if o.get("category") == "hand"]
    cable_pool = [o for o in object_library
                   if o.get("category") == "negative_wire_like"]
    other_pool = [o for o in object_library
                   if o.get("category") in ("gripper", "arm", "rope")]

    n_grasp = int(rng.randint(2, 5))         # 2..4
    n_hand_free = int(rng.randint(2, 5))      # 2..4
    n_cables_fg = int(rng.randint(3, 7))      # 3..6
    n_other = int(rng.randint(1, 4))          # 1..3

    total_items = n_grasp + n_hand_free + n_cables_fg + n_other
    if total_items == 0:
        return (np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8),
                {"placed": [], "n_points": 0,
                 "counts": {"grasping": 0, "hand_free": 0,
                             "cable": 0, "other": 0}})

    n_keep_per = max(256, n_points // total_items)

    pieces_pts: list[np.ndarray] = []
    pieces_rgb: list[np.ndarray] = []
    info: dict = {"placed": [],
                  "counts": {"grasping": 0, "hand_free": 0,
                              "cable": 0, "other": 0}}

    # Grasping: ≥1 if any graspable available; bias toward hands so skin
    # is dense in the foreground.
    placed_grasp = 0
    for _ in range(n_grasp):
        if hand_graspable and rng.uniform(0.0, 1.0) < 0.7:
            obj = hand_graspable[rng.randint(0, len(hand_graspable))]
        elif gripper_graspable:
            obj = gripper_graspable[rng.randint(0, len(gripper_graspable))]
        elif hand_graspable:
            obj = hand_graspable[rng.randint(0, len(hand_graspable))]
        else:
            continue
        result = _place_hand_on_wire(
            obj=obj, rng=rng,
            skeleton_nodes=skeleton_nodes, segments=segments,
            n_keep=n_keep_per,
        )
        if result is None:
            continue
        pts, rgb = result
        pieces_pts.append(pts)
        pieces_rgb.append(rgb)
        info["placed"].append({"slug": obj["slug"], "kind": "grasping"})
        placed_grasp += 1
    info["counts"]["grasping"] = placed_grasp

    # Free-floating hands (not grasping). Always present per spec.
    placed_hf = 0
    for _ in range(n_hand_free):
        if not hand_pool:
            break
        obj = hand_pool[rng.randint(0, len(hand_pool))]
        pts, rgb = _place_object_in_foreground(
            obj=obj, rng=rng,
            bbox_min=bbox_min, bbox_max=bbox_max,
            n_keep=n_keep_per,
        )
        pieces_pts.append(pts)
        pieces_rgb.append(rgb)
        info["placed"].append({"slug": obj["slug"], "kind": "hand_free"})
        placed_hf += 1
    info["counts"]["hand_free"] = placed_hf

    # Free-floating cables (wire-shaped negatives).
    placed_cable = 0
    for _ in range(n_cables_fg):
        if not cable_pool:
            break
        obj = cable_pool[rng.randint(0, len(cable_pool))]
        pts, rgb = _place_object_in_foreground(
            obj=obj, rng=rng,
            bbox_min=bbox_min, bbox_max=bbox_max,
            n_keep=n_keep_per,
        )
        pieces_pts.append(pts)
        pieces_rgb.append(rgb)
        info["placed"].append({"slug": obj["slug"], "kind": "cable"})
        placed_cable += 1
    info["counts"]["cable"] = placed_cable

    # Other clutter (grippers, arms, ropes).
    placed_other = 0
    for _ in range(n_other):
        if not other_pool:
            break
        obj = other_pool[rng.randint(0, len(other_pool))]
        pts, rgb = _place_object_in_foreground(
            obj=obj, rng=rng,
            bbox_min=bbox_min, bbox_max=bbox_max,
            n_keep=n_keep_per,
        )
        pieces_pts.append(pts)
        pieces_rgb.append(rgb)
        info["placed"].append({"slug": obj["slug"], "kind": "other"})
        placed_other += 1
    info["counts"]["other"] = placed_other

    if not pieces_pts:
        info["n_points"] = 0
        return (np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), info)

    pts_all = np.concatenate(pieces_pts, axis=0)
    rgb_all = np.concatenate(pieces_rgb, axis=0)
    info["n_points"] = int(pts_all.shape[0])
    info["n_objects"] = len(info["placed"])
    return pts_all, rgb_all, info


# ---------------------------------------------------------------------------
# (continued: _pick_clutter_position lives below)
# ---------------------------------------------------------------------------

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


#: Categories that should NEVER be placed as floor-level background clutter.
#: Hands/grippers/arms/cables are foreground objects: they only appear via
#: :func:`generate_foreground_scene`. Putting a hand on the floor as bg
#: clutter would imply detached hands lying around — visually wrong.
_FOREGROUND_ONLY_CATEGORIES: frozenset = frozenset({
    "hand", "gripper", "arm", "negative_wire_like",
})


def jitter_per_point_rgb(
    point_rgb: np.ndarray,
    rng: np.random.RandomState,
    max_hue_shift_deg: float = 25.0,
    max_sat_scale: float = 0.25,
    max_val_scale: float = 0.20,
) -> np.ndarray:
    """Apply per-source HSV jitter to a per-point BGR array.

    Phase 9: breaks the "harness texture statistics = DLO" shortcut that v2
    over-fit to. The jitter is a single random hue / saturation / value shift
    applied uniformly to ALL points in the array — i.e. it shifts the whole
    harness's colour identity per source, which means across the dataset the
    DLO class spans a much wider colour manifold than the v1/v2 texture set
    alone gives.

    This stays within the project-wide invariant (§0.5) that "all texture /
    colour MUST stay in the PCL or 2D photo, never as image post-processing"
    because it mutates per-point BGR BEFORE rasterisation.

    Parameters
    ----------
    point_rgb :
        ``(N, 3) uint8`` BGR — caller's compute_per_point_rgb output.
    rng :
        Source of randomness.
    max_hue_shift_deg, max_sat_scale, max_val_scale :
        Bounds on the per-call jitter. Hue shift is uniform in
        ``[-max_hue_shift_deg, max_hue_shift_deg]`` degrees; saturation and
        value scales are uniform in ``[1-max_*, 1+max_*]``.

    Returns
    -------
    ``(N, 3) uint8`` BGR with the jitter applied.
    """
    if point_rgb.size == 0:
        return point_rgb
    hue_shift = float(rng.uniform(-max_hue_shift_deg, max_hue_shift_deg))
    sat_scale = float(rng.uniform(1.0 - max_sat_scale, 1.0 + max_sat_scale))
    val_scale = float(rng.uniform(1.0 - max_val_scale, 1.0 + max_val_scale))

    bgr = point_rgb.reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    # OpenCV HSV: H in [0,180), S/V in [0,255]
    hsv[..., 0] = np.mod(hsv[..., 0] + hue_shift * 0.5, 180.0)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255)
    bgr_out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr_out.reshape(point_rgb.shape)


def _tint_texture(
    tex: np.ndarray,
    rng: np.random.RandomState,
    brightness_range: tuple[float, float] = (0.55, 1.20),
    hue_shift_deg: float = 40.0,
    sat_scale_range: tuple[float, float] = (0.40, 1.20),
) -> np.ndarray:
    """Return a tinted copy of ``tex`` (BGR uint8).

    Used to massively expand the effective wall-colour palette beyond the 11
    seed photos. Each wall samples its own tint so a single source frame can
    show 5 visibly distinct wall shades.
    """
    bgr = tex.astype(np.uint8)
    if bgr.size == 0:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = float(rng.uniform(-hue_shift_deg, hue_shift_deg))
    sat_scale = float(rng.uniform(sat_scale_range[0], sat_scale_range[1]))
    val_scale = float(rng.uniform(brightness_range[0], brightness_range[1]))
    hsv[..., 0] = np.mod(hsv[..., 0] + hue_shift * 0.5, 180.0)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Phase 13 levers: 3D scene lighting + gradient object colours.
# Both bake their effect into the per-point BGR BEFORE rasterisation (the
# §0.5 / §4.7 all-colour-in-PCL invariant) and depend only on world position,
# so they are identical across all 6 canonical views.
# ---------------------------------------------------------------------------

def sample_light_direction(
    rng: np.random.RandomState,
    elevation_range_deg: tuple[float, float] = (22.0, 58.0),
) -> np.ndarray:
    """Unit vector for the direction a distant scene light *travels*.

    Azimuth is uniform; elevation is above the horizon so the light comes from
    overhead-and-to-one-side and travels down into the scene (negative Y
    component). The per-source randomisation means the lit side rotates around
    the scene across the dataset.
    """
    az = float(rng.uniform(0.0, 2.0 * np.pi))
    el = float(rng.uniform(*[np.radians(e) for e in elevation_range_deg]))
    ce = np.cos(el)
    return np.array([np.cos(az) * ce, -np.sin(el), np.sin(az) * ce],
                    dtype=np.float64)


def apply_scene_lighting(
    points: np.ndarray,
    point_rgb: np.ndarray,
    light_dir: np.ndarray,
    centre: np.ndarray,
    ambient: float = 0.65,
    gain: float = 0.45,
    span: float = 1.6,
) -> np.ndarray:
    """Bake a directional 3D lighting term into a per-point BGR array.

    Models a distant light arriving along ``light_dir``: a point's exposure is
    a smooth ramp of how far it sits *toward* the light (opposite to
    ``light_dir``) along that axis, relative to the scene centre::

        t = -((p - centre) . light_dir) / span     # toward light → larger
        b = clip(0.5 + 0.5 * t, 0, 1)               # bright side .. shadow side
        s = ambient + gain * b
        bgr = clip(bgr * s, 0, 255)

    This is a true 3D lighting effect (depends only on world position, so the
    bright/shadow sides are consistent across all 6 canonical views and stable
    across LBS anim frames), baked into the point cloud — NOT a screen-space
    pass (§0.5). The 2D photographic backdrop is never a point and is therefore
    left untouched. ``points`` and ``point_rgb`` must be the same length ``N``.
    """
    if point_rgb.size == 0:
        return point_rgb
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    u = np.asarray(light_dir, dtype=np.float64).reshape(3)
    c = np.asarray(centre, dtype=np.float64).reshape(3)
    t = -((p - c[None, :]) @ u) / max(float(span), 1e-6)
    b = np.clip(0.5 + 0.5 * t, 0.0, 1.0)
    s = (float(ambient) + float(gain) * b)[:, None]
    return np.clip(point_rgb.astype(np.float32) * s, 0, 255).astype(np.uint8)


def apply_object_color_gradient(
    pts: np.ndarray,
    base_col: np.ndarray,
    rng: np.random.RandomState,
    max_hue_shift_deg: float = 60.0,
    sat_scale_range: tuple[float, float] = (0.60, 1.45),
    val_scale_range: tuple[float, float] = (0.85, 1.18),
) -> np.ndarray:
    """Bake a smooth per-object colour gradient into one object's BGR.

    A random gradient axis is drawn; each point's normalised projection
    ``t in [0, 1]`` along that axis interpolates between two HSV endpoint
    adjustments (hue shift + saturation/value scale). This turns a
    near-uniform object colour into a spatially-varying gradient and widens
    the object-colour manifold across the dataset. Colour mutation happens in
    the per-point BGR BEFORE rasterisation (§0.5 invariant).

    The gradient deliberately owns CHROMA (a wide hue ±60° + saturation swing)
    and keeps the VALUE/brightness swing modest, so it composes with — rather
    than fights / is overwritten by — the lighting lever (which owns luma via
    a hue-preserving multiply).
    """
    n = base_col.shape[0]
    if n == 0:
        return base_col
    g = rng.normal(size=3)
    ng = float(np.linalg.norm(g))
    if ng < 1e-9:
        return base_col
    g = g / ng
    proj = np.asarray(pts, dtype=np.float64).reshape(-1, 3) @ g
    lo, hi = float(proj.min()), float(proj.max())
    t = (proj - lo) / max(hi - lo, 1e-9)

    hue0 = float(rng.uniform(-max_hue_shift_deg, max_hue_shift_deg))
    hue1 = float(rng.uniform(-max_hue_shift_deg, max_hue_shift_deg))
    sat0 = float(rng.uniform(*sat_scale_range))
    sat1 = float(rng.uniform(*sat_scale_range))
    val0 = float(rng.uniform(*val_scale_range))
    val1 = float(rng.uniform(*val_scale_range))
    hue = hue0 + (hue1 - hue0) * t
    sat = sat0 + (sat1 - sat0) * t
    val = val0 + (val1 - val0) * t

    hsv = cv2.cvtColor(base_col.reshape(-1, 1, 3).astype(np.uint8),
                       cv2.COLOR_BGR2HSV).astype(np.float32).reshape(-1, 3)
    hsv[:, 0] = np.mod(hsv[:, 0] + hue * 0.5, 180.0)  # OpenCV hue is [0,180)
    hsv[:, 1] = np.clip(hsv[:, 1] * sat, 0, 255)
    hsv[:, 2] = np.clip(hsv[:, 2] * val, 0, 255)
    return cv2.cvtColor(hsv.reshape(-1, 1, 3).astype(np.uint8),
                        cv2.COLOR_HSV2BGR).reshape(-1, 3)


def generate_phase9_room_scene(
    rng: np.random.RandomState,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    texture_library: list[np.ndarray],
    object_library: list[dict] | None = None,
    n_points: int = 60000,
    enable_ceiling: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 9 multi-wall 3D backdrop.

    Replaces Phase 4's single-floor + 2D-photo composition with a fully 3D
    enclosure: floor + back wall + left wall + right wall + (optional)
    ceiling. Each wall is independently sampled from ``texture_library``
    AND gets its own random HSV tint, so a single source frame can show 4–5
    visibly distinct wall colours. The 2D photographic backdrop kwarg of
    ``rasterize_view`` is no longer needed — the walls fill the camera
    frustum from every randomised viewpoint.

    A small number of non-wire-shaped floor clutter objects (drawn from
    ``object_library`` — caller is responsible for filtering out
    wire-shaped / hand categories) are still placed in front of the
    harness so the scene reads as a real room with clutter, not an empty
    box.

    Parameters
    ----------
    rng :
        Source of randomness; same state → same scene.
    bbox_min, bbox_max :
        Per-axis (3,) extent of the harness at rest pose.
    texture_library :
        Flat list of BGR ``uint8`` images for wall textures (typically the
        output of :func:`load_background_library`).
    object_library :
        List of clutter dicts as returned by :func:`load_object_library`
        AFTER :func:`filter_library_by_category` has dropped wire-shaped /
        hand categories. When empty or ``None``, only the room shell is
        generated.
    n_points :
        Soft point budget across walls + clutter.
    enable_ceiling :
        Whether to render a 5th ceiling slab. Doesn't affect side/top views
        much but reads correctly for bottom-up shots.

    Returns
    -------
    bg_pcl : (M, 3) float64
    bg_rgb : (M, 3) uint8 BGR
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=np.float64).reshape(3)

    if not texture_library or n_points <= 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    # Room dimensions. The orthographic frustum is depth ∈ [-1.1, 1.1] in
    # cam-Z and image plane cam-X ∈ [-HALF_W=1.467, 1.467], cam-Y ∈ [-1.1,
    # 1.1]. With randomised camera azimuth+elevation, points outside cam-Z
    # range get frustum-culled but the visible projection of each wall
    # depends on which way the camera is looking. We make each wall's
    # face-on extents (the dimensions visible when the camera is
    # perpendicular to that wall) cover the image plane, while the
    # perpendicular dimension (the wall's distance from origin) stays
    # within cam-Z range. This ensures every randomised viewpoint sees
    # walls filling at least one image edge.
    FRUSTUM = 1.1
    HALF_W = 1.467  # frustum_half_horizontal from pcl_to_rgbd.py
    # Wall **plane distances** stay within cam-Z range (≤ FRUSTUM) so they
    # don't get culled when the camera views them face-on. The wall
    # **face-on extents** are pushed out to HALF_W in the wide image axis
    # so the wall fills the image frame from every viewing angle (face-on
    # wall extent = image width = 2 * HALF_W = 2.93 world units).
    floor_y = max(float(bbox_min[1] - 0.05), -FRUSTUM + 0.02)
    ceiling_y = min(float(bbox_max[1] + 0.50), FRUSTUM - 0.02)
    half_x = float(rng.uniform(1.00, 1.08))          # side wall plane depth
    half_z_back = float(rng.uniform(1.00, 1.08))     # back wall plane depth
    half_z_front = float(rng.uniform(1.00, 1.08))    # front wall plane depth
    floor_extent_xz = max(half_x, half_z_back, half_z_front) + 0.04
    wall_face_extent = HALF_W + 0.05  # ~1.52 — fills image plane

    n_walls = 5 if enable_ceiling else 4
    # Floor + 4 walls share the budget after clutter.
    n_clutter_budget = int(0.25 * n_points)
    n_per_wall = (n_points - n_clutter_budget) // (n_walls + 1)
    n_floor = n_per_wall
    n_ceiling = n_per_wall if enable_ceiling else 0
    n_back = n_per_wall
    n_front = n_per_wall
    n_left = n_per_wall
    n_right = n_per_wall

    def _pick_and_tint() -> np.ndarray:
        idx = rng.randint(0, len(texture_library))
        return _tint_texture(texture_library[idx], rng=rng)

    # ── Walls (each independently textured + tinted) ──────────────────
    pieces_pts: list[np.ndarray] = []
    pieces_rgb: list[np.ndarray] = []

    # Floor: xz plane at floor_y. Face-on extent in X uses wall_face_extent
    # (the image plane width) so top/bottom views see floor edge-to-edge.
    pts, rgb = _make_textured_plane(
        centre=np.array([0.0, floor_y, 0.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 0.0, 1.0]),
        half_u=wall_face_extent,
        half_v=floor_extent_xz,
        texture=_pick_and_tint(),
        rng=rng, n_points=n_floor,
        u_tile=float(rng.uniform(2.5, 4.5)),
        v_tile=float(rng.uniform(2.5, 4.5)),
    )
    pieces_pts.append(pts)
    pieces_rgb.append(rgb)

    # Ceiling: xz plane at ceiling_y, optionally rendered.
    if enable_ceiling and n_ceiling > 0:
        pts, rgb = _make_textured_plane(
            centre=np.array([0.0, ceiling_y, 0.0]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_u=wall_face_extent,
            half_v=floor_extent_xz,
            texture=_pick_and_tint(),
            rng=rng, n_points=n_ceiling,
            u_tile=float(rng.uniform(2.0, 3.5)),
            v_tile=float(rng.uniform(2.0, 3.5)),
        )
        pieces_pts.append(pts)
        pieces_rgb.append(rgb)

    # Back wall: xy plane at z=+half_z_back, face-on extent wall_face_extent
    # in X and ceiling-floor span in Y.
    half_y = (ceiling_y - floor_y) * 0.5
    centre_y = (ceiling_y + floor_y) * 0.5
    pts, rgb = _make_textured_plane(
        centre=np.array([0.0, centre_y, half_z_back]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        half_u=wall_face_extent,
        half_v=half_y,
        texture=_pick_and_tint(),
        rng=rng, n_points=n_back,
        u_tile=float(rng.uniform(2.0, 4.0)),
        v_tile=float(rng.uniform(1.5, 3.0)),
    )
    pieces_pts.append(pts)
    pieces_rgb.append(rgb)

    # Front wall: xy plane at z=-half_z_front.
    pts, rgb = _make_textured_plane(
        centre=np.array([0.0, centre_y, -half_z_front]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        half_u=wall_face_extent,
        half_v=half_y,
        texture=_pick_and_tint(),
        rng=rng, n_points=n_front,
        u_tile=float(rng.uniform(2.0, 4.0)),
        v_tile=float(rng.uniform(1.5, 3.0)),
    )
    pieces_pts.append(pts)
    pieces_rgb.append(rgb)

    # Left wall: yz plane at x=-half_x; face-on extent wall_face_extent in Z.
    pts, rgb = _make_textured_plane(
        centre=np.array([-half_x, centre_y, 0.0]),
        u_axis=np.array([0.0, 0.0, 1.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        half_u=wall_face_extent,
        half_v=half_y,
        texture=_pick_and_tint(),
        rng=rng, n_points=n_left,
        u_tile=float(rng.uniform(1.5, 3.0)),
        v_tile=float(rng.uniform(1.5, 3.0)),
    )
    pieces_pts.append(pts)
    pieces_rgb.append(rgb)

    # Right wall: yz plane at x=+half_x.
    pts, rgb = _make_textured_plane(
        centre=np.array([half_x, centre_y, 0.0]),
        u_axis=np.array([0.0, 0.0, 1.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        half_u=wall_face_extent,
        half_v=half_y,
        texture=_pick_and_tint(),
        rng=rng, n_points=n_right,
        u_tile=float(rng.uniform(1.5, 3.0)),
        v_tile=float(rng.uniform(1.5, 3.0)),
    )
    pieces_pts.append(pts)
    pieces_rgb.append(rgb)

    # ── Clutter on the floor (filtered library) ───────────────────────
    obj_lib = [
        o for o in (object_library or [])
        if o.get("category") not in _FOREGROUND_ONLY_CATEGORIES
    ]
    placed: list[tuple[float, float, float]] = []
    if obj_lib and n_clutter_budget > 0:
        n_pick = int(rng.randint(3, 7))  # 3..6 non-wire clutter items
        chosen = [obj_lib[rng.randint(0, len(obj_lib))] for _ in range(n_pick)]
        total_avail = sum(o["points"].shape[0] for o in chosen)
        keep_ratio = min(1.0, n_clutter_budget / max(total_avail, 1))
        for obj in chosen:
            n_avail = obj["points"].shape[0]
            n_keep = max(64, int(round(n_avail * keep_ratio)))
            n_keep = min(n_keep, n_avail)
            pts, rgb = _place_object_on_floor(
                obj=obj, rng=rng, floor_y=floor_y,
                bbox_min=bbox_min, bbox_max=bbox_max,
                floor_extent=floor_extent_xz,
                placed=placed,
                n_keep=n_keep,
            )
            pieces_pts.append(pts)
            pieces_rgb.append(rgb)

    nonempty_pts = [p for p in pieces_pts if p.size]
    nonempty_rgb = [r for r in pieces_rgb if r.size]
    if not nonempty_pts:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    bg_pcl = np.concatenate(nonempty_pts, axis=0)
    bg_rgb = np.concatenate(nonempty_rgb, axis=0)
    return bg_pcl, bg_rgb


def generate_phase9_foreground(
    rng: np.random.RandomState,
    object_library: list[dict],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    n_points: int = 8000,
    max_objects: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Phase 9 foreground: 0-2 benign occluders, no hands, no wire-shaped negs.

    Drops the v2 saturation entirely. The caller is responsible for passing
    an ``object_library`` that has already been filtered to drop
    ``negative_wire_like`` / ``rope`` / ``hand`` / ``gripper`` / ``arm`` —
    see :data:`PHASE9_DROP_CATEGORIES`.

    Returns ``(pts, rgb, info)`` matching the
    :func:`generate_foreground_scene` signature so the renderer can use the
    same downstream concat code.
    """
    info: dict = {"placed": [], "n_points": 0,
                  "counts": {"benign_fg": 0}}
    if not object_library or n_points <= 0 or max_objects <= 0:
        return (np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), info)

    n_place = int(rng.randint(0, max_objects + 1))  # 0..max_objects inclusive
    if n_place == 0:
        return (np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), info)

    n_keep_per = max(256, n_points // n_place)
    pieces_pts: list[np.ndarray] = []
    pieces_rgb: list[np.ndarray] = []
    for _ in range(n_place):
        obj = object_library[rng.randint(0, len(object_library))]
        pts, rgb = _place_object_in_foreground(
            obj=obj, rng=rng,
            bbox_min=bbox_min, bbox_max=bbox_max,
            n_keep=n_keep_per,
        )
        pieces_pts.append(pts)
        pieces_rgb.append(rgb)
        info["placed"].append({"slug": obj["slug"], "kind": "benign_fg"})
    info["counts"]["benign_fg"] = n_place

    if not pieces_pts:
        return (np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), info)
    pts_all = np.concatenate(pieces_pts, axis=0)
    rgb_all = np.concatenate(pieces_rgb, axis=0)
    info["n_points"] = int(pts_all.shape[0])
    info["n_objects"] = len(info["placed"])
    return pts_all, rgb_all, info


#: Categories Phase 9 drops entirely from foreground + background placement.
#: Wire-shaped negatives are the v2 over-specialisation smoking gun; hands /
#: grippers / arms are the saturating skin/finger-curvature density v2 used
#: to dense-grasp the wire. Phase 9 wants the input image harder but the
#: labelling decision rule simple (everything that's not harness is BG, no
#: wire-shaped or finger-shaped look-alikes in the negatives).
PHASE9_DROP_CATEGORIES: frozenset = frozenset({
    "hand", "gripper", "arm", "negative_wire_like", "rope",
})


#: Hand pose slugs that depict an OPEN / NEUTRAL hand (no grasp posture). Used
#: by :func:`generate_phase11_foreground` to bias the sampled hand toward
#: poses that won't read as "hand reaching for the wire" even when the hand
#: sits in front of the harness in 2D. Slugs not in this list (fist / pinch /
#: grasp_cylindrical / pointing) are still allowed but down-weighted.
_PHASE11_OPEN_HAND_TOKENS: tuple[str, ...] = (
    "open_palm_up", "open_palm_down", "half_open", "relaxed", "flat",
)


def generate_phase11_foreground(
    rng: np.random.RandomState,
    object_library: list[dict],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    n_points: int = 6000,
    p_hand: float = 0.3,
    safety_margin: float = 0.08,
    frustum_half: tuple[float, float, float] = (1.20, 1.00, 1.00),
    open_pose_bias: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Phase 11 foreground: AT MOST ONE hand per sample, NEVER touching the wire.

    Phase 11 keeps the Phase 4 backbone exactly (textured floor + photographic
    2D backdrop + 21-entry CC0 clutter) and adds light hand presence as the
    one foreground change. The rule is strict: with probability ``p_hand`` the
    function places exactly one hand from ``object_library`` somewhere in the
    camera frustum such that the hand's 3D AABB does NOT intersect the
    harness 3D AABB (expanded by ``safety_margin``). With probability
    ``1 - p_hand``, no foreground is placed and the sample looks identical to
    a vanilla Phase 4 sample.

    This intentionally diverges from Phase 8 v2's grasp-on-wire foreground
    (which catastrophically collapsed real-world transfer, see CONTEXT.md
    §0.13) and from :func:`generate_foreground_scene`'s dense composition.
    The hand is a single, sparse skin-toned object the model can use as a
    discriminative negative without overwhelming the gradient.

    Parameters
    ----------
    rng :
        Source of randomness; same state → same scene.
    object_library :
        Output of :func:`load_object_library`. Phase 11 filters to the
        ``"hand"`` category internally; non-hand entries are ignored by this
        function. Empty library → no foreground.
    bbox_min, bbox_max :
        Per-axis (3,) extent of the harness at rest pose. The hand AABB is
        forced to be disjoint from ``[bbox_min - safety_margin, bbox_max +
        safety_margin]``.
    n_points :
        Maximum point budget for the hand subsample (the underlying hand
        meshes have 8000 points; n_points caps the placed count).
    p_hand :
        Probability that this sample gets a hand (vs. no foreground).
    safety_margin :
        Extra padding (world units) around the harness AABB. The hand AABB
        is kept disjoint from the padded AABB.
    frustum_half :
        ``(half_x, half_y, half_z)`` bounds for the hand centroid sampling
        region. Defaults to slightly tighter than the orthographic camera so
        the hand stays mostly in frame from at least one of the 6 views.
    open_pose_bias :
        Probability of preferring an "open / non-grasp" hand pose (palm up,
        palm down, half-open, relaxed, flat). Reduces the visual chance of
        the hand reading as "grabbing a wire" even when there's no wire near
        it. A non-open pose is still allowed with probability
        ``1 - open_pose_bias``.

    Returns
    -------
    fg_pcl : (M, 3) float64
    fg_rgb : (M, 3) uint8 BGR
    info : dict
        Bookkeeping (matches the :func:`generate_foreground_scene` /
        :func:`generate_phase9_foreground` schema):

        * ``placed`` — list of ``{"slug", "kind"}`` per placed object.
        * ``counts`` — ``{"hand_off_wire": int, "no_fg": int}``.
        * ``n_points`` — total points returned.
        * ``safety_margin`` — echoed for downstream sanity-checking.
        * ``placement_attempts`` — how many rejection-sampler iterations the
          hand needed (helpful for diagnosing scenes where the harness fills
          the frustum so densely that the safe region is small).
    """
    info: dict = {"placed": [], "n_points": 0,
                  "counts": {"hand_off_wire": 0, "no_fg": 0},
                  "safety_margin": float(safety_margin),
                  "placement_attempts": 0}

    if not object_library or n_points <= 0:
        info["counts"]["no_fg"] = 1
        return (np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), info)

    if float(rng.uniform(0.0, 1.0)) >= p_hand:
        info["counts"]["no_fg"] = 1
        return (np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), info)

    hand_pool = [o for o in object_library if o.get("category") == "hand"]
    if not hand_pool:
        info["counts"]["no_fg"] = 1
        return (np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), info)

    open_hands = [
        o for o in hand_pool
        if any(tok in o.get("slug", "") for tok in _PHASE11_OPEN_HAND_TOKENS)
    ]
    if open_hands and float(rng.uniform(0.0, 1.0)) < open_pose_bias:
        pick_from = open_hands
    else:
        pick_from = hand_pool
    obj = pick_from[rng.randint(0, len(pick_from))]

    base_pts = obj["points"].astype(np.float64).copy()
    base_col = obj["colors"]
    if base_pts.shape[0] > n_points:
        idx = rng.choice(base_pts.shape[0], size=n_points, replace=False)
        base_pts = base_pts[idx]
        base_col = base_col[idx]

    lo, hi = obj["natural_scale_range"]
    scale = float(rng.uniform(lo, hi))
    pts = base_pts * scale

    theta_y = float(rng.uniform(0.0, 2.0 * np.pi))
    theta_x = float(rng.uniform(-np.deg2rad(45.0), np.deg2rad(45.0)))
    theta_z = float(rng.uniform(-np.deg2rad(45.0), np.deg2rad(45.0)))
    R = (_rotation_about_axis(np.array([0.0, 1.0, 0.0]), theta_y)
         @ _rotation_about_axis(np.array([1.0, 0.0, 0.0]), theta_x)
         @ _rotation_about_axis(np.array([0.0, 0.0, 1.0]), theta_z))
    pts_rot = pts @ R.T

    hand_min_local = pts_rot.min(axis=0)
    hand_max_local = pts_rot.max(axis=0)
    centroid_local = pts_rot.mean(axis=0)

    bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    harness_safe_min = bbox_min - safety_margin
    harness_safe_max = bbox_max + safety_margin

    fx, fy, fz = (float(frustum_half[0]),
                  float(frustum_half[1]),
                  float(frustum_half[2]))

    placed_pos: np.ndarray | None = None
    attempts = 0
    for _ in range(80):
        attempts += 1
        candidate = np.array([
            float(rng.uniform(-fx, fx)),
            float(rng.uniform(-fy, fy)),
            float(rng.uniform(-fz, fz)),
        ])
        shift = candidate - centroid_local
        hand_min_world = hand_min_local + shift
        hand_max_world = hand_max_local + shift
        # AABB intersection iff all three axes overlap.
        overlap = bool(np.all(
            (hand_min_world < harness_safe_max)
            & (hand_max_world > harness_safe_min)
        ))
        if not overlap:
            placed_pos = candidate
            break

    info["placement_attempts"] = attempts

    if placed_pos is None:
        # Rejection sampler exhausted. Force the hand off to a frustum corner
        # along whichever harness axis has the most headroom (almost always
        # ±Z because the harness lies near the XY plane).
        gaps = np.maximum(0.0, np.array([fx, fy, fz]) - np.maximum(
            np.abs(harness_safe_min), np.abs(harness_safe_max)))
        axis = int(np.argmax(gaps))
        sign = 1.0 if float(rng.uniform(0.0, 1.0)) > 0.5 else -1.0
        target_axis_val = sign * (
            max(abs(harness_safe_min[axis]), abs(harness_safe_max[axis]))
            + 0.5 * (hand_max_local[axis] - hand_min_local[axis])
            + 0.02
        )
        # Clamp to frustum.
        target_axis_val = float(np.clip(
            target_axis_val,
            -np.array([fx, fy, fz])[axis],
            np.array([fx, fy, fz])[axis],
        ))
        placed_pos = np.zeros(3, dtype=np.float64)
        placed_pos[axis] = target_axis_val

    pts_world = pts_rot - centroid_local[None, :] + placed_pos[None, :]

    info["placed"].append({"slug": obj["slug"], "kind": "hand_off_wire"})
    info["counts"]["hand_off_wire"] = 1
    info["n_points"] = int(pts_world.shape[0])
    info["n_objects"] = 1
    return pts_world, base_col, info


def generate_background_scene(
    rng: np.random.RandomState,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    texture_library: list[np.ndarray],
    object_library: list[dict] | None = None,
    n_points: int = 30000,
    n_objects_range: tuple[int, int] = (10, 16),
    object_color_gradient: bool = False,
    grad_rng: np.random.RandomState | None = None,
    grad_kwargs: dict | None = None,
    floor_tex_override: np.ndarray | None = None,
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
    floor_tex_override :
        Optional BGR ``uint8`` image replacing the floor texture (Phase 19
        busy-floor lever). The usual ``rng`` floor-texture draw still occurs
        (and is discarded) so the composition stream is unchanged.

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

    # Filter out foreground-only categories (hand/gripper/arm/cable). These
    # objects belong to ``generate_foreground_scene`` and should never become
    # detached "lying on the floor" clutter.
    obj_lib = [
        o for o in (object_library or [])
        if o.get("category") not in _FOREGROUND_ONLY_CATEGORIES
    ]

    # ``n_objects_range`` selects how many clutter PCLs to place on the floor.
    # v2's "extreme clutter" default is (10, 16) → 10-15 objects. Phase 4
    # baseline (and Phase 12) pass (3, 7) → 3-6 objects.
    chosen: list[dict] = []
    if obj_lib:
        lo, hi = int(n_objects_range[0]), int(n_objects_range[1])
        if hi <= lo:
            hi = lo + 1
        n_pick = int(rng.randint(lo, hi))
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
    # Phase 19 lever (busy floor): swap the floor TEXTURE only. The
    # original-pool draw above still executes (and is discarded) so the
    # scene-composition ``rng`` stream is identical with the lever ON or
    # OFF; point positions/labels are texture-independent either way.
    floor_tile = 4.0
    if floor_tex_override is not None:
        floor_tex = floor_tex_override
        # Busy swatches are broad-stroke patterns: 4x4 tiling over the
        # ~3-unit floor leaves one texel ≈ 0.3 screen px, aliasing the
        # strokes into colour noise. Tile ~1x1 so they stay legible.
        # ``u_tile``/``v_tile`` only affect the UV→colour lookup (after
        # the rng point draws), so the default path is the same.
        floor_tile = 1.0
    floor_pts, floor_rgb = _make_textured_plane(
        centre=np.array([0.0, floor_y, 0.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 0.0, 1.0]),
        half_u=floor_extent,
        half_v=floor_extent,
        texture=floor_tex,
        rng=rng, n_points=n_floor,
        u_tile=floor_tile, v_tile=floor_tile,
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
        # Phase 13 lever (b): bake a per-object colour gradient. Uses a
        # SEPARATE rng so the scene-composition ``rng`` stream above is
        # byte-identical whether or not the gradient is enabled (the
        # lever-OFF control then reproduces Phase 4 exactly).
        if object_color_gradient and grad_rng is not None:
            rgb = apply_object_color_gradient(pts, rgb, grad_rng,
                                              **(grad_kwargs or {}))
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
