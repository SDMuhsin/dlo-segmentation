"""Find connector SLOTS in a harness point cloud.

A slot is one cluster of source-class-3 points. It is described by a centre,
a principal axis oriented from the connector TOWARD the wire that is attached
to it, a robust length and width, and the indices of its points. Nothing here
mutates its inputs, and nothing is random (KMeans uses a fixed seed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans

from .config import ConnectorInsertionConfig
from .constants import SOURCE_CONNECTOR_CLASS, WIRE_LIKE_CLASSES

#: Farther than this (world units, ~55 px) from any wire point, a slot is
#: considered detached and its direction falls back to a deterministic sign.
MAX_WIRE_DIST = 0.25
#: Slot width never goes below this fraction of its length (flat clusters).
MIN_ASPECT = 0.15


@dataclass(frozen=True)
class Slot:
    """One connector slot. Lengths are in world units (``*_px`` = x px_per_unit)."""

    indices: np.ndarray               # (n,) indices into the harness arrays
    center: np.ndarray                # (3,) centre (axial midpoint of robust extent)
    axis: np.ndarray                  # (3,) unit vector pointing from connector toward its wire
    minor: np.ndarray                 # (3,) unit vector, second principal direction (wide side)
    length: float                     # robust axial extent (2-98 percentile)
    width: float                      # larger robust perpendicular extent
    attach_point: Optional[np.ndarray]  # mean of the nearest wire points at the wire end, or None
    has_wire: bool                    # a wire was found within MAX_WIRE_DIST
    degenerate: bool                  # PCA failed or extent ~0: never replaced
    n_points: int

    @property
    def wire_end(self) -> np.ndarray:
        """Point on the axis where the wire enters (centre + axis * length / 2)."""
        return self.center + self.axis * (self.length / 2.0)


def _describe(points: np.ndarray, idx: np.ndarray, wire_tree: Optional[cKDTree],
              wire_pts: np.ndarray) -> Slot:
    """Build a Slot for the cluster ``points[idx]``; guards every SVD edge case."""
    p = points[idx].astype(np.float64)
    n = len(p)
    c = p.mean(0)
    axis = np.array([1.0, 0.0, 0.0]); minor = np.array([0.0, 1.0, 0.0])
    degenerate = n < 3 or not np.isfinite(p).all()
    length = width = 0.0
    if not degenerate:
        try:
            _, s, vt = np.linalg.svd(p - c, full_matrices=False)
        except np.linalg.LinAlgError:
            s, vt = np.zeros(3), np.eye(3)
        if not np.isfinite(vt).all() or s[0] < 1e-9:
            degenerate = True
        else:
            axis, minor = vt[0], vt[1]
            q = (p - c) @ vt.T
            lo, hi = np.percentile(q[:, 0], [2, 98])
            length = float(hi - lo)
            c = c + axis * (lo + hi) / 2.0
            e1 = np.subtract(*np.percentile(q[:, 1], [98, 2]))
            e2 = np.subtract(*np.percentile(q[:, 2], [98, 2]))
            width = float(max(e1, e2, MIN_ASPECT * length))
            if length < 1e-6:
                degenerate = True
    # orient the axis toward the attached wire
    attach, has_wire = None, False
    if wire_tree is not None and not degenerate:
        end_a, end_b = c + axis * length / 2, c - axis * length / 2
        k = min(3, len(wire_pts))
        da = float(np.mean(wire_tree.query(end_a, k=k)[0]))
        db = float(np.mean(wire_tree.query(end_b, k=k)[0]))
        if db < da:
            axis = -axis
            end_a = end_b
        has_wire = min(da, db) <= MAX_WIRE_DIST
        if has_wire:
            kk = min(8, len(wire_pts))
            attach = wire_pts[wire_tree.query(end_a, k=kk)[1]].mean(0)
    if not has_wire:
        # deterministic fallback: the dominant component of the axis is positive
        if axis[int(np.argmax(np.abs(axis)))] < 0:
            axis = -axis
    return Slot(indices=idx, center=c, axis=axis, minor=minor, length=length,
                width=width, attach_point=attach, has_wire=has_wire,
                degenerate=degenerate, n_points=n)


def detect_slots(pcl: np.ndarray, seg: np.ndarray,
                 cfg: ConnectorInsertionConfig) -> list[Slot]:
    """Cluster source-class-3 points into slots (ordered by first point index)."""
    pcl = np.asarray(pcl)
    seg = np.asarray(seg).astype(int)
    conn = np.where(seg == SOURCE_CONNECTOR_CLASS)[0]
    if conn.size < cfg.min_cluster_points:
        return []
    labels = DBSCAN(eps=cfg.cluster_eps, min_samples=3).fit_predict(pcl[conn])
    wire_idx = np.where(np.isin(seg, WIRE_LIKE_CLASSES))[0]
    wire_pts = pcl[wire_idx].astype(np.float64)
    tree = cKDTree(wire_pts) if wire_idx.size else None

    groups: list[np.ndarray] = []
    for lab in sorted(set(labels) - {-1}):
        g = conn[labels == lab]
        if g.size < cfg.min_cluster_points:
            continue
        groups.append(g)

    slots: list[Slot] = []
    for g in groups:
        s = _describe(pcl, g, tree, wire_pts)
        len_px = s.length * cfg.px_per_unit
        if cfg.split_large and not s.degenerate and len_px > cfg.split_length_px:
            k = int(np.ceil(len_px / cfg.split_length_px))
            sub = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(pcl[g])
            pieces = [g[sub == j] for j in range(k) if (sub == j).sum() >= cfg.min_cluster_points]
            if len(pieces) > 1:
                slots.extend(_describe(pcl, pc, tree, wire_pts) for pc in pieces)
                continue
        slots.append(s)
    return slots
