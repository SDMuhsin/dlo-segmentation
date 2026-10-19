"""Zoom views: close-up renders centred on an inserted connector.

A zoom view is a normal orthographic view whose pixel scale is multiplied by a
zoom factor and whose camera window is moved onto a connector slot. The output
stays 640x480, so the staging and training stack do not change. Everything here
is opt-in (``ConnectorInsertionConfig.zoom_views > 0``).

Pipeline per source (called from ``convert_to_video_dataset.convert_one_video``
after the six standard views, so those stay byte-identical):

    plan_zoom_views      pure planning: slot, candidate rotations, centre, zoom
    ZoomRenderer.render  per candidate: build the view cloud, crop the backdrop,
                         rasterise with ``rasterize_view(zoom=, centre=)``

The view cloud is the standard combined cloud (already lit) with two optional
swaps inside the window: the harness wire points become a smooth tube, and the
inserted connectors are re-sampled at the zoomed pixel scale. Re-sampled
points get the SAME lighting function, parameters and world positions rule as
every other point (``light`` callable), so colours match the standard views.

All planning randomness comes from one private RandomState
(``set_id*1000 + frame_id + ZOOM_RNG_OFFSET + seed_salt``) that exists only
when zoom views are requested.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import numpy as np

from .config import ConnectorInsertionConfig
from .constants import INSERTED_CONNECTOR_LABEL, PX_PER_UNIT, ZOOM_RNG_OFFSET
from .insert import ConnectorInsertion, InsertedSlot, resample_inserted

IMG_W, IMG_H = 640, 480
_N_DRAWS = 12
OCC_MAX = 5          # background points in front of the harness above which a candidate is not preferred
VIS_TOL_MM = 3.0     # depth slack (mm) when deciding whether a connector pixel is occluded


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def project(points: np.ndarray, R: np.ndarray) -> np.ndarray:
    """World -> camera (right, down, depth), orthographic."""
    return np.asarray(points, dtype=np.float64) @ np.asarray(R).T


def projected_extent(cam_xy: np.ndarray) -> tuple[float, np.ndarray]:
    """Robust principal-axis extent (world units) of 2D points, and the axis."""
    c = cam_xy - cam_xy.mean(0)
    if len(c) < 3:
        return 0.0, np.array([1.0, 0.0])
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    ax = vt[0]
    q = c @ ax
    lo, hi = np.percentile(q, [1, 99])
    return float(hi - lo), ax


def perturb_rotation(R: np.ndarray, max_deg: float, rs: np.random.RandomState) -> np.ndarray:
    """``R`` followed by a rotation of U(0, max_deg) degrees about a random axis."""
    k = rs.normal(size=3)
    k /= max(float(np.linalg.norm(k)), 1e-12)
    ang = np.deg2rad(max_deg * rs.uniform())
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    Rr = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return Rr @ np.asarray(R, dtype=np.float64)


def window_half_extent(zoom: float, scale: float = PX_PER_UNIT) -> tuple[float, float]:
    """Half width / height of the camera window in world units."""
    s = scale * zoom
    return IMG_W / 2.0 / s, IMG_H / 2.0 / s


def zoom_for_target(proj_len_world: float, target_frac: float, cfg: ConnectorInsertionConfig,
                    scale: float = PX_PER_UNIT) -> float:
    """Zoom that makes ``proj_len_world`` span ``target_frac`` of the frame height, clamped."""
    z = target_frac * IMG_H / max(proj_len_world * scale, 1e-9)
    return float(np.clip(z, cfg.zoom_min, cfg.zoom_max))


def fit_centre(centre: np.ndarray, cam_xy: np.ndarray, zoom: float,
               margin_frac: float = 0.03, scale: float = PX_PER_UNIT) -> np.ndarray:
    """Clamp ``centre`` so the bbox of ``cam_xy`` lies inside the window (when it fits)."""
    hw, hh = window_half_extent(zoom, scale)
    m = margin_frac * IMG_H / (scale * zoom)
    lo, hi = cam_xy.min(0), cam_xy.max(0)
    out = np.array(centre, dtype=np.float64)
    for i, h in ((0, hw), (1, hh)):
        a, b = hi[i] - h + m, lo[i] + h - m
        out[i] = float(np.clip(out[i], a, b)) if a <= b else 0.5 * (lo[i] + hi[i])
    return out


def crop_backdrop(bg: Optional[np.ndarray], centre: np.ndarray, zoom: float,
                  mode: str = "crop", scale: float = PX_PER_UNIT) -> Optional[np.ndarray]:
    """Backdrop for a zoom view: the camera window of the photo, resized to 640x480.

    ``crop`` maps standard-view pixel (x, y) to ``(p - p_c) * zoom + (320, 240)``
    with ``p_c = (320 + cx*scale, 240 + cy*scale)`` (bicubic, edge reflection), the
    exact image a zoom-times larger full render would show there. ``static``
    returns the photo unchanged.
    """
    if bg is None or mode == "static" or (zoom == 1.0 and not np.any(centre)):
        return bg
    import cv2
    pc = np.array([IMG_W / 2.0 + centre[0] * scale, IMG_H / 2.0 + centre[1] * scale])
    M = np.array([[zoom, 0.0, IMG_W / 2.0 - pc[0] * zoom],
                  [0.0, zoom, IMG_H / 2.0 - pc[1] * zoom]], dtype=np.float64)
    return cv2.warpAffine(bg, M, (IMG_W, IMG_H), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT)


# ----------------------------------------------------------------------------
# planning
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ZoomCandidate:
    view: str                    # canonical view name the rotation derives from
    R: np.ndarray                # (3, 3) rotation actually used
    perturbed: bool
    centre: np.ndarray           # (2,) camera-plane centre, world units
    zoom: float
    target_frac: Optional[float]  # None for the random-wire fallback
    proj_len_1x_px: float        # connector projected length at zoom 1
    planned_len_px: float        # proj_len_1x_px * zoom
    planned_frac: float          # planned_len_px / frame height
    zoom_clamped: bool
    bbox_px: tuple               # planned connector bbox (x0, y0, x1, y1) in the zoom view


@dataclass(frozen=True)
class ZoomPlan:
    index: int
    slot: int                    # index into ConnectorInsertion.inserted, -1 for the wire fallback
    asset: str
    candidates: tuple            # tuple[ZoomCandidate, ...] tried in order


def _local_rng(set_id: int, frame_id: int, cfg: ConnectorInsertionConfig) -> np.random.RandomState:
    return np.random.RandomState(
        (set_id * 1000 + frame_id + ZOOM_RNG_OFFSET + cfg.seed_salt) % (2 ** 32))


def _candidate(isl: InsertedSlot, pts: np.ndarray, view: str, R: np.ndarray, perturbed: bool,
               target_frac: float, jit: np.ndarray, cfg: ConnectorInsertionConfig) -> ZoomCandidate:
    cam = project(pts, R)
    cxy = cam[:, :2]
    plen, _ = projected_extent(cxy)
    zoom = zoom_for_target(plen, target_frac, cfg)
    clamped = bool(abs(zoom * plen * PX_PER_UNIT / IMG_H - target_frac) > 1e-6)
    # frame centre: biased toward the wire-side (cable) end along the connector axis
    a = isl.slot.axis
    q = (pts - pts.mean(0)) @ a
    q_c = q.max() - cfg.zoom_centre_bias * (q.max() - q.min())
    c3 = pts.mean(0) + a * (q_c - 0.0)
    c2 = project(c3, R)[:2]
    c2 = c2 + jit * cfg.zoom_centre_jitter * IMG_H / (PX_PER_UNIT * zoom)
    c2 = fit_centre(c2, cxy, zoom)
    s = PX_PER_UNIT * zoom
    lo, hi = cxy.min(0), cxy.max(0)
    bbox = ((lo[0] - c2[0]) * s + IMG_W / 2, (lo[1] - c2[1]) * s + IMG_H / 2,
            (hi[0] - c2[0]) * s + IMG_W / 2, (hi[1] - c2[1]) * s + IMG_H / 2)
    return ZoomCandidate(view=view, R=R, perturbed=perturbed, centre=c2, zoom=zoom,
                         target_frac=target_frac, proj_len_1x_px=plen * PX_PER_UNIT,
                         planned_len_px=plen * PX_PER_UNIT * zoom,
                         planned_frac=plen * PX_PER_UNIT * zoom / IMG_H,
                         zoom_clamped=clamped, bbox_px=tuple(float(v) for v in bbox))


def plan_zoom_views(conn_ins: Optional[ConnectorInsertion], view_rotations: Mapping[str, np.ndarray],
                    cfg: ConnectorInsertionConfig, set_id: int, frame_id: int,
                    pcl: Optional[np.ndarray] = None, seg: Optional[np.ndarray] = None
                    ) -> list[ZoomPlan]:
    """Plan ``cfg.zoom_views`` close-ups. Returns [] when disabled or nothing to frame.

    Each view draws a fixed number of uniforms plus one sub-seed from the private
    stream, so changing a knob never reshuffles the other views. Slots are chosen
    without repetition while unused slots remain.
    """
    if not cfg.enabled or cfg.zoom_views <= 0:
        return []
    rs = _local_rng(set_id, frame_id, cfg)
    names = [n for n in view_rotations if n not in cfg.zoom_exclude_views]
    if not names:
        return []
    inserted = list(conn_ins.inserted) if conn_ins is not None else []
    plans: list[ZoomPlan] = []
    used: set[int] = set()
    for j in range(cfg.zoom_views):
        u = rs.uniform(size=_N_DRAWS)
        sub = int(rs.randint(0, 2 ** 31 - 1))
        lrs = np.random.RandomState(sub)
        if not inserted:
            if cfg.zoom_fallback != "wire" or pcl is None or seg is None:
                continue
            w = np.where(np.asarray(seg) == 0)[0]
            if not len(w):
                continue
            R = np.asarray(view_rotations[names[int(u[4] * len(names)) % len(names)]], float)
            p = np.asarray(pcl, float)[w[int(u[0] * len(w)) % len(w)]]
            zoom = float(np.exp(np.log(cfg.zoom_min) + u[1] * (np.log(cfg.zoom_max) - np.log(cfg.zoom_min))))
            c2 = project(p, R)[:2]
            cand = ZoomCandidate(view=names[int(u[4] * len(names)) % len(names)], R=R, perturbed=False,
                                 centre=c2, zoom=zoom, target_frac=None, proj_len_1x_px=0.0,
                                 planned_len_px=0.0, planned_frac=0.0, zoom_clamped=False,
                                 bbox_px=(0.0, 0.0, 0.0, 0.0))
            plans.append(ZoomPlan(j, -1, "", (cand,)))
            continue
        n = len(inserted)
        k = int(u[0] * n) % n
        for off in range(n):
            if (k + off) % n not in used:
                k = (k + off) % n
                break
        used.add(k)
        if len(used) == n:
            used.clear()
        isl = inserted[k]
        pts = np.asarray(conn_ins.extra_pts[isl.start:isl.stop], dtype=np.float64)
        frac = cfg.zoom_target_frac_min + u[1] * (cfg.zoom_target_frac_max - cfg.zoom_target_frac_min)
        jit = 2.0 * u[2:4] - 1.0
        # foreshortening reference: best projected length over the canonical views
        lens = {nm: projected_extent(project(pts, np.asarray(view_rotations[nm]))[:, :2])[0]
                for nm in names}
        best = max(lens.values())
        order = [names[i] for i in lrs.permutation(len(names))]
        ok = [nm for nm in order if lens[nm] >= cfg.zoom_min_projected_frac * best]
        if not ok:
            ok = [max(lens, key=lens.get)]
        cands = []
        for nm in ok[:cfg.zoom_max_attempts]:
            R0 = np.asarray(view_rotations[nm], dtype=np.float64)
            R, pert = (perturb_rotation(R0, cfg.zoom_perturb_deg, lrs), True) \
                if cfg.zoom_rotation_mode == "perturb" else (R0, False)
            cands.append(_candidate(isl, pts, nm, R, pert, float(frac), jit, cfg))
        plans.append(ZoomPlan(j, k, isl.asset.name, tuple(cands)))
    return plans


# ----------------------------------------------------------------------------
# wire tube
# ----------------------------------------------------------------------------
class WireTube:
    """Smooth hollow tube along the skeleton that carries harness wire points.

    Built once per source: the skeleton chains (``segments`` between structural
    nodes) are Laplacian-smoothed so bends do not open slits on the outer side,
    and a radius is estimated per node from the spread of the wire points around
    the raw skeleton. Per zoom view only the window is sampled: rings of points on
    the camera-facing half of the tube, spaced ``step_px`` screen pixels apart.
    Label and colour come from the nearest harness wire-like points of the render
    cloud (already lit): label of the nearest one, colour inverse-distance
    weighted over the 4 nearest so the mosaic of sparse source points does not
    show at high zoom.
    """

    def __init__(self, pcl, seg, na, nb, nodes, segments, wire_xyz, wire_rgb, wire_lab,
                 smooth_radius: float = 0.05, pct: float = 80.0, smooth_iters: int = 4):
        from scipy.spatial import cKDTree
        pcl = np.asarray(pcl, dtype=np.float64)
        seg = np.asarray(seg).astype(int)
        nodes = np.asarray(nodes, dtype=np.float64)
        m = np.isin(seg, (0, 1, 2))
        a, b = np.asarray(na)[m].astype(int), np.asarray(nb)[m].astype(int)
        ev = nodes[b] - nodes[a]
        t = np.clip(((pcl[m] - nodes[a]) * ev).sum(1) / np.maximum((ev ** 2).sum(1), 1e-12), 0, 1)
        dist = np.linalg.norm(pcl[m] - (nodes[a] + t[:, None] * ev), axis=1)
        mtree = cKDTree(pcl[m])
        # radius per node (NaN where no wire point is near, e.g. inside a connector)
        node_r = np.full(len(nodes), np.nan)
        # members bound to an edge that touches the node (a crossing strand nearby does not count)
        order_ = np.argsort(np.r_[a, b], kind="stable")
        both = np.r_[np.arange(len(a)), np.arange(len(a))][order_]
        bounds = np.searchsorted(np.r_[a, b][order_], np.arange(len(nodes) + 1))
        for i in range(len(nodes)):
            mem = both[bounds[i]:bounds[i + 1]]
            if len(mem) >= 3:
                node_r[i] = np.percentile(dist[mem], pct)
        fill = float(np.nanmedian(node_r)) if np.isfinite(node_r).any() else 4.0 / PX_PER_UNIT
        node_r = np.where(np.isfinite(node_r), node_r, fill)
        node_r = np.maximum(node_r, 0.5 / PX_PER_UNIT)
        edge_ok = {(int(min(x, y)), int(max(x, y))) for x, y in zip(a, b)}
        ek = lambda u_, v_: (int(min(u_, v_)), int(max(u_, v_)))
        for ch in segments:                      # fill single member-less edges inside a wire run
            for i in range(1, len(ch) - 2):
                if ek(ch[i], ch[i + 1]) not in edge_ok and ek(ch[i - 1], ch[i]) in edge_ok \
                        and ek(ch[i + 1], ch[i + 2]) in edge_ok:
                    edge_ok.add(ek(ch[i], ch[i + 1]))
        EA, EB, TA, TB, RA, RB = [], [], [], [], [], []
        ends: dict[int, int] = {}
        for chain in segments:
            for c_ in (chain[0], chain[-1]):
                ends[int(c_)] = ends.get(int(c_), 0) + 1
        CP, CT, CR = [], [], []          # free wire ends: cap centre, outward tangent, radius
        for chain in segments:
            chain = [int(c) for c in chain]
            P = nodes[chain].copy()
            for _ in range(smooth_iters if len(chain) >= 3 else 0):
                P[1:-1] = 0.5 * P[1:-1] + 0.25 * (P[:-2] + P[2:])
            T = np.gradient(P, axis=0) if len(chain) >= 2 else np.zeros_like(P)
            T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
            for i in range(len(chain) - 1):
                if (min(chain[i], chain[i + 1]), max(chain[i], chain[i + 1])) not in edge_ok:
                    continue
                ok_prev = i > 0 and (min(chain[i - 1], chain[i]), max(chain[i - 1], chain[i])) in edge_ok
                ok_next = (i + 2 < len(chain)
                           and (min(chain[i + 1], chain[i + 2]), max(chain[i + 1], chain[i + 2])) in edge_ok)
                # cap where the tube stops: a free wire end, or where the next edge carries
                # no wire points (the inserted connector takes over there)
                if (i == 0 and ends.get(chain[0], 0) == 1) or (i > 0 and not ok_prev):
                    d_ = P[i] - P[i + 1]
                    CP.append(P[i]); CT.append(d_ / max(np.linalg.norm(d_), 1e-12)); CR.append(node_r[chain[i]])
                if (i == len(chain) - 2 and ends.get(chain[-1], 0) == 1) or (i + 2 < len(chain) and not ok_next):
                    d_ = P[i + 1] - P[i]
                    CP.append(P[i + 1]); CT.append(d_ / max(np.linalg.norm(d_), 1e-12)); CR.append(node_r[chain[i + 1]])
                EA.append(P[i]); EB.append(P[i + 1]); TA.append(T[i]); TB.append(T[i + 1])
                RA.append(node_r[chain[i]]); RB.append(node_r[chain[i + 1]])
        self.EA, self.EB = np.array(EA).reshape(-1, 3), np.array(EB).reshape(-1, 3)
        self.TA, self.TB = np.array(TA).reshape(-1, 3), np.array(TB).reshape(-1, 3)
        self.RA, self.RB = np.array(RA), np.array(RB)
        self.CP, self.CT, self.CR = np.array(CP).reshape(-1, 3), np.array(CT).reshape(-1, 3), np.array(CR)
        self._tree = cKDTree(np.asarray(wire_xyz, dtype=np.float64))
        self._rgb = np.asarray(wire_rgb, dtype=np.uint8)
        self._lab = np.asarray(wire_lab, dtype=np.int64)

    @property
    def mean_radius_px(self) -> float:
        return float(np.mean(np.r_[self.RA, self.RB]) * PX_PER_UNIT) if len(self.RA) else 0.0

    def build(self, R: np.ndarray, centre: np.ndarray, zoom: float, step_px: float
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tube points (pts, bgr, labels) for the window of this zoom view."""
        s = PX_PER_UNIT * zoom
        R = np.asarray(R, dtype=np.float64)
        fwd = R[2]
        ca, cb = self.EA @ R.T, self.EB @ R.T
        rmax = np.maximum(self.RA, self.RB)[:, None]
        lo = np.minimum(ca[:, :2], cb[:, :2]) - rmax
        hi = np.maximum(ca[:, :2], cb[:, :2]) + rmax
        hw, hh = window_half_extent(zoom)
        pad = 4.0 / s
        sel = np.where((hi[:, 0] >= centre[0] - hw - pad) & (lo[:, 0] <= centre[0] + hw + pad)
                       & (hi[:, 1] >= centre[1] - hh - pad) & (lo[:, 1] <= centre[1] + hh + pad))[0]
        out = []
        for e in sel:
            L = float(np.linalg.norm(self.EB[e] - self.EA[e]))
            if L < 1e-9:
                continue
            n_s = max(int(np.ceil(L * s / step_px)) + 1, 2)
            tt = np.linspace(0.0, 1.0, n_s)[:, None]
            ctr = self.EA[e] + tt * (self.EB[e] - self.EA[e])
            tg = (1 - tt) * self.TA[e] + tt * self.TB[e]
            tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12)
            rad = (1 - tt) * self.RA[e] + tt * self.RB[e]                    # (n_s, 1)
            ref = np.where((np.abs(tg[:, 2:3]) < 0.9), np.array([[0.0, 0, 1]]), np.array([[0.0, 1, 0]]))
            u = np.cross(tg, ref); u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
            w = np.cross(tg, u)
            n_c = max(int(np.ceil(2 * np.pi * float(rad.max()) * s / step_px)), 6)
            ang = (np.arange(n_c)[None, :] + 0.5 * (np.arange(n_s)[:, None] % 2)) * (2 * np.pi / n_c)
            nrm = np.cos(ang)[..., None] * u[:, None, :] + np.sin(ang)[..., None] * w[:, None, :]
            front = (nrm @ fwd) < 0.25                                        # camera-facing half
            P = ctr[:, None, :] + rad[:, :, None] * nrm
            out.append(P[front])
        for c in range(len(self.CP)):                      # caps on free wire ends
            if float(self.CT[c] @ fwd) >= 0.25:
                continue
            cam_c = self.CP[c] @ R.T
            if (abs(cam_c[0] - centre[0]) > hw + self.CR[c] + pad
                    or abs(cam_c[1] - centre[1]) > hh + self.CR[c] + pad):
                continue
            tg = self.CT[c]
            ref = np.array([0.0, 0, 1]) if abs(tg[2]) < 0.9 else np.array([0.0, 1, 0])
            u = np.cross(tg, ref); u /= np.linalg.norm(u)
            w = np.cross(tg, u)
            m_r = max(int(np.ceil(self.CR[c] * s / step_px)), 1)
            pts_c = [self.CP[c][None, :]]
            for k_ in range(1, m_r + 1):
                rho = self.CR[c] * k_ / m_r
                n_c = max(int(np.ceil(2 * np.pi * rho * s / step_px)), 6)
                a_ = np.arange(n_c) * (2 * np.pi / n_c) + 0.5 * k_
                pts_c.append(self.CP[c] + rho * (np.cos(a_)[:, None] * u + np.sin(a_)[:, None] * w))
            out.append(np.concatenate(pts_c))
        if not out:
            return np.empty((0, 3)), np.empty((0, 3), np.uint8), np.empty(0, np.int64)
        P = np.concatenate(out)
        k = min(4, len(self._rgb))
        d, nn = self._tree.query(P, k=k)
        if k == 1:
            d, nn = d[:, None], nn[:, None]
        wgt = 1.0 / (d + 0.25 / PX_PER_UNIT)
        wgt /= wgt.sum(1, keepdims=True)
        rgb = np.rint((self._rgb[nn].astype(np.float64) * wgt[..., None]).sum(1)).astype(np.uint8)
        return P, rgb, self._lab[nn[:, 0]]


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------
@dataclass
class ZoomResult:
    index: int
    rgb: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    label: Optional[np.ndarray]
    info: dict = field(default_factory=dict)


class ZoomRenderer:
    """Renders the zoom views of one source from the standard (lit) render cloud.

    ``pts/labels/rgb`` are the standard combined cloud; the last
    ``len(conn_ins.extra_pts)`` points must be the inserted connectors.
    ``rasterize`` is ``pcl_to_rgbd.rasterize_view``; ``light(points, bgr)`` re-applies
    the scene lighting to NEW points (None when lighting is off).
    """

    def __init__(self, cfg: ConnectorInsertionConfig, conn_ins: Optional[ConnectorInsertion],
                 pts: np.ndarray, labels: np.ndarray, rgb: np.ndarray,
                 rasterize: Callable, background: Optional[np.ndarray],
                 light: Optional[Callable] = None, skeleton: Optional[Mapping[str, Any]] = None):
        self.cfg, self.ins = cfg, conn_ins
        self.pts, self.labels, self.rgb = pts, np.asarray(labels), rgb
        self.rasterize, self.background, self.light = rasterize, background, light
        n_ins = 0 if conn_ins is None else len(conn_ins.extra_pts)
        self.tail = len(pts) - n_ins
        if n_ins:
            assert np.array_equal(pts[self.tail:], conn_ins.extra_pts), "inserted points must be the cloud tail"
        self.wire_like = np.isin(self.labels, (0, 1, 2))
        self.tube: Optional[WireTube] = None
        if cfg.zoom_wire_densify and skeleton is not None and self.wire_like.any():
            wl = self.wire_like
            self.tube = WireTube(skeleton["pcl"], skeleton["seg"], skeleton["na"], skeleton["nb"],
                                 skeleton["nodes"], skeleton["segments"], pts[wl], rgb[wl],
                                 self.labels[wl])

    # -- cloud for one candidate -------------------------------------------------
    def build_cloud(self, cand: ZoomCandidate, slot: int, cull=None):
        cfg = self.cfg
        zoom = cand.zoom
        s = PX_PER_UNIT * zoom
        keep = np.ones(len(self.pts), dtype=bool)
        if cull is not None:
            keep[cull] = False
        r_bg = max(1, int(np.ceil(zoom * cfg.zoom_bg_radius_mult - 1e-9)))
        radius = np.full(len(self.pts), r_bg, dtype=np.int32)
        radius[self.labels == 4] = 1        # class-4 noise specks stay 1 px, they are not surfaces
        ex_p, ex_c, ex_l, ex_r = [], [], [], []
        info: dict[str, Any] = {"bg_radius": r_bg}
        hw, hh = window_half_extent(zoom)
        n_res = 0
        self._target = None
        if self.ins is not None:
            for k_, isl in enumerate(self.ins.inserted):
                std = np.asarray(self.ins.extra_pts[isl.start:isl.stop])
                cam = project(std, cand.R)[:, :2]
                pad = 8.0 / s
                inter = ((cam[:, 0].max() >= cand.centre[0] - hw - pad) & (cam[:, 0].min() <= cand.centre[0] + hw + pad)
                         & (cam[:, 1].max() >= cand.centre[1] - hh - pad) & (cam[:, 1].min() <= cand.centre[1] + hh + pad))
                if not inter:
                    continue
                keep[self.tail + isl.start:self.tail + isl.stop] = False
                P, C, n = resample_inserted(isl, cfg, s, cfg.zoom_connector_spacing_px, cfg.zoom_max_points)
                if self.light is not None:
                    C = self.light(P, C)
                if k_ == slot:
                    self._target = P
                ex_p.append(P); ex_c.append(C)
                ex_l.append(np.full(n, INSERTED_CONNECTOR_LABEL, dtype=np.int64))
                ex_r.append(np.full(n, cfg.zoom_connector_radius, dtype=np.int32))
                n_res += n
        info["n_connector_points"] = int(n_res)
        n_tube = 0
        if self.tube is not None:
            P, C, L = self.tube.build(cand.R, cand.centre, zoom, cfg.zoom_wire_step_px)
            keep &= ~self.wire_like
            n_tube = len(P)
            if n_tube:
                ex_p.append(P); ex_c.append(C); ex_l.append(L)
                ex_r.append(np.full(n_tube, cfg.zoom_wire_splat_radius, dtype=np.int32))
        info["n_tube_points"] = int(n_tube)
        P = np.concatenate([self.pts[keep]] + ex_p) if ex_p else self.pts[keep]
        C = np.concatenate([self.rgb[keep]] + ex_c) if ex_c else self.rgb[keep]
        L = np.concatenate([self.labels[keep]] + ex_l) if ex_l else self.labels[keep]
        Rr = np.concatenate([radius[keep]] + ex_r) if ex_r else radius[keep]
        return P, L, C, Rr, info

    def visibility(self, cand: ZoomCandidate, depth: np.ndarray) -> tuple[float, float]:
        """(unoccluded share of the target connector's pixels, share of its points in frame).

        A connector pixel is occluded when the rendered depth there is more than
        ``VIS_TOL_MM`` nearer than the connector's own front surface.
        """
        P = self._target
        if P is None or not len(P):
            return 1.0, 1.0
        cam = project(P, cand.R)
        s = PX_PER_UNIT * cand.zoom
        u = np.rint((cam[:, 0] - cand.centre[0]) * s + IMG_W / 2.0).astype(np.int64)
        v = np.rint((cam[:, 1] - cand.centre[1]) * s + IMG_H / 2.0).astype(np.int64)
        inside = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        in_frame = float(inside.mean())
        if not inside.any():
            return 0.0, in_frame
        mm = np.clip((cam[inside, 2] + 1.1) / 2.2 * 1000.0 + 500.0, 0, 65535)
        pix = v[inside] * IMG_W + u[inside]
        dmin = np.full(IMG_W * IMG_H, np.inf)
        np.minimum.at(dmin, pix, mm)
        pu = np.unique(pix)
        seen = depth.reshape(-1)[pu].astype(np.float64)
        return float((seen >= dmin[pu] - VIS_TOL_MM).mean()), in_frame

    def occluders(self, cand: ZoomCandidate) -> np.ndarray:
        """Indices of background points (floor, clutter) in the window that lie in front of the harness.

        Their enlarged splats would be drawn over the wire and connector. Reference depth is the
        nearest harness point (wire-like or inserted connector) inside the window.
        """
        s = PX_PER_UNIT * cand.zoom
        cam = project(self.pts, cand.R)
        hw, hh = window_half_extent(cand.zoom)
        pad = (int(np.ceil(cand.zoom)) + 2) / s
        win = ((np.abs(cam[:, 0] - cand.centre[0]) <= hw + pad) & (np.abs(cam[:, 1] - cand.centre[1]) <= hh + pad))
        harness = win & (self.wire_like | (self.labels == INSERTED_CONNECTOR_LABEL))
        if not harness.any():
            return np.empty(0, dtype=np.int64)
        zref = cam[harness, 2].min() - 0.004
        return np.where(win & (self.labels == 255) & (cam[:, 2] < zref))[0]

    def render_candidate(self, cand: ZoomCandidate, slot: int, cull=None):
        P, L, C, Rr, info = self.build_cloud(cand, slot, cull)
        info["culled_background_points"] = 0 if cull is None else int(len(cull))
        bg = crop_backdrop(self.background, cand.centre, cand.zoom, self.cfg.zoom_backdrop_mode)
        rgb, depth, lab = self.rasterize(P, L, cand.R, point_rgb=C, background=bg,
                                         zoom=cand.zoom, centre=tuple(cand.centre), point_radius=Rr)
        return rgb, depth, lab, info

    def render(self, plan: ZoomPlan) -> ZoomResult:
        t0 = time.time()
        best = None
        tried = []
        occ = [self.occluders(c) for c in plan.candidates]
        # prefer candidates with (almost) no background in front of the harness; the rest are
        # tried only when none is clean. Whatever still sits in front is culled from the cloud.
        idx = [i for i in range(len(occ)) if len(occ[i]) <= OCC_MAX] or list(range(len(occ)))
        for ai in idx:
            cand = plan.candidates[ai]
            rgb, depth, lab, info = self.render_candidate(cand, plan.slot, occ[ai] if len(occ[ai]) else None)
            x0, y0, x1, y1 = cand.bbox_px
            xa, ya = int(max(np.floor(x0) - 3, 0)), int(max(np.floor(y0) - 3, 0))
            xb, yb = int(min(np.ceil(x1) + 4, IMG_W)), int(min(np.ceil(y1) + 4, IMG_H))
            n_conn = int((lab[ya:yb, xa:xb] == 4).sum()) if plan.slot >= 0 else int((lab == 4).sum())
            vis, in_frame = self.visibility(cand, depth) if plan.slot >= 0 else (1.0, 1.0)
            info["visible_frac"], info["connector_in_frame_frac"] = round(vis, 4), round(in_frame, 4)
            tried.append({"view": cand.view, "occluding_bg_points": int(len(occ[ai])),
                          "connector_px_in_bbox": n_conn,
                          "visible_frac": round(vis, 4)})
            score = vis if n_conn >= self.cfg.zoom_min_connector_px else vis - 1.0
            if best is None or score > best[0]:
                best = (score, ai, rgb, depth, lab, info, n_conn)
            if plan.slot < 0 or (vis >= self.cfg.zoom_min_visible_frac
                                 and n_conn >= self.cfg.zoom_min_connector_px):
                break
        _, ai, rgb, depth, lab, info, n_conn = best
        cand = plan.candidates[ai]
        info.update(
            index=plan.index, slot=plan.slot, asset=plan.asset, view=cand.view,
            perturbed=cand.perturbed, rotation=np.asarray(cand.R).round(6).tolist(),
            centre_cam_xy=[float(v) for v in cand.centre], zoom=cand.zoom,
            target_frac=cand.target_frac, zoom_clamped=cand.zoom_clamped,
            proj_len_1x_px=cand.proj_len_1x_px, planned_len_px=cand.planned_len_px,
            planned_frac=cand.planned_frac, planned_bbox_px=list(cand.bbox_px),
            attempt=ai, attempts=tried, connector_px_in_bbox=n_conn,
            connector_px=int((lab == 4).sum()), seconds=round(time.time() - t0, 3))
        return ZoomResult(plan.index, rgb, depth, lab, info)


def zoom_view_name(j: int) -> str:
    return f"zoom{j}"
