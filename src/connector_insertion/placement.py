"""Pure geometry: fit a canonical connector asset into a slot.

Convention (see ``library.py``): the asset's CABLE end is at canonical -X and
its mating end at +X. A slot's ``axis`` points from the connector toward the
attached wire, so canonical -X is mapped onto ``+axis`` (cable end faces the
wire) and canonical +X onto ``-axis`` (mating end faces away).

The cable-side face is anchored at the slot's wire end (centre + axis * length / 2,
optionally pushed ``wire_overlap_frac`` of its length into the wire, and
laterally snapped to the attached wire's centreline when one was found). With
fit_mode=length and scale_mult=1 the asset therefore spans exactly the slot's
axial extent. Larger scales grow away from the wire, so the wire never detaches.

All randomness arrives pre-drawn in :class:`SlotDraws`, so these functions are
deterministic and the number of RNG values consumed per slot never depends on
knob values (changing ``roll_mode`` must not reshuffle later slots).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ConnectorInsertionConfig
from .library import ConnectorAsset
from .slots import Slot

_N_UNIFORMS = 12


@dataclass(frozen=True)
class SlotDraws:
    """Every random number one slot may need, drawn up front in a fixed order."""

    u_replace: float
    u_asset: float
    u_jitter: float
    u_roll: float
    u_tilt_dir: float
    u_tilt_mag: float
    u_flip: float
    u_color: tuple[float, float, float, float]
    sub_seed: int


def draw_slot_randoms(rng: np.random.RandomState) -> SlotDraws:
    """Consume exactly 12 uniforms + 1 integer from ``rng``."""
    u = rng.uniform(size=_N_UNIFORMS)
    seed = int(rng.randint(0, 2**31 - 1))
    return SlotDraws(u[0], u[1], u[2], u[3], u[4], u[5], u[6], tuple(u[7:11]), seed)


@dataclass(frozen=True)
class Fit:
    scale: float            # world units: canonical length 1.0 -> scale
    target_length_px: float  # final connector length in px-equivalents (after clamp)
    raw_length_px: float     # before the min/max clamp
    clamped: bool


def fit_scale(asset: ConnectorAsset, slot_length: float, slot_width: float,
              cfg: ConnectorInsertionConfig, u_jitter: float) -> Fit:
    """World-unit scale of the asset for this slot under ``cfg.fit_mode``.

    length: asset length = slot length. width: asset width = slot width.
    min / max: the smaller / larger of those two scales. geomean: their geometric mean.
    Then x scale_mult x (1 + jitter * U(-1, 1)), then clamp to [min, max]_length_px.
    """
    s_len = slot_length
    s_wid = slot_width / max(asset.width, 1e-6)
    s = {"length": s_len, "width": s_wid, "min": min(s_len, s_wid),
         "max": max(s_len, s_wid), "geomean": float(np.sqrt(s_len * s_wid))}[cfg.fit_mode]
    s *= cfg.scale_mult * (1.0 + cfg.scale_jitter * (2.0 * u_jitter - 1.0))
    raw_px = s * cfg.px_per_unit
    px = float(np.clip(raw_px, cfg.min_length_px, cfg.max_length_px))
    return Fit(scale=px / cfg.px_per_unit, target_length_px=px, raw_length_px=raw_px,
               clamped=bool(px != raw_px))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _perp_basis(axis: np.ndarray, hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal (m, n) perpendicular to ``axis``; m follows ``hint`` if possible."""
    m = hint - axis * float(hint @ axis)
    if np.linalg.norm(m) < 1e-6:
        ref = np.array([0.0, 0, 1]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
        m = ref - axis * float(ref @ axis)
    m = _unit(m)
    return m, np.cross(axis, m)


def _rodrigues(k: np.ndarray, ang: float) -> np.ndarray:
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def n_points_for(asset: ConnectorAsset, scale: float, cfg: ConnectorInsertionConfig,
                 sampling: "tuple[float, float, int] | None" = None) -> int:
    """Points to sample so splat spacing is ``spacing_px`` at the final size.

    ``sampling`` = (px_per_unit, spacing_px, max_points) overrides the config
    values (zoom views sample at ``px_per_unit * zoom``); None keeps the config.
    """
    ppu, spacing, cap = sampling if sampling else (cfg.px_per_unit, cfg.spacing_px, cfg.max_points)
    area_px2 = asset.area * (scale * ppu) ** 2
    n = int(round(area_px2 / spacing ** 2))
    return int(np.clip(n, min(cfg.min_points, cap), cap))


@dataclass(frozen=True)
class Placement:
    points: np.ndarray    # (n, 3) world coordinates
    normals: np.ndarray   # (n, 3) world unit normals
    colors: np.ndarray    # (n, 3) uint8 BGR, native asset colours (subset)
    fit: Fit
    roll_deg: float
    flipped: bool
    tilt_deg: float


def place_asset(asset: ConnectorAsset, slot: Slot, cfg: ConnectorInsertionConfig,
                draws: SlotDraws,
                sampling: "tuple[float, float, int] | None" = None) -> Placement:
    """Rigid + uniform-scale transform of ``asset`` into ``slot`` (inputs untouched).

    ``sampling`` only changes how many surface points are drawn (see
    :func:`n_points_for`); size, pose and the point subset prefix are identical
    to the default call, so a denser re-sampling is the same connector.
    """
    fit = fit_scale(asset, slot.length, slot.width, cfg, draws.u_jitter)
    s = fit.scale
    n = min(n_points_for(asset, s, cfg, sampling), asset.n_points)
    sel = np.random.RandomState(draws.sub_seed).choice(asset.n_points, n, replace=False)
    pc = asset.points[sel].astype(np.float64)
    nc = asset.normals[sel].astype(np.float64)
    col = asset.colors[sel]

    flipped = bool(draws.u_flip < cfg.flip_prob)
    if flipped:                                   # mirror across the XY plane
        pc = pc * np.array([1.0, 1.0, -1.0]); nc = nc * np.array([1.0, 1.0, -1.0])

    a = slot.axis
    m, nrm = _perp_basis(a, slot.minor)
    roll = {"random": 2 * np.pi * draws.u_roll,
            "fixed": np.deg2rad(cfg.roll_deg), "none": 0.0}[cfg.roll_mode]
    ycan = np.cos(roll) * m + np.sin(roll) * nrm
    xcan = -a                                      # cable end (-X) faces the wire
    R = np.stack([xcan, ycan, np.cross(xcan, ycan)], axis=1)   # columns = images of X,Y,Z

    # anchor: cable-side face centre at the wire end of the slot
    e = slot.wire_end + a * (cfg.wire_overlap_frac * s)
    if slot.attach_point is not None:
        lat = slot.attach_point - slot.wire_end
        lat = lat - a * float(lat @ a)
        cap = 0.5 * max(slot.width, 1e-6)
        ln = float(np.linalg.norm(lat))
        if ln > cap:
            lat *= cap / ln
        e = e + lat
    t = e - 0.5 * s * a

    pw = t + (R @ (s * pc).T).T
    nw = (R @ nc.T).T
    tilt = float(cfg.tilt_jitter_deg * draws.u_tilt_mag)
    if tilt > 0.0:
        phi = 2 * np.pi * draws.u_tilt_dir
        k = np.cos(phi) * m + np.sin(phi) * nrm
        T = _rodrigues(k, np.deg2rad(tilt))
        pw = e + (T @ (pw - e).T).T
        nw = (T @ nw.T).T
    return Placement(points=pw, normals=nw, colors=col, fit=fit,
                     roll_deg=float(np.rad2deg(roll)), flipped=flipped, tilt_deg=tilt)
