"""Public entry point of the connector insertion stage.

    result = plan_connector_insertion(pcl, seg, point_rgb, cfg, rng, library)

The renderer then (a) removes ``result.drop_mask`` points from its harness
buffers and (b) appends ``extra_pts / extra_rgb / extra_labels``. Nothing is
mutated here; with the stage disabled, no library or no slots the result is an
empty no-op (``drop_mask`` all False, zero extras).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .colour import mean_wire_bgr, tint_and_jitter
from .config import ConnectorInsertionConfig
from .constants import INSERTED_CONNECTOR_LABEL, RNG_OFFSET
from .library import ConnectorAsset, ConnectorLibrary
from .placement import SlotDraws, draw_slot_randoms, place_asset
from .slots import Slot, detect_slots


@dataclass(frozen=True)
class InsertedSlot:
    """Everything needed to re-sample one inserted connector more densely.

    ``start:stop`` is this connector's range inside ``ConnectorInsertion.extra_pts``.
    The zoom views use it to find the connector's geometry and to rebuild the
    SAME connector (same pose, scale, colours) with more surface points.
    """

    start: int
    stop: int
    slot: Slot
    asset: ConnectorAsset
    draws: SlotDraws
    wire_bgr: np.ndarray
    scale: float                 # world units, canonical length 1.0 -> scale


@dataclass(frozen=True)
class ConnectorInsertion:
    """Immutable plan: what to drop from the harness and what to add."""

    drop_mask: np.ndarray        # (n_harness,) bool, original connector points replaced
    extra_pts: np.ndarray        # (M, 3) float64 world coordinates
    extra_rgb: np.ndarray        # (M, 3) uint8 BGR (NOT yet lit; lighting runs later)
    extra_labels: np.ndarray     # (M,) int64, all INSERTED_CONNECTOR_LABEL
    report: dict[str, Any] = field(default_factory=dict)
    inserted: tuple = field(default=(), compare=False, repr=False)  # tuple[InsertedSlot, ...]

    @property
    def active(self) -> bool:
        return bool(self.extra_pts.shape[0])


def empty_insertion(n_harness: int, reason: str, **report: Any) -> ConnectorInsertion:
    return ConnectorInsertion(
        drop_mask=np.zeros(n_harness, dtype=bool),
        extra_pts=np.empty((0, 3)), extra_rgb=np.empty((0, 3), dtype=np.uint8),
        extra_labels=np.empty(0, dtype=np.int64),
        report=dict(enabled=False, reason=reason, n_slots=0, n_replaced=0, slots=[], **report))


def make_rng(set_id: int, frame_id: int,
             cfg: ConnectorInsertionConfig) -> np.random.RandomState:
    """The planner's private per-source RNG (does not touch any other stream)."""
    return np.random.RandomState(
        (set_id * 1000 + frame_id + RNG_OFFSET + cfg.seed_salt) % (2**32))


def plan_connector_insertion(pcl: np.ndarray, seg: np.ndarray, point_rgb: np.ndarray,
                             cfg: ConnectorInsertionConfig, rng: np.random.RandomState,
                             library: ConnectorLibrary) -> ConnectorInsertion:
    """Plan replacement of the harness connector points by 3D connector models."""
    n = len(pcl)
    if not cfg.enabled:
        return empty_insertion(n, "disabled")
    assets, probs = library.select(cfg.include, cfg.exclude, cfg.weights)
    if not assets:
        return empty_insertion(n, "no selectable assets")
    slots = detect_slots(pcl, seg, cfg)
    if not slots:
        return empty_insertion(n, "no connector slots")

    # biggest clusters first when capping, but keep spatial order for RNG stability
    keep_ids = sorted(np.argsort([-s.n_points for s in slots], kind="stable")[:cfg.max_per_frame])
    cum = np.cumsum(probs)
    drop = np.zeros(n, dtype=bool)
    pts, rgb, rep, ins_slots = [], [], [], []
    n_done = 0
    wire_bgr = None
    for si in keep_ids:
        slot = slots[int(si)]
        draws = draw_slot_randoms(rng)             # fixed consumption per slot
        info: dict[str, Any] = dict(
            slot=int(si), n_orig_points=slot.n_points, degenerate=slot.degenerate,
            has_wire=slot.has_wire, length_px_before=slot.length * cfg.px_per_unit,
            width_px_before=slot.width * cfg.px_per_unit, replaced=False)
        rep.append(info)
        if slot.degenerate or draws.u_replace >= cfg.replace_prob:
            info["skipped"] = "degenerate" if slot.degenerate else "replace_prob"
            continue
        asset = assets[min(int(np.searchsorted(cum, draws.u_asset, side="right")), len(assets) - 1)]
        pl = place_asset(asset, slot, cfg, draws)
        if wire_bgr is None:
            wire_bgr = mean_wire_bgr(point_rgb, seg, slot.indices)
        col = tint_and_jitter(pl.colors, wire_bgr, cfg.wire_tint,
                              cfg.asset_color_jitter, np.asarray(draws.u_color))
        drop[slot.indices] = True
        ins_slots.append(InsertedSlot(n_done, n_done + len(pl.points), slot, asset, draws,
                                      wire_bgr, pl.fit.scale))
        n_done += len(pl.points)
        pts.append(pl.points); rgb.append(col)
        info.update(replaced=True, asset=asset.name, scale=pl.fit.scale,
                    target_length_px=pl.fit.target_length_px,
                    raw_length_px=pl.fit.raw_length_px, clamped=pl.fit.clamped,
                    n_points=int(len(pl.points)), roll_deg=pl.roll_deg,
                    flipped=pl.flipped, tilt_deg=pl.tilt_deg)
    if not pts:
        out = empty_insertion(n, "no slot replaced")
        out.report.update(enabled=True, n_slots=len(slots), slots=rep)
        return out
    P = np.concatenate(pts)
    return ConnectorInsertion(
        drop_mask=drop, extra_pts=P, extra_rgb=np.concatenate(rgb).astype(np.uint8),
        extra_labels=np.full(len(P), INSERTED_CONNECTOR_LABEL, dtype=np.int64),
        report=dict(enabled=True, n_slots=len(slots), n_replaced=int(len(pts)),
                    n_degenerate=int(sum(s.degenerate for s in slots)),
                    n_dropped_points=int(drop.sum()), n_inserted_points=int(len(P)),
                    slots=rep),
        inserted=tuple(ins_slots))


def resample_inserted(ins: InsertedSlot, cfg: ConnectorInsertionConfig,
                      px_per_unit: float, spacing_px: float,
                      max_points: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Rebuild one inserted connector at splat spacing ``spacing_px`` on screen.

    ``px_per_unit`` is the effective pixels per world unit of the target view
    (renderer SCALE x zoom). Pose, scale, roll and colours are those of the
    original placement; only the number of surface points changes. The library
    holds ~150k points per asset and sampling never upsamples, so the achieved
    spacing is ``max(spacing_px, library spacing at this size)``. Returns
    ``(points, bgr uint8, n_points)``.
    """
    pl = place_asset(ins.asset, ins.slot, cfg, ins.draws,
                     sampling=(px_per_unit, spacing_px, max_points))
    col = tint_and_jitter(pl.colors, ins.wire_bgr, cfg.wire_tint,
                          cfg.asset_color_jitter, np.asarray(ins.draws.u_color))
    return pl.points, col, int(len(pl.points))
