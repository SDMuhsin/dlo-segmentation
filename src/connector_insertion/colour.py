"""Colour handling for inserted connectors (pure functions on BGR uint8)."""
from __future__ import annotations

import numpy as np


def mean_wire_bgr(point_rgb: np.ndarray, seg: np.ndarray,
                  fallback_idx: np.ndarray | None = None) -> np.ndarray:
    """Mean BGR of the harness wire-like points (classes 0,1,2), as floats.

    Uses the colours BEFORE scene lighting. Falls back to ``fallback_idx`` points
    (the replaced connector) and finally to mid-grey when no wire exists.
    """
    m = np.isin(np.asarray(seg).astype(int), (0, 1, 2))
    if m.any():
        return point_rgb[m].astype(np.float64).mean(0)
    if fallback_idx is not None and len(fallback_idx):
        return point_rgb[fallback_idx].astype(np.float64).mean(0)
    return np.full(3, 128.0)


def luminance(bgr: np.ndarray) -> np.ndarray:
    return 0.114 * bgr[:, 0] + 0.587 * bgr[:, 1] + 0.299 * bgr[:, 2]


def tint_and_jitter(colors: np.ndarray, wire_bgr: np.ndarray, wire_tint: float,
                    jitter: float, u: np.ndarray) -> np.ndarray:
    """Return the final per-point BGR uint8 colours of an inserted connector.

    ``wire_tint`` blends from the asset's native colours (0) toward the frame's
    wire colour (1). The tinted colour keeps the asset's luminance structure
    (pins, seams, shading stay visible) by scaling the wire colour with the
    per-point luminance relative to the asset mean (clipped to [0.5, 1.6]).
    ``jitter`` applies a global brightness gain U(1-j, 1+j) and a per-channel
    gain U(1-j/2, 1+j/2); ``u`` are 4 pre-drawn uniforms in [0, 1) so the draw
    count does not depend on the knob values.
    """
    c = colors.astype(np.float64)
    if wire_tint > 0.0:
        lum = luminance(c)
        ratio = np.clip(lum / max(float(lum.mean()), 1e-6), 0.5, 1.6)
        tinted = np.clip(wire_bgr[None, :] * ratio[:, None], 0, 255)
        c = (1.0 - wire_tint) * c + wire_tint * tinted
    if jitter > 0.0:
        gain = (1.0 + jitter * (2 * u[0] - 1)) * (1.0 + 0.5 * jitter * (2 * u[1:4] - 1))
        c = c * gain[None, :]
    return np.clip(np.rint(c), 0, 255).astype(np.uint8)
