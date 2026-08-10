#!/usr/bin/env python3
"""Generate the Phase 19 busy / painterly NEGATIVE textures.

Writes 36 deterministic 640x480 PNGs to ``data/textures/busy_negatives_p19/``
for use as (a) extra 2D photo BACKDROPS via ``KIAT_P19_BUSYBG_DIR`` /
``KIAT_P19_BUSYBG_P`` and (b) extra 3D FLOOR textures via
``KIAT_P19_BUSYFLOOR_P`` (see ``src/convert_to_video_dataset.py``).

WHY: forensics on the 62-frame real GT valset attribute 69.6 % of
false-positive pixels to texture blobs on BUSY surfaces — graffiti / mural
paint strokes (camera c2: curved painted swooshes/stripes/drips read as
wire), striped terminal blocks + red device bodies (c3), socket-panel
details and shadow-traced granite (c4). The synthetic pool has only 11 calm
photo backdrops, so the model has never seen a busy painterly surface
labelled background — and never a wire-LIKE curved stroke that is not a
wire. These textures supply exactly that negative pressure; they are only
ever composited as 2D backdrop / floor texture, so every pixel they touch is
labelled background by construction.

EDGE-SHARPNESS CONTRACT (P19b fix — see
``results/realism_campaign/p19_mechanism_probe/report.md``): the first
revision drew at 512x512 with a final per-texture GaussianBlur(σ0.5–0.8) and
was upscaled 512→640 (INTER_LINEAR) at composite time, landing busy-stroke
edges at median transition width 2.34 px vs 1.08 px for the splatted wires.
The model adopted "edge width ≲1.3 px = wire, softer = painted stroke" and
categorically rejected REAL wires (1.4–2.5 px). Therefore:

* textures are generated at NATIVE composite resolution 640x480
  (``pcl_to_rgbd.IMG_W/IMG_H``) so ``_get_busy_backdrop_library`` takes the
  no-resize branch — zero resampling between this file and the frame;
* there is NO final per-texture blur — ``cv2.LINE_AA`` / bilinear-rotation
  anti-aliasing alone puts stroke edges at ~1 px, matching the point-splat
  wire edges (1.02–1.08 px). Total extra Gaussian budget: σ ≤ 0.45 px (we
  spend 0). ``fam_shadowband``'s band-mask blur (σ15–40) is deliberate broad
  soft shadow, not a stroke edge, and stays.

Acceptance gate: busy-stroke edge transition width median in [0.95, 1.35] px
measured ON RENDERED busy-backdrop frames with the probe estimator
(contrast / max |dI/dt| along the edge normal).

Six families, 6 textures each, base lightness ramped dark -> light inside
each family:

  a. strokes     — painterly stroke fields ("fake murals"): many overlapping
                   curved cubic-Bezier strokes, width 2-30 px, cable-like
                   colours (orange/blue/red/grey/black/white) favoured.
  b. splatter    — Pollock-style splatters: blobs, tapered drip lines, spray.
  c. stripes     — stripe / grid panels (parallel stripes 4-20 px, checkers,
                   terminal-block-like separated bands, red device panel).
  d. speckle     — high-frequency granite / terrazzo noise.
  e. collage     — mixed: colour patches + strokes + splatter.
  f. shadowband  — soft illumination gradient + broad blurred dark diagonal
                   bands on mid-tone bases (shadow-tracing antidote).

No text, no real-photo dependencies — pure numpy/cv2 procedural.
Deterministic: fixed seed; re-running reproduces unchanged PNGs.

Usage:
    ./env/bin/python src/gen_busy_negative_textures.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "textures" / "busy_negatives_p19"

# Native composite resolution (pcl_to_rgbd.IMG_W / IMG_H): the 2D-backdrop
# path uses these textures pixel-for-pixel, the floor path samples them
# parametrically (any size works). MUST stay (640, 480) — any other size
# re-introduces a resample (and with it an edge-width shift) at composite.
W = 640             # texture width, px
H = 480             # texture height, px
N_PER_FAMILY = 6
SEED = 19

#: Cable-jacket-like stroke colours (RGB 0-255) — the anti-mural weapon: the
#: model must see wire-COLOURED curved strokes that are NOT wires.
CABLE_RGB: list[tuple[int, int, int]] = [
    (255, 115, 0),    # orange — the #1 real-world confusion bin
    (18, 60, 200),    # blue
    (208, 34, 34),    # red
    (128, 128, 128),  # grey
    (24, 24, 26),     # black
    (235, 235, 232),  # white
]
#: Generic mural / paint colours (RGB 0-255).
MURAL_RGB: list[tuple[int, int, int]] = [
    (24, 145, 60),    # green
    (235, 200, 30),   # yellow
    (122, 48, 160),   # purple
    (0, 168, 180),    # cyan
    (230, 70, 140),   # pink
    (110, 70, 40),    # brown
    (250, 250, 245),  # near-white
    (15, 15, 18),     # near-black
]


def _base_u8(rng: np.random.Generator, t: float,
             sat_max: float = 60.0) -> np.ndarray:
    """Plain low-saturation base canvas; lightness ramps dark -> light in t."""
    val = 45.0 + float(t) * 170.0           # V in [45, 215]
    hue = rng.uniform(0.0, 180.0)           # cv2 hue range
    sat = rng.uniform(8.0, sat_max)
    hsv = np.full((H, W, 3), (hue, sat, val), dtype=np.float32)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _pick_bgr(rng: np.random.Generator,
              p_cable: float = 0.6) -> tuple[int, int, int]:
    """One paint colour (BGR ints) with mild jitter; cable colours favoured."""
    pool = CABLE_RGB if rng.random() < p_cable else MURAL_RGB
    r, g, b = pool[int(rng.integers(0, len(pool)))]
    jit = rng.normal(0.0, 10.0, 3)
    return tuple(int(np.clip(c + j, 0, 255))
                 for c, j in zip((b, g, r), jit))


def _xy(rng: np.random.Generator) -> tuple[int, int]:
    """One uniform pixel position (x, y) on the canvas."""
    return int(rng.integers(0, W)), int(rng.integers(0, H))


def _bezier(rng: np.random.Generator, n_samples: int = 70) -> np.ndarray:
    """Random smooth cubic Bezier polyline roaming across (and past) the tile."""
    ctrl = rng.uniform((-0.15 * W, -0.15 * H), (1.15 * W, 1.15 * H),
                       size=(4, 2))
    t = np.linspace(0.0, 1.0, n_samples)[:, None]
    bern = np.concatenate(
        [(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3],
        axis=1)                              # (n, 4)
    return (bern @ ctrl).astype(np.int32)    # (n, 2) xy


def _draw_stroke(img: np.ndarray, rng: np.random.Generator,
                 width_range: tuple[int, int] = (2, 30),
                 p_cable: float = 0.6) -> None:
    """One curved painted stroke: wire-LIKE shape, flat paint appearance."""
    pts = _bezier(rng)
    w = int(rng.integers(width_range[0], width_range[1] + 1))
    cv2.polylines(img, [pts], False, _pick_bgr(rng, p_cable), w, cv2.LINE_AA)


def _spray(img: np.ndarray, rng: np.random.Generator, n: int,
           r_hi: int = 3, p_cable: float = 0.5) -> None:
    for _ in range(n):
        cv2.circle(img, _xy(rng), int(rng.integers(1, r_hi + 1)),
                   _pick_bgr(rng, p_cable), -1, cv2.LINE_AA)


# ── Families ────────────────────────────────────────────────────────────────

def fam_strokes(rng: np.random.Generator, t: float) -> np.ndarray:
    """(a) Painterly stroke field — many overlapping curved strokes."""
    img = _base_u8(rng, t)
    for _ in range(int(rng.integers(45, 90))):
        _draw_stroke(img, rng, (2, 30), p_cable=0.65)
    return img


def fam_splatter(rng: np.random.Generator, t: float) -> np.ndarray:
    """(b) Pollock-style splatter: blobs, tapered drips, spray dots."""
    img = _base_u8(rng, t)
    for _ in range(int(rng.integers(8, 16))):
        col = _pick_bgr(rng, 0.5)
        cx, cy = _xy(rng)
        ax, ay = (int(v) for v in rng.integers(8, 46, 2))
        cv2.ellipse(img, (cx, cy), (ax, ay), float(rng.uniform(0, 180)),
                    0, 360, col, -1, cv2.LINE_AA)
        # 0-3 paint drips running straight down from the blob, tapering.
        for _ in range(int(rng.integers(0, 4))):
            dx = int(np.clip(cx + rng.integers(-ax, ax + 1), 0, W - 1))
            length = int(rng.integers(30, 200))
            w_top = int(rng.integers(3, 8))
            n_seg = 4
            for s in range(n_seg):
                ya = cy + length * s // n_seg
                yb = cy + length * (s + 1) // n_seg
                w_seg = max(1, int(round(w_top * (1.0 - s / n_seg))))
                cv2.line(img, (dx, ya), (dx, yb), col, w_seg, cv2.LINE_AA)
            cv2.circle(img, (dx, cy + length), max(2, w_top // 2),
                       col, -1, cv2.LINE_AA)
    _spray(img, rng, int(rng.integers(150, 400)))
    return img


def fam_stripes(rng: np.random.Generator, t: float, kind: str) -> np.ndarray:
    """(c) Stripe / grid panels — terminal-block / device-panel-like."""
    img = _base_u8(rng, t)
    if kind in ("h", "v", "diag", "terminal", "redpanel"):
        # Striped pattern on an oversized canvas, optionally rotated, cropped.
        big = int(max(W, H) * 1.6)
        pat = np.zeros((big, big, 3), np.uint8)
        if kind == "terminal":
            # Regular alternating bands + thin dark separator lines.
            cols = [_pick_bgr(rng, 0.6), _pick_bgr(rng, 0.6)]
            sep = (28, 28, 30)
            sw = int(rng.integers(8, 21))
            x, i = 0, 0
            while x < big:
                pat[:, x:x + sw] = cols[i % 2]
                pat[:, x + sw:x + sw + 2] = sep
                x += sw + 2
                i += 1
        elif kind == "redpanel":
            # Red-dominant device panel: red bands of varying brightness.
            reds = [(34, 34, 208), (20, 20, 150), (60, 60, 235),
                    (28, 28, 30), (235, 235, 232)]
            x = 0
            while x < big:
                sw = int(rng.integers(4, 21))
                pat[:, x:x + sw] = reds[int(rng.integers(0, len(reds)))]
                x += sw
        else:
            n_col = int(rng.integers(2, 5))
            cols = [_pick_bgr(rng, 0.55) for _ in range(n_col)]
            x, i = 0, 0
            while x < big:
                sw = int(rng.integers(4, 21))
                pat[:, x:x + sw] = cols[i % n_col]
                x += sw
                i += 1
        if kind == "h":
            pat = np.transpose(pat, (1, 0, 2)).copy()
        elif kind == "diag":
            ang = float(rng.uniform(15.0, 75.0))
            M = cv2.getRotationMatrix2D((big / 2.0, big / 2.0), ang, 1.0)
            pat = cv2.warpAffine(pat, M, (big, big))
        offy = (big - H) // 2
        offx = (big - W) // 2
        img = pat[offy:offy + H, offx:offx + W].copy()
    else:  # checker
        cell = int(rng.integers(16, 65))
        c1, c2 = _pick_bgr(rng, 0.55), _pick_bgr(rng, 0.55)
        yy, xx = np.mgrid[0:H, 0:W]
        m = ((xx // cell + yy // cell) % 2).astype(bool)
        img[m] = c1
        img[~m] = c2
    return img


def fam_speckle(rng: np.random.Generator, t: float,
                terrazzo: bool) -> np.ndarray:
    """(d) High-frequency noise / speckle — granite / terrazzo-like."""
    img = _base_u8(rng, t).astype(np.float32)
    img += rng.normal(0.0, 16.0, img.shape).astype(np.float32)  # fine grain
    img = np.clip(img, 0, 255).astype(np.uint8)
    if terrazzo:
        # Larger multi-colour chips on the base.
        for _ in range(int(rng.integers(350, 700))):
            ax, ay = (int(v) for v in rng.integers(2, 9, 2))
            cv2.ellipse(img, _xy(rng), (ax, ay), float(rng.uniform(0, 180)),
                        0, 360, _pick_bgr(rng, 0.4), -1, cv2.LINE_AA)
    else:
        # Granite: dense 1-px dark/light flecks (luma-only, vectorised).
        n = int(rng.integers(40000, 70000))
        xs = rng.integers(0, W, n)
        ys = rng.integers(0, H, n)
        shade = rng.uniform(-90.0, 90.0, n).astype(np.float32)
        f = img.astype(np.float32)
        f[ys, xs] = np.clip(f[ys, xs] + shade[:, None], 0, 255)
        img = f.astype(np.uint8)
    return img


def fam_collage(rng: np.random.Generator, t: float) -> np.ndarray:
    """(e) Mixed collage: colour patches + strokes + splatter dots."""
    img = _base_u8(rng, t)
    for _ in range(int(rng.integers(4, 9))):
        col = _pick_bgr(rng, 0.5)
        x0 = int(rng.integers(0, W - 40))
        y0 = int(rng.integers(0, H - 40))
        w, h = (int(v) for v in rng.integers(60, 240, 2))
        cv2.rectangle(img, (x0, y0),
                      (min(W - 1, x0 + w), min(H - 1, y0 + h)),
                      col, -1)
    for _ in range(int(rng.integers(15, 35))):
        _draw_stroke(img, rng, (2, 24), p_cable=0.65)
    _spray(img, rng, int(rng.integers(60, 150)))
    return img


def fam_shadowband(rng: np.random.Generator, t: float) -> np.ndarray:
    """(f) Soft illumination gradient + broad blurred dark diagonal bands.

    The band-mask GaussianBlur (σ15-40) is INTENTIONAL: these are broad soft
    shadows (transition tens of px, max gradient far below any edge-anchor
    threshold), not stroke edges — they don't interact with the edge-width
    contract above.
    """
    img = _base_u8(rng, t, sat_max=45.0).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = rng.uniform(0.0, 2.0 * np.pi)
    g = np.cos(ang) * xx + np.sin(ang) * yy
    g = (g - g.min()) / max(g.max() - g.min(), 1e-9)
    img *= (0.85 + 0.30 * g)[..., None]
    for _ in range(int(rng.integers(1, 4))):
        theta = rng.uniform(0.0, np.pi)
        d = (np.cos(theta) * xx + np.sin(theta) * yy
             - rng.uniform(-0.2, 1.2) * max(W, H))
        band = (np.abs(d) < rng.uniform(30.0, 100.0)).astype(np.float32)
        band = cv2.GaussianBlur(band, (0, 0), float(rng.uniform(15.0, 40.0)))
        depth_f = rng.uniform(0.35, 0.60)   # multiplier at band core
        img *= (1.0 - band * (1.0 - depth_f))[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


# ── Driver ──────────────────────────────────────────────────────────────────

#: (family_name, callable(rng, t, j) -> img). j = in-family index 0..5; the
#: base lightness ramps dark -> light with t = j / 5 inside every family.
STRIPE_KINDS = ["v", "h", "diag", "checker", "terminal", "redpanel"]
FAMILIES: list[tuple[str, object]] = [
    ("strokes", lambda rng, t, j: fam_strokes(rng, t)),
    ("splatter", lambda rng, t, j: fam_splatter(rng, t)),
    ("stripes", lambda rng, t, j: fam_stripes(rng, t, STRIPE_KINDS[j])),
    ("speckle", lambda rng, t, j: fam_speckle(rng, t, terrazzo=(j % 2 == 1))),
    ("collage", lambda rng, t, j: fam_collage(rng, t)),
    ("shadowband", lambda rng, t, j: fam_shadowband(rng, t)),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)  # one stream; fixed loop order = draws
    idx = 0
    for fam_name, fn in FAMILIES:
        for j in range(N_PER_FAMILY):
            t = j / (N_PER_FAMILY - 1)
            img = fn(rng, t, j)
            assert img.shape == (H, W, 3) and img.dtype == np.uint8
            out = OUT_DIR / f"p19_{idx:02d}_{fam_name}_{j}.png"
            cv2.imwrite(str(out), img)
            print(f"wrote {out.name}  (family={fam_name}, lightness t={t:.1f})")
            idx += 1
    print(f"\n{idx} busy negative textures in {OUT_DIR}")


if __name__ == "__main__":
    main()
