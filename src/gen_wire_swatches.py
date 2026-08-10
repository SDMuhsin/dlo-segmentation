#!/usr/bin/env python3
"""Generate the Phase 18 solid-cable wire swatch textures.

Writes 16 deterministic 256x256 PNGs to ``data/textures/wire_swatches_p18/``
for use as EXTRA wire-segment textures via the ``KIAT_P18_WIRETEX_DIR`` knob
(see ``src/convert_to_video_dataset.py``).

WHY: the real-world GT eval (data/real_wires_valset) shows the model misses
wires by APPEARANCE — 19.9 % of real wire pixels are orange-binned (recall
0.144) and bright/saturated solid PVC jacket colours (green / purple / cyan
/ ...) are entirely absent from the 11 blurred-ambientCG-photo wire pool.
These swatches are solid-cable PVC jackets: a near-solid base colour + a
mild seamless lengthwise brightness gradient + low-amplitude blurred noise,
so they read as smooth extruded PVC rather than flat paint.

Texture orientation: the wire UV mapper (``src/texture_mapping.py``) tiles
the texture WIDTH (x) along the wire and the HEIGHT (y) around the
circumference, so the brightness gradient runs along x (and wraps
seamlessly), and the two-tone trace stripes are horizontal bands (a constant
y-range = a stripe running along the cable length, like mains flex).

Deterministic: fixed seed; re-running reproduces unchanged PNGs.

Usage:
    ./env/bin/python src/gen_wire_swatches.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "textures" / "wire_swatches_p18"

SIZE = 256          # square swatch, px
GRAD_AMP = 0.07     # lengthwise brightness gradient amplitude (fraction)
NOISE_SIGMA = 0.035 # multiplicative luma-noise sigma (fraction)
NOISE_BLUR = 1.2    # gaussian sigma (px) smoothing the noise → PVC sheen
SEED = 18

#: (name, base RGB, stripe RGB or None). Stripe = lengthwise trace band
#: covering STRIPE_BAND of the height (wraps radial_scale× around the cable).
#: RGB bases (sRGB 0-255):
SWATCHES: list[tuple[str, tuple[int, int, int], tuple[int, int, int] | None]] = [
    ("orange_pvc",        (255, 115,   0), None),  # #1 priority: PVC power-cable orange
    ("red_pvc",           (208,  34,  34), None),
    ("green_pvc",         ( 24, 145,  60), None),
    ("blue_pvc",          ( 18,  60, 200), None),
    ("midblue_pvc",       ( 72, 110, 190), None),
    ("cyan_pvc",          (  0, 168, 180), None),
    ("yellow_pvc",        (235, 200,  30), None),
    ("purple_pvc",        (122,  48, 160), None),
    ("white_pvc",         (235, 235, 232), None),
    ("lightgrey_pvc",     (192, 192, 192), None),
    ("midgrey_pvc",       (128, 128, 128), None),
    ("darkgrey_pvc",      ( 72,  72,  74), None),
    ("black_pvc",         ( 24,  24,  26), None),
    # Two-tone jackets (stripe along the cable length):
    ("tan_brownstripe",   (205, 175, 140), (110,  70,  40)),  # mains-flex
    ("grey_bluestripe",   (165, 165, 165), ( 40,  80, 185)),
    ("yellow_greenstripe",(225, 205,  40), ( 30, 140,  60)),  # earth-wire
]

STRIPE_BAND = (0.40, 0.56)  # stripe rows as a fraction of the height


def make_swatch(base_rgb, stripe_rgb, rng: np.random.Generator) -> np.ndarray:
    """Return one (SIZE, SIZE, 3) uint8 BGR swatch."""
    img = np.empty((SIZE, SIZE, 3), dtype=np.float64)
    img[:] = np.asarray(base_rgb, dtype=np.float64)[None, None, :]
    if stripe_rgb is not None:
        r0, r1 = (int(STRIPE_BAND[0] * SIZE), int(STRIPE_BAND[1] * SIZE))
        img[r0:r1] = np.asarray(stripe_rgb, dtype=np.float64)[None, None, :]

    # Mild lengthwise brightness gradient. One full sine period along the
    # width so the texture wraps seamlessly when tiled along the wire.
    x = np.arange(SIZE, dtype=np.float64) / SIZE
    grad = 1.0 + GRAD_AMP * np.sin(2.0 * np.pi * x)
    img *= grad[None, :, None]

    # Low-amplitude smoothed luma noise (same factor on all 3 channels so the
    # hue stays put — reads as PVC surface sheen, not chroma speckle).
    noise = rng.normal(0.0, NOISE_SIGMA, size=(SIZE, SIZE))
    noise = cv2.GaussianBlur(noise, ksize=(0, 0), sigmaX=NOISE_BLUR)
    img *= (1.0 + noise)[:, :, None]

    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return img[:, :, ::-1].copy()  # RGB → BGR for cv2.imwrite


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)  # one stream; SWATCHES order fixes draws
    for i, (name, base, stripe) in enumerate(SWATCHES):
        out = OUT_DIR / f"p18_{i:02d}_{name}.png"
        cv2.imwrite(str(out), make_swatch(base, stripe, rng))
        print(f"wrote {out}  base RGB={base}"
              + (f"  stripe RGB={stripe}" if stripe else ""))
    print(f"\n{len(SWATCHES)} swatches in {OUT_DIR}")


if __name__ == "__main__":
    main()
