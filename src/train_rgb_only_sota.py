"""Train SegFormer-B5 (RGB-only) for binary DLO/background segmentation on the CDLO dataset.

Phase 7 deliverable. Side experiment to the DFormer-v2-Large RGB-D Phase 5 model.
Motivated by the Phase 5 sanity-check finding that depth is being ignored on this
synthetic dataset (real RGB + zero depth ≡ real RGB + real depth at 0.923 IoU).

Reuses the mmap cache produced by `src/train_rgbd_seg.py:build_cache` — but loads
ONLY the RGB and label arrays (depth is ignored entirely; the model never sees it).

Binary collapse (matches src/train_dformer_v2_dlo.py):
    cache label gt_transform: 0=Wire, 1=Endpoint, 2=Bifurcation, 3=Connector, 4=Noise, 255=bg
    binary: classes 0..3 -> 1 (DLO); class 4 (Noise) and 255 (bg) -> 0 (bg)

Model: nvidia/mit-b5 (ImageNet-pretrained backbone) + fresh 2-class SegFormer decode head.
Logits emerge at H/4, W/4 from the decode head and are bilinear-upsampled to (480, 640)
before loss / argmax.

Avoiding the Phase-5 NaN failure mode (AMP + lr 1e-4 + effective batch 8 → NaN at ep 16):
    - Default lr = 6e-5 (smoke-tested stable for DFormer-v2)
    - Gradient clipping at max_norm=1.0 (always on, AMP-aware via scaler.unscale_)
    - Default --eval-every=1 + save best on every IoU(DLO) improvement
    - Default --ckpt-every=5 so a restart loses ≤ 5 epochs

Usage examples (run from project root, with env activated):
    source env/bin/activate
    # Single-GPU smoke (verify wire-up):
    CUDA_VISIBLE_DEVICES=1 python src/train_rgb_only_sota.py --single-gpu --smoke a --batch-size 4
    # Single-GPU full run on GPU 1:
    CUDA_VISIBLE_DEVICES=1 python src/train_rgb_only_sota.py --single-gpu --epochs 80 --batch-size 8
    # 2-GPU DDP (when both A40s are free):
    torchrun --nproc_per_node=2 src/train_rgb_only_sota.py --epochs 80 --batch-size 4
    # Full run with the KD 2D aug stack + strong hue randomisation (Phase 18):
    CUDA_VISIBLE_DEVICES=1 python src/train_rgb_only_sota.py --single-gpu --epochs 80 \
        --batch-size 8 --aug2d --aug-hue 0.4

PROJECT_ROOT is derived from this file's location. HF_HOME is auto-set to
$PROJECT_ROOT/data/hf_cache so backbone weights cache stays in ./data per repo standards.
"""

import argparse
import datetime
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

# ─────────────────────────── PATHS / HF CACHE ───────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE = os.path.join(PROJECT_ROOT, "data", "hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)

# Reuse the cache builder from the 5-class teacher script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rgbd_seg import build_cache  # noqa: E402

from transformers import SegformerForSemanticSegmentation  # noqa: E402

# ─────────────────────────── CONFIG ───────────────────────────

DATASET_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "data", "dformer_dataset")
RESULTS_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "results", "segformer_b5_rgb")
BACKBONE_DEFAULT = "nvidia/mit-b5"

# Binary mode (Phase 7 default): {bg=0, DLO=1}.
# 3-way mode: {bg=0, wire=1, connector=2}, sourced from a re-encoded label
# cache (see src/reencode_labels_3way.py) and pulled through build_cache
# without the legacy gt_transform shift.
CLASS_NAMES_BINARY = ["bg", "DLO"]
CLASS_NAMES_THREE_WAY = ["bg", "wire", "connector"]

# Back-compat aliases for external scripts that import NUM_CLASSES /
# CLASS_NAMES from this module (gen_rgb_only_sota_gifs.py, etc.).
NUM_CLASSES = 2
CLASS_NAMES = CLASS_NAMES_BINARY
IGNORE_INDEX = -1  # nothing in the prepared label tensors is == -1

IMAGE_H, IMAGE_W = 480, 640
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


# ─────────────────── AUGMENTATIONS (flag-gated; Phase 18) ───────────────────
# Copied verbatim from src/train_rgb_only_kd.py:RGBAugmentations (the KD
# student's validated 2D stack) so the --aug2d stack matches the KD trainer
# exactly: same transforms, same params, same probability semantics.
# NOT instantiated unless --aug2d is passed — the default path never touches
# this code and consumes exactly the same RNG draws as the historical trainer.


class RGBAugmentations:
    """Per-sample 2D augmentations applied on uint8 BGR HWC arrays.

    Geometric augs (flip, random-resized-crop) apply to BOTH rgb and label.
    Appearance augs (color jitter, blur) apply ONLY to rgb.

    Order: scale jitter → flip → color jitter → blur. Operating on uint8 BGR.
    """

    def __init__(
        self,
        flip_p=0.5,
        rrc_p=0.3,
        rrc_scale=(0.8, 1.0),
        color_jitter_p=1.0,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
        blur_p=0.3,
        blur_kernels=(3, 5, 7),
        blur_sigma=(0.1, 1.5),
    ):
        self.flip_p = flip_p
        self.rrc_p = rrc_p
        self.rrc_scale = rrc_scale
        self.color_jitter_p = color_jitter_p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.blur_p = blur_p
        self.blur_kernels = blur_kernels
        self.blur_sigma = blur_sigma

    @staticmethod
    def _random_crop_resize(rgb, label, scale_range):
        h, w = rgb.shape[:2]
        scale = random.uniform(*scale_range)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        y0 = random.randint(0, h - new_h)
        x0 = random.randint(0, w - new_w)
        rgb_crop = rgb[y0:y0 + new_h, x0:x0 + new_w, :]
        label_crop = label[y0:y0 + new_h, x0:x0 + new_w]
        rgb_out = cv2.resize(rgb_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        label_out = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)
        return rgb_out, label_out

    @staticmethod
    def _apply_brightness(rgb, factor):
        return np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_contrast(rgb, factor):
        mean = rgb.reshape(-1, 3).mean(axis=0)
        return np.clip((rgb.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_saturation(rgb_bgr, factor):
        hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def _apply_hue(rgb_bgr, hue_shift):
        # hue_shift in [-0.5, 0.5]; OpenCV H channel is 0-179.
        hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + hue_shift * 180.0) % 180.0
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def __call__(self, rgb, label):
        # rgb: (H, W, 3) uint8 BGR; label: (H, W) uint8 (with gt_transform applied: 0..4, 255 bg)
        # random scale jitter
        if random.random() < self.rrc_p:
            rgb, label = self._random_crop_resize(rgb, label, self.rrc_scale)
        # horizontal flip
        if random.random() < self.flip_p:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            label = np.ascontiguousarray(label[:, ::-1])
        # color jitter — apply each component with a random factor, in a random order
        if random.random() < self.color_jitter_p:
            ops = []
            if self.brightness > 0:
                ops.append(("b", random.uniform(1 - self.brightness, 1 + self.brightness)))
            if self.contrast > 0:
                ops.append(("c", random.uniform(1 - self.contrast, 1 + self.contrast)))
            if self.saturation > 0:
                ops.append(("s", random.uniform(1 - self.saturation, 1 + self.saturation)))
            if self.hue > 0:
                ops.append(("h", random.uniform(-self.hue, self.hue)))
            random.shuffle(ops)
            for op, val in ops:
                if op == "b":
                    rgb = self._apply_brightness(rgb, val)
                elif op == "c":
                    rgb = self._apply_contrast(rgb, val)
                elif op == "s":
                    rgb = self._apply_saturation(rgb, val)
                elif op == "h":
                    rgb = self._apply_hue(rgb, val)
        # Gaussian blur
        if random.random() < self.blur_p:
            k = random.choice(self.blur_kernels)
            sigma = random.uniform(*self.blur_sigma)
            rgb = cv2.GaussianBlur(rgb, (k, k), sigma)
        return np.ascontiguousarray(rgb), np.ascontiguousarray(label)


def hue_sat_jitter(rgb_bgr, hue_max, sat_jitter=0.3):
    """Extra-strong --aug-hue randomisation on a uint8 BGR image (image-only).

    ColorJitter-style: hue shift ~ U(-hue_max, +hue_max) in torchvision units
    (0.5 = full hue wheel; OpenCV H spans 0-179 so the shift is hue*180
    H-units, wrapped) + saturation factor ~ U(1-sat_jitter, 1+sat_jitter).
    Same HSV mechanics as RGBAugmentations._apply_hue/_apply_saturation, in a
    single HSV round-trip. Motivated by real-world recall failing on cable
    colours absent from synth (bright orange, certain greys/blues).
    """
    hue_shift = random.uniform(-hue_max, hue_max)
    sat_factor = random.uniform(1.0 - sat_jitter, 1.0 + sat_jitter)
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue_shift * 180.0) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


class HeavyAugmentations:
    """Heavy domain-randomisation / sim2real stack on uint8 BGR HWC arrays.

    Flag-gated by --aug-heavy. STACKS ON TOP of --aug2d / --aug-hue (it runs
    AFTER them in __getitem__, still on uint8 BGR, before binary-collapse /
    tensor / ImageNet-normalise). Motivated by real-world precision stuck ~0.70:
    the model over-relies on local texture (≈70% of real FP pixels are
    texture-blobs), so this stack attacks texture-shortcut reliance via sensor
    noise, codec artifacts, motion blur, projective warp, occlusion (random
    erasing) and aggressive photometrics — WITHOUT adding any synthetic content
    (we proved synthetic wire-shaped negatives poison via an edge-width
    shortcut, so this is pure train-time augmentation of the existing images).

    cv2/numpy (not torchvision.transforms.v2) to match the existing
    cv2-on-BGR-uint8 pipeline: the rest of the aug code (RGBAugmentations,
    hue_sat_jitter) all operates on numpy BGR uint8 with `random.*` draws, and
    keeping the same substrate makes label-handling and RNG accounting trivial
    to reason about (no tensor<->numpy round-trips, no separate RNG stream).

    RNG: uses the global `random` module ONLY (matching RGBAugmentations);
    np.random is NOT seeded per-sample so it is deliberately avoided for the
    stochastic draws to keep the RNG audit single-stream. Where numpy random IS
    needed (Gaussian/Poisson noise, erase fill, channel permutation) we derive
    it from `random` so the whole stack is reproducible off `random.seed`.

    Geometric op (perspective warp) transforms rgb AND label identically
    (label via INTER_NEAREST, border filled with bg=255 so the legacy
    gt_transform collapse maps it to background). Random-erasing is IMAGE-ONLY
    by design: the wire is "still there" under the occlusion, so the label is
    left intact — this teaches occlusion robustness / global-structure reliance
    rather than teaching the model that occluded wire == background.

    All other ops are image-only (label untouched).

    Order (each independently gated): sensor-noise → jpeg → motion-blur →
    perspective-warp(+label) → photometric-extras (gamma/contrast/brightness/
    grayscale/channel-shuffle) → random-erasing.
    Default per-op probabilities are tuned so the COMPOSITE does not routinely
    erase thin wires (see the wire-survival guard in the smoke report).
    """

    def __init__(
        self,
        gauss_noise_p=0.4,
        gauss_sigma=(0.0, 12.0),
        poisson_p=0.2,
        jpeg_p=0.3,
        jpeg_quality=(30, 75),
        motion_blur_p=0.2,
        motion_len=(3, 15),
        perspective_p=0.3,
        perspective_distortion=(0.20, 0.30),
        erase_p=0.4,
        erase_count=(1, 3),
        erase_area=(0.02, 0.12),
        gamma_p=0.3,
        gamma_range=(0.6, 1.6),
        bc_p=0.3,
        brightness=0.35,
        contrast=0.35,
        grayscale_p=0.1,
        channel_shuffle_p=0.1,
    ):
        self.gauss_noise_p = gauss_noise_p
        self.gauss_sigma = gauss_sigma
        self.poisson_p = poisson_p
        self.jpeg_p = jpeg_p
        self.jpeg_quality = jpeg_quality
        self.motion_blur_p = motion_blur_p
        self.motion_len = motion_len
        self.perspective_p = perspective_p
        self.perspective_distortion = perspective_distortion
        self.erase_p = erase_p
        self.erase_count = erase_count
        self.erase_area = erase_area
        self.gamma_p = gamma_p
        self.gamma_range = gamma_range
        self.bc_p = bc_p
        self.brightness = brightness
        self.contrast = contrast
        self.grayscale_p = grayscale_p
        self.channel_shuffle_p = channel_shuffle_p

    # ---- helpers (all derive randomness from `random` for single-stream RNG) ----

    @staticmethod
    def _rng():
        # A numpy Generator seeded from a `random` draw so noise sampling stays
        # reproducible off the global `random.seed` (single RNG stream) without
        # touching the un-seeded global np.random state.
        return np.random.default_rng(random.getrandbits(63))

    def _gaussian_noise(self, rgb):
        sigma = random.uniform(*self.gauss_sigma)
        noise = self._rng().normal(0.0, sigma, size=rgb.shape)
        return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _poisson_noise(self, rgb):
        # Shot-noise approximation: scale to a photon count, Poisson-sample,
        # scale back. Lower `peak` => stronger noise; pick a mild-ish range.
        rng = self._rng()
        peak = random.uniform(40.0, 120.0)
        scaled = rgb.astype(np.float32) / 255.0 * peak
        noisy = rng.poisson(np.maximum(scaled, 0.0)).astype(np.float32) / peak * 255.0
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _jpeg(self, rgb):
        q = int(round(random.uniform(*self.jpeg_quality)))
        ok, enc = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return rgb
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return dec if dec is not None else rgb

    def _motion_blur(self, rgb):
        length = random.randint(*self.motion_len)
        if length % 2 == 0:
            length += 1  # odd kernel so the line passes through the centre
        angle = random.uniform(0.0, 180.0)
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0
        rot = cv2.getRotationMatrix2D((length / 2.0 - 0.5, length / 2.0 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (length, length))
        s = kernel.sum()
        if s <= 1e-6:
            return rgb
        kernel /= s
        return cv2.filter2D(rgb, -1, kernel)

    def _perspective(self, rgb, label):
        h, w = rgb.shape[:2]
        d = random.uniform(*self.perspective_distortion)
        # Jitter each corner inward/outward by up to d*dim; src is the frame
        # corners, dst is the jittered corners. Same warp matrix for rgb+label.
        def jitter(x, y):
            return [x + random.uniform(-d, d) * w, y + random.uniform(-d, d) * h]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([jitter(0, 0), jitter(w, 0), jitter(w, h), jitter(0, h)])
        M = cv2.getPerspectiveTransform(src, dst)
        rgb_w = cv2.warpPerspective(
            rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        # Label: nearest-neighbour, border filled with 255 (bg under the
        # gt_transform collapse) so warped-in margins become background.
        label_w = cv2.warpPerspective(
            label, M, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255,
        )
        return rgb_w, label_w

    def _gamma(self, rgb):
        g = random.uniform(*self.gamma_range)
        inv = 1.0 / max(g, 1e-6)
        lut = (np.power(np.linspace(0, 1, 256), inv) * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.LUT(rgb, lut)

    def _brightness_contrast(self, rgb):
        b = random.uniform(1.0 - self.brightness, 1.0 + self.brightness)
        c = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        mean = rgb.reshape(-1, 3).mean(axis=0)
        out = (rgb.astype(np.float32) * b - mean) * c + mean
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _grayscale(rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _channel_shuffle(self, rgb):
        perm = [0, 1, 2]
        random.shuffle(perm)
        return np.ascontiguousarray(rgb[:, :, perm])

    def _random_erase(self, rgb):
        # IMAGE-ONLY occlusion. 1-3 rectangles, total area in erase_area of the
        # frame, each filled with a random solid colour or the image mean.
        h, w = rgb.shape[:2]
        out = rgb.copy()
        n = random.randint(*self.erase_count)
        total_area_frac = random.uniform(*self.erase_area)
        per_area = (total_area_frac * h * w) / max(n, 1)
        mean_col = rgb.reshape(-1, 3).mean(axis=0)
        for _ in range(n):
            aspect = random.uniform(0.3, 3.3)
            eh = int(round((per_area * aspect) ** 0.5))
            ew = int(round((per_area / max(aspect, 1e-6)) ** 0.5))
            eh = max(1, min(eh, h))
            ew = max(1, min(ew, w))
            y0 = random.randint(0, h - eh)
            x0 = random.randint(0, w - ew)
            if random.random() < 0.5:
                fill = np.array(
                    [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)],
                    dtype=np.uint8,
                )
            else:
                fill = mean_col.astype(np.uint8)
            out[y0:y0 + eh, x0:x0 + ew, :] = fill
        return out

    def __call__(self, rgb, label):
        # rgb: (H, W, 3) uint8 BGR; label: (H, W) uint8 (gt_transform: 0..4, 255 bg)
        # 1. sensor noise (additive gaussian; occasional poisson/shot)
        if random.random() < self.gauss_noise_p:
            rgb = self._gaussian_noise(rgb)
        if random.random() < self.poisson_p:
            rgb = self._poisson_noise(rgb)
        # 2. jpeg artifacts
        if random.random() < self.jpeg_p:
            rgb = self._jpeg(rgb)
        # 3. directional motion blur
        if random.random() < self.motion_blur_p:
            rgb = self._motion_blur(rgb)
        # 4. perspective warp (GEOMETRIC: warps rgb AND label identically)
        if random.random() < self.perspective_p:
            rgb, label = self._perspective(rgb, label)
        # 5. photometric extras
        if random.random() < self.gamma_p:
            rgb = self._gamma(rgb)
        if random.random() < self.bc_p:
            rgb = self._brightness_contrast(rgb)
        if random.random() < self.grayscale_p:
            rgb = self._grayscale(rgb)
        if random.random() < self.channel_shuffle_p:
            rgb = self._channel_shuffle(rgb)
        # 6. random erasing / coarse dropout (IMAGE-ONLY; label untouched)
        if random.random() < self.erase_p:
            rgb = self._random_erase(rgb)
        return np.ascontiguousarray(rgb), np.ascontiguousarray(label)


class WireColorAugmentation:
    """LABEL-AWARE wire recolouring on uint8 BGR HWC arrays (Phase 24).

    Flag-gated by --aug-wirecolor. STACKS on top of --aug2d/--aug-hue/
    --aug-heavy (runs LAST in __getitem__, still on uint8 BGR, before binary
    collapse / tensor / ImageNet-normalise).

    WHY: a real-world diagnostic found recall collapses specifically on
    BRIGHT + WARM/PALE cables (white, light-grey, tan, pale-/bright-yellow) —
    missed-wire pixels are ~42%% brighter than detected ones (luma 125 vs 88)
    and cluster at warm hue H≈25; dark wires are caught fine. The existing augs
    can't fix this: --aug-hue rotates hue (preserves luma, near no-op at the low
    saturation of pale cables) and global colour-jitter brightens wire AND
    background together (and we proved scene-like background changes recreate
    the v2 over-specialisation catastrophe). So we recolour ONLY the wire
    pixels toward the missed bright/pale/warm region, leaving every background
    pixel UNCHANGED, teaching "bright/pale/warm elongated structure = wire".

    WIRE MASK: selects pixels whose label collapses to the DLO/wire class. In
    the legacy gt_transform cache that is ``label <= 3`` (0=Wire, 1=Endpoint,
    2=Bifurcation, 3=Connector; 4=Noise and 255=bg are EXCLUDED, matching the
    binary collapse in __getitem__). In 3-way label3 mode it is ``label == 1``.

    PHOTOMETRIC ONLY: geometry and the label mask are never touched. Background
    (non-wire) pixels are returned exactly equal to the input.

    RNG: global ``random`` module ONLY (matching the other aug classes). The
    per-pixel jitter is sampled from a numpy Generator seeded off a ``random``
    draw (same single-stream trick as HeavyAugmentations._rng) so the whole
    stack stays reproducible off ``random.seed`` without touching the global
    np.random state. The ``is not None`` guard in __getitem__ short-circuits
    BEFORE any draw, so the default path (--aug-wirecolor off) consumes ZERO
    extra randomness and is identical to before.

    Mechanics (vectorised over the wire-pixel region): convert the wire pixels
    to HSV, then blend H/S/V toward a per-sample target "cable look" with a
    random global strength α∈[0.5,1.0], preserving cable-like shading by keeping
    the RELATIVE V variation around the (raised) target mean rather than
    flattening to a constant, plus small per-pixel H/S/V jitter so the cable
    isn't a flat colour. Convert back to BGR, clip, composite onto the wire
    pixels only.

    Target looks (OpenCV HSV: H∈[0,179], S∈[0,255], V∈[0,255]) span the
    missed region:
      pale-white   : S→very low,  V→very high
      light-grey   : S→very low,  V→mid-high
      warm-tan     : H≈20-35,     S→low-mid, V→high
      bright-yellow: H≈28-38,     S→high,    V→high
      pale-pastel  : S→low,       V→high,    H→random warm-ish
    """

    # Public so the smoke script / external callers can request a specific look.
    TARGETS = ("pale-white", "light-grey", "warm-tan", "bright-yellow", "pale-pastel")

    def __init__(self, p=0.5, strength=(0.5, 1.0), num_classes=2):
        self.p = float(p)
        self.strength = strength
        self.num_classes = int(num_classes)

    @staticmethod
    def _rng():
        # numpy Generator seeded from a `random` draw -> reproducible off the
        # global `random.seed` (single RNG stream), no global np.random touch.
        return np.random.default_rng(random.getrandbits(63))

    def _sample_target(self, name=None):
        """Return (H_target, S_target, V_target) in OpenCV HSV units for the
        named look (or a uniformly-random one). Values are sampled within the
        look's band so repeated draws aren't identical."""
        if name is None:
            name = random.choice(self.TARGETS)
        if name == "pale-white":
            # very low saturation, very high value
            h = random.uniform(0, 179)            # hue irrelevant at S≈0
            s = random.uniform(0, 25)
            v = random.uniform(225, 255)
        elif name == "light-grey":
            # very low saturation, mid-high value
            h = random.uniform(0, 179)
            s = random.uniform(0, 22)
            v = random.uniform(150, 200)
        elif name == "warm-tan":
            # warm hue, low-mid saturation, high value.
            # Spec hues are in 0-360 deg; OpenCV H is deg/2 (0-179).
            h = random.uniform(20, 35) / 2.0
            s = random.uniform(60, 120)
            v = random.uniform(190, 235)
        elif name == "bright-yellow":
            # warm hue, high saturation, high value
            h = random.uniform(28, 38) / 2.0
            s = random.uniform(170, 240)
            v = random.uniform(210, 255)
        else:  # pale-pastel
            # low saturation, high value, random warm-ish hue
            h = random.uniform(10, 45) / 2.0
            s = random.uniform(25, 70)
            v = random.uniform(205, 250)
        return float(h), float(s), float(v), name

    def recolor(self, rgb_bgr, wire_mask, name=None, strength=None):
        """Recolour ONLY pixels where wire_mask is True toward a target look.

        rgb_bgr : (H, W, 3) uint8 BGR
        wire_mask : (H, W) bool — True on wire pixels
        Returns a NEW (H, W, 3) uint8 BGR array; non-wire pixels are copied
        byte-for-byte from the input. Pure photometric, label never read here.
        """
        out = rgb_bgr.copy()
        n = int(wire_mask.sum())
        if n == 0:
            return out  # nothing to recolour; background already identical

        h_t, s_t, v_t, _name = self._sample_target(name)
        if strength is None:
            alpha = random.uniform(*self.strength)
        else:
            alpha = float(strength)

        # Pull the wire pixels out as a compact (n, 1, 3) image so the HSV
        # round-trip is vectorised over just the wire region (cheap even for
        # large wire masks).
        wire_px = rgb_bgr[wire_mask].reshape(-1, 1, 3)               # (n,1,3) uint8 BGR
        hsv = cv2.cvtColor(wire_px, cv2.COLOR_BGR2HSV).astype(np.float32)  # (n,1,3)
        H = hsv[:, 0, 0]
        S = hsv[:, 0, 1]
        V = hsv[:, 0, 2]

        rng = self._rng()
        # Per-pixel jitter so the recoloured cable isn't a flat colour.
        h_jit = rng.uniform(-6.0, 6.0, size=n).astype(np.float32)    # OpenCV H units
        s_jit = rng.uniform(-12.0, 12.0, size=n).astype(np.float32)
        v_jit = rng.uniform(-12.0, 12.0, size=n).astype(np.float32)

        # VALUE: preserve cable-like shading by keeping the RELATIVE V variation
        # (V - mean) and shifting the mean toward the target, blended by alpha.
        # New_mean = (1-alpha)*orig_mean + alpha*v_target; keep (V-orig_mean) so
        # highlights/shadows along the cable survive.
        v_mean = float(V.mean())
        v_centered = V - v_mean
        v_new_mean = (1.0 - alpha) * v_mean + alpha * v_t
        V_out = v_new_mean + v_centered + v_jit

        # SATURATION: scale DOWN toward the (usually low) target — blend the
        # actual S toward s_target. For pale/grey looks this collapses S; for
        # bright-yellow it raises it. Keep a little of the original so texture
        # in S survives.
        S_out = (1.0 - alpha) * S + alpha * s_t + s_jit

        # HUE: set toward the target hue (meaningful only where S is non-trivial,
        # e.g. tan/yellow). Blend on the circle-free assumption that target hues
        # are all in the warm 0-40deg band so no wrap issues; wrap mod 180 anyway.
        H_out = ((1.0 - alpha) * H + alpha * h_t + h_jit) % 180.0

        hsv[:, 0, 0] = np.clip(H_out, 0, 179)
        hsv[:, 0, 1] = np.clip(S_out, 0, 255)
        hsv[:, 0, 2] = np.clip(V_out, 0, 255)
        new_px = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)  # (n,1,3) uint8
        out[wire_mask] = new_px.reshape(-1, 3)
        return out

    def _wire_mask(self, label):
        if self.num_classes == 2:
            # Pixels that collapse to DLO: classes 0..3 (Wire/Endpoint/
            # Bifurcation/Connector). Noise(4) and bg(255) are NOT wire.
            return label <= 3
        # 3-way label3: wire is class 1.
        return label == 1

    def __call__(self, rgb, label):
        # rgb: (H, W, 3) uint8 BGR; label: (H, W) uint8. Photometric only:
        # geometry + label untouched; only wire pixels change. p<1 so it fires
        # often but not always (and the single RNG draw below means the cost of
        # "off this sample" is one random.random()).
        if random.random() < self.p:
            wire_mask = self._wire_mask(label)
            rgb = self.recolor(rgb, wire_mask)
        return np.ascontiguousarray(rgb), label


class BgClutterAugmentation:
    """LABEL-AWARE BACKGROUND clutter / texture domain-randomisation on uint8
    BGR HWC arrays (Phase 25). The mirror-image of WireColorAugmentation: it
    perturbs ONLY the BACKGROUND pixels and leaves every foreground (wire) pixel
    and the label map UNCHANGED.

    Flag-gated by --aug-bgclutter. STACKS on top of --aug2d/--aug-hue/
    --aug-heavy/--aug-wirecolor (runs LAST in __getitem__, still on uint8 BGR,
    before the binary label collapse / tensor / ImageNet-normalise). Because it
    only ever writes background pixels, its order relative to --aug-wirecolor
    (which only writes wire pixels) is irrelevant — neither can overwrite the
    other — but it is placed AFTER wirecolor so wire recolours are never at
    risk.

    WHY: real-world false positives concentrate on BUSY background surfaces
    (cluttered desks, carpet, wood grain, fabric) where high local texture /
    gradient energy is mistaken for wire. Synthetic backgrounds are comparatively
    flat, so the model never learns "high-texture background ≠ wire". We inject
    ISOTROPIC, NON-WIRE-LIKE texture + photometric domain randomisation onto the
    background ONLY, raising its local gradient energy with BLOBBY, non-directional
    structure — teaching the model to reject busy texture without ever seeing a
    thin/curved/elongated (i.e. wire-like) structure painted onto background.

    CRITICAL SAFETY CONTRAST WITH PRIOR BACKGROUND LEVERS: two earlier campaign
    phases that swapped or enriched whole-scene backgrounds (2D backdrop pool
    11→80, lighting DR) caused CATASTROPHIC real-world recall collapse by
    recreating scene-specialisation. This aug is deliberately different: it does
    NOT change the *content/identity* of the background (no new scenes, no photo
    swaps) and never touches a wire pixel or the label — it only adds isotropic
    HIGH-FREQUENCY texture + global photometric jitter to the existing background
    region, a precision-only signal. And it NEVER draws strokes / lines / thin
    elongated structure (the Phase-19 poison that destroyed recall): every op is
    region-like / isotropic. validate_bgclutter_aug.py gate 4 enforces this
    quantitatively (connected-component elongation of the injected texture).

    BACKGROUND MASK: the complement of the wire mask. In the legacy gt_transform
    cache (num_classes=2) the wire/foreground is ``label <= 3`` (0=Wire,
    1=Endpoint, 2=Bifurcation, 3=Connector), so BACKGROUND = ``label > 3``
    (i.e. 4=Noise and 255=bg). In 3-way label3 mode the wire is class 1, so
    BACKGROUND = ``label != 1``. Foreground (wire) pixels are returned exactly
    equal to the input.

    RNG: global ``random`` module for the p-gate + op selection (matching the
    other aug classes); the per-pixel/per-field noise is sampled from a numpy
    Generator seeded off a single ``random`` draw (same single-stream trick as
    HeavyAugmentations._rng / WireColorAugmentation._rng) so the whole stack stays
    reproducible off ``random.seed`` without touching the global np.random state.
    The ``is not None`` guard in __getitem__ short-circuits BEFORE any draw, so
    the default path (--aug-bgclutter off) consumes ZERO extra randomness and is
    the same as before.

    Ops (a random subset fires each call; ALL ISOTROPIC — no directional kernels,
    no line/stroke drawing, no thin elongated structure):
      1. multi-scale procedural texture overlay: 2–4 octaves of value noise
         (low-res white-noise grids bicubically upsampled, geometrically-decaying
         amplitudes, summed + one light end-blur) blended onto the background
         with alpha ~U(0.15,0.6). Blobby, non-directional.
      2. localized clutter patches: a few random near-square sub-regions filled
         with stronger isotropic multi-octave texture (cluttered desk/floor
         patches) — isotropic fill, never strokes.
      3. photometric DR on the background region: random contrast, gamma,
         brightness (+ mild per-channel shift) applied to bg pixels only.

    Compositing keeps foreground bytes exact:
        out = rgb.copy(); out[bg_mask] = perturbed[bg_mask]; out[fg_mask] = rgb[...]
    """

    def __init__(self, p=0.5, strength=(0.15, 0.6), num_classes=2):
        self.p = float(p)
        # `strength` is the (min,max) blend-alpha band for the procedural
        # texture overlay (op 1) and the clutter patches (op 2). Photometric
        # jitter magnitudes are derived from the upper end of this band so the
        # whole aug scales with one knob.
        self.strength = strength
        self.num_classes = int(num_classes)

    @staticmethod
    def _rng():
        # numpy Generator seeded from a `random` draw -> reproducible off the
        # global `random.seed` (single RNG stream), no global np.random touch.
        return np.random.default_rng(random.getrandbits(63))

    def _bg_mask(self, label):
        if self.num_classes == 2:
            # Pixels that collapse to DLO are wire: classes 0..3. BACKGROUND is
            # everything else (4=Noise, 255=bg), i.e. NOT wire.
            return label > 3
        # 3-way label3: wire is class 1; background is everything else.
        return label != 1

    @staticmethod
    def _lowres_field(h, w, rng, cells_h, cells_w):
        """One isotropic value-noise octave: a (cells_h, cells_w) white-noise grid
        bicubically upsampled to (h, w). Generating the randomness at LOW
        resolution (a few hundred values) then upsampling is ~100x cheaper than a
        full-res ``standard_normal`` (which alone is ~5 ms at 480x640) while still
        isotropic — the cubic interpolation kernel is separable but rotationally
        near-symmetric, and a single light end-blur in _texture_field removes any
        residual axis ringing. Returns float32 (h, w) in ~[-1, 1]."""
        lr = rng.standard_normal((cells_h, cells_w)).astype(np.float32)
        up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
        m = float(np.abs(up).max())
        if m > 1e-6:
            up /= m
        return up

    def _texture_field(self, h, w, rng):
        """Multi-scale ISOTROPIC procedural texture: sum 2..4 octaves of low-res
        value-noise (cell count doubling per octave, geometrically decaying
        amplitude) -> blobby, non-directional structure with high local gradient
        energy and NO thin/elongated ridges. A single light full-res Gaussian
        blur at the end softens any cubic-upsample ringing. validate_bgclutter_
        aug.py gate 4 confirms the orientation-isotropy (peak ~1.5, vs ~11 for a
        wire) and ZERO thin-elongated (wire-like) components. Returns float32
        (h, w) in ~[-1, 1]. Cost is a few ms (no full-res RNG draw)."""
        octaves = int(rng.integers(2, 5))            # 2..4 octaves
        field = np.zeros((h, w), np.float32)
        amp = 1.0
        amp_sum = 0.0
        base = int(rng.integers(6, 14))              # coarsest cells (height)
        for o in range(octaves):
            ch = base * (2 ** o)
            cw = int(ch * w / h)
            field += amp * self._lowres_field(h, w, rng, min(ch, h), max(min(cw, w), 2))
            amp_sum += amp
            amp *= 0.55
        field /= max(amp_sum, 1e-6)
        field = cv2.GaussianBlur(field, (0, 0), 1.0)   # one end-blur, isotropic
        m = float(np.abs(field).max())
        if m > 1e-6:
            field /= m
        return field

    def perturb(self, rgb_bgr, bg_mask, rng=None, force_ops=None):
        """Return a NEW (H, W, 3) uint8 BGR array where ONLY the bg_mask pixels
        are perturbed; foreground pixels are copied byte-for-byte from the input.

        rgb_bgr  : (H, W, 3) uint8 BGR
        bg_mask  : (H, W) bool — True on BACKGROUND pixels
        rng      : optional np.random.Generator (else a fresh isolated one)
        force_ops: optional iterable subset of {'texture','patches','photometric'}
                   to force-run specific ops (used by the validator); default
                   None -> sample a random non-empty subset.
        """
        out = rgb_bgr.copy()
        n = int(bg_mask.sum())
        if n == 0:
            return out  # no background to perturb; foreground already identical
        if rng is None:
            rng = self._rng()

        H, W = rgb_bgr.shape[:2]
        a_lo, a_hi = self.strength

        # choose which ops fire this call (>=1). texture is the workhorse, so it
        # fires most often; patches + photometric are sampled independently.
        if force_ops is None:
            ops = set()
            if rng.random() < 0.85:
                ops.add("texture")
            if rng.random() < 0.5:
                ops.add("patches")
            if rng.random() < 0.7:
                ops.add("photometric")
            if not ops:
                ops.add("texture")
        else:
            ops = set(force_ops)

        # Work in float32 on a full-frame buffer; we only ever composite the
        # bg_mask pixels back, so touching the whole frame is safe + vectorised.
        buf = out.astype(np.float32)

        # ── op 1: multi-scale procedural texture overlay (isotropic) ──────────
        if "texture" in ops:
            alpha = float(rng.uniform(a_lo, a_hi))
            # isotropic multi-octave field (shared luminance structure) + ONE
            # shared finer field for cheap per-channel colour decorrelation, so
            # colour (not just luma) gets busy while every channel's structure
            # stays the SAME non-directional texture -> no orientation injected.
            base = self._texture_field(H, W, rng)            # (H,W) ~[-1,1]
            decor = self._lowres_field(H, W, rng,
                                       int(rng.integers(12, 30)),
                                       int(rng.integers(16, 40)))
            amp = 70.0 * alpha                               # uint8-scale swing
            for c in range(3):
                gain = 1.0 + 0.25 * float(rng.uniform(-1, 1))
                dgain = 0.15 * float(rng.uniform(-1, 1))     # per-channel colour
                buf[:, :, c] += amp * gain * (base + dgain * decor)

        # ── op 2: localized clutter patches (isotropic fill, never strokes) ───
        if "patches" in ops:
            n_patch = int(rng.integers(1, 5))               # 1..4 patches
            for _ in range(n_patch):
                # patch size 12–45% of each dim, aspect kept near-square so the
                # patch itself is region-like (NOT an elongated bar).
                pw = int(rng.uniform(0.12, 0.45) * W)
                ph = int(rng.uniform(0.12, 0.45) * H)
                pw = max(pw, 12); ph = max(ph, 12)
                x0 = int(rng.integers(0, max(W - pw, 1)))
                y0 = int(rng.integers(0, max(H - ph, 1)))
                # stronger isotropic texture inside the patch -> a visibly busier
                # sub-region (same non-directional generator as op 1).
                ptex = self._texture_field(ph, pw, rng)
                palpha = float(rng.uniform(a_lo, a_hi))
                pamp = 90.0 * palpha
                sub = buf[y0:y0 + ph, x0:x0 + pw, :]
                for c in range(3):
                    jit = 1.0 + 0.2 * float(rng.uniform(-1, 1))
                    sub[:, :, c] += pamp * jit * ptex
                buf[y0:y0 + ph, x0:x0 + pw, :] = sub

        # ── op 3: photometric DR (contrast / gamma / brightness / channel) ────
        if "photometric" in ops:
            # contrast about mid-grey, then gamma, then brightness + per-channel.
            contrast = 1.0 + (a_hi) * float(rng.uniform(-0.6, 0.6))   # ~[0.64,1.36]
            buf = (buf - 128.0) * contrast + 128.0
            gamma = float(np.exp(0.6 * a_hi * float(rng.uniform(-1, 1))))  # ~[0.7,1.4]
            buf = np.clip(buf, 0, 255)
            buf = 255.0 * np.power(buf / 255.0, gamma)
            bright = 40.0 * a_hi * float(rng.uniform(-1, 1))
            buf = buf + bright
            chan = (20.0 * a_hi) * rng.uniform(-1, 1, size=3).astype(np.float32)
            buf += chan.reshape(1, 1, 3)

        perturbed = np.clip(buf, 0, 255).astype(np.uint8)
        # composite: ONLY background pixels change; foreground bytes preserved.
        out[bg_mask] = perturbed[bg_mask]
        return out

    def __call__(self, rgb, label):
        # rgb: (H, W, 3) uint8 BGR; label: (H, W) uint8. Background-only:
        # geometry + label + all FOREGROUND (wire) pixels untouched; only
        # background pixels change. p<1 so it fires often but not always (and the
        # single RNG draw below means the cost of "off this sample" is one
        # random.random()).
        if random.random() < self.p:
            bg_mask = self._bg_mask(label)
            rgb = self.perturb(rgb, bg_mask)
        return np.ascontiguousarray(rgb), label


# ─────────────────────────── DATASET ───────────────────────────


class CDLORGBOnlyDataset(Dataset):
    """RGB-only mmap CDLO dataset.

    Two label-tensor modes:

    * ``num_classes=2`` (Phase 7 binary): cache labels arrive as
      ``{0..4, 255}`` (Wire/Endpoint/Bifurcation/Connector/Noise/bg after
      ``gt_transform``). We collapse classes ``0..3`` to 1 (DLO);
      class 4 (Noise) and 255 (bg) collapse to 0.
    * ``num_classes=3`` (Phase 11 3-way): cache labels arrive already as
      ``{0,1,2}`` straight from ``label3/`` PNGs (no ``gt_transform``).
      We pass them through unmodified.

    Depth is intentionally not loaded — Phase 7 trains the model blind to depth.

    Flag-gated 2D augs (Phase 18; both default OFF so the default path is
    bit-identical to the historical trainer, including RNG consumption):

    * ``augmenter2d`` (--aug2d): the KD-validated RGBAugmentations stack
      (scale-jitter p0.3 → h-flip p0.5 → colour-jitter p1.0 → blur p0.3).
      When set it OWNS the horizontal flip, so the legacy flip draw below is
      skipped (flip stays at p=0.5 — one flip, never two).
    * ``hue_aug`` (--aug-hue): extra-strong hue wheel randomisation, p=0.8,
      hue ±hue_aug + saturation ±0.3, image-only. Stacks with --aug2d.
    * ``heavy_aug`` (--aug-heavy): HeavyAugmentations domain-randomisation
      stack (sensor noise, jpeg, motion blur, perspective warp+label, random
      erasing image-only, photometric extras). Runs AFTER aug2d/aug-hue, still
      on uint8 BGR. When ``None`` (default) NO heavy-aug RNG draw occurs, so
      the default path is bit-identical to before.
    """

    def __init__(self, rgb, label, augment=True, include_noise=False,
                 num_classes=2, augmenter2d=None, hue_aug=0.0, heavy_aug=None,
                 wirecolor_aug=None, bgclutter_aug=None, zoom_aug=None):
        self.rgb = rgb
        self.label = label
        self.augment = augment
        self.include_noise = include_noise
        self.num_classes = int(num_classes)
        self.augmenter2d = augmenter2d
        self.hue_aug = float(hue_aug)
        self.heavy_aug = heavy_aug
        self.wirecolor_aug = wirecolor_aug
        self.bgclutter_aug = bgclutter_aug
        self.zoom_aug = zoom_aug
        if self.num_classes not in (2, 3):
            raise ValueError(f"num_classes={num_classes} not in (2, 3)")

    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, idx):
        rgb = self.rgb[idx].copy()      # (H, W, 3) uint8 BGR
        lbl = self.label[idx].copy()     # (H, W) uint8

        if self.augment:
            if self.zoom_aug is not None:
                # --aug-zoom: SCALE-ONLY augmentation (no photometric ops), run
                # FIRST so any later geometric aug composes on the zoomed frame.
                #
                # Motivation is measured, not generic: connector recall on this
                # val set collapses with apparent size (mean recall by GT-blob
                # area quartile = 0.121 / 0.195 / 0.654 / 0.942), and the same
                # model loses connector IoU 0.745 -> 0.621 when merely shown 2x
                # input, i.e. it is scale-BRITTLE. Multi-scale training is the
                # standard fix and, unlike raising input resolution, costs no
                # extra memory or batch-size change. Crop-then-resize magnifies
                # (scale 0.5 -> 2x apparent size); label uses NEAREST so class
                # boundaries stay hard.
                p, lo, hi = self.zoom_aug
                if random.random() < p:
                    rgb, lbl = RGBAugmentations._random_crop_resize(
                        rgb, lbl, (lo, hi))
            if self.augmenter2d is not None:
                # --aug2d: KD stack (scale-jitter → flip → colour-jitter →
                # blur). Geometric ops transform rgb AND lbl; photometric ops
                # touch rgb only. Runs on uint8 BGR before binary collapse,
                # tensor conversion and ImageNet normalisation.
                rgb, lbl = self.augmenter2d(rgb, lbl)
            elif random.random() > 0.5:
                # Historical default: h-flip only. This branch (and its single
                # RNG draw) is byte-for-byte the pre-Phase-18 behaviour.
                rgb = np.ascontiguousarray(rgb[:, ::-1, :])
                lbl = np.ascontiguousarray(lbl[:, ::-1])
            if self.hue_aug > 0.0 and random.random() < 0.8:
                # --aug-hue: image-only; label untouched. The hue_aug > 0.0
                # guard short-circuits BEFORE the RNG draw, so the default
                # path (hue_aug=0.0) consumes no extra randomness.
                rgb = hue_sat_jitter(rgb, self.hue_aug)
            if self.heavy_aug is not None:
                # --aug-heavy: domain-randomisation stack. Runs AFTER
                # aug2d/aug-hue, on uint8 BGR. The geometric perspective op
                # warps rgb AND lbl identically; all other ops are image-only
                # (random-erasing intentionally leaves the label intact ->
                # occlusion robustness). The `is not None` guard short-circuits
                # BEFORE any heavy-aug RNG draw, so the default path
                # (heavy_aug=None) consumes ZERO extra randomness and stays
                # identical to the pre-heavy-aug trainer.
                rgb, lbl = self.heavy_aug(rgb, lbl)
            if self.wirecolor_aug is not None:
                # --aug-wirecolor: LABEL-AWARE photometric recolour of the wire
                # pixels ONLY (background returned identical), toward the
                # bright/pale/warm "missed cable" region. Runs LAST, on uint8
                # BGR, reading `lbl` while it is still in gt_transform encoding
                # (so the wire mask is `lbl <= 3` in binary mode). Geometry and
                # the label are untouched. The `is not None` guard short-
                # circuits BEFORE any RNG draw, so the default path
                # (wirecolor_aug=None) consumes ZERO extra randomness and stays
                # unchanged from the pre-wirecolor trainer.
                rgb, lbl = self.wirecolor_aug(rgb, lbl)
            if self.bgclutter_aug is not None:
                # --aug-bgclutter: LABEL-AWARE isotropic texture + photometric
                # domain-randomisation of the BACKGROUND pixels ONLY (every wire/
                # foreground pixel returned unchanged). Runs LAST, on uint8
                # BGR, reading `lbl` while it is still in gt_transform encoding
                # (so the background mask is `lbl > 3` in binary mode). Geometry
                # and the label are untouched, and — being the mirror of
                # wirecolor (writes bg, wirecolor writes wire) — its order vs
                # wirecolor cannot disturb wire recolours. The `is not None`
                # guard short-circuits BEFORE any RNG draw, so the default path
                # (bgclutter_aug=None) consumes ZERO extra randomness and stays
                # identical to the pre-bgclutter trainer.
                rgb, lbl = self.bgclutter_aug(rgb, lbl)

        if self.num_classes == 2:
            # Legacy gt_transform cache: classes 0..3 -> 1 (DLO); class 4
            # (Noise) and 255 (bg) -> 0. include_noise flips Noise into DLO.
            if self.include_noise:
                lbl_out = (lbl <= 4).astype(np.int64)
            else:
                lbl_out = (lbl <= 3).astype(np.int64)
        else:
            # 3-way cache: already {0=bg, 1=wire, 2=connector}.
            lbl_out = lbl.astype(np.int64)

        # Convert BGR uint8 to (3, H, W) — colour order is preserved through
        # ImageNet normalisation since SegFormer was trained on RGB. The cache
        # holds BGR, so swap to RGB before returning.
        rgb_rgb = rgb[:, :, ::-1].copy()              # BGR -> RGB
        rgb_t = torch.from_numpy(rgb_rgb.transpose(2, 0, 1).copy())  # (3, H, W) uint8 RGB
        label_t = torch.from_numpy(lbl_out)            # (H, W) int64
        return {"rgb": rgb_t, "label": label_t}


# Back-compat alias for any external code that imported the old name.
BinaryCDLORGBOnlyDataset = CDLORGBOnlyDataset


def normalize_batch(rgb_uint8, device):
    rgb = rgb_uint8.to(device, dtype=torch.float32, non_blocking=True) / 255.0
    rgb = (rgb - RGB_MEAN.to(device)) / RGB_STD.to(device)
    return rgb


def file_list(data_dir, split):
    txt = os.path.join(data_dir, "train.txt" if split == "train" else "test.txt")
    with open(txt) as f:
        return [line.strip() for line in f if line.strip()]


def filter_indices_by_set(file_names, allowed_sets):
    if allowed_sets is None:
        return list(range(len(file_names)))
    allowed = set(int(s) for s in allowed_sets)
    out = []
    for i, fn in enumerate(file_names):
        basename = os.path.basename(fn)
        try:
            sid = int(basename.split("_")[0])
        except ValueError:
            continue
        if sid in allowed:
            out.append(i)
    return out


# ─────────────────────────── MODEL ───────────────────────────


def lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension w.r.t. the sorted errors (Berman 2018)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:p - 1]
    return jaccard


def lovasz_softmax(logits, labels, classes="present", ignore=IGNORE_INDEX,
                   class_weights=None):
    """Lovasz-Softmax: a smooth, direct surrogate for the Jaccard (IoU) loss.

    Why this exists (Phase 3W): weighted cross-entropy optimises per-pixel
    likelihood, which is a poor proxy for IoU on a SMALL class whose error mass
    sits on its rim. The 3-way connector is exactly that case -- measured on the
    synthetic val set, 43 % of all connector error lies within 1 px of the GT
    connector boundary, and 92 % of connector false-negatives are lost to the
    WIRE class, not to background. Lovasz optimises the ranking of per-pixel
    errors against the IoU gain each would buy, so rim pixels that CE treats as
    a handful of cheap likelihood terms are weighted by what they are actually
    worth to the metric.

    Computed in fp32 regardless of the surrounding autocast context: the sort +
    cumsum are numerically fragile in fp16.

    Args:
        logits: (B, C, H, W) raw scores, already upsampled to label resolution.
        labels: (B, H, W) int64 ground truth.
        classes: 'present' (only classes in the batch) or 'all'.
        ignore: label value to drop from the computation.
        class_weights: optional (C,) tensor. The plain mean over classes is a
            poor allocation of gradient here: background IoU is already 0.998,
            so its Lovasz term is near-zero and contributes almost nothing,
            while connector -- the class that is actually 0.24 away from
            perfect -- gets the same 1/C share as everything else. Weighting
            lets the surrogate spend its gradient on the lagging class.
    Returns:
        scalar loss, weighted mean over the contributing classes.
    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        b, c, h, w = logits.shape
        probas = F.softmax(logits, dim=1)
        # (B, C, H, W) -> (B*H*W, C)
        probas = probas.permute(0, 2, 3, 1).reshape(-1, c)
        labels = labels.reshape(-1)
        if ignore is not None:
            keep = labels != ignore
            if not keep.all():
                probas = probas[keep]
                labels = labels[keep]
        if probas.numel() == 0:
            return logits.sum() * 0.0

        losses, weights = [], []
        class_list = range(c) if classes in ("all", "present") else classes
        for cls in class_list:
            fg = (labels == cls).to(probas.dtype)
            if classes == "present" and fg.sum() == 0:
                continue
            errors = (fg - probas[:, cls]).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
            weights.append(1.0 if class_weights is None
                           else float(class_weights[cls]))
        if not losses:
            return logits.sum() * 0.0
        stacked = torch.stack(losses)
        w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
        return (stacked * w).sum() / w.sum()


class SegFormerSegmenter(nn.Module):
    """Wraps a HuggingFace SegformerForSemanticSegmentation so the API mirrors the
    DFormer EncoderDecoder used in train_dformer_v2_dlo.py:

        forward(rgb)             -> logits at (B, num_classes, H, W)   [inference]
        forward(rgb, label)      -> scalar loss                        [training]

    The HF model emits logits at H/4, W/4; we bilinear-upsample to (H, W) before
    loss / argmax for parity with DFormer-style outputs.
    """

    def __init__(self, backbone_name=BACKBONE_DEFAULT, num_classes=2, criterion=None,
                 lovasz_weight=0.0, lovasz_classes="present",
                 lovasz_class_weights=None):
        super().__init__()
        names = (CLASS_NAMES_THREE_WAY if int(num_classes) == 3
                 else CLASS_NAMES_BINARY)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone_name,
            num_labels=int(num_classes),
            id2label={i: n for i, n in enumerate(names[:int(num_classes)])},
            label2id={n: i for i, n in enumerate(names[:int(num_classes)])},
            ignore_mismatched_sizes=True,
        )
        self.criterion = criterion
        self.lovasz_weight = float(lovasz_weight)
        self.lovasz_classes = lovasz_classes
        self.lovasz_class_weights = lovasz_class_weights

    def forward(self, rgb, label=None):
        out = self.model(pixel_values=rgb)
        logits = F.interpolate(
            out.logits, size=(IMAGE_H, IMAGE_W), mode="bilinear", align_corners=False
        )
        if label is None:
            return logits
        # criterion has reduction="none" -> (B, H, W); take a scalar mean over all valid pixels.
        per_pixel = self.criterion(logits, label)
        valid = (label != IGNORE_INDEX)
        if valid.all():
            loss = per_pixel.mean()
        else:
            loss = per_pixel[valid].mean()
        # --lovasz-weight: IoU-surrogate auxiliary term. The `> 0.0` guard
        # short-circuits BEFORE any extra compute, so the default path
        # (lovasz_weight=0.0) is bit-identical to the pre-Lovasz trainer.
        if self.lovasz_weight > 0.0:
            loss = loss + self.lovasz_weight * lovasz_softmax(
                logits, label, classes=self.lovasz_classes, ignore=IGNORE_INDEX,
                class_weights=self.lovasz_class_weights,
            )
        return loss


# ─────────────────────────── METRICS ───────────────────────────


class BinaryIoU:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, pred, label):
        p = pred.astype(np.int64)
        l = label.astype(np.int64)
        self.tp += int(((p == 1) & (l == 1)).sum())
        self.fp += int(((p == 1) & (l == 0)).sum())
        self.fn += int(((p == 0) & (l == 1)).sum())
        self.tn += int(((p == 0) & (l == 0)).sum())

    def compute(self):
        iou_dlo = self.tp / max(self.tp + self.fp + self.fn, 1)
        iou_bg = self.tn / max(self.tn + self.fp + self.fn, 1)
        miou = (iou_dlo + iou_bg) / 2.0
        acc = (self.tp + self.tn) / max(self.tp + self.tn + self.fp + self.fn, 1)
        prec = self.tp / max(self.tp + self.fp, 1)
        rec = self.tp / max(self.tp + self.fn, 1)
        return {
            "miou": miou,
            "iou_dlo": iou_dlo,
            "iou_bg": iou_bg,
            "pixel_acc": acc,
            "precision_dlo": prec,
            "recall_dlo": rec,
        }


class MultiClassIoU:
    """Per-class IoU + mIoU + per-class precision/recall.

    For the 3-way task: classes are {0=bg, 1=wire, 2=connector}.
    The headline metric (used for best-checkpoint selection) is IoU(wire),
    matching Phase 7's IoU(DLO) emphasis.
    """

    def __init__(self, num_classes, class_names=None):
        self.k = int(num_classes)
        self.names = list(class_names) if class_names else [
            f"c{i}" for i in range(self.k)]
        # confusion[i, j] = #pixels with gt=i, pred=j
        self.cm = np.zeros((self.k, self.k), dtype=np.int64)

    def update(self, pred, label):
        p = pred.astype(np.int64).ravel()
        l = label.astype(np.int64).ravel()
        valid = (l >= 0) & (l < self.k) & (p >= 0) & (p < self.k)
        idx = l[valid] * self.k + p[valid]
        binc = np.bincount(idx, minlength=self.k * self.k)
        self.cm += binc.reshape(self.k, self.k)

    def reduce_(self, device):
        # For DDP: collapse the confusion matrix across all ranks in-place.
        import torch.distributed as dist
        t = torch.from_numpy(self.cm).to(device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        self.cm = t.to("cpu").to(torch.int64).numpy()

    def compute(self):
        cm = self.cm
        diag = np.diag(cm)
        row = cm.sum(axis=1)   # gt sums
        col = cm.sum(axis=0)   # pred sums
        denom = row + col - diag
        out = {}
        ious = []
        for i, name in enumerate(self.names[:self.k]):
            iou = float(diag[i] / max(int(denom[i]), 1))
            prec = float(diag[i] / max(int(col[i]), 1))
            rec = float(diag[i] / max(int(row[i]), 1))
            out[f"iou_{name}"] = iou
            out[f"precision_{name}"] = prec
            out[f"recall_{name}"] = rec
            ious.append(iou)
        out["miou"] = float(np.mean(ious))
        out["pixel_acc"] = float(diag.sum() / max(int(cm.sum()), 1))
        return out


# ─────────────────────────── TRAINING ───────────────────────────


SMOKE_PRESETS = {
    # Same shape as the DFormer-v2 smoke set; lower throughput numbers expected
    # (SegFormer-B5 has heavier MIT-B5 attention than DFormerv2_L's MoCA blocks).
    "a": {"n_sets": 1, "epochs": 1, "eval_every": 1, "log_every": 1, "label": "code-runs"},
    "b": {"n_sets": 3, "epochs": 5, "eval_every": 1, "log_every": 5, "label": "learnability"},
    "c": {"n_sets": None, "epochs": 1, "eval_every": 0, "log_every": 5, "label": "throughput"},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--single-gpu", action="store_true")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8, help="per-GPU batch size")
    p.add_argument("--lr", type=float, default=6e-5,
                   help="default 6e-5 (Phase-5 smoke-tested stable; the NaN run used 1e-4)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=1,
                   help="default 1 — check every epoch so best ckpt fires on each improvement")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=5,
                   help="default 5 (Phase-5 lesson: --ckpt-every 20 made restart-cost too high)")
    p.add_argument("--dlo-weight", type=float, default=6.0,
                   help="(binary mode) positive class weight for the DLO class.")
    p.add_argument("--lovasz-weight", type=float, default=0.0,
                   help="Weight of the Lovasz-Softmax IoU-surrogate auxiliary "
                        "loss added to weighted CE. 0.0 (default) = OFF and the "
                        "loss path is bit-identical to the pre-Lovasz trainer. "
                        "Targets small-class rim error that CE under-weights.")
    p.add_argument("--lovasz-classes", type=str, default="present",
                   choices=("present", "all"),
                   help="'present' (default) averages only over classes that "
                        "occur in the batch; 'all' always averages over every "
                        "class. 'present' avoids a zero-gradient pull toward "
                        "absent classes on connector-free frames.")
    p.add_argument("--aug-zoom", type=str, default=None,
                   help="Scale-only multi-scale augmentation as 'p,lo,hi' "
                        "(e.g. '0.5,0.5,1.0' = with prob 0.5 crop a random "
                        "[lo,hi] fraction and resize back, magnifying up to 2x). "
                        "No photometric ops, so it is a SINGLE variable vs a "
                        "run without it. Default None = OFF, zero extra RNG "
                        "draws, path bit-identical to before.")
    p.add_argument("--lovasz-class-weights", type=str, default=None,
                   help="Comma-separated per-class weights for the Lovasz term "
                        "(e.g. '1,1,4' in 3-way mode to concentrate the "
                        "IoU-surrogate gradient on the connector). Default None "
                        "= plain mean over present classes.")
    p.add_argument("--select-metric", type=str, default="auto",
                   choices=("auto", "iou_dlo", "iou_wire", "iou_con", "miou"),
                   help="Validation metric used for best-checkpoint selection. "
                        "'auto' (default) preserves historical behaviour: "
                        "iou_dlo in binary mode, iou_wire in 3-way mode. For a "
                        "3-way run judged on connector+mIoU, pass 'miou' so the "
                        "selected checkpoint matches the reported objective.")
    p.add_argument("--num-classes", type=int, default=2, choices=(2, 3),
                   help="2 = binary {bg, DLO} (Phase 7 default). "
                        "3 = three-way {bg, wire, connector} "
                        "(expects re-encoded {0,1,2} label cache).")
    p.add_argument("--class-weights", type=str, default=None,
                   help="(3-way mode) comma-separated CE weights per class, "
                        "e.g. '1,6,1' for {bg=1, wire=6, connector=1}. "
                        "If omitted, defaults to [1.0, dlo_weight, 1.0].")
    p.add_argument("--include-noise", action="store_true")
    p.add_argument("--aug2d", action="store_true",
                   help="enable the KD-validated 2D train-time aug stack "
                        "copied from src/train_rgb_only_kd.py: scale-jitter "
                        "p=0.3 (crop scale 0.8-1.0, resized back), h-flip "
                        "p=0.5, colour-jitter p=1.0 (brightness/contrast/"
                        "saturation ±0.2, hue ±0.05), gaussian blur p=0.3 "
                        "(k∈{3,5,7}, σ∈[0.1,1.5]). Train split only. Default "
                        "OFF — without it the trainer is bit-identical to "
                        "the historical flip-only path.")
    p.add_argument("--aug-hue", type=float, default=0.0,
                   help="extra-strong hue randomisation (Phase 18, targets "
                        "real cable colours absent from synth): with p=0.8 "
                        "per train sample, shift hue by U(-HUE, +HUE) "
                        "(0.5 = full hue wheel) and jitter saturation ±0.3. "
                        "Image-only (labels untouched), train split only. "
                        "0.0 = off (default; no extra RNG draws). Stacks "
                        "with --aug2d.")
    p.add_argument("--aug-heavy", action="store_true",
                   help="enable the HEAVY domain-randomisation / sim2real aug "
                        "stack (texture-invariance for real-world precision). "
                        "STACKS on top of --aug2d/--aug-hue. Per train sample: "
                        "gaussian sensor noise σ∈[0,12] p=0.4 + poisson shot "
                        "noise p=0.2; jpeg re-encode q∈[30,75] p=0.3; "
                        "directional motion blur len∈[3,15] p=0.2; perspective "
                        "warp distort∈[0.2,0.3] p=0.3 (warps LABEL too, NN); "
                        "gamma∈[0.6,1.6] p=0.3; brightness/contrast ±0.35 "
                        "p=0.3; grayscale p=0.1; channel-shuffle p=0.1; random "
                        "erasing 1-3 rects area∈[2%%,12%%] p=0.4 (IMAGE-ONLY, "
                        "label kept -> occlusion robustness). Train split "
                        "only. Default OFF -> zero extra RNG draws, bit-"
                        "identical default path.")
    p.add_argument("--aug-wirecolor", action="store_true",
                   help="enable LABEL-AWARE wire recolouring (Phase 24, targets "
                        "the real-world recall hole on BRIGHT+WARM/PALE cables "
                        "— white/grey/tan/pale-/bright-yellow). Per train "
                        "sample with p=0.5: recolour ONLY the wire pixels "
                        "(label collapses to DLO) toward one of {pale-white, "
                        "light-grey, warm-tan, bright-yellow, pale-pastel}, "
                        "blending HSV toward the target with strength "
                        "U(0.5,1.0) while preserving cable-like V shading + "
                        "small per-pixel jitter. BACKGROUND pixels are returned "
                        "UNCHANGED; geometry + label untouched. STACKS on "
                        "top of --aug2d/--aug-hue/--aug-heavy (runs last, uint8 "
                        "BGR). Train split only. Default OFF -> zero extra RNG "
                        "draws, bit-identical default path.")
    p.add_argument("--aug-bgclutter", action="store_true",
                   help="enable LABEL-AWARE BACKGROUND clutter / texture domain-"
                        "randomisation (Phase 25, targets real-world FALSE "
                        "POSITIVES on busy textured surfaces). Per train sample "
                        "with p=0.5: perturb ONLY the background pixels (label "
                        "NOT collapsing to DLO, i.e. lbl>3 in binary mode) with "
                        "a random subset of ISOTROPIC ops — multi-scale "
                        "procedural value-noise overlay (2-4 octaves, alpha "
                        "U(0.15,0.6)), a few near-square clutter patches of "
                        "stronger texture, and photometric DR (contrast/gamma/"
                        "brightness/per-channel) — raising background texture/"
                        "gradient energy with BLOBBY, NON-WIRE-LIKE structure. "
                        "NO strokes/lines/thin elongated structure are ever "
                        "drawn (that poisoned recall in P19). FOREGROUND (wire) "
                        "pixels and the LABEL are returned UNCHANGED; "
                        "geometry untouched. STACKS on top of --aug2d/--aug-hue/"
                        "--aug-heavy/--aug-wirecolor (runs last, uint8 BGR; only "
                        "writes bg so it cannot disturb wire recolours). Train "
                        "split only. Default OFF -> zero extra RNG draws, bit-"
                        "identical default path.")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="max_norm for gradient clipping; set <=0 to disable. "
                        "Always on by default to dodge Phase-5's AMP NaN.")
    p.add_argument("--no-amp", action="store_true",
                   help="disable AMP FP16 (FP32 training)")
    p.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT)
    p.add_argument("--backbone", default=BACKBONE_DEFAULT,
                   help="HuggingFace backbone id (e.g. nvidia/mit-b5, nvidia/mit-b4, nvidia/mit-b3)")
    p.add_argument("--data-dir", default=DATASET_DIR_DEFAULT)
    p.add_argument("--smoke", choices=list(SMOKE_PRESETS), default=None)
    p.add_argument("--limit-sets", type=int, default=None)
    p.add_argument("--time-budget", type=float, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--init-checkpoint", default=None,
                   help="Path to a .pth (or torch.save'd dict containing "
                        "'model_state_dict') to load on top of the freshly-"
                        "initialised pretrained backbone. Used by the Phase 10 "
                        "ablation to fine-tune Phase 7's teacher for 1 epoch "
                        "on each Phase-4-plus-one-lever variant.")
    p.add_argument("--resume", default=None,
                   help="Path to an epoch_<N>.pth / best_model.pth produced by "
                        "this trainer. Warm-restart: restores model weights, "
                        "resumes at epoch N+1, advances the LR schedule to the "
                        "matching global step, and restores the best-IoU "
                        "watermark. Optimizer moments are NOT stored in the "
                        "checkpoint so they reinitialise (negligible at the "
                        "small LRs where crashes are recovered). Used to "
                        "recover from a server crash mid-run.")
    return p.parse_args()


def setup_distributed(single_gpu):
    distributed = (not single_gpu) and ("LOCAL_RANK" in os.environ)
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank, dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(0)
    return False, 0, 0, 1


def build_dataset(args, rank, world_size):
    train_rgb, _, train_label = build_cache(args.data_dir, "train")
    val_rgb, _, val_label = build_cache(args.data_dir, "val")

    train_files = file_list(args.data_dir, "train")
    val_files = file_list(args.data_dir, "val")

    smoke_cfg = SMOKE_PRESETS.get(args.smoke) if args.smoke else None

    def first_n_train_set_ids(n):
        seen = []
        for fn in train_files:
            sid = int(os.path.basename(fn).split("_")[0])
            if sid not in seen:
                seen.append(sid)
        return seen[:n]

    allowed_sets = None
    if args.limit_sets is not None:
        allowed_sets = first_n_train_set_ids(args.limit_sets)
    elif smoke_cfg is not None and smoke_cfg.get("n_sets") is not None:
        allowed_sets = first_n_train_set_ids(smoke_cfg["n_sets"])

    train_indices = filter_indices_by_set(train_files, allowed_sets)
    if len(train_indices) == 0:
        raise RuntimeError(f"no train images selected (allowed_sets={allowed_sets})")
    val_indices = filter_indices_by_set(val_files, None)

    if rank == 0:
        print(f"  Train subset: {len(train_indices)} images "
              f"(sets={allowed_sets if allowed_sets is not None else 'all'})")
        print(f"  Val:          {len(val_indices)} images")
        if (args.aug2d or args.aug_hue > 0.0 or args.aug_heavy
                or args.aug_wirecolor or args.aug_bgclutter):
            print(f"  2D augs (train only): aug2d={'ON (KD stack)' if args.aug2d else 'off'}, "
                  f"aug_hue={args.aug_hue}"
                  + (" (p=0.8, hue ±%.2f, sat ±0.3)" % args.aug_hue
                     if args.aug_hue > 0.0 else " (off)")
                  + f", aug_heavy={'ON (domain-randomisation)' if args.aug_heavy else 'off'}"
                  + f", aug_wirecolor={'ON (p=0.5, label-aware wire recolour)' if args.aug_wirecolor else 'off'}"
                  + f", aug_bgclutter={'ON (p=0.5, label-aware bg texture/clutter DR)' if args.aug_bgclutter else 'off'}")

    # --aug2d / --aug-hue / --aug-heavy / --aug-wirecolor / --aug-bgclutter
    # apply to the TRAIN dataset only; val_full below is constructed without
    # them (and has augment=False besides).
    augmenter2d = RGBAugmentations() if args.aug2d else None
    heavy_aug = HeavyAugmentations() if args.aug_heavy else None
    wirecolor_aug = (
        WireColorAugmentation(p=0.5, num_classes=args.num_classes)
        if args.aug_wirecolor else None
    )
    bgclutter_aug = (
        BgClutterAugmentation(p=0.5, num_classes=args.num_classes)
        if args.aug_bgclutter else None
    )
    zoom_aug = None
    if args.aug_zoom:
        parts = [float(x) for x in args.aug_zoom.split(",")]
        if len(parts) != 3:
            raise ValueError(f"--aug-zoom expects 'p,lo,hi', got {args.aug_zoom!r}")
        if not (0.0 < parts[1] <= parts[2] <= 1.0):
            raise ValueError(f"--aug-zoom needs 0 < lo <= hi <= 1.0 "
                             f"(crop-then-resize only magnifies), got {parts}")
        zoom_aug = tuple(parts)
    train_full = CDLORGBOnlyDataset(
        train_rgb, train_label, augment=True, include_noise=args.include_noise,
        num_classes=args.num_classes, augmenter2d=augmenter2d,
        hue_aug=args.aug_hue, heavy_aug=heavy_aug, wirecolor_aug=wirecolor_aug,
        bgclutter_aug=bgclutter_aug, zoom_aug=zoom_aug,
    )
    val_full = CDLORGBOnlyDataset(
        val_rgb, val_label, augment=False, include_noise=args.include_noise,
        num_classes=args.num_classes,
    )
    train_dataset = Subset(train_full, train_indices) if allowed_sets is not None else train_full
    val_dataset = val_full
    return train_dataset, val_dataset


def _parse_class_weights(args, device):
    """Return a (num_classes,) float32 tensor of CE class weights on device."""
    if args.num_classes == 2:
        return torch.tensor(
            [1.0, float(args.dlo_weight)],
            dtype=torch.float32, device=device,
        )
    # num_classes == 3
    if args.class_weights:
        parts = [float(x) for x in args.class_weights.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"--class-weights expected 3 comma-sep values for 3-way mode, "
                f"got {args.class_weights!r}"
            )
        return torch.tensor(parts, dtype=torch.float32, device=device)
    # Default for 3-way: re-use dlo_weight as the wire-class weight; keep
    # bg + connector at 1.0 so the loss puts heavy emphasis on wire
    # (wire ~ 1-6% of pixels, connector ~ 7-47%, bg ~ 50-90%).
    return torch.tensor(
        [1.0, float(args.dlo_weight), 1.0],
        dtype=torch.float32, device=device,
    )


def build_model(args, distributed, device):
    pos_weight = _parse_class_weights(args, device)
    criterion = nn.CrossEntropyLoss(weight=pos_weight, reduction="none", ignore_index=IGNORE_INDEX)
    lcw = getattr(args, "lovasz_class_weights", None)
    if lcw:
        lcw = [float(x) for x in lcw.split(",")]
        if len(lcw) != args.num_classes:
            raise ValueError(f"--lovasz-class-weights expected {args.num_classes} "
                             f"values, got {lcw!r}")
    model = SegFormerSegmenter(
        backbone_name=args.backbone, num_classes=args.num_classes, criterion=criterion,
        lovasz_weight=getattr(args, "lovasz_weight", 0.0),
        lovasz_classes=getattr(args, "lovasz_classes", "present"),
        lovasz_class_weights=lcw,
    ).to(device)
    if distributed:
        # SyncBatchNorm conversion (decode_head has BatchNorm2d).
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device.index], output_device=device.index,
                    find_unused_parameters=False)
    return model


def fmt_seconds(s):
    return str(datetime.timedelta(seconds=int(max(s, 0))))


def evaluate(model, val_loader, device, distributed, use_amp, num_classes=2):
    if num_classes == 2:
        metric = BinaryIoU()
    else:
        metric = MultiClassIoU(
            num_classes=num_classes, class_names=CLASS_NAMES_THREE_WAY,
        )
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            rgb = normalize_batch(batch["rgb"], device)
            label_np = batch["label"].numpy()
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    m = model.module if isinstance(model, DDP) else model
                    out = m(rgb)
            else:
                m = model.module if isinstance(model, DDP) else model
                out = m(rgb)
            pred = out.argmax(dim=1).cpu().numpy()
            for i in range(pred.shape[0]):
                metric.update(pred[i].flatten(), label_np[i].flatten())

    if distributed:
        if isinstance(metric, BinaryIoU):
            t = torch.tensor([metric.tp, metric.fp, metric.fn, metric.tn],
                             dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            metric.tp, metric.fp, metric.fn, metric.tn = [int(v.item()) for v in t]
        else:
            metric.reduce_(device)

    return metric.compute()


def main():
    args = parse_args()

    if not (0.0 <= args.aug_hue <= 0.5):
        raise SystemExit(f"--aug-hue must be in [0, 0.5] "
                         f"(0.5 = full hue wheel); got {args.aug_hue}")

    smoke_cfg = SMOKE_PRESETS.get(args.smoke) if args.smoke else None
    if smoke_cfg is not None:
        args.epochs = smoke_cfg["epochs"]
        args.eval_every = smoke_cfg["eval_every"]
        args.log_every = smoke_cfg["log_every"]
        args.warmup_epochs = 0
        args.ckpt_every = max(args.epochs, 1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    distributed, local_rank, rank, world_size = setup_distributed(args.single_gpu)
    device = torch.device(f"cuda:{local_rank}")
    torch.set_float32_matmul_precision("high")
    use_amp = not args.no_amp

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        tb_dir = os.path.join(args.results_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)
        if args.num_classes == 2:
            mode_str = "binary {bg, DLO}"
        else:
            mode_str = "3-way {bg, wire, connector}"
        print(f"Training SegFormer ({args.backbone}) — RGB-only, {mode_str}")
        print(f"  GPUs: {world_size}, batch/GPU: {args.batch_size}, "
              f"total batch: {args.batch_size * world_size}")
        cw_str = (f"DLO weight: {args.dlo_weight}" if args.num_classes == 2
                  else f"class weights: {args.class_weights or f'[1, {args.dlo_weight}, 1]'}")
        print(f"  Epochs: {args.epochs}, LR: {args.lr}, {cw_str}, "
              f"AMP: {use_amp}, grad clip: {args.grad_clip}")
        print(f"  Smoke preset: {args.smoke or 'none'}; results: {args.results_dir}")
    else:
        writer = None

    if distributed:
        if rank == 0:
            train_dataset, val_dataset = build_dataset(args, rank, world_size)
        dist.barrier()
        if rank != 0:
            train_dataset, val_dataset = build_dataset(args, rank, world_size)
    else:
        train_dataset, val_dataset = build_dataset(args, rank, world_size)

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=0, pin_memory=True)

    if rank == 0:
        print(f"  Train batches/epoch: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model(args, distributed, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if rank == 0:
        print(f"  Model: SegFormer ({args.backbone}), {n_params:.1f}M params")

    if args.init_checkpoint is not None:
        if not os.path.isfile(args.init_checkpoint):
            raise FileNotFoundError(
                f"--init-checkpoint not found: {args.init_checkpoint}")
        if rank == 0:
            print(f"  Loading init checkpoint: {args.init_checkpoint}")
        ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        target = model.module if isinstance(model, DDP) else model
        # Warm-start HEAD EXPANSION: loading a checkpoint whose classifier head
        # has fewer rows than the model's (e.g. a 2-class checkpoint into a
        # 3-class model) RAISES — strict=False does NOT skip shape mismatches.
        # Reconcile by row-copying the checkpoint's available rows into the
        # target's larger head and leaving the extra row(s) at the model's
        # fresh init. This ONLY fires on an actual shape mismatch, so a normal
        # matching-shape warm-start (binary->binary) is the same as before.
        tgt_sd = target.state_dict()
        for ck in ("model.decode_head.classifier.weight",
                   "model.decode_head.classifier.bias"):
            if (ck in state_dict and ck in tgt_sd
                    and state_dict[ck].shape != tgt_sd[ck].shape):
                n_old = state_dict[ck].shape[0]
                expanded = tgt_sd[ck].clone()
                expanded[:n_old] = state_dict[ck]
                state_dict[ck] = expanded
                if rank == 0:
                    print(f"    head expansion: {ck} {tuple(state_dict[ck].shape)} "
                          f"<- copied {n_old} checkpoint row(s), rest fresh-init")
        # Drop any OTHER shape-mismatched entry so the freshly-built model keeps
        # its own value. The only such key for a binary->3-class warm-start is
        # the args-derived ``criterion.weight`` CE class-weight buffer ([2] in a
        # binary checkpoint vs [3] here) — it is rebuilt from --class-weights /
        # --dlo-weight, NOT something to inherit. strict=False would still RAISE
        # on this shape mismatch, so it must be removed before the load. Like the
        # head expansion above this is a no-op when shapes already match, so a
        # matching-shape warm-start stays identical.
        for mk in [k for k in list(state_dict)
                   if k in tgt_sd and state_dict[k].shape != tgt_sd[k].shape]:
            if rank == 0:
                print(f"    drop shape-mismatched key (keep model init): {mk} "
                      f"ckpt{tuple(state_dict[mk].shape)} vs "
                      f"model{tuple(tgt_sd[mk].shape)}")
            del state_dict[mk]
        missing, unexpected = target.load_state_dict(state_dict, strict=False)
        if rank == 0:
            print(f"    missing keys ({len(missing)}): "
                  f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")
            print(f"    unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(args.epochs * len(train_loader), 1)
    warmup_steps = max(args.warmup_epochs * len(train_loader), 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / warmup_steps, 1e-6)
        return max((1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1)) ** 0.9, 1e-6)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_miou = 0.0
    best_iou_dlo = 0.0
    start_time = time.time()
    global_step = 0
    start_epoch = 1
    n_train_imgs = len(train_dataset)
    train_imgs_processed = 0
    eta_aborted = False

    # Warm-restart from a crash. Checkpoints store only model weights + epoch +
    # metrics (no optimizer/scaler state), so we restore weights, jump the
    # epoch counter to N+1, fast-forward the LR schedule to the matching global
    # step, and recover the best-IoU watermark. The headline metric key differs
    # between binary (iou_dlo) and 3-way (iou_wire) mode.
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume not found: {args.resume}")
        rs = torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(rs, dict) and "model_state_dict" in rs:
            rs_state = rs["model_state_dict"]
        else:
            rs_state = rs
        tgt = model.module if isinstance(model, DDP) else model
        miss, unexp = tgt.load_state_dict(rs_state, strict=False)
        done_epoch = int(rs.get("epoch", 0)) if isinstance(rs, dict) else 0
        start_epoch = done_epoch + 1
        global_step = (start_epoch - 1) * len(train_loader)
        for _ in range(global_step):
            scheduler.step()
        rs_metrics = rs.get("metrics", {}) if isinstance(rs, dict) else {}
        if args.select_metric != "auto":
            hk = ("iou_connector" if args.select_metric == "iou_con"
                  else args.select_metric)
        else:
            hk = "iou_wire" if args.num_classes == 3 else "iou_dlo"
        best_iou_dlo = float(rs_metrics.get(hk, rs_metrics.get("iou_dlo", 0.0)) or 0.0)
        best_miou = float(rs_metrics.get("miou", 0.0) or 0.0)
        if rank == 0:
            print(f"  RESUMING from {args.resume}")
            print(f"    resumed epoch {done_epoch} → start_epoch {start_epoch}; "
                  f"global_step={global_step}; "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")
            print(f"    restored best watermark: IoU={best_iou_dlo:.4f} "
                  f"mIoU={best_miou:.4f}")
            print(f"    missing keys ({len(miss)}); unexpected ({len(unexp)})")

    if rank == 0:
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_t0 = time.time()
        epoch_loss = 0.0
        epoch_iters = 0
        batch_t0 = time.time()
        nan_in_epoch = 0

        for bi, batch in enumerate(train_loader):
            rgb = normalize_batch(batch["rgb"], device)
            label = batch["label"].to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(rgb, label)
            else:
                loss = model(rgb, label)

            if not torch.isfinite(loss):
                # Skip the batch, don't poison the running mean — log and continue.
                nan_in_epoch += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            epoch_loss += float(loss.item())
            epoch_iters += 1
            global_step += 1
            train_imgs_processed += rgb.size(0) * world_size

            if rank == 0 and (bi + 1) % args.log_every == 0:
                dt = time.time() - batch_t0
                imgs_per_s = (args.log_every * args.batch_size * world_size) / max(dt, 1e-9)
                lr = optimizer.param_groups[0]["lr"]
                print(f"    ep{epoch:3d} batch {bi+1:4d}/{len(train_loader):4d}  "
                      f"loss={float(loss.item()):.4f}  lr={lr:.2e}  {imgs_per_s:.1f} img/s")
                if writer is not None:
                    writer.add_scalar("train/loss_step", float(loss.item()), global_step)
                    writer.add_scalar("train/imgs_per_sec", imgs_per_s, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                batch_t0 = time.time()

        epoch_dt = time.time() - epoch_t0
        avg_loss = epoch_loss / max(epoch_iters, 1)
        elapsed = time.time() - start_time
        per_epoch = elapsed / epoch
        remaining_epochs = args.epochs - epoch
        eta = per_epoch * remaining_epochs

        if rank == 0:
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"epoch_wall={fmt_seconds(epoch_dt)}  ETA={fmt_seconds(eta)}  "
                  f"peak_GPU={peak_mb:.0f}MB"
                  + (f"  NaN_skipped={nan_in_epoch}" if nan_in_epoch else ""))
            if writer is not None:
                writer.add_scalar("train/loss_epoch", avg_loss, epoch)
                writer.add_scalar("train/epoch_wall_sec", epoch_dt, epoch)
                writer.add_scalar("train/eta_sec", eta, epoch)
                writer.add_scalar("train/peak_gpu_mb", peak_mb, epoch)
                if nan_in_epoch:
                    writer.add_scalar("train/nan_batches", nan_in_epoch, epoch)

        if args.time_budget is not None:
            projected_total_h = (per_epoch * args.epochs) / 3600.0
            local_abort = 1.0 if projected_total_h > args.time_budget else 0.0
            if distributed:
                t = torch.tensor([local_abort], device=device)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                eta_aborted = bool(t.item() > 0.5)
            else:
                eta_aborted = bool(local_abort > 0.5)
            if rank == 0 and eta_aborted:
                print(f"  projected total wall {projected_total_h:.2f}h exceeds budget "
                      f"{args.time_budget:.2f}h — aborting")
        if eta_aborted:
            break

        do_eval = (
            args.eval_every > 0
            and (epoch % args.eval_every == 0 or epoch == args.epochs or epoch == 1)
        )
        if do_eval:
            metrics = evaluate(model, val_loader, device, distributed, use_amp,
                               num_classes=args.num_classes)
            if rank == 0:
                if args.num_classes == 2:
                    print(f"    val: mIoU={metrics['miou']:.4f}  "
                          f"IoU(DLO)={metrics['iou_dlo']:.4f}  "
                          f"IoU(bg)={metrics['iou_bg']:.4f}  "
                          f"acc={metrics['pixel_acc']:.4f}  "
                          f"prec(DLO)={metrics['precision_dlo']:.4f}  "
                          f"rec(DLO)={metrics['recall_dlo']:.4f}")
                else:
                    print(f"    val: mIoU={metrics['miou']:.4f}  "
                          f"IoU(wire)={metrics['iou_wire']:.4f}  "
                          f"IoU(bg)={metrics['iou_bg']:.4f}  "
                          f"IoU(con)={metrics['iou_connector']:.4f}  "
                          f"acc={metrics['pixel_acc']:.4f}  "
                          f"prec(wire)={metrics['precision_wire']:.4f}  "
                          f"rec(wire)={metrics['recall_wire']:.4f}")
                if writer is not None:
                    for k, v in metrics.items():
                        writer.add_scalar(f"val/{k}", v, epoch)

                # Headline metric for best-ckpt selection. 'auto' preserves the
                # historical choice (IoU(DLO) binary / IoU(wire) 3-way, both
                # measuring wire recovery); --select-metric overrides it so a run
                # judged on connector+mIoU selects on the metric it reports
                # instead of on wire alone.
                if args.select_metric != "auto":
                    headline_key = ("iou_connector" if args.select_metric == "iou_con"
                                    else args.select_metric)
                else:
                    headline_key = "iou_dlo" if args.num_classes == 2 else "iou_wire"
                if metrics[headline_key] > best_iou_dlo:
                    best_iou_dlo = metrics[headline_key]
                    best_miou = metrics["miou"]
                    class_names_for_cfg = (
                        CLASS_NAMES_BINARY if args.num_classes == 2
                        else CLASS_NAMES_THREE_WAY
                    )
                    ckpt = {
                        "epoch": epoch,
                        "metrics": metrics,
                        "args": vars(args),
                        "model_state_dict": (model.module if isinstance(model, DDP) else model).state_dict(),
                        "config": {"backbone": args.backbone,
                                   "num_classes": args.num_classes,
                                   "class_names": class_names_for_cfg,
                                   "image_size": [IMAGE_H, IMAGE_W]},
                    }
                    torch.save(ckpt, os.path.join(args.results_dir, "best_model.pth"))
                    headline_name = {
                        "iou_dlo": "IoU(DLO)", "iou_wire": "IoU(wire)",
                        "iou_connector": "IoU(con)", "miou": "mIoU",
                    }.get(headline_key, headline_key)
                    print(f"    ★ new best {headline_name}={best_iou_dlo:.4f}  "
                          f"mIoU={best_miou:.4f}")

        if rank == 0 and args.ckpt_every > 0 and epoch % args.ckpt_every == 0:
            ckpt = {
                "epoch": epoch,
                "args": vars(args),
                "model_state_dict": (model.module if isinstance(model, DDP) else model).state_dict(),
            }
            torch.save(ckpt, os.path.join(args.results_dir, f"epoch_{epoch}.pth"))

    final_eval = None
    if args.eval_every > 0:
        try:
            final_eval = evaluate(model, val_loader, device, distributed, use_amp,
                                  num_classes=args.num_classes)
        except Exception as e:
            if rank == 0:
                print(f"  final eval failed: {e}")

    if rank == 0:
        total_time = time.time() - start_time
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\nDone in {fmt_seconds(total_time)}")
        if best_iou_dlo > 0:
            print(f"Best IoU(DLO)={best_iou_dlo:.4f}  mIoU={best_miou:.4f}")
        print(f"Peak GPU memory: {peak_mb:.0f} MB")

        full_train_imgs = 7560
        imgs_per_s_overall = None
        full_epoch_secs = None
        if epoch_iters > 0 and total_time > 0:
            imgs_per_s_overall = train_imgs_processed / total_time
            full_epoch_secs = full_train_imgs / max(imgs_per_s_overall, 1e-9)
            print(f"Throughput: {imgs_per_s_overall:.1f} img/s overall  "
                  f"(full train epoch ≈ {fmt_seconds(full_epoch_secs)} ≈ {full_epoch_secs/3600:.2f}h)")

        if final_eval is not None:
            if args.num_classes == 2:
                print(f"Final eval: mIoU={final_eval['miou']:.4f}  "
                      f"IoU(DLO)={final_eval['iou_dlo']:.4f}  "
                      f"IoU(bg)={final_eval['iou_bg']:.4f}")
            else:
                print(f"Final eval: mIoU={final_eval['miou']:.4f}  "
                      f"IoU(wire)={final_eval['iou_wire']:.4f}  "
                      f"IoU(bg)={final_eval['iou_bg']:.4f}  "
                      f"IoU(con)={final_eval['iou_connector']:.4f}")

        report = {
            "args": vars(args),
            "wall_seconds": total_time,
            "peak_gpu_mb": peak_mb,
            "best_iou_dlo": best_iou_dlo,
            "best_miou": best_miou,
            "final_eval": final_eval,
            "epochs_completed": epoch,
            "n_train_imgs_used": n_train_imgs,
            "world_size": world_size,
            "batch_per_gpu": args.batch_size,
            "imgs_per_sec_overall": imgs_per_s_overall,
            "projected_full_epoch_seconds": full_epoch_secs,
        }
        with open(os.path.join(args.results_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
        print(f"Report: {os.path.join(args.results_dir, 'report.json')}")

        if writer is not None:
            writer.close()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
