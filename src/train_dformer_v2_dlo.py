"""Train DFormer-v2-Large for binary DLO/background segmentation on the CDLO RGB-D dataset.

Phase 5 deliverable. Reuses the mmap cache produced by `src/train_rgbd_seg.py:build_cache`
(after the 2026-04-28 textured-RGB patch — see CONTEXT.md §0.3 step 1). The 5-class cache
is collapsed to a 2-class label inside `__getitem__`:

    cache label gt_transform: 0=Wire, 1=Endpoint, 2=Bifurcation, 3=Connector, 4=Noise, 255=bg
    binary: classes 0..3 -> 1 (DLO); class 4 (Noise) and 255 (bg) -> 0 (bg)

Usage examples (run from project root):
    source env/bin/activate
    torchrun --nproc_per_node=2 src/train_dformer_v2_dlo.py --smoke a --batch-size 4
    torchrun --nproc_per_node=2 src/train_dformer_v2_dlo.py --smoke b --batch-size 4
    torchrun --nproc_per_node=2 src/train_dformer_v2_dlo.py --smoke c --batch-size 4
    torchrun --nproc_per_node=2 src/train_dformer_v2_dlo.py --epochs 80 --batch-size 4

PROJECT_ROOT is derived from this file's location, so the script works
unchanged whether it lives at /workspace/kiat_crefle/src/... (dev box) or
/home/<user>/.../dlo-segmentation/src/... (rorqual).  --data-dir and
--pretrained accept absolute paths to override the defaults.
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

DFORMER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dformer")
sys.path.insert(0, DFORMER_DIR)

from models.builder import EncoderDecoder as DFormerModel  # noqa: E402

# Reuse the cache builder from the 5-class teacher script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rgbd_seg import build_cache  # noqa: E402

# Reuse the EXACT RGB 2D aug stack from the RGB-only SOTA trainer so an RGB-D
# vs RGB-only comparison is fair (these augs were decisive for the RGB-only
# baseline). RGBAugmentations = scale-jitter -> flip -> colour-jitter -> blur
# (geometric ops touch rgb AND label; photometric ops touch rgb only).
# hue_sat_jitter = the --aug-hue extra-strong hue/saturation randomisation
# (image-only). Both operate on uint8 BGR HWC arrays with global-`random`
# draws, identical substrate to this trainer.
from train_rgb_only_sota import RGBAugmentations, hue_sat_jitter, MultiClassIoU  # noqa: E402

# Full-precision depth restaging (task 2): builds a per-frame DLO-windowed uint8
# depth cache from the raw uint16 render so the connector-vs-wire LOCAL contrast
# survives normalization (the staged 8-bit depth washes it out). See that module.
from restage_depth_fullprecision import build_depth_fp_cache  # noqa: E402


class RGBDAugmentations(RGBAugmentations):
    """RGBAugmentations extended to thread the DEPTH channel through the
    GEOMETRIC ops (scale-jitter, flip) so rgb/depth/label stay registered.

    Photometric ops (colour-jitter, blur) are applied to the RGB channel ONLY
    (depth gets its own domain aug elsewhere). The RNG-draw ORDER and the RGB
    transform are identical to the parent RGBAugmentations.__call__
    (same `random.*` calls in the same sequence), so an RGB-only vs RGB-D run
    with --aug2d augments the RGB channel identically.
    """

    def apply_rgbd(self, rgb, depth, label):
        # rgb: HxWx3 uint8 BGR; depth: HxW uint8; label: HxW uint8.
        # --- scale jitter (geometric: rgb + depth + label) ---
        if random.random() < self.rrc_p:
            h, w = rgb.shape[:2]
            scale = random.uniform(*self.rrc_scale)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            y0 = random.randint(0, h - new_h)
            x0 = random.randint(0, w - new_w)
            rgb = cv2.resize(rgb[y0:y0 + new_h, x0:x0 + new_w, :], (w, h),
                             interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth[y0:y0 + new_h, x0:x0 + new_w], (w, h),
                               interpolation=cv2.INTER_NEAREST)  # never blend dist
            label = cv2.resize(label[y0:y0 + new_h, x0:x0 + new_w], (w, h),
                               interpolation=cv2.INTER_NEAREST)
        # --- horizontal flip (geometric: rgb + depth + label) ---
        if random.random() < self.flip_p:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            depth = np.ascontiguousarray(depth[:, ::-1])
            label = np.ascontiguousarray(label[:, ::-1])
        # --- colour jitter (RGB only) — identical draws/order to parent ---
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
        # --- gaussian blur (RGB only) ---
        if random.random() < self.blur_p:
            k = random.choice(self.blur_kernels)
            sigma = random.uniform(*self.blur_sigma)
            rgb = cv2.GaussianBlur(rgb, (k, k), sigma)
        return (np.ascontiguousarray(rgb), np.ascontiguousarray(depth),
                np.ascontiguousarray(label))

# ─────────────────────────── CONFIG ───────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "data", "dformer_dataset")
RESULTS_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "results", "dformer_v2_dlo")
PRETRAINED_DEFAULT = os.path.join(
    PROJECT_ROOT, "data", "pretrained", "DFormerv2", "pretrained", "DFormerv2_Large_pretrained.pth"
)

# Binary mode (legacy P27 default): {bg=0, DLO=1}.
# 3-way mode (this task): {bg=0, wire=1, connector=2}, read straight from the
# re-encoded {0,1,2} Label cache (no binary collapse). Class names mirror the
# SegFormer 3-way reference (src/train_rgb_only_sota.py) so logs are readable.
NUM_CLASSES = 2
CLASS_NAMES = ["bg", "DLO"]
CLASS_NAMES_BINARY = ["bg", "DLO"]
CLASS_NAMES_THREE_WAY = ["bg", "wire", "connector"]
IGNORE_INDEX = -1  # sentinel: nothing in the label tensor is == -1, so loss covers all pixels

IMAGE_H, IMAGE_W = 480, 640
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
DEPTH_MEAN = torch.tensor([0.48, 0.48, 0.48], dtype=torch.float32).view(1, 3, 1, 1)
DEPTH_STD = torch.tensor([0.28, 0.28, 0.28], dtype=torch.float32).view(1, 3, 1, 1)


# ─────────────────────── DEPTH DOMAIN AUGMENTATION ───────────────────────
#
# WHY: synth render depth is ~86-91% zero pixels (unrendered background) and
# bimodal; the REAL valset depth (Intel-RealSense-like) is ~0.03% zero, dense,
# roughly uniform 0..254 with per-frame mean ~130 ± 33. A clean-synth model
# learns "depth == 0  =>  background" — a cue that covers 87% of synth bg but
# has 0.03% support on real data — so it collapses on real depth. This aug
# rewrites the synth-train depth (train split only, per-sample, on the raw
# uint8 BEFORE the /255 normalize) to look sensor-dense while PRESERVING the
# relative depth edges of the rendered wires/objects (the transferable signal).
#
# Pipeline (all on a single-channel uint8 HxW depth, returns uint8 HxW):
#   1. dense fill of the zero region  (THE shortcut killer; %zero 87% -> ~0)
#      - inpaint/NN-fill from the nearest rendered depth, then
#      - blend a smooth low-freq gradient ramp into the (formerly-zero) bg so
#        it is plausibly sensor-like rather than a flat silhouette.
#   2. sensor noise + quantization + a few dropout holes (sensor artefacts).
#   3. random per-frame gain + offset (real per-frame mean varies a lot).
# The rendered foreground (originally non-zero) keeps its depth structure; only
# the background structure and the global stats are altered.


def depth_domain_aug(depth_u8, rng=None):
    """Transform a synth-render depth map (uint8 HxW) to resemble dense sensor
    depth. Pure function of `depth_u8` + the supplied RNG; never mutates input.

    rng: a numpy Generator (default: a fresh one seeded off the global `random`
    stream, matching the single-RNG-stream convention used by the RGB augs).
    """
    if rng is None:
        rng = np.random.default_rng(random.getrandbits(63))
    d = depth_u8.astype(np.float32)
    H, W = d.shape
    valid = depth_u8 > 0  # rendered (foreground) pixels

    # ---- 1. dense background fill ----------------------------------------
    if valid.any() and (~valid).any():
        # Nearest-rendered-depth fill: for every zero pixel, copy the value of
        # the nearest non-zero (rendered) pixel. distanceTransform with
        # labels gives, for each pixel, the index of the nearest zero in a
        # mask — so we invert (zeros = the rendered pixels) to propagate
        # rendered depth outward into the background.
        # nearest non-zero (rendered) source for each pixel:
        _, labels = cv2.distanceTransformWithLabels(
            (depth_u8 == 0).astype(np.uint8), cv2.DIST_L2, 5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        # Build value lookup keyed by the label assigned to each *rendered* px.
        flat_vals = d.reshape(-1)
        ys, xs = np.where(depth_u8 > 0)
        src_labels = labels[ys, xs]
        lut = np.zeros(labels.max() + 1, dtype=np.float32)
        lut[src_labels] = flat_vals[ys * W + xs]
        nn_fill = lut[labels].reshape(H, W)

        # Smooth low-frequency gradient ramp (plausible receding background):
        gy = np.linspace(rng.uniform(-1, 1), rng.uniform(-1, 1), H)[:, None]
        gx = np.linspace(rng.uniform(-1, 1), rng.uniform(-1, 1), W)[None, :]
        ramp = (gy + gx)
        ramp = ramp / (np.abs(ramp).max() + 1e-6)  # [-1, 1]
        nz_mean = float(d[valid].mean())
        bg_base = np.clip(nz_mean + ramp * rng.uniform(20.0, 60.0), 1.0, 254.0)
        alpha = rng.uniform(0.4, 0.8)  # mix nearest-fill <-> smooth ramp in bg
        bg = (1.0 - alpha) * nn_fill + alpha * bg_base

        d = np.where(valid, d, bg)

    # ---- 3. random per-frame gain + offset (apply before noise so noise is
    #         in the final value scale; foreground edges preserved as gain is
    #         global) ----------------------------------------------------------
    gain = rng.uniform(0.7, 1.3)
    offset = rng.uniform(-40.0, 40.0)
    d = d * gain + offset

    # ---- 2. sensor noise + quantization + dropout holes ------------------
    # additive gaussian + mild multiplicative noise
    d = d + rng.normal(0.0, rng.uniform(1.0, 5.0), size=d.shape)
    d = d * (1.0 + rng.normal(0.0, 0.02, size=d.shape))
    # light quantization (coarser depth steps, like a real sensor)
    q = rng.integers(1, 4)  # step in {1,2,3}
    if q > 1:
        d = np.round(d / q) * q
    # a few small dropout patches (sensor holes -> 0). Kept rare and small so
    # post-aug %zero stays ~1% (real valset is ~0.03%); the goal is to break the
    # depth==0==background equivalence, NOT to recreate a large zero silhouette.
    if rng.random() < 0.5:
        n_holes = int(rng.integers(1, 4))
        for _ in range(n_holes):
            hh = int(rng.integers(3, max(4, H // 30)))   # <=16px tall
            hw = int(rng.integers(3, max(4, W // 30)))   # <=21px wide
            y0 = int(rng.integers(0, max(1, H - hh)))
            x0 = int(rng.integers(0, max(1, W - hw)))
            d[y0:y0 + hh, x0:x0 + hw] = 0.0

    return np.clip(d, 0.0, 255.0).astype(np.uint8)


# ─────────────────────────── DATASET ───────────────────────────


class BinaryCDLOMmapDataset(Dataset):
    """Mmap-backed CDLO RGB-D dataset.

    Two label modes (task 1):
    * ``num_classes=2`` (legacy P27 binary): cache labels arrive as {0..4, 255}
      (gt_transform). Collapse classes 0..3 -> 1 (DLO); 4 (Noise) and 255 (bg)
      -> 0. ``include_noise`` flips Noise into DLO.
    * ``num_classes=3`` (3-way de-cheat): cache labels arrive already as
      {0=bg, 1=wire, 2=connector} from the re-encoded Label PNGs (see
      src/reencode_labels_3way.py). Passed through UNMODIFIED -- NO binary
      collapse. ``include_noise`` is ignored.
    """

    def __init__(self, rgb, depth, label, augment=True, include_noise=False,
                 augmenter2d=None, hue_aug=0.0, depth_domain_aug=False,
                 num_classes=2, bin_from_3way=False):
        self.rgb = rgb
        self.depth = depth
        self.label = label
        self.augment = augment
        self.include_noise = include_noise
        self.num_classes = int(num_classes)
        # bin_from_3way: binary mode on the RE-ENCODED {0=bg,1=wire,2=connector}
        # cache -> collapse (lbl>=1) to {0=bg, 1=DLO(wire∪connector)} instead of
        # the legacy {0..4 fg, 255 bg} -> (lbl<=3) collapse (which would WRONGLY
        # map bg=0 to DLO on this data). Auto-detected in build_dataset by
        # label max<=2; the legacy collapse is untouched when False.
        self.bin_from_3way = bool(bin_from_3way)
        if self.num_classes not in (2, 3):
            raise ValueError(f"num_classes={num_classes} not in (2, 3)")
        # RGB 2D aug (task 3): applied to the RGB CHANNEL ONLY (depth has its
        # own aug). augmenter2d (--aug2d) does geometric ops that must also
        # transform depth+label to stay registered; hue_aug (--aug-hue) is
        # image-only. Both default OFF (None / 0.0) -> the same as the
        # historical flip-only path.
        self.augmenter2d = augmenter2d
        self.hue_aug = float(hue_aug)
        # depth domain aug (task 1): --depth-domain-aug, default OFF.
        self.depth_domain_aug_on = bool(depth_domain_aug)

    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, idx):
        rgb = self.rgb[idx].copy()      # (H, W, 3) uint8
        depth = self.depth[idx].copy()   # (H, W) uint8
        lbl = self.label[idx].copy()     # (H, W) uint8 — 0..4 fg, 255 bg

        if self.augment:
            if self.augmenter2d is not None:
                # --aug2d: KD stack (scale-jitter -> flip -> colour-jitter ->
                # blur). apply_rgbd() threads the GEOMETRIC ops (scale-jitter,
                # flip) through depth+lbl too (so they stay registered), while
                # the PHOTOMETRIC ops (colour-jitter, blur) touch RGB ONLY
                # (depth gets its own domain aug below). The RGB transform and
                # RNG-draw order are identical to the RGB-only trainer.
                rgb, depth, lbl = self.augmenter2d.apply_rgbd(rgb, depth, lbl)
            elif random.random() > 0.5:
                # Historical default: h-flip only (unchanged single draw).
                rgb = np.ascontiguousarray(rgb[:, ::-1, :])
                depth = np.ascontiguousarray(depth[:, ::-1])
                lbl = np.ascontiguousarray(lbl[:, ::-1])
            if self.hue_aug > 0.0 and random.random() < 0.8:
                # --aug-hue: RGB-only; depth/label untouched. The hue_aug > 0.0
                # guard short-circuits BEFORE the draw so the default path
                # (hue_aug=0.0) consumes no extra randomness.
                rgb = hue_sat_jitter(rgb, self.hue_aug)
            if self.depth_domain_aug_on:
                # task 1: rewrite synth depth to sensor-dense (train split only).
                depth = depth_domain_aug(depth)

        if self.num_classes == 3:
            # 3-way: labels are already {0=bg, 1=wire, 2=connector}. NO collapse.
            label_out = lbl.astype(np.int64)
        elif self.bin_from_3way:
            # Binary from RE-ENCODED {0=bg,1=wire,2=connector}: bg=0 stays bg,
            # wire+connector -> 1 (DLO). bg(0) is NOT swept into DLO.
            label_out = (lbl >= 1).astype(np.int64)
        else:
            # Legacy binary collapse: classes 0..3 -> 1 (DLO), class 4 (Noise)
            # and 255 (bg) -> 0.
            if self.include_noise:
                # Treat Noise as DLO (rare; not the default)
                label_out = (lbl <= 4).astype(np.int64)
            else:
                label_out = (lbl <= 3).astype(np.int64)

        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1))  # (3, H, W) uint8
        depth_t = torch.from_numpy(depth).unsqueeze(0)     # (1, H, W) uint8
        label_t = torch.from_numpy(label_out)              # (H, W) int64 {0,1[,2]}
        return {"rgb": rgb_t, "depth": depth_t, "label": label_t}


def normalize_batch(rgb_uint8, depth_uint8, device, no_depth=False):
    rgb = rgb_uint8.to(device, dtype=torch.float32, non_blocking=True) / 255.0
    rgb = (rgb - RGB_MEAN.to(device)) / RGB_STD.to(device)
    # task 4: --no-depth control. Zero the depth BEFORE normalize so the model
    # receives a CONSTANT depth map (carries no geometric information) -- the
    # same DFormer arch on the same data with depth disabled. This isolates
    # depth's contribution (depth-on vs depth-off) from the architecture change.
    if no_depth:
        depth = torch.zeros(
            (rgb.shape[0], 3, rgb.shape[2], rgb.shape[3]),
            dtype=torch.float32, device=device,
        )
        depth = (depth - DEPTH_MEAN.to(device)) / DEPTH_STD.to(device)
        return rgb, depth
    depth = depth_uint8.to(device, dtype=torch.float32, non_blocking=True) / 255.0
    depth = depth.expand(-1, 3, -1, -1)
    depth = (depth - DEPTH_MEAN.to(device)) / DEPTH_STD.to(device)
    return rgb, depth


def file_list(data_dir, split):
    txt = os.path.join(data_dir, "train.txt" if split == "train" else "test.txt")
    with open(txt) as f:
        return [line.strip() for line in f if line.strip()]


def filter_indices_by_set(file_names, allowed_sets):
    """Return indices into file_names whose set_id (first 3 digits of basename) is in allowed_sets."""
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


class ModelConfig:
    backbone = "DFormerv2_L"
    pretrained_model = PRETRAINED_DEFAULT
    decoder = "ham"
    decoder_embed_dim = 1024
    num_classes = NUM_CLASSES
    background = IGNORE_INDEX  # used by EncoderDecoder.forward to mask pixels in loss
    bn_eps = 1e-3
    bn_momentum = 0.1
    drop_path_rate = 0.3
    aux_rate = 0.0
    fix_bias = True


# ─────────────────────────── METRICS ───────────────────────────


class BinaryIoU:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, pred, label):
        # pred, label: 1d arrays of {0, 1}
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


# ─────────────────────────── TRAINING ───────────────────────────


SMOKE_PRESETS = {
    # n_sets selects the first N existing train set_ids (train set ids are non-contiguous;
    # val/test sets are interleaved, so a hardcoded [0,1,2] would silently miss set 001).
    # (a) code-runs check: 1 set, 1 epoch — verifies forward+backward+AMP+DDP wire-up
    "a": {"n_sets": 1, "epochs": 1, "eval_every": 1, "log_every": 1, "label": "code-runs"},
    # (b) learnability check: 3 sets, 5 epochs — confirms IoU climbs above bg-only baseline
    "b": {"n_sets": 3, "epochs": 5, "eval_every": 1, "log_every": 5, "label": "learnability"},
    # (c) throughput projection: full train set, 1 epoch, no eval — measure images/sec
    "c": {"n_sets": None, "epochs": 1, "eval_every": 0, "log_every": 5, "label": "throughput"},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--single-gpu", action="store_true")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4, help="per-GPU batch size")
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=20, help="batches per train log line")
    p.add_argument("--ckpt-every", type=int, default=20)
    p.add_argument("--dlo-weight", type=float, default=6.0,
                   help="positive class weight for the DLO/wire class")
    p.add_argument("--num-classes", type=int, default=2, choices=(2, 3),
                   help="TASK 1: 2 = binary {bg, DLO} (legacy P27). 3 = three-way "
                        "{bg, wire, connector} (reads the re-encoded {0,1,2} Label "
                        "cache; per-class val IoU incl. IoU(connector)).")
    p.add_argument("--class-weights", type=str, default=None,
                   help="(3-way) comma-separated CE weights per class, e.g. "
                        "'1,6,4' for {bg=1, wire=6, connector=4}. If omitted, "
                        "defaults to [1.0, dlo_weight, connector_weight].")
    p.add_argument("--connector-weight", type=float, default=4.0,
                   help="(3-way) CE weight for the connector class when "
                        "--class-weights is not given (connector is the hard "
                        "minority class ~7%% of DLO px). Default 4.0.")
    p.add_argument("--no-depth", action="store_true",
                   help="TASK 4: RGB-only CONTROL. Feed a CONSTANT (zeroed) depth "
                        "map so the SAME DFormer arch trains on the SAME data with "
                        "depth disabled -- isolates depth's contribution from the "
                        "architecture change. Default OFF (RGB-D).")
    p.add_argument("--depth-source", choices=("fullprec", "staged"), default="fullprec",
                   help="TASK 2: 'fullprec' (default) = per-frame DLO-windowed "
                        "uint8 depth rebuilt from the raw uint16 render "
                        "(connector contrast PRESERVED; cache/{split}_depth_fp.npy "
                        "via src/restage_depth_fullprecision.py). 'staged' = the "
                        "original globally-normalized 8-bit Depth/ (connector "
                        "contrast WASHED OUT; for ablation only).")
    p.add_argument("--render-dir",
                   default=os.path.join(PROJECT_ROOT, "data", "render_3way_decheat"),
                   help="raw uint16 render root (for --depth-source fullprec).")
    p.add_argument("--init-checkpoint", default=None,
                   help="TASK B: warm-start from a DFormer checkpoint (.pth or a "
                        "dict with 'model_state_dict'), loaded AFTER the pretrained "
                        "backbone. On a 3-class build loading a 2-class checkpoint, "
                        "the decode-head classifier [2,C,1,1]->[3,C,1,1] is expanded "
                        "by row-copy (bg->bg, DLO->wire, fresh-init connector); the "
                        "[2]-shaped criterion.weight buffer is dropped (rebuilt from "
                        "--class-weights). Row-copy fires ONLY on shape mismatch, so "
                        "binary->binary and 3->3 loads are identical.")
    p.add_argument("--include-noise", action="store_true",
                   help="treat class 4 (Noise) as DLO instead of bg (binary only)")
    # ── recipe parity / domain-aug flags (default OFF -> the same) ──
    p.add_argument("--depth-domain-aug", action="store_true",
                   help="TASK 1: rewrite synth TRAIN depth to look sensor-dense "
                        "(dense bg fill kills the depth==0 shortcut + sensor "
                        "noise/holes + per-frame gain/offset). Train split "
                        "only; NOT applied at eval. Default OFF -> depth tensor "
                        "unchanged from historical behaviour.")
    p.add_argument("--aug2d", action="store_true",
                   help="TASK 3: enable the KD-validated 2D RGB aug stack "
                        "(scale-jitter p=0.3, h-flip p=0.5, colour-jitter "
                        "p=1.0 brightness/contrast/saturation ±0.2 hue ±0.05, "
                        "gaussian blur p=0.3). Geometric ops also transform "
                        "depth+label (registered); photometric ops touch RGB "
                        "ONLY. Train split only. Default OFF -> historical "
                        "flip-only path.")
    p.add_argument("--aug-hue", type=float, default=0.0,
                   help="TASK 3: extra-strong hue randomisation on the RGB "
                        "channel only (p=0.8, hue ±HUE [0.5=full wheel], "
                        "saturation ±0.3). Depth/label untouched. 0.0=off "
                        "(default; no extra RNG draws). Stacks with --aug2d.")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="TASK 3: max_norm for AMP-aware gradient clipping "
                        "(scaler.unscale_ then clip_grad_norm_). <=0 disables "
                        "(default) -> identical optimisation step.")
    p.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT)
    p.add_argument("--pretrained", default=PRETRAINED_DEFAULT)
    p.add_argument("--data-dir", default=DATASET_DIR_DEFAULT,
                   help="path to dformer_dataset (with cache/, train.txt, test.txt). "
                        "Override to $SLURM_TMPDIR/dformer_dataset on HPC.")
    p.add_argument("--smoke", choices=list(SMOKE_PRESETS), default=None,
                   help="run a preset smoke test")
    p.add_argument("--limit-sets", type=int, default=None,
                   help="restrict to the first N train set_ids (overrides smoke set list)")
    p.add_argument("--time-budget", type=float, default=None,
                   help="abort if projected ETA exceeds this many hours")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
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
    train_rgb, train_depth, train_label = build_cache(args.data_dir, "train")
    val_rgb, val_depth, val_label = build_cache(args.data_dir, "val")

    # TASK 2: full-precision depth. Replace the washed-out staged 8-bit depth
    # with the per-frame DLO-windowed cache rebuilt from the raw uint16 render
    # (same N/order as build_cache, indexed identically). Deterministic +
    # identical for train and val. Skipped under --no-depth (depth is zeroed).
    if args.depth_source == "fullprec" and not args.no_depth:
        train_depth = build_depth_fp_cache(args.data_dir, args.render_dir, "train",
                                           verbose=(rank == 0))
        val_depth = build_depth_fp_cache(args.data_dir, args.render_dir, "val",
                                         verbose=(rank == 0))

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

    # Determine which train set_ids to keep
    allowed_sets = None
    if args.limit_sets is not None:
        allowed_sets = first_n_train_set_ids(args.limit_sets)
    elif smoke_cfg is not None and smoke_cfg.get("n_sets") is not None:
        allowed_sets = first_n_train_set_ids(smoke_cfg["n_sets"])

    train_indices = filter_indices_by_set(train_files, allowed_sets)
    if len(train_indices) == 0:
        raise RuntimeError(f"no train images selected (allowed_sets={allowed_sets})")

    val_indices = filter_indices_by_set(val_files, None)  # always full val

    # TASK A: in BINARY mode, auto-detect whether the label cache is the
    # re-encoded {0,1,2} 3-way cache (max<=2) vs the legacy {0..4,255} cache
    # (max=255). Mirrors build_cache's own label-mode auto-detect. On the
    # re-encoded cache we collapse (lbl>=1)->DLO so bg=0 is NOT swept into DLO.
    # 3-way mode is unaffected (passthrough stays unchanged).
    bin_from_3way = False
    if args.num_classes == 2:
        lbl_max = int(np.asarray(train_label[:200]).max())
        bin_from_3way = lbl_max <= 2
        if rank == 0:
            mode = ("re-encoded {0,1,2} -> (lbl>=1) {bg,DLO}" if bin_from_3way
                    else "legacy {0..4,255} -> (lbl<=3) {bg,DLO}")
            print(f"  Binary collapse: {mode} (label-cache sample max={lbl_max})")

    if rank == 0:
        print(f"  Train subset: {len(train_indices)} images "
              f"(sets={allowed_sets if allowed_sets is not None else 'all'})")
        print(f"  Val:          {len(val_indices)} images")

    # RGB 2D aug stack (task 3) — instantiated only when --aug2d (else None ->
    # the historical flip-only path; zero extra RNG draws).
    augmenter2d = RGBDAugmentations() if args.aug2d else None
    train_full = BinaryCDLOMmapDataset(
        train_rgb, train_depth, train_label, augment=True,
        include_noise=args.include_noise,
        augmenter2d=augmenter2d, hue_aug=args.aug_hue,
        depth_domain_aug=args.depth_domain_aug,
        num_classes=args.num_classes, bin_from_3way=bin_from_3way,
    )
    # Eval gets NO aug of any kind (augment=False short-circuits all three) —
    # depth-domain-aug and the RGB augs are train-only.
    val_full = BinaryCDLOMmapDataset(
        val_rgb, val_depth, val_label, augment=False, include_noise=args.include_noise,
        num_classes=args.num_classes, bin_from_3way=bin_from_3way,
    )

    train_dataset = Subset(train_full, train_indices) if allowed_sets is not None else train_full
    val_dataset = val_full

    return train_dataset, val_dataset


def _parse_class_weights(args, device):
    """Return a (num_classes,) float32 CE weight tensor on `device` (task 1)."""
    if args.num_classes == 2:
        return torch.tensor([1.0, float(args.dlo_weight)],
                            dtype=torch.float32, device=device)
    # 3-way
    if args.class_weights:
        parts = [float(x) for x in args.class_weights.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"--class-weights expected 3 comma-sep values for 3-way mode, "
                f"got {args.class_weights!r}")
        return torch.tensor(parts, dtype=torch.float32, device=device)
    # default: heavy emphasis on wire (dlo_weight) + connector (connector_weight),
    # mirroring the SegFormer 3-way reference's [1, dlo_weight, *] convention.
    return torch.tensor([1.0, float(args.dlo_weight), float(args.connector_weight)],
                        dtype=torch.float32, device=device)


def build_model(args, distributed, device):
    cfg = ModelConfig()
    cfg.num_classes = args.num_classes  # TASK 1: 3-way decoder head (Ham, num_classes=3)
    cfg.pretrained_model = None if args.no_pretrained else args.pretrained

    pos_weight = _parse_class_weights(args, device)
    criterion = nn.CrossEntropyLoss(weight=pos_weight, reduction="none", ignore_index=IGNORE_INDEX)
    model = DFormerModel(
        cfg=cfg,
        criterion=criterion,
        norm_layer=nn.SyncBatchNorm if distributed else nn.BatchNorm2d,
        syncbn=distributed,
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[device.index], output_device=device.index,
                    find_unused_parameters=False)
    return model


# Decode-head classifier weight/bias keys for the Ham decoder (LightHamHead's
# BaseDecodeHead.conv_seg). Used for the warm-start head expansion.
_HEAD_KEYS = ("decode_head.conv_seg.weight", "decode_head.conv_seg.bias")


def warmstart_from_checkpoint(model, init_checkpoint, distributed, rank=0):
    """TASK B: load a DFormer checkpoint ON TOP of the freshly-built model
    (after the pretrained backbone). When this build is 3-class and the
    checkpoint is binary, the decode-head classifier [2,C,1,1]->[3,C,1,1] is
    expanded by ROW-COPY (ckpt row 0->bg, row 1->wire, fresh-init connector
    row); the shape-mismatched criterion.weight buffer ([2] vs [3]) is dropped
    and rebuilt from --class-weights. Mirrors src/train_rgb_only_sota.py's
    head-expansion. Row-copy + drop fire ONLY on a shape mismatch, so a
    matching-shape load (binary->binary, 3->3) is identical to a plain
    strict=False load. Returns (missing, unexpected) key lists."""
    if not os.path.isfile(init_checkpoint):
        raise FileNotFoundError(f"--init-checkpoint not found: {init_checkpoint}")
    if rank == 0:
        print(f"  Loading init checkpoint: {init_checkpoint}")
    ckpt = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict)
                 and "model_state_dict" in ckpt) else ckpt
    # strip a DDP 'module.' prefix if present
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {(k[len("module."):] if k.startswith("module.") else k): v
                      for k, v in state_dict.items()}
    target = model.module if isinstance(model, DDP) else model
    tgt_sd = target.state_dict()
    # Head expansion: row-copy the checkpoint's classifier rows into the larger
    # target head; the extra row(s) keep the model's fresh init.
    for ck in _HEAD_KEYS:
        if (ck in state_dict and ck in tgt_sd
                and state_dict[ck].shape != tgt_sd[ck].shape):
            n_old = state_dict[ck].shape[0]
            expanded = tgt_sd[ck].clone()
            expanded[:n_old] = state_dict[ck]
            state_dict[ck] = expanded
            if rank == 0:
                print(f"    head expansion: {ck} {tuple(state_dict[ck].shape)} "
                      f"<- copied {n_old} checkpoint row(s), rest fresh-init")
    # Drop any OTHER shape-mismatched entry (e.g. criterion.weight [2] vs [3]) so
    # the freshly-built model keeps its own value; strict=False would still RAISE
    # on a shape mismatch, so it must be removed before the load.
    for mk in [k for k in list(state_dict)
               if k in tgt_sd and state_dict[k].shape != tgt_sd[k].shape]:
        if rank == 0:
            print(f"    drop shape-mismatched key (keep model init): {mk} "
                  f"ckpt{tuple(state_dict[mk].shape)} vs model{tuple(tgt_sd[mk].shape)}")
        del state_dict[mk]
    missing, unexpected = target.load_state_dict(state_dict, strict=False)
    if rank == 0:
        print(f"    missing keys ({len(missing)}): "
              f"{list(missing)[:5]}{' ...' if len(missing) > 5 else ''}")
        print(f"    unexpected keys ({len(unexpected)}): "
              f"{list(unexpected)[:5]}{' ...' if len(unexpected) > 5 else ''}")
    return missing, unexpected


def fmt_seconds(s):
    return str(datetime.timedelta(seconds=int(max(s, 0))))


def evaluate(model, val_loader, device, distributed, num_classes=2, no_depth=False):
    if num_classes == 3:
        metric = MultiClassIoU(num_classes=3, class_names=CLASS_NAMES_THREE_WAY)
    else:
        metric = BinaryIoU()
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            rgb, depth = normalize_batch(batch["rgb"], batch["depth"], device,
                                         no_depth=no_depth)
            label_np = batch["label"].numpy()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                m = model.module if isinstance(model, DDP) else model
                out = m(rgb, depth)
            pred = out.argmax(dim=1).cpu().numpy()
            for i in range(pred.shape[0]):
                metric.update(pred[i].flatten(), label_np[i].flatten())

    if distributed:
        if isinstance(metric, MultiClassIoU):
            metric.reduce_(device)
        else:
            # All-reduce raw counts
            t = torch.tensor([metric.tp, metric.fp, metric.fn, metric.tn],
                             dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            metric.tp, metric.fp, metric.fn, metric.tn = [int(v.item()) for v in t]

    return metric.compute()


def main():
    args = parse_args()

    smoke_cfg = SMOKE_PRESETS.get(args.smoke) if args.smoke else None
    if smoke_cfg is not None:
        # Override the relevant args from the smoke preset
        args.epochs = smoke_cfg["epochs"]
        args.eval_every = smoke_cfg["eval_every"]
        args.log_every = smoke_cfg["log_every"]
        args.warmup_epochs = 0  # no warmup for smoke
        args.ckpt_every = max(args.epochs, 1)  # only at end

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    distributed, local_rank, rank, world_size = setup_distributed(args.single_gpu)
    device = torch.device(f"cuda:{local_rank}")
    torch.set_float32_matmul_precision("high")

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        tb_dir = os.path.join(args.results_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)
        log_path = os.path.join(args.results_dir, "training.log")
        if args.num_classes == 3:
            mode_str = "3-way {bg, wire, connector}"
            depth_str = ("RGB-ONLY CONTROL (depth ZEROED)" if args.no_depth
                         else f"RGB-D (depth={args.depth_source})")
        else:
            mode_str = "binary {bg, DLO}"
            depth_str = ("RGB-ONLY CONTROL (depth ZEROED)" if args.no_depth
                         else f"RGB-D (depth={args.depth_source})")
        print(f"Training DFormer-v2-Large ({mode_str}) — {depth_str}")
        print(f"  GPUs: {world_size}, batch/GPU: {args.batch_size}, total batch: {args.batch_size * world_size}")
        cw = (f"class weights: {args.class_weights or f'[1, {args.dlo_weight}, {args.connector_weight}]'}"
              if args.num_classes == 3 else f"DLO weight: {args.dlo_weight}")
        print(f"  Epochs: {args.epochs}, LR: {args.lr}, {cw}")
        print(f"  Smoke preset: {args.smoke or 'none'}; results: {args.results_dir}")
    else:
        writer = None
        log_path = None

    # Build cache (rank 0 first)
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
        print(f"  Model: DFormer-v2-Large + ham, {n_params:.1f}M params")

    # TASK B: warm-start from a DFormer checkpoint (loaded ON TOP of the freshly
    # initialised pretrained backbone). See warmstart_from_checkpoint().
    if args.init_checkpoint is not None:
        warmstart_from_checkpoint(model, args.init_checkpoint, distributed, rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(args.epochs * len(train_loader), 1)
    warmup_steps = max(args.warmup_epochs * len(train_loader), 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / warmup_steps, 1e-6)
        return max((1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1)) ** 0.9, 1e-6)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler()

    best_miou = 0.0
    best_iou_dlo = 0.0
    start_time = time.time()
    global_step = 0
    skipped_nonfinite = 0  # count of batches skipped due to non-finite loss
    n_train_imgs = len(train_dataset)
    train_imgs_processed = 0
    eta_aborted = False

    if rank == 0:
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_t0 = time.time()
        epoch_loss = 0.0
        epoch_iters = 0
        batch_t0 = time.time()

        for bi, batch in enumerate(train_loader):
            rgb, depth = normalize_batch(batch["rgb"], batch["depth"], device,
                                         no_depth=args.no_depth)
            label = batch["label"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(rgb, depth, label)

            # ── DDP-safe non-finite-loss skip guard ──────────────────────────
            # fp16 autocast previously overflowed -> NaN loss -> divergence into
            # the all-background degenerate minimum. With bf16 this is far less
            # likely, but a single NaN/Inf batch (e.g. a degenerate depth map)
            # would still poison the optimiser. Skip the step on non-finite loss.
            # MUST stay rank-synchronised: if one rank skipped backward while the
            # other did not, the NCCL collective inside backward would hang. So
            # we all-reduce the finite flag with MIN — if ANY rank is non-finite,
            # ALL ranks skip together.
            finite = torch.isfinite(loss).all()
            flag = torch.tensor(
                [1.0 if bool(finite) else 0.0], device=device, dtype=torch.float32
            )
            if dist.is_initialized():
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if flag.item() < 0.5:
                # Non-finite on at least one rank: skip backward + optimizer step
                # on ALL ranks, but still advance the LR schedule + global_step so
                # the schedule is identical to a clean run (only the update is
                # skipped). Do NOT add the nan/inf to epoch_loss.
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                skipped_nonfinite += 1
                if rank == 0:
                    print(f"    [WARN] non-finite loss at epoch {epoch} batch {bi+1} "
                          f"— skipping step (total skipped: {skipped_nonfinite})")
                continue

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                # AMP-aware: unscale before clipping (matches RGB-only trainer).
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
                  f"peak_GPU={peak_mb:.0f}MB")
            if writer is not None:
                writer.add_scalar("train/loss_epoch", avg_loss, epoch)
                writer.add_scalar("train/epoch_wall_sec", epoch_dt, epoch)
                writer.add_scalar("train/eta_sec", eta, epoch)
                writer.add_scalar("train/peak_gpu_mb", peak_mb, epoch)

        # Time budget check — broadcast abort flag to all ranks so DDP doesn't hang on break.
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

        # Eval
        do_eval = (
            args.eval_every > 0
            and (epoch % args.eval_every == 0 or epoch == args.epochs or epoch == 1)
        )
        if do_eval:
            metrics = evaluate(model, val_loader, device, distributed,
                               num_classes=args.num_classes, no_depth=args.no_depth)
            if rank == 0:
                if args.num_classes == 3:
                    print(f"    val: mIoU={metrics['miou']:.4f}  "
                          f"IoU(wire)={metrics['iou_wire']:.4f}  "
                          f"IoU(connector)={metrics['iou_connector']:.4f}  "
                          f"IoU(bg)={metrics['iou_bg']:.4f}  acc={metrics['pixel_acc']:.4f}  "
                          f"prec(con)={metrics['precision_connector']:.4f}  "
                          f"rec(con)={metrics['recall_connector']:.4f}")
                else:
                    print(f"    val: mIoU={metrics['miou']:.4f}  IoU(DLO)={metrics['iou_dlo']:.4f}  "
                          f"IoU(bg)={metrics['iou_bg']:.4f}  acc={metrics['pixel_acc']:.4f}  "
                          f"prec(DLO)={metrics['precision_dlo']:.4f}  rec(DLO)={metrics['recall_dlo']:.4f}")
                if writer is not None:
                    for k, v in metrics.items():
                        writer.add_scalar(f"val/{k}", v, epoch)

                # Headline metric for best-ckpt selection: mIoU (robust to the
                # single-class selection-spike trap). 3-way connector IoU is
                # logged to tb every eval as val/iou_connector for the depth
                # on-vs-off comparison.
                if args.num_classes == 3:
                    headline = metrics["miou"]
                    class_names_cfg = CLASS_NAMES_THREE_WAY
                else:
                    headline = metrics["iou_dlo"]
                    class_names_cfg = CLASS_NAMES_BINARY
                if headline > best_iou_dlo:
                    best_iou_dlo = headline
                    best_miou = metrics["miou"]
                    ckpt = {
                        "epoch": epoch,
                        "metrics": metrics,
                        "args": vars(args),
                        "model_state_dict": (model.module if isinstance(model, DDP) else model).state_dict(),
                        "config": {"backbone": ModelConfig.backbone, "decoder": ModelConfig.decoder,
                                   "num_classes": args.num_classes, "class_names": class_names_cfg,
                                   "image_size": [IMAGE_H, IMAGE_W]},
                    }
                    torch.save(ckpt, os.path.join(args.results_dir, "best_model.pth"))
                    if args.num_classes == 3:
                        print(f"    ★ new best mIoU={best_miou:.4f}  "
                              f"IoU(connector)={metrics['iou_connector']:.4f}")
                    else:
                        print(f"    ★ new best IoU(DLO)={best_iou_dlo:.4f}  mIoU={best_miou:.4f}")

        # Periodic ckpt
        if rank == 0 and args.ckpt_every > 0 and epoch % args.ckpt_every == 0:
            ckpt = {
                "epoch": epoch,
                "args": vars(args),
                "model_state_dict": (model.module if isinstance(model, DDP) else model).state_dict(),
            }
            torch.save(ckpt, os.path.join(args.results_dir, f"epoch_{epoch}.pth"))

    # Final eval — run on all ranks so all_reduce inside evaluate() doesn't hang.
    final_eval = None
    if args.eval_every > 0:
        try:
            final_eval = evaluate(model, val_loader, device, distributed,
                                  num_classes=args.num_classes, no_depth=args.no_depth)
        except Exception as e:
            if rank == 0:
                print(f"  final eval failed: {e}")

    if rank == 0:
        total_time = time.time() - start_time
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\nDone in {fmt_seconds(total_time)}")
        if best_iou_dlo > 0:
            hl = "mIoU" if args.num_classes == 3 else "IoU(DLO)"
            print(f"Best {hl}={best_iou_dlo:.4f}  mIoU={best_miou:.4f}")
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
            if args.num_classes == 3:
                print(f"Final eval: mIoU={final_eval['miou']:.4f}  "
                      f"IoU(wire)={final_eval['iou_wire']:.4f}  "
                      f"IoU(connector)={final_eval['iou_connector']:.4f}  "
                      f"IoU(bg)={final_eval['iou_bg']:.4f}")
            else:
                print(f"Final eval: mIoU={final_eval['miou']:.4f}  "
                      f"IoU(DLO)={final_eval['iou_dlo']:.4f}  IoU(bg)={final_eval['iou_bg']:.4f}")

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
