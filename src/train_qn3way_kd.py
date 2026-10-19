#!/usr/bin/env python
"""QN-backbone -> 3-way (bg/wire/connector) fine-tuning with anti-forgetting tools.

Wraps ``src/train_rgb_only_sota.py`` (imported, NOT modified): same dataset, augs, losses
(weighted CE + Lovasz), LR schedule, AMP/grad-clip, eval and checkpoint format. With every new
flag at its default this script runs the P3W recipe step-for-step (same RNG consumption order,
same single-group AdamW), see results/qn_backbone_3way/toolkit/TOOLKIT.md for the verification.

New flags (all default OFF):

  --connector-init {random,bg_split}
      random   = trainer default (checkpoint rows copied, connector row = HF fresh init).
      bg_split = teacher head {bg, DLO} -> student {bg, wire, connector}: wire row = DLO row,
                 connector row = bg row, and bg & connector biases get -log 2. Then at step 0
                 P_s(bg)+P_s(conn) == P_t(bg) and P_s(wire) == P_t(DLO) exactly (the QN teacher
                 labels connectors as BACKGROUND, so connector is split out of bg).
  --freeze-stages N      freeze patch-embed + blocks + stage layer-norm of the first N MiT stages
                         (requires_grad=False, and those modules kept in eval() => no drop-path).
  --layer-lr-decay g     encoder stage i (1..4) LR multiplier g**(5-i) (stage4 g, stage1 g^4);
                         decode head x --head-lr-mult (default 1).
  --head-warmup-steps K  LP-FT: for the first K optimizer steps only decode_head.classifier trains
                         (everything else requires_grad=False and in eval mode). Optional
                         --head-warmup-lr sets a constant classifier LR during that phase (the
                         global warmup schedule would otherwise make it ~0).
  --kd-real-dir P[,P..]  unlabeled REAL pools (list .txt files of image paths or directories).
      --kd-weight l --kd-temp T --kd-conf-thresh c --kd-ratio r
      --kd-mode {both,pos_only,neg_weight} (--kd-neg-weight w for neg_weight)
      Each step also draws ceil(r * batch) real images (h-flip + the --aug-zoom crop, geometric
      only), runs the frozen QN teacher (eval, no-grad, fp16) and adds
          l * T^2 * KL( [P_t(bg), P_t(DLO)]_T || [P_s(bg)+P_s(conn), P_s(wire)]_T )
      averaged over active pixels (teacher max-prob at T=1 > c). Pixel weights:
          both       teacher-bg 1, teacher-DLO 1
          pos_only   teacher-bg 0, teacher-DLO 1  (protect wire; don't import teacher misses)
          neg_weight teacher-bg w, teacher-DLO 1
  --kd-on-synth          also apply the KD term on the SYNTHETIC batch, restricted to GT==bg
                         pixels (and the same confidence mask / mode weights), weight
                         --kd-synth-weight (default = --kd-weight). Off by default: on GT-wire/
                         connector pixels GT always wins; on GT-bg pixels it only acts where the
                         teacher is confident (teacher-DLO-on-GT-bg = teacher FP -> in pos_only
                         mode that would PULL towards wire against the GT; use `both` or
                         neg_weight with care, see TOOLKIT.md).
  --feat-anchor-weight m --feat-anchor-stages "4" | "3,4"
                         m * mean_s ||F_s - F_t||^2 of MiT stage outputs on the REAL batch only.

  --pl-list A.txt[,B.txt]  R2-PL: pseudo-labelled REAL frames ('<rgb>\\t<label png>' per line,
                         label 0/1/2, 255 = IGNORE). Each step adds a second supervised batch of
                         ceil(--pl-ratio * batch) PL frames (separate forward, summed into one
                         backward): loss = L_synth + --pl-weight * L_pl, where L_pl has the
                         synthetic loss semantics (class-weighted CE mean over NON-ignored px +
                         Lovasz with the same ignore), optional --pl-class-weights /
                         --pl-lovasz-weight. PL frames get the synthetic geometric augs
                         (--pl-zoom, default --aug-zoom; h-flip .5) and optional photometric
                         (--pl-photometric p, --pl-cj b,c,s,h, --pl-blur-p). Own RNG/generator, so
                         the synthetic shuffle/aug stream is unchanged. Note the parent trainers
                         use IGNORE_INDEX=-1 and do NOT accept 255; the PL dataset remaps 255->-1.

  --hand-paste-dir DIR   A4: paste REAL EgoHands hand cutouts (DIR/{rgb,hand}, LICENCE-PENDING, R&D
                         only) onto SYNTHETIC training frames as labelled bg (src/hand_paste_aug.py).
      --hand-paste-p p (frames that get 1-2 hands) --hand-paste-near-conn q (per-hand prob of
      fingertip placed 10-40 px from a GT connector blob) --hand-paste-scale lo,hi (hand max side
      as frac of H, log-uniform) --hand-paste-max-occ f (never occlude > f of a connector blob)
      --hand-paste-bank PKL (cutout cache). Own numpy RNG -> synthetic/KD/PL streams unchanged.
  --cpaste-bank JSON     A5: CONTRASTIVE cutout paste (src/contrastive_paste_aug.py): connector cutouts
                         (label 2) and clutter/tool cutouts (label 0) through the IDENTICAL compositing
                         pipeline, into synthetic frames (--cpaste-p-synth) and PL real frames
                         (--cpaste-p-pl). --cpaste-n lo,hi objects/frame, --cpaste-pos-frac, --cpaste-scale
                         lo,hi (max side frac of H, logU), --cpaste-wire-end (share placed at a GT wire end),
                         --cpaste-max-occ. Own numpy RNGs -> synthetic/KD/PL streams unchanged.
  --cpaste-wire-bank JSON  A6: add REAL WIRE strand cutouts (MovingCables train clips, label 1) to the
                         contrastive paste (3-way: connector / wire / negative), same compositing.
                         --cpaste-wire-n lo,hi strand events per pasted frame, --cpaste-wire-len lo,hi
                         (bbox max side frac of H, logU), --cpaste-wire-stroke lo,hi (px clamp),
                         --cpaste-wire-cross (prob of a 2-3 strand crossing group), --cpaste-wire-tries,
                         --cpaste-wire-min-vis, --cpaste-wire-max-cut (visible cut ends), --cpaste-wire-cover
                         (prob a visible cut end is hidden under a conn/neg object). Unset => A5 exactly.
  --cpaste-weighted      A7: draw bank cutouts with probability proportional to each entry's "weight"
                         field (default 1). Unset => uniform draw, bitwise identical to A6.

Logging every --log-every steps (stdout + tensorboard kd/*): KD loss, feature loss, teacher
agreement IoU (student argmax==wire vs teacher argmax==DLO, pooled over the window), student fg
(wire|conn) vs teacher DLO IoU, and connector rate on real pixels the teacher calls bg.

New flags require --single-gpu (two forwards per step / requires_grad toggling are not DDP-safe).
"""
import argparse
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_rgb_only_sota as S  # noqa: E402
from train_rgb_only_sota import (  # noqa: E402
    DDP, CLASS_NAMES_BINARY, CLASS_NAMES_THREE_WAY, IMAGE_H, IMAGE_W, IGNORE_INDEX,
    SMOKE_PRESETS, SummaryWriter, SegFormerSegmenter, build_dataset, build_model, evaluate,
    fmt_seconds, lovasz_softmax, normalize_batch, setup_distributed,
)

PROJECT_ROOT = S.PROJECT_ROOT
QN_CKPT_DEFAULT = os.path.join(PROJECT_ROOT, "results", "qn_backbone_3way", "recon",
                               "qn_best_model_hf4.pth")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
FORBIDDEN_SUBSTR = ("real_wires_valset", "dlo_real_sample")
MC_TEST_LIST = os.path.join(PROJECT_ROOT, "data", "dformer_dataset_movingcables", "test.txt")
HEAD_W = "model.decode_head.classifier.weight"
HEAD_B = "model.decode_head.classifier.bias"


# ─────────────────────────── ARGS ───────────────────────────

def parse_kd_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--connector-init", choices=("random", "bg_split"), default="random")
    p.add_argument("--freeze-stages", type=int, default=0, choices=(0, 1, 2, 3, 4))
    p.add_argument("--layer-lr-decay", type=float, default=1.0)
    p.add_argument("--head-lr-mult", type=float, default=1.0)
    p.add_argument("--head-warmup-steps", type=int, default=0)
    p.add_argument("--head-warmup-lr", type=float, default=None)
    p.add_argument("--kd-real-dir", default=None)
    p.add_argument("--kd-pool-probs", default=None,
                   help="comma list of sampling probs per --kd-real-dir entry "
                        "(default: proportional to pool size)")
    p.add_argument("--kd-weight", type=float, default=0.0)
    p.add_argument("--kd-temp", type=float, default=1.0)
    p.add_argument("--kd-conf-thresh", type=float, default=0.0)
    p.add_argument("--kd-ratio", type=float, default=0.5)
    p.add_argument("--kd-mode", choices=("both", "pos_only", "neg_weight"), default="both")
    p.add_argument("--kd-neg-weight", type=float, default=1.0)
    p.add_argument("--kd-on-synth", action="store_true")
    p.add_argument("--kd-synth-weight", type=float, default=None)
    p.add_argument("--kd-real-zoom", default=None,
                   help="'p,lo,hi' crop-zoom for real images (default: same as --aug-zoom, "
                        "else none)")
    p.add_argument("--kd-workers", type=int, default=2)
    p.add_argument("--teacher-ckpt", default=QN_CKPT_DEFAULT)
    p.add_argument("--feat-anchor-weight", type=float, default=0.0)
    p.add_argument("--feat-anchor-stages", default="4")
    # ---- R2-PL: pseudo-labelled REAL frames with ignore (all default OFF) ----
    p.add_argument("--pl-list", default=None,
                   help="comma list of .txt files, one '<rgb>\\t<label png>' pair per line "
                        "(repo-relative or abs). Label PNG values 0=bg 1=wire 2=connector "
                        "255=IGNORE. Enables a second supervised batch per step.")
    p.add_argument("--pl-pool-probs", default=None,
                   help="sampling probs per --pl-list entry (default: proportional to size)")
    p.add_argument("--pl-ratio", type=float, default=0.5,
                   help="PL images per step = ceil(r * --batch-size)")
    p.add_argument("--pl-weight", type=float, default=1.0,
                   help="loss = L_synth + w * L_pl")
    p.add_argument("--pl-zoom", default=None,
                   help="'p,lo,hi' crop-zoom for PL frames (default: same as --aug-zoom)")
    p.add_argument("--pl-photometric", type=float, default=0.0,
                   help="prob of colour jitter on a PL frame (0 = geometric augs only)")
    p.add_argument("--pl-cj", default="0.3,0.3,0.3,0.03",
                   help="colour-jitter magnitudes 'brightness,contrast,saturation,hue'")
    p.add_argument("--pl-blur-p", type=float, default=0.0,
                   help="prob of a Gaussian blur (k=3/5) on a PL frame")
    p.add_argument("--pl-lovasz-weight", type=float, default=None,
                   help="Lovasz weight on the PL batch (default: same as --lovasz-weight)")
    p.add_argument("--pl-class-weights", default=None,
                   help="CE class weights on the PL batch (default: the synthetic criterion's)")
    p.add_argument("--pl-workers", type=int, default=2)
    # ---- A4: real hand cutouts pasted on synthetic frames, labelled bg (all default OFF) ----
    p.add_argument("--hand-paste-dir", default=None)
    p.add_argument("--hand-paste-p", type=float, default=0.4)
    p.add_argument("--hand-paste-near-conn", type=float, default=0.5)
    p.add_argument("--hand-paste-scale", default="0.15,0.6")
    p.add_argument("--hand-paste-max-occ", type=float, default=0.30)
    p.add_argument("--hand-paste-bank", default=None)
    # ---- A5: contrastive connector(2)/clutter(0) cutout paste (all default OFF) ----
    p.add_argument("--cpaste-bank", default=None)
    p.add_argument("--cpaste-p-synth", type=float, default=0.5)
    p.add_argument("--cpaste-p-pl", type=float, default=0.5)
    p.add_argument("--cpaste-n", default="1,3")
    p.add_argument("--cpaste-pos-frac", type=float, default=0.45)
    p.add_argument("--cpaste-scale", default="0.05,0.40")
    p.add_argument("--cpaste-wire-end", type=float, default=0.2)
    p.add_argument("--cpaste-max-occ", type=float, default=0.30)
    p.add_argument("--cpaste-wire-bank", default=None)
    p.add_argument("--cpaste-wire-n", default="1,1")
    p.add_argument("--cpaste-wire-len", default="0.25,0.75")
    p.add_argument("--cpaste-wire-stroke", default="5,26")
    p.add_argument("--cpaste-wire-cross", type=float, default=0.3)
    p.add_argument("--cpaste-wire-tries", type=int, default=40)
    p.add_argument("--cpaste-wire-min-vis", type=float, default=0.5)
    p.add_argument("--cpaste-wire-max-cut", type=int, default=1)
    p.add_argument("--cpaste-wire-cover", type=float, default=0.5)
    p.add_argument("--cpaste-weighted", action="store_true",
                   help="A7: sample bank cutouts by their per-entry 'weight' field (default uniform)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop after this many optimizer steps (smoke tests)")
    p.add_argument("--save-final", action="store_true",
                   help="always write final_model.pth (P3W ckpt format) at the end")
    kd, rest = p.parse_known_args()
    return kd, rest


def parse_all_args():
    kd, rest = parse_kd_args()
    argv0 = sys.argv
    sys.argv = [argv0[0]] + rest
    try:
        args = S.parse_args()
    finally:
        sys.argv = argv0
    for k, v in vars(kd).items():
        setattr(args, k, v)
    return args


def new_flags_on(a):
    return (a.connector_init != "random" or a.freeze_stages > 0 or a.layer_lr_decay != 1.0
            or a.head_lr_mult != 1.0 or a.head_warmup_steps > 0 or kd_active(a)
            or a.feat_anchor_weight > 0.0 or pl_active(a) or a.hand_paste_dir is not None
            or a.cpaste_bank is not None)


def pl_active(a):
    return a.pl_list is not None and a.pl_weight > 0.0


def kd_active(a):
    return a.kd_real_dir is not None and (a.kd_weight > 0.0 or a.feat_anchor_weight > 0.0)


# ─────────────────────────── REAL POOL ───────────────────────────

def read_pool(entry):
    entry = entry.strip()
    full = entry if os.path.isabs(entry) else os.path.join(PROJECT_ROOT, entry)
    if os.path.isdir(full):
        paths = sorted(os.path.join(full, f) for f in os.listdir(full)
                       if f.lower().endswith(IMG_EXT))
    else:
        with open(full) as f:
            paths = [l.strip() for l in f if l.strip()]
        paths = [p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p) for p in paths]
    mc_test = set()
    if os.path.isfile(MC_TEST_LIST):
        with open(MC_TEST_LIST) as f:
            mc_test = {os.path.basename(l.strip()) for l in f if l.strip()}
    bad = [p for p in paths if any(s in p for s in FORBIDDEN_SUBSTR)
           or ("movingcables" in p and os.path.basename(p) in mc_test)]
    if bad:
        raise SystemExit(f"KD pool {entry} contains HELD-OUT evaluation images, e.g. {bad[:3]}")
    if not paths:
        raise SystemExit(f"KD pool {entry} is empty")
    return paths


def cover_resize(bgr, h=IMAGE_H, w=IMAGE_W):
    """Centre-crop to 4:3 then resize to (w, h) (aspect preserved; no-op for 640x480)."""
    ih, iw = bgr.shape[:2]
    if (ih, iw) == (h, w):
        return bgr
    tgt = w / h
    if iw / ih > tgt:
        nw = int(round(ih * tgt))
        x0 = (iw - nw) // 2
        bgr = bgr[:, x0:x0 + nw]
    else:
        nh = int(round(iw / tgt))
        y0 = (ih - nh) // 2
        bgr = bgr[y0:y0 + nh]
    interp = cv2.INTER_AREA if bgr.shape[0] > h else cv2.INTER_LINEAR
    return cv2.resize(bgr, (w, h), interpolation=interp)


class RealUnlabeledDataset(Dataset):
    """Unlabeled real images -> uint8 RGB (3, H, W). Geometric augs only (h-flip p=.5, optional
    crop-zoom). Uses its OWN python RNG (seeded per worker) so it never perturbs the synthetic
    stream's global `random` state."""

    def __init__(self, paths, zoom=None, seed=0):
        self.paths = paths
        self.zoom = zoom
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        bgr = cv2.imread(self.paths[idx], cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"unreadable KD image {self.paths[idx]}")
        bgr = cover_resize(bgr)
        if self.zoom is not None:
            p, lo, hi = self.zoom
            if self.rng.random() < p:
                s = self.rng.uniform(lo, hi)
                nh, nw = max(1, int(round(IMAGE_H * s))), max(1, int(round(IMAGE_W * s)))
                y0, x0 = self.rng.randint(0, IMAGE_H - nh), self.rng.randint(0, IMAGE_W - nw)
                bgr = cv2.resize(bgr[y0:y0 + nh, x0:x0 + nw], (IMAGE_W, IMAGE_H),
                                 interpolation=cv2.INTER_LINEAR)
        if self.rng.random() < 0.5:
            bgr = bgr[:, ::-1]
        rgb = np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1))
        return {"rgb": torch.from_numpy(rgb)}


def _worker_init(wid):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng.seed(info.seed)


def build_real_loader(args, n_real):
    pools = [read_pool(e) for e in args.kd_real_dir.split(",") if e.strip()]
    paths = [p for pool in pools for p in pool]
    if args.kd_pool_probs:
        probs = [float(x) for x in args.kd_pool_probs.split(",")]
        assert len(probs) == len(pools), "--kd-pool-probs must match --kd-real-dir entries"
    else:
        probs = [len(p) for p in pools]
    tot = sum(probs)
    weights = []
    for pool, pr in zip(pools, probs):
        weights += [pr / tot / len(pool)] * len(pool)
    zs = args.kd_real_zoom or args.aug_zoom
    zoom = tuple(float(x) for x in zs.split(",")) if zs else None
    g = torch.Generator()
    g.manual_seed(args.seed + 7919)
    ds = RealUnlabeledDataset(paths, zoom=zoom, seed=args.seed + 7919)
    sampler = WeightedRandomSampler(weights, num_samples=10 ** 8, replacement=True,
                                    generator=g)
    loader = DataLoader(ds, batch_size=n_real, sampler=sampler, num_workers=args.kd_workers,
                        pin_memory=True, drop_last=True, worker_init_fn=_worker_init,
                        persistent_workers=args.kd_workers > 0, generator=g)
    return loader, [len(p) for p in pools], probs, zoom


# ─────────────────────────── PSEUDO-LABELLED REAL POOL (R2-PL) ───────────────────────────

PL_IGNORE = 255


def read_pl_list(entry):
    """'<rgb>\\t<label>' lines -> [(rgb_abs, label_abs)], with the same held-out guards as KD."""
    full = entry.strip()
    full = full if os.path.isabs(full) else os.path.join(PROJECT_ROOT, full)
    pairs = []
    with open(full) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            parts = ln.split("\t")
            if len(parts) != 2:
                raise SystemExit(f"PL list {entry}: expected '<rgb>\\t<label>', got {ln!r}")
            pairs.append(tuple(p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)
                               for p in parts))
    mc_test = set()
    if os.path.isfile(MC_TEST_LIST):
        with open(MC_TEST_LIST) as f:
            mc_test = {os.path.basename(l.strip()) for l in f if l.strip()}
    bad = [r for r, _ in pairs if any(s in r for s in FORBIDDEN_SUBSTR)
           or ("movingcables" in r and os.path.basename(r) in mc_test)]
    if bad:
        raise SystemExit(f"PL list {entry} contains HELD-OUT evaluation images, e.g. {bad[:3]}")
    if not pairs:
        raise SystemExit(f"PL list {entry} is empty")
    return pairs


def cover_resize_label(lbl, h=IMAGE_H, w=IMAGE_W):
    """cover_resize for a label map (same crop, NEAREST)."""
    ih, iw = lbl.shape[:2]
    if (ih, iw) == (h, w):
        return lbl
    tgt = w / h
    if iw / ih > tgt:
        nw = int(round(ih * tgt)); x0 = (iw - nw) // 2
        lbl = lbl[:, x0:x0 + nw]
    else:
        nh = int(round(iw / tgt)); y0 = (ih - nh) // 2
        lbl = lbl[y0:y0 + nh]
    return cv2.resize(lbl, (w, h), interpolation=cv2.INTER_NEAREST)


class PseudoLabelDataset(Dataset):
    """Real RGB + pseudo-label (0/1/2/255). Geometric augs = the synthetic P3W ones (crop-zoom
    then h-flip p=.5; label NEAREST, so 255 stays 255). Optional photometric (colour jitter /
    blur, image only). Own python RNG (seeded per worker) -> never touches the global `random`
    stream that drives the synthetic shuffle/augs."""

    def __init__(self, pairs, zoom=None, cj_p=0.0, cj=(0.3, 0.3, 0.3, 0.03), blur_p=0.0,
                 seed=0):
        self.pairs = pairs
        self.zoom = zoom
        self.cj_p, self.cj, self.blur_p = float(cj_p), cj, float(blur_p)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.pairs)

    def _photometric(self, bgr):
        r = self.rng
        if self.cj_p > 0 and r.random() < self.cj_p:
            b, c, s, hh = self.cj
            x = bgr.astype(np.float32)
            ops = ["b", "c", "s", "h"]
            r.shuffle(ops)
            for op in ops:
                if op == "b" and b > 0:
                    x = x * r.uniform(1 - b, 1 + b)
                elif op == "c" and c > 0:
                    m = x.reshape(-1, 3).mean(0)
                    x = (x - m) * r.uniform(1 - c, 1 + c) + m
                elif op in ("s", "h") and (s > 0 if op == "s" else hh > 0):
                    hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8),
                                       cv2.COLOR_BGR2HSV).astype(np.float32)
                    if op == "s":
                        hsv[..., 1] = np.clip(hsv[..., 1] * r.uniform(1 - s, 1 + s), 0, 255)
                    else:
                        hsv[..., 0] = (hsv[..., 0] + r.uniform(-hh, hh) * 180.0) % 180.0
                    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
            bgr = np.clip(x, 0, 255).astype(np.uint8)
        if self.blur_p > 0 and r.random() < self.blur_p:
            k = r.choice((3, 5))
            bgr = cv2.GaussianBlur(bgr, (k, k), 0)
        return bgr

    def __getitem__(self, idx):
        rp, lp = self.pairs[idx]
        bgr = cv2.imread(rp, cv2.IMREAD_COLOR)
        lbl = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
        if bgr is None or lbl is None:
            raise RuntimeError(f"unreadable PL pair {rp} / {lp}")
        if lbl.ndim == 3:
            lbl = lbl[..., 0]
        bgr, lbl = cover_resize(bgr), cover_resize_label(lbl)
        if self.zoom is not None:
            p, lo, hi = self.zoom
            if self.rng.random() < p:
                s = self.rng.uniform(lo, hi)
                nh, nw = max(1, int(round(IMAGE_H * s))), max(1, int(round(IMAGE_W * s)))
                y0, x0 = self.rng.randint(0, IMAGE_H - nh), self.rng.randint(0, IMAGE_W - nw)
                bgr = cv2.resize(bgr[y0:y0 + nh, x0:x0 + nw], (IMAGE_W, IMAGE_H),
                                 interpolation=cv2.INTER_LINEAR)
                lbl = cv2.resize(lbl[y0:y0 + nh, x0:x0 + nw], (IMAGE_W, IMAGE_H),
                                 interpolation=cv2.INTER_NEAREST)
        if self.rng.random() < 0.5:
            bgr, lbl = bgr[:, ::-1], lbl[:, ::-1]
        bgr = self._photometric(np.ascontiguousarray(bgr))
        lab = np.ascontiguousarray(lbl).astype(np.int64)
        bad = ~np.isin(lab, (0, 1, 2, PL_IGNORE))
        if bad.any():
            raise RuntimeError(f"PL label {lp} has values outside {{0,1,2,255}}: "
                               f"{np.unique(lab[bad])[:5]}")
        lab[lab == PL_IGNORE] = IGNORE_INDEX          # -1: the trainer's CE/Lovasz ignore value
        rgb = np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1))
        return {"rgb": torch.from_numpy(rgb), "label": torch.from_numpy(lab)}


def build_pl_loader(args, n_pl):
    pools = [read_pl_list(e) for e in args.pl_list.split(",") if e.strip()]
    pairs = [p for pool in pools for p in pool]
    if args.pl_pool_probs:
        probs = [float(x) for x in args.pl_pool_probs.split(",")]
        assert len(probs) == len(pools), "--pl-pool-probs must match --pl-list entries"
    else:
        probs = [len(p) for p in pools]
    tot = sum(probs)
    weights = []
    for pool, pr in zip(pools, probs):
        weights += [pr / tot / len(pool)] * len(pool)
    zs = args.pl_zoom or args.aug_zoom
    zoom = tuple(float(x) for x in zs.split(",")) if zs else None
    cj = tuple(float(x) for x in args.pl_cj.split(","))
    assert len(cj) == 4, "--pl-cj needs 4 values b,c,s,h"
    g = torch.Generator()
    g.manual_seed(args.seed + 104729)
    ds = PseudoLabelDataset(pairs, zoom=zoom, cj_p=args.pl_photometric, cj=cj,
                            blur_p=args.pl_blur_p, seed=args.seed + 104729)
    sampler = WeightedRandomSampler(weights, num_samples=10 ** 8, replacement=True, generator=g)
    loader = DataLoader(ds, batch_size=n_pl, sampler=sampler, num_workers=args.pl_workers,
                        pin_memory=True, drop_last=True, worker_init_fn=_worker_init,
                        persistent_workers=args.pl_workers > 0, generator=g)
    return loader, [len(p) for p in pools], probs, zoom


def pl_sup_loss(logits, label, criterion, lovasz_w, lovasz_classes, lovasz_cw):
    """sup_loss semantics (weighted CE mean over valid px + Lovasz), ignore = IGNORE_INDEX (-1;
    PL 255 is remapped to it in the dataset). Safe when a batch is entirely ignored."""
    per_pixel = criterion(logits, label)          # ignore_index=-1 -> 0 on ignored px
    valid = label != IGNORE_INDEX
    if not bool(valid.any()):
        return logits.sum() * 0.0
    loss = per_pixel[valid].mean()
    if lovasz_w > 0.0:
        loss = loss + lovasz_w * lovasz_softmax(logits, label, classes=lovasz_classes,
                                                ignore=IGNORE_INDEX, class_weights=lovasz_cw)
    return loss


# ─────────────────────────── MODEL HELPERS ───────────────────────────

def load_teacher(path, device):
    """QN 2-class teacher, STRICT load (the raw hf5 file loads nothing under strict=False)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    nc = int(sd[HEAD_W].shape[0])
    assert nc == 2, f"teacher must be 2-class, got {nc}"
    t = SegFormerSegmenter(backbone_name="nvidia/mit-b5", num_classes=2,
                           criterion=nn.CrossEntropyLoss(weight=torch.ones(2)))
    t.load_state_dict(sd, strict=True)
    t.eval().to(device)
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def bg_split_head(w2, b2):
    """(2,C,1,1),(2,) teacher head -> 3-class head with P(bg)+P(conn)=P_t(bg), P(wire)=P_t(DLO)."""
    w3 = torch.stack([w2[0], w2[1], w2[0]], 0).clone()
    b3 = torch.stack([b2[0] - math.log(2.0), b2[1], b2[0] - math.log(2.0)], 0).clone()
    return w3, b3


def load_init_checkpoint(model, path, connector_init, rank=0):
    """Same semantics as train_rgb_only_sota (row-copy head expansion, drop shape-mismatched
    criterion.weight) PLUS: strict check that every encoder/decoder key was loaded, and the
    optional bg_split connector init."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = dict(ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck
              else ck)
    tgt = model.module if isinstance(model, DDP) else model
    tsd = tgt.state_dict()
    n_old = sd[HEAD_W].shape[0] if HEAD_W in sd else None
    n_new = tsd[HEAD_W].shape[0]
    if connector_init == "bg_split":
        if not (n_old == 2 and n_new == 3):
            raise SystemExit(f"--connector-init bg_split needs a 2-class ckpt into a 3-class "
                             f"model (got {n_old}->{n_new})")
        sd[HEAD_W], sd[HEAD_B] = bg_split_head(sd[HEAD_W].float(), sd[HEAD_B].float())
        if rank == 0:
            print("    connector-init bg_split: rows (bg-log2, DLO, bg-log2)")
    else:
        for k in (HEAD_W, HEAD_B):
            if k in sd and sd[k].shape != tsd[k].shape:
                e = tsd[k].clone()
                e[:sd[k].shape[0]] = sd[k]
                sd[k] = e
                if rank == 0:
                    print(f"    head expansion: {k} <- copied {n_old} row(s), rest fresh-init")
    for k in [k for k in list(sd) if k in tsd and sd[k].shape != tsd[k].shape]:
        if rank == 0:
            print(f"    drop shape-mismatched key (keep model init): {k}")
        del sd[k]
    missing, unexpected = tgt.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if k != "criterion.weight"]
    if bad_missing or unexpected:
        raise SystemExit(f"init checkpoint key mismatch: missing={bad_missing[:5]} "
                         f"(n={len(bad_missing)}) unexpected={unexpected[:5]} "
                         f"(n={len(unexpected)}) -- wrong key layout? (use the hf4-converted QN)")
    if rank == 0:
        print(f"    init ckpt loaded STRICT on model weights (missing only {missing})")


def stage_modules(seg):
    enc = seg.model.segformer.encoder
    return [[enc.patch_embeddings[i], enc.block[i], enc.layer_norm[i]] for i in range(4)]


def build_param_groups(seg, args):
    """Groups: stage1..4 (LLRD), decode head (non-classifier), classifier."""
    groups, seen = [], set()
    for i, mods in enumerate(stage_modules(seg)):
        ps = [p for m in mods for p in m.parameters()]
        groups.append({"params": ps, "lr": args.lr * args.layer_lr_decay ** (4 - i),
                       "name": f"stage{i + 1}"})
        seen.update(id(p) for p in ps)
    cls = list(seg.model.decode_head.classifier.parameters())
    dec = [p for p in seg.model.decode_head.parameters() if id(p) not in {id(c) for c in cls}]
    groups.append({"params": dec, "lr": args.lr * args.head_lr_mult, "name": "decoder"})
    groups.append({"params": cls, "lr": args.lr * args.head_lr_mult, "name": "classifier"})
    seen.update(id(p) for p in dec + cls)
    allp = list(seg.parameters())
    assert len(seen) == len(allp) and all(id(p) in seen for p in allp), \
        "param groups do not cover the model exactly"
    return groups


def set_trainable(seg, args, head_only):
    """requires_grad + module modes for the current phase. Call after every model.train()."""
    frozen = set()
    for i, mods in enumerate(stage_modules(seg)):
        if i < args.freeze_stages:
            for m in mods:
                frozen.update(id(p) for p in m.parameters())
                m.eval()
    cls_ids = {id(p) for p in seg.model.decode_head.classifier.parameters()}
    for p in seg.parameters():
        p.requires_grad_((id(p) in cls_ids) if head_only else (id(p) not in frozen))
    if head_only:
        # Linear probe on fixed features: no drop-path, decode-head BN uses running stats.
        seg.model.eval()


def student_forward(seg, rgb, want_feats):
    out = seg.model(pixel_values=rgb, output_hidden_states=want_feats)
    logits = F.interpolate(out.logits, size=(IMAGE_H, IMAGE_W), mode="bilinear",
                           align_corners=False)
    return logits, (out.hidden_states if want_feats else None)


def sup_loss(seg, logits, label):
    """Exactly SegFormerSegmenter.forward's loss given logits."""
    per_pixel = seg.criterion(logits, label)
    valid = label != IGNORE_INDEX
    loss = per_pixel.mean() if valid.all() else per_pixel[valid].mean()
    if seg.lovasz_weight > 0.0:
        loss = loss + seg.lovasz_weight * lovasz_softmax(
            logits, label, classes=seg.lovasz_classes, ignore=IGNORE_INDEX,
            class_weights=seg.lovasz_class_weights)
    return loss


def marginal_kd(s_logits, t_logits, T, conf_thresh, pos_w, neg_w, extra_mask=None):
    """Binary-marginal KD. Student 3-way {bg, wire, conn} -> {bg+conn, wire}; teacher {bg, DLO}.
    Returns (loss, n_active_weighted). fp32."""
    # Marginalise at T=1 into binary logits [logsumexp(z_bg, z_conn), z_wire], THEN temper.
    # (Tempering the 3-way logits first would make the bg_split init non-exact for T != 1.)
    s = s_logits.float()
    m = torch.stack([torch.logsumexp(s[:, [0, 2]], 1), s[:, 1]], 1)
    log_q = F.log_softmax(m / T, 1)
    t = t_logits.float()
    log_pt = F.log_softmax(t / T, 1)
    kl = (log_pt.exp() * (log_pt - log_q)).sum(1)                 # (B,H,W)
    with torch.no_grad():
        pt1 = F.softmax(t, 1)
        conf, targ = pt1.max(1)
        w = torch.where(targ == 1, torch.full_like(conf, pos_w), torch.full_like(conf, neg_w))
        w = w * (conf > conf_thresh).float()
        if extra_mask is not None:
            w = w * extra_mask.float()
        wsum = w.sum()
    loss = (kl * w).sum() / wsum.clamp_min(1.0) * (T * T)
    return loss, float(wsum)


# ─────────────────────────── MAIN ───────────────────────────

def main():
    args = parse_all_args()
    if not (0.0 <= args.aug_hue <= 0.5):
        raise SystemExit(f"--aug-hue must be in [0, 0.5]; got {args.aug_hue}")
    if new_flags_on(args) and not args.single_gpu:
        raise SystemExit("the new QN-KD flags require --single-gpu")
    if args.kd_weight > 0 and args.kd_real_dir is None:
        raise SystemExit("--kd-weight needs --kd-real-dir")
    if (args.kd_weight > 0 or args.feat_anchor_weight > 0) and args.num_classes != 3:
        raise SystemExit("KD / feature anchor are defined for --num-classes 3 students")
    if args.kd_mode == "both":
        pos_w, neg_w = 1.0, 1.0
    elif args.kd_mode == "pos_only":
        pos_w, neg_w = 1.0, 0.0
    else:
        pos_w, neg_w = 1.0, float(args.kd_neg_weight)
    kd_synth_w = args.kd_weight if args.kd_synth_weight is None else args.kd_synth_weight

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
        writer = SummaryWriter(os.path.join(args.results_dir, "tb"))
        print(f"QN-KD trainer ({args.backbone}) num_classes={args.num_classes}  "
              f"new-flags={'ON' if new_flags_on(args) else 'OFF (P3W-identical path)'}")
        print(f"  batch {args.batch_size}, epochs {args.epochs}, lr {args.lr}, AMP {use_amp}, "
              f"class weights {args.class_weights}, lovasz {args.lovasz_weight}, "
              f"zoom {args.aug_zoom}")
        print(f"  connector_init={args.connector_init} freeze_stages={args.freeze_stages} "
              f"llrd={args.layer_lr_decay} head_lr_mult={args.head_lr_mult} "
              f"head_warmup={args.head_warmup_steps}@{args.head_warmup_lr}")
        print(f"  KD: pools={args.kd_real_dir} w={args.kd_weight} T={args.kd_temp} "
              f"conf>{args.kd_conf_thresh} ratio={args.kd_ratio} mode={args.kd_mode}"
              f"(pos {pos_w}, neg {neg_w}) on_synth={args.kd_on_synth}; "
              f"feat_anchor={args.feat_anchor_weight}@stages{args.feat_anchor_stages}")
    else:
        writer = None

    # ---- data (identical call order to train_rgb_only_sota.main) ----
    train_dataset, val_dataset = build_dataset(args, rank, world_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    if rank == 0:
        print(f"  Train batches/epoch: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model(args, distributed, device)
    seg = model.module if isinstance(model, DDP) else model
    if args.init_checkpoint is not None:
        if rank == 0:
            print(f"  Loading init checkpoint: {args.init_checkpoint}")
        load_init_checkpoint(model, args.init_checkpoint, args.connector_init, rank)
    elif args.connector_init == "bg_split":
        raise SystemExit("--connector-init bg_split needs --init-checkpoint (the QN teacher)")

    # ---- teacher + real loader (only when KD / anchor active) ----
    teacher, real_iter, n_real = None, None, 0
    use_kd = kd_active(args)
    if use_kd or args.kd_on_synth:
        # fork the RNG so building the teacher (HF from_pretrained head init) does not shift
        # the synthetic shuffle / aug stream vs. a KD-off run with the same seed.
        with torch.random.fork_rng(devices=[device]):
            teacher = load_teacher(args.teacher_ckpt, device)
        if rank == 0:
            print(f"  teacher: {args.teacher_ckpt} (strict load OK, frozen, fp16 inference)")
    if use_kd:
        n_real = max(1, int(math.ceil(args.kd_ratio * args.batch_size)))
        real_loader, sizes, probs, rz = build_real_loader(args, n_real)
        real_iter = iter(real_loader)
        if rank == 0:
            print(f"  KD real pools sizes={sizes} probs={probs} zoom={rz}; "
                  f"{n_real} real imgs/step")
    feat_stages = [int(s) - 1 for s in str(args.feat_anchor_stages).split(",") if s.strip()]

    # ---- R2-PL pseudo-label loader (only when --pl-list and --pl-weight > 0) ----
    use_pl = pl_active(args)
    pl_iter, n_pl, pl_crit = None, 0, None
    if use_pl:
        if args.num_classes != 3:
            raise SystemExit("--pl-list is defined for --num-classes 3 students")
        n_pl = max(1, int(math.ceil(args.pl_ratio * args.batch_size)))
        pl_loader, pl_sizes, pl_probs, pl_z = build_pl_loader(args, n_pl)
        pl_iter = iter(pl_loader)
        if args.pl_class_weights:
            cw = torch.tensor([float(x) for x in args.pl_class_weights.split(",")],
                              dtype=torch.float32, device=device)
            assert cw.numel() == 3, "--pl-class-weights needs 3 values"
            pl_crit = nn.CrossEntropyLoss(weight=cw, reduction="none", ignore_index=IGNORE_INDEX)
        else:
            pl_crit = seg.criterion
        pl_lw = seg.lovasz_weight if args.pl_lovasz_weight is None else float(args.pl_lovasz_weight)
        if rank == 0:
            print(f"  PL lists sizes={pl_sizes} probs={pl_probs} zoom={pl_z} "
                  f"photometric p={args.pl_photometric} cj={args.pl_cj} blur_p={args.pl_blur_p}; "
                  f"{n_pl} PL imgs/step, weight {args.pl_weight}, lovasz {pl_lw}, "
                  f"class weights {args.pl_class_weights or 'same as synth'}; 255 -> ignore")

    # ---- A4 hand paster (only when --hand-paste-dir) ----
    paster = None
    if args.hand_paste_dir is not None:
        from hand_paste_aug import build_bank, HandPaster
        bank = build_bank(args.hand_paste_dir, cache=args.hand_paste_bank, verbose=rank == 0)
        lo, hi = (float(x) for x in args.hand_paste_scale.split(","))
        paster = HandPaster(bank, p=args.hand_paste_p, near_p=args.hand_paste_near_conn,
                            scale=(lo, hi), max_occ=args.hand_paste_max_occ,
                            seed=args.seed + 271828)
        if rank == 0:
            print(f"  HAND PASTE (EgoHands, LICENCE-PENDING): {len(bank)} cutouts, "
                  f"p={args.hand_paste_p} near_conn={args.hand_paste_near_conn} "
                  f"scale={lo},{hi} max_occ={args.hand_paste_max_occ}")

    # ---- A5 contrastive pasters (only when --cpaste-bank) ----
    cp_syn = cp_pl = None
    if args.cpaste_bank is not None:
        from contrastive_paste_aug import load_bank, ContrastivePaster
        cbank = load_bank(args.cpaste_bank, verbose=rank == 0)
        nlo, nhi = (int(x) for x in args.cpaste_n.split(","))
        slo, shi = (float(x) for x in args.cpaste_scale.split(","))
        ckw = dict(n_range=(nlo, nhi), pos_frac=args.cpaste_pos_frac, scale=(slo, shi),
                   wire_end_p=args.cpaste_wire_end, max_occ=args.cpaste_max_occ)
        if args.cpaste_weighted:                    # A7: per-entry bank sampling weights
            ckw.update(weighted=True)
            if rank == 0:
                import collections
                wc = collections.Counter((b["label"], b["weight"]) for b in cbank)
                print(f"  A7 WEIGHTED BANK SAMPLING: (label, weight) counts {dict(sorted(wc.items()))}")
        if args.cpaste_wire_bank is not None:      # A6: 3-way contrastive (adds real wire strands)
            from contrastive_paste_aug import load_wire_bank
            wbank = load_wire_bank(args.cpaste_wire_bank, verbose=rank == 0)
            pr = lambda t, f: tuple(f(x) for x in t.split(","))
            ckw.update(wire_bank=wbank, wire_n=pr(args.cpaste_wire_n, int),
                       wire_len=pr(args.cpaste_wire_len, float), wire_stroke=pr(args.cpaste_wire_stroke, float),
                       wire_cross=args.cpaste_wire_cross, wire_tries=args.cpaste_wire_tries,
                       wire_min_vis=args.cpaste_wire_min_vis, wire_max_cut=args.cpaste_wire_max_cut,
                       wire_cover=args.cpaste_wire_cover)
            if rank == 0:
                print(f"  A6 WIRE PASTE: n={args.cpaste_wire_n} len={args.cpaste_wire_len} "
                      f"stroke={args.cpaste_wire_stroke} cross={args.cpaste_wire_cross} "
                      f"tries={args.cpaste_wire_tries} min_vis={args.cpaste_wire_min_vis} "
                      f"max_cut={args.cpaste_wire_max_cut} cover={args.cpaste_wire_cover}")
        if args.cpaste_p_synth > 0:
            cp_syn = ContrastivePaster(cbank, p=args.cpaste_p_synth, seed=args.seed + 314159, **ckw)
        if use_pl and args.cpaste_p_pl > 0:
            cp_pl = ContrastivePaster(cbank, p=args.cpaste_p_pl, seed=args.seed + 161803, **ckw)
        if rank == 0:
            print(f"  CONTRASTIVE PASTE: p_synth={args.cpaste_p_synth} p_pl={args.cpaste_p_pl if use_pl else 0} "
                  f"n={nlo}-{nhi} pos_frac={args.cpaste_pos_frac} scale={slo},{shi} "
                  f"wire_end={args.cpaste_wire_end} max_occ={args.cpaste_max_occ}")

    # ---- optimizer (single group when no LR flags => identical to sota) ----
    if (args.layer_lr_decay != 1.0 or args.head_lr_mult != 1.0 or args.head_warmup_steps > 0
            or args.freeze_stages > 0):
        opt_params = build_param_groups(seg, args)
    else:
        opt_params = model.parameters()
    optimizer = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.freeze_stages > 0:
        set_trainable(seg, args, head_only=False)
        n_tr = sum(p.numel() for p in seg.parameters() if p.requires_grad) / 1e6
        if rank == 0:
            print(f"  frozen stages 1..{args.freeze_stages}; trainable {n_tr:.1f}M params")
    cls_group = next((g for g in optimizer.param_groups if g.get("name") == "classifier"), None)

    total_steps = max(args.epochs * len(train_loader), 1)
    warmup_steps = max(args.warmup_epochs * len(train_loader), 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / warmup_steps, 1e-6)
        return max((1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1)) ** 0.9,
                   1e-6)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_miou, best_head = 0.0, 0.0
    start_time = time.time()
    global_step, start_epoch = 0, 1
    train_imgs_processed = 0
    if args.resume is not None:
        rs = torch.load(args.resume, map_location="cpu", weights_only=False)
        seg.load_state_dict(rs["model_state_dict"], strict=True)
        start_epoch = int(rs.get("epoch", 0)) + 1
        global_step = (start_epoch - 1) * len(train_loader)
        for _ in range(global_step):
            scheduler.step()
        m = rs.get("metrics", {}) or {}
        hk = (("iou_connector" if args.select_metric == "iou_con" else args.select_metric)
              if args.select_metric != "auto" else "iou_wire")
        best_head, best_miou = float(m.get(hk, 0.0) or 0.0), float(m.get("miou", 0.0) or 0.0)
        if rank == 0:
            print(f"  RESUMED {args.resume} -> start_epoch {start_epoch}, step {global_step}")

    def apply_phase(step):
        """(re)apply requires_grad/eval modes; returns True while in the head-only phase."""
        head_only = step < args.head_warmup_steps
        if args.head_warmup_steps > 0 or args.freeze_stages > 0:
            set_trainable(seg, args, head_only)
        if head_only and cls_group is not None and args.head_warmup_lr is not None:
            cls_group["lr"] = args.head_warmup_lr
        return head_only

    win = {"kd": 0.0, "kds": 0.0, "feat": 0.0, "n": 0, "tp": 0, "fp": 0, "fn": 0,
           "ftp": 0, "ffp": 0, "ffn": 0, "tbg": 0, "conn_on_tbg": 0, "act": 0.0,
           "pl": 0.0, "pl_valid": 0, "pl_px": 0, "pl_ok": 0,
           "pl_wi": 0, "pl_wu": 0, "pl_ci": 0, "pl_cu": 0, "pl_bg": 0, "pl_fg_on_bg": 0}
    if rank == 0:
        torch.cuda.reset_peak_memory_stats()
    stop = False
    epoch = start_epoch - 1
    epoch_iters = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        head_only = apply_phase(global_step)
        epoch_t0 = time.time()
        epoch_loss, epoch_iters, nan_in_epoch = 0.0, 0, 0
        batch_t0 = time.time()
        for bi, batch in enumerate(train_loader):
            if head_only and global_step >= args.head_warmup_steps:
                model.train()
                head_only = apply_phase(global_step)
                if rank == 0:
                    print(f"  [LP-FT] head warm-up done at step {global_step}: full FT")
            if paster is not None:
                paster.apply_batch(batch["rgb"], batch["label"])
            if cp_syn is not None:
                cp_syn.apply_batch(batch["rgb"], batch["label"])
            rgb = normalize_batch(batch["rgb"], device)
            label = batch["label"].to(device, non_blocking=True)

            need_synth_logits = args.kd_on_synth
            kd_loss_v = feat_v = kds_v = None
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                if need_synth_logits:
                    s_logits, _ = student_forward(seg, rgb, False)
                    loss = sup_loss(seg, s_logits, label)
                else:
                    loss = model(rgb, label)   # the exact P3W path
            if args.kd_on_synth:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    t_syn = teacher(rgb)
                kds, _ = marginal_kd(s_logits, t_syn, args.kd_temp, args.kd_conf_thresh,
                                     pos_w, neg_w, extra_mask=(label == 0))
                loss = loss + kd_synth_w * kds
                kds_v = float(kds.detach())

            if use_kd:
                rb = next(real_iter)
                rrgb = normalize_batch(rb["rgb"], device)
                want_f = args.feat_anchor_weight > 0
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    tout = teacher.model(pixel_values=rrgb, output_hidden_states=want_f)
                    t_logits = F.interpolate(tout.logits, size=(IMAGE_H, IMAGE_W),
                                             mode="bilinear", align_corners=False)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    r_logits, r_feats = student_forward(seg, rrgb, want_f)
                if args.kd_weight > 0:
                    kd, act = marginal_kd(r_logits, t_logits, args.kd_temp,
                                          args.kd_conf_thresh, pos_w, neg_w)
                    loss = loss + args.kd_weight * kd
                    kd_loss_v = float(kd.detach())
                    win["act"] += act / r_logits[:, 0].numel()
                if want_f:
                    fl = sum(F.mse_loss(r_feats[s].float(), tout.hidden_states[s].float())
                             for s in feat_stages) / len(feat_stages)
                    loss = loss + args.feat_anchor_weight * fl
                    feat_v = float(fl.detach())
                with torch.no_grad():
                    sp = r_logits.argmax(1)
                    tp_ = t_logits.argmax(1) == 1
                    sw = sp == 1
                    sf = sp != 0
                    win["tp"] += int((sw & tp_).sum()); win["fp"] += int((sw & ~tp_).sum())
                    win["fn"] += int((~sw & tp_).sum())
                    win["ftp"] += int((sf & tp_).sum()); win["ffp"] += int((sf & ~tp_).sum())
                    win["ffn"] += int((~sf & tp_).sum())
                    win["tbg"] += int((~tp_).sum()); win["conn_on_tbg"] += int(((sp == 2) & ~tp_).sum())

            pl_loss_v = None
            if use_pl:
                pb = next(pl_iter)
                if cp_pl is not None:
                    cp_pl.apply_batch(pb["rgb"], pb["label"])
                prgb = normalize_batch(pb["rgb"], device)
                plab = pb["label"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    p_logits, _ = student_forward(seg, prgb, False)
                    pl = pl_sup_loss(p_logits, plab, pl_crit, pl_lw, seg.lovasz_classes,
                                     seg.lovasz_class_weights)
                loss = loss + args.pl_weight * pl
                pl_loss_v = float(pl.detach())
                with torch.no_grad():
                    sp = p_logits.argmax(1)
                    v = plab != IGNORE_INDEX
                    win["pl_valid"] += int(v.sum()); win["pl_px"] += plab.numel()
                    win["pl_ok"] += int(((sp == plab) & v).sum())
                    for c, ki, ku in ((1, "pl_wi", "pl_wu"), (2, "pl_ci", "pl_cu")):
                        a_, b_ = (sp == c) & v, (plab == c)
                        win[ki] += int((a_ & b_).sum()); win[ku] += int((a_ | b_).sum())
                    bg_ = plab == 0
                    win["pl_bg"] += int(bg_.sum()); win["pl_fg_on_bg"] += int(((sp != 0) & bg_).sum())

            if not torch.isfinite(loss):
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
            global_step += 1
            if global_step < args.head_warmup_steps and cls_group is not None \
                    and args.head_warmup_lr is not None:
                cls_group["lr"] = args.head_warmup_lr

            lv = float(loss.item())
            epoch_loss += lv
            epoch_iters += 1
            train_imgs_processed += rgb.size(0) * world_size
            win["n"] += 1
            win["kd"] += kd_loss_v or 0.0
            win["kds"] += kds_v or 0.0
            win["feat"] += feat_v or 0.0
            win["pl"] += pl_loss_v or 0.0

            if rank == 0 and (bi + 1) % args.log_every == 0:
                dt = time.time() - batch_t0
                ips = args.log_every * args.batch_size * world_size / max(dt, 1e-9)
                lr = optimizer.param_groups[-1]["lr"]   # classifier group (== the only group w/o LR flags)
                msg = (f"    ep{epoch:3d} batch {bi+1:4d}/{len(train_loader):4d}  "
                       f"loss={lv:.4f}  lr={lr:.2e}  {ips:.1f} img/s")
                if use_kd or args.kd_on_synth:
                    n = max(win["n"], 1)
                    agree = win["tp"] / max(win["tp"] + win["fp"] + win["fn"], 1)
                    fagree = win["ftp"] / max(win["ftp"] + win["ffp"] + win["ffn"], 1)
                    crate = win["conn_on_tbg"] / max(win["tbg"], 1)
                    msg += (f"  kd={win['kd']/n:.4f} kd_syn={win['kds']/n:.4f} "
                            f"feat={win['feat']/n:.5f} active={win['act']/n:.3f} "
                            f"agreeIoU(wire|DLO)={agree:.4f} agreeIoU(fg|DLO)={fagree:.4f} "
                            f"conn@t_bg={crate:.5f}")
                    if writer is not None:
                        for k, v in (("kd", win["kd"] / n), ("kd_synth", win["kds"] / n),
                                     ("feat", win["feat"] / n), ("agree_iou_wire", agree),
                                     ("agree_iou_fg", fagree), ("conn_rate_teacher_bg", crate),
                                     ("active_frac", win["act"] / n)):
                            writer.add_scalar(f"kd/{k}", v, global_step)
                if use_pl:
                    n = max(win["n"], 1)
                    pv = win["pl_valid"] / max(win["pl_px"], 1)
                    pacc = win["pl_ok"] / max(win["pl_valid"], 1)
                    pwi = win["pl_wi"] / max(win["pl_wu"], 1)
                    pci = win["pl_ci"] / max(win["pl_cu"], 1)
                    pfb = win["pl_fg_on_bg"] / max(win["pl_bg"], 1)
                    msg += (f"  pl={win['pl']/n:.4f} pl_valid={pv:.3f} pl_acc={pacc:.4f} "
                            f"plIoU(wire)={pwi:.4f} plIoU(conn)={pci:.4f} fg@pl_bg={pfb:.5f}")
                    if writer is not None:
                        for k, v in (("loss", win["pl"] / n), ("valid_frac", pv), ("acc", pacc),
                                     ("iou_wire", pwi), ("iou_conn", pci), ("fg_rate_pl_bg", pfb)):
                            writer.add_scalar(f"pl/{k}", v, global_step)
                if paster is not None:
                    hs = paster.pop_stats()
                    msg += (f"  hp: frames={hs['pasted_frames']}/{hs['frames']} "
                            f"hands={hs['hands']} near={hs['near_ok']}/{hs['near_req']} "
                            f"skip={hs['skipped']} occ_w={hs['px_wire_occ']} "
                            f"occ_c={hs['px_conn_occ']}/{hs['conn_px']}")
                if cp_syn is not None:
                    msg += "  cp_syn: " + ContrastivePaster.fmt(cp_syn.pop_stats())
                if cp_pl is not None:
                    msg += "  cp_pl: " + ContrastivePaster.fmt(cp_pl.pop_stats())
                print(msg, flush=True)
                if writer is not None:
                    writer.add_scalar("train/loss_step", lv, global_step)
                    writer.add_scalar("train/imgs_per_sec", ips, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                for k in win:
                    win[k] = 0 if isinstance(win[k], int) else 0.0
                batch_t0 = time.time()
            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break

        epoch_dt = time.time() - epoch_t0
        if rank == 0:
            peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={epoch_loss/max(epoch_iters,1):.4f}  "
                  f"epoch_wall={fmt_seconds(epoch_dt)}  peak_GPU={peak_mb:.0f}MB"
                  + (f"  NaN_skipped={nan_in_epoch}" if nan_in_epoch else ""), flush=True)
            if writer is not None:
                writer.add_scalar("train/loss_epoch", epoch_loss / max(epoch_iters, 1), epoch)
                writer.add_scalar("train/peak_gpu_mb", peak_mb, epoch)
        if stop:
            break

        do_eval = args.eval_every > 0 and (epoch % args.eval_every == 0
                                           or epoch == args.epochs or epoch == 1)
        if do_eval:
            metrics = evaluate(model, val_loader, device, distributed, use_amp,
                               num_classes=args.num_classes)
            if rank == 0:
                print("    val: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                              if isinstance(v, float)))
                if writer is not None:
                    for k, v in metrics.items():
                        writer.add_scalar(f"val/{k}", v, epoch)
                hk = (("iou_connector" if args.select_metric == "iou_con" else args.select_metric)
                      if args.select_metric != "auto"
                      else ("iou_dlo" if args.num_classes == 2 else "iou_wire"))
                if metrics[hk] > best_head:
                    best_head, best_miou = metrics[hk], metrics["miou"]
                    save_ckpt(args, seg, epoch, metrics, "best_model.pth")
                    print(f"    * new best {hk}={best_head:.4f} mIoU={best_miou:.4f}")
        if rank == 0 and args.ckpt_every > 0 and epoch % args.ckpt_every == 0:
            save_ckpt(args, seg, epoch, None, f"epoch_{epoch}.pth")

    if rank == 0:
        total = time.time() - start_time
        peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
        if args.save_final:
            save_ckpt(args, seg, epoch, None, "final_model.pth")
        print(f"\nDone in {fmt_seconds(total)}  steps={global_step}  peak GPU {peak_mb:.0f} MB  "
              f"{train_imgs_processed / max(total, 1e-9):.1f} synth img/s overall")
        import json
        with open(os.path.join(args.results_dir, "report.json"), "w") as f:
            json.dump({"args": vars(args), "wall_seconds": total, "peak_gpu_mb": peak_mb,
                       "best_headline": best_head, "best_miou": best_miou,
                       "steps": global_step}, f, indent=2, default=str)
        if writer is not None:
            writer.close()


def save_ckpt(args, seg, epoch, metrics, name):
    """P3W checkpoint format (epoch/metrics/args/model_state_dict/config)."""
    ck = {"epoch": epoch, "args": vars(args), "model_state_dict": seg.state_dict(),
          "config": {"backbone": args.backbone, "num_classes": args.num_classes,
                     "class_names": (CLASS_NAMES_BINARY if args.num_classes == 2
                                     else CLASS_NAMES_THREE_WAY),
                     "image_size": [IMAGE_H, IMAGE_W]}}
    if metrics is not None:
        ck["metrics"] = metrics
    torch.save(ck, os.path.join(args.results_dir, name))


if __name__ == "__main__":
    main()
