"""Knowledge-distil a small SegFormer student from the Phase 7 SegFormer-B5 teacher.

Phase 9 / KD-student deliverable. Spec: llmdocs/CONTEXT.md §0.14.

Teacher: nvidia/mit-b5  (84.6M params, results/segformer_b5_rgb/full_20260430_2032/best_model.pth)
Student: nvidia/mit-b0  (~3.8M, default) or nvidia/mit-b1 (~13.7M) via --backbone

Loss:    α × CE(student, hard_label) + (1-α) × KL(student/T || teacher/T) × T²
         α = 0.5, T = 4.0 (matches the 5-class KD recipe in src/train_kd.py)

Trained on the Phase 4 dataset (data/rgbd_videos/, NOT v2). 2D RGB augmentations
(color jitter, Gaussian blur, scale jitter) on the training data only — see §0.14
in CONTEXT.md and the v2 post-mortem (results/postmortem/POSTMORTEM.md) for why
augmentations matter here. The teacher and student see the SAME augmented image
per __getitem__ call (standard KD); labels track geometric augs.

Reuses the Phase 7 plumbing verbatim:
    - build_cache from src/train_rgbd_seg.py (mmap'd .npy cache)
    - SegFormerSegmenter wrapper from train_rgb_only_sota.py (HF model + bilinear upsample)
    - BinaryIoU metric, normalize_batch, lr_lambda, AMP, grad clip 1.0,
      isfinite-loss skip-batch, eval every epoch, ckpt every 5.

Usage (run from project root, with env activated):
    source env/bin/activate
    # smoke (1 set, 1 epoch) to verify wire-up:
    CUDA_VISIBLE_DEVICES=0 python src/train_rgb_only_kd.py --single-gpu --smoke a --batch-size 4
    # smoke b (3 sets, 5 epochs, learnability):
    CUDA_VISIBLE_DEVICES=0 python src/train_rgb_only_kd.py --single-gpu --smoke b --batch-size 4
    # full single-GPU run:
    CUDA_VISIBLE_DEVICES=0 python src/train_rgb_only_kd.py --single-gpu --epochs 80 --batch-size 8

Output: results/segformer_<backbone-tag>_rgb_kd/full_<timestamp>/
        best_model.pth, epoch_*.pth, report.json, tb/.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rgbd_seg import build_cache  # noqa: E402
from train_rgb_only_sota import (  # noqa: E402
    SegFormerSegmenter,
    NUM_CLASSES,
    CLASS_NAMES,
    IGNORE_INDEX,
    IMAGE_H,
    IMAGE_W,
    RGB_MEAN,
    RGB_STD,
    BinaryIoU,
    file_list,
    filter_indices_by_set,
)

# ─────────────────────────── CONFIG ───────────────────────────

DATASET_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "data", "dformer_dataset")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")

TEACHER_CKPT_DEFAULT = os.path.join(
    PROJECT_ROOT, "results", "segformer_b5_rgb", "full_20260430_2032", "best_model.pth"
)
TEACHER_BACKBONE_DEFAULT = "nvidia/mit-b5"
STUDENT_BACKBONE_DEFAULT = "nvidia/mit-b0"


# ─────────────────────────── AUGMENTATIONS ───────────────────────────


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


# ─────────────────────────── DATASET ───────────────────────────


class BinaryCDLOKDDataset(Dataset):
    """RGB-only mmap CDLO dataset for KD training.

    Cache labels (gt_transform applied): 0=Wire, 1=Endpoint, 2=Bifurcation, 3=Connector,
                                          4=Noise, 255=bg.
    Binary collapse: classes 0..3 → 1 (DLO); class 4 (Noise) and 255 (bg) → 0 (bg).

    With augmentations enabled (training), augs are applied BEFORE binary collapse so
    the geometric augs preserve label-class identity. Both teacher and student receive
    the same augmented RGB through `normalize_batch` downstream.
    """

    def __init__(self, rgb, label, augment=True, include_noise=False, augmenter=None):
        self.rgb = rgb
        self.label = label
        self.augment = augment
        self.include_noise = include_noise
        self.augmenter = augmenter if augmenter is not None else RGBAugmentations()

    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, idx):
        rgb = self.rgb[idx].copy()      # (H, W, 3) uint8 BGR
        lbl = self.label[idx].copy()     # (H, W) uint8 — 0..4 fg, 255 bg

        if self.augment:
            rgb, lbl = self.augmenter(rgb, lbl)

        if self.include_noise:
            binary = (lbl <= 4).astype(np.int64)
        else:
            binary = (lbl <= 3).astype(np.int64)

        rgb_rgb = rgb[:, :, ::-1].copy()
        rgb_t = torch.from_numpy(rgb_rgb.transpose(2, 0, 1).copy())  # (3, H, W) uint8 RGB
        label_t = torch.from_numpy(binary)
        return {"rgb": rgb_t, "label": label_t}


def normalize_batch(rgb_uint8, device):
    rgb = rgb_uint8.to(device, dtype=torch.float32, non_blocking=True) / 255.0
    rgb = (rgb - RGB_MEAN.to(device)) / RGB_STD.to(device)
    return rgb


# ─────────────────────────── TEACHER LOAD ───────────────────────────


def load_teacher(ckpt_path, fallback_backbone, device):
    """Load the frozen teacher exactly as gen_rgb_only_sota_gifs.load_model does."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = state.get("args") or {}
    cfg = state.get("config") or {}
    backbone_name = cfg.get("backbone") or saved_args.get("backbone") or fallback_backbone
    teacher = SegFormerSegmenter(
        backbone_name=backbone_name, num_classes=NUM_CLASSES, criterion=None
    )
    sd = state.get("model_state_dict", state)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  Teacher load: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"    first missing: {missing[:3]}")
        if unexpected:
            print(f"    first unexpected: {unexpected[:3]}")
    teacher = teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher, backbone_name


# ─────────────────────────── KD LOSS ───────────────────────────


def kd_loss(student_logits, teacher_logits, label, ce_criterion, alpha, temperature):
    """α × CE(student, hard_label) + (1-α) × KL(student/T || teacher/T) × T²

    student_logits / teacher_logits: (B, C, H, W) raw logits, full resolution.
    label: (B, H, W) int64 in {0, 1} (binary). No IGNORE_INDEX in binary mode.
    ce_criterion: nn.CrossEntropyLoss(weight=..., reduction="none", ignore_index=-1).

    KL is flattened to (B*H*W, C) so reduction="batchmean" divides by N_pixels —
    matching the prior 5-class KD recipe (src/train_kd.py:kd_loss) and keeping the
    KL term on a per-pixel scale comparable to CE.
    """
    per_pixel = ce_criterion(student_logits, label)
    valid = (label != IGNORE_INDEX)
    if valid.all():
        ce = per_pixel.mean()
    else:
        ce = per_pixel[valid].mean()

    B, C, H, W = student_logits.shape
    s_flat = student_logits.permute(0, 2, 3, 1).reshape(-1, C)
    t_flat = teacher_logits.permute(0, 2, 3, 1).reshape(-1, C)
    s_log_soft = F.log_softmax(s_flat / temperature, dim=1)
    t_soft = F.softmax(t_flat / temperature, dim=1)
    kl = F.kl_div(s_log_soft, t_soft, reduction="batchmean") * (temperature * temperature)

    total = alpha * ce + (1.0 - alpha) * kl
    return total, ce, kl


# ─────────────────────────── TRAINING ───────────────────────────


SMOKE_PRESETS = {
    "a": {"n_sets": 1, "epochs": 1, "eval_every": 1, "log_every": 1, "label": "code-runs"},
    "b": {"n_sets": 3, "epochs": 5, "eval_every": 1, "log_every": 5, "label": "learnability"},
    "c": {"n_sets": None, "epochs": 1, "eval_every": 0, "log_every": 5, "label": "throughput"},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--single-gpu", action="store_true")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--dlo-weight", type=float, default=6.0)
    p.add_argument("--include-noise", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--alpha", type=float, default=0.5, help="weight for hard-label CE")
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--results-root", default=RESULTS_ROOT)
    p.add_argument("--run-tag", default=None,
                   help="if not set, defaults to 'full_<timestamp>' (or 'smoke_<preset>')")
    p.add_argument("--backbone", default=STUDENT_BACKBONE_DEFAULT,
                   help=f"HF student backbone id (default {STUDENT_BACKBONE_DEFAULT})")
    p.add_argument("--teacher-ckpt", default=TEACHER_CKPT_DEFAULT)
    p.add_argument("--teacher-backbone-fallback", default=TEACHER_BACKBONE_DEFAULT,
                   help="used only if the teacher checkpoint has no config/args (legacy)")
    p.add_argument("--data-dir", default=DATASET_DIR_DEFAULT)
    p.add_argument("--smoke", choices=list(SMOKE_PRESETS), default=None)
    p.add_argument("--limit-sets", type=int, default=None)
    p.add_argument("--no-augment", action="store_true",
                   help="disable RGB augmentations (debug; not recommended)")
    p.add_argument("--time-budget", type=float, default=None,
                   help="abort if projected total wall (hours) > this")
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


def build_dataset(args, rank):
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

    do_augment = not args.no_augment
    augmenter = RGBAugmentations() if do_augment else None

    train_full = BinaryCDLOKDDataset(
        train_rgb, train_label, augment=do_augment,
        include_noise=args.include_noise, augmenter=augmenter,
    )
    val_full = BinaryCDLOKDDataset(
        val_rgb, val_label, augment=False, include_noise=args.include_noise,
    )
    train_dataset = Subset(train_full, train_indices) if allowed_sets is not None else train_full
    return train_dataset, val_full


def build_student(args, distributed, device):
    pos_weight = torch.tensor([1.0, float(args.dlo_weight)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=pos_weight, reduction="none", ignore_index=IGNORE_INDEX)
    student = SegFormerSegmenter(
        backbone_name=args.backbone, num_classes=NUM_CLASSES, criterion=None
    ).to(device)
    if distributed:
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        student = DDP(student, device_ids=[device.index], output_device=device.index,
                      find_unused_parameters=False)
    return student, criterion


def fmt_seconds(s):
    return str(datetime.timedelta(seconds=int(max(s, 0))))


def evaluate(model, val_loader, device, distributed, use_amp):
    metric = BinaryIoU()
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
        t = torch.tensor([metric.tp, metric.fp, metric.fn, metric.tn], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        metric.tp, metric.fp, metric.fn, metric.tn = [int(v.item()) for v in t]

    return metric.compute()


def make_run_tag(args):
    if args.run_tag is not None:
        return args.run_tag
    if args.smoke:
        return f"smoke_{args.smoke}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    return f"full_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"


def backbone_tag(backbone_name):
    # 'nvidia/mit-b0' -> 'b0'; 'mit-b3' -> 'b3'; fall back to last token after '-' / '/'.
    base = backbone_name.split("/")[-1]
    if base.startswith("mit-"):
        return base[len("mit-"):]
    return base.replace("/", "_")


def main():
    args = parse_args()

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

    # Compute output dir (run-tag subdir per Phase 7 convention)
    bk_tag = backbone_tag(args.backbone)
    run_tag = make_run_tag(args)
    results_dir = os.path.join(args.results_root, f"segformer_{bk_tag}_rgb_kd", run_tag)

    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        tb_dir = os.path.join(results_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)
        print(f"KD: teacher={args.teacher_ckpt}")
        print(f"    student backbone={args.backbone}")
        print(f"    output dir: {results_dir}")
        print(f"  GPUs: {world_size}, batch/GPU: {args.batch_size}, "
              f"total batch: {args.batch_size * world_size}")
        print(f"  Epochs: {args.epochs}, LR: {args.lr}, DLO weight: {args.dlo_weight}, "
              f"AMP: {use_amp}, grad clip: {args.grad_clip}")
        print(f"  alpha: {args.alpha}, T: {args.temperature}, augment: {not args.no_augment}")
        print(f"  Smoke preset: {args.smoke or 'none'}")
    else:
        writer = None

    if distributed:
        if rank == 0:
            train_dataset, val_dataset = build_dataset(args, rank)
        dist.barrier()
        if rank != 0:
            train_dataset, val_dataset = build_dataset(args, rank)
    else:
        train_dataset, val_dataset = build_dataset(args, rank)

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=0, pin_memory=True)

    if rank == 0:
        print(f"  Train batches/epoch: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Teacher ──
    teacher, teacher_backbone = load_teacher(args.teacher_ckpt, args.teacher_backbone_fallback, device)
    t_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    if rank == 0:
        print(f"  Teacher: {teacher_backbone}, {t_params:.2f}M params (frozen)")

    # ── Student ──
    student, ce_criterion = build_student(args, distributed, device)
    s_params = sum(p.numel() for p in student.parameters()) / 1e6
    compression = t_params / max(s_params, 1e-9)
    if rank == 0:
        print(f"  Student: {args.backbone}, {s_params:.2f}M params  (compression {compression:.2f}×)")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(args.epochs * len(train_loader), 1)
    warmup_steps = max(args.warmup_epochs * len(train_loader), 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / warmup_steps, 1e-6)
        return max((1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1)) ** 0.9, 1e-6)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_iou_dlo = 0.0
    best_miou = 0.0
    start_time = time.time()
    global_step = 0
    n_train_imgs = len(train_dataset)
    train_imgs_processed = 0
    eta_aborted = False

    if rank == 0:
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        student.train()
        epoch_t0 = time.time()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_kl = 0.0
        epoch_iters = 0
        batch_t0 = time.time()
        nan_in_epoch = 0

        for bi, batch in enumerate(train_loader):
            rgb = normalize_batch(batch["rgb"], device)
            label = batch["label"].to(device, non_blocking=True)

            # Teacher forward (frozen, no grad).
            with torch.no_grad():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        t_logits = teacher(rgb)
                else:
                    t_logits = teacher(rgb)

            # Student forward + KD loss.
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    s_logits = student(rgb)
                    loss, ce_val, kl_val = kd_loss(
                        s_logits, t_logits.float(), label,
                        ce_criterion, args.alpha, args.temperature,
                    )
            else:
                s_logits = student(rgb)
                loss, ce_val, kl_val = kd_loss(
                    s_logits, t_logits, label,
                    ce_criterion, args.alpha, args.temperature,
                )

            if not torch.isfinite(loss):
                nan_in_epoch += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            epoch_loss += float(loss.item())
            epoch_ce += float(ce_val.item())
            epoch_kl += float(kl_val.item())
            epoch_iters += 1
            global_step += 1
            train_imgs_processed += rgb.size(0) * world_size

            if rank == 0 and (bi + 1) % args.log_every == 0:
                dt = time.time() - batch_t0
                imgs_per_s = (args.log_every * args.batch_size * world_size) / max(dt, 1e-9)
                lr = optimizer.param_groups[0]["lr"]
                print(f"    ep{epoch:3d} batch {bi+1:4d}/{len(train_loader):4d}  "
                      f"loss={float(loss.item()):.4f}  "
                      f"CE={float(ce_val.item()):.4f}  KL={float(kl_val.item()):.4f}  "
                      f"lr={lr:.2e}  {imgs_per_s:.1f} img/s")
                if writer is not None:
                    writer.add_scalar("train/loss_step", float(loss.item()), global_step)
                    writer.add_scalar("train/ce_step", float(ce_val.item()), global_step)
                    writer.add_scalar("train/kl_step", float(kl_val.item()), global_step)
                    writer.add_scalar("train/imgs_per_sec", imgs_per_s, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                batch_t0 = time.time()

        epoch_dt = time.time() - epoch_t0
        avg_loss = epoch_loss / max(epoch_iters, 1)
        avg_ce = epoch_ce / max(epoch_iters, 1)
        avg_kl = epoch_kl / max(epoch_iters, 1)
        elapsed = time.time() - start_time
        per_epoch = elapsed / epoch
        remaining_epochs = args.epochs - epoch
        eta = per_epoch * remaining_epochs

        if rank == 0:
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"loss={avg_loss:.4f}  CE={avg_ce:.4f}  KL={avg_kl:.4f}  "
                  f"epoch_wall={fmt_seconds(epoch_dt)}  ETA={fmt_seconds(eta)}  "
                  f"peak_GPU={peak_mb:.0f}MB"
                  + (f"  NaN_skipped={nan_in_epoch}" if nan_in_epoch else ""))
            if writer is not None:
                writer.add_scalar("train/loss_epoch", avg_loss, epoch)
                writer.add_scalar("train/ce_epoch", avg_ce, epoch)
                writer.add_scalar("train/kl_epoch", avg_kl, epoch)
                writer.add_scalar("train/epoch_wall_sec", epoch_dt, epoch)
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
            metrics = evaluate(student, val_loader, device, distributed, use_amp)
            if rank == 0:
                print(f"    val: mIoU={metrics['miou']:.4f}  IoU(DLO)={metrics['iou_dlo']:.4f}  "
                      f"IoU(bg)={metrics['iou_bg']:.4f}  acc={metrics['pixel_acc']:.4f}  "
                      f"prec(DLO)={metrics['precision_dlo']:.4f}  rec(DLO)={metrics['recall_dlo']:.4f}")
                if writer is not None:
                    for k, v in metrics.items():
                        writer.add_scalar(f"val/{k}", v, epoch)

                if metrics["iou_dlo"] > best_iou_dlo:
                    best_iou_dlo = metrics["iou_dlo"]
                    best_miou = metrics["miou"]
                    ckpt = {
                        "epoch": epoch,
                        "metrics": metrics,
                        "args": vars(args),
                        "model_state_dict": (student.module if isinstance(student, DDP) else student).state_dict(),
                        "config": {"backbone": args.backbone, "num_classes": NUM_CLASSES,
                                   "class_names": CLASS_NAMES, "image_size": [IMAGE_H, IMAGE_W],
                                   "kd_alpha": args.alpha, "kd_temperature": args.temperature,
                                   "teacher_backbone": teacher_backbone,
                                   "teacher_ckpt": args.teacher_ckpt},
                    }
                    torch.save(ckpt, os.path.join(results_dir, "best_model.pth"))
                    print(f"    ★ new best IoU(DLO)={best_iou_dlo:.4f}  mIoU={best_miou:.4f}")

        if rank == 0 and args.ckpt_every > 0 and epoch % args.ckpt_every == 0:
            ckpt = {
                "epoch": epoch,
                "args": vars(args),
                "model_state_dict": (student.module if isinstance(student, DDP) else student).state_dict(),
                "config": {"backbone": args.backbone, "num_classes": NUM_CLASSES,
                           "class_names": CLASS_NAMES, "image_size": [IMAGE_H, IMAGE_W]},
            }
            torch.save(ckpt, os.path.join(results_dir, f"epoch_{epoch}.pth"))

    final_eval = None
    if args.eval_every > 0:
        try:
            final_eval = evaluate(student, val_loader, device, distributed, use_amp)
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

        imgs_per_s_overall = None
        full_epoch_secs = None
        if epoch_iters > 0 and total_time > 0:
            imgs_per_s_overall = train_imgs_processed / total_time
            full_train_imgs = 7560
            full_epoch_secs = full_train_imgs / max(imgs_per_s_overall, 1e-9)
            print(f"Throughput: {imgs_per_s_overall:.1f} img/s overall  "
                  f"(full train epoch ≈ {fmt_seconds(full_epoch_secs)} ≈ {full_epoch_secs/3600:.2f}h)")

        if final_eval is not None:
            print(f"Final eval: mIoU={final_eval['miou']:.4f}  "
                  f"IoU(DLO)={final_eval['iou_dlo']:.4f}  IoU(bg)={final_eval['iou_bg']:.4f}")

        report = {
            "args": vars(args),
            "results_dir": results_dir,
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
            "teacher_backbone": teacher_backbone,
            "teacher_params_M": t_params,
            "student_params_M": s_params,
            "compression_ratio": compression,
        }
        with open(os.path.join(results_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
        print(f"Report: {os.path.join(results_dir, 'report.json')}")

        if writer is not None:
            writer.close()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
