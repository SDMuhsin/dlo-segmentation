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

NUM_CLASSES = 2
CLASS_NAMES = ["bg", "DLO"]
IGNORE_INDEX = -1  # nothing in the binary label tensor is == -1

IMAGE_H, IMAGE_W = 480, 640
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


# ─────────────────────────── DATASET ───────────────────────────


class BinaryCDLORGBOnlyDataset(Dataset):
    """RGB-only mmap CDLO dataset that collapses 5 classes to binary {bg=0, DLO=1}.

    Depth is intentionally not loaded — Phase 7 trains the model blind to depth.
    """

    def __init__(self, rgb, label, augment=True, include_noise=False):
        self.rgb = rgb
        self.label = label
        self.augment = augment
        self.include_noise = include_noise

    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, idx):
        rgb = self.rgb[idx].copy()      # (H, W, 3) uint8 BGR
        lbl = self.label[idx].copy()     # (H, W) uint8 — 0..4 fg, 255 bg

        if self.augment and random.random() > 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            lbl = np.ascontiguousarray(lbl[:, ::-1])

        # Binary collapse: classes 0..3 -> 1 (DLO), class 4 (Noise) and 255 (bg) -> 0
        if self.include_noise:
            binary = (lbl <= 4).astype(np.int64)
        else:
            binary = (lbl <= 3).astype(np.int64)

        # Convert BGR uint8 to (3, H, W) — colour order is preserved through
        # ImageNet normalisation since SegFormer was trained on RGB. The cache
        # holds BGR, so swap to RGB before returning.
        rgb_rgb = rgb[:, :, ::-1].copy()              # BGR -> RGB
        rgb_t = torch.from_numpy(rgb_rgb.transpose(2, 0, 1).copy())  # (3, H, W) uint8 RGB
        label_t = torch.from_numpy(binary)             # (H, W) int64 {0,1}
        return {"rgb": rgb_t, "label": label_t}


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


class SegFormerSegmenter(nn.Module):
    """Wraps a HuggingFace SegformerForSemanticSegmentation so the API mirrors the
    DFormer EncoderDecoder used in train_dformer_v2_dlo.py:

        forward(rgb)             -> logits at (B, num_classes, H, W)   [inference]
        forward(rgb, label)      -> scalar loss                        [training]

    The HF model emits logits at H/4, W/4; we bilinear-upsample to (H, W) before
    loss / argmax for parity with DFormer-style outputs.
    """

    def __init__(self, backbone_name=BACKBONE_DEFAULT, num_classes=NUM_CLASSES, criterion=None):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone_name,
            num_labels=num_classes,
            id2label={i: n for i, n in enumerate(CLASS_NAMES[:num_classes])},
            label2id={n: i for i, n in enumerate(CLASS_NAMES[:num_classes])},
            ignore_mismatched_sizes=True,
        )
        self.criterion = criterion

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
            return per_pixel.mean()
        return per_pixel[valid].mean()


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
                   help="positive class weight for the DLO class")
    p.add_argument("--include-noise", action="store_true")
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

    train_full = BinaryCDLORGBOnlyDataset(
        train_rgb, train_label, augment=True, include_noise=args.include_noise
    )
    val_full = BinaryCDLORGBOnlyDataset(
        val_rgb, val_label, augment=False, include_noise=args.include_noise
    )
    train_dataset = Subset(train_full, train_indices) if allowed_sets is not None else train_full
    val_dataset = val_full
    return train_dataset, val_dataset


def build_model(args, distributed, device):
    pos_weight = torch.tensor([1.0, float(args.dlo_weight)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=pos_weight, reduction="none", ignore_index=IGNORE_INDEX)
    model = SegFormerSegmenter(
        backbone_name=args.backbone, num_classes=NUM_CLASSES, criterion=criterion
    ).to(device)
    if distributed:
        # SyncBatchNorm conversion (decode_head has BatchNorm2d).
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device.index], output_device=device.index,
                    find_unused_parameters=False)
    return model


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

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        tb_dir = os.path.join(args.results_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)
        print(f"Training SegFormer ({args.backbone}) for binary DLO segmentation (RGB-only)")
        print(f"  GPUs: {world_size}, batch/GPU: {args.batch_size}, "
              f"total batch: {args.batch_size * world_size}")
        print(f"  Epochs: {args.epochs}, LR: {args.lr}, DLO weight: {args.dlo_weight}, "
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
            metrics = evaluate(model, val_loader, device, distributed, use_amp)
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
                        "model_state_dict": (model.module if isinstance(model, DDP) else model).state_dict(),
                        "config": {"backbone": args.backbone, "num_classes": NUM_CLASSES,
                                   "class_names": CLASS_NAMES, "image_size": [IMAGE_H, IMAGE_W]},
                    }
                    torch.save(ckpt, os.path.join(args.results_dir, "best_model.pth"))
                    print(f"    ★ new best IoU(DLO)={best_iou_dlo:.4f}  mIoU={best_miou:.4f}")

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
            final_eval = evaluate(model, val_loader, device, distributed, use_amp)
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
