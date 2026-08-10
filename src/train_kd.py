"""
Knowledge Distillation: DFormer-Tiny (teacher) → DFormer-Nano (student).

Teacher: DFormer-Tiny, 5.35M params, 92.06% mIoU (frozen)
Student: DFormer-Nano, ~2.67M params (half the teacher)

Loss: α × CE(student_pred, hard_labels) + (1-α) × KL(student_logits/T, teacher_logits/T) × T²

Uses the same mmap data pipeline and DDP setup as the teacher training script.

Usage:
  source env/bin/activate
  torchrun --nproc_per_node=2 src/train_kd.py
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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Add DFormer to path
DFORMER_DIR = os.path.join(os.path.dirname(__file__), "dformer")
sys.path.insert(0, DFORMER_DIR)

from models.builder import EncoderDecoder as DFormerModel

# ─────────────────────────── CONFIG ───────────────────────────

PROJECT_ROOT = "/workspace/kiat_crefle"
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dformer_dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "kd_student")
TEACHER_CKPT = os.path.join(PROJECT_ROOT, "results", "dformer_cdlo", "best_model.pth")

NUM_CLASSES = 5
CLASS_NAMES = ["Wire", "Endpoint", "Bifurcation", "Connector", "Noise"]
BACKGROUND = 255

CLASS_WEIGHTS = torch.tensor([1.0, 3.8, 2.2, 3.4, 13.5], dtype=torch.float32)

IMAGE_H, IMAGE_W = 480, 640
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
DEPTH_MEAN = torch.tensor([0.48, 0.48, 0.48], dtype=torch.float32).view(1, 3, 1, 1)
DEPTH_STD = torch.tensor([0.28, 0.28, 0.28], dtype=torch.float32).view(1, 3, 1, 1)


# ─────────────────────────── CACHE ───────────────────────────

def build_cache(root_dir, split):
    """Build or load memory-mapped numpy cache of images."""
    txt_file = os.path.join(root_dir, "train.txt" if split == "train" else "test.txt")
    with open(txt_file) as f:
        file_names = [line.strip() for line in f if line.strip()]

    N = len(file_names)
    cache_dir = os.path.join(root_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    rgb_path = os.path.join(cache_dir, f"{split}_rgb.npy")
    depth_path = os.path.join(cache_dir, f"{split}_depth.npy")
    label_path = os.path.join(cache_dir, f"{split}_label.npy")

    if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(label_path):
        print(f"  Loading cached {split} data...")
        rgb = np.load(rgb_path, mmap_mode="r")
        depth = np.load(depth_path, mmap_mode="r")
        label = np.load(label_path, mmap_mode="r")
        print(f"  {split}: {rgb.shape[0]} images from cache")
        return rgb, depth, label

    raise RuntimeError(f"Cache not found for {split}. Run teacher training first to build cache.")


# ─────────────────────────── DATASET ───────────────────────────

class CDLOMmapDataset(Dataset):
    """Memory-mapped dataset with on-the-fly augmentation."""

    def __init__(self, rgb, depth, label, augment=True):
        self.rgb = rgb
        self.depth = depth
        self.label = label
        self.augment = augment

    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, idx):
        rgb = self.rgb[idx].copy()
        depth = self.depth[idx].copy()
        label = self.label[idx].copy()

        if self.augment and random.random() > 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            depth = np.ascontiguousarray(depth[:, ::-1])
            label = np.ascontiguousarray(label[:, ::-1])

        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1))
        depth_t = torch.from_numpy(depth).unsqueeze(0)
        label_t = torch.from_numpy(label.astype(np.int64))

        return {"rgb": rgb_t, "depth": depth_t, "label": label_t}


def normalize_batch(rgb_uint8, depth_uint8, device):
    """Normalize a batch on GPU. Input: uint8 tensors. Output: float32 normalized."""
    rgb = rgb_uint8.to(device, dtype=torch.float32, non_blocking=True) / 255.0
    rgb = (rgb - RGB_MEAN.to(device)) / RGB_STD.to(device)

    depth = depth_uint8.to(device, dtype=torch.float32, non_blocking=True) / 255.0
    depth = depth.expand(-1, 3, -1, -1)
    depth = (depth - DEPTH_MEAN.to(device)) / DEPTH_STD.to(device)

    return rgb, depth


# ─────────────────────────── MODEL CONFIGS ───────────────────────────

class TeacherConfig:
    backbone = "DFormer-Tiny"
    pretrained_model = None
    decoder = "MLPDecoder"
    decoder_embed_dim = 256
    num_classes = NUM_CLASSES
    background = BACKGROUND
    bn_eps = 1e-3
    bn_momentum = 0.1
    drop_path_rate = 0.1
    aux_rate = 0.0
    fix_bias = True


class StudentConfig:
    backbone = "DFormer-Nano"
    pretrained_model = None
    decoder = "MLPDecoder"
    decoder_embed_dim = 128
    num_classes = NUM_CLASSES
    background = BACKGROUND
    bn_eps = 1e-3
    bn_momentum = 0.1
    drop_path_rate = 0.1
    aux_rate = 0.0
    fix_bias = True


# ─────────────────────────── METRICS ───────────────────────────

class SegMetric:
    def __init__(self, n):
        self.n = n
        self.hist = np.zeros((n, n), dtype=np.int64)

    def update(self, pred, label):
        k = (label >= 0) & (label < self.n)
        self.hist += np.bincount(self.n * label[k].astype(int) + pred[k].astype(int),
                                  minlength=self.n ** 2).reshape(self.n, self.n)

    def compute(self):
        iou = np.diag(self.hist) / (self.hist.sum(1) + self.hist.sum(0) - np.diag(self.hist) + 1e-10)
        return np.nanmean(iou), iou, np.diag(self.hist).sum() / (self.hist.sum() + 1e-10)


# ─────────────────────────── KD LOSS ───────────────────────────

def kd_loss(student_logits, teacher_logits, labels, ce_criterion, alpha, temperature):
    """
    Knowledge distillation loss.

    Args:
        student_logits: (B, C, H, W) raw logits from student
        teacher_logits: (B, C, H, W) raw logits from teacher
        labels: (B, H, W) ground truth (255 = ignore)
        ce_criterion: weighted CrossEntropyLoss with ignore_index=255
        alpha: weight for hard label CE loss
        temperature: softening temperature for KL divergence

    Returns:
        total_loss: α × CE_hard + (1-α) × KL_soft × T²
    """
    # Hard label loss (weighted CE, ignoring background)
    # The criterion has reduction='none', so we manually mask and mean
    ce_hard = ce_criterion(student_logits, labels)
    valid_mask = labels != BACKGROUND
    ce_hard = ce_hard[valid_mask].mean()

    # Soft label loss (KL divergence on softened probabilities)
    # Only compute on valid (non-background) pixels for efficiency and correctness
    B, C, H, W = student_logits.shape
    # Reshape to (B*H*W, C) for masking
    s_flat = student_logits.permute(0, 2, 3, 1).reshape(-1, C)
    t_flat = teacher_logits.permute(0, 2, 3, 1).reshape(-1, C)
    valid_flat = valid_mask.reshape(-1)

    s_valid = s_flat[valid_flat]  # (N_valid, C)
    t_valid = t_flat[valid_flat]  # (N_valid, C)

    s_soft = F.log_softmax(s_valid / temperature, dim=1)
    t_soft = F.softmax(t_valid / temperature, dim=1)
    kl_soft = F.kl_div(s_soft, t_soft, reduction="batchmean") * (temperature ** 2)

    total_loss = alpha * ce_hard + (1.0 - alpha) * kl_soft
    return total_loss, ce_hard.item(), kl_soft.item()


# ─────────────────────────── TRAINING ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for hard CE loss (1-alpha for KL soft loss)")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Temperature for knowledge distillation")
    args = parser.parse_args()

    distributed = not args.single_gpu and "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank, rank, world_size = 0, 0, 1
        torch.cuda.set_device(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.set_float32_matmul_precision("high")

    if rank == 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"Knowledge Distillation: DFormer-Tiny → DFormer-Nano")
        print(f"  GPUs: {world_size}, Batch/GPU: {args.batch_size}, Total: {args.batch_size * world_size}")
        print(f"  Epochs: {args.epochs}, LR: {args.lr}")
        print(f"  Alpha: {args.alpha}, Temperature: {args.temperature}")

    # ── Load data ──
    if distributed:
        if rank == 0:
            train_rgb, train_depth, train_label = build_cache(DATASET_DIR, "train")
            val_rgb, val_depth, val_label = build_cache(DATASET_DIR, "val")
        dist.barrier()
        if rank != 0:
            train_rgb, train_depth, train_label = build_cache(DATASET_DIR, "train")
            val_rgb, val_depth, val_label = build_cache(DATASET_DIR, "val")
    else:
        train_rgb, train_depth, train_label = build_cache(DATASET_DIR, "train")
        val_rgb, val_depth, val_label = build_cache(DATASET_DIR, "val")

    train_dataset = CDLOMmapDataset(train_rgb, train_depth, train_label, augment=True)
    val_dataset = CDLOMmapDataset(val_rgb, val_depth, val_label, augment=False)

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, sampler=val_sampler,
                            num_workers=0, pin_memory=True)

    if rank == 0:
        print(f"  Train: {len(train_dataset)} imgs, {len(train_loader)} batches/epoch")
        print(f"  Val:   {len(val_dataset)} imgs")

    # ── Teacher model (frozen) ──
    teacher = DFormerModel(cfg=TeacherConfig(), criterion=None,
                           norm_layer=nn.BatchNorm2d, syncbn=False).to(device)
    state = torch.load(TEACHER_CKPT, map_location=device, weights_only=False)
    teacher.load_state_dict(state["model_state_dict"], strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    if rank == 0:
        print(f"  Teacher: DFormer-Tiny, {teacher_params:.2f}M params (frozen)")
        print(f"  Teacher checkpoint: epoch {state.get('epoch', '?')}, mIoU={state.get('miou', '?')}")

    # ── Student model ──
    ce_criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device), reduction="none", ignore_index=BACKGROUND)
    student = DFormerModel(cfg=StudentConfig(), criterion=ce_criterion,
                           norm_layer=nn.SyncBatchNorm if distributed else nn.BatchNorm2d,
                           syncbn=distributed).to(device)
    if distributed:
        student = DDP(student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    student_params = sum(p.numel() for p in student.parameters()) / 1e6
    if rank == 0:
        print(f"  Student: DFormer-Nano, {student_params:.2f}M params")
        print(f"  Compression ratio: {teacher_params / student_params:.2f}×")

    # ── Optimizer & scheduler ──
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = 10 * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-6)
        return max((1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1)) ** 0.9, 1e-6)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler()

    best_miou = 0.0
    start_time = time.time()

    if rank == 0:
        print(f"  Total steps: {total_steps}, warmup: {warmup_steps}\n")

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ── Train ──
        student.train()
        total_loss, total_ce, total_kl, n_iter = 0.0, 0.0, 0.0, 0

        for batch in train_loader:
            rgb, depth = normalize_batch(batch["rgb"], batch["depth"], device)
            label = batch["label"].to(device, non_blocking=True)

            # Teacher forward (frozen, no grad)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    teacher_logits = teacher(rgb, depth)  # (B, C, H, W)

            # Student forward
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                student_model = student.module if distributed else student
                student_logits = student_model(rgb, depth)  # (B, C, H, W)

                loss, ce_val, kl_val = kd_loss(
                    student_logits, teacher_logits.float(), label,
                    ce_criterion, args.alpha, args.temperature
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            total_loss += loss.item()
            total_ce += ce_val
            total_kl += kl_val
            n_iter += 1

        avg_loss = total_loss / max(n_iter, 1)
        avg_ce = total_ce / max(n_iter, 1)
        avg_kl = total_kl / max(n_iter, 1)

        if rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{args.epochs}: loss={avg_loss:.4f} (CE={avg_ce:.4f}, KL={avg_kl:.4f}), lr={lr:.2e}")

        # ── Evaluate ──
        do_eval = (epoch % args.eval_every == 0 or epoch == args.epochs or epoch == 1)
        if do_eval:
            student.eval()
            metric = SegMetric(NUM_CLASSES)

            with torch.no_grad():
                for batch in val_loader:
                    rgb, depth = normalize_batch(batch["rgb"], batch["depth"], device)
                    label_np = batch["label"].numpy()

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        student_model = student.module if distributed else student
                        out = student_model(rgb, depth)

                    pred = out.argmax(dim=1).cpu().numpy()
                    for i in range(pred.shape[0]):
                        metric.update(pred[i].flatten(), label_np[i].flatten())

            miou, ious, acc = metric.compute()

            if distributed:
                t = torch.tensor([miou], device=device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                miou = t.item()

            if rank == 0:
                elapsed = time.time() - start_time
                eta = str(datetime.timedelta(seconds=int(elapsed / epoch * (args.epochs - epoch))))
                print(f"  ▸ mIoU={miou:.4f}, pxAcc={acc:.4f}, ETA={eta}")
                for i, name in enumerate(CLASS_NAMES):
                    print(f"      {name}: {ious[i]:.4f}")

                if miou > best_miou:
                    best_miou = miou
                    ckpt = {
                        "epoch": epoch, "miou": miou,
                        "ious": {n: float(ious[i]) for i, n in enumerate(CLASS_NAMES)},
                        "model_state_dict": (student.module if distributed else student).state_dict(),
                        "config": {
                            "backbone": "DFormer-Nano", "decoder": "MLPDecoder",
                            "decoder_embed_dim": 128,
                            "num_classes": NUM_CLASSES, "class_names": CLASS_NAMES,
                            "image_size": [IMAGE_H, IMAGE_W],
                            "kd_alpha": args.alpha, "kd_temperature": args.temperature,
                            "teacher": "DFormer-Tiny",
                        },
                    }
                    torch.save(ckpt, os.path.join(RESULTS_DIR, "best_student.pth"))
                    print(f"  ★ New best mIoU={miou:.4f}")

    # ── Save final ──
    if rank == 0:
        torch.save({
            "epoch": args.epochs, "best_miou": best_miou,
            "model_state_dict": (student.module if distributed else student).state_dict(),
            "config": {
                "backbone": "DFormer-Nano", "decoder": "MLPDecoder",
                "decoder_embed_dim": 128,
                "num_classes": NUM_CLASSES, "class_names": CLASS_NAMES,
                "image_size": [IMAGE_H, IMAGE_W],
                "kd_alpha": args.alpha, "kd_temperature": args.temperature,
                "teacher": "DFormer-Tiny",
            },
        }, os.path.join(RESULTS_DIR, "final_student.pth"))

        total_time = time.time() - start_time
        print(f"\nDone in {str(datetime.timedelta(seconds=int(total_time)))}")
        print(f"Best mIoU: {best_miou:.4f}")
        print(f"Teacher mIoU: {state.get('miou', 'N/A')}")
        print(f"Models: {RESULTS_DIR}/best_student.pth, final_student.pth")

        # Save summary
        summary = {
            "teacher": {"backbone": "DFormer-Tiny", "params_M": teacher_params,
                        "miou": state.get("miou", None)},
            "student": {"backbone": "DFormer-Nano", "params_M": student_params,
                        "best_miou": best_miou},
            "kd": {"alpha": args.alpha, "temperature": args.temperature},
            "training": {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size * world_size,
                         "total_time_s": int(total_time)},
        }
        with open(os.path.join(RESULTS_DIR, "kd_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
