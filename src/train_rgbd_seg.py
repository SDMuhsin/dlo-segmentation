"""
Train DFormer-Tiny for RGB-D semantic segmentation on the CDLO dataset.

Uses DFormer's model architecture with a clean PyTorch DDP training loop.
Handles class imbalance via weighted cross-entropy loss.

Data is stored as memory-mapped uint8 arrays (shared between GPU processes)
and normalized on-the-fly for fast, memory-efficient training.

Usage:
  source env/bin/activate
  torchrun --nproc_per_node=2 src/train_rgbd_seg.py
"""

import argparse
import datetime
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "dformer_cdlo_honest")

NUM_CLASSES = 5
CLASS_NAMES = ["Wire", "Endpoint", "Bifurcation", "Connector", "Noise"]
BACKGROUND = 255

CLASS_WEIGHTS = torch.tensor([1.0, 3.8, 2.2, 3.4, 13.5], dtype=torch.float32)

IMAGE_H, IMAGE_W = 480, 640
# Normalization constants as tensors for GPU ops
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

    print(f"  Building cache for {split} ({N} images)...")
    rgb_dir = os.path.join(root_dir, "RGB")
    depth_dir = os.path.join(root_dir, "Depth")
    label_dir = os.path.join(root_dir, "Label")

    # Create on-disk arrays
    rgb_arr = np.lib.format.open_memmap(rgb_path, mode="w+", dtype=np.uint8, shape=(N, IMAGE_H, IMAGE_W, 3))
    depth_arr = np.lib.format.open_memmap(depth_path, mode="w+", dtype=np.uint8, shape=(N, IMAGE_H, IMAGE_W))
    label_arr = np.lib.format.open_memmap(label_path, mode="w+", dtype=np.uint8, shape=(N, IMAGE_H, IMAGE_W))

    for i, fn in enumerate(file_names):
        base_name = fn.split("/")[1].replace(".png", "")
        bgr = cv2.imread(os.path.join(rgb_dir, base_name + ".png"), cv2.IMREAD_COLOR)
        d = cv2.imread(os.path.join(depth_dir, base_name + ".png"), cv2.IMREAD_GRAYSCALE)
        lbl = cv2.imread(os.path.join(label_dir, base_name + ".png"), cv2.IMREAD_GRAYSCALE)

        # Phase 4 RGB is real-textured; preserve it. The old uniform-gray strip was a
        # defensive hack for the OLD class-coloured RGB and would discard 28h of textured
        # rendering on Phase 4 data.
        rgb_arr[i] = bgr
        depth_arr[i] = d
        # gt_transform: 0→255 (bg), 1-5→0-4 (classes)
        lbl_t = lbl.astype(np.int16) - 1
        lbl_t[lbl_t < 0] = 255
        label_arr[i] = lbl_t.astype(np.uint8)

        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{N}")

    rgb_arr.flush()
    depth_arr.flush()
    label_arr.flush()
    print(f"  {split}: {N} images cached to {cache_dir}")

    # Reopen as read-only mmap
    return (np.load(rgb_path, mmap_mode="r"),
            np.load(depth_path, mmap_mode="r"),
            np.load(label_path, mmap_mode="r"))


# ─────────────────────────── DATASET ───────────────────────────

class CDLOMmapDataset(Dataset):
    """Memory-mapped dataset with on-the-fly normalization."""

    def __init__(self, rgb, depth, label, augment=True):
        self.rgb = rgb
        self.depth = depth
        self.label = label
        self.augment = augment

    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, idx):
        # Read from mmap (copies to RAM automatically)
        rgb = self.rgb[idx].copy()      # (H, W, 3) uint8
        depth = self.depth[idx].copy()   # (H, W) uint8
        label = self.label[idx].copy()   # (H, W) uint8

        # Random horizontal flip
        if self.augment and random.random() > 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            depth = np.ascontiguousarray(depth[:, ::-1])
            label = np.ascontiguousarray(label[:, ::-1])

        # Convert to tensors (uint8 → float32 normalized happens on GPU for speed)
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1))      # (3, H, W) uint8
        depth_t = torch.from_numpy(depth).unsqueeze(0)          # (1, H, W) uint8
        label_t = torch.from_numpy(label.astype(np.int64))      # (H, W) int64

        return {"rgb": rgb_t, "depth": depth_t, "label": label_t}


def normalize_batch(rgb_uint8, depth_uint8, device):
    """Normalize a batch on GPU. Input: uint8 tensors. Output: float32 normalized."""
    # RGB: uint8 → float32 [0,1] → ImageNet normalize
    rgb = rgb_uint8.to(device, dtype=torch.float32, non_blocking=True) / 255.0
    rgb = (rgb - RGB_MEAN.to(device)) / RGB_STD.to(device)

    # Depth: uint8 (1ch) → 3ch float32 → normalize
    depth = depth_uint8.to(device, dtype=torch.float32, non_blocking=True) / 255.0
    depth = depth.expand(-1, 3, -1, -1)  # (B, 1, H, W) → (B, 3, H, W)
    depth = (depth - DEPTH_MEAN.to(device)) / DEPTH_STD.to(device)

    return rgb, depth


# ─────────────────────────── MODEL ───────────────────────────

class ModelConfig:
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


# ─────────────────────────── TRAINING ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--eval_every", type=int, default=10)
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
        print(f"Training DFormer-Tiny on CDLO RGB-D dataset")
        print(f"  GPUs: {world_size}, Batch/GPU: {args.batch_size}, Total: {args.batch_size * world_size}")
        print(f"  Epochs: {args.epochs}, LR: {args.lr}")

    # Build cache (only rank 0 creates, others wait)
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

    # Model
    cfg = ModelConfig()
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device), reduction="none", ignore_index=BACKGROUND)
    model = DFormerModel(cfg=cfg, criterion=criterion,
                         norm_layer=nn.SyncBatchNorm if distributed else nn.BatchNorm2d,
                         syncbn=distributed).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if rank == 0:
        print(f"  Model: DFormer-Tiny + MLP, {n_params:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

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
        model.train()
        total_loss, n_iter = 0.0, 0

        for batch in train_loader:
            rgb, depth = normalize_batch(batch["rgb"], batch["depth"], device)
            label = batch["label"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(rgb, depth, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            total_loss += loss.item()
            n_iter += 1

        avg_loss = total_loss / max(n_iter, 1)
        if rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{args.epochs}: loss={avg_loss:.4f}, lr={lr:.2e}")

        # ── Evaluate ──
        do_eval = (epoch % args.eval_every == 0 or epoch == args.epochs or epoch == 1)
        if do_eval:
            model.eval()
            metric = SegMetric(NUM_CLASSES)

            with torch.no_grad():
                for batch in val_loader:
                    rgb, depth = normalize_batch(batch["rgb"], batch["depth"], device)
                    label_np = batch["label"].numpy()

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        m = model.module if distributed else model
                        out = m(rgb, depth)

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
                # Sanity: check model predicts all 5 classes (not degenerate)
                pred_classes = set(np.unique(pred))
                print(f"  ▸ mIoU={miou:.4f}, pxAcc={acc:.4f}, predicted_classes={sorted(pred_classes)}, ETA={eta}")
                for i, name in enumerate(CLASS_NAMES):
                    print(f"      {name}: {ious[i]:.4f}")

                if miou > best_miou:
                    best_miou = miou
                    torch.save({
                        "epoch": epoch, "miou": miou,
                        "ious": {n: float(ious[i]) for i, n in enumerate(CLASS_NAMES)},
                        "model_state_dict": (model.module if distributed else model).state_dict(),
                        "config": {"backbone": "DFormer-Tiny", "decoder": "MLPDecoder",
                                   "num_classes": NUM_CLASSES, "class_names": CLASS_NAMES,
                                   "image_size": [IMAGE_H, IMAGE_W]},
                    }, os.path.join(RESULTS_DIR, "best_model.pth"))
                    print(f"  ★ New best mIoU={miou:.4f}")

    # Save final
    if rank == 0:
        torch.save({
            "epoch": args.epochs, "best_miou": best_miou,
            "model_state_dict": (model.module if distributed else model).state_dict(),
            "config": {"backbone": "DFormer-Tiny", "decoder": "MLPDecoder",
                       "num_classes": NUM_CLASSES, "class_names": CLASS_NAMES,
                       "image_size": [IMAGE_H, IMAGE_W]},
        }, os.path.join(RESULTS_DIR, "final_model.pth"))

        total_time = time.time() - start_time
        print(f"\nDone in {str(datetime.timedelta(seconds=int(total_time)))}")
        print(f"Best mIoU: {best_miou:.4f}")
        print(f"Models: {RESULTS_DIR}/best_model.pth, final_model.pth")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
