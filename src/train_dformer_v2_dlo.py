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

# ─────────────────────────── CONFIG ───────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "data", "dformer_dataset")
RESULTS_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "results", "dformer_v2_dlo")
PRETRAINED_DEFAULT = os.path.join(
    PROJECT_ROOT, "data", "pretrained", "DFormerv2", "pretrained", "DFormerv2_Large_pretrained.pth"
)

NUM_CLASSES = 2
CLASS_NAMES = ["bg", "DLO"]
IGNORE_INDEX = -1  # sentinel: nothing in the label tensor is == -1, so loss covers all pixels

IMAGE_H, IMAGE_W = 480, 640
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
DEPTH_MEAN = torch.tensor([0.48, 0.48, 0.48], dtype=torch.float32).view(1, 3, 1, 1)
DEPTH_STD = torch.tensor([0.28, 0.28, 0.28], dtype=torch.float32).view(1, 3, 1, 1)


# ─────────────────────────── DATASET ───────────────────────────


class BinaryCDLOMmapDataset(Dataset):
    """Mmap-backed CDLO dataset that collapses 5 classes to binary {bg=0, DLO=1}."""

    def __init__(self, rgb, depth, label, augment=True, include_noise=False):
        self.rgb = rgb
        self.depth = depth
        self.label = label
        self.augment = augment
        self.include_noise = include_noise

    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, idx):
        rgb = self.rgb[idx].copy()      # (H, W, 3) uint8
        depth = self.depth[idx].copy()   # (H, W) uint8
        lbl = self.label[idx].copy()     # (H, W) uint8 — 0..4 fg, 255 bg

        if self.augment and random.random() > 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            depth = np.ascontiguousarray(depth[:, ::-1])
            lbl = np.ascontiguousarray(lbl[:, ::-1])

        # Binary collapse: classes 0..3 -> 1 (DLO), class 4 (Noise) and 255 (bg) -> 0
        if self.include_noise:
            # Treat Noise as DLO (rare; not the default)
            binary = (lbl <= 4).astype(np.int64)
        else:
            binary = (lbl <= 3).astype(np.int64)

        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1))  # (3, H, W) uint8
        depth_t = torch.from_numpy(depth).unsqueeze(0)     # (1, H, W) uint8
        label_t = torch.from_numpy(binary)                  # (H, W) int64 {0,1}
        return {"rgb": rgb_t, "depth": depth_t, "label": label_t}


def normalize_batch(rgb_uint8, depth_uint8, device):
    rgb = rgb_uint8.to(device, dtype=torch.float32, non_blocking=True) / 255.0
    rgb = (rgb - RGB_MEAN.to(device)) / RGB_STD.to(device)
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
                   help="positive class weight for the DLO class")
    p.add_argument("--include-noise", action="store_true",
                   help="treat class 4 (Noise) as DLO instead of bg")
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

    if rank == 0:
        print(f"  Train subset: {len(train_indices)} images "
              f"(sets={allowed_sets if allowed_sets is not None else 'all'})")
        print(f"  Val:          {len(val_indices)} images")

    train_full = BinaryCDLOMmapDataset(
        train_rgb, train_depth, train_label, augment=True, include_noise=args.include_noise
    )
    val_full = BinaryCDLOMmapDataset(
        val_rgb, val_depth, val_label, augment=False, include_noise=args.include_noise
    )

    train_dataset = Subset(train_full, train_indices) if allowed_sets is not None else train_full
    val_dataset = val_full

    return train_dataset, val_dataset


def build_model(args, distributed, device):
    cfg = ModelConfig()
    cfg.pretrained_model = None if args.no_pretrained else args.pretrained

    pos_weight = torch.tensor([1.0, float(args.dlo_weight)], dtype=torch.float32, device=device)
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


def fmt_seconds(s):
    return str(datetime.timedelta(seconds=int(max(s, 0))))


def evaluate(model, val_loader, device, distributed):
    metric = BinaryIoU()
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            rgb, depth = normalize_batch(batch["rgb"], batch["depth"], device)
            label_np = batch["label"].numpy()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                m = model.module if isinstance(model, DDP) else model
                out = m(rgb, depth)
            pred = out.argmax(dim=1).cpu().numpy()
            for i in range(pred.shape[0]):
                metric.update(pred[i].flatten(), label_np[i].flatten())

    if distributed:
        # All-reduce raw counts
        t = torch.tensor([metric.tp, metric.fp, metric.fn, metric.tn], dtype=torch.float64, device=device)
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
        print(f"Training DFormer-v2-Large for binary DLO segmentation")
        print(f"  GPUs: {world_size}, batch/GPU: {args.batch_size}, total batch: {args.batch_size * world_size}")
        print(f"  Epochs: {args.epochs}, LR: {args.lr}, DLO weight: {args.dlo_weight}")
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
            rgb, depth = normalize_batch(batch["rgb"], batch["depth"], device)
            label = batch["label"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(rgb, depth, label)

            scaler.scale(loss).backward()
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
            metrics = evaluate(model, val_loader, device, distributed)
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
                        "config": {"backbone": ModelConfig.backbone, "decoder": ModelConfig.decoder,
                                   "num_classes": NUM_CLASSES, "class_names": CLASS_NAMES,
                                   "image_size": [IMAGE_H, IMAGE_W]},
                    }
                    torch.save(ckpt, os.path.join(args.results_dir, "best_model.pth"))
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
            final_eval = evaluate(model, val_loader, device, distributed)
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
