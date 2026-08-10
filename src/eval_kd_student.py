"""
Evaluate a trained KD student (DFormer-Nano) and compare with teacher (DFormer-Tiny).

Computes: mIoU, per-class IoU, pixel accuracy, param count, FLOPs, inference speed.

Usage:
  source env/bin/activate
  python src/eval_kd_student.py
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

DFORMER_DIR = os.path.join(os.path.dirname(__file__), "dformer")
sys.path.insert(0, DFORMER_DIR)

from models.builder import EncoderDecoder as DFormerModel

PROJECT_ROOT = "/workspace/kiat_crefle"
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dformer_dataset")

NUM_CLASSES = 5
CLASS_NAMES = ["Wire", "Endpoint", "Bifurcation", "Connector", "Noise"]
BACKGROUND = 255
IMAGE_H, IMAGE_W = 480, 640

VIS_COLORS = {
    0: (180, 180, 180),  # Wire
    1: (0, 0, 255),      # Endpoint
    2: (255, 0, 0),      # Bifurcation
    3: (0, 255, 0),      # Connector
    4: (0, 255, 255),    # Noise
}

RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEPTH_MEAN = torch.tensor([0.48, 0.48, 0.48]).view(1, 3, 1, 1)
DEPTH_STD = torch.tensor([0.28, 0.28, 0.28]).view(1, 3, 1, 1)


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


def load_model(cfg_class, checkpoint_path, device):
    cfg = cfg_class()
    model = DFormerModel(cfg=cfg, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model = model.to(device).eval()
    return model, state


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_flops(model, device):
    """Estimate FLOPs using torch.utils.flop_counter."""
    rgb = torch.randn(1, 3, IMAGE_H, IMAGE_W, device=device)
    depth = torch.randn(1, 3, IMAGE_H, IMAGE_W, device=device)
    from torch.utils.flop_counter import FlopCounterMode
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        model(rgb, depth)
    return flop_counter.get_total_flops()


def measure_inference_speed(model, device, n_warmup=50, n_runs=200):
    """Measure average inference time in ms."""
    rgb = torch.randn(1, 3, IMAGE_H, IMAGE_W, device=device)
    depth = torch.randn(1, 3, IMAGE_H, IMAGE_W, device=device)

    # Warmup
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(n_warmup):
            model(rgb, depth)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(rgb, depth)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    return np.mean(times), np.std(times)


def predict_single(model, rgb_bgr, depth_8bit, device):
    rgb_t = torch.from_numpy(rgb_bgr.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0) / 255.0
    rgb_t = (rgb_t - RGB_MEAN) / RGB_STD
    d_t = torch.from_numpy(depth_8bit.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255.0
    d_t = d_t.expand(-1, 3, -1, -1)
    d_t = (d_t - DEPTH_MEAN) / DEPTH_STD
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(rgb_t.to(device), d_t.to(device))
    return out.argmax(dim=1).squeeze(0).cpu().numpy()


def evaluate_model(model, file_names, device, model_name):
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    print(f"\nEvaluating {model_name} on val ({len(file_names)} images)...")

    for i, fn in enumerate(file_names):
        base_name = fn.split("/")[1].replace(".png", "")
        rgb = cv2.imread(os.path.join(DATASET_DIR, "RGB", base_name + ".png"), cv2.IMREAD_COLOR)
        depth = cv2.imread(os.path.join(DATASET_DIR, "Depth", base_name + ".png"), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(DATASET_DIR, "Label", base_name + ".png"), cv2.IMREAD_GRAYSCALE)

        label_t = label.astype(np.int16) - 1
        label_t[label_t < 0] = 255

        pred = predict_single(model, rgb, depth, device)
        k = (label_t >= 0) & (label_t < NUM_CLASSES)
        hist += np.bincount(NUM_CLASSES * label_t[k].astype(int) + pred[k].astype(int),
                            minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(file_names)}")

    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    miou = np.nanmean(iou)
    acc = np.diag(hist).sum() / (hist.sum() + 1e-10)
    return miou, iou, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_ckpt", default=os.path.join(PROJECT_ROOT, "results", "kd_student", "best_student.pth"))
    parser.add_argument("--teacher_ckpt", default=os.path.join(PROJECT_ROOT, "results", "dformer_cdlo", "best_model.pth"))
    parser.add_argument("--num_vis", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda:0")

    # Load val file list
    txt_file = os.path.join(DATASET_DIR, "test.txt")
    with open(txt_file) as f:
        file_names = [l.strip() for l in f if l.strip()]

    results = {}

    for name, cfg_cls, ckpt in [("teacher", TeacherConfig, args.teacher_ckpt),
                                  ("student", StudentConfig, args.student_ckpt)]:
        print(f"\n{'='*60}")
        print(f"  {name.upper()}: {cfg_cls.backbone}")
        print(f"{'='*60}")

        model, state = load_model(cfg_cls, ckpt, device)

        # Params
        params = count_params(model)
        print(f"  Parameters: {params:,} ({params/1e6:.3f}M)")

        # FLOPs
        try:
            flops = measure_flops(model, device)
            print(f"  FLOPs: {flops/1e9:.2f}G")
        except Exception as e:
            print(f"  FLOPs: error ({e})")
            flops = None

        # Inference speed
        mean_ms, std_ms = measure_inference_speed(model, device)
        fps = 1000.0 / mean_ms
        print(f"  Inference: {mean_ms:.2f} ± {std_ms:.2f} ms ({fps:.1f} FPS)")

        # Accuracy
        miou, ious, acc = evaluate_model(model, file_names, device, name)
        print(f"\n  mIoU: {miou:.4f}")
        print(f"  Pixel Accuracy: {acc:.4f}")
        print(f"  Per-class IoU:")
        for i, cname in enumerate(CLASS_NAMES):
            print(f"    {cname:15s}: {ious[i]:.4f}")

        results[name] = {
            "backbone": cfg_cls.backbone,
            "checkpoint": ckpt,
            "epoch": state.get("epoch"),
            "params": params,
            "params_M": round(params / 1e6, 3),
            "flops_G": round(flops / 1e9, 2) if flops else None,
            "inference_ms": round(mean_ms, 2),
            "inference_fps": round(fps, 1),
            "miou": round(float(miou), 4),
            "pixel_accuracy": round(float(acc), 4),
            "per_class_iou": {cname: round(float(ious[i]), 4) for i, cname in enumerate(CLASS_NAMES)},
        }

    # Comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Teacher vs Student")
    print(f"{'='*60}")
    t, s = results["teacher"], results["student"]
    print(f"  {'Metric':<20s} {'Teacher':>12s} {'Student':>12s} {'Delta':>10s}")
    print(f"  {'-'*54}")
    print(f"  {'Params (M)':<20s} {t['params_M']:>12.3f} {s['params_M']:>12.3f} {s['params_M']/t['params_M']*100:>9.1f}%")
    if t['flops_G'] and s['flops_G']:
        print(f"  {'FLOPs (G)':<20s} {t['flops_G']:>12.2f} {s['flops_G']:>12.2f} {s['flops_G']/t['flops_G']*100:>9.1f}%")
    print(f"  {'Inference (ms)':<20s} {t['inference_ms']:>12.2f} {s['inference_ms']:>12.2f} {s['inference_ms']/t['inference_ms']*100:>9.1f}%")
    print(f"  {'mIoU':<20s} {t['miou']:>12.4f} {s['miou']:>12.4f} {s['miou']-t['miou']:>+10.4f}")
    print(f"  {'Pixel Accuracy':<20s} {t['pixel_accuracy']:>12.4f} {s['pixel_accuracy']:>12.4f} {s['pixel_accuracy']-t['pixel_accuracy']:>+10.4f}")
    for cname in CLASS_NAMES:
        ti, si = t['per_class_iou'][cname], s['per_class_iou'][cname]
        print(f"  {cname+' IoU':<20s} {ti:>12.4f} {si:>12.4f} {si-ti:>+10.4f}")

    # Save comparison
    out_path = os.path.join(PROJECT_ROOT, "results", "kd_student", "eval_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save student predictions for visualization
    vis_dir = os.path.join(PROJECT_ROOT, "results", "kd_student", "predictions")
    os.makedirs(vis_dir, exist_ok=True)
    student_model, _ = load_model(StudentConfig, args.student_ckpt, device)
    for i in range(min(args.num_vis, len(file_names))):
        fn = file_names[i]
        base_name = fn.split("/")[1].replace(".png", "")
        rgb = cv2.imread(os.path.join(DATASET_DIR, "RGB", base_name + ".png"), cv2.IMREAD_COLOR)
        depth = cv2.imread(os.path.join(DATASET_DIR, "Depth", base_name + ".png"), cv2.IMREAD_GRAYSCALE)
        pred = predict_single(student_model, rgb, depth, device)
        vis = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cls, color in VIS_COLORS.items():
            vis[pred == cls] = color
        cv2.imwrite(os.path.join(vis_dir, f"{base_name}_pred.png"), vis)
    print(f"Predictions saved to {vis_dir}/")


if __name__ == "__main__":
    main()
