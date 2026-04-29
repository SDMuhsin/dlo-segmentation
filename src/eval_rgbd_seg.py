"""
Evaluate a trained DFormer RGB-D segmentation model on the test set.

Loads the best model checkpoint and runs evaluation on the test split,
producing per-class IoU, mIoU, confusion matrix, and sample predictions.

Usage:
  source env/bin/activate
  python src/eval_rgbd_seg.py [--checkpoint results/dformer_cdlo/best_model.pth]
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

DFORMER_DIR = os.path.join(os.path.dirname(__file__), "dformer")
sys.path.insert(0, DFORMER_DIR)

from models.builder import EncoderDecoder as DFormerModel

PROJECT_ROOT = "/workspace/kiat_crefle"
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dformer_dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "dformer_cdlo")

NUM_CLASSES = 5
CLASS_NAMES = ["Wire", "Endpoint", "Bifurcation", "Connector", "Noise"]
BACKGROUND = 255
IMAGE_H, IMAGE_W = 480, 640

# Visualization colors (BGR) for predictions
VIS_COLORS = {
    0: (180, 180, 180),  # Wire - gray
    1: (0, 0, 255),      # Endpoint - red
    2: (255, 0, 0),      # Bifurcation - blue
    3: (0, 255, 0),      # Connector - green
    4: (0, 255, 255),    # Noise - yellow
}

RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEPTH_MEAN = torch.tensor([0.48, 0.48, 0.48]).view(1, 3, 1, 1)
DEPTH_STD = torch.tensor([0.28, 0.28, 0.28]).view(1, 3, 1, 1)


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


def load_model(checkpoint_path, device):
    cfg = ModelConfig()
    model = DFormerModel(cfg=cfg, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model = model.to(device).eval()
    return model, state


def predict_single(model, rgb_bgr, depth_8bit, device):
    """Run inference on a single image pair."""
    # Normalize
    rgb_t = torch.from_numpy(rgb_bgr.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0) / 255.0
    rgb_t = (rgb_t - RGB_MEAN) / RGB_STD

    d_t = torch.from_numpy(depth_8bit.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255.0
    d_t = d_t.expand(-1, 3, -1, -1)
    d_t = (d_t - DEPTH_MEAN) / DEPTH_STD

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(rgb_t.to(device), d_t.to(device))
    return out.argmax(dim=1).squeeze(0).cpu().numpy()


def visualize_prediction(pred, save_path):
    """Convert prediction map to colored visualization."""
    vis = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for cls, color in VIS_COLORS.items():
        vis[pred == cls] = color
    cv2.imwrite(save_path, vis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(RESULTS_DIR, "best_model.pth"))
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--num_vis", type=int, default=10, help="Number of visualization samples")
    args = parser.parse_args()

    device = torch.device("cuda:0")

    print(f"Loading model from {args.checkpoint}")
    model, state = load_model(args.checkpoint, device)
    print(f"  Epoch: {state.get('epoch', '?')}, mIoU: {state.get('miou', '?'):.4f}")
    if "ious" in state:
        for name, iou in state["ious"].items():
            print(f"    {name}: {iou:.4f}")

    # Load data
    txt_file = os.path.join(DATASET_DIR, "train.txt" if args.split == "train" else "test.txt")
    with open(txt_file) as f:
        file_names = [l.strip() for l in f if l.strip()]

    # Evaluate
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    vis_dir = os.path.join(RESULTS_DIR, "predictions")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"\nEvaluating on {args.split} ({len(file_names)} images)...")
    for i, fn in enumerate(file_names):
        base_name = fn.split("/")[1].replace(".png", "")
        rgb = cv2.imread(os.path.join(DATASET_DIR, "RGB", base_name + ".png"), cv2.IMREAD_COLOR)
        depth = cv2.imread(os.path.join(DATASET_DIR, "Depth", base_name + ".png"), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(DATASET_DIR, "Label", base_name + ".png"), cv2.IMREAD_GRAYSCALE)

        # gt_transform
        label_t = label.astype(np.int16) - 1
        label_t[label_t < 0] = 255

        pred = predict_single(model, rgb, depth, device)

        # Update confusion matrix
        k = (label_t >= 0) & (label_t < NUM_CLASSES)
        hist += np.bincount(NUM_CLASSES * label_t[k].astype(int) + pred[k].astype(int),
                            minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)

        # Save visualizations
        if i < args.num_vis:
            visualize_prediction(pred, os.path.join(vis_dir, f"{base_name}_pred.png"))
            # Also save ground truth for comparison
            gt_vis = np.zeros_like(rgb)
            for cls, color in VIS_COLORS.items():
                gt_vis[label_t == cls] = color
            cv2.imwrite(os.path.join(vis_dir, f"{base_name}_gt.png"), gt_vis)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(file_names)}")

    # Compute metrics
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    miou = np.nanmean(iou)
    acc = np.diag(hist).sum() / (hist.sum() + 1e-10)

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({args.split})")
    print(f"{'='*50}")
    print(f"  mIoU:     {miou:.4f}")
    print(f"  Pixel Acc: {acc:.4f}")
    print(f"\n  Per-class IoU:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:15s}: {iou[i]:.4f}")

    print(f"\n  Confusion Matrix:")
    header = "  " + " " * 15 + "".join(f"{n[:8]:>10s}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name:15s}" + "".join(f"{hist[i, j]:10d}" for j in range(NUM_CLASSES))
        print(row)

    # Save results
    results = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "epoch": state.get("epoch"),
        "miou": float(miou),
        "pixel_accuracy": float(acc),
        "per_class_iou": {name: float(iou[i]) for i, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": hist.tolist(),
    }
    results_path = os.path.join(RESULTS_DIR, f"eval_{args.split}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Predictions saved to {vis_dir}/")


if __name__ == "__main__":
    main()
