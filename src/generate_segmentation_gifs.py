"""
Generate side-by-side GIFs showing animated DLO segmentation results.

Each GIF has 3 panels:
  Left:   Ground truth labels (class-colored)
  Middle: Unlabeled DLO (single color on black background)
  Right:  Model prediction (class-colored)

Uses all 20 animation frames per source frame for smooth animation.
"""

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

DFORMER_DIR = os.path.join(os.path.dirname(__file__), "dformer")
sys.path.insert(0, DFORMER_DIR)

from models.builder import EncoderDecoder as DFormerModel

# ─── Config ───
PROJECT_ROOT = "/workspace/kiat_crefle"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "rgbd_videos")
CHECKPOINT = os.path.join(PROJECT_ROOT, "results", "dformer_cdlo_honest", "best_model.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "dformer_cdlo_honest", "gifs")

NUM_CLASSES = 5
CLASS_NAMES = ["Wire", "Endpoint", "Bifurcation", "Connector", "Noise"]
IMAGE_H, IMAGE_W = 480, 640
DEPTH_MIN_MM, DEPTH_MAX_MM = 500, 1500
NUM_ANIM_FRAMES = 20

# Class colors (BGR for OpenCV)
CLASS_COLORS_BGR = {
    0: (180, 180, 180),  # Wire - gray
    1: (0, 0, 255),      # Endpoint - red
    2: (255, 0, 0),      # Bifurcation - blue
    3: (0, 255, 0),      # Connector - green
    4: (0, 255, 255),    # Noise - yellow
}

# Single color for unlabeled view (light cyan/teal)
UNLABELED_COLOR_BGR = (210, 180, 90)  # pleasant teal

# Normalization
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEPTH_MEAN = torch.tensor([0.48, 0.48, 0.48]).view(1, 3, 1, 1)
DEPTH_STD = torch.tensor([0.28, 0.28, 0.28]).view(1, 3, 1, 1)

# Panel labels
PANEL_LABELS = ["Ground Truth", "Input (Unlabeled)", "Model Prediction"]
LABEL_HEIGHT = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
SEPARATOR_WIDTH = 4


class ModelConfig:
    backbone = "DFormer-Tiny"
    pretrained_model = None
    decoder = "MLPDecoder"
    decoder_embed_dim = 256
    num_classes = NUM_CLASSES
    background = 255
    bn_eps = 1e-3
    bn_momentum = 0.1
    drop_path_rate = 0.1
    aux_rate = 0.0
    fix_bias = True


def load_model(device):
    cfg = ModelConfig()
    model = DFormerModel(cfg=cfg, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model = model.to(device).eval()
    print(f"Loaded model from {CHECKPOINT} (epoch {state.get('epoch', '?')}, mIoU={state.get('miou', 0):.4f})")
    return model


def depth_16to8(depth_16):
    """Convert 16-bit depth (mm) to 8-bit. 0→0, 500-1500→1-255."""
    out = np.zeros(depth_16.shape, dtype=np.uint8)
    mask = depth_16 > 0
    clipped = np.clip(depth_16[mask].astype(np.float32), DEPTH_MIN_MM, DEPTH_MAX_MM)
    out[mask] = ((clipped - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 254 + 1).astype(np.uint8)
    return out


def make_model_input(rgb_bgr):
    """Convert class-colored image to uniform gray foreground (what the model was trained on)."""
    fg_mask = rgb_bgr.sum(axis=2) > 0
    uniform = np.zeros_like(rgb_bgr)
    uniform[fg_mask] = 128
    return uniform


@torch.no_grad()
def predict(model, rgb_bgr, depth_8, device):
    """Run model inference on a single image pair.
    Feeds uniform-gray foreground (not class-colored) to match training."""
    model_rgb = make_model_input(rgb_bgr)
    rgb_t = torch.from_numpy(model_rgb.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0) / 255.0
    rgb_t = (rgb_t - RGB_MEAN) / RGB_STD

    d_t = torch.from_numpy(depth_8.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255.0
    d_t = d_t.expand(-1, 3, -1, -1)
    d_t = (d_t - DEPTH_MEAN) / DEPTH_STD

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(rgb_t.to(device), d_t.to(device))
    return out.argmax(dim=1).squeeze(0).cpu().numpy()


def colorize_prediction(pred):
    """Convert prediction map to class-colored BGR image."""
    vis = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS_BGR.items():
        vis[pred == cls] = color
    return vis


def make_unlabeled(rgb_bgr):
    """Show what the model actually sees: uniform gray foreground on black background."""
    return make_model_input(rgb_bgr)


def add_panel_labels(combined):
    """Add text labels above each panel."""
    h, w = combined.shape[:2]
    panel_w = (w - 2 * SEPARATOR_WIDTH) // 3

    # Create label bar
    label_bar = np.zeros((LABEL_HEIGHT, w, 3), dtype=np.uint8)
    label_bar[:] = (30, 30, 30)  # dark gray background

    for i, label in enumerate(PANEL_LABELS):
        x_start = i * (panel_w + SEPARATOR_WIDTH)
        text_size = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = x_start + (panel_w - text_size[0]) // 2
        text_y = LABEL_HEIGHT - 8
        cv2.putText(label_bar, label, (text_x, text_y), FONT, FONT_SCALE,
                    (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)

    return np.vstack([label_bar, combined])


def generate_gif(model, device, split, set_id, src_frame, view, output_path):
    """Generate a 3-panel GIF for one animated DLO sample."""
    frames = []
    rgb_dir = os.path.join(DATA_ROOT, split, set_id, "rgb")
    depth_dir = os.path.join(DATA_ROOT, split, set_id, "depth")

    for anim in range(NUM_ANIM_FRAMES):
        fname = f"{src_frame:04d}_{anim:02d}_{view}.png"

        # Load original images
        rgb_path = os.path.join(rgb_dir, fname)
        depth_path = os.path.join(depth_dir, fname)

        if not os.path.exists(rgb_path):
            print(f"  Missing: {rgb_path}")
            continue

        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_8 = depth_16to8(depth_16)

        # Left: ground truth (the class-colored image itself)
        gt_panel = rgb_bgr.copy()

        # Middle: unlabeled (single color)
        unlabeled_panel = make_unlabeled(rgb_bgr)

        # Right: model prediction (masked to DLO region only)
        pred = predict(model, rgb_bgr, depth_8, device)
        pred_panel = colorize_prediction(pred)
        # Mask out background (where original image is black / depth is 0)
        bg_mask = rgb_bgr.sum(axis=2) == 0
        pred_panel[bg_mask] = 0

        # Separator (white vertical line)
        sep = np.ones((IMAGE_H, SEPARATOR_WIDTH, 3), dtype=np.uint8) * 255

        # Combine panels
        combined = np.hstack([gt_panel, sep, unlabeled_panel, sep, pred_panel])

        # Add labels
        combined = add_panel_labels(combined)

        # Convert BGR → RGB for PIL
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(combined_rgb))

    if not frames:
        print(f"  No frames generated!")
        return

    # Save as GIF (loop, 100ms per frame = 10 fps, bounce effect)
    # Create bounce: forward + reverse
    bounce_frames = frames + frames[-2:0:-1]

    bounce_frames[0].save(
        output_path,
        save_all=True,
        append_images=bounce_frames[1:],
        duration=100,
        loop=0,
    )
    print(f"  Saved: {output_path} ({len(bounce_frames)} frames)")


def main():
    device = torch.device("cuda:0")
    model = load_model(device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate GIFs for several samples across val and test sets
    # Pick diverse samples: different sets, different source frames, front view
    samples = [
        # (split, set_id, src_frame, view)
        ("val", "032", 0, "front"),
        ("val", "032", 50, "front"),
        ("val", "032", 150, "front"),
        ("val", "034", 0, "front"),
        ("val", "034", 100, "front"),
        ("val", "035", 0, "front"),
        ("val", "035", 75, "front"),
        # Also some non-front views for variety
        ("val", "032", 0, "top"),
        ("val", "034", 50, "right"),
        ("test", "036", 0, "front"),
        ("test", "036", 100, "front"),
        ("test", "037", 0, "front"),
    ]

    print(f"\nGenerating {len(samples)} GIFs...")
    for split, set_id, src_frame, view in samples:
        name = f"{split}_{set_id}_{src_frame:04d}_{view}"
        output_path = os.path.join(OUTPUT_DIR, f"{name}.gif")
        print(f"\n{name}:")
        generate_gif(model, device, split, set_id, src_frame, view, output_path)

    print(f"\nAll GIFs saved to {OUTPUT_DIR}/")
    print(f"Total: {len(os.listdir(OUTPUT_DIR))} files")


if __name__ == "__main__":
    main()
