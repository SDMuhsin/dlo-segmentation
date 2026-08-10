"""Run the trained SegFormer-B5 (RGB-only, binary DLO) on a real-world video.

Phase 7 deliverable extension. The model was trained on 640x480 BGR synthetic
renders (orthographic, photographic backdrop, 3D real-object clutter). A real
phone-camera video has perspective projection, real lighting, and a hand in
frame — so the goal is to bring the input as close to the training format as
possible without lying about the content (no fake depth, no synthetic backdrop).

Preprocessing pipeline (matches training distribution):
    1. read frame as BGR uint8 (OpenCV default, same as cache)
    2. center-crop to 4:3 aspect ratio (training was 640x480, all sources 4:3)
       — preserves pixel content; no stretch, no black bars (which the model
       has never seen at the periphery)
    3. resize to 640x480 (cv2.INTER_AREA when shrinking, INTER_LINEAR otherwise)
    4. hand off to predict(model, bgr) which does the BGR->RGB swap + ImageNet
       normalisation that the training pipeline used (verbatim from
       gen_rgb_only_sota_gifs.py)

Two modes:
    --sanity-check   extract N evenly-spaced frames; save preprocessed PNG +
                     overlay PNG + a stats JSON. Use this BEFORE running on the
                     full video to verify orientation, color order, mask sanity.
    (default)        run on the full video; write a side-by-side mp4
                     [preprocessed | DLO overlay] at input fps + a per-frame
                     CSV of mask coverage %.
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE = os.path.join(PROJECT_ROOT, "data", "hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_rgb_only_sota_gifs import (  # noqa: E402
    load_model,
    predict,
    overlay_mask,
    add_label_bar,
)
from train_rgb_only_sota import IMAGE_H, IMAGE_W, BACKBONE_DEFAULT  # noqa: E402

assert (IMAGE_H, IMAGE_W) == (480, 640), "training image size changed; update preprocess"
TARGET_AR = IMAGE_W / IMAGE_H  # 4:3


def center_crop_to_aspect(frame_bgr, target_ar=TARGET_AR):
    """Center-crop frame_bgr (H, W, 3) to the given aspect ratio (W/H)."""
    h, w = frame_bgr.shape[:2]
    src_ar = w / h
    if abs(src_ar - target_ar) < 1e-6:
        return frame_bgr
    if src_ar > target_ar:
        # too wide — crop horizontally
        new_w = int(round(h * target_ar))
        x0 = (w - new_w) // 2
        return frame_bgr[:, x0:x0 + new_w, :]
    # too tall — crop vertically
    new_h = int(round(w / target_ar))
    y0 = (h - new_h) // 2
    return frame_bgr[y0:y0 + new_h, :, :]


def preprocess(frame_bgr):
    """Center-crop to 4:3 then resize to 640x480 BGR uint8."""
    cropped = center_crop_to_aspect(frame_bgr, TARGET_AR)
    h, w = cropped.shape[:2]
    interp = cv2.INTER_AREA if (w >= IMAGE_W and h >= IMAGE_H) else cv2.INTER_LINEAR
    out = cv2.resize(cropped, (IMAGE_W, IMAGE_H), interpolation=interp)
    assert out.shape == (IMAGE_H, IMAGE_W, 3) and out.dtype == np.uint8
    return out


# Per-class overlay colours (BGR) for 3-way inference. Wire keeps green
# (matches the binary overlay so side-by-side review reads the same way);
# backdrop is transparent.
#
# Class 2 means different things in the two 3-way model families that exist
# here: the Phase 11 model called it "objects" (background clutter), the
# connector model calls it "connector". Mislabelling it makes a visual review
# read the wrong thing, so the name and colour are selectable and the default
# preserves the historical (objects/cyan) behaviour.
CLS2_STYLES = {
    "objects":   {"color": (255, 200, 0), "name": "objects",   "legend": "cyan"},
    "connector": {"color": (200, 0, 200), "name": "connector", "legend": "magenta"},
}

OVERLAY_COLORS_3WAY = {
    1: (0, 255, 0),    # wire, green
    2: CLS2_STYLES["objects"]["color"],
}

OVERLAY_LEGEND_LABELS_3WAY = "wire=green   objects=cyan"


def set_class2_style(kind):
    """Point the 3-way overlay at the right meaning for class 2."""
    global OVERLAY_LEGEND_LABELS_3WAY
    style = CLS2_STYLES[kind]
    OVERLAY_COLORS_3WAY[2] = style["color"]
    OVERLAY_LEGEND_LABELS_3WAY = f"wire=green   {style['name']}={style['legend']}"
    return style["name"]


def _overlay_three_way(frame_bgr, mask, alpha=0.5):
    """Composite a per-class translucent overlay for {1=wire, 2=objects}."""
    out = frame_bgr.copy()
    for cls, color in OVERLAY_COLORS_3WAY.items():
        sel = (mask == cls)
        if not sel.any():
            continue
        color_layer = np.zeros_like(frame_bgr)
        color_layer[sel] = color
        out[sel] = (out[sel] * (1 - alpha) + color_layer[sel] * alpha).astype(np.uint8)
    return out


def make_side_by_side(frame_bgr, mask, three_way=False):
    if three_way:
        overlay = _overlay_three_way(frame_bgr, mask)
        title = (f"Predicted 3-way mask "
                 f"({OVERLAY_LEGEND_LABELS_3WAY})")
    else:
        overlay = overlay_mask(frame_bgr, mask, color_bgr=(0, 255, 0), alpha=0.5)
        title = "Predicted DLO mask (RGB-only SegFormer-B5)"
    left = add_label_bar(frame_bgr, "Input RGB (preprocessed 640x480)")
    right = add_label_bar(overlay, title)
    sep = np.full((left.shape[0], 4, 3), 200, dtype=np.uint8)
    return np.hstack([left, sep, right])


def make_overlay_only(frame_bgr, mask, three_way=False):
    if three_way:
        return _overlay_three_way(frame_bgr, mask)
    return overlay_mask(frame_bgr, mask, color_bgr=(0, 255, 0), alpha=0.5)


def _infer_num_classes(model):
    """Read num_labels off the wrapped HuggingFace SegformerForSemanticSegmentation."""
    try:
        return int(model.model.config.num_labels)
    except Exception:
        return 2


def sanity_check(video_path, model, device, out_dir, n_samples=6,
                 num_classes=2):
    three_way = (num_classes == 3)
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {video_path}")
    print(f"  source: {src_w}x{src_h}  ({src_w / src_h:.3f} AR), {n_frames} frames @ {fps:.2f} fps")
    print(f"  model output: {'3-way (backdrop/wire/objects)' if three_way else 'binary (bg/DLO)'}")

    # Pick n_samples evenly-spaced frame indices
    if n_frames <= 0:
        raise RuntimeError("no frames in video")
    sample_idx = np.linspace(0, n_frames - 1, n_samples, dtype=int).tolist()
    print(f"  sampling frame indices: {sample_idx}")

    info = {
        "video": video_path,
        "source_dims": [src_w, src_h],
        "source_aspect_ratio": src_w / src_h,
        "training_dims": [IMAGE_W, IMAGE_H],
        "training_aspect_ratio": TARGET_AR,
        "preprocessing": "center-crop to 4:3 then resize to 640x480",
        "n_frames": n_frames,
        "fps": fps,
        "samples": [],
    }

    for sample_n, idx in enumerate(sample_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"    frame {idx}: read failed; skipping")
            continue
        # Preprocess
        pre = preprocess(frame)
        # Save the preprocessed BGR frame as PNG (cv2 writes BGR).
        pre_png = os.path.join(out_dir, f"sample{sample_n:02d}_idx{idx:05d}_preprocessed.png")
        cv2.imwrite(pre_png, pre)

        # Run model
        t0 = time.time()
        mask = predict(model, pre, device)
        dt_ms = (time.time() - t0) * 1000
        # Sanity stats
        per_chan_mean = pre.reshape(-1, 3).mean(axis=0).tolist()  # B, G, R order
        if three_way:
            n_wire = int((mask == 1).sum())
            n_obj  = int((mask == 2).sum())
            cov_wire = n_wire / mask.size
            cov_obj  = n_obj / mask.size
            sample_stats = {
                "mask_wire_pixels":         n_wire,
                "mask_wire_coverage_pct":   round(100 * cov_wire, 3),
                "mask_objects_pixels":      n_obj,
                "mask_objects_coverage_pct": round(100 * cov_obj, 3),
            }
        else:
            n_dlo = int(mask.sum())
            cov = n_dlo / mask.size
            sample_stats = {
                "mask_dlo_pixels": n_dlo,
                "mask_dlo_coverage_pct": round(100 * cov, 3),
            }

        # Overlay PNG
        side = make_side_by_side(pre, mask, three_way=three_way)
        side_png = os.path.join(out_dir, f"sample{sample_n:02d}_idx{idx:05d}_overlay.png")
        cv2.imwrite(side_png, side)

        info["samples"].append({
            "sample_n": sample_n,
            "frame_idx": idx,
            "preprocessed_png": os.path.relpath(pre_png, PROJECT_ROOT),
            "overlay_png": os.path.relpath(side_png, PROJECT_ROOT),
            "preprocessed_bgr_mean": per_chan_mean,
            "preprocessed_min_max": [int(pre.min()), int(pre.max())],
            "predict_ms": round(dt_ms, 1),
            **sample_stats,
        })
        if three_way:
            print(f"    frame {idx:5d}: BGR_mean=[{per_chan_mean[0]:.1f},{per_chan_mean[1]:.1f},{per_chan_mean[2]:.1f}]  "
                  f"wire={100*cov_wire:.2f}%  obj={100*cov_obj:.2f}%  predict={dt_ms:.0f}ms")
        else:
            print(f"    frame {idx:5d}: BGR_mean=[{per_chan_mean[0]:.1f},{per_chan_mean[1]:.1f},{per_chan_mean[2]:.1f}]  "
                  f"DLO cov={100*cov:.2f}%  predict={dt_ms:.0f}ms")

    cap.release()
    info_path = os.path.join(out_dir, "sanity.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nsanity-check complete. Inspect:")
    print(f"  {os.path.relpath(out_dir, PROJECT_ROOT)}/sample*_preprocessed.png  (look right-side-up? colour right? wire visible?)")
    print(f"  {os.path.relpath(out_dir, PROJECT_ROOT)}/sample*_overlay.png       (does the green mask follow the wire?)")
    print(f"  {os.path.relpath(info_path, PROJECT_ROOT)}                         (per-sample stats)")


def run_full(video_path, model, device, out_dir, output_fps=None, max_frames=None,
             overlay_only=False, also_gif=False, gif_max_frames=200, gif_scale=0.5,
             num_classes=2):
    three_way = (num_classes == 3)
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if output_fps is None:
        output_fps = fps
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    if overlay_only:
        def make_frame(pre, mask):
            return make_overlay_only(pre, mask, three_way=three_way)
    else:
        def make_frame(pre, mask):
            return make_side_by_side(pre, mask, three_way=three_way)
    out_name = "overlay.mp4" if overlay_only else "side_by_side.mp4"

    # Probe one frame to get the output dims
    ok, f0 = cap.read()
    if not ok:
        raise RuntimeError("could not read first frame")
    pre0 = preprocess(f0)
    out0 = make_frame(pre0, np.zeros((IMAGE_H, IMAGE_W), dtype=np.int64))
    out_h, out_w = out0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_mp4 = os.path.join(out_dir, out_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, output_fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {out_mp4}")

    csv_path = os.path.join(out_dir, "per_frame.csv")
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    if three_way:
        csv_w.writerow(["frame_idx",
                        "wire_coverage_pct", "wire_pixels",
                        "objects_coverage_pct", "objects_pixels"])
    else:
        csv_w.writerow(["frame_idx", "dlo_coverage_pct", "dlo_pixels"])

    # GIF prep — sample evenly to fit gif_max_frames
    gif_frames = []
    gif_keep_every = max(1, n_frames // gif_max_frames) if also_gif else 1

    print(f"writing {out_mp4} ({out_w}x{out_h} @ {output_fps:.2f} fps), processing {n_frames} frames ...")
    if also_gif:
        print(f"  also writing gif (every {gif_keep_every}th frame, scale {gif_scale}, target ~{n_frames//gif_keep_every} frames)")
    t0 = time.time()
    processed = 0
    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        pre = preprocess(frame)
        mask = predict(model, pre, device)
        out_frame = make_frame(pre, mask)
        writer.write(out_frame)
        if also_gif and (fi % gif_keep_every == 0):
            small = cv2.resize(out_frame, (int(out_w * gif_scale), int(out_h * gif_scale)),
                               interpolation=cv2.INTER_AREA)
            gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if three_way:
            n_wire = int((mask == 1).sum())
            n_obj  = int((mask == 2).sum())
            csv_w.writerow([fi,
                            round(100 * n_wire / mask.size, 4), n_wire,
                            round(100 * n_obj  / mask.size, 4), n_obj])
        else:
            n_dlo = int(mask.sum())
            csv_w.writerow([fi, round(100 * n_dlo / mask.size, 4), n_dlo])
        processed += 1
        if processed % 120 == 0:
            elapsed = time.time() - t0
            eta = elapsed / processed * (n_frames - processed)
            print(f"  frame {processed}/{n_frames}  ({processed / elapsed:.1f} fps, ETA {eta:.0f}s)")

    writer.release()
    cap.release()
    csv_f.close()
    elapsed = time.time() - t0
    print(f"\ndone. {processed} frames in {elapsed:.1f}s ({processed/elapsed:.1f} fps).")
    print(f"  {out_mp4}")
    print(f"  {csv_path}")

    if also_gif and gif_frames:
        from PIL import Image
        gif_path = os.path.join(out_dir, out_name.replace(".mp4", ".gif"))
        gif_fps = output_fps / gif_keep_every
        duration = max(20, int(round(1000 / gif_fps)))
        pil = [Image.fromarray(f) for f in gif_frames]
        pil[0].save(gif_path, save_all=True, append_images=pil[1:],
                    duration=duration, loop=0, optimize=True)
        print(f"  {gif_path}  ({len(gif_frames)} frames @ {gif_fps:.1f} fps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sanity-check", action="store_true",
                    help="extract N samples, save preprocessed/overlay PNGs, exit.")
    ap.add_argument("--n-samples", type=int, default=6,
                    help="number of evenly-spaced samples for sanity-check mode")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap full-mode frame count (e.g. for a quick test)")
    ap.add_argument("--output-fps", type=float, default=None,
                    help="override mp4 output fps (default = source fps)")
    ap.add_argument("--overlay-only", action="store_true",
                    help="single-panel output (overlay only, 640x480) instead of side-by-side")
    ap.add_argument("--also-gif", action="store_true",
                    help="also write a downsampled GIF alongside the mp4")
    ap.add_argument("--gif-max-frames", type=int, default=200,
                    help="cap GIF frame count by sampling evenly (default 200)")
    ap.add_argument("--gif-scale", type=float, default=0.5,
                    help="GIF resolution scale relative to mp4 (default 0.5)")
    ap.add_argument("--backbone", default=BACKBONE_DEFAULT)
    ap.add_argument("--cls2", default="objects", choices=tuple(CLS2_STYLES),
                    help="What class 2 means in a 3-way checkpoint: 'objects' "
                         "(default, the Phase 11 model) or 'connector' (the "
                         "connector model). Sets the overlay colour and legend "
                         "so a visual review is not mislabelled.")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.ckpt, device, backbone=args.backbone)
    nc = _infer_num_classes(model)
    if nc == 3:
        print(f"  class 2 rendered as: {set_class2_style(args.cls2)}")
    if args.sanity_check:
        sanity_check(args.video, model, device, args.out_dir,
                     n_samples=args.n_samples, num_classes=nc)
    else:
        run_full(args.video, model, device, args.out_dir,
                 output_fps=args.output_fps, max_frames=args.max_frames,
                 overlay_only=args.overlay_only, also_gif=args.also_gif,
                 gif_max_frames=args.gif_max_frames, gif_scale=args.gif_scale,
                 num_classes=nc)


if __name__ == "__main__":
    main()
