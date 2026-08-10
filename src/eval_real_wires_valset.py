#!/usr/bin/env python
"""
Evaluate a trained RGB-only SegFormer on the partner-team REAL electric-wires RGB-D
validation set (data/real_wires_valset), which ships human-annotated binary GT masks.

This is the project's first QUANTITATIVE real-world IoU. Every prior real-world
assessment was judging data/dlo_real_sample_{1..4}.mp4 by eye, with no GT;
synthetic-val IoU is a known liar (stayed ~0.90 on runs that collapsed on real data).

Preprocessing reuses the EXACT path used for those 4 videos
(infer_video_rgb_only.preprocess: center-crop to 4:3 -> resize 640x480; then
gen_rgb_only_sota_gifs.predict: BGR->RGB, /255, ImageNet-normalize, argmax), so the
number here is directly comparable to all prior real-world work.

IoU is computed at the cropped human-GT resolution (the prediction is upsampled with
nearest-neighbour; the human annotation is never downsampled) -- the standard semantic
segmentation convention. Caveat: the 16:9 -> 4:3 center-crop excludes the left/right
margins from both prediction and GT (the model's input aspect is 4:3).

Outputs (results/real_wires_valset_eval/):
  summary.json       overall + per-camera pooled IoU / mean IoU / precision / recall
  per_image.csv      one row per frame
  compare/<bn>.png   4-panel overlay [input | GT | pred | error(TP green/FP red/FN blue)]
"""
import os
import sys
import glob
import json
import csv
import argparse

import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# infer_video_rgb_only sets HF_HOME/TRANSFORMERS_CACHE -> data/hf_cache on import.
from infer_video_rgb_only import preprocess, center_crop_to_aspect  # noqa: E402
from gen_rgb_only_sota_gifs import (  # noqa: E402
    predict,
    overlay_mask,
    add_label_bar,
)
from train_rgb_only_sota import (  # noqa: E402
    SegFormerSegmenter,
    NUM_CLASSES,
    IMAGE_H,
    IMAGE_W,
    RGB_MEAN,
    RGB_STD,
    BACKBONE_DEFAULT,
)

# ── RGB-D (DFormer) bridge constants — match the trainer EXACTLY ──
# These mirror src/train_dformer_v2_dlo.py (depth normalisation + model cfg) and
# the validated /tmp/rgbd_smoke.py depth preprocessing. They are referenced ONLY
# by the --model-type dformer path; the default segformer path never touches them.
DEPTH_MEAN = torch.tensor([0.48, 0.48, 0.48], dtype=torch.float32).view(1, 3, 1, 1)
DEPTH_STD = torch.tensor([0.28, 0.28, 0.28], dtype=torch.float32).view(1, 3, 1, 1)
DFORMER_PRETRAINED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "pretrained", "DFormerv2", "pretrained", "DFormerv2_Large_pretrained.pth",
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "real_wires_valset")
DEFAULT_CKPT = os.path.join(
    PROJECT_ROOT,
    "results/segformer_b5_rgb_phase15_wirefree_ft/full_20260605_0409/best_model.pth",
)
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "results", "real_wires_valset_eval")
DEFAULT_SIZE = f"{IMAGE_H}x{IMAGE_W}"  # 480x640 = training/eval default
MASK_THRESH = 127  # masks are lossy-JPEG but effectively binary (0% mid-range pixels)


def parse_size(s):
    """'HxW' -> (H, W)."""
    try:
        h, w = (int(t) for t in s.lower().split("x"))
        assert h > 0 and w > 0
        return h, w
    except Exception:
        raise argparse.ArgumentTypeError(f"--infer-size must be HxW, got {s!r}")


def load_model_auto(ckpt_path, device):
    """gen_rgb_only_sota_gifs.load_model + a state-dict-shape fallback for
    num_classes, so checkpoints that never saved num_classes in config/args
    (e.g. legacy multi-class heads) still build the right head. Same mechanism
    as the 4-video inference path (cfg > saved args > default), with the head
    weight shape inserted before the final default. Wire = argmax>0 downstream,
    so any num_classes works."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = state.get("args") or {}
    cfg = state.get("config") or {}
    backbone_name = cfg.get("backbone") or saved_args.get("backbone") or BACKBONE_DEFAULT
    sd = state.get("model_state_dict", state)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    num_classes = int(cfg.get("num_classes") or saved_args.get("num_classes") or 0)
    if num_classes <= 0:  # fallback: read the classifier head shape
        for k, v in sd.items():
            if k.endswith("decode_head.classifier.weight"):
                num_classes = int(v.shape[0])
                break
    num_classes = num_classes or NUM_CLASSES
    model = SegFormerSegmenter(
        backbone_name=backbone_name, num_classes=num_classes, criterion=None
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")
    print(f"  model: backbone={backbone_name}  num_classes={num_classes}")
    return model.eval().to(device)


# ──────────────────────── RGB-D (DFormer) bridge ────────────────────────
#
# These functions implement --model-type dformer. They are completely separate
# from the segformer code path above; the default (segformer) path is unchanged
# and identical. The crop/resize/IoU/threshold-127/upsample-nearest logic
# in eval_one() is SHARED and identical for both model types — the ONLY
# differences are (1) the model construction and (2) the extra depth input.


def build_dformer(ckpt_path, device, pretrained=DFORMER_PRETRAINED):
    """Build DFormerv2-Large + ham (the SAME config the trainer uses) and load
    --ckpt. Returns (model, loaded_ok): loaded_ok is False if the ckpt could not
    be loaded (then the model is randomly-initialised, for a plumbing smoke)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dformer"))
    from models.builder import EncoderDecoder as DFormerModel  # noqa: E402

    class _Cfg:
        backbone = "DFormerv2_L"
        decoder = "ham"
        decoder_embed_dim = 1024
        num_classes = 2
        background = -1
        bn_eps = 1e-3
        bn_momentum = 0.1
        drop_path_rate = 0.3
        aux_rate = 0.0
        fix_bias = True

    cfg = _Cfg()
    loaded_ok = False
    # Only load the heavy pretrained backbone if we have no real ckpt to load;
    # if a ckpt is given, the full state dict below supersedes the pretrained.
    cfg.pretrained_model = None
    model = DFormerModel(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d,
                         syncbn=False).to(device)
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = state.get("model_state_dict", state)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # A genuine DFormer ckpt loads the encoder+decoder; if essentially
        # nothing matched, treat as not-loaded (e.g. a segformer ckpt).
        loaded_ok = len(sd) > 0 and len(unexpected) < len(sd)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys (first 3): {list(missing)[:3]}")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys (first 3): {list(unexpected)[:3]}")
        print(f"  dformer: loaded {len(sd) - len(unexpected)}/{len(sd)} ckpt tensors "
              f"({'OK' if loaded_ok else 'FAILED -> random init'})")
    else:
        print(f"  dformer: no ckpt at {ckpt_path!r} -> RANDOMLY-INITIALISED model "
              f"(plumbing smoke only)")
    return model.eval().to(device), loaded_ok


def preprocess_depth_dformer(depth_path, infer_hw):
    """Depth preprocess for the arbiter — IDENTICAL geometry to the RGB path
    (center-crop to the input aspect, then resize to HxW) but with INTER_NEAREST
    (never blend distances). Returns HxW uint8. Mirrors the validated
    /tmp/rgbd_smoke.py:preprocess_depth_real."""
    h, w = infer_hw
    d = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if d is None:
        raise RuntimeError(f"cannot read depth {depth_path}")
    d3 = np.repeat(d[:, :, None], 3, axis=2)  # 3ch so crop maths == RGB exactly
    crop = center_crop_to_aspect(d3, w / h)
    out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)[:, :, 0]
    return out


@torch.no_grad()
def predict_dformer(model, rgb_bgr, depth_u8, device, ablate_depth=False):
    """DFormer RGB-D forward -> argmax. rgb_bgr: HxWx3 BGR uint8 (already
    preprocessed); depth_u8: HxW uint8 (already preprocessed). RGB is normalised
    with the SAME ImageNet stats the trainer uses; depth uint8/255 -> expand 3ch
    -> mean 0.48/std 0.28. NO depth-domain-aug (real depth used raw).

    ablate_depth (P27 diagnostic, default OFF): replace the depth input with a
    SPATIALLY-CONSTANT map equal to the dataset's normalized mean (uint8 0.48*255
    -> normalizes to exactly 0), removing ALL spatial depth information while
    leaving the RGB path and every other the same. Lets us measure whether
    the model actually USES depth (synth-val/real-val IoU drop when ablated)."""
    h, w = rgb_bgr.shape[:2]
    rgb_rgb = rgb_bgr[:, :, ::-1].copy()  # BGR->RGB (matches segformer predict())
    rgb = torch.from_numpy(rgb_rgb.transpose(2, 0, 1)).unsqueeze(0).to(
        device, dtype=torch.float32) / 255.0
    rgb = (rgb - RGB_MEAN.to(device)) / RGB_STD.to(device)
    if ablate_depth:
        # constant = dataset normalized mean (0.48) -> after /255 already-in-[0,1]
        # path we feed 0.48 directly so (0.48 - 0.48)/0.28 = 0 everywhere.
        dep = torch.full((1, 1, h, w), float(DEPTH_MEAN.flatten()[0]),
                         device=device, dtype=torch.float32)
    else:
        dep = torch.from_numpy(depth_u8).unsqueeze(0).unsqueeze(0).to(
            device, dtype=torch.float32) / 255.0
    dep = dep.expand(-1, 3, -1, -1)
    dep = (dep - DEPTH_MEAN.to(device)) / DEPTH_STD.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16,
                        enabled=(device.type == "cuda")):
        logits = model(rgb, dep)
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy()


def preprocess_size(frame_bgr, infer_hw):
    """preprocess() generalised to any inference size (center-crop to the
    target aspect, then resize). Delegates to the imported preprocess() at the
    default 480x640 so default behaviour stays identical."""
    h, w = infer_hw
    if (h, w) == (IMAGE_H, IMAGE_W):
        return preprocess(frame_bgr)
    cropped = center_crop_to_aspect(frame_bgr, w / h)
    ch, cw = cropped.shape[:2]
    interp = cv2.INTER_AREA if (cw >= w and ch >= h) else cv2.INTER_LINEAR
    return cv2.resize(cropped, (w, h), interpolation=interp)


@torch.no_grad()
def predict_size(model, rgb_bgr, device, infer_hw):
    """predict() generalised to any inference size. The SegFormerSegmenter
    wrapper hard-upsamples logits to 480x640, so for other sizes we call the
    inner HF model and upsample the H/4 logits to the actual input size
    (identical maths to the wrapper, just a parametric size)."""
    h, w = infer_hw
    if (h, w) == (IMAGE_H, IMAGE_W):
        return predict(model, rgb_bgr, device)
    rgb_rgb = rgb_bgr[:, :, ::-1].copy()
    rgb = torch.from_numpy(rgb_rgb.transpose(2, 0, 1)).unsqueeze(0).to(
        device, dtype=torch.float32) / 255.0
    rgb = (rgb - RGB_MEAN.to(device)) / RGB_STD.to(device)
    logits = model.model(pixel_values=rgb).logits
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy()


def crop_mask_to_aspect(mask_bin, target_ar=IMAGE_W / IMAGE_H):
    """Apply the SAME center_crop_to_aspect used on the RGB, to a 2D binary mask.

    Stacking to 3 channels guarantees unchanged crop maths to the RGB path
    (the RGB is cropped by the same function), so prediction and GT stay aligned.
    """
    m3 = np.repeat(mask_bin[:, :, None], 3, axis=2)
    return center_crop_to_aspect(m3, target_ar)[:, :, 0]


def load_gt_binary(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"cannot read mask {mask_path}")
    return (m >= MASK_THRESH).astype(np.uint8)


def error_panel(pre_bgr, pred640, gt640):
    """TP=green, FP=red, FN=blue overlaid translucently on the preprocessed input."""
    out = pre_bgr.copy()
    p = pred640.astype(bool)
    g = gt640.astype(bool)
    layer = np.zeros_like(pre_bgr)
    layer[p & g] = (0, 255, 0)      # TP green
    layer[p & ~g] = (0, 0, 255)     # FP red
    layer[~p & g] = (255, 0, 0)     # FN blue
    sel = (p | g)
    alpha = 0.6
    out[sel] = (out[sel] * (1 - alpha) + layer[sel] * alpha).astype(np.uint8)
    return out


def eval_one(model, device, rgb_path, mask_path, save_compare=None,
             infer_hw=(IMAGE_H, IMAGE_W), wire_rule="fg",
             model_type="segformer", depth_path=None, ablate_depth=False):
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"cannot read image {rgb_path}")
    ih, iw = infer_hw
    pre = preprocess_size(rgb_bgr, infer_hw)        # HxW BGR uint8 (default 480x640)
    if model_type == "dformer":
        # RGB-D path: same RGB preprocessing as above; load + preprocess the
        # matching depth with IDENTICAL geometry (INTER_NEAREST). Everything
        # below (crop/IoU/threshold/upsample) is shared with the segformer path.
        depth_u8 = preprocess_depth_dformer(depth_path, infer_hw)
        pred_small = predict_dformer(model, pre, depth_u8, device,
                                     ablate_depth=ablate_depth)  # HxW argmax
    else:
        pred_small = predict_size(model, pre, device, infer_hw)  # HxW argmax
    if wire_rule == "class1":                        # wire = exactly class 1 (3-way ckpts)
        pred_small = (pred_small == 1).astype(np.uint8)
    else:                                            # any fg -> wire (binary teacher: ==1)
        pred_small = (pred_small >= 1).astype(np.uint8)

    gt_bin = load_gt_binary(mask_path)              # native res {0,1}
    gt_crop = crop_mask_to_aspect(gt_bin, iw / ih)  # cropped native res
    Hc, Wc = gt_crop.shape
    # benchmark-standard: upsample prediction to GT resolution, never degrade the GT
    pred_full = cv2.resize(pred_small, (Wc, Hc), interpolation=cv2.INTER_NEAREST)

    p = pred_full.astype(bool)
    g = gt_crop.astype(bool)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    union = tp + fp + fn
    iou = float(tp) / union if union > 0 else float("nan")

    if save_compare is not None:
        gt640 = cv2.resize(gt_crop, (pre.shape[1], pre.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        panels = [
            add_label_bar(pre, f"Input RGB (preprocessed {iw}x{ih})"),
            add_label_bar(overlay_mask(pre, gt640, (0, 255, 0), 0.5), "Ground truth (wire)"),
            add_label_bar(overlay_mask(pre, pred_small, (0, 255, 0), 0.5), "Teacher prediction"),
            add_label_bar(error_panel(pre, pred_small, gt640),
                          f"Error  TP=grn FP=red FN=blu   IoU={iou:.3f}"),
        ]
        sep = np.full((panels[0].shape[0], 4, 3), 200, dtype=np.uint8)
        sheet = panels[0]
        for pnl in panels[1:]:
            sheet = np.hstack([sheet, sep, pnl])
        cv2.imwrite(save_compare, sheet)

    return dict(tp=tp, fp=fp, fn=fn, iou=iou,
                gt_px=int(g.sum()), pred_px=int(p.sum()), H=Hc, W=Wc)


def agg(rows):
    """Pooled + mean metrics over a list of per-image result dicts."""
    if not rows:
        return {}
    TP = sum(r["tp"] for r in rows)
    FP = sum(r["fp"] for r in rows)
    FN = sum(r["fn"] for r in rows)
    ious = [r["iou"] for r in rows if not np.isnan(r["iou"])]
    pooled_union = TP + FP + FN
    return {
        "n_images": len(rows),
        "n_iou_defined": len(ious),
        "pooled_iou": (TP / pooled_union) if pooled_union > 0 else float("nan"),
        "mean_iou": float(np.mean(ious)) if ious else float("nan"),
        "precision": (TP / (TP + FP)) if (TP + FP) > 0 else float("nan"),
        "recall": (TP / (TP + FN)) if (TP + FN) > 0 else float("nan"),
        "tp": TP, "fp": FP, "fn": FN,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--model-type", choices=["segformer", "dformer"], default="segformer",
                    help="segformer (default; RGB-only, unchanged identical "
                         "path) or dformer (RGB-D: DFormerv2-L + ham, loads depth "
                         "from data-dir/depth/<basename>.png, runs model(rgb,depth)). "
                         "Crop/resize/IoU/threshold/upsample are identical for both.")
    ap.add_argument("--infer-size", type=parse_size, default=DEFAULT_SIZE, metavar="HxW",
                    help=f"network input size (default {DEFAULT_SIZE}; pred is always "
                         "resized back to the GT crop with nearest)")
    ap.add_argument("--wire-rule", choices=["fg", "class1"], default="fg",
                    help="wire mask = any foreground class (default) or class==1 only "
                         "(for 3-way ckpts where class 2 = objects)")
    ap.add_argument("--ablate-depth", action="store_true",
                    help="P27 diagnostic (dformer only, default OFF): feed a "
                         "spatially-constant depth = dataset normalized mean (zero "
                         "spatial info) to test whether the model uses depth. The "
                         "default arbiter path is unchanged when this is off.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-sheets", action="store_true", help="skip per-image overlay PNGs")
    args = ap.parse_args()
    infer_hw = args.infer_size if isinstance(args.infer_size, tuple) else parse_size(args.infer_size)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    cmp_dir = os.path.join(args.out_dir, "compare")
    if not args.no_sheets:
        os.makedirs(cmp_dir, exist_ok=True)

    print(f"device      : {device}")
    print(f"checkpoint  : {args.ckpt}")
    print(f"data-dir    : {args.data_dir}")
    print(f"infer-size  : {infer_hw[0]}x{infer_hw[1]} (HxW)   wire-rule: {args.wire_rule}")
    print(f"model-type  : {args.model_type}")
    if args.model_type == "dformer":
        model, _loaded_ok = build_dformer(args.ckpt, device)
    else:
        model = load_model_auto(args.ckpt, device)

    basenames = sorted(
        (os.path.splitext(os.path.basename(p))[0]
         for p in glob.glob(os.path.join(args.data_dir, "imgs", "*"))),
        key=lambda b: (b.split("_")[0], int(b.split("_")[1])),
    )
    if not basenames:
        raise SystemExit(f"no images under {args.data_dir}/imgs")
    print(f"images      : {len(basenames)}")

    rows = []
    per_cam = {}
    for i, bn in enumerate(basenames):
        rgb = os.path.join(args.data_dir, "imgs", f"{bn}.jpg")
        mask = os.path.join(args.data_dir, "masks", f"{bn}.jpg")
        depth = os.path.join(args.data_dir, "depth", f"{bn}.png")  # used only by dformer
        save = None if args.no_sheets else os.path.join(cmp_dir, f"compare_{bn}.png")
        r = eval_one(model, device, rgb, mask, save_compare=save,
                     infer_hw=infer_hw, wire_rule=args.wire_rule,
                     model_type=args.model_type, depth_path=depth,
                     ablate_depth=args.ablate_depth)
        cam = bn.split("_")[0]
        r["basename"] = bn
        r["camera"] = cam
        r["precision"] = (r["tp"] / (r["tp"] + r["fp"])) if (r["tp"] + r["fp"]) > 0 else float("nan")
        r["recall"] = (r["tp"] / (r["tp"] + r["fn"])) if (r["tp"] + r["fn"]) > 0 else float("nan")
        rows.append(r)
        per_cam.setdefault(cam, []).append(r)
        print(f"  [{i+1:2d}/{len(basenames)}] {bn:8s}  IoU={r['iou']:.3f}  "
              f"gt={r['gt_px']:7d}  pred={r['pred_px']:7d}")

    overall = agg(rows)
    cams = {cam: agg(rs) for cam, rs in sorted(per_cam.items())}

    # per-image CSV
    csv_path = os.path.join(args.out_dir, "per_image.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["basename", "camera", "iou", "precision", "recall",
                    "tp", "fp", "fn", "gt_px", "pred_px", "H", "W"])
        for r in rows:
            w.writerow([r["basename"], r["camera"], f"{r['iou']:.6f}",
                        f"{r['precision']:.6f}", f"{r['recall']:.6f}",
                        r["tp"], r["fp"], r["fn"], r["gt_px"], r["pred_px"], r["H"], r["W"]])

    summary = {
        "checkpoint": args.ckpt,
        "data_dir": args.data_dir,
        "model_type": args.model_type,
        "infer_size_hw": list(infer_hw),
        "wire_rule": args.wire_rule,
        "metric_note": ("IoU(wire) at cropped human-GT resolution; pred upsampled nearest; "
                        f"center-crop 4:3 then {infer_hw[1]}x{infer_hw[0]} input; "
                        "mask threshold 127"),
        "overall": overall,
        "per_camera": cams,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==================== RESULTS ====================")
    print(f"OVERALL  pooled IoU = {overall['pooled_iou']:.4f}   "
          f"mean IoU = {overall['mean_iou']:.4f}   "
          f"P = {overall['precision']:.4f}   R = {overall['recall']:.4f}   "
          f"(n={overall['n_images']})")
    print("per camera:")
    for cam, m in cams.items():
        print(f"  {cam}: pooled IoU={m['pooled_iou']:.4f}  mean IoU={m['mean_iou']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}  (n={m['n_images']})")
    print(f"\nwrote {csv_path}")
    print(f"wrote {os.path.join(args.out_dir, 'summary.json')}")
    if not args.no_sheets:
        print(f"wrote {len(basenames)} overlay sheets -> {cmp_dir}")


if __name__ == "__main__":
    main()
