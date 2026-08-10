#!/usr/bin/env python
"""
Matched A/B spatial diff for the P38 structured-confuser experiment (CPU-ONLY).

Both arms (CONTROL and TREATMENT/P38) are warm-started from the SAME checkpoint
with a matching recipe+seed; only 2000 training frames differ. At a MATCHED
epoch the shared warm-start relaxation cancels in the per-pixel prediction
DIFFERENCE, leaving the causal effect of the confuser frames. This tool emits a
SPATIAL decomposition (not aggregate metrics) over the precision-limiter cameras
c3/c4, reusing the EXACT prediction path of src/eval_real_wires_valset.py so the
maps reconcile with the live A/B numbers.

For each image and each checkpoint (A, B) we compute pred_full (argmax, wire-rule
fg, upsampled nearest to cropped-GT resolution), then decompose against GT:

  GT-bg (~g):
    fp_a       = (~g)&pa            fp_b = (~g)&pb
    removed_fp = fp_a & ~pb         # A predicted FP, B does NOT -> B SUPPRESSED (GOOD)
    added_fp   = fp_b & ~pa         # B predicted FP, A does NOT -> B ADDED    (BAD)
  GT-wire (g):
    lost_tp    = g & pa & ~pb       # recall LOST by B
    gained_tp  = g & ~pa & pb       # recall GAINED by B

Distance transform on (~g) gives px distance from nearest GT-wire pixel; removed/
added FP are bucketed near(<30)/mid(30-100)/far(>100). FAR FP = genuine
surface-leak (the c4 problem). Connected components of fp_a and fp_b give the
top-5 surface-FP blob areas per image.

CPU ONLY. Set CUDA_VISIBLE_DEVICES="" before launching python. Never uses cuda.
Imports from eval_real_wires_valset; does not modify it or any data.

Usage:
  CUDA_VISIBLE_DEVICES="" python src/diff_matched_ab.py \
      --ckpt-a control.pth --ckpt-b p38.pth \
      --label-a control --label-b p38 --epoch 3 --cams c3,c4 \
      --out-dir results/realism_campaign/p38_matched_control/diff_epoch_3
"""
import os
import sys
import glob
import json
import argparse

# Hard CPU enforcement: blank CUDA before torch is imported anywhere.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_real_wires_valset import (  # noqa: E402
    load_model_auto,
    preprocess_size,
    predict_size,
    crop_mask_to_aspect,
    load_gt_binary,
    IMAGE_H,
    IMAGE_W,
    DEFAULT_DATA,
)

INFER_HW = (IMAGE_H, IMAGE_W)
TARGET_AR = IMAGE_W / IMAGE_H

NEAR_MAX = 30.0   # dist < 30 px  -> near
MID_MAX = 100.0   # 30 <= dist <= 100 -> mid ; dist > 100 -> far


def list_basenames(data_dir, cams):
    """Sorted basenames for the requested cameras, matching the arbiter's order
    (camera, then integer index)."""
    all_bns = [
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(data_dir, "imgs", "*"))
    ]
    sel = [b for b in all_bns if b.split("_")[0] in cams]
    sel.sort(key=lambda b: (b.split("_")[0], int(b.split("_")[1])))
    return sel


def pred_full_for(model, device, rgb_path, gt_crop_shape):
    """Replicate the arbiter prediction recipe EXACTLY and return a bool
    pred_full at the cropped-GT resolution. (gt_crop_shape = (Hc, Wc).)"""
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"cannot read image {rgb_path}")
    pre = preprocess_size(rgb_bgr, INFER_HW)                 # HxW BGR uint8
    pred_small = predict_size(model, pre, device, INFER_HW)  # HxW argmax
    pred_small = (pred_small >= 1).astype(np.uint8)          # wire-rule "fg"
    Hc, Wc = gt_crop_shape
    pred_full = cv2.resize(pred_small, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
    return pred_full.astype(bool), pre


def top5_blobs(mask_bool):
    """(top-5 blob areas desc as ints, total #blobs, total px) for a bool mask."""
    m = mask_bool.astype(np.uint8)
    total_px = int(m.sum())
    if total_px == 0:
        return [], 0, 0
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    # label 0 is background; areas for labels 1..n-1
    areas = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)), reverse=True)
    return areas[:5], len(areas), total_px


def bucket_by_dist(mask_bool, dist):
    """Count mask px in near/mid/far buckets given the GT distance transform."""
    if not mask_bool.any():
        return 0, 0, 0
    d = dist[mask_bool]
    near = int((d < NEAR_MAX).sum())
    far = int((d > MID_MAX).sum())
    mid = int(d.size - near - far)
    return far, mid, near


def diff_one(pa, pb, g, pre_a):
    """Compute the full per-image decomposition. pa, pb, g are bool arrays at
    cropped-GT resolution; pre_a is the preprocessed RGB (for overlay sizing).
    Returns (record_dict, masks_dict) where masks_dict holds the four diff masks
    at cropped-GT resolution for the overlay."""
    not_g = ~g
    fp_a = not_g & pa
    fp_b = not_g & pb
    removed_fp = fp_a & ~pb
    added_fp = fp_b & ~pa
    lost_tp = g & pa & ~pb
    gained_tp = g & ~pa & pb

    # distance from nearest GT-wire pixel, over GT-bg region (px units)
    dist = cv2.distanceTransform(not_g.astype(np.uint8), cv2.DIST_L2, 5)

    rem_far, rem_mid, rem_near = bucket_by_dist(removed_fp, dist)
    add_far, add_mid, add_near = bucket_by_dist(added_fp, dist)

    top5_a, nblob_a, fppx_a = top5_blobs(fp_a)
    top5_b, nblob_b, fppx_b = top5_blobs(fp_b)

    rec = dict(
        gt_px=int(g.sum()),
        fp_a_px=fppx_a,
        fp_b_px=fppx_b,
        removed_fp_px=int(removed_fp.sum()),
        added_fp_px=int(added_fp.sum()),
        net_fp_change=int(added_fp.sum()) - int(removed_fp.sum()),
        lost_tp_px=int(lost_tp.sum()),
        gained_tp_px=int(gained_tp.sum()),
        removed_far=rem_far, removed_mid=rem_mid, removed_near=rem_near,
        added_far=add_far, added_mid=add_mid, added_near=add_near,
        n_fp_blobs_a=nblob_a, n_fp_blobs_b=nblob_b,
        top5_blob_a=top5_a, top5_blob_b=top5_b,
    )
    masks = dict(removed_fp=removed_fp, added_fp=added_fp,
                 lost_tp=lost_tp, gained_tp=gained_tp)
    return rec, masks


# Px-count fields to pool (sum) into aggregates.
SUM_FIELDS = [
    "gt_px", "fp_a_px", "fp_b_px", "removed_fp_px", "added_fp_px",
    "net_fp_change", "lost_tp_px", "gained_tp_px",
    "removed_far", "removed_mid", "removed_near",
    "added_far", "added_mid", "added_near",
    "n_fp_blobs_a", "n_fp_blobs_b",
]


def aggregate(records):
    """Pooled (summed) px-count aggregates over a list of per-image records,
    plus precision-relevant pooled totals."""
    if not records:
        return {"n_images": 0}
    out = {"n_images": len(records)}
    for f in SUM_FIELDS:
        out[f] = int(sum(r[f] for r in records))
    # convenience: net FP change recomputed from pooled (== added - removed)
    out["net_fp_change"] = out["added_fp_px"] - out["removed_fp_px"]
    return out


def make_overlay(pre_bgr, masks, label_a, label_b, bn):
    """hstack [preprocessed RGB | RGB with removed_fp=GREEN, added_fp=RED,
    lost_tp=YELLOW, gained_tp=CYAN]. Diff masks resized nearest to the
    preprocessed RGB size."""
    h, w = pre_bgr.shape[:2]
    over = pre_bgr.copy()
    layer = np.zeros_like(pre_bgr)
    sel = np.zeros((h, w), dtype=bool)
    # BGR colours
    palette = [
        ("removed_fp", (0, 255, 0)),    # GREEN  = B suppressed A's FP (GOOD)
        ("added_fp", (0, 0, 255)),      # RED    = B added a new FP   (BAD)
        ("lost_tp", (0, 255, 255)),     # YELLOW = recall lost by B
        ("gained_tp", (255, 255, 0)),   # CYAN   = recall gained by B
    ]
    for name, colour in palette:
        m = masks[name]
        mr = cv2.resize(m.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        layer[mr] = colour
        sel |= mr
    alpha = 0.6
    over[sel] = (over[sel] * (1 - alpha) + layer[sel] * alpha).astype(np.uint8)

    def label_bar(img, text):
        bar = np.full((26, img.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(bar, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    left = label_bar(pre_bgr, f"{bn}  input RGB")
    right = label_bar(
        over,
        f"diff {label_b}-{label_a}: GRN=removedFP RED=addedFP YEL=lostTP CYN=gainedTP",
    )
    sep = np.full((left.shape[0], 4, 3), 200, dtype=np.uint8)
    return np.hstack([left, sep, right])


def render_md(epoch, label_a, label_b, cams, per_cam_records, agg_blocks):
    """Human-readable per-camera + per-image table."""
    lines = []
    lines.append(f"# Matched A/B diff — epoch {epoch}  (B={label_b} minus A={label_a})")
    lines.append("")
    lines.append(f"Cameras: {','.join(cams)}.  Wire-rule fg, cropped-GT resolution, "
                 "arbiter prediction path.")
    lines.append("removed_fp = A's FP suppressed by B (GOOD). added_fp = new FP from B (BAD).")
    lines.append("net_fp_change = added - removed (negative = B is cleaner).")
    lines.append("")
    # Aggregate table
    lines.append("## Aggregates (pooled px sums)")
    lines.append("")
    hdr = ("| scope | n | gt_px | fp_a | fp_b | removed | added | net | "
           "lost_tp | gained_tp | rem far/mid/near | add far/mid/near |")
    sep = "|" + "---|" * 12
    lines.append(hdr)
    lines.append(sep)
    for scope in list(cams) + (["c3c4"] if len(cams) > 1 else []):
        a = agg_blocks.get(scope)
        if not a or a.get("n_images", 0) == 0:
            continue
        lines.append(
            f"| {scope} | {a['n_images']} | {a['gt_px']} | {a['fp_a_px']} | "
            f"{a['fp_b_px']} | {a['removed_fp_px']} | {a['added_fp_px']} | "
            f"{a['net_fp_change']} | {a['lost_tp_px']} | {a['gained_tp_px']} | "
            f"{a['removed_far']}/{a['removed_mid']}/{a['removed_near']} | "
            f"{a['added_far']}/{a['added_mid']}/{a['added_near']} |"
        )
    lines.append("")
    # Per-image table
    lines.append("## Per-image")
    lines.append("")
    hdr2 = ("| bn | cam | gt_px | fp_a | fp_b | removed | added | net | "
            "lost_tp | gained_tp | rem f/m/n | add f/m/n | top5_a | top5_b |")
    sep2 = "|" + "---|" * 14
    lines.append(hdr2)
    lines.append(sep2)
    for cam in cams:
        for r in per_cam_records.get(cam, []):
            lines.append(
                f"| {r['bn']} | {r['camera']} | {r['gt_px']} | {r['fp_a_px']} | "
                f"{r['fp_b_px']} | {r['removed_fp_px']} | {r['added_fp_px']} | "
                f"{r['net_fp_change']} | {r['lost_tp_px']} | {r['gained_tp_px']} | "
                f"{r['removed_far']}/{r['removed_mid']}/{r['removed_near']} | "
                f"{r['added_far']}/{r['added_mid']}/{r['added_near']} | "
                f"{r['top5_blob_a']} | {r['top5_blob_b']} |"
            )
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-a", required=True, help="CONTROL checkpoint (A)")
    ap.add_argument("--ckpt-b", required=True, help="TREATMENT/P38 checkpoint (B)")
    ap.add_argument("--label-a", default="control")
    ap.add_argument("--label-b", default="p38")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--cams", default="c3,c4",
                    help="comma list of cameras to diff (default c3,c4)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-overlays", action="store_true",
                    help="skip writing overlay sheets")
    args = ap.parse_args()

    # Hard CPU guard.
    if torch.cuda.is_available():
        # CUDA_VISIBLE_DEVICES should have hidden it; never proceed on GPU.
        raise SystemExit("REFUSING TO RUN: CUDA is visible. Export "
                         "CUDA_VISIBLE_DEVICES=\"\" before launching.")
    device = torch.device("cpu")

    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    out_dir = args.out_dir
    ov_dir = os.path.join(out_dir, "overlays")
    os.makedirs(out_dir, exist_ok=True)
    if not args.no_overlays:
        os.makedirs(ov_dir, exist_ok=True)

    basenames = list_basenames(args.data_dir, cams)
    if not basenames:
        raise SystemExit(f"no images for cams {cams} under {args.data_dir}/imgs")

    print(f"device   : {device}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')!r})")
    print(f"A ({args.label_a}) : {args.ckpt_a}")
    print(f"B ({args.label_b}) : {args.ckpt_b}")
    print(f"epoch    : {args.epoch}   cams: {cams}   images: {len(basenames)}")

    # Load A, predict all; load B, predict all. (Avoid holding both B5 models in
    # RAM at once is not necessary on CPU, but we still load sequentially.)
    print(f"\n[load A] {args.ckpt_a}")
    model_a = load_model_auto(args.ckpt_a, device)
    preds_a = {}
    gt_crops = {}
    pre_imgs = {}
    for i, bn in enumerate(basenames):
        rgb = os.path.join(args.data_dir, "imgs", f"{bn}.jpg")
        mask = os.path.join(args.data_dir, "masks", f"{bn}.jpg")
        gt_bin = load_gt_binary(mask)
        gt_crop = crop_mask_to_aspect(gt_bin, TARGET_AR).astype(bool)
        gt_crops[bn] = gt_crop
        pa, pre = pred_full_for(model_a, device, rgb, gt_crop.shape)
        preds_a[bn] = pa
        pre_imgs[bn] = pre
        print(f"  A [{i+1:2d}/{len(basenames)}] {bn}")
    del model_a

    print(f"\n[load B] {args.ckpt_b}")
    model_b = load_model_auto(args.ckpt_b, device)
    preds_b = {}
    for i, bn in enumerate(basenames):
        rgb = os.path.join(args.data_dir, "imgs", f"{bn}.jpg")
        pb, _pre = pred_full_for(model_b, device, rgb, gt_crops[bn].shape)
        preds_b[bn] = pb
        print(f"  B [{i+1:2d}/{len(basenames)}] {bn}")
    del model_b

    # Decompose.
    per_image = []
    per_cam_records = {c: [] for c in cams}
    for bn in basenames:
        cam = bn.split("_")[0]
        rec, masks = diff_one(preds_a[bn], preds_b[bn], gt_crops[bn], pre_imgs[bn])
        rec["camera"] = cam
        rec["bn"] = bn
        per_image.append(rec)
        per_cam_records[cam].append(rec)
        if not args.no_overlays:
            sheet = make_overlay(pre_imgs[bn], masks, args.label_a, args.label_b, bn)
            cv2.imwrite(os.path.join(ov_dir, f"diff_{bn}.png"), sheet)

    # Aggregates.
    agg_blocks = {}
    for c in cams:
        agg_blocks[c] = aggregate(per_cam_records[c])
    if len(cams) > 1:
        agg_blocks["c3c4"] = aggregate(per_image)

    payload = {
        "epoch": args.epoch,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "ckpt_a": os.path.abspath(args.ckpt_a),
        "ckpt_b": os.path.abspath(args.ckpt_b),
        "cams": cams,
        "n_images": len(basenames),
        "per_image": per_image,
        "agg": agg_blocks,
    }
    json_path = os.path.join(out_dir, f"diff_epoch_{args.epoch}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    md = render_md(args.epoch, args.label_a, args.label_b, cams,
                   per_cam_records, agg_blocks)
    md_path = os.path.join(out_dir, f"diff_epoch_{args.epoch}.md")
    with open(md_path, "w") as f:
        f.write(md)

    # Console summary.
    print("\n==================== DIFF SUMMARY ====================")
    for scope in list(cams) + (["c3c4"] if len(cams) > 1 else []):
        a = agg_blocks.get(scope)
        if not a or a.get("n_images", 0) == 0:
            continue
        print(f"  {scope:5s} n={a['n_images']:2d}  fp_a={a['fp_a_px']:7d} "
              f"fp_b={a['fp_b_px']:7d}  removed={a['removed_fp_px']:7d} "
              f"added={a['added_fp_px']:7d}  net={a['net_fp_change']:+8d}  "
              f"lost_tp={a['lost_tp_px']:6d} gained_tp={a['gained_tp_px']:6d}  "
              f"rem(f/m/n)={a['removed_far']}/{a['removed_mid']}/{a['removed_near']} "
              f"add(f/m/n)={a['added_far']}/{a['added_mid']}/{a['added_near']}")
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")
    if not args.no_overlays:
        print(f"wrote {len(basenames)} overlays -> {ov_dir}")


if __name__ == "__main__":
    main()
