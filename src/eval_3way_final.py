"""Full-val evaluation of a 3-way {bg, wire, connector} checkpoint.

Reports the pooled metrics AND the per-held-out-set breakdown. The per-set view
is not decoration: set 035 carries 65.5 % of all connector pixels in this val
split, so a pooled connector gain could in principle come from one harness
alone. A lever is only credible if it moves 032, 034 AND 035 in the same
direction.

Optionally evaluates with horizontal-flip TTA (a deterministic inference-time
procedure, reported separately from the plain argmax number so the two are
never conflated).

Usage:
  python src/eval_3way_final.py --ckpt <path> [--ckpt <path> ...] \
      --data-dir data/dformer_dataset_3way_connscale3 --out <json> [--tta]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_rgb_only_sota import IMAGE_H, IMAGE_W, RGB_MEAN, RGB_STD  # noqa: E402
from gen_rgb_only_sota_gifs import load_model  # noqa: E402

CLASSES = {0: "bg", 1: "wire", 2: "connector"}


def metrics_from_cm(cm):
    out = {}
    ious = []
    for c, name in CLASSES.items():
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum()) - tp
        fp = int(cm[:, c].sum()) - tp
        iou = tp / max(tp + fp + fn, 1)
        out[f"iou_{name}"] = iou
        out[f"precision_{name}"] = tp / max(tp + fp, 1)
        out[f"recall_{name}"] = tp / max(tp + fn, 1)
        ious.append(iou)
    out["miou"] = float(np.mean(ious))
    out["pixel_acc"] = float(np.trace(cm) / max(cm.sum(), 1))
    return out


@torch.no_grad()
def predict_batch(model, rgb_bgr, device, tta=False):
    rgb = rgb_bgr[:, :, :, ::-1].copy()
    t = torch.from_numpy(rgb.transpose(0, 3, 1, 2)).to(device).float() / 255.0
    t = (t - RGB_MEAN.to(device)) / RGB_STD.to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(t).float()
        if tta:
            flipped = model(torch.flip(t, dims=[3])).float()
            logits = logits + torch.flip(flipped, dims=[3])
    return logits.argmax(1).cpu().numpy().astype(np.uint8)


def evaluate_train_sample(model, data_dir, device, batch=8, n=480):
    """Metrics on a fixed, evenly-spaced sample of TRAIN images.

    The train-minus-val IoU gap is the standard quantitative overfitting
    measure: a lever that buys val IoU by memorising the train harnesses shows
    a widening gap, one that genuinely generalises does not.
    """
    cache = os.path.join(data_dir, "cache")
    rgb = np.load(os.path.join(cache, "train_rgb.npy"), mmap_mode="r")
    lab = np.load(os.path.join(cache, "train_label.npy"), mmap_mode="r")
    idx = np.linspace(0, len(rgb) - 1, min(n, len(rgb))).astype(int)
    cm = np.zeros((3, 3), dtype=np.int64)
    for i0 in range(0, len(idx), batch):
        sel = idx[i0:i0 + batch]
        pr = predict_batch(model, np.asarray(rgb[sel]), device)
        gt = np.asarray(lab[sel]).astype(np.int64)
        for b in range(len(sel)):
            j = gt[b].ravel() * 3 + pr[b].ravel().astype(np.int64)
            cm += np.bincount(j, minlength=9).reshape(3, 3)
    m = metrics_from_cm(cm)
    m["n_images"] = int(len(idx))
    return m


def evaluate(ckpt, data_dir, device, batch=8, tta=False, train_sample=0):
    model = load_model(ckpt, device)
    cache = os.path.join(data_dir, "cache")
    rgb = np.load(os.path.join(cache, "val_rgb.npy"), mmap_mode="r")
    lab = np.load(os.path.join(cache, "val_label.npy"), mmap_mode="r")
    names = [l.strip() for l in open(os.path.join(data_dir, "test.txt")) if l.strip()]
    assert len(names) == len(rgb), f"{len(names)} names vs {len(rgb)} imgs"

    cm_all = np.zeros((3, 3), dtype=np.int64)
    cm_set = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))

    for i0 in range(0, len(rgb), batch):
        i1 = min(i0 + batch, len(rgb))
        pr = predict_batch(model, np.asarray(rgb[i0:i1]), device, tta=tta)
        gt = np.asarray(lab[i0:i1]).astype(np.int64)
        for b in range(i1 - i0):
            idx = gt[b].ravel() * 3 + pr[b].ravel().astype(np.int64)
            cm = np.bincount(idx, minlength=9).reshape(3, 3)
            cm_all += cm
            cm_set[names[i0 + b].split("/")[1].split("_")[0]] += cm

    res = {"pooled": metrics_from_cm(cm_all),
           "per_set": {s: metrics_from_cm(c) for s, c in sorted(cm_set.items())}}
    res["pooled"]["confusion"] = cm_all.tolist()
    if train_sample:
        tr = evaluate_train_sample(model, data_dir, device, batch, train_sample)
        res["train_sample"] = tr
        res["generalization_gap"] = {
            k: tr[k] - res["pooled"][k]
            for k in ("iou_connector", "iou_wire", "miou")
        }
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="repeatable; use name=path to label an arm")
    ap.add_argument("--data-dir", default="data/dformer_dataset_3way_connscale3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--train-sample", type=int, default=0,
                    help="also score N evenly-spaced TRAIN images to report the "
                         "train-minus-val generalization gap (0 = skip)")
    args = ap.parse_args()

    device = "cuda"
    report = {}
    for spec in args.ckpt:
        name, path = spec.split("=", 1) if "=" in spec else (os.path.basename(
            os.path.dirname(spec)), spec)
        print(f"\n[eval] {name}  <- {path}", flush=True)
        report[name] = {"ckpt": path,
                        "argmax": evaluate(path, args.data_dir, device, args.batch,
                                           train_sample=args.train_sample)}
        if args.tta:
            report[name]["hflip_tta"] = evaluate(path, args.data_dir, device,
                                                 args.batch, tta=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # ---- tables -------------------------------------------------------------
    print(f"\n{'='*92}\nPOOLED (full 1080-image val)\n{'='*92}")
    print(f"{'arm':<24}{'IoU(con)':>11}{'IoU(wire)':>11}{'IoU(bg)':>10}"
          f"{'mIoU':>10}{'P(con)':>9}{'R(con)':>9}")
    for name, r in report.items():
        m = r["argmax"]["pooled"]
        print(f"{name:<24}{m['iou_connector']:>11.4f}{m['iou_wire']:>11.4f}"
              f"{m['iou_bg']:>10.4f}{m['miou']:>10.4f}"
              f"{m['precision_connector']:>9.3f}{m['recall_connector']:>9.3f}")

    print(f"\n{'='*92}\nPER HELD-OUT SET — a credible lever moves ALL THREE\n{'='*92}")
    sets = sorted(next(iter(report.values()))["argmax"]["per_set"])
    print(f"{'arm':<24}" + "".join(
        f"{'con_'+s:>12}" for s in sets) + "".join(f"{'mIoU_'+s:>12}" for s in sets))
    for name, r in report.items():
        ps = r["argmax"]["per_set"]
        print(f"{name:<24}" + "".join(f"{ps[s]['iou_connector']:>12.4f}" for s in sets)
              + "".join(f"{ps[s]['miou']:>12.4f}" for s in sets))

    if args.tta:
        print(f"\n{'='*92}\nHFLIP TTA (inference-time only; reported separately)"
              f"\n{'='*92}")
        print(f"{'arm':<24}{'IoU(con)':>11}{'IoU(wire)':>11}{'mIoU':>10}")
        for name, r in report.items():
            m = r["hflip_tta"]["pooled"]
            print(f"{name:<24}{m['iou_connector']:>11.4f}{m['iou_wire']:>11.4f}"
                  f"{m['miou']:>10.4f}")

    if args.train_sample:
        print(f"\n{'='*92}\nOVERFITTING CHECK — train-minus-val gap "
              f"(smaller/stable = generalising, not memorising)\n{'='*92}")
        print(f"{'arm':<24}{'train con':>11}{'val con':>10}{'gap con':>10}"
              f"{'train mIoU':>12}{'val mIoU':>10}{'gap mIoU':>10}")
        for name, r in report.items():
            t = r["argmax"]["train_sample"]
            v = r["argmax"]["pooled"]
            print(f"{name:<24}{t['iou_connector']:>11.4f}{v['iou_connector']:>10.4f}"
                  f"{t['iou_connector']-v['iou_connector']:>+10.4f}"
                  f"{t['miou']:>12.4f}{v['miou']:>10.4f}"
                  f"{t['miou']-v['miou']:>+10.4f}")

    names = list(report)
    if len(names) > 1:
        base = report[names[0]]["argmax"]
        print(f"\n{'='*92}\nDELTA vs {names[0]}\n{'='*92}")
        print(f"{'arm':<24}{'d con':>10}{'d wire':>10}{'d mIoU':>10}   per-set d con")
        for name in names[1:]:
            m = report[name]["argmax"]
            d = [f"{m['per_set'][s]['iou_connector']-base['per_set'][s]['iou_connector']:+.4f}"
                 for s in sets]
            print(f"{name:<24}"
                  f"{m['pooled']['iou_connector']-base['pooled']['iou_connector']:>+10.4f}"
                  f"{m['pooled']['iou_wire']-base['pooled']['iou_wire']:>+10.4f}"
                  f"{m['pooled']['miou']-base['pooled']['miou']:>+10.4f}   "
                  + "  ".join(d))
    print(f"\n[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
