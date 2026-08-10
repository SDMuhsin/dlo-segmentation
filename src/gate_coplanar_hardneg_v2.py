#!/usr/bin/env python
"""P29-v2 hard-negative HARDNESS GATE + final dataset assembly (PHASE 2).

Scores every candidate from src/gen_coplanar_hardneg_v2.py with SegFormer-B5 epoch_15 (the
current best, results/realism_campaign/p28_realtextured/segformer_b5_cotrain/epoch_15.pth),
using the EXACT eval-script preprocessing (center-crop 4:3 -> 480x640 BGR, BGR->RGB, /255,
ImageNet-norm) and the network's OWN wire-decision axis: per-pixel wire-logit = (logit_wire -
logit_bg), computed at full 480x640 resolution exactly like
results/realism_campaign/p29_feature_separability/extract_features.py.

GATE (model-confusability, valset-FREE): a candidate is KEPT iff the model fires
(wire-logit > 0) on >= FRAC_THRESH of its SEAM-region pixels (the vicinity of the drawn dark
continuous lines, written by the generator to Seam/<base>.png). The hardness-calibration probe
showed the model mis-fires "wire" specifically on the dark continuous line ON a busy surround,
so the seam-region fired-fraction is the operative confusability measure (a thin line is a tiny
% of the whole frame, so whole-frame fired-% saturates near 0 even for genuinely hard frames —
exactly v1's failure mode of measuring the wrong denominator). We auto-tune FRAC_THRESH over a
grid (>=0.30 floor per spec) to retain ~target_n (default 760) frames, and report the chosen
threshold + retention. We ALSO report whole-frame fired-% and the kept FIRED-pixel wire-logit
median (the direct apples-to-apples vs v1's NEG_seam −14.5 / real FP_seam +1.66).

After selection we write the final dformer dataset under --out-dir (RGB/Label/Depth + train.txt
/test.txt, split BY DONOR TEXTURE so val/train share no donor), mirroring v1's format & meta.
We also report the kept set's median wire-logit (vs v1's −17.67 and real-FP +3.2) and median
local surround texstd (vs target 25-35 / real-FP 28.9), and (SANITY ONLY, per guardrail) the
mean fused-feature z-distance of the kept FIRED pixels to the real FP_seam / TN_surface / TP_wire
centroids from p29_feature_separability/features.npz — selection is by the model gate, NOT by
minimizing valset distance.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/workspace/kiat_crefle"
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(ROOT, "data", "hf_cache"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from infer_video_rgb_only import preprocess  # noqa: E402  (center-crop 4:3 -> 480x640)
from train_rgb_only_sota import (  # noqa: E402
    SegFormerSegmenter, IMAGE_H, IMAGE_W, RGB_MEAN, RGB_STD, BACKBONE_DEFAULT,
)

CKPT = os.path.join(ROOT, "results/realism_campaign/p28_realtextured/segformer_b5_cotrain/epoch_15.pth")
FEAT_NPZ = os.path.join(ROOT, "results/realism_campaign/p29_feature_separability/features.npz")
WIRE_IDX = 1


def load_model(device):
    state = torch.load(CKPT, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model = SegFormerSegmenter(backbone_name=BACKBONE_DEFAULT, num_classes=2, criterion=None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model.eval().to(device)


def local_texstd(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (9, 9), normalize=True)
    sq = cv2.boxFilter(gray * gray, -1, (9, 9), normalize=True)
    return np.sqrt(np.clip(sq - mean * mean, 0, None))


@torch.no_grad()
def score_frame(model, cls_feats, w_dir_t, b_dir, pre_bgr, device):
    """Return (wlogit_full 480x640, fused 768xFHxFW). pre_bgr already 480x640 BGR uint8."""
    rgb = pre_bgr[:, :, ::-1].copy()
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device, torch.float32) / 255.0
    x = (x - RGB_MEAN.to(device)) / RGB_STD.to(device)
    out = model.model(pixel_values=x)
    logits_small = out.logits                                  # (1,2,FH,FW)
    logits_full = F.interpolate(logits_small, size=(IMAGE_H, IMAGE_W),
                                mode="bilinear", align_corners=False)
    wlogit = (logits_full[0, WIRE_IDX] - logits_full[0, 0]).cpu().numpy()
    fused = cls_feats.pop("fused")[0]                          # (768,FH,FW) on device
    return wlogit, fused


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand-dir",
                    default="results/realism_campaign/p29v2_coplanar_hardneg/candidates")
    ap.add_argument("--out-dir", default="data/dformer_dataset_coplanar_hardneg_v2")
    ap.add_argument("--report-dir", default="results/realism_campaign/p29v2_coplanar_hardneg")
    ap.add_argument("--target-n", type=int, default=760,
                    help="aim to retain ~this many (700-800) frames; threshold auto-tuned")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    model = load_model(device)

    # hook the fused classifier-input feature (768ch) exactly as the separability extractor.
    cls_feats = {}
    dh = model.model.decode_head

    def cls_pre(mod, inp):
        cls_feats["fused"] = inp[0].detach()
    dh.classifier.register_forward_pre_hook(cls_pre)

    # the network's own wire-vs-bg axis in the 768-d fused space (== eval wire-logit direction)
    Wc = dh.classifier.weight.detach().squeeze(-1).squeeze(-1).cpu().numpy()  # (2,768)
    bc = dh.classifier.bias.detach().cpu().numpy()
    w_dir = (Wc[WIRE_IDX] - Wc[0]).astype(np.float64)
    b_dir = float(bc[WIRE_IDX] - bc[0])
    w_dir_t = torch.from_numpy(w_dir).to(device)

    cand_rgb = os.path.join(args.cand_dir, "RGB")
    cand_seam = os.path.join(args.cand_dir, "Seam")
    # Derive candidate bases by scanning the RGB dir directly (crash-resilient: does NOT
    # depend on candidates_meta.json, which the generator only writes at the very end).
    # Donor names (cosmetic, for the val-split) come from the meta if present, else "unknown".
    rec_by_base = {}
    meta_path = os.path.join(args.cand_dir, "candidates_meta.json")
    if os.path.isfile(meta_path):
        try:
            rec_by_base = {r["base"]: r for r in json.load(open(meta_path)).get("records", [])}
        except Exception:
            rec_by_base = {}
    bases = sorted(os.path.splitext(f)[0] for f in os.listdir(cand_rgb) if f.endswith(".png"))
    print(f"candidates: {len(bases)}  (meta records: {len(rec_by_base)})", flush=True)

    # ---- score every candidate ----
    cand = []   # dicts: base, donor, frac_fire_seam, frac_fire_frame, med_wlogit_all, surround
    per_base_fired = {}      # base -> (fused_fired Nx768, wlogit_fired, surround_fired)
    for i, base in enumerate(bases):
        pre = cv2.imread(os.path.join(cand_rgb, base + ".png"), cv2.IMREAD_COLOR)
        # candidates are already 480x640; preprocess() is identity-geometry on 4:3 480x640
        # but run it anyway so the path is unchanged from eval.
        pre = preprocess(pre)
        seam_m = cv2.imread(os.path.join(cand_seam, base + ".png"), cv2.IMREAD_GRAYSCALE)
        seam_m = (seam_m > 127) if seam_m is not None else np.zeros(pre.shape[:2], bool)
        wlogit, fused = score_frame(model, cls_feats, w_dir_t, b_dir, pre, device)
        ts = local_texstd(pre)
        fire = wlogit > 0.0
        frac_fire_frame = float(fire.mean())
        # GATE measure = fraction of SEAM-region pixels that fire
        frac_fire_seam = float((fire & seam_m).sum() / max(seam_m.sum(), 1))
        # fired-pixel fused features (map full-res fired coords -> feature grid cell y//4,x//4)
        ys, xs = np.where(fire)
        if len(ys) > 0:
            FH, FW = fused.shape[1], fused.shape[2]
            fy = np.clip(ys // 4, 0, FH - 1)
            fx = np.clip(xs // 4, 0, FW - 1)
            # subsample to cap memory
            if len(ys) > 1500:
                sel = np.random.default_rng(0).choice(len(ys), 1500, replace=False)
                fy, fx, yy, xx = fy[sel], fx[sel], ys[sel], xs[sel]
            else:
                yy, xx = ys, xs
            fused_fired = fused[:, fy, fx].T.float().cpu().numpy()
            surround_fired = ts[yy, xx]
            wlogit_fired = wlogit[yy, xx]
        else:
            fused_fired = np.zeros((0, fused.shape[0]), np.float32)
            surround_fired = np.zeros((0,), np.float32)
            wlogit_fired = np.zeros((0,), np.float32)
        per_base_fired[base] = (fused_fired, wlogit_fired, surround_fired)
        cand.append({
            "base": base,
            "donor": rec_by_base.get(base, {}).get("donor", "unknown"),
            "frac_fire": frac_fire_seam,            # GATE measure = seam-region fired fraction
            "frac_fire_frame": frac_fire_frame,
            "med_wlogit_all": float(np.median(wlogit)),
            "surround_texstd_median": float(np.median(ts)),
        })
        if (i + 1) % 200 == 0:
            print(f"  scored {i+1}/{len(bases)}  (last seam-fire={frac_fire_seam:.3f})", flush=True)

    frac = np.array([c["frac_fire"] for c in cand])

    # ---- auto-tune FRAC_THRESH on a grid (>=0.30 floor per spec) to retain ~target_n ----
    grid = np.round(np.arange(0.30, 0.901, 0.01), 2)
    best_thr, best_keep = None, None
    for thr in grid:
        keep_n = int((frac >= thr).sum())
        if best_keep is None or abs(keep_n - args.target_n) < abs(best_keep - args.target_n):
            best_keep, best_thr = keep_n, float(thr)
        if keep_n <= args.target_n:   # grid ascends -> first time we drop at/below target
            # prefer the threshold whose keep is closest to target
            pass
    # choose threshold giving keep closest to target while >= 0.30
    keeps = [(float(t), int((frac >= t).sum())) for t in grid]
    keeps_sorted = sorted(keeps, key=lambda tk: abs(tk[1] - args.target_n))
    chosen_thr, chosen_keep = keeps_sorted[0]
    # guardrail: never go below 0.30; if even 0.30 retains < ~700, keep 0.30 and report
    if chosen_keep < 700 and (frac >= 0.30).sum() >= chosen_keep:
        chosen_thr = 0.30
        chosen_keep = int((frac >= 0.30).sum())
    print(f"\nGATE TUNE: threshold grid retention (frac_fire >= thr):", flush=True)
    for t, k in keeps[::5]:
        print(f"    thr={t:.2f}  keep={k}", flush=True)
    print(f"CHOSEN frac_fire threshold = {chosen_thr:.2f}  -> keep {chosen_keep} "
          f"(target {args.target_n})", flush=True)

    kept = [c for c in cand if c["frac_fire"] >= chosen_thr]
    kept_bases = set(c["base"] for c in kept)
    retention = len(kept) / len(cand) if cand else 0.0

    # ---- assemble final dataset (copy RGB/Label/Depth, write train/test split by donor) ----
    rgb_out = os.path.join(args.out_dir, "RGB")
    depth_out = os.path.join(args.out_dir, "Depth")
    label_out = os.path.join(args.out_dir, "Label")
    for d in (rgb_out, depth_out, label_out):
        os.makedirs(d, exist_ok=True)
    # split by donor texture (no near-dup leak) when donor info is available; else fall back
    # to a deterministic per-frame random split (donor unknown after a resumed/crash-recovered
    # generation). Each frame is an independent procedural render (no donor near-dup risk within
    # a single donor since scale/rotation/photometric/seam are all randomized), so a per-frame
    # split is acceptable here.
    rng = np.random.default_rng(args.seed)
    donors = sorted(set(c["donor"] for c in kept))
    use_donor_split = len(donors) >= 5 and "unknown" not in donors
    if use_donor_split:
        k = max(1, int(round(len(donors) * args.val_frac)))
        val_donors = set(np.array(donors)[rng.permutation(len(donors))[:k]].tolist())
        is_val = lambda c: c["donor"] in val_donors
    else:
        kept_sorted0 = sorted(kept, key=lambda c: c["base"])
        nval = max(1, int(round(len(kept_sorted0) * args.val_frac)))
        val_idx = set(rng.permutation(len(kept_sorted0))[:nval].tolist())
        base2idx = {c["base"]: i for i, c in enumerate(kept_sorted0)}
        val_donors = set()
        is_val = lambda c: base2idx[c["base"]] in val_idx

    zero_depth = np.zeros((IMAGE_H, IMAGE_W), np.uint16)
    zero_label = np.zeros((IMAGE_H, IMAGE_W), np.uint8)
    train_lines, val_lines = [], []
    # re-number the kept frames 000.. so the dataset is contiguous (new basenames)
    for newid, c in enumerate(sorted(kept, key=lambda c: c["base"])):
        old = c["base"]
        nb = f"{newid:03d}_{0:04d}_00_hn2"
        src = cv2.imread(os.path.join(cand_rgb, old + ".png"), cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(rgb_out, nb + ".png"), src)
        cv2.imwrite(os.path.join(depth_out, nb + ".png"), zero_depth)
        cv2.imwrite(os.path.join(label_out, nb + ".png"), zero_label)
        line = f"RGB/{nb}.png"
        (val_lines if is_val(c) else train_lines).append(line)
        c["new_base"] = nb
    with open(os.path.join(args.out_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_lines) + ("\n" if train_lines else ""))
    with open(os.path.join(args.out_dir, "test.txt"), "w") as f:
        f.write("\n".join(val_lines) + ("\n" if val_lines else ""))

    # ---- kept-set statistics for the acceptance report ----
    kept_med_wlogit_all = float(np.median([c["med_wlogit_all"] for c in kept]))
    kept_med_surround = float(np.median([c["surround_texstd_median"] for c in kept]))
    # pooled FIRED-pixel wire-logit (the pixels the model mis-fires on) — the apples-to-apples
    # vs v1's NEG_seam −14.5 and real FP_seam +1.66
    fired_wlogit = np.concatenate([per_base_fired[c["base"]][1] for c in kept
                                   if per_base_fired[c["base"]][1].size > 0])
    fired_surround = np.concatenate([per_base_fired[c["base"]][2] for c in kept
                                     if per_base_fired[c["base"]][2].size > 0])
    fired_fused = np.concatenate([per_base_fired[c["base"]][0] for c in kept
                                  if per_base_fired[c["base"]][0].shape[0] > 0], 0)

    # ---- SANITY (guardrail): z-distance of kept FIRED features to real centroids ----
    sanity = {}
    if os.path.isfile(FEAT_NPZ):
        d = np.load(FEAT_NPZ)
        fp = d["FP_seam__fused"]; tn = d["TN_surface__fused"]; tp = d["TP_wire__fused"]
        stack = np.concatenate([fp, tn, tp], 0)
        mu, sd = stack.mean(0), stack.std(0) + 1e-6
        def zc(a):
            return (a - mu) / sd
        c_fp, c_tn, c_tp = zc(fp.mean(0)), zc(tn.mean(0)), zc(tp.mean(0))
        if fired_fused.shape[0] > 0:
            kf = zc(fired_fused)
            d_fp = float(np.linalg.norm(kf - c_fp, axis=1).mean())
            d_tn = float(np.linalg.norm(kf - c_tn, axis=1).mean())
            d_tp = float(np.linalg.norm(kf - c_tp, axis=1).mean())
            nearest = min([("FP_seam", d_fp), ("TN_surface", d_tn), ("TP_wire", d_tp)],
                          key=lambda x: x[1])[0]
            sanity = {"kept_fired_to_FP_seam": d_fp, "kept_fired_to_TN_surface": d_tn,
                      "kept_fired_to_TP_wire": d_tp, "nearest": nearest,
                      "real_FP_seam_texstd_median": float(np.median(d["FP_seam__texstd"])),
                      "real_FP_seam_wlogit_median": float(np.median(d["FP_seam__wlogit"]))}

    report = {
        "out_dir": args.out_dir,
        "n_candidates": len(cand),
        "gate_frac_fire_threshold": chosen_thr,
        "n_accepted": len(kept),
        "retention_rate": retention,
        "n_train": len(train_lines),
        "n_val": len(val_lines),
        "val_donors": sorted(val_donors),
        "kept_median_wlogit_whole_frame": kept_med_wlogit_all,
        "kept_FIRED_pixel_wlogit_median": float(np.median(fired_wlogit)) if fired_wlogit.size else None,
        "kept_FIRED_pixel_wlogit_mean": float(np.mean(fired_wlogit)) if fired_wlogit.size else None,
        "kept_median_surround_texstd": kept_med_surround,
        "kept_FIRED_surround_texstd_median": float(np.median(fired_surround)) if fired_surround.size else None,
        "v1_reference": {"neg_all_median_wlogit": -17.67, "neg_seam_median_wlogit": -14.53,
                         "neg_seam_surround_texstd": 14.55},
        "real_FP_reference": {"wlogit_median": 1.66, "wlogit_mean": 3.0, "surround_texstd": 28.86},
        "frac_fire_distribution_SEAM": {
            "min": float(frac.min()), "median": float(np.median(frac)),
            "mean": float(frac.mean()), "max": float(frac.max()),
            "p25": float(np.percentile(frac, 25)), "p75": float(np.percentile(frac, 75)),
        },
        "kept_whole_frame_fire_pct_median": float(np.median(
            [c["frac_fire_frame"] for c in kept])) if kept else None,
        "gate_measure": "fraction of SEAM-region pixels (drawn dark-line vicinity) with "
                        "wire-logit>0 (model fires); seam region from generator Seam/ mask",
        "threshold_grid_retention": {f"{t:.2f}": int(k) for t, k in keeps},
        "sanity_centroid_distances": sanity,
        "ckpt": CKPT,
        "note": "Gate = model-confusability (wire-logit>0 fraction); valset used ONLY as a "
                "sanity measurement anchor, NOT for selection.",
    }
    os.makedirs(args.report_dir, exist_ok=True)
    with open(os.path.join(args.report_dir, "gate_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    # meta in the dataset dir (mirror v1 gen_meta.json)
    with open(os.path.join(args.out_dir, "gen_meta.json"), "w") as f:
        json.dump({"n_total": len(kept), "n_train": len(train_lines), "n_val": len(val_lines),
                   "val_donors": sorted(val_donors),
                   "gate_frac_fire_threshold": chosen_thr, "retention_rate": retention,
                   "all_labels": "all-zero (wire-free, background only)",
                   "report": os.path.join(args.report_dir, "gate_report.json")}, f, indent=2)

    print("\n=== P29-v2 HARDNESS GATE + ASSEMBLY DONE ===")
    print(f"out-dir: {args.out_dir}")
    print(f"accepted {len(kept)}/{len(cand)}  retention {100*retention:.1f}%  "
          f"(frac_fire >= {chosen_thr:.2f})")
    print(f"train {len(train_lines)} / val {len(val_lines)}")
    print(f"kept median whole-frame wire-logit: {kept_med_wlogit_all:.2f}  (v1 −17.67, real-FP +1.66)")
    if fired_wlogit.size:
        print(f"kept FIRED-pixel wire-logit median: {np.median(fired_wlogit):.2f}")
    print(f"kept median local surround texstd: {kept_med_surround:.2f}  (target 25-35, real-FP 28.9)")
    if sanity:
        print(f"SANITY nearest real centroid of kept fired px: {sanity['nearest']} "
              f"(FP {sanity['kept_fired_to_FP_seam']:.1f} / TN {sanity['kept_fired_to_TN_surface']:.1f} "
              f"/ TP {sanity['kept_fired_to_TP_wire']:.1f})")


if __name__ == "__main__":
    main()
