"""Forensic diagnosis of CONNECTOR (class 2) recall for the 3-way SegFormer-B5.

Goal: of the connector GT pixels the model MISSES (false negatives), what
fraction are predicted as WIRE (class 1, connector<->wire confusion) vs BACKGROUND
(class 0, under-firing / sub-resolution)?  Plus: does recall depend on connector
blob size, FN confidence, and spatial relationship to wire / predicted connector.

REUSES the real trainer (src/train_rgb_only_sota.py) build + dataset + metric so
the numbers reproduce the in-training IoU(connector) ~ 0.4576 at epoch 10.

Run (GPU1 ONLY; GPU0 is busy training):
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
        ./env/bin/python src/analyze_3way_connector_fn.py
"""

import os
import sys
import json

# Must be set BEFORE transformers is imported (train_rgb_only_sota imports it).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
import cv2
import scipy.ndimage as ndi

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

# Reuse the EXACT trainer pieces so preprocessing / model build / metric match.
from train_rgb_only_sota import (  # noqa: E402
    build_cache,
    CDLORGBOnlyDataset,
    normalize_batch,
    file_list,
    SegFormerSegmenter,
    MultiClassIoU,
    CLASS_NAMES_THREE_WAY,
)
from torch.utils.data import DataLoader  # noqa: E402

# ─────────────────────────── CONFIG ───────────────────────────
DATA_DIR = os.environ.get("FORENSIC_DATA_DIR",
    os.path.join(PROJECT_ROOT, "data", "dformer_dataset_3way_decheat"))
CKPT = os.environ.get("FORENSIC_CKPT", os.path.join(
    PROJECT_ROOT, "results", "realism_campaign", "p_3way_decheat",
    "seg_b5_3way_v1", "epoch_10.pth",
))
OUT_DIR = os.environ.get("FORENSIC_OUT_DIR", os.path.join(
    PROJECT_ROOT, "results", "realism_campaign", "p_3way_decheat", "forensic",
))
BACKBONE = "nvidia/mit-b5"
NUM_CLASSES = 3
BATCH = 8
USE_AMP = True  # trainer eval default (use_amp = not --no-amp)

# class ids
BG, WIRE, CON = 0, 1, 2

# component-size buckets (px)
SIZE_EDGES = [0, 10, 30, 100, 300, 1000, 10**12]
SIZE_LABELS = ["<10", "10-30", "30-100", "100-300", "300-1000", ">1000"]

# connector-prob buckets for FN px
PROB_EDGES = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0001])
PROB_LABELS = ["<0.05", "0.05-0.1", "0.1-0.2", "0.2-0.3",
               "0.3-0.4", "0.4-0.5", ">=0.5"]

# distance buckets (px)
DIST_EDGES = np.array([0, 1, 2, 3, 5, 8, 12, 20, 40, 1e9])
DIST_LABELS = ["0", "1", "2", "3-4", "5-7", "8-11", "12-19", "20-39", ">=40"]


def build_and_load():
    device = torch.device("cuda:0")
    # criterion only needs the right weight-buffer shape [3] for a clean strict
    # load of the checkpoint's criterion.weight; values irrelevant at inference.
    criterion = nn.CrossEntropyLoss(
        weight=torch.ones(NUM_CLASSES), reduction="none", ignore_index=-1)
    model = SegFormerSegmenter(
        backbone_name=BACKBONE, num_classes=NUM_CLASSES, criterion=criterion
    ).to(device)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=True)  # strict
    print(f"[load] strict load OK  missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model, device


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    print("[data] building / loading val cache (test.txt)…")
    val_rgb, _, val_label = build_cache(DATA_DIR, "val")
    val_files = file_list(DATA_DIR, "val")  # ordered, parallel to cache rows
    n = val_rgb.shape[0]
    assert len(val_files) == n, (len(val_files), n)
    print(f"[data] {n} val images")

    val_ds = CDLORGBOnlyDataset(
        val_rgb, val_label, augment=False, include_noise=False,
        num_classes=NUM_CLASSES,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            num_workers=0, pin_memory=True)

    model, device = build_and_load()

    # ── accumulators ──
    metric = MultiClassIoU(num_classes=NUM_CLASSES, class_names=CLASS_NAMES_THREE_WAY)

    # connector decomposition
    con_gt = 0           # total connector GT px
    con_tp = 0           # gt2 & pred2
    fn_to_wire = 0       # gt2 & pred1
    fn_to_bg = 0         # gt2 & pred0
    fp_from_wire = 0     # pred2 & gt1
    fp_from_bg = 0       # pred2 & gt0
    pred_con_total = 0   # total pred2 px

    # FN connector prob histograms (the model's connector-class softmax prob on
    # FN px) + comparison on TP px
    fn_prob_hist = np.zeros(len(PROB_LABELS), dtype=np.int64)
    tp_prob_hist = np.zeros(len(PROB_LABELS), dtype=np.int64)
    fn_prob_sum = 0.0
    fn_prob_sumsq = 0.0
    # fine hist for percentiles of FN connector prob
    fine_bins = np.linspace(0, 1, 101)
    fn_prob_fine = np.zeros(100, dtype=np.int64)
    # what class wins on FN px? (argmax already known: 0 or 1) and how confident
    # is the WINNING (wrong) class on FN px
    fn_winprob_sum = 0.0

    # component analysis: per-bucket component counts + recall + pixel recall
    comp_count = np.zeros(len(SIZE_LABELS), dtype=np.int64)
    comp_reced = np.zeros(len(SIZE_LABELS), dtype=np.int64)   # components w/ recall>0.5
    comp_recall_sum = np.zeros(len(SIZE_LABELS), dtype=np.float64)  # sum of per-comp recall
    comp_gtpx = np.zeros(len(SIZE_LABELS), dtype=np.int64)    # total gt px in bucket
    comp_tppx = np.zeros(len(SIZE_LABELS), dtype=np.int64)    # total caught px in bucket
    all_comp_sizes = []   # every connector component size (for median)
    all_comp_recall = []  # parallel per-component recall (frac px pred==2)

    # spatial: FN px distance to nearest GT wire & nearest predicted connector
    fn_dist_wire_hist = np.zeros(len(DIST_LABELS), dtype=np.int64)
    fn_dist_predcon_hist = np.zeros(len(DIST_LABELS), dtype=np.int64)
    fn_no_wire_in_frame = 0       # FN px in frames with zero GT wire
    fn_no_predcon_in_frame = 0    # FN px in frames with zero predicted connector
    # comparison: TP px distance to nearest GT wire (is TP further from wire?)
    tp_dist_wire_sum = 0.0
    tp_dist_wire_cnt = 0
    fn_dist_wire_sum = 0.0
    fn_dist_wire_cnt = 0

    # per-image connector recall (for selecting worst frames)
    per_img = []  # (idx, gt2_count, tp2_count, recall, fn_wire, fn_bg)

    print("[eval] running forensic pass…")
    with torch.no_grad():
        gidx = 0
        for batch in val_loader:
            rgb = normalize_batch(batch["rgb"], device)
            label_np = batch["label"].numpy()  # (B,H,W) int64 in {0,1,2}
            if USE_AMP:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(rgb)
            else:
                logits = model(rgb)
            logits = logits.float()  # (B,3,H,W)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B,3,H,W)
            pred = logits.argmax(dim=1).cpu().numpy()           # (B,H,W)

            for b in range(pred.shape[0]):
                p = pred[b]
                g = label_np[b]
                pc = probs[b, CON]   # connector prob map (H,W)

                metric.update(p.ravel(), g.ravel())

                gt2 = (g == CON)
                pr2 = (p == CON)
                tp_m = gt2 & pr2
                fn_m = gt2 & ~pr2
                fnw_m = gt2 & (p == WIRE)
                fnb_m = gt2 & (p == BG)

                ngt2 = int(gt2.sum())
                ntp = int(tp_m.sum())
                nfnw = int(fnw_m.sum())
                nfnb = int(fnb_m.sum())

                con_gt += ngt2
                con_tp += ntp
                fn_to_wire += nfnw
                fn_to_bg += nfnb
                fp_from_wire += int((pr2 & (g == WIRE)).sum())
                fp_from_bg += int((pr2 & (g == BG)).sum())
                pred_con_total += int(pr2.sum())

                rec = ntp / ngt2 if ngt2 > 0 else float("nan")
                per_img.append((gidx + b, ngt2, ntp, rec, nfnw, nfnb))

                # FN connector prob distribution
                if fn_m.any():
                    fnp = pc[fn_m]
                    fn_prob_hist += np.histogram(fnp, bins=PROB_EDGES)[0]
                    fn_prob_fine += np.histogram(fnp, bins=fine_bins)[0]
                    fn_prob_sum += float(fnp.sum())
                    fn_prob_sumsq += float((fnp * fnp).sum())
                    # winning (wrong) class prob on FN px
                    winp = probs[b].max(axis=0)[fn_m]
                    fn_winprob_sum += float(winp.sum())
                if tp_m.any():
                    tp_prob_hist += np.histogram(pc[tp_m], bins=PROB_EDGES)[0]

                # connected components on GT connector mask
                if ngt2 > 0:
                    lab, ncomp = ndi.label(gt2)
                    if ncomp > 0:
                        sizes = ndi.sum(np.ones_like(lab), lab,
                                        index=np.arange(1, ncomp + 1)).astype(np.int64)
                        caught = ndi.sum(pr2.astype(np.float64), lab,
                                         index=np.arange(1, ncomp + 1))
                        for s, c in zip(sizes, caught):
                            r = c / s if s > 0 else 0.0
                            bidx = int(np.digitize(s, SIZE_EDGES) - 1)
                            bidx = min(max(bidx, 0), len(SIZE_LABELS) - 1)
                            comp_count[bidx] += 1
                            comp_recall_sum[bidx] += r
                            comp_gtpx[bidx] += int(s)
                            comp_tppx[bidx] += int(round(c))
                            if r > 0.5:
                                comp_reced[bidx] += 1
                            all_comp_sizes.append(int(s))
                            all_comp_recall.append(float(r))

                # spatial: distances on FN px
                if fn_m.any():
                    wire_mask = (g == WIRE)
                    if wire_mask.any():
                        dwire = ndi.distance_transform_edt(~wire_mask)
                        dvals = dwire[fn_m]
                        fn_dist_wire_hist += np.histogram(dvals, bins=DIST_EDGES)[0]
                        fn_dist_wire_sum += float(dvals.sum())
                        fn_dist_wire_cnt += int(dvals.size)
                        # TP distance to wire (comparison)
                        if tp_m.any():
                            tvals = dwire[tp_m]
                            tp_dist_wire_sum += float(tvals.sum())
                            tp_dist_wire_cnt += int(tvals.size)
                    else:
                        fn_no_wire_in_frame += int(fn_m.sum())
                    if pr2.any():
                        dpc = ndi.distance_transform_edt(~pr2)
                        fn_dist_predcon_hist += np.histogram(dpc[fn_m], bins=DIST_EDGES)[0]
                    else:
                        fn_no_predcon_in_frame += int(fn_m.sum())

            gidx += pred.shape[0]
            if gidx % 200 == 0:
                print(f"    {gidx}/{n}")

    # ── compute aggregate IoU via the trainer metric ──
    res = metric.compute()
    cm = metric.cm.astype(np.float64)  # rows=gt, cols=pred

    # ── derived headline numbers ──
    fn_total = fn_to_wire + fn_to_bg
    con_recall = con_tp / max(con_gt, 1)
    fn_wire_pct = 100.0 * fn_to_wire / max(fn_total, 1)
    fn_bg_pct = 100.0 * fn_to_bg / max(fn_total, 1)
    fp_total = fp_from_wire + fp_from_bg
    fp_wire_pct = 100.0 * fp_from_wire / max(fp_total, 1)
    fp_bg_pct = 100.0 * fp_from_bg / max(fp_total, 1)

    fn_prob_mean = fn_prob_sum / max(fn_total, 1)
    fn_prob_var = fn_prob_sumsq / max(fn_total, 1) - fn_prob_mean ** 2
    fn_prob_std = float(np.sqrt(max(fn_prob_var, 0)))
    # median / percentiles of FN connector prob from fine hist
    cumsum = np.cumsum(fn_prob_fine)
    tot = cumsum[-1] if cumsum[-1] > 0 else 1
    centers = 0.5 * (fine_bins[:-1] + fine_bins[1:])

    def pct(q):
        thr = q * tot
        i = int(np.searchsorted(cumsum, thr))
        i = min(i, len(centers) - 1)
        return float(centers[i])

    fn_prob_median = pct(0.5)
    fn_prob_p90 = pct(0.9)
    fn_winprob_mean = fn_winprob_sum / max(fn_total, 1)

    med_comp_size = float(np.median(all_comp_sizes)) if all_comp_sizes else 0.0

    # ── normalized confusion matrix (rows) ──
    cm_rownorm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    # ── PRINT REPORT ──
    P = print
    P("\n" + "=" * 74)
    P("SANITY GATE  (forensic vs in-training epoch_10)")
    P("=" * 74)
    P(f"  IoU(connector) = {res['iou_connector']:.4f}   (expected ~0.4576)")
    P(f"  IoU(wire)      = {res['iou_wire']:.4f}   (expected ~0.8215)")
    P(f"  IoU(bg)        = {res['iou_bg']:.4f}   (expected ~0.9972)")
    P(f"  mIoU           = {res['miou']:.4f}   (expected ~0.7588)")
    P(f"  recall(con)    = {res['recall_connector']:.4f}   (expected ~0.5693)")
    P(f"  prec(con)      = {res['precision_connector']:.4f}   (expected ~0.6999)")
    dcon = abs(res['iou_connector'] - 0.4576)
    P(f"  --> |IoU(con) - 0.4576| = {dcon:.4f}  "
      f"{'PASS' if dcon <= 0.01 else 'FAIL (preprocessing differs!)'}")

    P("\n" + "=" * 74)
    P("3x3 CONFUSION MATRIX  (rows = GT, cols = PRED), row-normalized")
    P("=" * 74)
    P(f"           {'pred bg':>12}{'pred wire':>12}{'pred con':>12}")
    for i, nm in enumerate(["gt bg", "gt wire", "gt con"]):
        P(f"  {nm:>8} " + "".join(f"{cm_rownorm[i, j]:12.4f}" for j in range(3)))
    P("  raw pixel counts:")
    for i, nm in enumerate(["gt bg", "gt wire", "gt con"]):
        P(f"  {nm:>8} " + "".join(f"{int(cm[i, j]):>14d}" for j in range(3)))

    P("\n" + "=" * 74)
    P("HEADLINE: CONNECTOR FALSE-NEGATIVE BREAKDOWN")
    P("=" * 74)
    P(f"  connector GT px ........ {con_gt:,}")
    P(f"  connector TP (gt2&pred2) {con_tp:,}   recall={con_recall:.4f}")
    P(f"  connector FN (gt2&!=2) . {fn_total:,}")
    P(f"     FN -> WIRE (gt2,pred1) {fn_to_wire:,}   = {fn_wire_pct:.1f}% of FN   "
      f"<== connector<->wire confusion")
    P(f"     FN -> BG   (gt2,pred0) {fn_to_bg:,}   = {fn_bg_pct:.1f}% of FN   "
      f"<== under-firing / sub-resolution")
    P(f"  connector FP (pred2,gt!=2) {fp_total:,}")
    P(f"     FP from WIRE (pred2,gt1) {fp_from_wire:,}  = {fp_wire_pct:.1f}% of FP")
    P(f"     FP from BG   (pred2,gt0) {fp_from_bg:,}  = {fp_bg_pct:.1f}% of FP")

    P("\n" + "=" * 74)
    P("FN CONNECTOR-PROB DISTRIBUTION  (model's softmax p(connector) on FN px)")
    P("=" * 74)
    for lbl, c in zip(PROB_LABELS, fn_prob_hist):
        P(f"  p(con) {lbl:>9} : {c:>12,}  ({100.0*c/max(fn_total,1):5.1f}%)")
    P(f"  FN p(con): mean={fn_prob_mean:.3f} median={fn_prob_median:.3f} "
      f"p90={fn_prob_p90:.3f} std={fn_prob_std:.3f}")
    P(f"  FN winning(wrong)-class prob: mean={fn_winprob_mean:.3f}  "
      f"(how confident the model is in the WRONG class)")
    P("  (for comparison) TP p(con) distribution:")
    tp_tot = max(int(tp_prob_hist.sum()), 1)
    for lbl, c in zip(PROB_LABELS, tp_prob_hist):
        P(f"  p(con) {lbl:>9} : {c:>12,}  ({100.0*c/tp_tot:5.1f}%)")

    P("\n" + "=" * 74)
    P("CONNECTOR COMPONENT-SIZE RECALL  (GT connector connected-components)")
    P("=" * 74)
    P(f"  total components: {int(comp_count.sum()):,}   median size: {med_comp_size:.0f} px")
    P(f"  {'size bucket':>12}{'#comp':>9}{'%comp':>8}{'comp-mean rec':>15}"
      f"{'pixel rec':>12}{'%comp rec>0.5':>15}")
    tc = max(int(comp_count.sum()), 1)
    for i, lbl in enumerate(SIZE_LABELS):
        cc = int(comp_count[i])
        cmean = comp_recall_sum[i] / max(cc, 1)
        prec_px = comp_tppx[i] / max(comp_gtpx[i], 1)
        pgt = 100.0 * comp_reced[i] / max(cc, 1)
        P(f"  {lbl:>12}{cc:>9}{100.0*cc/tc:>7.1f}%{cmean:>15.4f}"
          f"{prec_px:>12.4f}{pgt:>14.1f}%")

    P("\n" + "=" * 74)
    P("SPATIAL: FN connector px relationship to wire / predicted-connector")
    P("=" * 74)
    P(f"  FN px (with wire in frame): mean dist to nearest GT wire = "
      f"{fn_dist_wire_sum/max(fn_dist_wire_cnt,1):.2f} px")
    P(f"  TP px (with wire in frame): mean dist to nearest GT wire = "
      f"{tp_dist_wire_sum/max(tp_dist_wire_cnt,1):.2f} px  (comparison)")
    P(f"  FN px in frames with ZERO GT wire: {fn_no_wire_in_frame:,} "
      f"({100.0*fn_no_wire_in_frame/max(fn_total,1):.1f}% of FN)")
    P("  FN px distance to nearest GT WIRE pixel:")
    dw_tot = max(int(fn_dist_wire_hist.sum()), 1)
    for lbl, c in zip(DIST_LABELS, fn_dist_wire_hist):
        P(f"    dist {lbl:>6} px : {c:>12,}  ({100.0*c/dw_tot:5.1f}%)")
    P(f"  FN px in frames with ZERO predicted connector: {fn_no_predcon_in_frame:,} "
      f"({100.0*fn_no_predcon_in_frame/max(fn_total,1):.1f}% of FN)")
    P("  FN px distance to nearest PREDICTED-connector pixel:")
    dp_tot = max(int(fn_dist_predcon_hist.sum()), 1)
    for lbl, c in zip(DIST_LABELS, fn_dist_predcon_hist):
        P(f"    dist {lbl:>6} px : {c:>12,}  ({100.0*c/dp_tot:5.1f}%)")

    # ── overlay sheets for worst connector-recall frames (meaningful area) ──
    P("\n[overlays] selecting worst connector-recall frames (gt2>=500 px)…")
    cand = [r for r in per_img if r[1] >= 500]
    cand.sort(key=lambda r: r[3])  # ascending recall
    worst = cand[:6]
    sheet_paths = render_overlays(worst, val_rgb, val_label, val_files, model, device)
    for sp in sheet_paths:
        P(f"  wrote {sp}")

    # ── save forensic.json ──
    out = {
        "checkpoint": CKPT,
        "n_val_images": int(n),
        "sanity": {
            "iou_connector": res["iou_connector"],
            "iou_wire": res["iou_wire"],
            "iou_bg": res["iou_bg"],
            "miou": res["miou"],
            "recall_connector": res["recall_connector"],
            "precision_connector": res["precision_connector"],
            "expected": {"iou_connector": 0.4576, "iou_wire": 0.8215,
                         "recall_connector": 0.5693, "precision_connector": 0.6999},
            "iou_connector_abs_err": dcon,
            "pass": bool(dcon <= 0.01),
        },
        "confusion_matrix_raw": cm.astype(np.int64).tolist(),
        "confusion_matrix_rownorm": cm_rownorm.tolist(),
        "connector_decomposition": {
            "gt_px": con_gt, "tp_px": con_tp, "recall": con_recall,
            "fn_total": fn_total,
            "fn_to_wire_px": fn_to_wire, "fn_to_wire_pct": fn_wire_pct,
            "fn_to_bg_px": fn_to_bg, "fn_to_bg_pct": fn_bg_pct,
            "fp_total": fp_total, "pred_con_total": pred_con_total,
            "fp_from_wire_px": fp_from_wire, "fp_from_wire_pct": fp_wire_pct,
            "fp_from_bg_px": fp_from_bg, "fp_from_bg_pct": fp_bg_pct,
        },
        "fn_connector_prob": {
            "buckets": PROB_LABELS,
            "hist": fn_prob_hist.tolist(),
            "mean": fn_prob_mean, "median": fn_prob_median,
            "p90": fn_prob_p90, "std": fn_prob_std,
            "winning_wrong_class_prob_mean": fn_winprob_mean,
            "tp_hist": tp_prob_hist.tolist(),
        },
        "component_size_recall": {
            "buckets": SIZE_LABELS,
            "edges": SIZE_EDGES,
            "n_components": comp_count.tolist(),
            "comp_mean_recall": (comp_recall_sum / np.clip(comp_count, 1, None)).tolist(),
            "pixel_recall": (comp_tppx / np.clip(comp_gtpx, 1, None)).tolist(),
            "n_comp_recall_gt0p5": comp_reced.tolist(),
            "gt_px": comp_gtpx.tolist(),
            "tp_px": comp_tppx.tolist(),
            "median_component_size": med_comp_size,
            "total_components": int(comp_count.sum()),
        },
        "spatial": {
            "dist_buckets": DIST_LABELS,
            "fn_dist_to_wire_hist": fn_dist_wire_hist.tolist(),
            "fn_dist_to_predcon_hist": fn_dist_predcon_hist.tolist(),
            "fn_mean_dist_to_wire": fn_dist_wire_sum / max(fn_dist_wire_cnt, 1),
            "tp_mean_dist_to_wire": tp_dist_wire_sum / max(tp_dist_wire_cnt, 1),
            "fn_no_wire_in_frame_px": fn_no_wire_in_frame,
            "fn_no_predcon_in_frame_px": fn_no_predcon_in_frame,
        },
        "worst_frames": [
            {"idx": r[0], "file": val_files[r[0]], "gt2_px": r[1],
             "tp2_px": r[2], "recall": r[3], "fn_to_wire": r[4], "fn_to_bg": r[5]}
            for r in worst
        ],
        "overlay_sheets": sheet_paths,
    }
    json_path = os.path.join(OUT_DIR, "forensic.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o)
                  if isinstance(o, (np.floating, np.integer)) else str(o))
    P(f"\n[done] forensic.json -> {json_path}")


# ── 3-color + error-map rendering ──
# BGR colors for cv2.imwrite
COL_BG = (50, 50, 50)
COL_WIRE = (0, 200, 0)       # green
COL_CON = (0, 0, 255)        # red
ERR_TP = (0, 255, 0)         # connector TP   = green
ERR_FN_WIRE = (0, 165, 255)  # FN->wire       = orange
ERR_FN_BG = (0, 0, 255)      # FN->bg         = red
ERR_FP = (255, 0, 255)       # FP             = magenta


def colorize(seg):
    h, w = seg.shape
    out = np.zeros((h, w, 3), np.uint8)
    out[seg == BG] = COL_BG
    out[seg == WIRE] = COL_WIRE
    out[seg == CON] = COL_CON
    return out


def error_map(g, p, base_bgr):
    out = (base_bgr.astype(np.float32) * 0.35).astype(np.uint8)  # dim context
    gt2 = (g == CON)
    pr2 = (p == CON)
    out[gt2 & pr2] = ERR_TP
    out[gt2 & (p == WIRE)] = ERR_FN_WIRE
    out[gt2 & (p == BG)] = ERR_FN_BG
    out[pr2 & (g != CON)] = ERR_FP
    return out


def label_panel(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_overlays(worst, val_rgb, val_label, val_files, model, device):
    paths = []
    for rank, r in enumerate(worst):
        idx = r[0]
        bgr = np.array(val_rgb[idx])           # (H,W,3) uint8 BGR (cache order)
        g = np.array(val_label[idx]).astype(np.int64)
        rgb_in = bgr[:, :, ::-1].copy()        # BGR->RGB for the model
        x = torch.from_numpy(rgb_in.transpose(2, 0, 1).copy()).unsqueeze(0)
        xn = normalize_batch(x, device)
        with torch.no_grad():
            if USE_AMP:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(xn)
            else:
                logits = model(xn)
        p = logits.float().argmax(dim=1)[0].cpu().numpy()

        panel_in = label_panel(bgr, f"input {os.path.basename(val_files[idx])}")
        panel_gt = label_panel(colorize(g), "GT (bg/wire=grn/con=red)")
        panel_pr = label_panel(colorize(p), "PRED")
        panel_er = label_panel(
            error_map(g, p, bgr),
            f"err TPgrn FN>wire orng FN>bg red FP mag  rec={r[3]:.2f}")
        sheet = cv2.hconcat([panel_in, panel_gt, panel_pr, panel_er])
        path = os.path.join(OUT_DIR, f"worst_con_recall_{rank:02d}_idx{idx}.png")
        cv2.imwrite(path, sheet)
        paths.append(path)
    return paths


if __name__ == "__main__":
    main()
