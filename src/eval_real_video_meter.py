#!/usr/bin/env python
"""Real-video PRECISION meter + RECALL proxies for bg/wire/connector models (no GT needed).

The 4 real clips (data/dlo_real_sample_{1..4}.mp4) have no ground truth. "Wire coverage %"
rewards false positives (P3W paints skin), so this meter splits the question in two:

  PRECISION (definite FP): a person/hand mask from an off-the-shelf panoptic segmenter
    (facebook/mask2former-swin-large-coco-panoptic, class "person"), eroded by --erode px.
    Wire/connector px inside it are counted as definite FP:
      fp_rate_<cls>     = FP px / all predicted <cls> px (pooled)
      fp_px_per_frame   = FP px / n frames
    Connector blobs (8-conn, area >= 50 px, same as eval_qn3way_suite): count, bbox max-dim
    p50/p90, fraction of blobs touching the eroded person mask.
  RECALL proxies (because the FP meter rewards blank models):
    consensus ref = px where >= 2 of {p3w, qn, a1_ep30} predict wire, outside the (non-eroded)
                    person mask;  union ref = any of the three.
      CIRCULAR for those three models - see METER.md.
    ridge ref   = model-independent thin-dark/bright ridge detector (see ridge_reference()),
                  outside person; a structure-recall proxy, low ceiling, not cable-specific.
  Recall is reported for predicted wire (cls 1) and for wire|connector (fg).

Everything model-independent is cached once under --cache (frames, person masks, refs);
per-model argmax maps are cached under --cache/preds keyed by ckpt path+mtime+size.

Usage:
  ./env/bin/python src/eval_real_video_meter.py NAME=CKPT [NAME=CKPT ...] [--stride 5] [--gpu 1]
  ./env/bin/python src/eval_real_video_meter.py --table          # just rebuild the table
  ./env/bin/python src/eval_real_video_meter.py --audit          # person-mask validity audit
"""
import argparse
import glob
import json
import os
import sys
import time


def _early_gpu():
    for i, a in enumerate(sys.argv):
        if a == "--gpu" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--gpu="):
            return a.split("=", 1)[1]
    return "1"


os.environ["CUDA_VISIBLE_DEVICES"] = _early_gpu()  # never touch any other GPU

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "hf_cache"))
from infer_video_rgb_only import preprocess  # noqa: E402

DEV = torch.device("cuda:0")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
VIDEOS = {f"v{i}": os.path.join(ROOT, f"data/dlo_real_sample_{i}.mp4") for i in (1, 2, 3, 4)}
OUT_DEFAULT = os.path.join(ROOT, "results/qn_backbone_3way/meter")
PERSON_MODEL = "facebook/mask2former-swin-large-coco-panoptic"
REFS_DEFAULT = {
    "p3w": "results/realism_campaign/p3w_lovasz/lovasz050_zoom/best_model.pth",
    "qn": "results/qn_backbone_3way/recon/qn_best_model_hf4.pth",
    "a1_ep30": "results/qn_backbone_3way/a1_kdpos/epoch_30.pth",
}
BLOB_MIN_AREA = 50
RIDGE_TOL = 4  # px tolerance for the model-free ridge recall proxy
ORPHAN_D = 25  # px: prediction farther than this from union-wire ref | ridge | person = "orphan"
BS = 8
H, W = 480, 640


def pack(m):
    return np.packbits(m.astype(bool), axis=-1)


def unpack(p):
    return np.unpackbits(p, axis=-1)[..., :W].astype(bool)


# ================================================================ stage 1: frames
def frames_cache(cache, stride):
    """Every stride-th frame of each clip, preprocessed (centre-crop 4:3 -> 640x480 BGR)."""
    out = {}
    for v, path in VIDEOS.items():
        p = os.path.join(cache, f"frames_s{stride}_{v}.npy")
        pi = os.path.join(cache, f"frames_s{stride}_{v}_idx.json")
        if not os.path.exists(p):
            cap = cv2.VideoCapture(path)
            fr, idx, fi = [], [], 0
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                if fi % stride == 0:
                    fr.append(preprocess(f)); idx.append(fi)
                fi += 1
            cap.release()
            np.save(p, np.stack(fr))
            json.dump({"frame_idx": idx, "n_total": fi}, open(pi, "w"))
            print(f"[frames] {v}: {len(fr)}/{fi} frames cached", flush=True)
        out[v] = (np.load(p, mmap_mode="r"), json.load(open(pi))["frame_idx"])
    return out


# ================================================================ stage 2: person masks
def person_cache(cache, stride, frames):
    """uint8 person score (0-255) + bool argmax==person, per frame. Model-independent."""
    out = {}
    todo = [v for v in VIDEOS if not os.path.exists(os.path.join(cache, f"person_s{stride}_{v}.npz"))]
    if todo:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        proc = AutoImageProcessor.from_pretrained(PERSON_MODEL)
        net = Mask2FormerForUniversalSegmentation.from_pretrained(PERSON_MODEL).eval().to(DEV)
        pid = [k for k, n in net.config.id2label.items() if n == "person"]
        assert pid == [0], pid
        for v in todo:
            t0 = time.time()
            fr = frames[v][0]
            scores, sems = [], []
            for i in range(0, len(fr), 4):
                rgb = [f[:, :, ::-1].copy() for f in fr[i:i + 4]]
                inp = proc(images=rgb, size={"height": H, "width": W}, return_tensors="pt").to(DEV)
                ph, pw = inp["pixel_values"].shape[-2:]
                assert abs(pw / ph - W / H) < 0.02, ("processor padded/cropped", ph, pw)
                with torch.no_grad():
                    o = net(**inp)
                # semantic map exactly as HF post_process_semantic_segmentation, keep person channel
                cls = o.class_queries_logits.softmax(-1)[..., :-1]          # B,Q,133
                msk = o.masks_queries_logits                                # B,Q,h,w
                msk = torch.nn.functional.interpolate(msk, size=inp["pixel_values"].shape[-2:],
                                                      mode="bilinear", align_corners=False).sigmoid()
                seg = torch.einsum("bqc,bqhw->bchw", cls, msk)              # B,133,H',W'
                # processor pads/resizes; resize the whole valid canvas back to 480x640
                seg = torch.nn.functional.interpolate(seg, size=(H, W), mode="bilinear",
                                                      align_corners=False)
                sems.append((seg.argmax(1) == 0).cpu().numpy())
                scores.append((seg[:, 0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy())
            sc, se = np.concatenate(scores), np.concatenate(sems)
            np.savez_compressed(os.path.join(cache, f"person_s{stride}_{v}.npz"), score=sc, sem=pack(se))
            print(f"[person] {v}: {len(sc)} frames, mean person area {se.mean()*100:.1f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)
        del net
        torch.cuda.empty_cache()
    for v in VIDEOS:
        z = np.load(os.path.join(cache, f"person_s{stride}_{v}.npz"))
        out[v] = unpack(z["sem"])
    return out


def erode(masks, px):
    if px <= 0:
        return masks
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return np.stack([cv2.erode(m.astype(np.uint8), k).astype(bool) for m in masks])


SKIN_T, SKIN_LMIN, SKIN_DIL = 10.0, 30.0, 2


def skin_region(bgr, person):
    """Skin-consistent part of the person mask: the panoptic 'person' mask INCLUDES cables held
    across the hand (verified on v2/v3 overlays), so counting every wire px inside it as FP would
    punish correct held-cable recall. Keep only person px whose Lab chroma lies within SKIN_T of
    the median chroma of their own person component (per hand -> handles 2 skin tones/shading)
    and L > SKIN_LMIN (black cable), then remove a SKIN_DIL-px dilation of the non-skin px
    (mixed-colour cable borders). Black/white/grey/red cables fall out; skin stays."""
    if not person.any():
        return person
    lab = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1], lab[..., 2]
    ok = np.zeros(person.shape, bool)
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(person.astype(np.uint8), connectivity=8)
    ga, gb = np.median(a[person]), np.median(b[person])
    for j in range(1, n):
        cc = lab_cc == j
        ma, mb = (np.median(a[cc]), np.median(b[cc])) if st[j, cv2.CC_STAT_AREA] >= 500 else (ga, gb)
        ok |= cc & (np.sqrt((a - ma) ** 2 + (b - mb) ** 2) < SKIN_T)
    ok &= lab[..., 0] > SKIN_LMIN
    non = person & ~ok
    if SKIN_DIL:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SKIN_DIL + 1, 2 * SKIN_DIL + 1))
        non = cv2.dilate(non.astype(np.uint8), k).astype(bool)
    return person & ~non


def skin_cache(cache, stride, frames, person):
    p = os.path.join(cache, f"skin_s{stride}_T{SKIN_T:g}_L{SKIN_LMIN:g}_d{SKIN_DIL}.npz")
    if not os.path.exists(p):
        np.savez_compressed(p, **{v: pack(np.stack([skin_region(f, m) for f, m in zip(frames[v][0], person[v])]))
                                  for v in VIDEOS})
    z = np.load(p)
    return {v: unpack(z[v])[:len(frames[v][0])] for v in VIDEOS}


# ================================================================ stage 3: model preds
def load_model(ckpt):
    from train_rgb_only_sota import SegFormerSegmenter
    st = torch.load(os.path.join(ROOT, ckpt), map_location="cpu", weights_only=False)
    sd = st.get("model_state_dict", st)
    sd = {k.replace("module.", ""): v for k, v in sd.items() if k != "criterion.weight"}
    nc = int(sd["model.decode_head.classifier.weight"].shape[0])
    m = SegFormerSegmenter(backbone_name="nvidia/mit-b5", num_classes=nc, criterion=None)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{ckpt}: missing={missing[:4]} unexpected={unexpected[:4]} "
                         "(v5 checkpoint? run src/convert_hf5_segformer_ckpt.py)")
    return m.eval().to(DEV), nc


def pred_cache(cache, stride, name, ckpt, frames):
    """argmax class maps (uint8) for every cached frame; 2-class maps keep DLO=1."""
    ap = os.path.join(ROOT, ckpt)
    if not os.path.exists(ap):
        raise SystemExit(f"missing checkpoint: {ckpt}")
    st = os.stat(ap)
    key = {"ckpt": os.path.relpath(ap, ROOT), "mtime": int(st.st_mtime), "size": st.st_size, "stride": stride}
    d = os.path.join(cache, "preds")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{name}_s{stride}.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        if json.loads(str(z["key"])) == key:
            return {v: z[v] for v in VIDEOS}, int(z["nc"]), 0.0
        print(f"[preds] {name}: cache key changed -> recompute", flush=True)
    t0 = time.time()
    m, nc = load_model(ckpt)
    out = {}
    with torch.no_grad():
        for v in VIDEOS:
            fr = frames[v][0]
            ps = []
            for i in range(0, len(fr), BS):
                x = torch.from_numpy(np.ascontiguousarray(fr[i:i + BS][..., ::-1])).permute(0, 3, 1, 2)
                x = (x.float() / 255.0 - MEAN) / STD
                ps.append(m(x.to(DEV)).argmax(1).to(torch.uint8).cpu().numpy())
            out[v] = np.concatenate(ps)
    del m
    torch.cuda.empty_cache()
    dt = time.time() - t0
    np.savez_compressed(p, key=json.dumps(key), nc=nc, **out)
    print(f"[preds] {name}: nc={nc}, inference {dt:.0f}s", flush=True)
    return out, nc, dt


# ================================================================ stage 4: references
def ridge_reference(bgr, person):
    """Model-independent thin-structure reference.

    Cables are long, thin (3-15 px at 640x480), and contrast with the desk on both sides.
    Multi-scale Hessian ridge (both polarities) on luma, sigma in {1.5, 2.5, 4}; keep the
    strongest ridges (|lambda_max| normalised by sigma^2, >= frame p97 and >= absolute floor),
    then keep only ELONGATED components: skeleton length >= 60 px and length/width >= 8.
    Outside the (dilated) person mask. Not cable-specific: desk seams, cords, keyboard edges,
    shadows pass too, so recall against it has a low ceiling; use it RELATIVELY.
    """
    from skimage.morphology import skeletonize
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    best = np.zeros_like(g)
    for s in (1.5, 2.5, 4.0):
        b = cv2.GaussianBlur(g, (0, 0), s)
        dxx = cv2.Sobel(b, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(b, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(b, cv2.CV_32F, 1, 1, ksize=3)
        tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
        l1 = 0.5 * (dxx + dyy + tmp)
        l2 = 0.5 * (dxx + dyy - tmp)
        lmax = np.where(np.abs(l1) > np.abs(l2), l1, l2)
        lmin = np.where(np.abs(l1) > np.abs(l2), l2, l1)
        # ridge: one strong curvature, one weak (line-like, not blob-like)
        r = np.abs(lmax) * (np.abs(lmin) < 0.35 * np.abs(lmax)) * s ** 2
        best = np.maximum(best, r)
    thr = max(np.percentile(best, 97), 0.02)
    m = (best >= thr).astype(np.uint8)
    pd = cv2.dilate(person.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    m[pd] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    for j in range(1, n):
        a = stats[j, cv2.CC_STAT_AREA]
        if a < 60:
            continue
        sub = (lab[stats[j, 1]:stats[j, 1] + stats[j, 3], stats[j, 0]:stats[j, 0] + stats[j, 2]] == j)
        length = int(skeletonize(sub).sum())
        width = a / max(length, 1)
        keep[j] = length >= 60 and length / max(width, 1) >= 8
    return keep[lab]


REF_KINDS = ("cons", "union", "held", "ridge")
# pair refs (both of two ref models say wire, outside person) -> leave-one-out recall for refs
PAIRS = [("p3w", "qn"), ("p3w", "a1_ep30"), ("qn", "a1_ep30")]
PAIR_KINDS = tuple(f"pair_{a}+{b}" for a, b in PAIRS)


def refs_cache(cache, stride, frames, person, skin, ref_ckpts):
    """cons/union: >=2 / >=1 of the ref models say wire, OUTSIDE the person mask.
    held: >=2 say wire INSIDE the person mask but NOT on skin-consistent px (= cable held in /
    across the hand; the QN-lineage under-segmentation case). ridge: model-free, outside person."""
    p = os.path.join(cache, f"refs_s{stride}.npz")
    names = sorted(ref_ckpts)
    key = json.dumps({n: ref_ckpts[n] for n in names}, sort_keys=True)
    if os.path.exists(p):
        z = np.load(p)
        if str(z["names"]) == key:
            return {v: {k: unpack(z[f"{v}_{k}"])[:len(frames[v][0])] for k in REF_KINDS + PAIR_KINDS}
                    for v in VIDEOS}
    preds = {n: pred_cache(cache, stride, n, ref_ckpts[n], frames)[0] for n in names}
    out, save = {}, {"names": key}
    for v in VIDEOS:
        votes = sum((preds[n][v] == 1).astype(np.uint8) for n in names)
        outside = ~person[v]
        cons, uni = (votes >= 2) & outside, (votes >= 1) & outside
        held = (votes >= 2) & person[v] & ~skin[v]
        ridge = np.stack([ridge_reference(f, pm) for f, pm in zip(frames[v][0], person[v])])
        out[v] = {"cons": cons, "union": uni, "held": held, "ridge": ridge}
        for (a, b), k in zip(PAIRS, PAIR_KINDS):
            out[v][k] = (preds[a][v] == 1) & (preds[b][v] == 1) & outside
        for k, a in out[v].items():
            save[f"{v}_{k}"] = pack(a)
    np.savez_compressed(p, **save)
    print("[refs] consensus/union/ridge references cached", flush=True)
    return out


# ================================================================ metrics
def blobs(conn, sk):
    rows = []
    for mk, pm in zip(conn, sk):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8), connectivity=8)
        if n <= 1:
            continue
        inside = np.bincount(lab[pm], minlength=n)
        for j in range(1, n):
            a = int(stats[j, cv2.CC_STAT_AREA])
            if a >= BLOB_MIN_AREA:
                rows.append((max(int(stats[j, 2]), int(stats[j, 3])), a, inside[j] / a))
    return rows


def blob_summary(rows):
    if not rows:
        return {"n": 0, "p50": None, "p90": None, "frac_touch_skin": None, "frac_mostly_skin": None}
    d = np.array([r[0] for r in rows]); f = np.array([r[2] for r in rows])
    return {"n": len(rows), "p50": float(np.percentile(d, 50)), "p90": float(np.percentile(d, 90)),
            "frac_touch_skin": float((f > 0).mean()), "frac_mostly_skin": float((f > 0.5).mean())}


def support_mask(refs_v, person_v):
    """Everything a wire/connector prediction could plausibly sit near: any ref model's wire,
    any model-free ridge, or a person; dilated by ORPHAN_D. Outside it = orphan prediction."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ORPHAN_D + 1, 2 * ORPHAN_D + 1))
    base = refs_v["union"] | refs_v["ridge"] | refs_v["held"] | person_v
    return np.stack([cv2.dilate(m.astype(np.uint8), k) for m in base]).astype(bool)


def counts(pred, nc, pe, sk, refs, sup):
    """pe = eroded person mask (raw), sk = eroded person & skin-consistent (PRIMARY FP region)."""
    c = {"n_frames": len(pred), "person_px_eroded": int(pe.sum()), "skin_px": int(sk.sum())}
    cls = {"wire": pred == 1, "conn": pred == 2 if nc == 3 else np.zeros_like(pred, bool)}
    cls["fg"] = cls["wire"] | cls["conn"]
    for k, m in cls.items():
        c[f"{k}_px"] = int(m.sum())
        c[f"{k}_fp_px"] = int((m & sk).sum())
        c[f"{k}_fpraw_px"] = int((m & pe).sum())
        c[f"{k}_orphan_px"] = int((m & ~sup).sum())
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RIDGE_TOL + 1, 2 * RIDGE_TOL + 1))
    tol = {"wire": np.stack([cv2.dilate(m.astype(np.uint8), k) for m in cls["wire"]]).astype(bool),
           "fg": np.stack([cv2.dilate(m.astype(np.uint8), k) for m in cls["fg"]]).astype(bool)}
    for r, ref in refs.items():
        # ridge ref fires on the cable's shadow edges a few px OUTSIDE the cable body ->
        # score it with a RIDGE_TOL-px tolerance; model refs are scored exactly.
        pw, pf = (tol["wire"], tol["fg"]) if r == "ridge" else (cls["wire"], cls["fg"])
        c[f"{r}_ref_px"] = int(ref.sum())
        c[f"{r}_tp_wire"] = int((pw & ref).sum())
        c[f"{r}_tp_fg"] = int((pf & ref).sum())
    return c


def finish(c, blob_rows):
    n = max(c["n_frames"], 1)
    r = {"n_frames": c["n_frames"]}
    for k in ("wire", "conn", "fg"):
        r[f"{k}_pct"] = 100 * c[f"{k}_px"] / (n * H * W)
        r[f"{k}_fp_rate"] = c[f"{k}_fp_px"] / c[f"{k}_px"] if c[f"{k}_px"] else None
        r[f"{k}_fp_px_per_frame"] = c[f"{k}_fp_px"] / n
        r[f"{k}_fpraw_rate"] = c[f"{k}_fpraw_px"] / c[f"{k}_px"] if c[f"{k}_px"] else None
        r[f"{k}_fpraw_px_per_frame"] = c[f"{k}_fpraw_px"] / n
        r[f"{k}_orphan_rate"] = c[f"{k}_orphan_px"] / c[f"{k}_px"] if c[f"{k}_px"] else None
        r[f"{k}_clean_px_per_frame"] = (c[f"{k}_px"] - c[f"{k}_fp_px"]) / n
    r["skin_paint_frac_fg"] = c["fg_fp_px"] / max(c["skin_px"], 1)
    for ref in REF_KINDS + PAIR_KINDS:
        rp = c[f"{ref}_ref_px"]
        r[f"recall_{ref}_wire"] = c[f"{ref}_tp_wire"] / rp if rp else None
        r[f"recall_{ref}_fg"] = c[f"{ref}_tp_fg"] / rp if rp else None
    r["conn_blobs"] = blob_summary(blob_rows)
    return r


def evaluate(name, ckpt, args, frames, pe, sk, refs, sup):
    preds, nc, dt = pred_cache(args.cache, args.stride, name, ckpt, frames)
    res = {"name": name, "ckpt": ckpt, "num_classes": nc, "stride": args.stride, "erode_px": args.erode,
           "skin_params": {"T": SKIN_T, "Lmin": SKIN_LMIN, "dil": SKIN_DIL},
           "inference_s": round(dt, 1), "per_video": {}}
    tot, all_blobs = None, []
    for v in VIDEOS:
        c = counts(preds[v], nc, pe[v], sk[v], refs[v], sup[v])
        b = blobs(preds[v] == 2, sk[v]) if nc == 3 else []
        all_blobs += b
        res["per_video"][v] = finish(c, b)
        tot = c if tot is None else {k: tot[k] + c[k] for k in tot}
    res["pooled"] = finish(tot, all_blobs)
    # leave-one-out consensus recall: a ref model is scored on the pair of the OTHER two refs;
    # any other model on the mean over the three pairs
    q = res["pooled"]
    own = [k for (a, b), k in zip(PAIRS, PAIR_KINDS) if name not in (a, b)]
    if len(own) == 1:
        q["recall_loo_fg"], q["loo_ref"] = q[f"recall_{own[0]}_fg"], own[0]
    else:
        q["recall_loo_fg"], q["loo_ref"] = float(np.mean([q[f"recall_{k}_fg"] for k in PAIR_KINDS])), "mean_of_pairs"
    res["pooled"]["mean_video_wire_pct"] = float(np.mean([res["per_video"][v]["wire_pct"] for v in VIDEOS]))
    return res


# ================================================================ table
def fmt(x, p=3, pct=False):
    if x is None:
        return "-"
    return f"{100*x:.1f}" if pct else f"{x:.{p}f}"


COLS = [  # (header, getter)
    ("wire %", lambda q: f"{q['wire_pct']:.2f}"),
    ("conn %", lambda q: f"{q['conn_pct']:.2f}"),
    ("wire FP rate %", lambda q: fmt(q["wire_fp_rate"], pct=True)),
    ("wire FP px/fr", lambda q: f"{q['wire_fp_px_per_frame']:.0f}"),
    ("conn FP rate %", lambda q: fmt(q["conn_fp_rate"], pct=True)),
    ("conn FP px/fr", lambda q: f"{q['conn_fp_px_per_frame']:.0f}"),
    ("skin painted %", lambda q: fmt(q["skin_paint_frac_fg"], pct=True)),
    ("raw-person FP rate wire/conn %", lambda q: f"{fmt(q['wire_fpraw_rate'], pct=True)} / {fmt(q['conn_fpraw_rate'], pct=True)}"),
    ("conn blobs", lambda q: str(q["conn_blobs"]["n"])),
    ("blob p50/p90 px", lambda q: f"{fmt(q['conn_blobs']['p50'], 0)} / {fmt(q['conn_blobs']['p90'], 0)}"),
    ("blobs on skin %", lambda q: fmt(q["conn_blobs"]["frac_touch_skin"], pct=True)),
    ("orphan rate wire/conn %", lambda q: f"{fmt(q['wire_orphan_rate'], pct=True)} / {fmt(q['conn_orphan_rate'], pct=True)}"),
    ("R cons wire/fg %", lambda q: f"{fmt(q['recall_cons_wire'], pct=True)} / {fmt(q['recall_cons_fg'], pct=True)}"),
    ("R LOO fg %", lambda q: fmt(q.get("recall_loo_fg"), pct=True)),
    ("R union wire %", lambda q: fmt(q["recall_union_wire"], pct=True)),
    ("R held fg %", lambda q: fmt(q["recall_held_fg"], pct=True)),
    ("R ridge fg %", lambda q: fmt(q["recall_ridge_fg"], pct=True)),
]


def table(out_dir):
    js = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, "models", "*.json")))]
    hdr = "| model | " + " | ".join(h for h, _ in COLS) + " |"
    sep = "|" + "---|" * (len(COLS) + 1)
    rows = ["| " + r["name"] + " | " + " | ".join(g(r["pooled"]) for _, g in COLS) + " |" for r in js]
    txt = "\n".join([hdr, sep] + rows) + "\n"
    pv = ["| model | video | wire % | conn % | wire FP rate % | conn FP rate % | skin painted % "
          "| R cons fg % | R held fg % | R ridge fg % |", "|" + "---|" * 10]
    for r in js:
        for v, q in r["per_video"].items():
            pv.append(f"| {r['name']} | {v} | {q['wire_pct']:.2f} | {q['conn_pct']:.2f} | "
                      f"{fmt(q['wire_fp_rate'], pct=True)} | {fmt(q['conn_fp_rate'], pct=True)} | "
                      f"{fmt(q['skin_paint_frac_fg'], pct=True)} | {fmt(q['recall_cons_fg'], pct=True)} | "
                      f"{fmt(q['recall_held_fg'], pct=True)} | {fmt(q['recall_ridge_fg'], pct=True)} |")
    open(os.path.join(out_dir, "meter_table.md"), "w").write(txt)
    open(os.path.join(out_dir, "meter_table_per_video.md"), "w").write("\n".join(pv) + "\n")
    print(txt)


# ================================================================ audit (person-mask validity)
ADE_MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"


def ade_person_cache(cache, stride, frames):
    """Independent second person segmenter (ADE20K SegFormer-B5, class 12) - audit only."""
    out = {}
    todo = [v for v in VIDEOS if not os.path.exists(os.path.join(cache, f"person_ade_s{stride}_{v}.npz"))]
    if todo:
        from transformers import SegformerForSemanticSegmentation
        net = SegformerForSemanticSegmentation.from_pretrained(ADE_MODEL).eval().to(DEV)
        assert net.config.id2label[12] == "person"
        for v in todo:
            fr, ms = frames[v][0], []
            with torch.no_grad():
                for i in range(0, len(fr), BS):
                    x = torch.from_numpy(np.ascontiguousarray(fr[i:i + BS][..., ::-1])).permute(0, 3, 1, 2)
                    x = ((x.float() / 255.0 - MEAN) / STD).to(DEV)
                    lg = torch.nn.functional.interpolate(net(pixel_values=x).logits, size=(H, W),
                                                         mode="bilinear", align_corners=False)
                    ms.append((lg.argmax(1) == 12).cpu().numpy())
            np.savez_compressed(os.path.join(cache, f"person_ade_s{stride}_{v}.npz"), sem=pack(np.concatenate(ms)))
        del net
        torch.cuda.empty_cache()
    for v in VIDEOS:
        out[v] = unpack(np.load(os.path.join(cache, f"person_ade_s{stride}_{v}.npz"))["sem"])[:len(frames[v][0])]
    return out


def audit(args, frames, person, pe, sk, refs):
    """Person-mask validity: cross-segmenter disagreement (edge vs interior), held-cable
    inclusion (how much reference-model wire lies inside the person mask), overlays."""
    ad = os.path.join(args.out, "audit")
    os.makedirs(ad, exist_ok=True)
    ade = ade_person_cache(args.cache, args.stride, frames)
    preds = {n: dict(np.load(os.path.join(args.cache, "preds", f"{n}_s{args.stride}.npz"))) for n in ("qn", "p3w", "a1_ep30")}
    border = np.zeros((H, W), bool)
    border[:16] = border[-16:] = True
    border[:, :16] = border[:, -16:] = True
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    t = dict(fr=0, fr_m2f=0, fr_ade=0, inter=0, union=0, fr_miss=0, fr_miss_edge=0, fr_miss_int=0,
             miss_px=0, fr_extra=0, qn_w=0, qn_w_in=0, qn_w_in_e=0, cons_w=0, cons_w_in=0, cons_w_in_e=0,
             qn_w_in_skin=0, cons_w_in_skin=0, pe_px=0, sk_px=0)
    per_frame = []
    for v in VIDEOS:
        for i in range(len(frames[v][0])):
            a, b = person[v][i], ade[v][i]
            adil = cv2.dilate(a.astype(np.uint8), dk).astype(bool)
            bdil = cv2.dilate(b.astype(np.uint8), dk).astype(bool)
            miss = b & ~adil                                    # ADE person, Mask2Former not (missed hand?)
            n, lab, st, _ = cv2.connectedComponentsWithStats(miss.astype(np.uint8), connectivity=8)
            big = [j for j in range(1, n) if st[j, cv2.CC_STAT_AREA] >= 150]
            edge = [j for j in big if border[lab == j].any()]
            extra = (a & ~bdil).sum() >= 150
            qw = preds["qn"][v][i] == 1
            votes = sum((preds[k][v][i] == 1).astype(np.uint8) for k in preds)
            cw = votes >= 2
            t["fr"] += 1; t["fr_m2f"] += int(a.any()); t["fr_ade"] += int(b.any())
            t["inter"] += int((a & b).sum()); t["union"] += int((a | b).sum())
            t["fr_miss"] += int(bool(big)); t["fr_miss_edge"] += int(bool(edge))
            t["fr_miss_int"] += int(len(big) > len(edge)); t["miss_px"] += int(sum(st[j, 4] for j in big))
            t["fr_extra"] += int(extra)
            t["qn_w"] += int(qw.sum()); t["qn_w_in"] += int((qw & a).sum()); t["qn_w_in_e"] += int((qw & pe[v][i]).sum())
            t["cons_w"] += int(cw.sum()); t["cons_w_in"] += int((cw & a).sum())
            t["cons_w_in_e"] += int((cw & pe[v][i]).sum())
            t["qn_w_in_skin"] += int((qw & sk[v][i]).sum()); t["cons_w_in_skin"] += int((cw & sk[v][i]).sum())
            t["pe_px"] += int(pe[v][i].sum()); t["sk_px"] += int(sk[v][i].sum())
            per_frame.append(dict(v=v, i=i, f=frames[v][1][i], person=int(a.sum()),
                                  miss=int(sum(st[j, 4] for j in big)), miss_edge=bool(edge),
                                  qn_in=int((qw & a).sum())))
    rep = {k: t[k] for k in t}
    rep.update({
        "m2f_vs_ade_person_iou": t["inter"] / max(t["union"], 1),
        "frac_frames_with_m2f_person": t["fr_m2f"] / t["fr"],
        "frac_frames_with_ade_person": t["fr_ade"] / t["fr"],
        "frac_frames_ade_only_component": t["fr_miss"] / t["fr"],
        "frac_frames_ade_only_component_at_edge": t["fr_miss_edge"] / t["fr"],
        "frac_frames_ade_only_component_interior": t["fr_miss_int"] / t["fr"],
        "frac_frames_m2f_only_region": t["fr_extra"] / t["fr"],
        "qn_wire_frac_inside_person": t["qn_w_in"] / max(t["qn_w"], 1),
        "qn_wire_frac_inside_eroded_person": t["qn_w_in_e"] / max(t["qn_w"], 1),
        "consensus_wire_frac_inside_person": t["cons_w_in"] / max(t["cons_w"], 1),
        "consensus_wire_frac_inside_eroded_person": t["cons_w_in_e"] / max(t["cons_w"], 1),
        "qn_wire_frac_inside_skin_region": t["qn_w_in_skin"] / max(t["qn_w"], 1),
        "consensus_wire_frac_inside_skin_region": t["cons_w_in_skin"] / max(t["cons_w"], 1),
        "skin_region_frac_of_eroded_person": t["sk_px"] / max(t["pe_px"], 1),
        "per_video_person_area_pct": {v: float(person[v].mean() * 100) for v in VIDEOS},
        "per_video_frames_with_person": {v: float(person[v].any(axis=(1, 2)).mean()) for v in VIDEOS},
    })
    # overlays: 12 frames with the biggest person area (3 per clip that has any, rest top-up),
    # 6 worst cross-segmenter misses, 6 most QN-wire-inside-person (held-cable check)
    sel = []
    for v in VIDEOS:
        pf = sorted([p for p in per_frame if p["v"] == v and p["person"] > 0], key=lambda p: -p["person"])
        sel += [("person", p) for p in pf[:: max(len(pf) // 4, 1)][:4]]
    sel = sel[:12]
    sel += [("miss", p) for p in sorted(per_frame, key=lambda p: -p["miss"])[:6] if p["miss"] > 0]
    sel += [("heldcable", p) for p in sorted(per_frame, key=lambda p: -p["qn_in"])[:6] if p["qn_in"] > 0]
    for v in VIDEOS:
        pf = [p for p in per_frame if p["v"] == v]
        sel += [("ridge", pf[int(k)]) for k in np.linspace(len(pf) * 0.15, len(pf) * 0.85, 2)]
    for tag, p in sel:
        v, i = p["v"], p["i"]
        f = np.ascontiguousarray(frames[v][0][i])
        a, b, e = person[v][i], ade[v][i], pe[v][i]
        o = f.copy()
        o[a] = (o[a] * 0.55 + np.array([0, 0, 255]) * 0.45).astype(np.uint8)          # red = M2F person
        cs, _ = cv2.findContours(b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, cs, -1, (255, 160, 0), 2)                                  # blue = ADE person
        ce, _ = cv2.findContours(e.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, ce, -1, (255, 255, 255), 1)                                # white = eroded edge
        ex = e & ~sk[v][i]
        o[ex] = (0, 200, 255)                                                          # orange = excluded non-skin
        q = f.copy()
        qw = preds["qn"][v][i] == 1
        q[qw] = (0, 255, 0)
        pp = preds["p3w"][v][i]
        r = f.copy()
        r[pp == 1] = (0, 255, 0); r[pp == 2] = (255, 0, 255)
        r[(pp >= 1) & sk[v][i]] = (0, 0, 255)
        g = f.copy()
        g[refs[v]["ridge"][i]] = (255, 0, 255); g[refs[v]["cons"][i]] = (0, 255, 0)
        g[refs[v]["ridge"][i] & refs[v]["cons"][i]] = (255, 255, 0)
        for img, txt in ((o, f"{v} f{p['f']} red=person orange=non-skin blue=ADE"),
                         (q, "QN wire"), (r, "P3W wire/conn, red=counted FP"),
                         (g, "ref: grn=cons mag=ridge cyan=both")):
            cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(ad, f"{tag}_{v}_f{p['f']:04d}.png"),
                    np.vstack([np.hstack([o, q]), np.hstack([r, g])]))
    json.dump(rep, open(os.path.join(ad, "audit.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    return rep


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="NAME=CKPT (path relative to repo root or absolute)")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--erode", type=int, default=4, help="person-mask erosion radius (px)")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--cache", default=None, help="default <out>/cache")
    ap.add_argument("--ref", action="append", default=[], help="override a recall-reference model NAME=CKPT")
    ap.add_argument("--table", action="store_true", help="only rebuild meter_table.md")
    ap.add_argument("--audit", action="store_true", help="person-mask validity audit + overlays")
    ap.add_argument("--max-frames", type=int, default=0, help="debug: limit frames per clip")
    args = ap.parse_args()
    args.cache = args.cache or os.path.join(args.out, "cache_debug" if args.max_frames else "cache")
    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
    if args.table:
        return table(args.out)
    ref_ckpts = dict(REFS_DEFAULT)
    for a in args.ref:
        k, _, v = a.partition("=")
        if k not in ref_ckpts:
            raise SystemExit(f"--ref name must be one of {sorted(ref_ckpts)} (pair refs are wired to them)")
        ref_ckpts[k] = v
    t0 = time.time()
    frames = frames_cache(args.cache, args.stride)
    if args.max_frames:
        frames = {v: (f[:args.max_frames], i[:args.max_frames]) for v, (f, i) in frames.items()}
    person = person_cache(args.cache, args.stride, frames)
    if args.max_frames:
        person = {v: m[:args.max_frames] for v, m in person.items()}
    pe = {v: erode(m, args.erode) for v, m in person.items()}
    skin = skin_cache(args.cache, args.stride, frames, person)
    sk = {v: skin[v] & pe[v] for v in VIDEOS}
    refs = refs_cache(args.cache, args.stride, frames, person, skin, ref_ckpts)
    sup = {v: support_mask(refs[v], person[v]) for v in VIDEOS}
    print(f"[setup] cached inputs ready ({time.time()-t0:.0f}s)", flush=True)
    if args.audit:
        audit(args, frames, person, pe, sk, refs)
    for a in args.models:
        name, _, ck = a.partition("=")
        if not ck:
            raise SystemExit(f"bad model arg {a!r}; expected NAME=CKPT")
        t1 = time.time()
        r = evaluate(name, ck, args, frames, pe, sk, refs, sup)
        r["wall_s"] = round(time.time() - t1, 1)
        json.dump(r, open(os.path.join(args.out, "models", f"{name}.json"), "w"), indent=1)
        q = r["pooled"]
        print(f"[{name}] wire {q['wire_pct']:.2f}% FPrate {fmt(q['wire_fp_rate'], pct=True)}% | "
              f"conn {q['conn_pct']:.2f}% FPrate {fmt(q['conn_fp_rate'], pct=True)}% | "
              f"Rcons(fg) {fmt(q['recall_cons_fg'], pct=True)}% | wall {r['wall_s']}s", flush=True)
    if args.models:
        table(args.out)


if __name__ == "__main__":
    main()
