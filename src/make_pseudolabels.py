#!/usr/bin/env python
"""R2-PL: write pseudo-labels (0=bg 1=wire 2=connector 255=ignore) for a REAL unlabeled pool.

Input pool: a .txt with one image path per line (repo-relative or abs; if a line has a TAB, the
first column is used). Images are centre-cropped to 4:3 and resized to 640x480 exactly like the
trainer (train_qn3way_kd.cover_resize), so the label PNG is pixel-aligned with what the trainer
feeds the network.

Teachers (--teacher):
  p3w_thresh          PLACEHOLDER: P3W 3-way argmax where its max softmax prob >= --conf, else 255.
  fusion:<cfg>        the R2-W dual teacher (src/fusion_teacher.py): P3W (+) QN [+ A1 connector
                      vote] probabilities -> fusion_teacher.pseudo_label(..., conf=--conf,
                      qn_wire_hi=--qn-wire-hi). <cfg> = a preset ('veto05_r95_cg40') or a
                      'k=v,...' list, see fusion_teacher.parse_cfg.
  module:<mod>:<fn>   any callable fn(device, args) -> f(x) where x = (B,3,480,640) ImageNet-
                      normalised RGB float tensor on device and f returns (B,480,640) uint8
                      labels in {0,1,2,255}.

Outputs (--out-dir):
  labels/<uid>.png    uint8 label maps (640x480)
  pl_list.txt         '<rgb path>\\t<label path>' (repo-relative) -> train_qn3way_kd.py --pl-list
  stats.json          per-frame class / ignore fractions + aggregates + config
  sheet.jpg           overlay contact sheet [rgb | PL overlay] for --sheet-n random frames
                      (green = wire, red = connector, grey hatch = ignore)

Optional hard-bg constraint (default OFF; flags-off output is byte-identical):
  --force-bg-mask-list F   TSV '<rgb path>\\t<mask png>' (e.g. EgoHands hand masks). AFTER the teacher,
                           pixels inside the mask (nonzero, cover-resized NEAREST, then ERODED by
                           --force-bg-erode px) are set to 0 (bg) for frames listed in F; all other
                           pixels keep the teacher label. Also writes labels_teacher/<uid>.png (the
                           pre-force teacher map), stats.json 'force_bg' (teacher wire/conn/ignore
                           fraction on hand px, full and eroded mask) and sheet_forcebg.jpg.
  --force-bg-erode N       erosion radius in px (default 3; elliptical (2N+1)^2 kernel).

  CUDA_VISIBLE_DEVICES=1 ./env/bin/python src/make_pseudolabels.py \\
      --pool results/qn_backbone_3way/r2pl/pools/pl_pool_all.txt \\
      --teacher fusion:veto05_r95_cg40 --conf 0.7 --out-dir results/qn_backbone_3way/r2pl/pl_fusion
"""
import argparse
import hashlib
import importlib
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "hf_cache"))

P3W_CKPT = "results/realism_campaign/p3w_lovasz/lovasz050_zoom/best_model.pth"
H, W = 480, 640
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
FORBIDDEN_SUBSTR = ("real_wires_valset", "dlo_real_sample", "fairbench", "rmdlo_probe",
                    "/mBEST/", "putvision_extract", "CVF-DLO", "cables_dataset", "fastdlo")
MC_TEST_LIST = os.path.join(ROOT, "data", "dformer_dataset_movingcables", "test.txt")


def cover_resize(bgr):
    ih, iw = bgr.shape[:2]
    if (ih, iw) == (H, W):
        return bgr
    tgt = W / H
    if iw / ih > tgt:
        nw = int(round(ih * tgt)); x0 = (iw - nw) // 2
        bgr = bgr[:, x0:x0 + nw]
    else:
        nh = int(round(iw / tgt)); y0 = (ih - nh) // 2
        bgr = bgr[y0:y0 + nh]
    interp = cv2.INTER_AREA if bgr.shape[0] > H else cv2.INTER_LINEAR
    return cv2.resize(bgr, (W, H), interpolation=interp)


def read_pool(path):
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    out = []
    for ln in open(full):
        ln = ln.strip()
        if ln:
            out.append(ln.split("\t")[0])
    mc_test = set()
    if os.path.isfile(MC_TEST_LIST):
        mc_test = {os.path.basename(l.strip()) for l in open(MC_TEST_LIST) if l.strip()}
    bad = [p for p in out if any(s in p for s in FORBIDDEN_SUBSTR)
           or ("movingcables" in p and os.path.basename(p) in mc_test)]
    if bad:
        raise SystemExit(f"pool {path} contains HELD-OUT / eval images, e.g. {bad[:3]}")
    return out


def _load_seg(ckpt, device):
    from train_rgb_only_sota import SegFormerSegmenter
    st = torch.load(os.path.join(ROOT, ckpt), map_location="cpu", weights_only=False)
    sd = st.get("model_state_dict", st)
    sd = {k.replace("module.", ""): v for k, v in sd.items() if k != "criterion.weight"}
    nc = int(sd["model.decode_head.classifier.weight"].shape[0])
    m = SegFormerSegmenter(backbone_name="nvidia/mit-b5", num_classes=nc, criterion=None)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{ckpt}: missing={missing[:4]} unexpected={unexpected[:4]} "
                         "(hf5 checkpoint? use src/convert_hf5_segformer_ckpt.py)")
    return m.eval().to(device), nc


# ─────────────────────────── teachers ───────────────────────────

def teacher_p3w_thresh(device, args):
    m, nc = _load_seg(args.p3w_ckpt, device)
    assert nc == 3

    @torch.no_grad()
    def f(x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=not args.fp32):
            lg = m(x)
        pr = lg.float().softmax(1)
        conf, lab = pr.max(1)
        lab = lab.to(torch.uint8)
        lab[conf < args.conf] = 255
        return lab
    return f


def teacher_fusion(device, args, spec):
    import fusion_teacher as FT  # the R2-W teacher (built by another agent; imported, not edited)
    cfg = FT.parse_cfg(spec)
    fm = FT.FusionModel(cfg, device).eval()

    @torch.no_grad()
    def f(x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=not args.fp32):
            lp, lq = fm.p3w(x), fm.qn(x)
            lr = fm.ref(x) if fm.ref is not None else None
        lp, lq = lp.float().softmax(1), lq.float().softmax(1)
        lr = lr.float().softmax(1) if lr is not None else None
        with torch.autocast("cuda", enabled=False):
            return FT.pseudo_label(lp, lq, fm.cfg, conf=args.conf, qn_wire_hi=args.qn_wire_hi,
                                   ref_probs=lr).to(torch.uint8)
    return f, fm.cfg


def build_teacher(args, device):
    t = args.teacher
    if t == "p3w_thresh":
        return teacher_p3w_thresh(device, args), {"teacher": t, "conf": args.conf,
                                                   "p3w": args.p3w_ckpt}
    if t.startswith("fusion:"):
        f, cfg = teacher_fusion(device, args, t.split(":", 1)[1])
        return f, {"teacher": t, "cfg": cfg, "conf": args.conf, "qn_wire_hi": args.qn_wire_hi}
    if t.startswith("module:"):
        _, mod, fn = t.split(":")
        return getattr(importlib.import_module(mod), fn)(device, args), {"teacher": t}
    raise SystemExit(f"unknown --teacher {t}")


# ─────────────────────────── outputs ───────────────────────────

def uid_for(path):
    rel = os.path.relpath(path, ROOT) if os.path.isabs(path) else path
    base = os.path.splitext(rel.replace("/", "__"))[0]
    if len(base) > 150:
        base = base[-120:] + "_" + hashlib.md5(rel.encode()).hexdigest()[:8]
    return base


def overlay(bgr, lab):
    o = bgr.copy()
    o[lab == 1] = (0.35 * o[lab == 1] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    o[lab == 2] = (0.35 * o[lab == 2] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    ign = lab == 255
    hatch = ((np.add.outer(np.arange(H), np.arange(W)) // 4) % 2 == 0)
    o[ign & hatch] = (0.4 * o[ign & hatch] + 0.6 * 160).astype(np.uint8)
    return o


def make_sheet(items, out_path, tile_w=320):
    rows = []
    th = int(tile_w * H / W)
    for bgr, lab, name, st in items:
        a = cv2.resize(bgr, (tile_w, th))
        b = cv2.resize(overlay(bgr, lab), (tile_w, th), interpolation=cv2.INTER_NEAREST)
        cv2.putText(a, name[-40:], (3, 13), 0, 0.38, (0, 255, 255), 1)
        cv2.putText(b, f"w{st['wire']:.3f} c{st['conn']:.3f} ign{st['ignore']:.3f}", (3, 13), 0,
                    0.4, (255, 255, 255), 1)
        rows.append(np.hstack([a, b]))
    while len(rows) % 2:
        rows.append(np.zeros_like(rows[0]))
    grid = np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows), 2)])
    cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


def read_force_bg_list(path):
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    m = {}
    for ln in open(full):
        ln = ln.rstrip("\n")
        if ln.strip():
            a, b = ln.split("\t")[:2]
            m[a] = b
    return m


def load_force_mask(mpath, erode):
    """Binary mask (H,W) bool -> (full, eroded), cover-resized NEAREST to 640x480."""
    mk = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
    if mk is None:
        raise SystemExit(f"unreadable force-bg mask {mpath}")
    mk = (mk > 0).astype(np.uint8)
    if mk.shape != (H, W):
        ih, iw = mk.shape
        if iw / ih > W / H:
            nw = int(round(ih * W / H)); x0 = (iw - nw) // 2; mk = mk[:, x0:x0 + nw]
        else:
            nh = int(round(iw * H / W)); y0 = (ih - nh) // 2; mk = mk[y0:y0 + nh]
        mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
    er = mk
    if erode > 0:
        er = cv2.erode(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1,) * 2))
    return mk.astype(bool), er.astype(bool)


def make_forcebg_sheet(items, out_path, tile_w=320):
    """[rgb + hand contour (yellow=full, cyan=eroded) | teacher | forced]."""
    rows = []
    th = int(tile_w * H / W)
    for bgr, raw, lab, full, er, name, fs in items:
        a = bgr.copy()
        for m, col in ((full, (0, 255, 255)), (er, (255, 255, 0))):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(a, cs, -1, col, 2)
        a = cv2.resize(a, (tile_w, th))
        b = cv2.resize(overlay(bgr, raw), (tile_w, th), interpolation=cv2.INTER_NEAREST)
        c = cv2.resize(overlay(bgr, lab), (tile_w, th), interpolation=cv2.INTER_NEAREST)
        cv2.putText(a, name[-40:], (3, 13), 0, 0.38, (0, 255, 255), 1)
        cv2.putText(b, f"teacher: hand w{fs['hand_wire']:.3f} c{fs['hand_conn']:.3f}", (3, 13), 0,
                    0.4, (255, 255, 255), 1)
        cv2.putText(c, "forced bg (eroded hand)", (3, 13), 0, 0.4, (255, 255, 255), 1)
        rows.append(np.hstack([a, b, c]))
    grid = np.vstack(rows) if len(rows) <= 8 else np.vstack(
        [np.hstack(rows[i:i + 2]) if i + 1 < len(rows) else np.hstack([rows[i], np.zeros_like(rows[i])])
         for i in range(0, len(rows), 2)])
    cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="comma list of pool .txt files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--teacher", default="p3w_thresh")
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--qn-wire-hi", type=float, default=0.5)
    ap.add_argument("--p3w-ckpt", default=P3W_CKPT)
    ap.add_argument("--limit", type=int, default=None, help="random subset of N frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--min-valid", type=float, default=0.0,
                    help="drop frames whose non-ignored fraction is below this from pl_list.txt")
    ap.add_argument("--sheet-n", type=int, default=24)
    ap.add_argument("--device", default=None,
                    help="default cuda:0 if CUDA_VISIBLE_DEVICES is set, else cuda:1")
    ap.add_argument("--force-bg-mask-list", default=None,
                    help="TSV '<rgb>\\t<mask png>': force eroded-mask pixels to bg (0) after the teacher")
    ap.add_argument("--force-bg-erode", type=int, default=3)
    args = ap.parse_args()
    os.chdir(ROOT)
    dev = torch.device(args.device or ("cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES")
                                       else "cuda:1"))
    paths = [p for e in args.pool.split(",") if e.strip() for p in read_pool(e.strip())]
    paths = list(dict.fromkeys(paths))
    if args.limit is not None and args.limit < len(paths):
        paths = sorted(random.Random(args.seed).sample(paths, args.limit))
    os.makedirs(os.path.join(args.out_dir, "labels"), exist_ok=True)
    teacher, tinfo = build_teacher(args, dev)
    print(f"teacher {tinfo}; {len(paths)} frames -> {args.out_dir}", flush=True)
    fbg, fbg_stats = None, {}
    if args.force_bg_mask_list:
        fbg = read_force_bg_list(args.force_bg_mask_list)
        os.makedirs(os.path.join(args.out_dir, "labels_teacher"), exist_ok=True)
        print(f"force-bg: {len(fbg)} masks, erode {args.force_bg_erode}px; "
              f"{sum(p in fbg for p in paths)}/{len(paths)} pool frames covered", flush=True)

    per, lines, t0 = {}, [], time.time()
    tot = np.zeros(4, np.int64)
    for i in range(0, len(paths), args.batch):
        chunk = paths[i:i + args.batch]
        bgrs = []
        for p in chunk:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                raise SystemExit(f"unreadable {p}")
            bgrs.append(cover_resize(im))
        x = torch.from_numpy(np.stack([b[:, :, ::-1] for b in bgrs]).copy()).permute(0, 3, 1, 2)
        x = ((x.float() / 255.0 - MEAN) / STD).to(dev)
        labs = teacher(x).cpu().numpy().astype(np.uint8)
        assert labs.shape == (len(chunk), H, W), labs.shape
        assert np.isin(labs, (0, 1, 2, 255)).all()
        for p, lab in zip(chunk, labs):
            u = uid_for(p)
            lp = os.path.join(args.out_dir, "labels", u + ".png")
            if fbg is not None and p in fbg:
                cv2.imwrite(os.path.join(args.out_dir, "labels_teacher", u + ".png"), lab)
                full, er = load_force_mask(fbg[p], args.force_bg_erode)
                nf, ne = max(int(full.sum()), 1), max(int(er.sum()), 1)
                fbg_stats[p] = {
                    "hand_frac": float(full.mean()), "hand_eroded_frac": float(er.mean()),
                    "hand_wire": float((lab[full] == 1).sum() / nf),
                    "hand_conn": float((lab[full] == 2).sum() / nf),
                    "hand_ignore": float((lab[full] == 255).sum() / nf),
                    "hander_wire": float((lab[er] == 1).sum() / ne),
                    "hander_conn": float((lab[er] == 2).sum() / ne),
                    "nonhand_wire": float((lab[~full] == 1).mean()),
                    "nonhand_conn": float((lab[~full] == 2).mean())}
                lab = lab.copy()
                lab[er] = 0
            cv2.imwrite(lp, lab)
            c = np.array([(lab == 0).sum(), (lab == 1).sum(), (lab == 2).sum(), (lab == 255).sum()])
            tot += c
            st = {"bg": float(c[0] / lab.size), "wire": float(c[1] / lab.size),
                  "conn": float(c[2] / lab.size), "ignore": float(c[3] / lab.size)}
            per[p] = st
            if 1.0 - st["ignore"] >= args.min_valid:
                lines.append(f"{os.path.relpath(p, ROOT) if os.path.isabs(p) else p}\t"
                             f"{os.path.relpath(lp, ROOT)}\n")
        if (i // args.batch) % 50 == 0:
            print(f"  {i + len(chunk)}/{len(paths)}  {time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(args.out_dir, "pl_list.txt"), "w") as f:
        f.writelines(lines)
    fr = tot / max(tot.sum(), 1)
    ign = np.array([s["ignore"] for s in per.values()])
    agg = {"n_frames": len(per), "n_in_pl_list": len(lines),
           "pixel_frac": {"bg": fr[0], "wire": fr[1], "conn": fr[2], "ignore": fr[3]},
           "frac_frames_with_conn": float(np.mean([s["conn"] > 0 for s in per.values()])),
           "frac_frames_with_wire": float(np.mean([s["wire"] > 0 for s in per.values()])),
           "ignore_frac_p10_p50_p90": [float(np.percentile(ign, q)) for q in (10, 50, 90)],
           "seconds": time.time() - t0}
    by_src = {}
    for p, s in per.items():
        k = p.split("/")[1] if p.startswith("data/") else p.split("/")[0]
        d = by_src.setdefault(k, {"n": 0, "wire": 0.0, "conn": 0.0, "ignore": 0.0})
        d["n"] += 1
        for kk in ("wire", "conn", "ignore"):
            d[kk] += s[kk]
    for d in by_src.values():
        for kk in ("wire", "conn", "ignore"):
            d[kk] /= d["n"]
    agg["by_source_mean"] = by_src
    out = {"config": {**tinfo, "pool": args.pool, "limit": args.limit, "seed": args.seed,
                      "min_valid": args.min_valid},
           "aggregate": agg, "per_frame": per}
    if fbg is not None:
        out["config"].update(force_bg_mask_list=args.force_bg_mask_list,
                             force_bg_erode=args.force_bg_erode)
        v = list(fbg_stats.values())
        fa = {"n_forced_frames": len(v)}
        if v:
            for k in v[0]:
                fa["mean_" + k] = float(np.mean([s[k] for s in v]))
            hf = np.array([s["hand_wire"] + s["hand_conn"] for s in v])
            fa["hand_fg_frac_p50_p90_max"] = [float(np.percentile(hf, 50)),
                                              float(np.percentile(hf, 90)), float(hf.max())]
            fa["frac_frames_hand_fg_gt_1pct"] = float((hf > 0.01).mean())
            fa["frac_frames_hand_fg_gt_5pct"] = float((hf > 0.05).mean())
        out["force_bg"] = {"aggregate": fa, "per_frame": fbg_stats}
        agg["force_bg"] = fa
    json.dump(out, open(os.path.join(args.out_dir, "stats.json"), "w"), indent=1, default=float)
    print(json.dumps(agg, indent=1, default=float))

    if args.sheet_n > 0:
        pick = sorted(random.Random(args.seed + 1).sample(list(per), min(args.sheet_n, len(per))))
        items = []
        for p in pick:
            bgr = cover_resize(cv2.imread(p, cv2.IMREAD_COLOR))
            lab = cv2.imread(os.path.join(args.out_dir, "labels", uid_for(p) + ".png"),
                             cv2.IMREAD_UNCHANGED)
            items.append((bgr, lab, os.path.relpath(p, ROOT) if os.path.isabs(p) else p, per[p]))
        make_sheet(items, os.path.join(args.out_dir, "sheet.jpg"))
        print("sheet ->", os.path.join(args.out_dir, "sheet.jpg"))
        fitems = []
        for p, (bgr, lab, name, _) in zip(pick, items):
            if fbg is None or p not in fbg:
                continue
            raw = cv2.imread(os.path.join(args.out_dir, "labels_teacher", uid_for(p) + ".png"),
                             cv2.IMREAD_UNCHANGED)
            full, er = load_force_mask(fbg[p], args.force_bg_erode)
            fitems.append((bgr, raw, lab, full, er, name, fbg_stats[p]))
        if fitems:
            make_forcebg_sheet(fitems, os.path.join(args.out_dir, "sheet_forcebg.jpg"))
            print("force-bg sheet ->", os.path.join(args.out_dir, "sheet_forcebg.jpg"))


if __name__ == "__main__":
    main()
