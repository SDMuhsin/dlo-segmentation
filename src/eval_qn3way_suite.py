#!/usr/bin/env python
"""QN-backbone 3-way evaluation suite (one command, one row per model).

Given one or more NAME=CKPT (2-class bg/DLO or 3-class bg/wire/connector; the class count is
read from the classifier weight shape; the key-converted QN ckpt is accepted as-is), compute:

 1. REAL VIDEOS (data/dlo_real_sample_{1..4}.mp4, unannotated): mean predicted wire % and
    connector % per clip + mean over clips; connector connected-component (8-conn) bbox
    max-dimension p50/p90/p99/max and blob count; teacher (QN) agreement on the same frames.
    Preprocessing = src/infer_video_rgb_only.preprocess (centre-crop 4:3 -> 640x480).
    SUBSAMPLE (identical for every model):
       coverage + teacher agreement: every VIDEO_STRIDE-th frame (0, 3, 6, ...);
       blob stats: BLOB_FRAMES_PER_CLIP evenly spaced frames per clip (np.linspace), 100 total,
       components with area >= BLOB_MIN_AREA=50 px
       (same protocol size as results/p443_diagnosis/REPORT.md).
    Also dumps the 12 recon frame indices as [camera | P3W | QN | model] PNGs
    (wire=green, connector=magenta) once the p3w/qn reference masks are cached.
 2. MOVINGCABLES TEST (1172 real frames, GT label 4 = cable): IoU/P/R of predicted WIRE (class 1;
    = DLO for 2-class) and of WIRE u CONNECTOR vs GT cable; fraction of GT-cable px predicted connector.
 3. data/real_wires_valset (62 imgs) pooled IoU, wire-only (class1) and wire u conn (fg), via
    eval_real_wires_valset.eval_one. CONTAMINATED for QN lineage -> DRIFT meter only.
 4. SYNTHETIC 3-way val (data/dformer_dataset_3way_connscale3/test.txt, 1080): per-class IoU + mIoU
    (pooled confusion); 2-class models get collapsed IoU(DLO) (wire+conn GT -> DLO).
 5. TEACHER AGREEMENT on MovingCables test: IoU(model wire, QN DLO), IoU(model wire u conn, QN DLO),
    fraction of model-connector px where QN says bg. (QN masks cached once.)
 6. PROBES src/probe_p44_wire_cue.py / src/probe_p44_texture_cue.py (subprocess, --auto-num-classes,
    --extra-model), headline numbers only (--no-probes to skip).

All inference fp32, argmax, batch 8 on ONE GPU (--gpu, default 1; set via CUDA_VISIBLE_DEVICES).

Usage:
  ./env/bin/python src/eval_qn3way_suite.py p3w=results/.../best_model.pth \
      qn=results/qn_backbone_3way/recon/qn_best_model_hf4.pth [--out results/qn_backbone_3way/suite]
Reference names: the QN teacher must be available as --qn-ckpt (default the converted file);
frame composites use the cached masks of models named by --ref-p3w / --ref-qn.
"""
import argparse
import glob
import json
import os
import subprocess
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
from train_rgb_only_sota import SegFormerSegmenter  # noqa: E402
import eval_real_wires_valset as ev  # noqa: E402

DEV = torch.device("cuda:0")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
VIDEOS = [os.path.join(ROOT, f"data/dlo_real_sample_{i}.mp4") for i in (1, 2, 3, 4)]
VIDEO_STRIDE = 3
BLOB_FRAMES_PER_CLIP = 25
# blob = 8-connected component of predicted connector with area >= BLOB_MIN_AREA px; bbox max-dim.
# 50 px reproduces the p443 REPORT baseline protocol (P3W: 283 blobs, p90 82 px vs documented 300 / 82);
# conn_blobs_all keeps every component (incl. 1-px speckle).
BLOB_MIN_AREA = 50
DUMP_FRAMES = {1: [256, 640, 1024], 2: [196, 491, 786], 3: [150, 375, 600], 4: [85, 214, 343]}
MC_DIR = os.path.join(ROOT, "data/dformer_dataset_movingcables")
SYN_DIR = os.path.join(ROOT, "data/dformer_dataset_3way_connscale3")
QN_DEFAULT = "results/qn_backbone_3way/recon/qn_best_model_hf4.pth"
BS = 8


# ---------------------------------------------------------------- model
def load(ckpt):
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


@torch.no_grad()
def predict_batch(m, bgrs):
    x = torch.from_numpy(np.stack([b[:, :, ::-1] for b in bgrs]).copy()).permute(0, 3, 1, 2)
    x = (x.float() / 255.0 - MEAN) / STD
    return m(x.to(DEV)).argmax(1).to(torch.uint8).cpu().numpy()


def run_batched(m, frames):
    out = []
    for i in range(0, len(frames), BS):
        out.append(predict_batch(m, frames[i:i + BS]))
    return np.concatenate(out) if out else np.zeros((0, 480, 640), np.uint8)


def prf(tp, fp, fn):
    return {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1)}


# ---------------------------------------------------------------- data (read once)
def read_video_frames(path, wanted):
    wanted = sorted(set(wanted))
    cap = cv2.VideoCapture(path)
    out, fi, wi = {}, 0, 0
    while wi < len(wanted):
        ok, f = cap.read()
        if not ok:
            break
        if fi == wanted[wi]:
            out[fi] = preprocess(f)
            wi += 1
        fi += 1
    cap.release()
    return out


def video_plan():
    plan = {}
    for vi, v in enumerate(VIDEOS, 1):
        n = int(cv2.VideoCapture(v).get(cv2.CAP_PROP_FRAME_COUNT))
        cov = list(range(0, n, VIDEO_STRIDE))
        blob = np.linspace(0, n - 1, BLOB_FRAMES_PER_CLIP, dtype=int).tolist()
        plan[vi] = {"n_total": n, "cov": cov, "blob": blob, "dump": DUMP_FRAMES[vi]}
    return plan


def mc_names():
    return [os.path.splitext(os.path.basename(l.split()[0]))[0]
            for l in open(os.path.join(MC_DIR, "test.txt")) if l.strip()]


def syn_names():
    return [os.path.splitext(os.path.basename(l.split()[0]))[0]
            for l in open(os.path.join(SYN_DIR, "test.txt")) if l.strip()]


# ---------------------------------------------------------------- QN teacher cache
def teacher_cache(cache_dir, qn_ckpt, plan):
    """Binary QN DLO masks on MC-test and on the video coverage frames (packbits)."""
    p = os.path.join(cache_dir, "qn_teacher_masks.npz")
    if os.path.exists(p):
        z = np.load(p)
        return {k: np.unpackbits(z[k], axis=-1)[..., :640].astype(bool) for k in z.files}
    print("[teacher] computing QN masks (one-off)", flush=True)
    m, nc = load(qn_ckpt)
    assert nc == 2, "QN teacher must be 2-class"
    out = {}
    names = mc_names()
    preds = []
    for i in range(0, len(names), BS):
        fr = [cv2.imread(os.path.join(MC_DIR, "RGB", n + ".png")) for n in names[i:i + BS]]
        preds.append(predict_batch(m, fr) >= 1)
    out["mc"] = np.concatenate(preds)
    for vi, v in enumerate(VIDEOS, 1):
        fr = read_video_frames(v, plan[vi]["cov"])
        out[f"vid{vi}"] = run_batched(m, [fr[k] for k in plan[vi]["cov"]]) >= 1
    del m
    torch.cuda.empty_cache()
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(p, **{k: np.packbits(v, axis=-1) for k, v in out.items()})
    return out


# ---------------------------------------------------------------- 1. videos
def blob_dims(masks):
    dims = []
    for mk in masks:
        n, _, stats, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8), connectivity=8)
        for j in range(1, n):
            dims.append((int(max(stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT])),
                         int(stats[j, cv2.CC_STAT_AREA])))
    return dims


def eval_videos(m, nc, plan, teacher, mdir):
    res = {"protocol": f"coverage/agreement on every {VIDEO_STRIDE}th frame; blobs on "
                       f"{BLOB_FRAMES_PER_CLIP} linspace frames/clip", "clips": {}}
    all_dims = []
    dumps = {}
    ag = dict(i_w=0, u_w=0, i_f=0, u_f=0, con=0, con_qbg=0)
    for vi, v in enumerate(VIDEOS, 1):
        pl = plan[vi]
        fr = read_video_frames(v, pl["cov"] + pl["blob"] + pl["dump"])
        pc = run_batched(m, [fr[k] for k in pl["cov"]])
        wire = (pc == 1).mean(axis=(1, 2)) * 100
        con = (pc == 2).mean(axis=(1, 2)) * 100 if nc == 3 else np.full(len(pc), np.nan)
        t = teacher[f"vid{vi}"]
        pw, pf = pc == 1, pc >= 1
        ag["i_w"] += int((pw & t).sum()); ag["u_w"] += int((pw | t).sum())
        ag["i_f"] += int((pf & t).sum()); ag["u_f"] += int((pf | t).sum())
        if nc == 3:
            ag["con"] += int((pc == 2).sum()); ag["con_qbg"] += int(((pc == 2) & ~t).sum())
        clip = {"wire_pct": float(wire.mean()), "conn_pct": float(np.nanmean(con)) if nc == 3 else None,
                "n_frames_used": len(pc), "n_frames_total": pl["n_total"]}
        if nc == 3:
            pb = run_batched(m, [fr[k] for k in pl["blob"]])
            d = blob_dims(pb == 2)
            all_dims += d
            clip["n_blobs"] = sum(1 for _, ar in d if ar >= BLOB_MIN_AREA)
        pd = run_batched(m, [fr[k] for k in pl["dump"]])
        for k, p in zip(pl["dump"], pd):
            dumps[f"sample{vi}_f{k:04d}"] = (fr[k], p)
        res["clips"][f"sample_{vi}"] = clip
    cl = res["clips"].values()
    res["mean_wire_pct"] = float(np.mean([c["wire_pct"] for c in cl]))
    res["mean_conn_pct"] = float(np.mean([c["conn_pct"] for c in cl])) if nc == 3 else None
    if nc == 3:
        def st(ds):
            a = np.array(ds) if ds else np.array([0])
            return {"n": len(ds), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
                    "p99": float(np.percentile(a, 99)), "max": int(a.max())}
        res["conn_blobs"] = st([dm for dm, ar in all_dims if ar >= BLOB_MIN_AREA])
        res["conn_blobs"]["min_area_px"] = BLOB_MIN_AREA
        res["conn_blobs_all"] = st([dm for dm, _ in all_dims])
    res["teacher_agree_iou_wire"] = ag["i_w"] / max(ag["u_w"], 1)
    res["teacher_agree_iou_fg"] = ag["i_f"] / max(ag["u_f"], 1)
    res["conn_where_qn_bg"] = (ag["con_qbg"] / ag["con"]) if (nc == 3 and ag["con"]) else None
    # cache the dump masks for composites
    np.savez_compressed(os.path.join(mdir, "dump_masks.npz"), **{k: v[1] for k, v in dumps.items()})
    return res, {k: v[0] for k, v in dumps.items()}


# ---------------------------------------------------------------- composites
def overlay(bgr, mask):
    out = bgr.copy()
    for cls, col in ((1, (0, 255, 0)), (2, (255, 0, 255))):
        sel = mask == cls
        out[sel] = (out[sel] * 0.5 + np.array(col) * 0.5).astype(np.uint8)
    return out


def bar(img, text):
    b = np.full((28, img.shape[1], 3), 25, np.uint8)
    cv2.putText(b, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([b, img])


def write_composites(name, frames, out_root, ref_p3w, ref_qn):
    refs = []
    for rn in (ref_p3w, ref_qn):
        p = os.path.join(out_root, rn, "dump_masks.npz")
        refs.append((rn, np.load(p) if os.path.exists(p) else None))
    own = np.load(os.path.join(out_root, name, "dump_masks.npz"))
    fdir = os.path.join(out_root, name, "frames")
    os.makedirs(fdir, exist_ok=True)
    for k, bgr in frames.items():
        panels = [bar(bgr, "camera")]
        for rn, z in refs:
            if z is not None and rn != name:
                panels.append(bar(overlay(bgr, z[k]), rn))
        panels.append(bar(overlay(bgr, own[k]), f"{name}  (wire=green conn=magenta)"))
        sep = np.full((panels[0].shape[0], 4, 3), 200, np.uint8)
        row = panels[0]
        for p in panels[1:]:
            row = np.hstack([row, sep, p])
        cv2.imwrite(os.path.join(fdir, f"{k}.png"), row)


# ---------------------------------------------------------------- 2+5. MovingCables
def eval_mc(m, nc, teacher):
    names = mc_names()
    c = dict(tw=0, fw=0, nw=0, tf=0, ff=0, nf=0, gt=0, gt_con=0,
             ti_w=0, tu_w=0, ti_f=0, tu_f=0, con=0, con_qbg=0)
    qn = teacher["mc"]
    for i in range(0, len(names), BS):
        chunk = names[i:i + BS]
        fr = [cv2.imread(os.path.join(MC_DIR, "RGB", n + ".png")) for n in chunk]
        gt = np.stack([cv2.imread(os.path.join(MC_DIR, "Label", n + ".png"), -1) == 4 for n in chunk])
        p = predict_batch(m, fr)
        t = qn[i:i + len(chunk)]
        pw, pf = p == 1, p >= 1
        c["tw"] += int((pw & gt).sum()); c["fw"] += int((pw & ~gt).sum()); c["nw"] += int((~pw & gt).sum())
        c["tf"] += int((pf & gt).sum()); c["ff"] += int((pf & ~gt).sum()); c["nf"] += int((~pf & gt).sum())
        c["gt"] += int(gt.sum()); c["gt_con"] += int(((p == 2) & gt).sum())
        c["ti_w"] += int((pw & t).sum()); c["tu_w"] += int((pw | t).sum())
        c["ti_f"] += int((pf & t).sum()); c["tu_f"] += int((pf | t).sum())
        c["con"] += int((p == 2).sum()); c["con_qbg"] += int(((p == 2) & ~t).sum())
    r = {"n_images": len(names), "wire": prf(c["tw"], c["fw"], c["nw"]),
         "wire_or_conn": prf(c["tf"], c["ff"], c["nf"]),
         "gt_cable_pred_conn_frac": c["gt_con"] / max(c["gt"], 1),
         "teacher_agree_iou_wire": c["ti_w"] / max(c["tu_w"], 1),
         "teacher_agree_iou_fg": c["ti_f"] / max(c["tu_f"], 1),
         "conn_pct_of_px": 100 * c["con"] / (len(names) * 480 * 640),
         "conn_where_qn_bg": (c["con_qbg"] / c["con"]) if c["con"] else None}
    return r


# ---------------------------------------------------------------- 3. valset
def eval_valset(m, nc):
    imgs = sorted(glob.glob(os.path.join(ROOT, "data/real_wires_valset/imgs/*.jpg")))
    out = {}
    rules = ("fg", "class1") if nc == 3 else ("fg",)
    for rule in rules:
        rows = []
        for ip in imgs:
            bn = os.path.splitext(os.path.basename(ip))[0]
            mp = os.path.join(ROOT, "data/real_wires_valset/masks", bn + ".jpg")
            rows.append(ev.eval_one(m, DEV, ip, mp, wire_rule=rule))
        a = ev.agg(rows)
        out[rule] = {k: a[k] for k in ("pooled_iou", "precision", "recall", "n_images")}
    if nc == 2:
        out["class1"] = out["fg"]
    return {"wire": out["class1"], "wire_or_conn": out["fg"],
            "note": "CONTAMINATED for QN lineage (EWD/Zanella overlap) - drift meter only"}


# ---------------------------------------------------------------- 4. synthetic val
def eval_synth(m, nc):
    names = syn_names()
    cm = np.zeros((3, 3), np.int64)  # gt x pred (pred collapsed to 3 cols; 2-class never predicts 2)
    for i in range(0, len(names), BS):
        chunk = names[i:i + BS]
        fr = [cv2.imread(os.path.join(SYN_DIR, "RGB", n + ".png")) for n in chunk]
        lab = np.stack([cv2.imread(os.path.join(SYN_DIR, "Label", n + ".png"), -1) for n in chunk])
        p = predict_batch(m, fr)
        v = lab != 255
        cm += np.bincount(lab[v].astype(np.int64) * 3 + p[v], minlength=9).reshape(3, 3)
    r = {"n_images": len(names)}
    if nc == 3:
        ious = []
        for k, nm in enumerate(("bg", "wire", "connector")):
            tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k, :].sum() - tp
            r[f"iou_{nm}"] = float(tp / max(tp + fp + fn, 1)); ious.append(r[f"iou_{nm}"])
        r["miou"] = float(np.mean(ious))
    gd = cm[1:, :].sum(1)  # gt dlo rows
    tp = int(cm[1:, 1:].sum()); fp = int(cm[0, 1:].sum()); fn = int(cm[1:, 0].sum())
    r["iou_dlo_collapsed"] = tp / max(tp + fp + fn, 1)
    return r


# ---------------------------------------------------------------- 6. probes
def run_probes(name, ckpt, mdir):
    env = dict(os.environ)
    res = {}
    for key, script in (("wire_cue", "probe_p44_wire_cue.py"), ("texture_cue", "probe_p44_texture_cue.py")):
        od = os.path.join(mdir, f"probe_{key}")
        sj = os.path.join(od, "summary.json")
        if not os.path.exists(sj):
            cmd = [os.path.join(ROOT, "env/bin/python"), os.path.join(SRC, script), "--device", "cuda:0",
                   "--auto-num-classes", "--extra-model", f"{name}={ckpt}", "--out", od]
            if key == "wire_cue":
                cmd += ["--dump", "4"]
            r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
            open(os.path.join(mdir, f"probe_{key}.log"), "w").write(r.stdout + r.stderr)
            if r.returncode:
                print(f"[warn] probe {key} failed: {r.stderr[-800:]}")
                continue
        res[key] = json.load(open(sj))[name]
    return res


# ---------------------------------------------------------------- report
def fmt(x, pct=False, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.2f}" if pct else f"{x:.{d}f}"


HDR = ("| model | vid wire% | vid conn% | blobs n / p50 / p90 / p99 / max | vid QN-agree IoU(w / w+c) | "
       "vid conn@QN-bg | MC IoU wire (P/R) | MC IoU w+c | MC cable->conn | MC QN-agree IoU(w / w+c) | "
       "MC conn@QN-bg | valset* w / w+c | synth mIoU (wire/con) | synth IoU(DLO) | "
       "probe real-cable W/C | probe synth-wire W | probe texture real C/W |")


def md_row(name, R):
    v, mc, vs, sy, pr = R["videos"], R["movingcables"], R["valset"], R["synth"], R.get("probes", {})
    b = v.get("conn_blobs")
    blobs = f"{b['n']} / {b['p50']:.0f} / {b['p90']:.0f} / {b['p99']:.0f} / {b['max']}" if b else "-"
    syn = (f"{sy['miou']:.4f} ({sy['iou_wire']:.3f}/{sy['iou_connector']:.3f})" if "miou" in sy else "-")
    wc, tc = pr.get("wire_cue"), pr.get("texture_cue")
    pw = f"{100*wc['real_cable']['wire']:.1f} / {100*wc['real_cable']['conn']:.1f}" if wc else "-"
    ps = f"{100*wc['synth_wire']['wire']:.1f}" if wc else "-"
    pt = f"{100*tc['real']:.1f} / {100*tc.get('real_wire', float('nan')):.1f}" if tc else "-"
    return (f"| {name} | {fmt(v['mean_wire_pct'], True)} | {fmt(v['mean_conn_pct'], True)} | {blobs} | "
            f"{fmt(v['teacher_agree_iou_wire'])} / {fmt(v['teacher_agree_iou_fg'])} | {fmt(v['conn_where_qn_bg'])} | "
            f"{mc['wire']['iou']:.3f} ({mc['wire']['precision']:.3f}/{mc['wire']['recall']:.3f}) | "
            f"{mc['wire_or_conn']['iou']:.3f} | {mc['gt_cable_pred_conn_frac']:.3f} | "
            f"{mc['teacher_agree_iou_wire']:.3f} / {mc['teacher_agree_iou_fg']:.3f} | {fmt(mc['conn_where_qn_bg'])} | "
            f"{vs['wire']['pooled_iou']:.4f} / {vs['wire_or_conn']['pooled_iou']:.4f} | {syn} | "
            f"{sy['iou_dlo_collapsed']:.3f} | {pw} | {ps} | {pt} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", metavar="NAME=CKPT")
    ap.add_argument("--out", default="results/qn_backbone_3way/suite")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--qn-ckpt", default=QN_DEFAULT)
    ap.add_argument("--ref-p3w", default="p3w")
    ap.add_argument("--ref-qn", default="qn")
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute even if metrics.json exists")
    a = ap.parse_args()
    out_root = os.path.join(ROOT, a.out)
    os.makedirs(out_root, exist_ok=True)
    plan = video_plan()
    teacher = teacher_cache(os.path.join(out_root, "_cache"), a.qn_ckpt, plan)
    for spec in a.models:
        name, _, ckpt = spec.partition("=")
        mdir = os.path.join(out_root, name)
        mj = os.path.join(mdir, "metrics.json")
        os.makedirs(mdir, exist_ok=True)
        if os.path.exists(mj) and not a.force:
            R = json.load(open(mj))
            print(f"[skip] {name} (cached)")
        else:
            t0 = time.time()
            m, nc = load(ckpt)
            print(f"[{name}] {ckpt} num_classes={nc}", flush=True)
            R = {"name": name, "ckpt": ckpt, "num_classes": nc}
            R["videos"], frames = eval_videos(m, nc, plan, teacher, mdir)
            print(f"  videos wire {R['videos']['mean_wire_pct']:.2f} conn {R['videos']['mean_conn_pct']}", flush=True)
            R["movingcables"] = eval_mc(m, nc, teacher)
            print(f"  MC IoU(wire) {R['movingcables']['wire']['iou']:.3f}", flush=True)
            R["valset"] = eval_valset(m, nc)
            print(f"  valset {R['valset']['wire_or_conn']['pooled_iou']:.4f}", flush=True)
            R["synth"] = eval_synth(m, nc)
            print(f"  synth {R['synth']}", flush=True)
            del m
            torch.cuda.empty_cache()
            if not a.no_probes:
                R["probes"] = run_probes(name, ckpt, mdir)
            R["wall_s"] = time.time() - t0
            json.dump(R, open(mj, "w"), indent=2)
            fc = os.path.join(out_root, "_cache", "dump_frames.npz")
            if not os.path.exists(fc):
                np.savez_compressed(fc, **frames)
        fc = os.path.join(out_root, "_cache", "dump_frames.npz")
        if os.path.exists(fc) and os.path.exists(os.path.join(mdir, "dump_masks.npz")):
            z = np.load(fc)
            write_composites(name, {k: z[k] for k in z.files}, out_root, a.ref_p3w, a.ref_qn)
        row = md_row(name, R)
        open(os.path.join(mdir, "row.md"), "w").write(HDR + "\n" + row + "\n")
        print(HDR); print(row, flush=True)


if __name__ == "__main__":
    main()
