"""Real-video connector comparison sheet (decision artifact for the QN-backbone 3-way campaign).

For a fixed set of frames from the 4 real PI videos (data/dlo_real_sample_{1..4}.mp4, same
preprocessing as src/eval_qn3way_suite.py: centre-crop 4:3 -> 640x480 BGR), render one PNG per
frame: [camera | model_1 | model_2 | ...] with wire = green, connector = red (+ red contour so small
connector blobs stay visible), each panel labelled with the model name and its wire% / conn% on
that frame.  Predictions are cached per model (keyed on ckpt path + mtime + size) in
<out>/_cache/, so re-running with an extra NAME=CKPT only computes the new model.

Usage:
  ./env/bin/python src/make_connector_sheet.py p3w=... a0_ep5=... wenc_a0e5_a07=... a1_ep10=... [a2_ep30=...]
      [--frames 1:640,4:214,...] [--out results/qn_backbone_3way/sheet] [--gpu 1]
  ./env/bin/python src/make_connector_sheet.py --scan p3w=... a1_ep10=...   # candidate finder:
      every 15th frame of each clip, per-model connector % -> <out>/scan.json (+ top list on stdout)

Outputs: <out>/s{clip}_f{frame:04d}.png, <out>/index.md (auto table of per-frame per-model wire% /
conn% / connector blob count, followed by the hand-written <out>/assessment.md if it exists).
"""
import argparse
import hashlib
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+", help="NAME=CKPT (order = panel order)")
ap.add_argument("--frames", default=None, help="comma list clip:frame (default: FRAMES below)")
ap.add_argument("--out", default="results/qn_backbone_3way/sheet")
ap.add_argument("--gpu", default="1")
ap.add_argument("--scan", action="store_true")
ap.add_argument("--scan-stride", type=int, default=15)
A = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = A.gpu  # never touch any other GPU

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "hf_cache"))
from infer_video_rgb_only import preprocess  # noqa: E402
from train_rgb_only_sota import SegFormerSegmenter  # noqa: E402

VIDEOS = {i: os.path.join(ROOT, f"data/dlo_real_sample_{i}.mp4") for i in (1, 2, 3, 4)}
# Chosen by looking at the scan (frames where any model predicts connector) + known cases.
# (clip, frame, note)  -- the note is the ground-truth-by-eye description of what is in view.
FRAMES = [
    (1, 105, "close-up black USB-A plug + cable knot"),
    (1, 640, "USB-A plug + knot on paper (known)"),
    (1, 1110, "USB-A plug + barrel charger plug (bottom) + knot"),
    (1, 1170, "USB-A plug + barrel charger plug (bottom) + knot"),
    (2, 105, "black plug in left hand (edge) + fingertips"),
    (2, 210, "USB-A + micro-USB plugs held in hands"),
    (2, 555, "white plug/strain relief in hand"),
    (2, 645, "white 2-prong mains plug in hand"),
    (2, 720, "white mains plug held in both hands"),
    (2, 780, "white C7 connector head + strain relief, hands"),
    (2, 300, "CLUTTER: black cable + fingers, no connector"),
    (2, 360, "CLUTTER: open hand in front of black cable"),
    (3, 585, "CLUTTER: fingers close-up, cable behind"),
    (3, 630, "WIRE: blurred thick close-up cable + fingers"),
    (4, 90, "CLUTTER: keyboard + mic stand, thin cable"),
    (4, 135, "USB-C plug into laptop + laptop/chair clutter"),
    (4, 214, "CLUTTER: cluttered desk, tangled cable (known)"),
    (4, 300, "black puck device at cable end + desk clutter"),
]
DEV = torch.device("cuda:0")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
BLOB_MIN_AREA = 50


def load(ckpt):
    st = torch.load(os.path.join(ROOT, ckpt), map_location="cpu", weights_only=False)
    sd = st.get("model_state_dict", st)
    sd = {k.replace("module.", ""): v for k, v in sd.items() if k != "criterion.weight"}
    nc = int(sd["model.decode_head.classifier.weight"].shape[0])
    m = SegFormerSegmenter(backbone_name="nvidia/mit-b5", num_classes=nc, criterion=None)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{ckpt}: missing={missing[:4]} unexpected={unexpected[:4]}")
    return m.eval().to(DEV)


@torch.no_grad()
def predict(m, bgrs, bs=4):
    out = []
    for i in range(0, len(bgrs), bs):
        x = torch.from_numpy(np.stack([b[:, :, ::-1] for b in bgrs[i:i + bs]]).copy()).permute(0, 3, 1, 2)
        x = (x.float() / 255.0 - MEAN) / STD
        with torch.autocast("cuda", dtype=torch.float16):
            lg = m(x.to(DEV))
        out.append(lg.float().argmax(1).to(torch.uint8).cpu().numpy())
    return np.concatenate(out)


def read_frames(clip, wanted):
    wanted = sorted(set(wanted))
    cap = cv2.VideoCapture(VIDEOS[clip])
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


def ckey(ckpt, frames):
    p = os.path.join(ROOT, ckpt)
    s = f"{os.path.abspath(p)}|{os.path.getmtime(p)}|{os.path.getsize(p)}|{sorted(frames)}"
    return hashlib.md5(s.encode()).hexdigest()[:12]


def get_masks(name, ckpt, frames_by_key, cache_dir):
    """frames_by_key: {'s1_f0640': bgr}. Returns {key: mask}; cached per model."""
    os.makedirs(cache_dir, exist_ok=True)
    cp = os.path.join(cache_dir, f"{name}_{ckey(ckpt, frames_by_key)}.npz")
    if os.path.exists(cp):
        z = np.load(cp)
        return {k: z[k] for k in z.files}
    print(f"  predicting {name} ({ckpt}) on {len(frames_by_key)} frames", flush=True)
    m = load(ckpt)
    keys = sorted(frames_by_key)
    pr = predict(m, [frames_by_key[k] for k in keys])
    del m
    torch.cuda.empty_cache()
    res = {k: pr[i] for i, k in enumerate(keys)}
    np.savez_compressed(cp, **res)
    return res


def n_blobs(mask):
    n, _, st, _ = cv2.connectedComponentsWithStats((mask == 2).astype(np.uint8), connectivity=8)
    return int(sum(st[i, cv2.CC_STAT_AREA] >= BLOB_MIN_AREA for i in range(1, n)))


def overlay(bgr, mask):
    out = bgr.copy()
    for cls, col in ((1, (0, 220, 0)), (2, (0, 0, 255))):
        sel = mask == cls
        out[sel] = (out[sel] * 0.45 + np.array(col) * 0.55).astype(np.uint8)
    cm = (mask == 2).astype(np.uint8)
    if cm.any():
        cs, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, (0, 0, 255), 2)
    return out


def bar(img, text, sub=""):
    b = np.full((46, img.shape[1], 3), 25, np.uint8)
    cv2.putText(b, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(b, sub, (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([b, img])


def grid(panels, ncol=3):
    h, w = panels[0].shape[:2]
    blank = np.full((h, w, 3), 255, np.uint8)
    rows = []
    for r in range(0, len(panels), ncol):
        row = panels[r:r + ncol] + [blank] * (ncol - len(panels[r:r + ncol]))
        rows.append(np.hstack([np.pad(p, ((4, 4), (4, 4), (0, 0)), constant_values=255) for p in row]))
    return np.vstack(rows)


def main():
    models = [s.split("=", 1) for s in A.models]
    out = os.path.join(ROOT, A.out)
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, "_cache")

    if A.scan:
        frames = {}
        for c in VIDEOS:
            n = int(cv2.VideoCapture(VIDEOS[c]).get(cv2.CAP_PROP_FRAME_COUNT))
            for f, bgr in read_frames(c, range(0, n, A.scan_stride)).items():
                frames[f"s{c}_f{f:04d}"] = bgr
        res = {}
        for name, ck in models:
            ms = get_masks(name, ck, frames, cache)
            res[name] = {k: {"wire": float((v == 1).mean() * 100), "conn": float((v == 2).mean() * 100),
                             "blobs": n_blobs(v)} for k, v in ms.items()}
        json.dump(res, open(os.path.join(out, "scan.json"), "w"), indent=1)
        keys = sorted(frames)
        tot = {k: sum(res[n][k]["conn"] for n, _ in models) for k in keys}
        for k in sorted(keys, key=lambda k: -tot[k])[:60]:
            print(k, " ".join(f"{n}={res[n][k]['conn']:.2f}/{res[n][k]['blobs']}" for n, _ in models))
        return

    if A.frames:
        sel = [(int(s.split(":")[0]), int(s.split(":")[1]), "") for s in A.frames.split(",")]
    else:
        sel = FRAMES
    frames = {}
    for c in sorted({c for c, _, _ in sel}):
        got = read_frames(c, [f for cc, f, _ in sel if cc == c])
        for f, bgr in got.items():
            frames[f"s{c}_f{f:04d}"] = bgr
    notes = {f"s{c}_f{f:04d}": n for c, f, n in sel}
    masks = {name: get_masks(name, ck, frames, cache) for name, ck in models}
    stats = {}
    for k in [f"s{c}_f{f:04d}" for c, f, _ in sel]:
        bgr = frames[k]
        panels = [bar(bgr, f"camera  {k}", notes.get(k, "")[:60])]
        for name, _ in models:
            mk = masks[name][k]
            w, cn, nb = (mk == 1).mean() * 100, (mk == 2).mean() * 100, n_blobs(mk)
            stats.setdefault(k, {})[name] = (w, cn, nb)
            panels.append(bar(overlay(bgr, mk), name, f"wire {w:.2f}%  conn {cn:.2f}%  blobs {nb}"))
        cv2.imwrite(os.path.join(out, f"{k}.png"), grid(panels))
    names = [n for n, _ in models]
    L = ["# Real-video connector comparison sheet", "",
         "Generated by `src/make_connector_sheet.py` (wire = green, connector = red + red contour). "
         "Frames: 4 real PI videos, suite preprocessing (centre-crop 4:3 -> 640x480). "
         f"blobs = 8-connected connector components >= {BLOB_MIN_AREA} px. No GT: all judgements are by eye.", "",
         "Models: " + ", ".join(f"`{n}` = `{c}`" for n, c in models), "",
         "## Per-frame numbers (wire% / conn% / conn blobs)", "",
         "| frame | content | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for k, d in stats.items():
        L.append(f"| [{k}]({k}.png) | {notes.get(k, '')} | " +
                 " | ".join(f"{d[n][0]:.1f} / {d[n][1]:.2f} / {d[n][2]}" for n in names) + " |")
    ap_ = os.path.join(out, "assessment.md")
    if os.path.exists(ap_):
        L += ["", open(ap_).read()]
    open(os.path.join(out, "index.md"), "w").write("\n".join(L) + "\n")
    print(f"wrote {len(stats)} sheets + index.md to {out}")


if __name__ == "__main__":
    main()
