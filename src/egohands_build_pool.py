#!/usr/bin/env python
"""R2-PL: EgoHands (Bambach et al. ICCV'15) hand hard-negative pool.  ** LICENCE PENDING: R&D ONLY **

Reads data/egohands_raw (official egohands_data.zip extracted: metadata.mat + _LABELLED_SAMPLES),
rasterises ALL hand polygons (own + partner, left + right) at 720x1280, cover-crops to 4:3 and resizes
to 640x480 exactly like make_pseudolabels.cover_resize / the trainer (centre 960x720 crop, INTER_AREA
for rgb, NEAREST for masks).

Leakage gate (same as src/r2pl_build_pools.py): every raw frame AND its 640x480 version vs valset,
MovingCables TEST, FAIRBENCH eval lists, fastdlo EWD samples, 4 PI videos every 5th frame (raw + pp):
file-md5, pixel-md5, pHash Hamming <= 6. Hits are dropped.

Selection: hand area (all hands, 640x480) >= --min-hand; within-video pHash de-dup (Hamming <= --dedup);
then round-robin over the 48 videos (4 activities x 3 locations x 4 viewers) up to --n frames.

Writes:
  data/egohands_raw/r2pl_640/{rgb,hand,hand_eroded3}/<VIDEO>__frame_XXXX.png
  results/qn_backbone_3way/r2pl/pools/egohands_hands.txt         (rgb path per line -> --pool)
  results/qn_backbone_3way/r2pl/pools/egohands_hands_masks.tsv   ('<rgb>\\t<hand mask>' -> --force-bg-mask-list;
                                                                  UN-eroded masks, erosion = --force-bg-erode 3)
  results/qn_backbone_3way/r2pl/egohos/egohands_{gate,select}.json

  ./env/bin/python src/egohands_build_pool.py [--n 2000 --min-hand 0.05 --workers 6]
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from multiprocessing import Pool

import cv2
import imagehash
import numpy as np
import scipy.io as sio
from PIL import Image

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
import r2pl_build_pools as RB  # noqa: E402  (references(), video_hashes(), preprocess())

RAW = "data/egohands_raw"
OUT_IMG = f"{RAW}/r2pl_640"
POOLS = "results/qn_backbone_3way/r2pl/pools"
REP = "results/qn_backbone_3way/r2pl/egohos"
HAMMING_THR = 6
H, W = 480, 640


def load_meta():
    m = sio.loadmat(f"{RAW}/metadata.mat", squeeze_me=True, struct_as_record=False)["video"]
    items = []
    for v in m:
        for lf in v.labelled_frames:
            polys = []
            for k in ("myleft", "myright", "yourleft", "yourright"):
                a = np.asarray(getattr(lf, k), dtype=np.float64)
                if a.ndim == 2 and a.shape[0] >= 3:
                    polys.append(a)
            items.append({"video": v.video_id, "activity": v.activity_id, "location": v.location_id,
                          "viewer": v.ego_viewer_id, "frame": int(lf.frame_num),
                          "img": f"{RAW}/_LABELLED_SAMPLES/{v.video_id}/frame_{int(lf.frame_num):04d}.jpg",
                          "polys": polys})
    return items


def cover(im, interp):
    ih, iw = im.shape[:2]
    if iw / ih > W / H:
        nw = int(round(ih * W / H)); x0 = (iw - nw) // 2; im = im[:, x0:x0 + nw]
    else:
        nh = int(round(iw * H / W)); y0 = (ih - nh) // 2; im = im[y0:y0 + nh]
    return cv2.resize(im, (W, H), interpolation=interp)


def process(it):
    bgr = cv2.imread(it["img"], cv2.IMREAD_COLOR)
    assert bgr is not None and bgr.shape[:2] == (720, 1280), it["img"]
    mk = np.zeros(bgr.shape[:2], np.uint8)
    for a in it["polys"]:   # MATLAB 1-based (x, y)
        cv2.fillPoly(mk, [np.round(a - 1).astype(np.int32)], 1)
    rgb6 = cover(bgr, cv2.INTER_AREA)
    mk6 = cover(mk, cv2.INTER_NEAREST)
    md5f = hashlib.md5(open(it["img"], "rb").read()).hexdigest()
    hr = RB.h_img(bgr)
    hp = RB.h_img(rgb6)
    return {"key": f"{it['video']}__frame_{it['frame']:04d}", "md5f": md5f, "md5p_raw": hr[0],
            "ph_raw": hr[1], "md5p_pp": hp[0], "ph_pp": hp[1],
            "hand_frac": float(mk6.mean()), "hand_frac_raw": float(mk.mean()),
            "n_hands": len(it["polys"])}


def write_one(args):
    it, key = args
    bgr = cv2.imread(it["img"], cv2.IMREAD_COLOR)
    mk = np.zeros(bgr.shape[:2], np.uint8)
    for a in it["polys"]:
        cv2.fillPoly(mk, [np.round(a - 1).astype(np.int32)], 1)
    rgb6 = cover(bgr, cv2.INTER_AREA)
    mk6 = cover(mk, cv2.INTER_NEAREST) * 255
    er = cv2.erode(mk6, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    paths = {d: f"{OUT_IMG}/{d}/{key}.png" for d in ("rgb", "hand", "hand_eroded3")}
    cv2.imwrite(paths["rgb"], rgb6)
    cv2.imwrite(paths["hand"], mk6)
    cv2.imwrite(paths["hand_eroded3"], er)
    return key, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--min-hand", type=float, default=0.05)
    ap.add_argument("--dedup", type=int, default=6, help="within-video pHash Hamming de-dup threshold")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.chdir(ROOT)
    items = load_meta()
    print(len(items), "labelled frames", flush=True)
    with Pool(a.workers) as pool:
        res = pool.map(process, items, chunksize=16)
        R = RB.references()
        refp = sorted({p for v in R.values() for p in v})
        RH = {p: (f, x, h) for p, f, x, h in pool.map(RB.h_file, refp, chunksize=32)}
    Rh = {k: [RH[p] for p in v] for k, v in R.items()}
    Rh["pi_videos_every5"] = [(f, x, h) for _, f, x, h in RB.video_hashes()]

    # ---- leakage gate ----
    gate, hits = {"n": len(res)}, set()
    for tag in ("raw", "pp"):
        cb = np.unpackbits(np.stack([r["ph_" + tag] for r in res]), axis=1)
        for rn, rv in Rh.items():
            rb = np.unpackbits(np.stack([h for _, _, h in rv]), axis=1)
            rf = {f for f, _, _ in rv if f}; rx = {x for _, x, _ in rv}
            md5f = [r["key"] for r in res if r["md5f"] in rf]
            md5p = [r["key"] for r in res if r["md5p_" + tag] in rx]
            mn = np.full(len(res), 64)
            for j in range(0, len(res), 1024):
                mn[j:j + 1024] = (cb[j:j + 1024, None, :] != rb[None]).sum(-1).min(1)
            ph = [res[i]["key"] for i in np.where(mn <= HAMMING_THR)[0]]
            gate[f"{tag}__{rn}"] = {"n_ref": len(rv), "md5_file_hits": len(md5f) if tag == "raw" else None,
                                    "md5_pixel_hits": len(md5p), "phash_min_hamming": int(mn.min()),
                                    "phash_hits_le6": len(ph), "phash_hit_examples": ph[:5]}
            hits |= set(md5f) | set(md5p) | set(ph)
    gate["n_dropped"] = len(hits)
    gate["dropped"] = sorted(hits)
    os.makedirs(REP, exist_ok=True)
    json.dump({"hamming_thr": HAMMING_THR, "gate": gate, "ref_sizes": {k: len(v) for k, v in Rh.items()}},
              open(f"{REP}/egohands_gate.json", "w"), indent=1)
    for k, v in gate.items():
        if isinstance(v, dict):
            print(f"  gate {k}: md5f {v['md5_file_hits']} md5p {v['md5_pixel_hits']} "
                  f"phash_min {v['phash_min_hamming']} hits {v['phash_hits_le6']}")
    print("gate dropped", len(hits), flush=True)

    # ---- selection ----
    byk = {r["key"]: r for r in res}
    itk = {f"{it['video']}__frame_{it['frame']:04d}": it for it in items}
    big = [r for r in res if r["hand_frac"] >= a.min_hand and r["key"] not in hits]
    per_vid = defaultdict(list)
    for r in sorted(big, key=lambda r: -r["hand_frac"]):
        v = itk[r["key"]]["video"]
        hb = np.unpackbits(r["ph_pp"])
        if all((hb != np.unpackbits(q["ph_pp"])).sum() > a.dedup for q in per_vid[v]):
            per_vid[v].append(r)
    n_dedup = sum(len(v) for v in per_vid.values())
    rng = random.Random(a.seed)
    for v in per_vid.values():
        rng.shuffle(v)
    sel, i = [], 0
    vids = sorted(per_vid)
    while len(sel) < a.n and any(i < len(per_vid[v]) for v in vids):
        for v in vids:
            if i < len(per_vid[v]) and len(sel) < a.n:
                sel.append(per_vid[v][i]["key"])
        i += 1
    sel.sort()
    for d in ("rgb", "hand", "hand_eroded3"):
        os.makedirs(f"{OUT_IMG}/{d}", exist_ok=True)
    with Pool(a.workers) as pool:
        out = dict(pool.map(write_one, [(itk[k], k) for k in sel], chunksize=16))
    os.makedirs(POOLS, exist_ok=True)
    with open(f"{POOLS}/egohands_hands.txt", "w") as f:
        f.writelines(out[k]["rgb"] + "\n" for k in sel)
    with open(f"{POOLS}/egohands_hands_masks.tsv", "w") as f:
        f.writelines(f"{out[k]['rgb']}\t{out[k]['hand']}\n" for k in sel)
    def cnt(fld):
        return {x: sum(itk[k][fld] == x for k in sel) for x in sorted({itk[k][fld] for k in sel})}
    hf = np.array([byk[k]["hand_frac"] for k in sel])
    summ = {"n_labelled": len(res), "n_gate_dropped": len(hits),
            "n_hand_ge_min": len(big), "n_after_dedup": n_dedup, "n_selected": len(sel),
            "min_hand": a.min_hand, "dedup_hamming": a.dedup,
            "hand_frac_all_p10_p50_p90": [float(np.percentile([r["hand_frac"] for r in res], q)) for q in (10, 50, 90)],
            "hand_frac_sel_p10_p50_p90": [float(np.percentile(hf, q)) for q in (10, 50, 90)],
            "by_activity": cnt("activity"), "by_location": cnt("location"), "by_viewer": cnt("viewer"),
            "n_videos": len({itk[k]["video"] for k in sel}),
            "per_video_min_max": [min(sum(itk[k]["video"] == v for k in sel) for v in vids),
                                  max(sum(itk[k]["video"] == v for k in sel) for v in vids)]}
    json.dump({"summary": summ, "selected": {k: {"hand_frac": byk[k]["hand_frac"], "n_hands": byk[k]["n_hands"]}
                                             for k in sel}},
              open(f"{REP}/egohands_select.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
