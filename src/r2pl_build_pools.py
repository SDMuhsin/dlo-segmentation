#!/usr/bin/env python
"""R2-PL: assemble the REAL unlabeled pools for pseudo-label self-training + leakage gate.

Writes results/qn_backbone_3way/r2pl/pools/{<pool>.txt, pl_pool_all.txt, gate.json, counts.json}
(one repo-relative image path per line).

Candidate pools (survey in results/qn_backbone_3way/r2pl/R2PL.md):
  mc_train        MovingCables TRAIN split (real cable footage composited on photo backgrounds,
                  poking stick, NO hands) = results/qn_backbone_3way/kd_pools/mc_train.txt
  fixtures_wc     Wikimedia Commons photos of terminals / plugs / sockets / harnesses
                  (data/hardneg_fixtures_raw, licensed rows only) = kd_pools clutter minus bgs
  backgrounds_cc0 CC0 surface/scene photos (data/textures/backgrounds)
  rmdlo_realbg    data/dformer_dataset_rmdlo frames that kept the REAL TrackDLO/RM-DLO white-table
                  background (=> real HANDS manipulating rope/tubing; rope chroma is RE-SKINNED);
                  the 60 % background-composited frames are dropped by a border heuristic (white-table
                  fraction >= .6 AND smooth border: mean |Laplacian| < 2);
                  setid 12 (failure_scenario_2 bag = FAIRBENCH rmdlo_fs2 eval scene) excluded.

Gate (every candidate vs every held-out/eval reference): file-md5, decoded-pixel-md5 and 64-bit
pHash Hamming <= 6 (same thresholds as src/fairbench_overlap_gate.py). References:
  valset (62), MovingCables TEST (1172), FAIRBENCH eval images (mbest_sbhc, rmdlo_fs2,
  p3wseen_bwh/labd_real/m4nh/putvision), fastdlo EWD samples, 4 PI videos every 5th frame
  (pre-processed exactly like inference: centre-crop 4:3 -> 640x480) plus raw frames.
Any hit is DROPPED from the pool. Path guards also refuse dlo_real_sample / real_wires_valset /
MC-test basenames.

  ./env/bin/python src/r2pl_build_pools.py [--workers 6] [--rmdlo-white-thr 0.6 --rmdlo-lap-thr 2]
"""
import argparse
import glob
import hashlib
import json
import os
from multiprocessing import Pool

import cv2
import imagehash
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results/qn_backbone_3way/r2pl/pools")
KD = "results/qn_backbone_3way/kd_pools"
FB = "results/qn_backbone_3way/fairbench/lists"
HAMMING_THR = 6
RMDLO_EXCLUDE_SETIDS = {12}   # failure_scenario_2_fast_dlo_tip_movement.bag == FAIRBENCH rmdlo_fs2


def rd(p):
    return [l.strip().split("\t")[0] for l in open(p) if l.strip()]


def white_table_frac(p):
    """Fraction of a 20-px border band that is white-table-like (low saturation, bright)."""
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = np.zeros(hsv.shape[:2], bool)
    m[:20] = m[-20:] = True
    m[:, :20] = m[:, -20:] = True
    s, v = hsv[..., 1][m], hsv[..., 2][m]
    g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
    lap = float(np.abs(cv2.Laplacian(g, cv2.CV_32F))[m].mean())   # composites are textured
    return p, (float(((s < 45) & (v > 110)).mean()), lap)


def candidates(args, pool):
    c = {}
    c["mc_train"] = rd(f"{KD}/mc_train.txt")
    clutter = rd(f"{KD}/clutter.txt")
    c["fixtures_wc"] = [p for p in clutter if "hardneg_fixtures_raw" in p]
    c["backgrounds_cc0"] = [p for p in clutter if "textures/backgrounds" in p]
    rm = sorted(glob.glob("data/dformer_dataset_rmdlo/RGB/*.png"))
    rm = [p for p in rm if int(os.path.basename(p).split("_")[0]) not in RMDLO_EXCLUDE_SETIDS]
    wt = dict(pool.map(white_table_frac, rm, chunksize=16))
    c["rmdlo_realbg"] = [p for p in rm if wt[p][0] >= args.rmdlo_white_thr
                         and wt[p][1] < args.rmdlo_lap_thr]
    return c, wt


def references():
    r = {}
    r["valset"] = sorted(glob.glob("data/real_wires_valset/imgs/*"))
    r["mc_test"] = ["data/dformer_dataset_movingcables/" + l for l in
                    rd("data/dformer_dataset_movingcables/test.txt")]
    for f in sorted(glob.glob(f"{FB}/*.txt")):
        r["fairbench_" + os.path.basename(f)[:-4]] = rd(f)
    r["fastdlo_ewd_samples"] = sorted(glob.glob("data/realtextured_raw/fastdlo/test_images/*"))
    return r


def h_img(im):
    md5p = hashlib.md5(im.tobytes()).hexdigest()
    ph = imagehash.phash(Image.fromarray(im[:, :, ::-1])).hash.flatten()
    return md5p, np.packbits(ph)


def h_file(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"unreadable {p}")
    md5f = hashlib.md5(open(p, "rb").read()).hexdigest()
    return (p, md5f) + h_img(im)


def preprocess(f):
    h, w = f.shape[:2]
    if w / h > 4 / 3:
        nw = int(round(h * 4 / 3)); f = f[:, (w - nw) // 2:(w - nw) // 2 + nw]
    else:
        nh = int(round(w * 3 / 4)); f = f[(h - nh) // 2:(h - nh) // 2 + nh]
    return cv2.resize(f, (640, 480), interpolation=cv2.INTER_AREA)


def video_hashes(step=5):
    out = []
    for i in (1, 2, 3, 4):
        cap = cv2.VideoCapture(f"data/dlo_real_sample_{i}.mp4")
        fi = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if fi % step == 0:
                for tag, im in (("raw", f), ("pp", preprocess(f))):
                    out.append((f"video{i}_f{fi}_{tag}", "") + h_img(im))
            fi += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rmdlo-white-thr", type=float, default=0.6)
    ap.add_argument("--rmdlo-lap-thr", type=float, default=2.0,
                    help="max mean |Laplacian| of the border band (real table ~0.6-0.8, "
                         "textured composites 4-9)")
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)
    with Pool(a.workers) as pool:
        C, wt = candidates(a, pool)
        R = references()
        allp = sorted({p for v in C.values() for p in v} | {p for v in R.values() for p in v})
        H = {p: (f, x, h) for p, f, x, h in pool.map(h_file, allp, chunksize=32)}
    Rh = {k: [H[p] for p in v] for k, v in R.items()}
    Rh["pi_videos_every5"] = [(f, x, h) for _, f, x, h in video_hashes()]
    ref_bits = {k: np.unpackbits(np.stack([h for _, _, h in v]), axis=1) for k, v in Rh.items()}

    forbidden = ("real_wires_valset", "dlo_real_sample", "fairbench", "rmdlo_probe")
    mc_test_b = {os.path.basename(p) for p in R["mc_test"]}
    gate, kept = {}, {}
    for cn, ps in C.items():
        g = {"n": len(ps)}
        hits = {p for p in ps if any(s in p for s in forbidden)
                or ("movingcables" in p and os.path.basename(p) in mc_test_b)}
        g["path_guard_hits"] = len(hits)
        cb = np.unpackbits(np.stack([H[p][2] for p in ps]), axis=1)
        for rn, rv in Rh.items():
            rf = {f for f, _, _ in rv if f}
            rx = {x for _, x, _ in rv}
            md5f = [p for p in ps if H[p][0] in rf]
            md5p = [p for p in ps if H[p][1] in rx]
            mn = np.full(len(ps), 64)
            for j in range(0, len(ps), 2048):   # chunk: n_c x n_r x 64 bool
                mn[j:j + 2048] = (cb[j:j + 2048, None, :] != ref_bits[rn][None]).sum(-1).min(1)
            ph = [ps[i] for i in np.where(mn <= HAMMING_THR)[0]]
            g[rn] = {"n_ref": len(rv), "md5_file_hits": len(md5f), "md5_pixel_hits": len(md5p),
                     "phash_min_hamming": int(mn.min()), "phash_hits_le6": len(ph),
                     "phash_hit_examples": ph[:5]}
            hits |= set(md5f) | set(md5p) | set(ph)
        g["n_dropped"] = len(hits)
        kept[cn] = [p for p in ps if p not in hits]
        g["n_kept"] = len(kept[cn])
        gate[cn] = g
        with open(os.path.join(OUT, cn + ".txt"), "w") as f:
            f.writelines(p + "\n" for p in kept[cn])
        print(cn, "n", g["n"], "kept", g["n_kept"],
              {rn: (v["md5_file_hits"], v["md5_pixel_hits"], v["phash_min_hamming"],
                    v["phash_hits_le6"]) for rn, v in g.items() if isinstance(v, dict)}, flush=True)
    allk = [p for cn in C for p in kept[cn]]
    with open(os.path.join(OUT, "pl_pool_all.txt"), "w") as f:
        f.writelines(p + "\n" for p in allk)
    counts = {cn: len(v) for cn, v in kept.items()}
    counts["pl_pool_all"] = len(allk)
    rmall = [p for p in wt]
    counts["rmdlo_frames_considered(excl setid12)"] = len(rmall)
    json.dump({"hamming_thr": HAMMING_THR, "rmdlo_white_thr": a.rmdlo_white_thr, "rmdlo_lap_thr": a.rmdlo_lap_thr,
               "gate": gate, "ref_sizes": {k: len(v) for k, v in Rh.items()}},
              open(os.path.join(OUT, "gate.json"), "w"), indent=2)
    json.dump(counts, open(os.path.join(OUT, "counts.json"), "w"), indent=2)
    json.dump(wt, open(os.path.join(OUT, "rmdlo_white_table_frac.json"), "w"))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
