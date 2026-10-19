#!/usr/bin/env python
"""Build the FAIR real-wire benchmark lists + md5/pHash overlap gate.

Writes results/qn_backbone_3way/fairbench/{lists/*.txt, overlap_gate.json}.
Each list line: "<image path>\t<mask path>" (repo-relative). Masks: fg = nonzero
(mBEST .npy instance ids > 0; rmdlo HSV png 255; realtextured-source binary >0, putvision >=128).

Gate: every candidate image vs references (valset, ALL MovingCables frames, KD pool lists,
fastdlo EWD sample images, P3W-lineage realtextured train pool, frames of the 4 PI videos every
30th): file-md5, decoded-pixel-md5, and 64-bit DCT pHash (imagehash.phash) Hamming <= 6.
Reports min Hamming per (candidate set, reference) and aborts listing any hit image.

  ./env/bin/python src/fairbench_overlap_gate.py [--workers 6]
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
OUT = os.path.join(ROOT, "results/qn_backbone_3way/fairbench")
RT = "data/realtextured_raw"
HAMMING_THR = 6


def cands():
    c = {}
    c["mbest_sbhc"] = [(p, p.replace("/images/", "/gt_labels/").replace(".jpg", ".npy"))
                       for s in ("S1", "S2", "S3")
                       for p in sorted(glob.glob(f"{RT}/mBEST/dataset/{s}/images/*.jpg"),
                                       key=lambda x: int(os.path.basename(x)[3:-4]))]
    c["rmdlo_fs2"] = [(p, p.replace("/rgb/", "/mask/")) for p in
                      sorted(glob.glob("results/qn_backbone_3way/fairbench/rmdlo_fs2/rgb/*.png"))]
    # P3W-SEEN (trained/validated in the P3W lineage via data/dformer_dataset_realtextured): info only
    c["p3wseen_bwh"] = [(p, p.replace("/imgs/", "/masks/").replace(".jpg", ".png")) for p in
                        sorted(glob.glob(f"{RT}/CVF-DLO/DATASETS/BWH/imgs/*.jpg"))]
    c["p3wseen_labd_real"] = [(p, p.replace("/imgs/", "/masks/").replace(".jpg", ".png")) for p in
                              sorted(glob.glob(f"{RT}/CVF-DLO/DATASETS/LABD_Real/imgs/*.jpg"))]
    c["p3wseen_m4nh_cables"] = [(p, p[:-4] + "_mask_all.png") for sub in ("simple", "hard") for p in
                                sorted(glob.glob(f"{RT}/cables_dataset/cable_dataset_{sub}/*.jpg"))]
    c["p3wseen_putvision"] = [(p, p.replace("/img/", "/mask/")) for p in
                              sorted(glob.glob(f"{RT}/putvision_extract/data/img/*.jpg"))]
    for k, v in c.items():
        for ip, mp in v:
            assert os.path.exists(ip) and os.path.exists(mp), (k, ip, mp)
    return c


def refs():
    r = {}
    r["valset"] = sorted(glob.glob("data/real_wires_valset/imgs/*.jpg"))
    r["movingcables_all"] = sorted(glob.glob("data/dformer_dataset_movingcables/RGB/*.png"))
    for f in sorted(glob.glob("results/qn_backbone_3way/kd_pools/*.txt")):
        r["kdpool_" + os.path.basename(f)[:-4]] = [l.strip() for l in open(f) if l.strip()]
    r["fastdlo_ewd_samples"] = sorted(glob.glob(f"{RT}/fastdlo/test_images/*"))
    r["p3w_realtextured_pool"] = sorted(glob.glob("data/dformer_dataset_realtextured/RGB/*.png"))
    return r


def h_file(p):
    try:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        md5f = hashlib.md5(open(p, "rb").read()).hexdigest()
        md5p = hashlib.md5(im.tobytes()).hexdigest()
        ph = imagehash.phash(Image.fromarray(im[:, :, ::-1])).hash.flatten()
        return p, md5f, md5p, np.packbits(ph)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{p}: {e}")


def video_hashes():
    out = []
    for i in (1, 2, 3, 4):
        cap = cv2.VideoCapture(f"data/dlo_real_sample_{i}.mp4")
        fi = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if fi % 30 == 0:
                ph = imagehash.phash(Image.fromarray(f[:, :, ::-1])).hash.flatten()
                out.append((f"video{i}_f{fi}", "", hashlib.md5(f.tobytes()).hexdigest(), np.packbits(ph)))
            fi += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.chdir(ROOT)
    C, R = cands(), refs()
    allp = sorted({ip for v in C.values() for ip, _ in v} | {p for v in R.values() for p in v})
    with Pool(a.workers) as pool:
        H = {p: (f, x, h) for p, f, x, h in pool.map(h_file, allp, chunksize=32)}
    Rh = {k: [H[p] for p in v] for k, v in R.items()}
    Rh["pi_videos_every30"] = [(f, x, h) for _, f, x, h in video_hashes()]
    ref_bits = {k: np.unpackbits(np.stack([h for _, _, h in v]), axis=1) for k, v in Rh.items()}
    gate, hits_all = {}, {}
    for cn, pairs in C.items():
        gate[cn] = {"n": len(pairs)}
        hits = set()
        cb = np.unpackbits(np.stack([H[ip][2] for ip, _ in pairs]), axis=1)
        for rn, rv in Rh.items():
            rf = {f for f, _, _ in rv if f}; rx = {x for _, x, _ in rv}
            md5f = [ip for ip, _ in pairs if H[ip][0] in rf]
            md5p = [ip for ip, _ in pairs if H[ip][1] in rx]
            ham = (cb[:, None, :] != ref_bits[rn][None, :, :]).sum(-1)  # n_c x n_r
            mn = ham.min(1)
            ph = [pairs[i][0] for i in np.where(mn <= HAMMING_THR)[0]]
            gate[cn][rn] = {"n_ref": len(rv), "md5_file_hits": len(md5f), "md5_pixel_hits": len(md5p),
                            "phash_min_hamming": int(mn.min()), "phash_hits_le6": len(ph),
                            "phash_hit_examples": ph[:5]}
            if not (cn.startswith("p3wseen_") and rn == "p3w_realtextured_pool"):  # expected: known P3W-seen
                hits |= set(md5f) | set(md5p) | set(ph)
        gate[cn]["n_flagged_any"] = len(hits)
        hits_all[cn] = sorted(hits)
    os.makedirs(os.path.join(OUT, "lists"), exist_ok=True)
    for cn, pairs in C.items():
        kept = [(ip, mp) for ip, mp in pairs if ip not in set(hits_all[cn])]
        gate[cn]["n_kept"] = len(kept)
        with open(os.path.join(OUT, "lists", cn + ".txt"), "w") as f:
            f.writelines(f"{ip}\t{mp}\n" for ip, mp in kept)
    json.dump({"hamming_thr": HAMMING_THR, "gate": gate, "flagged": hits_all},
              open(os.path.join(OUT, "overlap_gate.json"), "w"), indent=2)
    for cn, g in gate.items():
        print(cn, "n", g["n"], "kept", g["n_kept"], {rn: (v["md5_file_hits"], v["md5_pixel_hits"],
              v["phash_min_hamming"], v["phash_hits_le6"]) for rn, v in g.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
