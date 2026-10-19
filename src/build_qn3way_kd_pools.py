#!/usr/bin/env python
"""Build the unlabeled REAL image pools for QN->3-way distillation (train_qn3way_kd.py).

Writes list files (one repo-relative image path per line) to
``results/qn_backbone_3way/kd_pools/``:

* ``mc_train.txt``  -- MovingCables TRAIN-split frames ONLY (data/dformer_dataset_movingcables/train.txt).
* ``clutter.txt``   -- real clutter / backdrop photos: data/textures/backgrounds (CC0 Polyhaven),
                       data/textures/backgrounds_p4orig11 (CC0 ambientCG; md5-deduped against the former),
                       data/hardneg_fixtures_raw (Wikimedia Commons fixtures, per-file licence in manifest.csv).
* ``disjointness.json`` -- the evidence for the hard rules below.

HARD RULES (asserted, the script aborts if any fails):
1. The MovingCables TEST split (our real evaluation meter) is never in any pool: disjoint by
   filename, by clip id (MC split is clip-wise), by file md5 AND by decoded-pixel md5.
2. data/real_wires_valset and the 4 PI videos (data/dlo_real_sample_*) are never in any pool
   (path check + file/pixel md5 against every valset image).
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DEFAULT = os.path.join("results", "qn_backbone_3way", "kd_pools")
MC_DIR = os.path.join("data", "dformer_dataset_movingcables")
FORBIDDEN_SUBSTR = ("real_wires_valset", "dlo_real_sample")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _hashes(path):
    with open(os.path.join(PROJECT_ROOT, path), "rb") as f:
        fmd5 = hashlib.md5(f.read()).hexdigest()
    img = cv2.imread(os.path.join(PROJECT_ROOT, path), cv2.IMREAD_COLOR)
    if img is None:
        return path, fmd5, None, None
    return path, fmd5, hashlib.md5(img.tobytes()).hexdigest(), img.shape[:2]


def hash_all(paths, workers):
    with Pool(workers) as p:
        return {r[0]: r[1:] for r in p.map(_hashes, paths, chunksize=64)}


def list_images(d):
    return sorted(os.path.relpath(p, PROJECT_ROOT)
                  for p in glob.glob(os.path.join(PROJECT_ROOT, d, "*"))
                  if p.lower().endswith(IMG_EXT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.chdir(PROJECT_ROOT)
    os.makedirs(args.out, exist_ok=True)

    def split(name):
        with open(os.path.join(MC_DIR, f"{name}.txt")) as f:
            return [os.path.join(MC_DIR, "RGB", os.path.basename(l.strip()))
                    for l in f if l.strip()]
    mc_train, mc_test = split("train"), split("test")
    clip = lambda p: os.path.basename(p).split("_")[0]  # noqa: E731

    # clutter pools
    bg = list_images("data/textures/backgrounds")
    bg11 = list_images("data/textures/backgrounds_p4orig11")
    fx = list_images("data/hardneg_fixtures_raw")
    valset = list_images("data/real_wires_valset/imgs")

    all_paths = sorted(set(mc_train + mc_test + bg + bg11 + fx + valset))
    print(f"hashing {len(all_paths)} files ...", flush=True)
    H = hash_all(all_paths, args.workers)
    unreadable = [p for p in all_paths if H[p][1] is None]
    for p in unreadable:
        print(f"  WARN unreadable image, dropped: {p}")

    # clutter: dedupe by pixel md5 (backgrounds_p4orig11 overlaps backgrounds/01-11)
    clutter, seen, n_dup = [], set(), 0
    fx_manifest = set()
    with open(os.path.join("data", "hardneg_fixtures_raw", "manifest.csv"), newline="") as f:
        fx_manifest = {row["filename"] for row in csv.DictReader(f)}
    no_licence = [p for p in fx if os.path.basename(p) not in fx_manifest]
    for p in bg + bg11 + fx:
        if H[p][1] is None or p in no_licence:
            continue
        if H[p][1] in seen:
            n_dup += 1
            continue
        seen.add(H[p][1])
        clutter.append(p)
    mc_train = [p for p in mc_train if H[p][1] is not None]

    # ---------------- disjointness checks ----------------
    test_names = {os.path.basename(p) for p in mc_test}
    test_clips = {clip(p) for p in mc_test}
    test_f = {H[p][0] for p in mc_test}
    test_px = {H[p][1] for p in mc_test}
    val_f = {H[p][0] for p in valset}
    val_px = {H[p][1] for p in valset}
    checks = {}
    for pname, pool in (("mc_train", mc_train), ("clutter", clutter)):
        c = {
            "n": len(pool),
            "filename_overlap_mc_test": sum(os.path.basename(p) in test_names for p in pool)
            if pname == "mc_train" else 0,
            "clip_overlap_mc_test": sorted({clip(p) for p in pool} & test_clips)
            if pname == "mc_train" else [],
            "file_md5_overlap_mc_test": sum(H[p][0] in test_f for p in pool),
            "pixel_md5_overlap_mc_test": sum(H[p][1] in test_px for p in pool),
            "file_md5_overlap_valset": sum(H[p][0] in val_f for p in pool),
            "pixel_md5_overlap_valset": sum(H[p][1] in val_px for p in pool),
            "forbidden_path_hits": [p for p in pool if any(s in p for s in FORBIDDEN_SUBSTR)],
        }
        checks[pname] = c
        bad = (c["filename_overlap_mc_test"] or c["clip_overlap_mc_test"]
               or c["file_md5_overlap_mc_test"] or c["pixel_md5_overlap_mc_test"]
               or c["file_md5_overlap_valset"] or c["pixel_md5_overlap_valset"]
               or c["forbidden_path_hits"])
        assert not bad, f"DISJOINTNESS VIOLATION in {pname}: {c}"
    # within-MC: are there train frames byte-identical to one another? (info only)
    dup_within = len(mc_train) - len({H[p][1] for p in mc_train})

    # licences
    lic = Counter()
    man = os.path.join("data", "hardneg_fixtures_raw", "manifest.csv")
    fx_lic = {}
    with open(man, newline="") as f:
        for row in csv.DictReader(f):
            fx_lic[row["filename"]] = row["license"]
    for p in clutter:
        if "hardneg_fixtures_raw" in p:
            lic["hardneg_fixtures_raw: " + fx_lic.get(os.path.basename(p), "UNKNOWN")] += 1
        elif "backgrounds_p4orig11" in p:
            lic["backgrounds_p4orig11: CC0 (ambientCG)"] += 1
        else:
            lic["backgrounds: CC0 (Polyhaven)"] += 1

    with open(os.path.join(args.out, "mc_train.txt"), "w") as f:
        f.write("\n".join(mc_train) + "\n")
    with open(os.path.join(args.out, "clutter.txt"), "w") as f:
        f.write("\n".join(clutter) + "\n")
    res = {
        "mc_train": checks["mc_train"], "clutter": checks["clutter"],
        "mc_test_n": len(mc_test), "mc_test_clips": sorted(test_clips),
        "mc_train_clips_n": len({clip(p) for p in mc_train}),
        "valset_n_checked": len(valset),
        "clutter_pixel_duplicates_removed": n_dup,
        "mc_train_internal_pixel_duplicates": dup_within,
        "unreadable_dropped": unreadable,
        "no_licence_row_dropped": no_licence,
        "clutter_licences": dict(sorted(lic.items())),
        "clutter_sources": {"backgrounds": len(bg), "backgrounds_p4orig11": len(bg11),
                            "hardneg_fixtures_raw": len(fx)},
    }
    with open(os.path.join(args.out, "disjointness.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("ALL DISJOINTNESS ASSERTIONS PASSED")


if __name__ == "__main__":
    sys.exit(main())
