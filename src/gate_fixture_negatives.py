#!/usr/bin/env python
"""P32 fixture-surface hard-negative STRICT GATE (md5/sha256 + perceptual hash).

Self-contained pHash (DCT, 64-bit) identical in algorithm to ``imagehash.phash``
(``imagehash`` is not installed in this env, so we reimplement it with numpy/PIL):

    pHash(img):
        gray = img.convert("L").resize((32, 32), LANCZOS)
        dct  = DCT-II along both axes  (scipy-free: orthonormal-less type-II)
        low  = dct[:8, :8]            (top-left 8x8 low-frequency block)
        med  = median(low[1:])        (exclude the DC term low[0,0])
        bits = low > med              (64 bits)
    hamming(a, b) = popcount(a XOR b)

REJECT a candidate if:
  * its sha256 collides with any forbidden image (exact dup), OR
  * its min pHash Hamming distance to the forbidden set is <= --phash-thresh (default 10).

The forbidden set = data/real_wires_valset/imgs (the literal EWD-overlapping valset)
+ the realtextured pool actually used by the current best model
(data/dformer_dataset_realtextured/RGB) + CVF-DLO BWH/LABD_Real source imgs.

Usage:
  python src/gate_fixture_negatives.py \
      --candidates-dir data/hardneg_fixtures_raw \
      --manifest data/hardneg_fixtures_raw/manifest.csv \
      --forbidden-json <scratch>/forbidden_list.json \
      --out-json <scratch>/gate_result.json \
      --phash-thresh 10
"""
import argparse
import csv
import hashlib
import json
import os

import numpy as np
from PIL import Image


def _dct_1d(a):
    """Orthogonal-less DCT-II along the last axis (matches scipy.fftpack.dct type 2)."""
    N = a.shape[-1]
    n = np.arange(N)
    k = n.reshape(-1, 1)
    # cos[k, n] = cos(pi/N * (n + 0.5) * k)
    basis = np.cos(np.pi / N * (n + 0.5) * k)  # (N, N)
    return a @ basis.T


def phash(img, hash_size=8, highfreq_factor=4):
    """64-bit perceptual hash as a numpy bool array of length hash_size**2."""
    img_size = hash_size * highfreq_factor  # 32
    image = img.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float64)
    dct = _dct_1d(_dct_1d(pixels).T).T  # DCT along both axes
    low = dct[:hash_size, :hash_size]
    # imagehash.phash uses the median of the WHOLE low block (incl. DC) — match it exactly.
    med = np.median(low)
    return (low > med).flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_phash(path):
    try:
        with Image.open(path) as im:
            return phash(im)
    except Exception as e:
        print(f"  WARN cannot hash {path}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-dir", required=True)
    ap.add_argument("--forbidden-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--phash-thresh", type=int, default=10)
    args = ap.parse_args()

    forbidden_paths = json.load(open(args.forbidden_json))
    print(f"forbidden images: {len(forbidden_paths)}")

    # forbidden sha256 + pHash
    fb_sha = {}
    fb_hashes = []
    for p in forbidden_paths:
        try:
            fb_sha[sha256_of(p)] = p
        except Exception:
            pass
        h = load_phash(p)
        if h is not None:
            fb_hashes.append(h)
    fb_mat = np.stack(fb_hashes).astype(np.uint8)  # (F, 64)
    print(f"forbidden pHashes computed: {fb_mat.shape[0]}")

    cand_paths = sorted(
        os.path.join(args.candidates_dir, f)
        for f in os.listdir(args.candidates_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    )
    print(f"candidates: {len(cand_paths)}")

    results = []
    n_md5 = n_phash = n_kept = n_unreadable = 0
    for p in cand_paths:
        sha = None
        try:
            sha = sha256_of(p)
        except Exception:
            pass
        ch = load_phash(p)
        if ch is None:
            n_unreadable += 1
            results.append({"path": p, "sha256": sha, "min_phash": None,
                            "verdict": "REJECT_UNREADABLE", "nearest": None})
            continue
        if sha in fb_sha:
            n_md5 += 1
            results.append({"path": p, "sha256": sha, "min_phash": 0,
                            "verdict": "REJECT_EXACTDUP", "nearest": fb_sha[sha]})
            continue
        # vectorized hamming to all forbidden
        dists = np.count_nonzero(fb_mat != ch.astype(np.uint8)[None, :], axis=1)
        j = int(np.argmin(dists))
        mind = int(dists[j])
        if mind <= args.phash_thresh:
            n_phash += 1
            verdict = "REJECT_PHASH"
        else:
            n_kept += 1
            verdict = "KEEP"
        results.append({"path": p, "sha256": sha, "min_phash": mind,
                        "verdict": verdict, "nearest": forbidden_paths[j]})

    mins = [r["min_phash"] for r in results
            if r["min_phash"] is not None and r["verdict"] != "REJECT_EXACTDUP"]
    summary = {
        "n_candidates": len(cand_paths),
        "n_forbidden": int(fb_mat.shape[0]),
        "phash_thresh": args.phash_thresh,
        "n_reject_exactdup": n_md5,
        "n_reject_phash": n_phash,
        "n_reject_unreadable": n_unreadable,
        "n_kept": n_kept,
        "min_phash_distribution": {
            "min": int(np.min(mins)) if mins else None,
            "p1": float(np.percentile(mins, 1)) if mins else None,
            "p5": float(np.percentile(mins, 5)) if mins else None,
            "median": float(np.median(mins)) if mins else None,
            "mean": float(np.mean(mins)) if mins else None,
            "max": int(np.max(mins)) if mins else None,
        },
        "histogram_le20": {str(d): int(sum(1 for m in mins if m == d)) for d in range(0, 21)},
    }
    json.dump({"summary": summary, "results": results}, open(args.out_json, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
