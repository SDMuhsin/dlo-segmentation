#!/usr/bin/env python
"""Build the P44 connector-paste training set.

IDENTICAL to ``data/dformer_dataset_3way_connscale3`` (the Phase-3W / July-deck base)
EXCEPT the chosen TRAIN frames are replaced 1:1 by their ``_p44conn`` composited
versions (real connector cutouts pasted as label 2). Drift-controlled substitution:
train.txt and test.txt are byte-identical to the base, so the arm differs from its
baseline in frame CONTENT only - not in dataset size, set ids, or split grouping.
That isolation is what P37 lacked when +2000 added frames confounded its result.

Unchanged frames are symlinked, so the build costs almost no disk.

Usage:
  python src/build_p44_connpaste_dataset.py \
      --paste-dir data/dformer_dataset_p44_connpaste \
      --out-dir   data/dformer_dataset_p44_connpaste_train
"""
import argparse, os, shutil, sys

BASE = "data/dformer_dataset_3way_connscale3"
SUBS = ("RGB", "Label", "Depth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--paste-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--suffix", default="_p44conn")
    args = ap.parse_args()

    base, out = os.path.abspath(args.base), os.path.abspath(args.out_dir)
    paste = os.path.abspath(args.paste_dir)

    subbed = set()
    sb = os.path.join(paste, "substituted_basenames.txt")
    if os.path.exists(sb):
        subbed = {l.strip() for l in open(sb) if l.strip()}
    if not subbed:
        sys.exit(f"no substituted_basenames.txt under {paste}")

    for sub in SUBS:
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    n_sub = n_link = n_missing = 0
    for sub in SUBS:
        src_dir = os.path.join(base, sub)
        if not os.path.isdir(src_dir):
            continue
        for fn in sorted(os.listdir(src_dir)):
            b = os.path.splitext(fn)[0]
            dst = os.path.join(out, sub, fn)
            if os.path.lexists(dst):
                os.unlink(dst)
            if b in subbed:
                cand = os.path.join(paste, sub, b + args.suffix + ".png")
                if os.path.exists(cand):
                    os.symlink(cand, dst); n_sub += 1; continue
                n_missing += 1
            os.symlink(os.path.join(src_dir, fn), dst); n_link += 1

    # splits copied verbatim -> byte-identical train/test lists
    for f in ("train.txt", "test.txt"):
        p = os.path.join(base, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, f))

    print(f"substituted {n_sub}  symlinked {n_link}  missing-paste {n_missing}")
    print(f"-> {out}")
    for f in ("train.txt", "test.txt"):
        a, b_ = os.path.join(base, f), os.path.join(out, f)
        if os.path.exists(a):
            same = open(a, "rb").read() == open(b_, "rb").read()
            print(f"  {f}: identical to base = {same}")


if __name__ == "__main__":
    main()
