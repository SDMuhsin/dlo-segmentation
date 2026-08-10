#!/usr/bin/env python3
"""Re-encode a STAGED segmentation dataset's Label PNGs to a 3-class scheme.

The staged dataset (produced by the render -> stage pipeline) has the layout::

    <src>/
        RGB/    *.png   (real-textured RGB, kept as-is)
        Depth/  *.png   (depth, kept as-is)
        Label/  *.png   (uint8, values 0..5)
        train.txt
        test.txt

where the integer Label values mean::

    0 = bg          1 = wire        2 = endpoint
    3 = bifurcation 4 = connector   5 = noise

This script writes a NEW dataset dir whose Label/*.png are re-encoded to exactly
``{0=bg, 1=wire, 2=connector}`` via::

    new 0  <-  old {0 (bg), 5 (noise)}
    new 1  <-  old {1 (wire), 2 (endpoint), 3 (bifurcation)}   # whole cable body
    new 2  <-  old {4 (connector)}

RGB/ and Depth/ in the new dir are per-file SYMLINKS to the source files (the
GBs of imagery are NOT duplicated); train.txt and test.txt are copied verbatim.

Why {0,1,2} specifically: the trainer's build_cache (src/train_rgbd_seg.py)
auto-detects label mode from the FIRST PNG's max value -- if max<=2 it stores
labels as-is (which is what the num_classes==3 pass-through in
src/train_rgb_only_sota.py expects); if max>=3 it applies a legacy -1 shift
that would mangle these labels. Re-encoding to {0,1,2} guarantees the clean
pass-through.

Usage::

    python src/reencode_labels_3way.py --src <staged_dir> --dst <new_dir>
    python src/reencode_labels_3way.py --src <staged_dir> --dst <new_dir> --limit 30
"""

import argparse
import os
import random
import shutil
import sys

import cv2
import numpy as np

# Source label values that are legal in a staged dataset.
ALLOWED_SRC = {0, 1, 2, 3, 4, 5}

# old value -> new value mapping (lookup table over the full uint8 range).
# Unmapped source values are caught explicitly before remap (see _remap).
_LUT = np.zeros(256, dtype=np.uint8)
_LUT[0] = 0   # bg          -> bg
_LUT[5] = 0   # noise       -> bg
_LUT[1] = 1   # wire        -> wire
_LUT[2] = 1   # endpoint    -> wire
_LUT[3] = 1   # bifurcation -> wire
_LUT[4] = 2   # connector   -> connector

# Reference groupings used by both the remap and the validation.
OLD_BG = (0, 5)
OLD_WIRE = (1, 2, 3)
OLD_CONNECTOR = (4,)


def _read_label(path):
    lbl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if lbl is None:
        raise FileNotFoundError(f"could not read label PNG: {path}")
    return lbl


def _remap(src_lbl, path):
    """Apply the {0..5}->{0,1,2} remap; fail loudly on unexpected source values."""
    uniq = set(int(v) for v in np.unique(src_lbl))
    extra = uniq - ALLOWED_SRC
    if extra:
        raise ValueError(
            f"unexpected label value(s) {sorted(extra)} in {path}; "
            f"allowed source values are {sorted(ALLOWED_SRC)}"
        )
    return _LUT[src_lbl]


def _symlink(src_file, dst_file, absolute=True):
    """Create (or refresh) a symlink dst_file -> src_file."""
    target = os.path.abspath(src_file) if absolute else os.path.relpath(
        src_file, os.path.dirname(dst_file)
    )
    if os.path.islink(dst_file) or os.path.exists(dst_file):
        os.remove(dst_file)
    os.symlink(target, dst_file)


def reencode(src, dst, limit=None, relative=False):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    src_rgb = os.path.join(src, "RGB")
    src_depth = os.path.join(src, "Depth")
    src_label = os.path.join(src, "Label")
    for d in (src_rgb, src_depth, src_label):
        if not os.path.isdir(d):
            raise NotADirectoryError(f"missing expected source subdir: {d}")

    dst_rgb = os.path.join(dst, "RGB")
    dst_depth = os.path.join(dst, "Depth")
    dst_label = os.path.join(dst, "Label")
    for d in (dst_rgb, dst_depth, dst_label):
        os.makedirs(d, exist_ok=True)

    label_files = sorted(f for f in os.listdir(src_label) if f.endswith(".png"))
    if not label_files:
        raise RuntimeError(f"no Label/*.png found under {src_label}")
    if limit is not None:
        label_files = label_files[: int(limit)]
    n = len(label_files)
    print(f"Re-encoding {n} label file(s)")
    print(f"  src: {src}")
    print(f"  dst: {dst}")

    processed = []
    missing_imgs = 0
    for i, fn in enumerate(label_files):
        src_lbl = _read_label(os.path.join(src_label, fn))
        new_lbl = _remap(src_lbl, os.path.join(src_label, fn))
        ok = cv2.imwrite(os.path.join(dst_label, fn), new_lbl)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {os.path.join(dst_label, fn)}")

        for sub_src, sub_dst in ((src_rgb, dst_rgb), (src_depth, dst_depth)):
            s = os.path.join(sub_src, fn)
            if not os.path.isfile(s):
                missing_imgs += 1
                print(f"  WARNING: missing source image {s}", file=sys.stderr)
                continue
            _symlink(s, os.path.join(sub_dst, fn), absolute=not relative)

        processed.append(fn)
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{n}")

    # Copy split files verbatim (if present).
    for txt in ("train.txt", "test.txt"):
        s = os.path.join(src, txt)
        if os.path.isfile(s):
            shutil.copyfile(s, os.path.join(dst, txt))
            print(f"  copied {txt}")
        else:
            print(f"  NOTE: {txt} not present in src, skipped")

    if missing_imgs:
        print(f"  WARNING: {missing_imgs} missing source image symlink(s)",
              file=sys.stderr)

    _validate(src_label, dst_label, processed, n_sample=max(20, 25))
    print("OK: re-encode + validation passed")
    return processed


def _validate(src_label, dst_label, processed, n_sample=25):
    """Sample processed labels and assert px-count conservation + value range."""
    if not processed:
        raise RuntimeError("nothing was processed; cannot validate")
    n_sample = min(n_sample, len(processed))
    if n_sample < 20 and len(processed) >= 20:
        n_sample = 20
    rng = random.Random(1234)
    sample = rng.sample(processed, n_sample)
    print(f"Validating a sample of {n_sample} label file(s)...")

    tot_old_bg = tot_old_wire = tot_old_con = 0
    tot_new_bg = tot_new_wire = tot_new_con = 0
    for fn in sample:
        old = _read_label(os.path.join(src_label, fn)).astype(np.int64)
        new = _read_label(os.path.join(dst_label, fn)).astype(np.int64)

        # 1) new labels must be a subset of {0,1,2}.
        mx = int(new.max())
        mn = int(new.min())
        if mx > 2 or mn < 0:
            raise AssertionError(
                f"{fn}: new label range [{mn},{mx}] not within {{0,1,2}}")

        o_bg = int(np.isin(old, OLD_BG).sum())
        o_wire = int(np.isin(old, OLD_WIRE).sum())
        o_con = int(np.isin(old, OLD_CONNECTOR).sum())
        n_bg = int((new == 0).sum())
        n_wire = int((new == 1).sum())
        n_con = int((new == 2).sum())

        # 2) per-file conservation (fail loudly with the offending file).
        if n_con != o_con:
            raise AssertionError(
                f"{fn}: connector px new={n_con} != old(==4)={o_con}")
        if n_wire != o_wire:
            raise AssertionError(
                f"{fn}: wire px new={n_wire} != old(in 1,2,3)={o_wire}")
        if n_bg != o_bg:
            raise AssertionError(
                f"{fn}: bg px new={n_bg} != old(in 0,5)={o_bg}")

        tot_old_bg += o_bg
        tot_old_wire += o_wire
        tot_old_con += o_con
        tot_new_bg += n_bg
        tot_new_wire += n_wire
        tot_new_con += n_con

    # 3) aggregate conservation over the sample.
    assert tot_new_bg == tot_old_bg, (tot_new_bg, tot_old_bg)
    assert tot_new_wire == tot_old_wire, (tot_new_wire, tot_old_wire)
    assert tot_new_con == tot_old_con, (tot_new_con, tot_old_con)

    print(f"  bg px       : new={tot_new_bg}  == old{{0,5}}={tot_old_bg}")
    print(f"  wire px     : new={tot_new_wire}  == old{{1,2,3}}={tot_old_wire}")
    print(f"  connector px: new={tot_new_con}  == old{{4}}={tot_old_con}")
    print(f"  all new label values <= 2: confirmed over {n_sample} files")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="staged dataset dir (RGB/ Depth/ Label/)")
    p.add_argument("--dst", required=True, help="output dataset dir (new)")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N label files (for testing)")
    p.add_argument("--relative", action="store_true",
                   help="make RGB/Depth symlinks relative instead of absolute")
    return p.parse_args()


def main():
    args = parse_args()
    reencode(args.src, args.dst, limit=args.limit, relative=args.relative)


if __name__ == "__main__":
    main()
