#!/usr/bin/env python
"""Build the R2 co-train dataset = P25 best-synth (phase15_wirefree) + MovingCables,
mixed ~1:1, in the trainer's RGB/Depth/Label + train.txt/test.txt layout.

Strategy: symlink both source datasets' PNGs into one output dir. The synth set-IDs
(0..35) stay as-is; MovingCables set-IDs are offset by +1000 so they never collide
with synth sets (filter_indices_by_set parses int(basename.split('_')[0])). train.txt
and test.txt are the concatenation of the two sources' lists. Both sources already use
legacy-mode-compatible Label encoding (synth {0..5}, MovingCables {0,4}; both max>=3),
so build_cache reads both correctly with one shared cache.
"""
import os
import sys
import argparse
import glob

SYNTH = "data/dformer_dataset_phase15_wirefree"
MC = "data/dformer_dataset_movingcables"
MC_OFFSET = 1000


def link_all(src, dst, rename=None):
    """Symlink every PNG in src/{RGB,Depth,Label} into dst, optionally renaming base."""
    for sub in ("RGB", "Depth", "Label"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
        for p in glob.glob(os.path.join(src, sub, "*.png")):
            base = os.path.basename(p)
            if rename is not None:
                base = rename(base)
            link = os.path.join(dst, sub, base)
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(os.path.abspath(p), link)


def offset_base(base):
    # 000_0000_00_mc.png -> 1000_0000_00_mc.png
    parts = base.split("_")
    parts[0] = str(int(parts[0]) + MC_OFFSET)
    return "_".join(parts)


def remap_list(lines, rename=None):
    out = []
    for l in lines:
        l = l.strip()
        if not l:
            continue
        # "RGB/<base>.png"
        pre, base = l.split("/", 1)
        if rename is not None:
            base = rename(base)
        out.append(f"{pre}/{base}")
    return out


def read_lines(path):
    with open(path) as f:
        return [x.strip() for x in f if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/dformer_dataset_p28_cotrain")
    ap.add_argument("--synth", default=SYNTH)
    ap.add_argument("--mc", default=MC)
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(args.mc, "RGB")):
        raise SystemExit(f"MovingCables dataset not found at {args.mc} — run the converter first")

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    print("symlinking synth ...")
    link_all(args.synth, out, rename=None)
    print("symlinking MovingCables (set-id +%d) ..." % MC_OFFSET)
    link_all(args.mc, out, rename=offset_base)

    synth_tr = remap_list(read_lines(os.path.join(args.synth, "train.txt")))
    synth_va = remap_list(read_lines(os.path.join(args.synth, "test.txt")))
    mc_tr = remap_list(read_lines(os.path.join(args.mc, "train.txt")), rename=offset_base)
    mc_va = remap_list(read_lines(os.path.join(args.mc, "test.txt")), rename=offset_base)

    train = synth_tr + mc_tr
    val = synth_va + mc_va
    with open(os.path.join(out, "train.txt"), "w") as f:
        f.write("\n".join(train) + "\n")
    with open(os.path.join(out, "test.txt"), "w") as f:
        f.write("\n".join(val) + "\n")

    print("\n=== R2 CO-TRAIN BUILT ===")
    print(f"out-dir: {out}")
    print(f"train: {len(train)}  (synth {len(synth_tr)} + MovingCables {len(mc_tr)}; "
          f"ratio synth:mc = 1 : {len(mc_tr)/max(1,len(synth_tr)):.2f})")
    print(f"val:   {len(val)}  (synth {len(synth_va)} + MovingCables {len(mc_va)})")


if __name__ == "__main__":
    main()
