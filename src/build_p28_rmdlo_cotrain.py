#!/usr/bin/env python
"""Build the P28 3-way co-train dataset = P25 best-synth (phase15_wirefree)
+ MovingCables + re-skinned RMDLO, symlink-merged into one trainer dir.

Extends ``src/build_p28_cotrain.py`` (synth + MovingCables) with a THIRD source,
the re-skinned RMDLO dataset, using distinct set-id offsets so the three sources
never collide under ``filter_indices_by_set`` (int(basename.split('_')[0])):

    synth     set-ids  0 ..  N          (unchanged)
    MC        set-ids  +1000            (offset_base)
    RMDLO     set-ids  +2000            (offset_base, --rmdlo-offset)

All three sources already use legacy-mode-compatible Label encoding (synth {0..5},
MovingCables {0,4}, RMDLO {0,4}; every label max>=3) so build_cache reads them all
in one shared legacy cache.

RMDLO is a deliberate MINORITY of the co-train (segmented-rope-silhouette risk):
``--rmdlo-share`` controls how many RMDLO TRAIN frames are linked, expressed as a
fraction of the MovingCables train count (default 0.5 = RMDLO ≈ half of MC). All
RMDLO val frames (if any) are kept. Set ``--rmdlo-share`` to a value > the natural
ratio (or use ``--all-rmdlo``) to take every RMDLO frame.
"""
import argparse
import glob
import os

SYNTH = "data/dformer_dataset_phase15_wirefree"
MC = "data/dformer_dataset_movingcables"
RMDLO = "data/dformer_dataset_rmdlo"
MC_OFFSET = 1000
RMDLO_OFFSET = 2000


def offset_base(base, offset):
    # 000_0000_00_xx.png -> {offset}_0000_00_xx.png
    parts = base.split("_")
    parts[0] = str(int(parts[0]) + offset)
    return "_".join(parts)


def link_subset(src, dst, bases, offset):
    """Symlink the given basenames' RGB/Depth/Label PNGs from src into dst,
    offsetting the set-id. ``bases`` are basenames WITHOUT extension, as they
    exist in src. Returns the list of remapped 'RGB/<newbase>.png' lines."""
    lines = []
    for sub in ("RGB", "Depth", "Label"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
    for base in bases:
        newbase = offset_base(base, offset)
        for sub in ("RGB", "Depth", "Label"):
            sp = os.path.join(src, sub, base + ".png")
            if not os.path.exists(sp):
                raise SystemExit(f"missing {sp}")
            link = os.path.join(dst, sub, newbase + ".png")
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(os.path.abspath(sp), link)
        lines.append(f"RGB/{newbase}.png")
    return lines


def read_bases(list_path):
    """Read a train.txt/test.txt -> [basename-without-ext]."""
    out = []
    if not os.path.exists(list_path):
        return out
    with open(list_path) as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            out.append(l.split("/", 1)[1][:-4])   # 'RGB/<base>.png' -> <base>
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/dformer_dataset_p28_rmdlo_cotrain")
    ap.add_argument("--synth", default=SYNTH)
    ap.add_argument("--mc", default=MC)
    ap.add_argument("--rmdlo", default=RMDLO)
    ap.add_argument("--rmdlo-share", type=float, default=0.5,
                    help="RMDLO train frames as a fraction of MC train count "
                         "(default 0.5 = RMDLO ~ half of MovingCables)")
    ap.add_argument("--all-rmdlo", action="store_true",
                    help="take every RMDLO frame (ignore --rmdlo-share cap)")
    ap.add_argument("--rmdlo-offset", type=int, default=RMDLO_OFFSET)
    ap.add_argument("--mc-offset", type=int, default=MC_OFFSET)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    for name, d in (("synth", args.synth), ("mc", args.mc), ("rmdlo", args.rmdlo)):
        if not os.path.isdir(os.path.join(d, "RGB")):
            raise SystemExit(f"{name} dataset not found at {d} — run its converter first")

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    # --- read source file lists as basenames ---
    synth_tr = read_bases(os.path.join(args.synth, "train.txt"))
    synth_va = read_bases(os.path.join(args.synth, "test.txt"))
    mc_tr = read_bases(os.path.join(args.mc, "train.txt"))
    mc_va = read_bases(os.path.join(args.mc, "test.txt"))
    rm_tr = read_bases(os.path.join(args.rmdlo, "train.txt"))
    rm_va = read_bases(os.path.join(args.rmdlo, "test.txt"))

    # --- cap RMDLO train share (deliberate minority) ---
    import numpy as np
    rng = np.random.default_rng(args.seed)
    if not args.all_rmdlo and mc_tr:
        cap = int(round(len(mc_tr) * args.rmdlo_share))
        if len(rm_tr) > cap:
            idx = rng.permutation(len(rm_tr))[:cap]
            rm_tr = [rm_tr[i] for i in sorted(idx)]

    print("symlinking synth ...")
    s_tr = link_subset(args.synth, out, synth_tr, 0)
    s_va = link_subset(args.synth, out, synth_va, 0)
    print(f"symlinking MovingCables (set-id +{args.mc_offset}) ...")
    m_tr = link_subset(args.mc, out, mc_tr, args.mc_offset)
    m_va = link_subset(args.mc, out, mc_va, args.mc_offset)
    print(f"symlinking RMDLO (set-id +{args.rmdlo_offset}) ...")
    r_tr = link_subset(args.rmdlo, out, rm_tr, args.rmdlo_offset)
    r_va = link_subset(args.rmdlo, out, rm_va, args.rmdlo_offset)

    train = s_tr + m_tr + r_tr
    val = s_va + m_va + r_va
    with open(os.path.join(out, "train.txt"), "w") as f:
        f.write("\n".join(train) + "\n")
    with open(os.path.join(out, "test.txt"), "w") as f:
        f.write("\n".join(val) + "\n")

    print("\n=== P28 3-WAY CO-TRAIN BUILT ===")
    print(f"out-dir: {out}")
    print(f"train: {len(train)}  = synth {len(s_tr)} + MC {len(m_tr)} + RMDLO {len(r_tr)}")
    print(f"val:   {len(val)}  = synth {len(s_va)} + MC {len(m_va)} + RMDLO {len(r_va)}")
    if m_tr:
        print(f"ratios (train) synth:MC:RMDLO = "
              f"1 : {len(m_tr)/max(1,len(s_tr)):.2f} : {len(r_tr)/max(1,len(s_tr)):.2f}"
              f"   (RMDLO = {len(r_tr)/max(1,len(m_tr)):.2f}x MC)")


if __name__ == "__main__":
    main()
