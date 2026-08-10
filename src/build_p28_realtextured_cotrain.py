#!/usr/bin/env python
"""Build the P28 real-textured 3-way co-train dataset =
  P25 best-synth (phase15_wirefree)  +  MovingCables  +  NEW real-textured pool,
symlink-merged into one trainer dir.

Mirrors ``src/build_p28_cotrain.py`` / ``src/build_p28_rmdlo_cotrain.py`` with distinct
set-id offsets per source so they never collide under ``filter_indices_by_set``
(int(basename.split('_')[0])):

    synth          set-ids  0 .. N            (unchanged)
    MovingCables   set-ids  +1000             (--mc-offset)
    real-textured  set-ids  +3000 .. +6000    (OVERSAMPLED: --rt-mult replicas, each
                                               replica block offset by +1000 so the
                                               oversampled copies are distinct sets)

All three sources use legacy-mode Label encoding (synth {0..5}, MC {0,4}, real-textured
{0,4}; every label max>=3) so build_cache reads them all in one shared legacy cache.

The real-textured pool is SMALL, so its TRAIN frames are oversampled by an integer
``--rt-mult`` (capped at 4) toward ~10% of total train frames. Each replica is symlinked
under a distinct +1000 set-id block (sharing the same underlying PNG via symlink) and
referenced ``--rt-mult`` times in train.txt. Real-textured VAL frames are kept once.
"""
import argparse
import glob
import os

SYNTH = "data/dformer_dataset_phase15_wirefree"
MC = "data/dformer_dataset_movingcables"
RT = "data/dformer_dataset_realtextured"
MC_OFFSET = 1000
RT_OFFSET = 3000          # first real-textured replica block
RT_REPLICA_STRIDE = 1000  # each oversample replica gets +1000 more


def offset_base(base, offset):
    parts = base.split("_")
    parts[0] = str(int(parts[0]) + offset)
    return "_".join(parts)


def link_subset(src, dst, bases, offset):
    """Symlink RGB/Depth/Label PNGs for ``bases`` from src into dst at set-id +offset.
    Returns the remapped 'RGB/<newbase>.png' lines."""
    for sub in ("RGB", "Depth", "Label"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
    lines = []
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
    out = []
    if not os.path.exists(list_path):
        return out
    with open(list_path) as f:
        for l in f:
            l = l.strip()
            if l:
                out.append(l.split("/", 1)[1][:-4])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/dformer_dataset_p28_realtextured_cotrain")
    ap.add_argument("--synth", default=SYNTH)
    ap.add_argument("--mc", default=MC)
    ap.add_argument("--rt", default=RT)
    ap.add_argument("--rt-mult", type=int, default=4,
                    help="real-textured oversample multiplier (capped at 4)")
    ap.add_argument("--mc-offset", type=int, default=MC_OFFSET)
    ap.add_argument("--rt-offset", type=int, default=RT_OFFSET)
    args = ap.parse_args()

    rt_mult = max(1, min(4, args.rt_mult))

    for name, d in (("synth", args.synth), ("mc", args.mc), ("rt", args.rt)):
        if not os.path.isdir(os.path.join(d, "RGB")):
            raise SystemExit(f"{name} dataset not found at {d} — run its converter first")

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    synth_tr = read_bases(os.path.join(args.synth, "train.txt"))
    synth_va = read_bases(os.path.join(args.synth, "test.txt"))
    mc_tr = read_bases(os.path.join(args.mc, "train.txt"))
    mc_va = read_bases(os.path.join(args.mc, "test.txt"))
    rt_tr = read_bases(os.path.join(args.rt, "train.txt"))
    rt_va = read_bases(os.path.join(args.rt, "test.txt"))

    print("symlinking synth ...")
    s_tr = link_subset(args.synth, out, synth_tr, 0)
    s_va = link_subset(args.synth, out, synth_va, 0)
    print(f"symlinking MovingCables (set-id +{args.mc_offset}) ...")
    m_tr = link_subset(args.mc, out, mc_tr, args.mc_offset)
    m_va = link_subset(args.mc, out, mc_va, args.mc_offset)

    # real-textured: rt_mult replicas of TRAIN (distinct +1000 set-id blocks), VAL once.
    print(f"symlinking real-textured (oversample x{rt_mult}, "
          f"set-id +{args.rt_offset} .. +{args.rt_offset + (rt_mult-1)*RT_REPLICA_STRIDE}) ...")
    r_tr = []
    for k in range(rt_mult):
        off = args.rt_offset + k * RT_REPLICA_STRIDE
        r_tr += link_subset(args.rt, out, rt_tr, off)
    # val: single copy, at the base rt offset
    r_va = link_subset(args.rt, out, rt_va, args.rt_offset)

    train = s_tr + m_tr + r_tr
    val = s_va + m_va + r_va

    # IMPORTANT: train.txt FIRST line must reference a frame whose Label max>=3 so
    # build_cache picks legacy/three-way mode. synth frames are first and synth labels
    # use {0..5} (max>=3), so the first line is already legacy-safe. Do NOT reshuffle.
    with open(os.path.join(out, "train.txt"), "w") as f:
        f.write("\n".join(train) + "\n")
    with open(os.path.join(out, "test.txt"), "w") as f:
        f.write("\n".join(val) + "\n")

    rt_unique = len(rt_tr)
    rt_eff = len(r_tr)
    total = len(train)
    print("\n=== P28 REAL-TEXTURED 3-WAY CO-TRAIN BUILT ===")
    print(f"out-dir: {out}")
    print(f"train: {total}  = synth {len(s_tr)} + MC {len(m_tr)} + real-textured {rt_eff}")
    print(f"val:   {len(val)}  = synth {len(s_va)} + MC {len(m_va)} + real-textured {len(r_va)}")
    print(f"real-textured: unique-train={rt_unique}  multiplier=x{rt_mult}  "
          f"effective={rt_eff}  share={100*rt_eff/max(1,total):.2f}% of train")
    if len(s_tr):
        print(f"ratios (train) synth:MC:RT = "
              f"1 : {len(m_tr)/len(s_tr):.2f} : {rt_eff/len(s_tr):.3f}")


if __name__ == "__main__":
    main()
