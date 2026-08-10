#!/usr/bin/env python
"""Build the P41 HALF-DOSE cutout-negative co-train dataset.

IDENTICAL to ``src/build_p39_cutoutneg_cotrain.py`` EXCEPT only the first ``--max-sub``
(default 900) of the 1800 substituted bases keep their ``_p39neg`` composite; the rest
revert to the CLEAN phase15_wirefree synth frame. This is a pure DOSE-reduction control:
same recipe, same warm-start, same total train.txt count (16623) — only the NUMBER of
negative-composite frames changes (1800 -> 900). Deterministic subset (sorted order) for
reproducibility. Matched A/B vs P39 (full dose) tests whether the recall cost is
volume-driven (favorable trade => frontier expansion reachable) or 1:1-coupled (ceiling).

CPU only. No torch / no GPU.
"""
import argparse, os

SYNTH = "data/dformer_dataset_phase15_wirefree"
MC = "data/dformer_dataset_movingcables"
RT = "data/dformer_dataset_realtextured"
NEG = "data/dformer_dataset_phase15_wirefree_p39neg"
MC_OFFSET = 1000
RT_OFFSET = 3000
RT_REPLICA_STRIDE = 1000


def offset_base(base, offset):
    parts = base.split("_")
    parts[0] = str(int(parts[0]) + offset)
    return "_".join(parts)


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


def symlink(sp, link):
    if not os.path.exists(sp):
        raise SystemExit(f"missing {sp}")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.abspath(sp), link)


def link_subset(src, dst, bases, offset, sub_map=None, neg_dir=NEG):
    for sub in ("RGB", "Depth", "Label"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
    lines, subbed = [], []
    for base in bases:
        newbase = offset_base(base, offset)
        use_sub = sub_map is not None and base in sub_map
        for sub in ("RGB", "Depth", "Label"):
            if use_sub:
                sp = os.path.join(neg_dir, sub, base + "_p39neg.png")
                if not os.path.exists(sp):
                    sp = os.path.join(src, sub, base + ".png")
            else:
                sp = os.path.join(src, sub, base + ".png")
            symlink(sp, os.path.join(dst, sub, newbase + ".png"))
        if use_sub:
            subbed.append(base)
        lines.append(f"RGB/{newbase}.png")
    return lines, subbed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/dformer_dataset_p41_halfdose_cotrain")
    ap.add_argument("--synth", default=SYNTH)
    ap.add_argument("--mc", default=MC)
    ap.add_argument("--rt", default=RT)
    ap.add_argument("--neg", default=NEG)
    ap.add_argument("--rt-mult", type=int, default=4)
    ap.add_argument("--mc-offset", type=int, default=MC_OFFSET)
    ap.add_argument("--rt-offset", type=int, default=RT_OFFSET)
    ap.add_argument("--max-sub", type=int, default=900,
                    help="keep only the first N (sorted) substituted bases as _p39neg composites")
    args = ap.parse_args()
    neg_dir = args.neg

    rt_mult = max(1, min(4, args.rt_mult))
    for name, d in (("synth", args.synth), ("mc", args.mc), ("rt", args.rt), ("neg", args.neg)):
        if not os.path.isdir(os.path.join(d, "RGB")):
            raise SystemExit(f"{name} dataset not found at {d}")

    sub_list_path = os.path.join(args.neg, "p39_substituted_basenames.txt")
    with open(sub_list_path) as f:
        full_sub = sorted(l.strip() for l in f if l.strip())
    sub_bases = set(full_sub[:args.max_sub])
    print(f"full substitution set: {len(full_sub)}  ->  half-dose keep: {len(sub_bases)} (first {args.max_sub} sorted)")

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    synth_tr = read_bases(os.path.join(args.synth, "train.txt"))
    synth_va = read_bases(os.path.join(args.synth, "test.txt"))
    mc_tr = read_bases(os.path.join(args.mc, "train.txt"))
    mc_va = read_bases(os.path.join(args.mc, "test.txt"))
    rt_tr = read_bases(os.path.join(args.rt, "train.txt"))
    rt_va = read_bases(os.path.join(args.rt, "test.txt"))

    print("symlinking synth (with HALF-DOSE p39neg substitution) ...")
    s_tr, subbed = link_subset(args.synth, out, synth_tr, 0, sub_map=sub_bases, neg_dir=neg_dir)
    s_va, _ = link_subset(args.synth, out, synth_va, 0)
    print(f"  substituted {len(subbed)} synth train frames with _p39neg composites")

    print(f"symlinking MovingCables (set-id +{args.mc_offset}) ...")
    m_tr, _ = link_subset(args.mc, out, mc_tr, args.mc_offset)
    m_va, _ = link_subset(args.mc, out, mc_va, args.mc_offset)

    print(f"symlinking real-textured (oversample x{rt_mult}) ...")
    r_tr = []
    for k in range(rt_mult):
        off = args.rt_offset + k * RT_REPLICA_STRIDE
        lines, _ = link_subset(args.rt, out, rt_tr, off)
        r_tr += lines
    r_va, _ = link_subset(args.rt, out, rt_va, args.rt_offset)

    train = s_tr + m_tr + r_tr
    val = s_va + m_va + r_va

    with open(os.path.join(out, "train.txt"), "w") as f:
        f.write("\n".join(train) + "\n")
    with open(os.path.join(out, "test.txt"), "w") as f:
        f.write("\n".join(val) + "\n")
    with open(os.path.join(out, "p41_substituted_basenames.txt"), "w") as f:
        f.write("\n".join(sorted(subbed)) + "\n")

    print("\n=== P41 HALF-DOSE CUTOUT-NEG CO-TRAIN BUILT ===")
    print(f"out-dir: {out}")
    print(f"train: {len(train)}  = synth {len(s_tr)} + MC {len(m_tr)} + RT {len(r_tr)}")
    print(f"val:   {len(val)}")
    print(f"substituted (p39neg) train frames: {len(subbed)}  (P39 full-dose was 1800)")


if __name__ == "__main__":
    main()
