#!/usr/bin/env python
"""Build the P29 WARM-START co-train dataset =
  CURRENT BEST MIX  (data/dformer_dataset_p28_realtextured_cotrain = synth + MovingCables
                     + real-textured)
  +  the NEW targeted COPLANAR HARD-NEGATIVES (data/dformer_dataset_coplanar_hardneg),
     OVERSAMPLED so the negatives are ~12-15% of total train frames (oversample <= 4x),
symlink-merged into one trainer dir. The val/test split is the realtextured co-train's
test.txt UNCHANGED (so eval stays comparable).

Mirrors src/build_p28_realtextured_cotrain.py:
  - every frame uses legacy-mode Label encoding (synth {0..5}, MC {0,4}, RT {0,4} all have
    label max>=3); the hard-NEGATIVES are all-zero (max==0). build_cache auto-detects label
    mode from the FIRST train.txt line's max, so the negatives MUST NOT be line 1 — we
    copy the base train.txt order verbatim (synth frame is line 1, max=5) and APPEND the
    oversampled negatives, guaranteeing a legacy-safe line 1.
  - set-id collision avoidance: the base mix already occupies set-ids 0..~6264 (synth 0-999,
    MC +1000, RT +3000..+6000). The negatives are placed far above that at +NEG_OFFSET
    (10000), each oversample replica block offset by +NEG_REPLICA_STRIDE (1000) so the
    replicated copies are distinct sets under filter_indices_by_set.

The negatives' OWN val frames (coplanar_hardneg/test.txt) are NOT added to this co-train's
val (eval comparability is preserved against the realtextured baseline); they are held in
the standalone data/dformer_dataset_coplanar_hardneg for separate inspection if wanted.
"""
import argparse
import os

BASE = "data/dformer_dataset_p28_realtextured_cotrain"   # current best mix
NEG = "data/dformer_dataset_coplanar_hardneg"            # new hard-negatives
NEG_OFFSET = 10000
NEG_REPLICA_STRIDE = 1000


def offset_base(base, offset):
    parts = base.split("_")
    parts[0] = str(int(parts[0]) + offset)
    return "_".join(parts)


def read_bases(list_path):
    out = []
    with open(list_path) as f:
        for l in f:
            l = l.strip()
            if l:
                out.append(l.split("/", 1)[1][:-4])
    return out


def link_one(src, dst, base, newbase):
    for sub in ("RGB", "Depth", "Label"):
        sp = os.path.join(src, sub, base + ".png")
        if not os.path.exists(sp):
            raise SystemExit(f"missing {sp}")
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
        link = os.path.join(dst, sub, newbase + ".png")
        if os.path.lexists(link):
            os.remove(link)
        # resolve through any existing symlink in the BASE so the new link points at the
        # real underlying PNG (the base mix is itself a symlink farm).
        real = os.path.realpath(sp)
        os.symlink(real, link)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/dformer_dataset_p29_coplanar_cotrain")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--neg", default=NEG)
    ap.add_argument("--neg-mult", type=int, default=4,
                    help="hard-negative oversample multiplier (capped at 4)")
    args = ap.parse_args()

    neg_mult = max(1, min(4, args.neg_mult))
    out = args.out_dir

    for name, d in (("base", args.base), ("neg", args.neg)):
        if not os.path.isdir(os.path.join(d, "RGB")):
            raise SystemExit(f"{name} dataset not found at {d}")

    # CRITICAL: remove any stale prior dataset dir / cache under the new name so build_cache
    # cannot reuse a cache built for a different frame set.
    if os.path.exists(out):
        import shutil
        shutil.rmtree(out)
    os.makedirs(out)

    # --- base mix: copy train.txt order VERBATIM (line 1 = synth, label max=5 -> legacy) ---
    base_tr = read_bases(os.path.join(args.base, "train.txt"))
    base_va = read_bases(os.path.join(args.base, "test.txt"))
    print(f"base mix: train {len(base_tr)}  val {len(base_va)}")
    for b in base_tr + base_va:
        link_one(args.base, out, b, b)   # same set-ids, base occupies 0..~6264
    base_tr_lines = [f"RGB/{b}.png" for b in base_tr]
    base_va_lines = [f"RGB/{b}.png" for b in base_va]

    # --- hard-negatives: TRAIN oversampled x neg_mult at distinct +NEG_OFFSET blocks ---
    neg_tr = read_bases(os.path.join(args.neg, "train.txt"))
    print(f"hard-negatives: unique-train {len(neg_tr)}  oversample x{neg_mult}")
    neg_lines = []
    for k in range(neg_mult):
        off = NEG_OFFSET + k * NEG_REPLICA_STRIDE
        for b in neg_tr:
            nb = offset_base(b, off)
            link_one(args.neg, out, b, nb)
            neg_lines.append(f"RGB/{nb}.png")

    # train.txt = base order (legacy-safe line 1) + appended negatives. val = base val UNCHANGED.
    train = base_tr_lines + neg_lines
    val = base_va_lines
    with open(os.path.join(out, "train.txt"), "w") as f:
        f.write("\n".join(train) + "\n")
    with open(os.path.join(out, "test.txt"), "w") as f:
        f.write("\n".join(val) + "\n")

    neg_eff = len(neg_lines)
    total = len(train)
    print("\n=== P29 COPLANAR WARM-START CO-TRAIN BUILT ===")
    print(f"out-dir: {out}")
    print(f"train: {total}  = base {len(base_tr_lines)} + hard-neg {neg_eff}")
    print(f"val:   {len(val)}  (= realtextured co-train test.txt, UNCHANGED)")
    print(f"hard-negatives: unique={len(neg_tr)}  x{neg_mult}  effective={neg_eff}  "
          f"share={100*neg_eff/total:.2f}% of train")
    print(f"line-1 train base: {train[0]}  (must be a wire frame, label max>=3)")


if __name__ == "__main__":
    main()
