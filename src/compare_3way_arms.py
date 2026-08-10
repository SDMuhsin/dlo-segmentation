"""Compare 3-way training arms on the STABLE val level, not on a single peak.

The project has a documented failure mode where a one-epoch winner's-curse
spike was mistaken for a real gain (the epoch_15 selection-spike finding). So
the headline comparison here is the MEAN OF THE LAST N EVALUATIONS, which a
single lucky epoch cannot move, and the per-epoch curve is printed so a reader
can see for themselves that val is flat-or-rising rather than degrading.

Usage:
  python src/compare_3way_arms.py \
      baseline=results/.../seg_b5_3way_connscale3/train_connscale3.log \
      lovasz05=results/.../p3w_lovasz/lovasz05/train.log
"""
from __future__ import annotations

import re
import sys

VAL_RE = re.compile(
    r"val:\s*mIoU=([0-9.]+)\s+IoU\(wire\)=([0-9.]+)\s+IoU\(bg\)=([0-9.]+)\s+"
    r"IoU\(con\)=([0-9.]+)"
)
TRAIN_LOSS_RE = re.compile(r"ep\s+(\d+)\s+batch.*loss=([0-9.]+)")


def parse(path):
    evals, losses = [], {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = VAL_RE.search(line)
            if m:
                evals.append({
                    "miou": float(m.group(1)),
                    "wire": float(m.group(2)),
                    "bg": float(m.group(3)),
                    "con": float(m.group(4)),
                })
            m2 = TRAIN_LOSS_RE.search(line)
            if m2:
                losses.setdefault(int(m2.group(1)), []).append(float(m2.group(2)))
    return evals, losses


def stable(evals, n=5):
    tail = evals[-n:]
    if not tail:
        return None
    return {k: sum(e[k] for e in tail) / len(tail) for k in ("miou", "wire", "con", "bg")}


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    arms = {}
    for a in args:
        name, path = a.split("=", 1)
        evals, losses = parse(path)
        arms[name] = (evals, losses)

    n_tail = 5
    print(f"\n{'='*78}\nSTABLE LEVEL — mean of the last {n_tail} evaluations "
          f"(spike-proof headline)\n{'='*78}")
    print(f"{'arm':<16}{'n_ev':>5}{'IoU(con)':>11}{'IoU(wire)':>11}"
          f"{'mIoU':>10}{'IoU(bg)':>10}")
    base = None
    for name, (evals, _) in arms.items():
        s = stable(evals, n_tail)
        if s is None:
            print(f"{name:<16}{0:>5}   (no eval lines yet)")
            continue
        if base is None:
            base = s
        print(f"{name:<16}{len(evals):>5}{s['con']:>11.4f}{s['wire']:>11.4f}"
              f"{s['miou']:>10.4f}{s['bg']:>10.4f}")

    if base is not None and len(arms) > 1:
        print(f"\n{'-'*78}\nDELTA vs first arm (the baseline)\n{'-'*78}")
        print(f"{'arm':<16}{'d IoU(con)':>13}{'d IoU(wire)':>13}{'d mIoU':>11}")
        for name, (evals, _) in list(arms.items())[1:]:
            s = stable(evals, n_tail)
            if s is None:
                continue
            print(f"{name:<16}{s['con']-base['con']:>+13.4f}"
                  f"{s['wire']-base['wire']:>+13.4f}{s['miou']-base['miou']:>+11.4f}")

    print(f"\n{'='*78}\nPER-EVAL CURVES (overfitting check: val must not decline)"
          f"\n{'='*78}")
    for name, (evals, losses) in arms.items():
        print(f"\n-- {name} --")
        print(f"{'eval':>5}{'IoU(con)':>11}{'IoU(wire)':>11}{'mIoU':>10}"
              f"{'train_loss':>12}")
        ep_keys = sorted(losses)
        for i, e in enumerate(evals):
            # evals happen every --eval-every epochs; map index -> epoch bucket
            lo = ""
            if ep_keys:
                j = min(int((i + 1) * len(ep_keys) / max(len(evals), 1)) - 1,
                        len(ep_keys) - 1)
                if j >= 0:
                    vals = losses[ep_keys[j]]
                    lo = f"{sum(vals)/len(vals):>12.4f}"
            print(f"{i+1:>5}{e['con']:>11.4f}{e['wire']:>11.4f}{e['miou']:>10.4f}{lo}")
        if len(evals) >= 4:
            first_half = evals[len(evals)//2:]
            best_con = max(e["con"] for e in evals)
            last_con = evals[-1]["con"]
            print(f"   peak IoU(con)={best_con:.4f} at eval "
                  f"{1+max(range(len(evals)), key=lambda i: evals[i]['con'])}; "
                  f"final={last_con:.4f}; "
                  f"peak-minus-final={best_con-last_con:+.4f} "
                  f"(large positive = spiky//overfit-ish, ~0 = converged flat)")


if __name__ == "__main__":
    main()
