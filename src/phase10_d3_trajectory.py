"""Phase 10 D3 secondary — trajectory of real-world coverage across Phase 9
dlow=3 epoch checkpoints.

Run inference at epochs {5, 10, 20, 40, 60, 80} on sample_1 (fastest video).
Save per-epoch mean coverage to D3/trajectory_sample1.json.
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCH_DIR = os.path.join(PROJECT_ROOT, "results", "segformer_b5_rgb_phase9",
                         "full_dlow3_20260515_1330")
INFER = os.path.join(PROJECT_ROOT, "src", "infer_video_rgb_only.py")
VIDEO = os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_1.mp4")
OUT_BASE = os.path.join(PROJECT_ROOT, "results", "phase10_investigation", "D3", "trajectory_sample1")

EPOCHS = [5, 10, 20, 40, 60, 80]


def coverage_from_csv(p):
    df = pd.read_csv(p)
    return float(df["dlo_coverage_pct"].mean()), int(len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(OUT_BASE, exist_ok=True)

    results = {}
    for e in EPOCHS:
        ckpt = os.path.join(EPOCH_DIR, f"epoch_{e}.pth")
        if not os.path.exists(ckpt):
            print(f"[skip] no ckpt at {ckpt}")
            continue
        out_dir = os.path.join(OUT_BASE, f"epoch_{e:03d}")
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(os.path.join(out_dir, "per_frame.csv")):
            print(f"=== epoch {e} -> {out_dir} ===")
            cmd = [sys.executable, INFER,
                   "--video", VIDEO,
                   "--ckpt", ckpt,
                   "--out-dir", out_dir,
                   "--device", args.device,
                   "--overlay-only"]
            subprocess.run(cmd, check=True)
        mean_cov, n = coverage_from_csv(os.path.join(out_dir, "per_frame.csv"))
        results[f"epoch_{e}"] = {"mean_coverage_pct": mean_cov, "n_frames": n, "ckpt": ckpt}
        print(f"  epoch {e}: cov={mean_cov:.4f}%  ({n} frames)")

    out_json = os.path.join(PROJECT_ROOT, "results", "phase10_investigation", "D3",
                            "trajectory_sample1.json")
    with open(out_json, "w") as f:
        json.dump({"video": VIDEO,
                   "canonical_dlow3_final": 2.88,
                   "results": results}, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
