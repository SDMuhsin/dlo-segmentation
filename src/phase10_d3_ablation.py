"""Phase 10 D3 — Ablation: dlow=6 vs dlow=3 on Phase 9 dataset.

Primary deliverable: Run inference on all 4 real videos with the
Phase 9 dlow=6 best_model checkpoint (full_20260515_0509). Same Phase 9
dataset & recipe as dlow=3 except --dlo-weight 6.0 (matching Phase 7).

Outputs:
  D3/dlow6/real_video_<i>/per_frame.csv
  D3/dlow6/real_video_<i>/side_by_side.mp4  (and side_by_side.gif)
  D3/ablation_summary.json
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DLOW6 = os.path.join(PROJECT_ROOT, "results", "segformer_b5_rgb_phase9",
                          "full_20260515_0509", "best_model.pth")
CKPT_DLOW3 = os.path.join(PROJECT_ROOT, "results", "segformer_b5_rgb_phase9",
                          "full_dlow3_20260515_1330", "best_model.pth")
CKPT_P7 = os.path.join(PROJECT_ROOT, "results", "segformer_b5_rgb",
                       "full_20260430_2032", "best_model.pth")
OUT_DLOW6 = os.path.join(PROJECT_ROOT, "results", "phase10_investigation", "D3", "dlow6")

VIDEOS = {
    1: os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_1.mp4"),
    2: os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_2.mp4"),
    3: os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_3.mp4"),
    4: os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_4.mp4"),
}
INFER = os.path.join(PROJECT_ROOT, "src", "infer_video_rgb_only.py")

CANONICAL = {
    "p7":         {1: 6.77, 2: 12.37, 3: 12.53, 4: 12.28},
    "p9_dlow3":   {1: 2.88, 2: 2.52, 3: 0.37,  4: 0.97},
}


def coverage_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return float(df["dlo_coverage_pct"].mean()), int(len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(OUT_DLOW6, exist_ok=True)

    # Run dlow=6 inference on each of the 4 real videos
    for i, vp in VIDEOS.items():
        out_dir = os.path.join(OUT_DLOW6, f"real_video_{i}")
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(os.path.join(out_dir, "per_frame.csv")):
            print(f"[skip] dlow=6 sample_{i} already done at {out_dir}")
            continue
        print(f"\n=== dlow=6 sample_{i}: {vp} -> {out_dir} ===")
        cmd = [
            sys.executable, INFER,
            "--video", vp,
            "--ckpt", CKPT_DLOW6,
            "--out-dir", out_dir,
            "--device", args.device,
            "--also-gif",
        ]
        r = subprocess.run(cmd, check=True)

    # Aggregate
    table = {"models": ["Phase 7 teacher (canonical)",
                        "Phase 9 dlow=3 final (canonical)",
                        "Phase 9 dlow=6 ep49 (measured)"],
             "rows": {}}
    p7_cov = CANONICAL["p7"]
    p9_3_cov = CANONICAL["p9_dlow3"]
    p9_6_cov = {}
    for i in [1, 2, 3, 4]:
        csv_p = os.path.join(OUT_DLOW6, f"real_video_{i}", "per_frame.csv")
        mean_cov, n = coverage_from_csv(csv_p)
        p9_6_cov[i] = mean_cov

    table["rows"]["Phase 7 teacher (canonical)"] = p7_cov
    table["rows"]["Phase 9 dlow=3 final (canonical)"] = p9_3_cov
    table["rows"]["Phase 9 dlow=6 ep49 (measured)"] = p9_6_cov

    # Decision rule
    def relative_delta(meas, ref):
        if ref == 0:
            return float("inf")
        return (meas - ref) / ref

    avg_p7 = sum(p7_cov.values()) / 4
    avg_p9_3 = sum(p9_3_cov.values()) / 4
    avg_p9_6 = sum(p9_6_cov.values()) / 4

    # Per-video relative deltas to P7 and to P9 dlow=3
    rel_to_p7 = {i: relative_delta(p9_6_cov[i], p7_cov[i]) for i in [1, 2, 3, 4]}
    rel_to_p9_3 = {i: relative_delta(p9_6_cov[i], p9_3_cov[i]) for i in [1, 2, 3, 4]}

    # Decision per-video and aggregate
    close_to_p7 = all(abs(d) <= 0.20 for d in rel_to_p7.values())
    close_to_p9_3 = all(abs(d) <= 0.20 for d in rel_to_p9_3.values())
    avg_rel_p7 = relative_delta(avg_p9_6, avg_p7)
    avg_rel_p9_3 = relative_delta(avg_p9_6, avg_p9_3)
    avg_close_to_p7 = abs(avg_rel_p7) <= 0.20
    avg_close_to_p9_3 = abs(avg_rel_p9_3) <= 0.20

    if close_to_p7 and not close_to_p9_3:
        outcome = "loss_weight_was_the_cause_dataset_is_fine"
    elif close_to_p9_3 and not close_to_p7:
        outcome = "loss_weight_is_NOT_the_cause_dataset_is_the_cause"
    elif close_to_p7 and close_to_p9_3:
        outcome = "ambiguous_within_20pct_of_both"
    else:
        outcome = "mixed_effect_neither_within_20pct"

    avg_outcome = (
        "loss_weight_was_the_cause_dataset_is_fine" if avg_close_to_p7 and not avg_close_to_p9_3 else
        "loss_weight_is_NOT_the_cause_dataset_is_the_cause" if avg_close_to_p9_3 and not avg_close_to_p7 else
        "ambiguous_within_20pct_of_both" if avg_close_to_p7 and avg_close_to_p9_3 else
        "mixed_effect_neither_within_20pct"
    )

    summary = {
        "checkpoints": {
            "p7": CKPT_P7,
            "p9_dlow3": CKPT_DLOW3,
            "p9_dlow6": CKPT_DLOW6,
        },
        "per_video_coverage_pct": {
            "p7_canonical": p7_cov,
            "p9_dlow3_canonical": p9_3_cov,
            "p9_dlow6_measured": p9_6_cov,
        },
        "avg_coverage_pct": {
            "p7_canonical": avg_p7,
            "p9_dlow3_canonical": avg_p9_3,
            "p9_dlow6_measured": avg_p9_6,
        },
        "relative_delta_dlow6_to_p7": rel_to_p7,
        "relative_delta_dlow6_to_p9_dlow3": rel_to_p9_3,
        "average_relative_delta_dlow6_to_p7": avg_rel_p7,
        "average_relative_delta_dlow6_to_p9_dlow3": avg_rel_p9_3,
        "decision_rule_per_video_threshold_pm20pct": {
            "all_within_20pct_of_p7": close_to_p7,
            "all_within_20pct_of_p9_dlow3": close_to_p9_3,
            "outcome": outcome,
        },
        "decision_rule_on_average_threshold_pm20pct": {
            "avg_within_20pct_of_p7": avg_close_to_p7,
            "avg_within_20pct_of_p9_dlow3": avg_close_to_p9_3,
            "outcome": avg_outcome,
        },
    }

    out_json = os.path.join(PROJECT_ROOT, "results", "phase10_investigation", "D3",
                            "ablation_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {out_json}")
    print("\nTable (mean DLO coverage % across the full video):")
    print(f"  {'Model':50s}  S1     S2     S3     S4     avg")
    print(f"  {'Phase 7 teacher (canonical)':50s}  "
          f"{p7_cov[1]:5.2f}  {p7_cov[2]:5.2f}  {p7_cov[3]:5.2f}  {p7_cov[4]:5.2f}  {avg_p7:5.2f}")
    print(f"  {'Phase 9 dlow=3 final (canonical)':50s}  "
          f"{p9_3_cov[1]:5.2f}  {p9_3_cov[2]:5.2f}  {p9_3_cov[3]:5.2f}  {p9_3_cov[4]:5.2f}  {avg_p9_3:5.2f}")
    print(f"  {'Phase 9 dlow=6 ep49 (measured)':50s}  "
          f"{p9_6_cov[1]:5.2f}  {p9_6_cov[2]:5.2f}  {p9_6_cov[3]:5.2f}  {p9_6_cov[4]:5.2f}  {avg_p9_6:5.2f}")
    print(f"\nDecision (per-video, threshold ±20%): {outcome}")
    print(f"Decision (on average, threshold ±20%): {avg_outcome}")


if __name__ == "__main__":
    main()
