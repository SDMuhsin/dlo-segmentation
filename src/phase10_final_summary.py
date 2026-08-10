"""Phase 10 final summary — consolidate D1+D2+D3 numbers into one JSON +
print a markdown-style table that can be copy-pasted into the REPORT.md.
"""
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(PROJECT_ROOT, "results", "phase10_investigation")


def safe_read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    out = {}

    # ---------- D1 ----------
    d1 = safe_read_json(os.path.join(BASE, "D1", "disagreement_stats.json"))
    if d1 is not None:
        out["D1"] = {
            "sanity_coverage_full_video": d1["sanity_coverage_full_video"],
            "canonical": d1["canonical"],
            "per_video_agg": {v: {k: d1["per_video"][v][k] for k in d1["per_video"][v]
                                  if k != "frames"} for v in d1["per_video"]},
        }
        out["D1"]["rgb_at_disagreement"] = {}
        for v in ["sample_1", "sample_2", "sample_3", "sample_4"]:
            p = os.path.join(BASE, "D1", v, "rgb_at_disagreement.json")
            j = safe_read_json(p)
            if j is None:
                continue
            out["D1"]["rgb_at_disagreement"][v] = {
                set_tag: {
                    "n_pixels": j[set_tag]["n_pixels"],
                    "channel_mean_bgr": j[set_tag]["channel_mean_bgr"],
                    "channel_std_bgr": j[set_tag]["channel_std_bgr"],
                }
                for set_tag in ["p7_only", "p9_only", "both"]
            }

    # ---------- D2 ----------
    d2 = safe_read_json(os.path.join(BASE, "D2", "feature_drift_summary.json"))
    if d2 is not None:
        out["D2"] = d2

    # ---------- D3 primary ----------
    d3_primary = safe_read_json(os.path.join(BASE, "D3", "ablation_summary.json"))
    if d3_primary is not None:
        out["D3_primary"] = d3_primary

    # ---------- D3 secondary ----------
    d3_secondary = safe_read_json(os.path.join(BASE, "D3", "trajectory_sample1.json"))
    if d3_secondary is not None:
        out["D3_secondary"] = d3_secondary

    with open(os.path.join(BASE, "final_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- pretty-print key numbers ----
    print("\n=== D1 sanity (mine vs canonical) ===")
    if d1:
        for v in ["sample_1", "sample_2", "sample_3", "sample_4"]:
            s = d1["sanity_coverage_full_video"][v]
            cP7 = d1["canonical"]["p7"][v]
            cP9 = d1["canonical"]["p9_dlow3"][v]
            print(f"  {v}: P7 {s['p7_mean_coverage_pct']:7.4f} vs {cP7:5.2f} (canon); "
                  f"P9 {s['p9_mean_coverage_pct']:7.4f} vs {cP9:5.2f} (canon)")

    print("\n=== D1 disagreement at 20 sampled frames (per-video means, %) ===")
    if d1:
        for v in ["sample_1", "sample_2", "sample_3", "sample_4"]:
            d = d1["per_video"][v]
            print(f"  {v}: P7_only={d['mean_p7_only_pct_at_20_sampled']:6.2f}  "
                  f"P9_only={d['mean_p9_only_pct_at_20_sampled']:6.2f}  "
                  f"both={d['mean_both_pct_at_20_sampled']:6.2f}  "
                  f"IoU(P7,P9)={d['mean_iou_p7_vs_p9_at_20_sampled']:.3f}  "
                  f"corr={d['p7_p9_correlation_coefficient']:.3f}")

    print("\n=== D2 feature drift (each model vs its own synth) ===")
    if d2:
        for model_tag in ["p7", "p9_dlow3"]:
            print(f"  {model_tag} (reference = {d2[model_tag]['reference_set']}):")
            for s in ["stage0", "stage1", "stage2", "stage3"]:
                r = d2[model_tag]["stages"][s]
                print(f"    {s}: max|z|={r['max_abs_z']:6.2f}  p95|z|={r['p95_abs_z']:6.2f}  "
                      f"mean|z|={r['mean_abs_z']:6.2f}  frac(|z|>3)={r['frac_channels_abs_z_gt_3']:.4f}")

    print("\n=== D3 primary: per-video coverage (dlow=6 measured vs P7/dlow3 canonical) ===")
    if d3_primary:
        for k in ["p7_canonical", "p9_dlow3_canonical", "p9_dlow6_measured"]:
            cov = d3_primary["per_video_coverage_pct"][k]
            print(f"  {k:25s}: S1={cov['1']:.2f}  S2={cov['2']:.2f}  S3={cov['3']:.2f}  S4={cov['4']:.2f}")
        print(f"  outcome (per-video pm20%): {d3_primary['decision_rule_per_video_threshold_pm20pct']['outcome']}")
        print(f"  outcome (avg pm20%): {d3_primary['decision_rule_on_average_threshold_pm20pct']['outcome']}")

    if d3_secondary:
        print("\n=== D3 secondary: dlow=3 epoch trajectory on sample_1 ===")
        for k, v in sorted(d3_secondary["results"].items()):
            print(f"  {k}: cov={v['mean_coverage_pct']:.4f}%")


if __name__ == "__main__":
    main()
