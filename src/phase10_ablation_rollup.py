"""Roll up the 4 per-variant gate.json files into one summary.

Adds Phase 7 (=1.00 / 1.00 by construction) and Phase 9 dlow=3 references
from the M1 / M4 artefacts so the user can compare apples-to-apples in
the REPORT.md gate table.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABL_ROOT = os.path.join(PROJECT_ROOT, "results", "phase10_ablation")
M4_PATH = os.path.join(PROJECT_ROOT, "results", "phase10_mechanism", "M4",
                       "soft_metric_comparison.json")
M1_PATH = os.path.join(PROJECT_ROOT, "results", "phase10_mechanism", "M1",
                       "p9_prob_at_p7_only_pixels.json")

VIDEOS = ["sample_1", "sample_2", "sample_3", "sample_4"]
VARIANTS = ["wall", "camera", "harness_hsv", "objects"]


def main():
    with open(M4_PATH) as f:
        m4 = json.load(f)
    with open(M1_PATH) as f:
        m1 = json.load(f)

    # Phase 7: by construction
    p7_row = {
        "label": "Phase 7 (baseline)",
        "per_video": {v: {"gate_a": 1.000, "gate_b": 1.000} for v in VIDEOS},
        "all_pass_a": True,
        "all_pass_b": True,
    }

    # Phase 9 dlow=3 reference, per PI's spec:
    #   gate_a from M4 (P9 mean P_DLO / P7 mean P_DLO)
    #   gate_b: prefer end-to-end measurement at results/phase10_ablation/
    #   _sanity_p9/gate.json (full P7-positive pixel set, includes P7∩P9);
    #   fall back to M1's (1 - hist_counts[0]/n_pixels) at P7-only pixels
    #   if the sanity file isn't present.
    p9_sanity_p = os.path.join(ABL_ROOT, "_sanity_p9", "gate.json")
    p9_sanity = None
    if os.path.isfile(p9_sanity_p):
        with open(p9_sanity_p) as f:
            p9_sanity = json.load(f)
    p9_row_per_video = {}
    for v in VIDEOS:
        a = (m4["per_video"]["P9_dlow3"][v]["mean_per_frame_mean_pDLO"]
             / m4["per_video"]["P7"][v]["mean_per_frame_mean_pDLO"])
        if p9_sanity is not None:
            b = p9_sanity["per_video"][v]["gate_b_frac_P7pos_softmax_gt_0.01"]
            b_source = "measured (end-to-end on Phase 9 dlow=3)"
        else:
            n_p7only = m1["per_video"][v]["n_pixels"]
            hist_lt_001 = m1["per_video"][v]["hist_counts"][0]
            b = 1.0 - (hist_lt_001 / max(n_p7only, 1))
            b_source = "M1 lower-bound (at P7-only pixels)"
        p9_row_per_video[v] = {"gate_a": a, "gate_b": b}
    p9_row = {
        "label": "Phase 9 dlow=3 (reference)",
        "per_video": p9_row_per_video,
        "gate_b_source": b_source,
    }

    rows = {"P7": p7_row, "P9_dlow3": p9_row}
    for v in VARIANTS:
        gate_p = os.path.join(ABL_ROOT, v, "gate.json")
        if not os.path.isfile(gate_p):
            print(f"  WARN: {gate_p} missing — skipping variant '{v}'")
            continue
        with open(gate_p) as f:
            g = json.load(f)
        row = {
            "label": f"Phase 10 ablation: {v}",
            "per_video": {
                vid: {"gate_a": g["per_video"][vid]["gate_a_ratio_mean_pDLO"],
                      "gate_b": g["per_video"][vid]["gate_b_frac_P7pos_softmax_gt_0.01"]}
                for vid in VIDEOS
            },
            "overall": g["overall"],
        }
        rows[v] = row

    with open(os.path.join(ABL_ROOT, "gate_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {os.path.join(ABL_ROOT, 'gate_summary.json')}")

    # Pretty-print to stdout
    print("\n=== Gate summary ===")
    headers = ["row"] + [f"a:{v}" for v in VIDEOS] + [f"b:{v}" for v in VIDEOS]
    print(" | ".join(h.rjust(10) for h in headers))
    for key in ["P7", "P9_dlow3"] + VARIANTS:
        if key not in rows:
            continue
        cells = [key]
        for v in VIDEOS:
            a = rows[key]["per_video"].get(v, {}).get("gate_a", float("nan"))
            cells.append(f"{a:.3f}")
        for v in VIDEOS:
            b = (rows[key]["per_video"].get(v, {}).get("gate_b")
                 or float("nan"))
            cells.append(f"{b:.3f}")
        print(" | ".join(c.rjust(10) for c in cells))


if __name__ == "__main__":
    main()
