"""Phase 10 mechanism investigation — distinguish among 4 candidate mechanisms
(A) argmax artifact, (B) confident-wrong, (C) mass-shifted, (D) smooth-degradation
that could explain why Phase 9's small encoder drift becomes a catastrophic
real-world coverage collapse.

Inputs (verified):
  - Phase 7 ckpt:     results/segformer_b5_rgb/full_20260430_2032/best_model.pth
  - Phase 9 dlow=3:   results/segformer_b5_rgb_phase9/full_dlow3_20260515_1330/best_model.pth
  - Real videos:      data/dlo_real_sample_{1,2,3,4}.mp4
  - 80 D1 frames:     results/phase10_investigation/D1/sample_indices.json (20 per video)
  - D1 argmax masks:  results/phase10_investigation/D1/<vid>/frame_<idx>_p{7,9}.png (uint8 x255)

Outputs (under results/phase10_mechanism/):
  M1/<vid>/frame_<idx>_p7_prob.npy             float32 (480,640)
  M1/<vid>/frame_<idx>_p9_prob.npy             float32 (480,640)
  M1/<vid>/frame_<idx>_p7_heatmap.png          uint8 jet
  M1/<vid>/frame_<idx>_p9_heatmap.png          uint8 jet
  M1/<vid>/p9_softmax_at_<set>.json            mean/median/std/p5/p95 + 100-bin hist
  M1/<vid>/p7_softmax_at_<set>.json            symmetric reference
  M1/p9_prob_at_p7_only_pixels.json            cross-video roll-up (the key table)

  M2/threshold_sweep.json                      {model: {sample_i: {tau: cov_pct}}}
  M2/threshold_sweep.png                       per-video lineplot P7+P9 vs tau

  M3/mass_shift_per_video.json                 per-frame and per-video aggregates

  M4/soft_metric_comparison.json               soft vs binary, plus AUC vs P7-pseudo-GT

  REPORT.md                                    final summary

Compute pattern:
  Pass A: for each model in {P7, P9}, walk every frame of all 4 videos once.
      For each frame, capture (1) p_DLO softmax (480,640) and (2) the 0.5 argmax mask.
      Threshold-sweep softmax in numpy at {0.05,...,0.5}. Accumulate M2/M3/M4-soft.
  Pass B: on the 80 D1 frames, save the softmax maps, run the partition into
      4 disagreement sets using the existing D1 argmax PNGs, accumulate M1
      histograms, and accumulate M4-AUC via P7-as-pseudo-GT.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE = os.path.join(PROJECT_ROOT, "data", "hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_rgb_only_sota_gifs import load_model  # noqa: E402
from infer_video_rgb_only import preprocess  # noqa: E402
from train_rgb_only_sota import RGB_MEAN, RGB_STD  # noqa: E402

VIDEOS = {
    "sample_1": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_1.mp4"),
    "sample_2": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_2.mp4"),
    "sample_3": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_3.mp4"),
    "sample_4": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_4.mp4"),
}
CKPT_P7 = os.path.join(PROJECT_ROOT, "results", "segformer_b5_rgb",
                      "full_20260430_2032", "best_model.pth")
CKPT_P9 = os.path.join(PROJECT_ROOT, "results", "segformer_b5_rgb_phase9",
                      "full_dlow3_20260515_1330", "best_model.pth")
D1_DIR = os.path.join(PROJECT_ROOT, "results", "phase10_investigation", "D1")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "phase10_mechanism")

# Canonical Phase 7 / Phase 9 numbers used as the sanity check.
CANON_P7 = {"sample_1": 6.77, "sample_2": 12.37, "sample_3": 12.53, "sample_4": 12.28}
CANON_P9 = {"sample_1": 2.88, "sample_2": 2.52, "sample_3": 0.37, "sample_4": 0.97}

TAU_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
                    dtype=np.float64)
N_BINS = 100  # for M1 histograms
IMAGE_H, IMAGE_W = 480, 640
N_PIX = IMAGE_H * IMAGE_W


@torch.no_grad()
def softmax_dlo(model, bgr_uint8, device):
    """Mirror gen_rgb_only_sota_gifs.predict but keep softmax over class 1.

    Returns: float32 (H, W) array of P(DLO).
    """
    rgb_rgb = bgr_uint8[:, :, ::-1].copy()
    rgb = torch.from_numpy(rgb_rgb.transpose(2, 0, 1)).unsqueeze(0).to(
        device, dtype=torch.float32) / 255.0
    rgb = (rgb - RGB_MEAN.to(device)) / RGB_STD.to(device)
    logits = model(rgb)  # (1, 2, H, W)
    prob = F.softmax(logits, dim=1)[0, 1].detach().cpu().numpy().astype(np.float32)
    return prob


def load_d1_indices():
    with open(os.path.join(D1_DIR, "sample_indices.json")) as f:
        return json.load(f)


def load_d1_argmax(vid, idx):
    """Return (m7_bool, m9_bool) for the existing D1 frame_<idx>_p{7,9}.png masks."""
    p7 = cv2.imread(os.path.join(D1_DIR, vid, f"frame_{idx:05d}_p7.png"),
                    cv2.IMREAD_UNCHANGED)
    p9 = cv2.imread(os.path.join(D1_DIR, vid, f"frame_{idx:05d}_p9.png"),
                    cv2.IMREAD_UNCHANGED)
    assert p7.shape == (IMAGE_H, IMAGE_W) and p9.shape == (IMAGE_H, IMAGE_W), \
        f"D1 mask shape mismatch at {vid}/{idx}"
    return (p7 == 255), (p9 == 255)


def jet_heatmap(prob_2d):
    """Map float prob in [0, 1] to a BGR jet image."""
    u8 = (np.clip(prob_2d, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def hist_stats(values, n_bins=N_BINS):
    """Return mean/median/std/p5/p95 + 100-bin hist over [0,1] for a flat array.

    Coerces values to float32; safely handles empty arrays.
    """
    v = np.asarray(values, dtype=np.float32).ravel()
    if v.size == 0:
        return {
            "n_pixels": 0,
            "mean": None, "median": None, "std": None,
            "p5": None, "p95": None,
            "hist_counts": [0] * n_bins,
            "hist_edges": np.linspace(0.0, 1.0, n_bins + 1).tolist(),
        }
    hist, edges = np.histogram(v, bins=n_bins, range=(0.0, 1.0))
    return {
        "n_pixels": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std()),
        "p5": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "hist_counts": hist.astype(int).tolist(),
        "hist_edges": edges.tolist(),
    }


def auc_from_scores(scores, labels):
    """Compute ROC-AUC of `scores` against binary `labels` using the rank trick.

    Implementation matches sklearn.metrics.roc_auc_score for the binary case.
    Falls back to None on degenerate input.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int32).ravel()
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    # rankdata "average" via argsort (manual to avoid scipy)
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    # Average ties
    n = scores.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_ranks_pos = float(ranks[labels == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def process_full_video_one_model(model, vid_path, device, tag, n_frames_cap=None):
    """Single pass over every frame: returns
        cov_per_tau              dict tau -> list of per-frame coverages
        argmax_cov_list          list (P>=0.5) coverages (== threshold 0.5)
        mass_total_list          list of sum(p)
        mean_p_list              list of mean(p)
        soft_area_list           list of sum(p) == total expected DLO area
        # For pass A we don't need the masks per-frame to compute mass-shift,
        # those are computed in process_with_p7_overlay when both models are present.
    """
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {vid_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames_cap is not None:
        n_frames = min(n_frames, n_frames_cap)

    cov_per_tau = {float(t): [] for t in TAU_GRID}
    mass_total_list = []
    mean_p_list = []

    fi = 0
    while fi < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        pre = preprocess(frame)
        prob = softmax_dlo(model, pre, device)  # (H, W) float32
        # threshold sweep
        for t in TAU_GRID:
            cov_per_tau[float(t)].append(100.0 * float((prob > t).sum()) / N_PIX)
        mass_total_list.append(float(prob.sum()))
        mean_p_list.append(float(prob.mean()))
        fi += 1
    cap.release()
    print(f"  [{tag}] processed {fi} frames")
    return {
        "n_frames_used": fi,
        "cov_per_tau": cov_per_tau,
        "mass_total_list": mass_total_list,
        "mean_p_list": mean_p_list,
    }


def process_full_video_both(p7_model, p9_model, vid_path, device, tag):
    """Single pass to capture: per-frame
        - p9_total_mass, p9_mass_outside_P7=BG-aware, p9_mass_inside_P7=DLO-aware
        - p9_softmax for threshold sweep
        - p7_softmax for threshold sweep
        - p7 argmax pixel count (mass_inside/outside is taken vs P7 argmax)
    Returns dict with the same per-tau coverage lists for BOTH models, plus
    the mass-shift series. Doing both models in one cap-loop saves IO."""
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {vid_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cov_p7 = {float(t): [] for t in TAU_GRID}
    cov_p9 = {float(t): [] for t in TAU_GRID}

    p7_mass_list = []
    p9_mass_list = []
    p9_mean_list = []
    p7_mean_list = []
    p9_mass_outside_p7 = []
    p9_mass_inside_p7 = []
    p7_dlo_area_list = []   # binary
    p9_dlo_area_list = []   # binary

    fi = 0
    while fi < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        pre = preprocess(frame)
        p7_prob = softmax_dlo(p7_model, pre, device)
        p9_prob = softmax_dlo(p9_model, pre, device)
        p7_mask = (p7_prob >= 0.5)
        p9_mask = (p9_prob >= 0.5)
        for t in TAU_GRID:
            cov_p7[float(t)].append(100.0 * float((p7_prob > t).sum()) / N_PIX)
            cov_p9[float(t)].append(100.0 * float((p9_prob > t).sum()) / N_PIX)
        p7_total = float(p7_prob.sum())
        p9_total = float(p9_prob.sum())
        p7_mass_list.append(p7_total)
        p9_mass_list.append(p9_total)
        p7_mean_list.append(float(p7_prob.mean()))
        p9_mean_list.append(float(p9_prob.mean()))
        if p7_mask.any():
            p9_mass_inside_p7.append(float(p9_prob[p7_mask].sum()))
            p9_mass_outside_p7.append(float(p9_prob[~p7_mask].sum()))
        else:
            p9_mass_inside_p7.append(0.0)
            p9_mass_outside_p7.append(p9_total)
        p7_dlo_area_list.append(int(p7_mask.sum()))
        p9_dlo_area_list.append(int(p9_mask.sum()))
        fi += 1
        if fi % 200 == 0:
            print(f"  [{tag}] {fi}/{n_frames}")
    cap.release()
    print(f"  [{tag}] processed {fi} frames")
    return {
        "n_frames_used": fi,
        "cov_p7": cov_p7, "cov_p9": cov_p9,
        "p7_mass_list": p7_mass_list, "p9_mass_list": p9_mass_list,
        "p7_mean_list": p7_mean_list, "p9_mean_list": p9_mean_list,
        "p9_mass_outside_p7": p9_mass_outside_p7,
        "p9_mass_inside_p7": p9_mass_inside_p7,
        "p7_dlo_area_list": p7_dlo_area_list,
        "p9_dlo_area_list": p9_dlo_area_list,
    }


def process_d1_frames_one_model(model, vid_path, idxs, device, tag):
    """Open a cap, walk to the listed indices, return dict idx -> softmax(p)."""
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {vid_path}")
    out = {}
    target = set(int(i) for i in idxs)
    fi = 0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while fi < n_frames and len(out) < len(target):
        ok, frame = cap.read()
        if not ok:
            break
        if fi in target:
            pre = preprocess(frame)
            out[fi] = softmax_dlo(model, pre, device)
        fi += 1
    cap.release()
    print(f"  [{tag}] D1 frames captured: {len(out)}/{len(target)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frame-cap", type=int, default=None,
                    help="if set, process at most this many frames per video (for testing)")
    args = ap.parse_args()
    device = torch.device(args.device)

    os.makedirs(OUT_DIR, exist_ok=True)
    for sub in ("M1", "M2", "M3", "M4"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    for v in VIDEOS:
        os.makedirs(os.path.join(OUT_DIR, "M1", v), exist_ok=True)

    print(f"Loading P7 from {CKPT_P7}")
    p7_model = load_model(CKPT_P7, device)
    print(f"Loading P9-dlow3 from {CKPT_P9}")
    p9_model = load_model(CKPT_P9, device)

    sample_indices = load_d1_indices()

    # ───── PASS A: full-video sweep (M2 + M3 + M4 soft) ─────
    threshold_sweep = {"P7": {}, "P9_dlow3": {}}
    mass_shift = {}
    soft_metric = {"P7": {}, "P9_dlow3": {}}
    sanity_report = {}

    for vid, vid_path in VIDEOS.items():
        print(f"\n=== Pass A: {vid} ({vid_path}) ===")
        agg = process_full_video_both(p7_model, p9_model, vid_path, device, vid)

        # M2: per-tau mean coverage
        threshold_sweep["P7"][vid] = {
            f"{t:.2f}": float(np.mean(agg["cov_p7"][float(t)])) for t in TAU_GRID
        }
        threshold_sweep["P9_dlow3"][vid] = {
            f"{t:.2f}": float(np.mean(agg["cov_p9"][float(t)])) for t in TAU_GRID
        }

        # Sanity check: tau=0.5 == argmax-coverage
        sanity_report[vid] = {
            "n_frames": agg["n_frames_used"],
            "p7_mean_coverage_pct_tau050": threshold_sweep["P7"][vid]["0.50"],
            "p9_mean_coverage_pct_tau050": threshold_sweep["P9_dlow3"][vid]["0.50"],
            "p7_canonical": CANON_P7[vid],
            "p9_canonical": CANON_P9[vid],
            "p7_delta_vs_canon": threshold_sweep["P7"][vid]["0.50"] - CANON_P7[vid],
            "p9_delta_vs_canon": threshold_sweep["P9_dlow3"][vid]["0.50"] - CANON_P9[vid],
        }
        print(f"  -> {vid}: P7 tau=0.5 cov = {sanity_report[vid]['p7_mean_coverage_pct_tau050']:.4f}% "
              f"(canon {CANON_P7[vid]:.2f}, delta {sanity_report[vid]['p7_delta_vs_canon']:+.3f})")
        print(f"     P9 tau=0.5 cov = {sanity_report[vid]['p9_mean_coverage_pct_tau050']:.4f}% "
              f"(canon {CANON_P9[vid]:.2f}, delta {sanity_report[vid]['p9_delta_vs_canon']:+.3f})")

        # M3: mass-shift aggregates (per-frame -> per-video sums and means)
        p9_total = np.asarray(agg["p9_mass_list"])
        p9_in = np.asarray(agg["p9_mass_inside_p7"])
        p9_out = np.asarray(agg["p9_mass_outside_p7"])
        p7_total = np.asarray(agg["p7_mass_list"])
        # Aggregate ratio = sum/sum to be size-weighted (not mean of ratios).
        sum_p9_total = float(p9_total.sum())
        ratio_mass_outside = float(p9_out.sum()) / sum_p9_total if sum_p9_total > 0 else None
        ratio_mass_inside = float(p9_in.sum()) / sum_p9_total if sum_p9_total > 0 else None
        # Per-frame ratios for stats (skip frames with zero p9 mass)
        per_frame_out_ratio = []
        for tot, out_v in zip(p9_total, p9_out):
            if tot > 1e-9:
                per_frame_out_ratio.append(float(out_v) / float(tot))
        mass_shift[vid] = {
            "n_frames": agg["n_frames_used"],
            "p7_total_prob_mass_mean": float(p7_total.mean()),
            "p9_total_prob_mass_mean": float(p9_total.mean()),
            "p9_total_prob_mass_per_frame_sum": sum_p9_total,
            "p9_mass_inside_p7_dlo_total": float(p9_in.sum()),
            "p9_mass_outside_p7_dlo_total": float(p9_out.sum()),
            "ratio_mass_outside_p7_aggregate": ratio_mass_outside,
            "ratio_mass_inside_p7_aggregate": ratio_mass_inside,
            "ratio_mass_outside_per_frame_mean": float(np.mean(per_frame_out_ratio)) if per_frame_out_ratio else None,
            "ratio_mass_outside_per_frame_median": float(np.median(per_frame_out_ratio)) if per_frame_out_ratio else None,
            "p7_binary_area_mean_px": float(np.mean(agg["p7_dlo_area_list"])),
            "p9_binary_area_mean_px": float(np.mean(agg["p9_dlo_area_list"])),
            "p7_binary_cov_mean_pct": 100.0 * float(np.mean(agg["p7_dlo_area_list"])) / N_PIX,
            "p9_binary_cov_mean_pct": 100.0 * float(np.mean(agg["p9_dlo_area_list"])) / N_PIX,
        }

        # M4: soft-metric (per video, both models)
        soft_metric["P7"][vid] = {
            "mean_total_expected_DLO_area_px": float(np.mean(agg["p7_mass_list"])),
            "mean_total_expected_DLO_area_pct_of_image": 100.0 * float(np.mean(agg["p7_mass_list"])) / N_PIX,
            "mean_per_frame_mean_pDLO": float(np.mean(agg["p7_mean_list"])),
            "binary_area_mean_pct": 100.0 * float(np.mean(agg["p7_dlo_area_list"])) / N_PIX,
        }
        soft_metric["P9_dlow3"][vid] = {
            "mean_total_expected_DLO_area_px": float(np.mean(agg["p9_mass_list"])),
            "mean_total_expected_DLO_area_pct_of_image": 100.0 * float(np.mean(agg["p9_mass_list"])) / N_PIX,
            "mean_per_frame_mean_pDLO": float(np.mean(agg["p9_mean_list"])),
            "binary_area_mean_pct": 100.0 * float(np.mean(agg["p9_dlo_area_list"])) / N_PIX,
        }

    # Save M2 + M3 + M4 partial outputs.
    with open(os.path.join(OUT_DIR, "M2", "threshold_sweep.json"), "w") as f:
        json.dump({"taus": [float(t) for t in TAU_GRID],
                   "coverage_pct_by_video_and_tau": threshold_sweep,
                   "sanity_vs_canonical_at_tau050": sanity_report}, f, indent=2)
    with open(os.path.join(OUT_DIR, "M3", "mass_shift_per_video.json"), "w") as f:
        json.dump(mass_shift, f, indent=2)
    with open(os.path.join(OUT_DIR, "M4", "soft_metric_comparison.json"), "w") as f:
        # AUC will be added in pass B.
        json.dump({"per_video": soft_metric, "auc_p9_pseudo_gt_p7": None}, f, indent=2)
    print("\nPass A complete. M2/M3/M4-soft saved.")

    # Plot M2 (per-video line plot)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for ax, vid in zip(axes, VIDEOS):
            ys_p7 = [threshold_sweep["P7"][vid][f"{t:.2f}"] for t in TAU_GRID]
            ys_p9 = [threshold_sweep["P9_dlow3"][vid][f"{t:.2f}"] for t in TAU_GRID]
            ax.plot(TAU_GRID, ys_p7, "o-", label="P7", color="tab:blue")
            ax.plot(TAU_GRID, ys_p9, "o-", label="P9 dlow3", color="tab:red")
            ax.axhline(CANON_P7[vid], linestyle="--", color="tab:blue", alpha=0.4,
                       label=f"P7 canon ({CANON_P7[vid]:.2f}%)")
            ax.axhline(CANON_P9[vid], linestyle="--", color="tab:red", alpha=0.4,
                       label=f"P9 canon ({CANON_P9[vid]:.2f}%)")
            ax.set_xlabel("threshold tau on P(DLO)")
            ax.set_ylabel("mean coverage (%)")
            ax.set_title(vid)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "M2", "threshold_sweep.png"), dpi=120)
        plt.close()
        print(f"  threshold_sweep.png written.")
    except Exception as e:
        print(f"  WARN: could not produce M2 plot: {e}")

    # ───── PASS B: D1 80 frames — M1 (probability maps + histograms) + M4 AUC ─────
    print(f"\n=== Pass B: M1 + M4-AUC over 80 D1 frames ===")
    m1_summary = {"per_video": {}, "rollup": {}}
    auc_per_video = {}
    auc_global_scores = []
    auc_global_labels = []

    for vid in VIDEOS:
        vid_path = VIDEOS[vid]
        idxs = sample_indices[vid]
        print(f"\n--- {vid}: {len(idxs)} frames ---")

        # Capture softmax maps for both models on these specific frames.
        p7_maps = process_d1_frames_one_model(p7_model, vid_path, idxs, device,
                                              f"{vid}/P7")
        p9_maps = process_d1_frames_one_model(p9_model, vid_path, idxs, device,
                                              f"{vid}/P9")

        # Accumulators
        # For each disagreement-set, gather Phase 9 prob and Phase 7 prob lists.
        sets = ("both_dlo", "p7_only", "p9_only", "both_bg")
        p7_vals = {s: [] for s in sets}
        p9_vals = {s: [] for s in sets}
        pixel_count = {s: 0 for s in sets}

        # AUC accumulators per video
        auc_scores = []
        auc_labels = []

        for idx in idxs:
            p7_prob = p7_maps[int(idx)]
            p9_prob = p9_maps[int(idx)]
            assert p7_prob.shape == (IMAGE_H, IMAGE_W) and p9_prob.shape == (IMAGE_H, IMAGE_W)

            # Save raw float32 probability maps.
            np.save(os.path.join(OUT_DIR, "M1", vid,
                                 f"frame_{int(idx):05d}_p7_prob.npy"), p7_prob)
            np.save(os.path.join(OUT_DIR, "M1", vid,
                                 f"frame_{int(idx):05d}_p9_prob.npy"), p9_prob)

            # Heatmaps
            cv2.imwrite(os.path.join(OUT_DIR, "M1", vid,
                                     f"frame_{int(idx):05d}_p7_heatmap.png"),
                        jet_heatmap(p7_prob))
            cv2.imwrite(os.path.join(OUT_DIR, "M1", vid,
                                     f"frame_{int(idx):05d}_p9_heatmap.png"),
                        jet_heatmap(p9_prob))

            # Partition using the existing D1 argmax masks on disk (the same ones the
            # canonical Phase 10 D1 report uses; this guarantees set-membership is
            # identical to Phase 10).
            m7, m9 = load_d1_argmax(vid, int(idx))
            both_dlo = m7 & m9
            p7_only = m7 & (~m9)
            p9_only = (~m7) & m9
            both_bg = (~m7) & (~m9)

            for s, mask in (("both_dlo", both_dlo), ("p7_only", p7_only),
                            ("p9_only", p9_only), ("both_bg", both_bg)):
                if mask.any():
                    p7_vals[s].append(p7_prob[mask])
                    p9_vals[s].append(p9_prob[mask])
                pixel_count[s] += int(mask.sum())

            # AUC: pseudo-GT = P7 argmax mask (m7). Score = P9's softmax.
            auc_scores.append(p9_prob.ravel())
            auc_labels.append(m7.astype(np.int32).ravel())
            auc_global_scores.append(p9_prob.ravel())
            auc_global_labels.append(m7.astype(np.int32).ravel())

        # Concat per-set arrays
        p7_concat = {s: np.concatenate(p7_vals[s]) if p7_vals[s] else np.zeros(0, dtype=np.float32)
                     for s in sets}
        p9_concat = {s: np.concatenate(p9_vals[s]) if p9_vals[s] else np.zeros(0, dtype=np.float32)
                     for s in sets}

        # Save per-set JSONs
        for s in sets:
            with open(os.path.join(OUT_DIR, "M1", vid, f"p9_softmax_at_{s}.json"), "w") as f:
                json.dump(hist_stats(p9_concat[s]), f, indent=2)
            with open(os.path.join(OUT_DIR, "M1", vid, f"p7_softmax_at_{s}.json"), "w") as f:
                json.dump(hist_stats(p7_concat[s]), f, indent=2)

        # Per-video summary entry
        m1_summary["per_video"][vid] = {
            "pixel_counts_by_set": pixel_count,
            "p9_at_p7_only": hist_stats(p9_concat["p7_only"]),
            "p9_at_both_dlo": hist_stats(p9_concat["both_dlo"]),
            "p9_at_p9_only": hist_stats(p9_concat["p9_only"]),
            "p9_at_both_bg": hist_stats(p9_concat["both_bg"]),
            "p7_at_p7_only": hist_stats(p7_concat["p7_only"]),
            "p7_at_both_dlo": hist_stats(p7_concat["both_dlo"]),
            "p7_at_p9_only": hist_stats(p7_concat["p9_only"]),
            "p7_at_both_bg": hist_stats(p7_concat["both_bg"]),
        }

        # Per-video AUC
        s_v = np.concatenate(auc_scores)
        l_v = np.concatenate(auc_labels)
        # Subsample to keep AUC fast: 1M pixels are fine, but 20 frames * 307200 = 6.14M.
        # The rank-trick implementation is O(n log n) so 6M is acceptable.
        try:
            auc_v = auc_from_scores(s_v, l_v)
        except Exception as e:
            print(f"  WARN: AUC computation failed for {vid}: {e}")
            auc_v = None
        auc_per_video[vid] = {"auc_P9_softmax_vs_P7_pseudo_gt": auc_v,
                              "n_pixels": int(s_v.size),
                              "n_positive_pixels": int((l_v == 1).sum())}

    # Roll-up: P9 prob at p7_only pixels, across all videos
    rollup = {}
    for s in ("both_dlo", "p7_only", "p9_only", "both_bg"):
        all_p9 = []
        all_p7 = []
        for vid in VIDEOS:
            v = m1_summary["per_video"][vid][f"p9_at_{s}"]
            # the per-video histogram has n_pixels; we'll just aggregate the per-frame raw
            # via a re-read of npy files is heavy — instead aggregate counts across videos.
            # Hist_stats stores hist counts, not raw. We'll combine histograms here.
        # Re-aggregate by summing hist counts across videos (and weighting mean/median is
        # not exact from histograms; we use the per-set raw arrays cached earlier instead).
        pass
    # Easier: re-collect raw per-set arrays for the roll-up by reading npys back is too slow.
    # Instead, store the per-set values during the loop. We already have access to them per
    # video, but they were freed. Redo cheaply: re-scan saved npys.
    # The cost of reading 80 npys * 2 == 160 small files is trivial; do it.
    rollup_arrays_p9 = {s: [] for s in ("both_dlo", "p7_only", "p9_only", "both_bg")}
    rollup_arrays_p7 = {s: [] for s in ("both_dlo", "p7_only", "p9_only", "both_bg")}
    for vid in VIDEOS:
        for idx in sample_indices[vid]:
            p7_prob = np.load(os.path.join(OUT_DIR, "M1", vid,
                              f"frame_{int(idx):05d}_p7_prob.npy"))
            p9_prob = np.load(os.path.join(OUT_DIR, "M1", vid,
                              f"frame_{int(idx):05d}_p9_prob.npy"))
            m7, m9 = load_d1_argmax(vid, int(idx))
            both_dlo = m7 & m9
            p7_only = m7 & (~m9)
            p9_only = (~m7) & m9
            both_bg = (~m7) & (~m9)
            for s, mask in (("both_dlo", both_dlo), ("p7_only", p7_only),
                            ("p9_only", p9_only), ("both_bg", both_bg)):
                if mask.any():
                    rollup_arrays_p9[s].append(p9_prob[mask])
                    rollup_arrays_p7[s].append(p7_prob[mask])
    rollup_p9 = {}
    rollup_p7 = {}
    for s in ("both_dlo", "p7_only", "p9_only", "both_bg"):
        if rollup_arrays_p9[s]:
            rollup_p9[s] = hist_stats(np.concatenate(rollup_arrays_p9[s]))
            rollup_p7[s] = hist_stats(np.concatenate(rollup_arrays_p7[s]))
        else:
            rollup_p9[s] = hist_stats(np.zeros(0, dtype=np.float32))
            rollup_p7[s] = hist_stats(np.zeros(0, dtype=np.float32))
    m1_summary["rollup"] = {"p9_softmax_at": rollup_p9, "p7_softmax_at": rollup_p7}

    # Save the "smoking-gun number"
    with open(os.path.join(OUT_DIR, "M1", "p9_prob_at_p7_only_pixels.json"), "w") as f:
        json.dump({"per_video": {v: m1_summary["per_video"][v]["p9_at_p7_only"]
                                 for v in VIDEOS},
                   "rollup_all_videos": rollup_p9["p7_only"],
                   "explanation":
                   "P9's softmax(DLO) distribution at pixels where P7=DLO and P9=BG. "
                   "Near 0 -> (B) confident-wrong. Near 0.4 -> (A) argmax artifact. "
                   "Bimodal -> mixture. Compare against p7_softmax counterpart and "
                   "against p9_at_both_dlo (P9's prob at agreed DLO pixels) for context."}, f,
                  indent=2)

    # Save the rest of M1 summary.
    with open(os.path.join(OUT_DIR, "M1", "m1_summary.json"), "w") as f:
        json.dump(m1_summary, f, indent=2)

    # Global AUC
    auc_global = None
    try:
        s_all = np.concatenate(auc_global_scores)
        l_all = np.concatenate(auc_global_labels)
        # Optional subsample to keep fast; with 80 frames = ~24M pixels — fine.
        auc_global = auc_from_scores(s_all, l_all)
    except Exception as e:
        print(f"  WARN: global AUC failed: {e}")

    # Update M4 with AUC numbers.
    soft_path = os.path.join(OUT_DIR, "M4", "soft_metric_comparison.json")
    with open(soft_path) as f:
        m4 = json.load(f)
    m4["auc_p9_pseudo_gt_p7"] = {
        "per_video": auc_per_video,
        "global_80_frames": {
            "auc_P9_softmax_vs_P7_pseudo_gt": auc_global,
        },
        "explanation":
            "Pseudo-GT is the Phase 7 argmax mask on the same 80 D1 frames "
            "(P7=DLO=1 vs P7=BG=0). Score is Phase 9's softmax(DLO). "
            "AUC>=0.85 -> P9 has DLO ranking right, just wrong threshold -> (A) or (D). "
            "AUC<=0.6 -> P9 broken at the ranking level -> (B)."
    }
    with open(soft_path, "w") as f:
        json.dump(m4, f, indent=2)

    print("\nDone. All artifacts in", OUT_DIR)


if __name__ == "__main__":
    main()
