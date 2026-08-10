"""Phase 10 D1 — Per-pixel disagreement masks (P7 vs P9-dlow3) on real videos.

Picks 20 evenly-spaced frame indices from each of the 4 real videos, runs both
checkpoints, and saves:
  D1/<vid>/frame_<idx>_rgb.png           preprocessed 640x480 BGR input
  D1/<vid>/frame_<idx>_p7.png            P7 binary mask (uint8, x255)
  D1/<vid>/frame_<idx>_p9.png            P9-dlow3 binary mask (uint8, x255)
  D1/<vid>/frame_<idx>_disagree.png      RGBA overlay: GREEN=P7-only,
                                          RED=P9-only, GRAY=both, transparent elsewhere
  D1/sample_indices.json
  D1/disagreement_stats.json
  D1/<vid>/rgb_at_disagreement.json      per-channel BGR histograms at P7-only,
                                          P9-only, both pixels.

In addition, this script computes the per-video mean coverage over the FULL
video (not just the 20 sampled frames) for both P7 and P9-dlow3 — this is the
sanity check against the canonical numbers.
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE = os.path.join(PROJECT_ROOT, "data", "hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_rgb_only_sota_gifs import load_model, predict  # noqa: E402
from infer_video_rgb_only import preprocess  # noqa: E402


N_SAMPLES_PER_VIDEO = 20
VIDEOS = {
    "sample_1": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_1.mp4"),
    "sample_2": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_2.mp4"),
    "sample_3": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_3.mp4"),
    "sample_4": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_4.mp4"),
}
CKPT_P7 = os.path.join(PROJECT_ROOT, "results", "segformer_b5_rgb", "full_20260430_2032", "best_model.pth")
CKPT_P9 = os.path.join(PROJECT_ROOT, "results", "segformer_b5_rgb_phase9", "full_dlow3_20260515_1330", "best_model.pth")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "phase10_investigation", "D1")


def disagree_overlay(rgb, m7, m9):
    """Return a 3-channel BGR overlay on top of rgb (faded), with:
        GREEN where P7=1 and P9=0
        RED   where P9=1 and P7=0
        GRAY  where both
    Pixels with no DLO from either get the original RGB.
    """
    out = rgb.copy()
    # Fade non-DLO background just a little so the overlay reads well
    p7_only = (m7 == 1) & (m9 == 0)
    p9_only = (m9 == 1) & (m7 == 0)
    both = (m7 == 1) & (m9 == 1)
    out[p7_only] = (0, 255, 0)  # GREEN (BGR)
    out[p9_only] = (0, 0, 255)  # RED (BGR)
    out[both] = (128, 128, 128)  # GRAY
    return out


def per_channel_hist(rgb_bgr, mask_bool):
    """Return (3, 256) uint64 histograms over BGR channels at mask-true pixels."""
    if mask_bool.sum() == 0:
        return np.zeros((3, 256), dtype=np.uint64)
    px = rgb_bgr[mask_bool]
    hist = np.zeros((3, 256), dtype=np.uint64)
    for c in range(3):
        h, _ = np.histogram(px[:, c], bins=np.arange(257))
        hist[c] = h.astype(np.uint64)
    return hist


def channel_mean_std(rgb_bgr, mask_bool):
    if mask_bool.sum() == 0:
        return [None, None, None], [None, None, None]
    px = rgb_bgr[mask_bool].astype(np.float64)
    return px.mean(axis=0).tolist(), px.std(axis=0).tolist()


def correlation(xs, ys):
    if len(xs) < 2:
        return None
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.std() == 0 or ys.std() == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0",
                    help="(CUDA_VISIBLE_DEVICES=1 should already be set so cuda:0 -> physical GPU 1)")
    args = ap.parse_args()
    device = torch.device(args.device)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading P7 from {CKPT_P7}")
    p7 = load_model(CKPT_P7, device)
    print(f"Loading P9-dlow3 from {CKPT_P9}")
    p9 = load_model(CKPT_P9, device)

    # First pass: full-video coverage for both models (sanity check vs canonical).
    sanity_coverage = {}
    sample_indices = {}
    per_frame_stats = {v: [] for v in VIDEOS}

    for vid_name, vid_path in VIDEOS.items():
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            raise RuntimeError(f"could not open {vid_path}")
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, n_frames - 1, N_SAMPLES_PER_VIDEO, dtype=int).tolist()
        sample_indices[vid_name] = idxs
        print(f"\n=== {vid_name} ({n_frames} frames, sampling {N_SAMPLES_PER_VIDEO}) ===")

        vid_out = os.path.join(OUT_DIR, vid_name)
        os.makedirs(vid_out, exist_ok=True)

        # Full-video coverage stats (for canonical sanity check) + saved 20-frame artefacts.
        cov_p7_full = []
        cov_p9_full = []

        # Histograms for D1 BGR distributions across all sampled frames.
        hist_p7only = np.zeros((3, 256), dtype=np.uint64)
        hist_p9only = np.zeros((3, 256), dtype=np.uint64)
        hist_both = np.zeros((3, 256), dtype=np.uint64)

        # We iterate over every frame for coverage, but only save artefacts for sampled idxs.
        for fi in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            pre = preprocess(frame)
            with torch.no_grad():
                m7 = predict(p7, pre, device)
                m9 = predict(p9, pre, device)
            n_total = m7.size
            cov_p7_full.append(100.0 * (m7 == 1).sum() / n_total)
            cov_p9_full.append(100.0 * (m9 == 1).sum() / n_total)

            if fi in idxs:
                # Save artefacts
                cv2.imwrite(os.path.join(vid_out, f"frame_{fi:05d}_rgb.png"), pre)
                cv2.imwrite(os.path.join(vid_out, f"frame_{fi:05d}_p7.png"),
                            (m7 * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(vid_out, f"frame_{fi:05d}_p9.png"),
                            (m9 * 255).astype(np.uint8))
                overlay = disagree_overlay(pre, m7, m9)
                cv2.imwrite(os.path.join(vid_out, f"frame_{fi:05d}_disagree.png"), overlay)

                # Per-frame stats
                p7_dlo = int((m7 == 1).sum())
                p9_dlo = int((m9 == 1).sum())
                p7only = int(((m7 == 1) & (m9 == 0)).sum())
                p9only = int(((m9 == 1) & (m7 == 0)).sum())
                both = int(((m7 == 1) & (m9 == 1)).sum())
                union = (m7 | m9).astype(bool).sum()
                iou_p7_p9 = float(both) / float(union) if union > 0 else float("nan")
                per_frame_stats[vid_name].append({
                    "frame_idx": int(fi),
                    "p7_dlo_px": p7_dlo,
                    "p9_dlo_px": p9_dlo,
                    "p7_only_px": p7only,
                    "p9_only_px": p9only,
                    "both_px": both,
                    "iou_p7_vs_p9": iou_p7_p9,
                    "p7_coverage_pct": 100.0 * p7_dlo / n_total,
                    "p9_coverage_pct": 100.0 * p9_dlo / n_total,
                })

                # Accumulate BGR histograms at disagreement pixels.
                hist_p7only += per_channel_hist(pre, (m7 == 1) & (m9 == 0))
                hist_p9only += per_channel_hist(pre, (m9 == 1) & (m7 == 0))
                hist_both += per_channel_hist(pre, (m7 == 1) & (m9 == 1))

            if (fi + 1) % 200 == 0:
                print(f"  {fi+1}/{n_frames} frames "
                      f"(running mean P7 cov={np.mean(cov_p7_full):.3f}%, P9 cov={np.mean(cov_p9_full):.3f}%)")

        cap.release()

        mean_p7 = float(np.mean(cov_p7_full))
        mean_p9 = float(np.mean(cov_p9_full))
        sanity_coverage[vid_name] = {
            "n_frames": n_frames,
            "p7_mean_coverage_pct": mean_p7,
            "p9_mean_coverage_pct": mean_p9,
        }
        print(f"  -> {vid_name}: P7 mean cov={mean_p7:.4f}%, P9 mean cov={mean_p9:.4f}%")

        # Per-video disagreement-stats and BGR-hist summaries.
        # Build means from per-channel histograms.
        def hist_mean_std(hist):
            # hist: (3, 256)
            n = int(hist[0].sum())
            if n == 0:
                return None, None, n
            bins = np.arange(256, dtype=np.float64)
            means = ((hist * bins[None, :]).sum(axis=1) / n).tolist()
            # variance via E[x^2] - E[x]^2
            sq = (hist * (bins[None, :] ** 2)).sum(axis=1) / n
            stds = np.sqrt(sq - np.asarray(means) ** 2).tolist()
            return means, stds, n

        p7o_means, p7o_stds, p7o_n = hist_mean_std(hist_p7only)
        p9o_means, p9o_stds, p9o_n = hist_mean_std(hist_p9only)
        b_means, b_stds, b_n = hist_mean_std(hist_both)

        with open(os.path.join(vid_out, "rgb_at_disagreement.json"), "w") as f:
            json.dump({
                "channel_order": "BGR",
                "p7_only": {
                    "n_pixels": p7o_n,
                    "channel_mean_bgr": p7o_means,
                    "channel_std_bgr": p7o_stds,
                    "hist_bgr": hist_p7only.tolist(),
                },
                "p9_only": {
                    "n_pixels": p9o_n,
                    "channel_mean_bgr": p9o_means,
                    "channel_std_bgr": p9o_stds,
                    "hist_bgr": hist_p9only.tolist(),
                },
                "both": {
                    "n_pixels": b_n,
                    "channel_mean_bgr": b_means,
                    "channel_std_bgr": b_stds,
                    "hist_bgr": hist_both.tolist(),
                },
            }, f, indent=2)

    # sample_indices.json
    with open(os.path.join(OUT_DIR, "sample_indices.json"), "w") as f:
        json.dump(sample_indices, f, indent=2)

    # disagreement_stats.json: per-frame + per-video aggregates + canonical sanity table.
    summary = {"per_video": {}, "sanity_coverage_full_video": sanity_coverage,
               "canonical": {
                   "p7": {"sample_1": 6.77, "sample_2": 12.37, "sample_3": 12.53, "sample_4": 12.28},
                   "p9_dlow3": {"sample_1": 2.88, "sample_2": 2.52, "sample_3": 0.37, "sample_4": 0.97},
               }}
    for vid, frames in per_frame_stats.items():
        if not frames:
            summary["per_video"][vid] = {"frames": []}
            continue
        m7_cov = [r["p7_coverage_pct"] for r in frames]
        m9_cov = [r["p9_coverage_pct"] for r in frames]
        iou = [r["iou_p7_vs_p9"] for r in frames if not np.isnan(r["iou_p7_vs_p9"])]
        p7only_cov = [100.0 * r["p7_only_px"] / 307200 for r in frames]
        p9only_cov = [100.0 * r["p9_only_px"] / 307200 for r in frames]
        both_cov = [100.0 * r["both_px"] / 307200 for r in frames]
        summary["per_video"][vid] = {
            "frames": frames,
            "mean_p7_coverage_pct_at_20_sampled": float(np.mean(m7_cov)),
            "mean_p9_coverage_pct_at_20_sampled": float(np.mean(m9_cov)),
            "mean_p7_only_pct_at_20_sampled": float(np.mean(p7only_cov)),
            "mean_p9_only_pct_at_20_sampled": float(np.mean(p9only_cov)),
            "mean_both_pct_at_20_sampled": float(np.mean(both_cov)),
            "mean_iou_p7_vs_p9_at_20_sampled": float(np.mean(iou)) if iou else None,
            "p7_p9_correlation_coefficient": correlation(m7_cov, m9_cov),
        }
    with open(os.path.join(OUT_DIR, "disagreement_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Pretty sanity check vs canonical
    print("\nSANITY CHECK — FULL-VIDEO MEAN COVERAGE vs CANONICAL:")
    can_p7 = {"sample_1": 6.77, "sample_2": 12.37, "sample_3": 12.53, "sample_4": 12.28}
    can_p9 = {"sample_1": 2.88, "sample_2": 2.52, "sample_3": 0.37, "sample_4": 0.97}
    for v in VIDEOS:
        s = sanity_coverage[v]
        dp7 = s["p7_mean_coverage_pct"] - can_p7[v]
        dp9 = s["p9_mean_coverage_pct"] - can_p9[v]
        ok7 = "OK" if abs(dp7) <= 0.5 else "FAIL"
        ok9 = "OK" if abs(dp9) <= 0.5 else "FAIL"
        print(f"  {v}: P7 mine={s['p7_mean_coverage_pct']:.4f} vs canon={can_p7[v]:.2f} (delta {dp7:+.3f} {ok7})  "
              f"|  P9 mine={s['p9_mean_coverage_pct']:.4f} vs canon={can_p9[v]:.2f} (delta {dp9:+.3f} {ok9})")

    print(f"\nAll D1 artefacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
