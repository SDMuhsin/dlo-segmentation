#!/usr/bin/env python3
"""Generate presentation figures for CREFLE March 2026 update.

Produces 5 figures into llmdocs/presentations/KIAT_CREFLE_UPDATE_MARCH2026/ using sample_09 (set=035):
  slide04_method.png      - 3D point cloud + projection plane
  slide06_rgbd_grid.png   - 6-column RGB/depth grid
  slide07_single_view.png - Single-view roundtrip comparison
  slide08_all_views.png   - All-views roundtrip comparison
  slide09_info_loss.png   - Per-class coverage + NN distance bar charts

Usage:
    python src/generate_presentation_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Add src to path for imports
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from pcl_to_rgbd import (
    load_sample, CLASS_COLORS_RGB, CLASS_NAMES, VIEWS, SAMPLES,
    reproject_single_view, reproject_views_to_pointcloud, labels_to_rgb,
    make_view_matrix,
)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = SRC_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "llmdocs" / "presentations" / "KIAT_CREFLE_UPDATE_MARCH2026"
SAMPLE_DIR = PROJECT_ROOT / "results" / "rgbd" / "sample_09"

SET_ID = 35
SAMPLE_ID = 0
DPI = 300
ELEV = 25
AZIM = -135
NN_THRESHOLD = 0.01


# ── Helpers ──────────────────────────────────────────────────────────────────

def class_color_norm(label):
    """Return normalized [0,1] RGB for a class label."""
    r, g, b = CLASS_COLORS_RGB.get(int(label), (128, 128, 128))
    return (r / 255, g / 255, b / 255)


def labels_to_colors(labels):
    """Convert label array to Nx3 float colors for matplotlib."""
    return np.array([class_color_norm(l) for l in labels])


def plot_pointcloud_3d(ax, points, colors, title, s=4):
    """Scatter 3D point cloud with consistent view angle and clean panes."""
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors, s=s, alpha=0.8, edgecolors="none", rasterized=True,
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.view_init(elev=ELEV, azim=AZIM)
    # Equal aspect ratio
    maxr = np.max(np.ptp(points, axis=0)) / 2
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - maxr, mid[0] + maxr)
    ax.set_ylim(mid[1] - maxr, mid[1] + maxr)
    ax.set_zlim(mid[2] - maxr, mid[2] + maxr)
    # Clean panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)
    ax.tick_params(labelsize=6)


def compute_coverage_metrics(orig_pts, orig_labels, rt_pts):
    """Compute per-class coverage and NN distances.

    Returns dict with:
        per_class: {label: {coverage_pct, mean_dist, count}}
        aggregate: {coverage_pct, mean_dist, median_dist, max_dist}
    """
    if rt_pts.shape[0] == 0:
        empty = {l: {"coverage_pct": 0.0, "mean_dist": float("inf"), "count": 0}
                 for l in range(5)}
        return {
            "per_class": empty,
            "aggregate": {"coverage_pct": 0.0, "mean_dist": float("inf"),
                          "median_dist": float("inf"), "max_dist": float("inf")},
        }
    tree = KDTree(rt_pts)
    dists, _ = tree.query(orig_pts)

    results = {"per_class": {}}
    for lbl in range(5):
        mask = orig_labels == lbl
        if mask.sum() == 0:
            results["per_class"][lbl] = {"coverage_pct": 100.0, "mean_dist": 0.0, "count": 0}
            continue
        d = dists[mask]
        cov = (d < NN_THRESHOLD).sum() / mask.sum() * 100
        results["per_class"][lbl] = {
            "coverage_pct": float(cov),
            "mean_dist": float(d.mean()),
            "count": int(mask.sum()),
        }

    cov_all = (dists < NN_THRESHOLD).sum() / len(dists) * 100
    results["aggregate"] = {
        "coverage_pct": float(cov_all),
        "mean_dist": float(dists.mean()),
        "median_dist": float(np.median(dists)),
        "max_dist": float(dists.max()),
    }
    return results


def class_legend():
    """Return list of matplotlib patches for class legend."""
    return [
        mpatches.Patch(color=class_color_norm(k), label=CLASS_NAMES[k])
        for k in sorted(CLASS_NAMES.keys())
    ]


# ── Figure Generators ────────────────────────────────────────────────────────

def generate_slide04(points, labels):
    """Slide 4: Method visualization — 3D cloud + projection plane."""
    print("  Generating slide04_method.png ...")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = labels_to_colors(labels)
    plot_pointcloud_3d(ax, points, colors, "Orthographic Projection Method", s=4)

    # Add semi-transparent projection plane at z_min - 0.2
    z_plane = points[:, 2].min() - 0.2
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    verts = [
        [xl[0], yl[0], z_plane],
        [xl[1], yl[0], z_plane],
        [xl[1], yl[1], z_plane],
        [xl[0], yl[1], z_plane],
    ]
    plane = Poly3DCollection([verts], alpha=0.15, facecolor="steelblue",
                             edgecolor="steelblue", linewidths=1.5)
    ax.add_collection3d(plane)

    # Draw dashed projection lines from sample points to the plane
    np.random.seed(42)
    sample_idxs = np.random.choice(len(points), size=3, replace=False)
    for idx in sample_idxs:
        px, py, pz = points[idx]
        ax.plot(
            [px, px], [py, py], [pz, z_plane],
            color="gray", linestyle="--", linewidth=1.0, alpha=0.7,
        )
        ax.scatter([px], [py], [z_plane], color="steelblue", s=15,
                   marker="x", zorder=5)

    # Text annotations
    ax.text2D(0.02, 0.95, "RGB = class colors", transform=ax.transAxes,
              fontsize=9, color="dimgray",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.8))
    ax.text2D(0.02, 0.89, "D = distance to image plane", transform=ax.transAxes,
              fontsize=9, color="dimgray",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.8))

    # Class legend
    ax.legend(handles=class_legend(), loc="upper right", fontsize=7,
              framealpha=0.9, title="Classes", title_fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "slide04_method.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def generate_slide06():
    """Slide 6: RGB-D grid — 6 views x 2 rows (RGB top, depth bottom)."""
    print("  Generating slide06_rgbd_grid.png ...")
    view_names = list(VIEWS.keys())
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    # Preload depth images for shared vmin/vmax
    depth_imgs = {}
    for vname in view_names:
        d = cv2.imread(str(SAMPLE_DIR / f"depth_{vname}.png"), cv2.IMREAD_UNCHANGED)
        depth_imgs[vname] = d

    # Compute shared depth range (excluding zeros)
    all_valid = np.concatenate([d[d > 0] for d in depth_imgs.values()])
    vmin, vmax = all_valid.min(), all_valid.max()

    for i, vname in enumerate(view_names):
        # Top row: RGB
        color = cv2.imread(str(SAMPLE_DIR / f"color_{vname}.png"), cv2.IMREAD_COLOR)
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(color_rgb)
        axes[0, i].set_title(vname.capitalize(), fontsize=10, fontweight="bold")
        axes[0, i].axis("off")

        # Bottom row: Depth
        d = depth_imgs[vname].astype(np.float32)
        d[d == 0] = np.nan
        im = axes[1, i].imshow(d, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[1, i].axis("off")

    # Row labels
    axes[0, 0].set_ylabel("RGB", fontsize=12, fontweight="bold", rotation=0,
                          labelpad=40, va="center")
    axes[1, 0].set_ylabel("Depth", fontsize=12, fontweight="bold", rotation=0,
                          labelpad=40, va="center")

    # Shared colorbar for depth
    cbar = fig.colorbar(im, ax=axes[1, :].tolist(), orientation="horizontal",
                        fraction=0.06, pad=0.08, aspect=40)
    cbar.set_label("Depth (mm)", fontsize=10)

    fig.suptitle("RGB-D Output: 6 Canonical Views", fontsize=14,
                 fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "slide06_rgbd_grid.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def generate_slide07(points, labels, sv_pts):
    """Slide 7: Single-view roundtrip — original / front roundtrip / diff."""
    print("  Generating slide07_single_view.png ...")
    colors_orig = labels_to_colors(labels)
    sv_colors = sv_pts[1] / 255.0 if sv_pts[1].shape[0] > 0 else np.empty((0, 3))
    sv_points = sv_pts[0]

    # Subsample roundtrip for rendering
    if sv_points.shape[0] > 10000:
        idx = np.random.choice(sv_points.shape[0], 10000, replace=False)
        sv_points_plot = sv_points[idx]
        sv_colors_plot = sv_colors[idx]
    else:
        sv_points_plot = sv_points
        sv_colors_plot = sv_colors

    # Compute NN distances for diff panel
    if sv_points.shape[0] > 0:
        tree = KDTree(sv_points)
        dists, _ = tree.query(points)
    else:
        dists = np.full(len(points), np.inf)

    # Axis limits from original bounding box
    maxr = np.max(np.ptp(points, axis=0)) / 2 * 1.1
    mid = points.mean(axis=0)

    fig = plt.figure(figsize=(16, 5.5))

    # Panel 1: Original
    ax1 = fig.add_subplot(131, projection="3d")
    plot_pointcloud_3d(ax1, points, colors_orig, f"Original ({len(points)} pts)", s=4)

    # Panel 2: Front roundtrip
    ax2 = fig.add_subplot(132, projection="3d")
    plot_pointcloud_3d(ax2, sv_points_plot, sv_colors_plot,
                       f"Front-View Roundtrip ({sv_points.shape[0]} pts)", s=0.5)

    # Panel 3: Diff — only show lost points (above threshold)
    ax3 = fig.add_subplot(133, projection="3d")
    captured_mask = dists < NN_THRESHOLD
    lost_mask = ~captured_mask
    n_captured = captured_mask.sum()
    n_lost = lost_mask.sum()
    pct_captured = n_captured / len(dists) * 100
    pct_lost = 100 - pct_captured

    # Ghost captured points for spatial context
    if n_captured > 0:
        ax3.scatter(
            points[captured_mask, 0], points[captured_mask, 1], points[captured_mask, 2],
            c="lightgray", s=1, alpha=0.15, edgecolors="none", rasterized=True,
        )
    # Plot lost points colored by NN distance
    if n_lost > 0:
        lost_dists = dists[lost_mask]
        cmap = plt.cm.YlOrRd
        norm_lost = np.clip(lost_dists / max(lost_dists.max(), NN_THRESHOLD * 5), 0, 1)
        lost_colors = cmap(norm_lost)[:, :3]
        ax3.scatter(
            points[lost_mask, 0], points[lost_mask, 1], points[lost_mask, 2],
            c=lost_colors, s=6, alpha=0.9, edgecolors="none", rasterized=True,
        )
    ax3.set_title(f"Lost Points ({n_lost} pts, {pct_lost:.1f}%)",
                  fontsize=11, fontweight="bold", pad=8)
    ax3.view_init(elev=ELEV, azim=AZIM)

    stats_text = (
        f"Captured: {pct_captured:.1f}%\n"
        f"Lost: {pct_lost:.1f}%\n"
        f"Mean NN: {dists.mean():.4f}\n"
        f"Max NN: {dists.max():.4f}"
    )
    ax3.text2D(0.02, 0.95, stats_text, transform=ax3.transAxes, fontsize=8,
               verticalalignment="top",
               bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

    # Set consistent limits
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(mid[0] - maxr, mid[0] + maxr)
        ax.set_ylim(mid[1] - maxr, mid[1] + maxr)
        ax.set_zlim(mid[2] - maxr, mid[2] + maxr)

    fig.suptitle("Single-View (Front) Roundtrip Analysis", fontsize=14,
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "slide07_single_view.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def generate_slide08(points, labels, av_pts):
    """Slide 8: All-views roundtrip — original / all-6-views roundtrip / diff."""
    print("  Generating slide08_all_views.png ...")
    colors_orig = labels_to_colors(labels)
    av_points, av_rgb = av_pts
    av_colors = av_rgb / 255.0

    # Subsample roundtrip for rendering
    if av_points.shape[0] > 10000:
        idx = np.random.choice(av_points.shape[0], 10000, replace=False)
        av_points_plot = av_points[idx]
        av_colors_plot = av_colors[idx]
    else:
        av_points_plot = av_points
        av_colors_plot = av_colors

    # NN distances for diff
    tree = KDTree(av_points)
    dists, _ = tree.query(points)

    maxr = np.max(np.ptp(points, axis=0)) / 2 * 1.1
    mid = points.mean(axis=0)

    fig = plt.figure(figsize=(16, 5.5))

    ax1 = fig.add_subplot(131, projection="3d")
    plot_pointcloud_3d(ax1, points, colors_orig, f"Original ({len(points)} pts)", s=4)

    ax2 = fig.add_subplot(132, projection="3d")
    plot_pointcloud_3d(ax2, av_points_plot, av_colors_plot,
                       f"All-Views Roundtrip ({av_points.shape[0]} pts)", s=0.5)

    # Panel 3: Diff — only show lost points (above threshold)
    ax3 = fig.add_subplot(133, projection="3d")
    captured_mask = dists < NN_THRESHOLD
    lost_mask = ~captured_mask
    n_captured = captured_mask.sum()
    n_lost = lost_mask.sum()
    pct_captured = n_captured / len(dists) * 100
    pct_lost = 100 - pct_captured

    # Ghost captured points for spatial context
    if n_captured > 0:
        ax3.scatter(
            points[captured_mask, 0], points[captured_mask, 1], points[captured_mask, 2],
            c="lightgray", s=1, alpha=0.15, edgecolors="none", rasterized=True,
        )
    # Plot lost points colored by NN distance
    if n_lost > 0:
        lost_dists = dists[lost_mask]
        cmap = plt.cm.YlOrRd
        norm_lost = np.clip(lost_dists / max(lost_dists.max(), NN_THRESHOLD * 5), 0, 1)
        lost_colors = cmap(norm_lost)[:, :3]
        ax3.scatter(
            points[lost_mask, 0], points[lost_mask, 1], points[lost_mask, 2],
            c=lost_colors, s=6, alpha=0.9, edgecolors="none", rasterized=True,
        )
    ax3.set_title(f"Lost Points ({n_lost} pts, {pct_lost:.1f}%)",
                  fontsize=11, fontweight="bold", pad=8)
    ax3.view_init(elev=ELEV, azim=AZIM)

    stats_text = (
        f"Captured: {pct_captured:.1f}%\n"
        f"Lost: {pct_lost:.1f}%\n"
        f"Mean NN: {dists.mean():.4f}\n"
        f"Max NN: {dists.max():.4f}"
    )
    ax3.text2D(0.02, 0.95, stats_text, transform=ax3.transAxes, fontsize=8,
               verticalalignment="top",
               bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(mid[0] - maxr, mid[0] + maxr)
        ax.set_ylim(mid[1] - maxr, mid[1] + maxr)
        ax.set_zlim(mid[2] - maxr, mid[2] + maxr)

    fig.suptitle("Multi-View (All 6) Roundtrip Analysis", fontsize=14,
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "slide08_all_views.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def generate_slide09(points, labels, sv_pts, av_pts):
    """Slide 9: Information loss — per-class coverage + NN distance bar charts."""
    print("  Generating slide09_info_loss.png ...")
    sv_metrics = compute_coverage_metrics(points, labels, sv_pts[0])
    av_metrics = compute_coverage_metrics(points, labels, av_pts[0])

    class_labels = [CLASS_NAMES[i] for i in range(5)]
    x = np.arange(5)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Per-class coverage
    sv_cov = [sv_metrics["per_class"][i]["coverage_pct"] for i in range(5)]
    av_cov = [av_metrics["per_class"][i]["coverage_pct"] for i in range(5)]

    bars1 = ax1.bar(x - width / 2, sv_cov, width, label="Single View (Front)",
                    color="#e74c3c", alpha=0.85)
    bars2 = ax1.bar(x + width / 2, av_cov, width, label="All 6 Views",
                    color="#2ecc71", alpha=0.85)

    ax1.set_ylabel("Coverage (%)", fontsize=11)
    ax1.set_title("Per-Class Point Coverage", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_labels, fontsize=9)
    ax1.set_ylim(0, 110)
    ax1.legend(fontsize=9)
    ax1.axhline(y=100, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax1.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7)

    # Panel B: NN distance statistics
    stat_labels = ["Mean", "Median", "Max"]
    sv_stats = [
        sv_metrics["aggregate"]["mean_dist"],
        sv_metrics["aggregate"]["median_dist"],
        sv_metrics["aggregate"]["max_dist"],
    ]
    av_stats = [
        av_metrics["aggregate"]["mean_dist"],
        av_metrics["aggregate"]["median_dist"],
        av_metrics["aggregate"]["max_dist"],
    ]

    x2 = np.arange(len(stat_labels))
    bars3 = ax2.bar(x2 - width / 2, sv_stats, width, label="Single View (Front)",
                    color="#e74c3c", alpha=0.85)
    bars4 = ax2.bar(x2 + width / 2, av_stats, width, label="All 6 Views",
                    color="#2ecc71", alpha=0.85)

    ax2.set_ylabel("NN Distance (world units)", fontsize=11)
    ax2.set_title("Nearest-Neighbor Distance Statistics", fontsize=12,
                  fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(stat_labels, fontsize=9)
    ax2.legend(fontsize=9)
    ax2.axhline(y=NN_THRESHOLD, color="orange", linestyle="--", linewidth=1.5,
                label=f"Threshold ({NN_THRESHOLD})")
    ax2.legend(fontsize=9)

    # Value labels
    for bar in bars3:
        h = bar.get_height()
        ax2.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7)
    for bar in bars4:
        h = bar.get_height()
        ax2.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7)

    fig.suptitle("Information Loss: Single-View vs Multi-View", fontsize=14,
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "slide09_info_loss.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Generating CREFLE March 2026 Presentation Figures")
    print(f"  Sample: sample_09 (set={SET_ID:03d}, sample={SAMPLE_ID})")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Load data once
    print("\nLoading data ...")
    points, labels = load_sample(SET_ID, SAMPLE_ID)
    print(f"  Point cloud: {points.shape}, labels: {np.unique(labels)}")

    # Reproject single view (front)
    print("  Reprojecting front view ...")
    sv_pts = reproject_single_view(SAMPLE_DIR, "front")
    print(f"  Front roundtrip: {sv_pts[0].shape[0]} points")

    # Reproject all 6 views
    print("  Reprojecting all 6 views ...")
    av_pts = reproject_views_to_pointcloud(SAMPLE_DIR)
    print(f"  All-views roundtrip: {av_pts[0].shape[0]} points")

    # Generate all figures
    print("\nGenerating figures ...")
    generate_slide04(points, labels)
    generate_slide06()
    generate_slide07(points, labels, sv_pts)
    generate_slide08(points, labels, av_pts)
    generate_slide09(points, labels, sv_pts, av_pts)

    print("\nAll figures generated successfully!")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
