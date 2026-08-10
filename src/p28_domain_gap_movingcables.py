"""P28: quantify the domain gap between MovingCables and our REAL valset.
Model-free. Compares foreground fraction, typical wire WIDTH (px), and color,
to predict transfer before committing GPU. No training, no valset training use.
"""
import glob, os, numpy as np, cv2

def wire_width_px(mask_fg):
    """Typical local wire width = 2 * median distance-transform over foreground."""
    if mask_fg.sum() == 0:
        return np.nan, 0.0
    dt = cv2.distanceTransform(mask_fg.astype(np.uint8), cv2.DIST_L2, 5)
    vals = dt[mask_fg > 0]
    return float(2.0 * np.median(vals)), float(mask_fg.mean())

def summarize(name, mask_paths, rgb_paths, fg_from):
    widths, fgs, cols = [], [], []
    for mp in mask_paths:
        m = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[..., 0]
        fg = fg_from(m)
        w, f = wire_width_px(fg)
        if not np.isnan(w):
            widths.append(w); fgs.append(f)
    for rp in rgb_paths[:120]:
        im = cv2.imread(rp)  # BGR
        if im is not None:
            cols.append(im.reshape(-1, 3).mean(0))
    widths, fgs = np.array(widths), np.array(fgs)
    col = np.array(cols).mean(0) if cols else np.array([np.nan]*3)
    print(f"\n=== {name}  (n_masks={len(widths)}) ===")
    print(f"  fg_fraction   median={np.median(fgs):.4f}  mean={fgs.mean():.4f}  "
          f"p05={np.percentile(fgs,5):.4f} p95={np.percentile(fgs,95):.4f}")
    print(f"  wire_width_px median={np.median(widths):.2f}  mean={widths.mean():.2f}  "
          f"p05={np.percentile(widths,5):.2f} p95={np.percentile(widths,95):.2f}")
    print(f"  color BGR mean=({col[0]:.0f},{col[1]:.0f},{col[2]:.0f})  "
          f"R-B={col[2]-col[0]:+.0f} (warm>0/cool<0)")
    return dict(fg_med=float(np.median(fgs)), w_med=float(np.median(widths)))

# --- our REAL valset (target morphology) ---
val_masks = sorted(glob.glob("data/real_wires_valset/masks/*.jpg"))
val_rgb   = sorted(glob.glob("data/real_wires_valset/imgs/*.jpg"))
v = summarize("REAL VALSET (target)", val_masks, val_rgb, fg_from=lambda m: (m > 127).astype(np.uint8))

# --- MovingCables converted (built only after the 11.5GB download finishes) ---
mc_masks = sorted(glob.glob("data/dformer_dataset_movingcables/Label/*.png"))
mc_rgb   = sorted(glob.glob("data/dformer_dataset_movingcables/RGB/*.png"))
if mc_masks:
    m = summarize("MovingCables", mc_masks, mc_rgb, fg_from=lambda m: (m >= 3).astype(np.uint8))
    print("\n=== GAP (MovingCables / valset) ===")
    print(f"  fg-fraction ratio : {m['fg_med']/max(v['fg_med'],1e-9):.1f}x")
    print(f"  wire-width  ratio : {m['w_med']/max(v['w_med'],1e-9):.1f}x")
else:
    print("\n[MovingCables converted set not built yet] — sample fg≈0.33, visually thick cables.")
    print(f"  fg-fraction gap vs valset median {v['fg_med']:.4f}: ≈ {0.33/max(v['fg_med'],1e-9):.0f}x denser")
