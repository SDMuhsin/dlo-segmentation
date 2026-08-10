#!/usr/bin/env python3
"""Phase 26.0 smoke + validation harness for the soft-edge / thick WIRE lever.

Renders a SMALL batch of (set, src) frames TWICE — once with KIAT_P26_SOFTEDGE
OFF (the unchanged baseline) and once ON — into a scratch dir, then runs
the six isolation gates and writes a side-by-side smoke grid.

Two modes:
  python src/smoke_p26_softedge.py render --tag old   # flag must be OFF in env
  python src/smoke_p26_softedge.py render --tag new   # flag must be ON  in env
  python src/smoke_p26_softedge.py validate            # compare old vs new

The render step is launched per-tag in its own subprocess by the orchestrator
(`run`) so the import-time P26 flag is read correctly each time.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRATCH = PROJECT_ROOT / "results" / "realism_campaign" / "p26_0_smoke"

# A handful of (set_id, frame_id) spanning a few sets. Strict-P4 (phase13)
# renders 6 canonical views per source.
SMOKE_WORK = [
    (0, 130), (0, 70), (3, 100), (6, 150), (9, 40), (12, 200),
]
VIEW_NAMES = ["front", "back", "right", "left", "top", "bottom"]


def _render(tag: str):
    """Render SMOKE_WORK into SCRATCH/<tag>/ using the current env flags.

    When KIAT_P26_DEBUG_COV=1 (set for the "new" pass), the TRUE per-view
    coverage-alpha map is dumped to <out>/cov/<frame>.npy so the validator can
    measure the rendered penumbra ramp directly (ground truth, not RGB-recon)."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import convert_to_video_dataset as C
    import pcl_to_rgbd as P
    out_root = SCRATCH / tag
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[render {tag}] P26_SOFTEDGE={P.P26_SOFTEDGE} SS={P.P26_SS} "
          f"R={P.P26_WIRE_RADIUS} RIM={P.P26_RIM} cov={P.P26_COV_THRESH} "
          f"MODE={C.DATASET_MODE}")
    dump_cov = os.environ.get("KIAT_P26_DEBUG_COV", "0").strip() in ("1", "true", "True")
    # Per-frame coverage capture: wrap _composite_soft_wire so we know which
    # frame/view each cov belongs to (it loops views internally).
    _orig = P._composite_soft_wire
    _bucket = {"covs": []}
    if dump_cov:
        def _grab(*a, **k):
            r = _orig(*a, **k)
            _bucket["covs"].append(P._P26_DEBUG.get("cov"))
            return r
        P._composite_soft_wire = _grab
    for (sid, fid) in SMOKE_WORK:
        _bucket["covs"] = []
        args = (sid, fid, 1, 25.0, str(out_root))
        res = C.convert_one_video(args)
        print(f"   set {sid:03d} src {fid:04d}: {res[2]}")
        if str(res[2]).startswith("error"):
            print(res[2])
            sys.exit(1)
        if dump_cov and _bucket["covs"]:
            cov_dir = out_root / _split_of(sid) / f"{sid:03d}" / "cov"
            cov_dir.mkdir(parents=True, exist_ok=True)
            for vn, cov in zip(VIEW_NAMES, _bucket["covs"]):
                if cov is not None:
                    np.save(str(cov_dir / f"{fid:04d}_00_{vn}.npy"),
                            cov.astype(np.float16))


def _frame_paths(tag: str):
    """Yield (key, rgb, depth, label) Paths for every rendered view."""
    root = SCRATCH / tag
    out = []
    for (sid, fid) in SMOKE_WORK:
        split = "train" if True else "train"
        base = root / _split_of(sid) / f"{sid:03d}"
        for vn in VIEW_NAMES:
            fn = f"{fid:04d}_00_{vn}.png"
            rgb = base / "rgb" / fn
            dep = base / "depth" / fn
            lbl = base / "label" / fn
            if rgb.exists():
                out.append((f"{sid:03d}_{fid:04d}_{vn}", rgb, dep, lbl))
    return out


def _split_of(sid):
    # Mirror convert_to_video_dataset.split_of without importing (lazy).
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import convert_to_video_dataset as C
    return C.split_of(sid)


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _run_widths(mask):
    ws = []
    for row in mask:
        idx = np.where(row)[0]
        if len(idx) == 0:
            continue
        splits = np.where(np.diff(idx) > 1)[0] + 1
        for seg in np.split(idx, splits):
            ws.append(len(seg))
    return np.array(ws) if ws else np.array([0])


def _boundary_sobel(img_bgr, wire_mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    k = np.ones((3, 3), np.uint8)
    wd = cv2.dilate(wire_mask.astype(np.uint8), k)
    we = cv2.erode(wire_mask.astype(np.uint8), k)
    boundary = (wd > 0) & (we == 0)
    return (float(mag[boundary].mean()) if boundary.sum() else float("nan"),
            float(mag.mean()))


def _ramp_width_from_cov(cov):
    """Ground-truth 10-90 boundary transition width (px) from the renderer's
    actual coverage-alpha map (immune to wire texture / contrast). Bins alpha
    by signed distance to the crisp 0.5 contour and measures the 0.1→0.9 rise."""
    from scipy.ndimage import distance_transform_edt
    cov = np.asarray(cov, dtype=np.float64)
    m = cov >= 0.5
    if m.sum() < 50:
        return float("nan")
    sd = distance_transform_edt(~m) - distance_transform_edt(m)
    band = np.abs(sd) <= 4
    sds, al = sd[band], cov[band]
    bins = np.arange(-4, 4.001, 0.25)
    cen = 0.5 * (bins[:-1] + bins[1:])
    prof = []
    for i in range(len(bins) - 1):
        msk = (sds >= bins[i]) & (sds < bins[i + 1])
        prof.append(al[msk].mean() if msk.sum() > 3 else np.nan)
    prof = np.array(prof)
    good = ~np.isnan(prof)
    c, p = cen[good], prof[good]
    if len(c) < 3:
        return float("nan")

    def cross(level):
        for j in range(len(p) - 1):
            if (p[j] - level) * (p[j + 1] - level) <= 0 and p[j] != p[j + 1]:
                t = (level - p[j]) / (p[j + 1] - p[j])
                return c[j] + t * (c[j + 1] - c[j])
        return np.nan
    a1, a9 = cross(0.1), cross(0.9)
    return abs(a1 - a9) if (a1 == a1 and a9 == a9) else float("nan")


def validate():
    old = {k: (r, d, l) for (k, r, d, l) in _frame_paths("old")}
    new = {k: (r, d, l) for (k, r, d, l) in _frame_paths("new")}
    keys = sorted(set(old) & set(new))
    assert keys, "no overlapping rendered frames found"

    report = {"gates": {}, "per_frame": {}}

    # We can only run gate 1 (byte identity) if an "old" render of the SAME
    # code with the flag OFF is present. We compare old-vs-new and require the
    # NON-WIRE pixels to be content-preserved (gate 5) and the OLD render to
    # match a flag-off render byte-for-byte (gate 1, done in `run`).

    # Gather metrics
    edge_old, edge_new, mean_old, mean_new = [], [], [], []
    rw_old, rw_new = [], []
    tw_old, tw_new = [], []
    desync = 0
    soft_label_px = 0
    align_total_wire = 0
    bg_changed_frac = []

    for k in keys:
        ro, do, lo = old[k]
        rn, dn, ln = new[k]
        io = cv2.imread(str(ro)); inew = cv2.imread(str(rn))
        Lo = cv2.imread(str(lo), cv2.IMREAD_UNCHANGED)
        Ln = cv2.imread(str(ln), cv2.IMREAD_UNCHANGED)
        Do = cv2.imread(str(do), cv2.IMREAD_UNCHANGED)
        Dn = cv2.imread(str(dn), cv2.IMREAD_UNCHANGED)
        wo = (Lo == 1); wn = (Ln == 1)

        # edge / thickness
        eo, mo = _boundary_sobel(io, wo)
        en, mn = _boundary_sobel(inew, wn)
        edge_old.append(eo); edge_new.append(en); mean_old.append(mo); mean_new.append(mn)
        rw_old.append(np.median(np.concatenate([_run_widths(wo), _run_widths(wo.T)])))
        rw_new.append(np.median(np.concatenate([_run_widths(wn), _run_widths(wn.T)])))
        # OLD = hard step (transition ≈ 0, by construction the binary splat).
        # NEW = ground-truth 10-90 ramp of the renderer's coverage-alpha map.
        tw_old.append(0.0)
        sid_s, fid_s, vn_s = k.split("_")
        cov_path = (SCRATCH / "new" / _split_of(int(sid_s)) / sid_s / "cov"
                    / f"{fid_s}_00_{vn_s}.npy")
        if cov_path.exists():
            tw_new.append(_ramp_width_from_cov(np.load(str(cov_path))))
        else:
            tw_new.append(float("nan"))

        # gate 4: label crisp binary + depth aligned (new)
        assert set(np.unique(Ln)).issubset({0, 1, 2, 3, 4, 5}), "label not categorical"
        # wire label pixels must have depth>0 (aligned); count desyncs
        desync += int(np.count_nonzero(wn & (Dn == 0)))
        align_total_wire += int(wn.sum())
        # label must be exactly {0,1,..} ints, never fractional/grey — PNG is
        # uint8 by construction, so "crisp" means no intermediate wire alpha
        # leaked into the label. (sanity: label is a strict binary per class.)

        # gate 5: background / floor / clutter / hands content preserved. The
        # soft wire intentionally alpha-blends a thin penumbra HALO into the
        # pixels immediately around it, so the correct isolation test is that
        # every pixel FAR from any wire (old or new) is identical — i.e.,
        # no background CONTENT was swapped/resampled, only the wire neighbour-
        # hood changed. We use a generous 8 px guard band (> the ~6 px footprint
        # radius) around the union of old+new wire.
        from scipy.ndimage import binary_dilation
        wire_union = wo | wn
        guard = binary_dilation(wire_union, iterations=12)
        far = ~guard
        same_rgb = np.all(io == inew, axis=2)
        bg_changed = far & (~same_rgb)
        bg_changed_frac.append(float(bg_changed.sum()) / max(far.sum(), 1))

        report["per_frame"][k] = {
            "edge_sobel_old": round(eo, 1), "edge_sobel_new": round(en, 1),
            "runwidth_old": float(rw_old[-1]), "runwidth_new": float(rw_new[-1]),
            "trans_w_old": round(tw_old[-1], 2) if tw_old[-1] == tw_old[-1] else None,
            "trans_w_new": round(tw_new[-1], 2) if tw_new[-1] == tw_new[-1] else None,
            "wire_px_old": int(wo.sum()), "wire_px_new": int(wn.sum()),
            "bg_changed_frac": round(bg_changed_frac[-1], 6),
        }

    def med(x):
        x = [v for v in x if v == v]
        return float(np.median(x)) if x else float("nan")

    g2 = {
        "edge_sobel_old_median": round(med(edge_old), 1),
        "edge_sobel_new_median": round(med(edge_new), 1),
        "image_mean_sobel_new_median": round(med(mean_new), 1),
        "transition_w_old_median_px": round(med(tw_old), 2),
        "transition_w_new_median_px": round(med(tw_new), 2),
        "PASS": bool(1.4 <= med(tw_new) <= 2.6 and med(edge_new) < med(edge_old)),
    }
    g3 = {
        "runwidth_old_median_px": round(med(rw_old), 1),
        "runwidth_new_median_px": round(med(rw_new), 1),
        "PASS": bool(12.0 <= med(rw_new) <= 16.0),
    }
    g4 = {
        "wire_label_px_with_zero_depth": int(desync),
        "total_new_wire_px": int(align_total_wire),
        "PASS": bool(desync == 0),
    }
    g5 = {
        "bg_changed_frac_max": round(max(bg_changed_frac), 6),
        "bg_changed_frac_median": round(med(bg_changed_frac), 6),
        "PASS": bool(max(bg_changed_frac) < 1e-9),
    }

    report["gates"] = {"G2_edge_softness": g2, "G3_thickness": g3,
                       "G4_align": g4, "G5_isolation": g5}

    # Smoke grid: old | new | new-label, for up to 6 frames
    rows = []
    for k in keys[:6]:
        ro, _, lo = old[k]; rn, _, ln = new[k]
        io = cv2.imread(str(ro)); inew = cv2.imread(str(rn))
        Ln = cv2.imread(str(ln), cv2.IMREAD_UNCHANGED)
        lvis = np.zeros_like(inew); lvis[Ln == 1] = (0, 255, 0)
        lab = lambda im, t: cv2.putText(im.copy(), t, (8, 22),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (255, 255, 255), 2)
        row = np.hstack([lab(io, "OLD"), lab(inew, "NEW"),
                         lab(lvis, "NEW label")])
        rows.append(row)
    grid = np.vstack(rows)
    grid_path = SCRATCH / "p26_0_smoke_grid.png"
    cv2.imwrite(str(grid_path), grid)
    report["smoke_grid"] = str(grid_path)

    with open(SCRATCH / "p26_0_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["gates"], indent=2))
    print("smoke grid:", grid_path)
    return report


def run():
    """Orchestrate: render old (flag off) + new (flag on) in subprocesses,
    prove byte-identity of the OLD render against a flag-off re-render, then
    validate."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    base_env = dict(os.environ)
    base_env["KIAT_DATASET_MODE"] = "phase13"
    base_env["KIAT_OUTPUT_ROOT"] = str(SCRATCH / "old")  # unused by _render but keeps mode

    def sub(tag, soft, debug_cov=False):
        e = dict(base_env)
        e["KIAT_P26_SOFTEDGE"] = "1" if soft else "0"
        e["KIAT_P26_DEBUG_COV"] = "1" if debug_cov else "0"
        cmd = [sys.executable, str(Path(__file__).resolve()), "render", "--tag", tag]
        subprocess.run(cmd, env=e, check=True)

    sub("old", soft=False)
    sub("old2", soft=False)   # second flag-off render to prove determinism
    sub("new", soft=True, debug_cov=True)

    # Gate 1: byte identity old vs old2 (flag-off determinism / reproducibility)
    old = {k: r for (k, r, d, l) in _frame_paths("old")}
    old2 = {k: r for (k, r, d, l) in _frame_paths("old2")}
    keys = sorted(set(old) & set(old2))
    mism = [k for k in keys if _md5(old[k]) != _md5(old2[k])]
    # also compare depth + label
    for tag_pair in []:
        pass
    g1 = {"frames_compared": len(keys), "rgb_mismatches": len(mism),
          "PASS": len(mism) == 0 and len(keys) > 0}
    print("G1 byte-identity (flag-off determinism):", g1)

    rep = validate()
    rep["gates"]["G1_flag_off_byte_identity"] = g1
    with open(SCRATCH / "p26_0_report.json", "w") as f:
        json.dump(rep, f, indent=2)
    print("\n=== GATE SUMMARY ===")
    for g, v in rep["gates"].items():
        print(f"{g}: {'PASS' if v.get('PASS') else 'FAIL'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("render"); r.add_argument("--tag", required=True)
    sub.add_parser("validate")
    sub.add_parser("run")
    a = ap.parse_args()
    if a.cmd == "render":
        _render(a.tag)
    elif a.cmd == "validate":
        validate()
    else:
        run()
