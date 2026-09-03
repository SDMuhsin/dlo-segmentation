#!/usr/bin/env python
"""Validate P44 connector-paste composites BEFORE any training run.

Checks, in order of how badly this project has been burned by each:
  1. GT integrity   - every non-background pixel of the source label is unchanged.
  2. Edge width     - pasted connector edges must match RENDERED connector edges
                      (~0.97 px measured). A systematic difference is a shortcut the
                      model will happily learn instead of connector appearance.
  3. Scale          - pasted bbox distribution vs the rendered one, so we know exactly
                      how far the arm moves the scale statistics (the P3W cue).
  4. Label sanity   - values stay in {0,1,2}; connector pixel share per frame.
  5. Wire pastes    - P44.3 only: pasted-cable edge width and apparent STROKE WIDTH must
                      match the RENDERED wire, and the achieved
                      P(class | pixel came from a real photograph) must be near 1/3 each.
                      A stroke-width mismatch is the same shortcut as an edge-width one
                      ("real cable is always fatter/thinner than rendered cable"), and an
                      unbalanced P(class|photographic) is precisely what sank P44 (1.00
                      connector) and P44.2 (0.51 connector / 0.00 wire).
"""
import argparse, json, os, random
import cv2, numpy as np

BG, WIRE, CONN = 0, 1, 2


def edge_width(rgb, mask):
    """Approximate 10-90% luma rise distance across a mask boundary, in px."""
    g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = mask.astype(np.uint8)
    er = cv2.erode(m, np.ones((3, 3), np.uint8))
    dl = cv2.dilate(m, np.ones((3, 3), np.uint8))
    band = (dl - er).astype(bool)
    if band.sum() < 30 or er.sum() < 10:
        return None
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx ** 2 + gy ** 2)[band].mean()
    inside, outside = g[er.astype(bool)], g[(dl - m).astype(bool)]
    if len(outside) < 10:
        return None
    c = abs(inside.mean() - outside.mean())
    if c < 5 or gm < 1e-6:
        return None
    return c / (gm / 4.0)


def stroke_width(mask):
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    v = d[mask.astype(bool)]
    return float(2 * np.percentile(v, 95)) if v.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paste-dir", required=True)
    ap.add_argument("--base", default="data/dformer_dataset_3way_connscale3")
    ap.add_argument("--n", type=int, default=250)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.paste_dir, "composite_meta.json")))
    frames = meta["frames"]
    random.seed(0)
    sample = random.sample(frames, min(a.n, len(frames)))

    gt_fail, bad_vals = 0, 0
    ew_paste, ew_render, paste_dims, render_dims, conn_share = [], [], [], [], []
    ew_wpaste, ew_wrender, sw_wpaste, sw_wrender, wire_share = [], [], [], [], []

    for fr in sample:
        lp_new = os.path.join(a.paste_dir, "Label", fr["out"] + ".png")
        rp_new = os.path.join(a.paste_dir, "RGB", fr["out"] + ".png")
        lp_old = os.path.join(a.base, "Label", fr["base"] + ".png")
        rp_old = os.path.join(a.base, "RGB", fr["base"] + ".png")
        ln, rn = cv2.imread(lp_new, cv2.IMREAD_UNCHANGED), cv2.imread(rp_new)
        lo, ro = cv2.imread(lp_old, cv2.IMREAD_UNCHANGED), cv2.imread(rp_old)
        if any(x is None for x in (ln, rn, lo, ro)):
            continue
        if not set(np.unique(ln)).issubset({BG, WIRE, CONN}):
            bad_vals += 1
        keep = lo != BG
        if not (ln[keep] == lo[keep]).all():
            gt_fail += 1
        conn_share.append((ln == CONN).mean())
        wire_share.append((ln == WIRE).mean())

        # pasted connector regions = new CONN pixels that were BG before
        pasted = ((ln == CONN) & (lo == BG)).astype(np.uint8)
        rendered = (lo == CONN).astype(np.uint8)
        # P44.3: pasted WIRE regions = new WIRE pixels that were BG before
        w_pasted = ((ln == WIRE) & (lo == BG)).astype(np.uint8)
        w_render = (lo == WIRE).astype(np.uint8)
        for m_, ews, sws, img in ((w_pasted, ew_wpaste, sw_wpaste, rn),
                                  (w_render, ew_wrender, sw_wrender, ro)):
            n, lbl, st, _ = cv2.connectedComponentsWithStats(m_, 8)
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] < 60:
                    continue
                comp = (lbl == i).astype(np.uint8)
                w = edge_width(img, comp)
                if w is not None:
                    ews.append(w)
                sws.append(stroke_width(comp))

        for m_, store, img in ((pasted, ew_paste, rn), (rendered, ew_render, ro)):
            n, lbl, st, _ = cv2.connectedComponentsWithStats(m_, 8)
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] < 40:
                    continue
                comp = (lbl == i).astype(np.uint8)
                w = edge_width(img, comp)
                if w is not None:
                    store.append(w)
                (paste_dims if store is ew_paste else render_dims).append(
                    max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]))

    def q(v):
        return np.percentile(v, [5, 25, 50, 75, 95]).round(2) if len(v) else "n/a"

    print(f"frames checked            : {len(sample)}")
    print(f"1. GT overwritten         : {gt_fail}  (MUST be 0)")
    print(f"   label values not 0/1/2 : {bad_vals}  (MUST be 0)")
    print(f"2. edge width px, pasted  : median {np.median(ew_paste):.2f}  n={len(ew_paste)}"
          if ew_paste else "2. edge width px, pasted  : n/a")
    print(f"   edge width px, rendered: median {np.median(ew_render):.2f}  n={len(ew_render)}"
          if ew_render else "   edge width px, rendered: n/a")
    if ew_paste and ew_render:
        d = np.median(ew_paste) - np.median(ew_render)
        print(f"   difference             : {d:+.2f} px  "
              f"{'OK' if abs(d) < 0.35 else 'WARNING - potential paste shortcut'}")
    print(f"3. bbox px pasted   p5-95 : {q(paste_dims)}")
    print(f"   bbox px rendered p5-95 : {q(render_dims)}")
    print(f"4. connector px share/frame: mean {np.mean(conn_share):.4f} "
          f"(base render mean 0.0039)")
    if sw_wpaste:
        print(f"5. WIRE pastes (P44.3)")
        print(f"   edge width px, pasted  : median {np.median(ew_wpaste):.2f}  "
              f"n={len(ew_wpaste)}")
        print(f"   edge width px, rendered: median {np.median(ew_wrender):.2f}  "
              f"n={len(ew_wrender)}")
        d = np.median(ew_wpaste) - np.median(ew_wrender)
        print(f"   difference             : {d:+.2f} px  "
              f"{'OK' if abs(d) < 0.35 else 'WARNING - potential paste shortcut'}")
        print(f"   stroke width px pasted  p5-95 : {q(sw_wpaste)}")
        print(f"   stroke width px rendered p5-95: {q(sw_wrender)}")
        print(f"   wire px share/frame    : mean {np.mean(wire_share):.4f}")
        pc = meta.get("p_class_given_photographic")
        if pc:
            print("   P(class | photographic), by pasted area: " +
                  "  ".join(f"{k}={v:.3f}" for k, v in pc.items()))
            ok = all(0.20 <= v <= 0.45 for v in pc.values())
            print(f"   balance                : {'OK' if ok else 'WARNING - unbalanced'} "
                  f"(target ~0.333 each; P44 conn=1.000, P44.2 wire=0.000)")
    else:
        print("5. WIRE pastes (P44.3)   : none (no --wire-cutouts in this build)")


if __name__ == "__main__":
    main()
