#!/usr/bin/env python
"""A5: CONTRASTIVE online cutout pasting (licence-clean: Wikimedia Commons CC/PD cutouts only).

Positives = curated real connector cutouts (data/connector_cutouts, label 2); negatives = real
clutter/tool/household cutouts (data/neg_cutouts_p39 + p44, label 0). BOTH classes go through the
IDENTICAL pipeline, so nothing about the paste artefact (edge, scale, placement, jitter, blur,
matte style) carries class information - only object appearance does:

  * bank: every cutout's alpha is RE-BINARISED (a > 127) and cropped to its bbox, so the different
    matting tools (SAM for connectors, threshold/grabCut for P39) leave no edge-style signature;
    crops are pre-shrunk (INTER_AREA) to <= max_side px.
  * per pasted frame: n ~ U{n_lo..n_hi} objects, each class 2 / 0 with prob pos_frac (.5).
  * appearance (both classes): brightness/contrast/saturation/hue jitter, blur p, h-flip .5.
  * scale (both classes): object bbox max side ~ logU(scale_lo, scale_hi) * H.
  * placement (both classes): with prob wire_end_p at a GT wire END (skeleton endpoint >= 8 px
    from the frame border), principal axis aligned with the outgoing wire tangent, the object
    overlapping the wire end by U(.1,.3) of its length (a plug terminating its cable / an object
    lying on the cable end); otherwise wire-independent uniform position + rotation.
  * alpha: warped binary mask, eroded 1 px, Gaussian-feathered sigma ~ U(feather) (~1-2 px edge).
  * reject a placement that occludes > max_occ of ANY existing GT connector blob (retries, skip).
  * label = object label where alpha > .5 (occluded wire / ignore becomes that label).

Own numpy Generator per paster -> never touches python `random` / torch RNG (trainer streams
unchanged; only the pixels/labels of pasted frames differ).

A6 (only when a WIRE bank is given, `wire_bank=`; otherwise every code path and RNG draw is
identical to A5/A5b): real cable STRAND cutouts (MovingCables train clips, GT masks) pasted as
label 1 through the same _render (same jitter / feather / erode / flip), on the same pasted
frames, BEFORE the connector/negative objects (which then may land on strand ends). Own RNG
(seed+7919) so the connector/negative draws stay the A5b sequence.
  * scale by LENGTH: bbox max side ~ logU(wire_len) * H, clamped so the apparent stroke width
    lies in wire_stroke px.
  * cut ends (mask px on the window-crop border) are hidden by EDGE-EXIT placement: of
    wire_tries candidate poses the one with the most cut-end points outside the frame wins
    (>= wire_min_vis of the strand must stay visible).
  * strands (free and crossing) must leave <= wire_max_cut cut-end clusters visible and no
    oblique (pointed) cut visible; each visible cut end is,
    with prob wire_cover, hidden under a connector/negative object (A5 wire-end placement).
  * bank cleaning: corner wedges / short stubs (skeleton < 3 strokes or median width < .6
    stroke) are dropped.
  * with prob wire_cross a CROSSING group of 2-3 strands: the first anchored on existing GT wire
    (if any, p .5) else free, each next strand anchored on a pixel of the previous one; the
    crossing region is wire.
"""
import json
import math
import os

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────── BANK ───────────────────────────

def load_bank(spec_json, max_side=480, verbose=True):
    """spec_json: list of {"file": path, "label": 0|2, ...}. Returns list of cutout dicts."""
    with open(spec_json) as f:
        spec = json.load(f)
    bank = []
    for e in spec:
        p = e["file"] if os.path.isabs(e["file"]) else os.path.join(PROJECT_ROOT, e["file"])
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3 or im.shape[2] != 4:
            raise RuntimeError(f"bad RGBA cutout {p}")
        m = (im[..., 3] > 127).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if len(ys) < 50:
            raise RuntimeError(f"empty cutout {p}")
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rgb = np.ascontiguousarray(im[y0:y1, x0:x1, 2::-1])          # BGR -> RGB
        m = m[y0:y1, x0:x1]
        s = max_side / max(m.shape)
        if s < 1:
            nw, nh = max(1, int(round(m.shape[1] * s))), max(1, int(round(m.shape[0] * s)))
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            m = (cv2.resize(m.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) > .5
                 ).astype(np.uint8)
        # principal axis (unit vector, crop xy coords) of the mask
        yy, xx = np.nonzero(m)
        c = np.array([xx.mean(), yy.mean()])
        cov = np.cov(np.stack([xx - c[0], yy - c[1]]))
        w, v = np.linalg.eigh(cov)
        ax = v[:, int(np.argmax(w))]
        proj = (xx - c[0]) * ax[0] + (yy - c[1]) * ax[1]
        bank.append({"rgb": rgb, "mask": m, "label": int(e["label"]), "id": e.get("id", p),
                     "center": c, "axis": ax, "axlen": float(proj.max() - proj.min()),
                     "weight": float(e.get("weight", 1.0))})   # A7: only used when weighted=True
    if verbose:
        n2 = sum(b["label"] == 2 for b in bank)
        print(f"  contrastive paste bank: {len(bank)} cutouts ({n2} connector / "
              f"{len(bank) - n2} negative) from {spec_json}")
    return bank


def load_wire_bank(spec_json, max_side=480, verbose=True, min_comp_frac=0.15):
    """A6 strand bank: list of {"file", "label": 1, "stroke_px"}. Alpha re-binarised (a > 127),
    small fragments / corner slivers dropped (component area < max(2*stroke^2, min_comp_frac *
    largest)). cut-end points = mask px on the ORIGINAL window border in runs >= .4 * stroke."""
    with open(spec_json) as f:
        spec = json.load(f)
    bank = []
    for e in spec:
        p = e["file"] if os.path.isabs(e["file"]) else os.path.join(PROJECT_ROOT, e["file"])
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3 or im.shape[2] != 4:
            raise RuntimeError(f"bad RGBA cutout {p}")
        m = (im[..., 3] > 127).astype(np.uint8)
        stroke = float(e.get("stroke_px", 20.0))
        n, lab, sts, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n < 2:
            continue
        big = sts[1:, 4].max()
        from skimage.morphology import skeletonize
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        keep = []
        for i in range(1, n):
            if sts[i, 4] < max(2 * stroke ** 2, min_comp_frac * big):
                continue
            sk = skeletonize(lab == i)
            # drop corner wedges / short stubs: skeleton length >= 3 strokes, median width >= .6 stroke
            if sk.sum() < 3 * stroke or 2 * np.median(dt[sk]) < 0.6 * stroke:
                continue
            keep.append(i)
        if not keep:
            continue
        m = np.isin(lab, keep).astype(np.uint8)
        # cut-end mask: border pixels belonging to runs >= .4*stroke along that border
        H0, W0 = m.shape
        ce = np.zeros_like(m)
        runs = []
        minrun = max(3, int(0.4 * stroke))
        for side in range(4):
            line = (m[0, :], m[-1, :], m[:, 0], m[:, -1])[side]
            v = np.concatenate([[0], line, [0]]).astype(np.int8)
            d = np.diff(v)
            for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
                if b - a >= minrun:
                    mid = (a + b - 1) / 2
                    runs.append(((mid, 0.0, 0.0, -1.0), (mid, H0 - 1.0, 0.0, 1.0),
                                 (0.0, mid, -1.0, 0.0), (W0 - 1.0, mid, 1.0, 0.0))[side]
                                + (float(b - a) / stroke,))
                    if side == 0: ce[0, a:b] = 1
                    elif side == 1: ce[-1, a:b] = 1
                    elif side == 2: ce[a:b, 0] = 1
                    else: ce[a:b, -1] = 1
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rgb = np.ascontiguousarray(im[y0:y1, x0:x1, 2::-1])
        m, ce = m[y0:y1, x0:x1], ce[y0:y1, x0:x1]
        runs = np.array([(x - x0, y - y0, ux, uy, rl) for x, y, ux, uy, rl in runs], float).reshape(-1, 5)
        sc = max_side / max(m.shape)
        if sc < 1:
            nw, nh = max(1, int(round(m.shape[1] * sc))), max(1, int(round(m.shape[0] * sc)))
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            m = (cv2.resize(m.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) > .5).astype(np.uint8)
            ce = cv2.resize(ce.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) > 0
            stroke *= sc
            runs[:, :2] *= sc
        yy, xx = np.nonzero(m)
        rs = np.random.default_rng(len(bank))
        sel = rs.choice(len(yy), size=min(400, len(yy)), replace=False)
        cy, cx = np.nonzero(ce)
        bank.append({"rgb": rgb, "mask": m, "label": 1, "id": e.get("id", p),
                     "center": np.array([(m.shape[1] - 1) / 2, (m.shape[0] - 1) / 2]),
                     "stroke": stroke, "pts": np.stack([xx[sel], yy[sel]], 1).astype(np.float64),
                     "cut": np.stack([cx, cy], 1).astype(np.float64), "runs": runs})
    if verbose:
        nc = np.array([len(b["cut"]) > 0 for b in bank])
        print(f"  A6 wire strand bank: {len(bank)} strands ({len(spec) - len(bank)} dropped: no strand-like "
              f"component) from {spec_json} "
              f"(stroke px p50 {np.median([b['stroke'] for b in bank]):.1f}; {nc.mean():.2f} with cut ends)")
    return bank


# ─────────────────────────── helpers ───────────────────────────

def wire_endpoints(lbl, border=8, min_px=60):
    """Skeleton endpoints of the GT wire mask -> list of (xy endpoint, unit outward tangent)."""
    from skimage.morphology import skeletonize
    w = (lbl == 1)
    if w.sum() < min_px:
        return []
    sk = skeletonize(w)
    if sk.sum() < 10:
        return []
    k = np.ones((3, 3), np.uint8)
    nb = cv2.filter2D(sk.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    ey, ex = np.nonzero(sk & (nb == 2))            # itself + exactly one neighbour
    H, W = lbl.shape
    out = []
    sy, sx = np.nonzero(sk)
    for y, x in zip(ey, ex):
        if x < border or y < border or x >= W - border or y >= H - border:
            continue
        d2 = (sx - x) ** 2 + (sy - y) ** 2
        near = (d2 > 0) & (d2 <= 15 ** 2)
        if near.sum() < 5:
            continue
        u = np.array([x - sx[near].mean(), y - sy[near].mean()], float)
        n = np.linalg.norm(u)
        if n < 1e-3:
            continue
        out.append((np.array([x, y], float), u / n))
    return out


# ─────────────────────────── PASTER ───────────────────────────

class ContrastivePaster:
    def __init__(self, bank, p=0.5, n_range=(1, 3), pos_frac=0.45, scale=(0.05, 0.40),
                 wire_end_p=0.2, max_occ=0.30, feather=(0.6, 1.2), cj=(0.15, 0.15, 0.15, 0.02),
                 blur_p=0.25, min_conn_px=20, tries=8, seed=0,
                 wire_bank=None, wire_n=(1, 2), wire_len=(0.25, 0.9), wire_stroke=(6.0, 32.0),
                 wire_cross=0.3, wire_tries=12, wire_min_vis=0.4, wire_max_cut=1, wire_cover=0.5,
                 weighted=False):
        self.pos = [b for b in bank if b["label"] == 2]
        self.neg = [b for b in bank if b["label"] == 0]
        assert self.pos and self.neg, "bank needs both connector (2) and negative (0) cutouts"
        self.p, self.n_range, self.pos_frac = p, n_range, pos_frac
        self.scale, self.wire_end_p, self.max_occ = scale, wire_end_p, max_occ
        self.feather, self.cj, self.blur_p = feather, cj, blur_p
        self.min_conn_px, self.tries = min_conn_px, tries
        self.rng = np.random.default_rng(seed)
        # A7: per-entry sampling weights (bank "weight" field). Off -> uniform r.integers draw (unchanged).
        self.wprob = None
        if weighted:
            self.wprob = {}
            for lab, pool in ((2, self.pos), (0, self.neg)):
                w = np.array([b["weight"] for b in pool], np.float64)
                self.wprob[lab] = w / w.sum()
        self.wire = list(wire_bank) if wire_bank else None
        if self.wire is not None:
            self.wire_n, self.wire_len, self.wire_stroke = wire_n, wire_len, wire_stroke
            self.wire_cross, self.wire_tries, self.wire_min_vis = wire_cross, wire_tries, wire_min_vis
            self.wire_max_cut, self.wire_cover = wire_max_cut, wire_cover
            self.wrng = np.random.default_rng(seed + 7919)
        self._zero_stats()

    def _zero_stats(self):
        self.stats = {"frames": 0, "pasted_frames": 0, "obj": 0, "obj_pos": 0, "obj_neg": 0,
                      "end_req": 0, "end_ok": 0, "end_ok_pos": 0, "end_ok_neg": 0, "skipped": 0,
                      "px_pos": 0, "px_neg": 0, "px_wire_occ": 0, "px_conn_occ": 0,
                      "px_ign_occ": 0}
        if getattr(self, "wire", None) is not None:
            self.stats.update({"obj_wire": 0, "px_wire": 0, "wire_groups": 0, "wire_cross": 0,
                               "wire_skip": 0, "cut_tot": 0, "cut_vis": 0, "wire_conn_occ": 0,
                               "ends_tot": 0, "ends_vis": 0, "cover_req": 0, "cover_ok": 0,
                               "cover_px_pos": 0, "cover_px_neg": 0})

    # -- appearance (identical for both classes) --
    def _jitter(self, rgb):
        r = self.rng
        x = rgb.astype(np.float32)
        b, c, s, h = self.cj
        x = x * r.uniform(1 - b, 1 + b)
        m = x.reshape(-1, 3).mean(0)
        x = (x - m) * r.uniform(1 - c, 1 + c) + m
        x = np.clip(x, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * r.uniform(1 - s, 1 + s), 0, 255)
        hsv[..., 0] = (hsv[..., 0] + r.uniform(-h, h) * 180.0) % 180.0
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _render(self, c, k, flip, theta, center_src_scaled, dst, H, W, sigma):
        """scale crop by k (INTER_AREA), flip, rotate theta about center_src_scaled -> dst."""
        h0, w0 = c["mask"].shape
        nw, nh = max(2, int(round(w0 * k))), max(2, int(round(h0 * k)))
        rgb = cv2.resize(c["rgb"], (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        m = cv2.resize(c["mask"].astype(np.float32), (nw, nh),
                       interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        if flip:
            rgb, m = rgb[:, ::-1], m[:, ::-1]
        rgb = self._jitter(np.ascontiguousarray(rgb))
        m = np.ascontiguousarray(m)
        cs, sn = math.cos(theta), math.sin(theta)
        R = np.array([[cs, -sn], [sn, cs]])
        M = np.zeros((2, 3))
        M[:, :2] = R
        M[:, 2] = np.asarray(dst, float) - R @ np.asarray(center_src_scaled, float)
        # work in the destination ROI only (bbox of the warped crop + feather margin)
        cor = np.array([[0, 0], [nw - 1, 0], [0, nh - 1], [nw - 1, nh - 1]], float) @ M[:, :2].T + M[:, 2]
        mg = int(math.ceil(4 * sigma)) + 3
        x0, y0 = max(0, int(cor[:, 0].min()) - mg), max(0, int(cor[:, 1].min()) - mg)
        x1, y1 = min(W, int(math.ceil(cor[:, 0].max())) + mg + 1), min(H, int(math.ceil(cor[:, 1].max())) + mg + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        Mr = M.copy()
        Mr[:, 2] -= (x0, y0)
        rw, rh = x1 - x0, y1 - y0
        a = cv2.warpAffine(m, Mr, (rw, rh), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = (a > 0.5).astype(np.float32)
        a = cv2.erode(a, np.ones((3, 3), np.uint8))
        a = cv2.GaussianBlur(a, (0, 0), sigma)
        im = cv2.warpAffine(rgb, Mr, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return a, im, (slice(y0, y1), slice(x0, x1))

    def paste_one(self, img, lbl, conn_blobs, ends, label, at_end):
        r = self.rng
        H, W = lbl.shape
        st = self.stats
        pool = self.pos if label == 2 else self.neg
        if at_end:
            st["end_req"] += 1
        for _ in range(self.tries):
            if self.wprob is None:
                c = pool[int(r.integers(len(pool)))]
            else:
                c = pool[int(r.choice(len(pool), p=self.wprob[label]))]
            k = float(np.exp(r.uniform(math.log(self.scale[0]), math.log(self.scale[1])))
                      * H / max(c["mask"].shape))
            flip = bool(r.random() < 0.5)
            cx, cy = c["center"] * k
            ax = c["axis"].copy()
            if flip:
                cx = c["mask"].shape[1] * k - 1 - cx
                ax[0] = -ax[0]
            if at_end and ends:
                pt, u = ends[int(r.integers(len(ends)))]
                L = c["axlen"] * k                                 # length along principal axis
                sgn = 1.0 if r.random() < 0.5 else -1.0          # which end faces the wire
                theta = math.atan2(u[1], u[0]) - math.atan2(sgn * ax[1], sgn * ax[0])
                theta += math.radians(r.uniform(-15, 15))
                dst = pt + u * (L / 2 - r.uniform(0.1, 0.3) * L)
            else:
                theta = r.uniform(0, 2 * math.pi)
                dst = (r.uniform(0.05 * W, 0.95 * W), r.uniform(0.05 * H, 0.95 * H))
            sigma = r.uniform(*self.feather)
            out = self._render(c, k, flip, theta, (cx, cy), dst, H, W, sigma)
            if out is None:
                continue
            a, oim, sl = out
            om = a > 0.5
            if om.sum() < 80:
                continue
            if any((om & b[sl]).sum() > self.max_occ * bs for b, bs in conn_blobs):
                continue
            lr, ir = lbl[sl], img[sl]
            st["px_wire_occ"] += int((om & (lr == 1)).sum())
            st["px_conn_occ"] += int((om & (lr == 2)).sum())
            st["px_ign_occ"] += int((om & (lr < 0)).sum())
            st["px_pos" if label == 2 else "px_neg"] += int(om.sum())
            a3 = a[..., None]
            ir[:] = np.clip(a3 * oim.astype(np.float32) + (1 - a3) * ir.astype(np.float32)
                            + 0.5, 0, 255).astype(np.uint8)
            lr[om] = label
            st["obj"] += 1
            st["obj_pos" if label == 2 else "obj_neg"] += 1
            if at_end and ends:
                st["end_ok"] += 1
                st["end_ok_pos" if label == 2 else "end_ok_neg"] += 1
            return True
        st["skipped"] += 1
        return False

    # ---------------- A6 wire strands ----------------
    def _strand_pose(self, c, H, W, anchor):
        """Sample one candidate pose -> (k, flip, theta, dst, score) or None."""
        r = self.wrng
        lf = float(np.exp(r.uniform(math.log(self.wire_len[0]), math.log(self.wire_len[1]))))
        k = lf * H / max(c["mask"].shape)
        k = min(max(k, self.wire_stroke[0] / c["stroke"]), self.wire_stroke[1] / c["stroke"])
        flip = bool(r.random() < 0.5)
        theta = r.uniform(0, 2 * math.pi)
        h0, w0 = c["mask"].shape
        nw = max(2, int(round(w0 * k)))
        cen = c["center"] * k
        if flip:
            cen = np.array([nw - 1 - cen[0], cen[1]])
        cs, sn = math.cos(theta), math.sin(theta)
        R = np.array([[cs, -sn], [sn, cs]])

        def tf(p):
            q = p * k
            if flip:
                q = np.stack([nw - 1 - q[:, 0], q[:, 1]], 1)
            return (q - cen) @ R.T
        rel = tf(c["pts"])
        if anchor is None:
            dst = np.array([r.uniform(-0.15 * W, 1.15 * W), r.uniform(-0.15 * H, 1.15 * H)])
        else:
            dst = np.asarray(anchor, float) - rel[int(r.integers(len(rel)))]
        pt = rel + dst
        inside = lambda q: (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
        vis = float(inside(pt).mean())
        if vis < self.wire_min_vis or vis * (c["mask"].sum() * k * k) < 150:
            return None
        if len(c["cut"]):
            ce = tf(c["cut"]) + dst
            ci = (ce[:, 0] >= 2) & (ce[:, 0] < W - 2) & (ce[:, 1] >= 2) & (ce[:, 1] < H - 2)
            hid = 1.0 - float(ci.mean())
            ncut, nvis = len(ce), int(ci.sum())
        else:
            hid, ncut, nvis = 1.0, 0, 0
        vends, nobl = [], 0
        for x, y, ux, uy, rl in c["runs"]:
            q = tf(np.array([[x, y]]))[0] + dst
            if 2 <= q[0] < W - 2 and 2 <= q[1] < H - 2:
                d = np.array([-ux if flip else ux, uy]) @ R.T
                vends.append((q, d / (np.linalg.norm(d) + 1e-9)))
                nobl += int(rl > 1.6)          # oblique cut (run >> stroke) -> pointed "leaf" tip
        if nobl:
            return None                        # never leave an oblique cut visible
        # rank: fewest visible cut-end clusters, then most hidden cut-end px
        score = -len(vends) + hid + 1e-3 * r.random()
        return k, flip, theta, cen, dst, score, ncut, nvis, len(c["runs"]), vends

    def _paste_strand(self, img, lbl, conn_blobs, anchor):
        r = self.wrng
        H, W = lbl.shape
        st = self.stats
        cands = []
        for _ in range(self.wire_tries):
            c = self.wire[int(r.integers(len(self.wire)))]
            o = self._strand_pose(c, H, W, anchor)
            if o is not None:
                cands.append((o[5], c, o))
        cands.sort(key=lambda t: -t[0])
        cands = [t for t in cands if len(t[2][9]) <= self.wire_max_cut]   # <= max visible cut ends
        for _, c, (k, flip, theta, cen, dst, _s, ncut, nvis, nends, vends) in cands[:4]:
            sigma = r.uniform(*self.feather)
            # _render uses self.rng for jitter -> swap in the wire RNG so A5b draws are untouched
            main = self.rng
            self.rng = r
            try:
                out = self._render(c, k, flip, theta, cen, dst, H, W, sigma)
            finally:
                self.rng = main
            if out is None:
                continue
            a, oim, sl = out
            om = a > 0.5
            if om.sum() < 80:
                continue
            if any((om & b[sl]).sum() > self.max_occ * bs for b, bs in conn_blobs):
                continue
            lr, ir = lbl[sl], img[sl]
            st["wire_conn_occ"] += int((om & (lr == 2)).sum())
            st["px_wire"] += int(om.sum())
            a3 = a[..., None]
            ir[:] = np.clip(a3 * oim.astype(np.float32) + (1 - a3) * ir.astype(np.float32)
                            + 0.5, 0, 255).astype(np.uint8)
            lr[om] = 1
            st["obj_wire"] += 1
            st["cut_tot"] += ncut
            st["cut_vis"] += nvis
            st["ends_tot"] += nends
            st["ends_vis"] += len(vends)
            ys, xs = np.nonzero(om)
            j = int(r.integers(len(ys)))
            self._pend_ends.extend(vends)
            return (xs[j] + sl[1].start, ys[j] + sl[0].start)
        st["wire_skip"] += 1
        return None

    def _paste_wires(self, img, lbl, conn_blobs):
        r = self.wrng
        st = self.stats
        n = int(r.integers(self.wire_n[0], self.wire_n[1] + 1))
        got = 0
        self._pend_ends = []
        for _ in range(n):
            st["wire_groups"] += 1
            if r.random() < self.wire_cross:
                st["wire_cross"] += 1
                anchor = None
                gy, gx = np.nonzero(lbl == 1)
                if len(gy) >= 200 and r.random() < 0.5:
                    j = int(r.integers(len(gy)))
                    anchor = (gx[j], gy[j])
                for _ in range(int(r.integers(2, 4))):
                    a = self._paste_strand(img, lbl, conn_blobs, anchor)
                    if a is not None:
                        got += 1
                        anchor = a
            else:
                got += int(self._paste_strand(img, lbl, conn_blobs, None) is not None)
        # visible cut ends: with prob wire_cover hide each under a connector / negative object placed
        # by the A5 wire-END rule (axis along the outgoing tangent, overlapping the end); the class
        # follows pos_frac. Uses the wire RNG so the A5b connector/negative draws are untouched.
        main = self.rng
        self.rng = r
        try:
            for pt, u in self._pend_ends:
                if r.random() >= self.wire_cover:
                    continue
                st["cover_req"] += 1
                label = 2 if r.random() < self.pos_frac else 0
                b0 = st["px_pos" if label == 2 else "px_neg"]
                ok = self.paste_one(img, lbl, conn_blobs, [(pt, u)], label, True)
                st["end_req"] -= 1          # keep the A5 end/obj counters about A5 objects only
                if ok:
                    st["cover_ok"] += 1
                    st["obj"] -= 1
                    st["obj_pos" if label == 2 else "obj_neg"] -= 1
                    st["end_ok"] -= 1
                    st["end_ok_pos" if label == 2 else "end_ok_neg"] -= 1
                    d = st["px_pos" if label == 2 else "px_neg"] - b0
                    st["cover_px_pos" if label == 2 else "cover_px_neg"] += d
                else:
                    st["skipped"] -= 1
        finally:
            self.rng = main
        return got

    def __call__(self, img, lbl):
        """img (H,W,3) uint8 RGB, lbl (H,W) int (0/1/2, may hold -1 ignore). In place."""
        r = self.rng
        st = self.stats
        st["frames"] += 1
        if r.random() >= self.p:
            return 0
        cm = (lbl == 2).astype(np.uint8)
        n, lab, sts, _ = cv2.connectedComponentsWithStats(cm, connectivity=8)
        blobs = [(lab == i, int(sts[i, 4])) for i in range(1, n) if sts[i, 4] >= self.min_conn_px]
        got = 0
        if self.wire is not None:
            got += self._paste_wires(img, lbl, blobs)
        nobj = int(r.integers(self.n_range[0], self.n_range[1] + 1))
        ends = None
        for _ in range(nobj):
            label = 2 if r.random() < self.pos_frac else 0
            at_end = bool(r.random() < self.wire_end_p)
            if at_end and ends is None:
                ends = wire_endpoints(lbl)
            got += int(self.paste_one(img, lbl, blobs, ends or [], label, at_end))
        if got:
            st["pasted_frames"] += 1
        return got

    def apply_batch(self, rgb_t, lbl_t):
        """torch uint8 (B,3,H,W) RGB + int64 (B,H,W) CPU tensors, modified in place."""
        import torch
        n = 0
        for i in range(rgb_t.shape[0]):
            img = np.ascontiguousarray(rgb_t[i].numpy().transpose(1, 2, 0))
            lbl = lbl_t[i].numpy()          # shares memory
            k = self(img, lbl)
            if k:
                rgb_t[i].copy_(torch.from_numpy(img.transpose(2, 0, 1).copy()))
            n += k
        return n

    def pop_stats(self):
        s = dict(self.stats)
        self._zero_stats()
        return s

    @staticmethod
    def fmt(s):
        return (f"frames={s['pasted_frames']}/{s['frames']} obj={s['obj_pos']}c/{s['obj_neg']}n "
                f"end={s['end_ok_pos']}c/{s['end_ok_neg']}n of {s['end_req']} skip={s['skipped']} "
                f"px={s['px_pos']}c/{s['px_neg']}n occ_w={s['px_wire_occ']} "
                f"occ_c={s['px_conn_occ']} occ_ign={s['px_ign_occ']}"
                + (f" wire={s['obj_wire']}obj/{s['px_wire']}px grp={s['wire_groups']} "
                   f"cross={s['wire_cross']} wskip={s['wire_skip']} cutvis={s['cut_vis']}/{s['cut_tot']} "
                   f"wocc_c={s['wire_conn_occ']} ends={s['ends_vis']}/{s['ends_tot']} "
                   f"cover={s['cover_ok']}/{s['cover_req']} cover_px={s['cover_px_pos']}c/{s['cover_px_neg']}n"
                   if "obj_wire" in s else ""))
