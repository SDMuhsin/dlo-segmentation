#!/usr/bin/env python
"""A4: online pasting of REAL EgoHands hand cutouts onto SYNTHETIC training frames, labelled bg.

EgoHands is LICENCE-PENDING (informal author grant) -> any model trained with this aug is R&D only,
NOT shippable (see results/qn_backbone_3way/r2pl/egohos/EGOHANDS.md).

Bank (build_bank): per-hand RGBA cutouts from data/egohands_raw/r2pl_640/{rgb,hand} (un-eroded hand
masks), one per connected component with area >= min_area px. Per cutout we store
  * rgb crop (RGB uint8), mask crop (uint8 0/1)
  * palm centre (argmax of the distance transform) and fingertip (contour point farthest from the
    palm centre, restricted to the side opposite the frame-border cut when there is one)
  * cut side: the image border the component is truncated by (>= 10 px of mask on that border), as
    an outward unit vector in crop coords, or None (wrist-cut only, the polygons stop at the wrist).

Paster (HandPaster.__call__): on a uint8 RGB (3,H,W) image + int64 (H,W) label {0,1,2}:
  with prob p the frame gets 1-2 hands. Per hand:
    * mild colour jitter on the hand only (brightness/contrast/saturation/hue), optional mild blur,
      h-flip p .5, scale so the hand's bbox max side ~ logU(scale_lo, scale_hi) * H,
    * placement: with prob near_p "near connector": a fingertip is put 10-40 px outside a random
      synthetic connector blob (from the GT) with the fingers pointing at it (+-30 deg); else
      wire-independent random. Border-cut cutouts in random mode are anchored to a frame edge
      (cut side pointing out of frame, +-20 deg) so the cut never floats mid-frame.
    * alpha = warped mask, feathered with a Gaussian (sigma ~ U(.6, 1.2) => ~1-2 px edge),
    * any placement that would occlude > max_occ of ANY connector blob's pixels is rejected
      (retries, then skip),
    * composite; label = 0 (bg) wherever alpha > .5 (occluded wire/connector GT becomes bg: hidden).
  Uses its OWN numpy Generator -> never touches python `random` / torch RNG (synthetic stream of
  the trainer is unchanged; only the pixels of pasted frames differ).
"""
import glob
import math
import os
import pickle

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(PROJECT_ROOT, "data", "egohands_raw", "r2pl_640")


# ─────────────────────────── BANK ───────────────────────────

def _cutouts_from_frame(rgb_bgr, mask, min_area, margin=6):
    H, W = mask.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_area:
            continue
        comp = (lab == i)
        # border truncation -> cut side (outward normal in image coords, (dx, dy))
        sides = {(0, -1): comp[0].sum(), (0, 1): comp[-1].sum(),
                 (-1, 0): comp[:, 0].sum(), (1, 0): comp[:, -1].sum()}
        cut = None
        tot = sum(sides.values())
        if tot >= 10:
            v = np.zeros(2)
            for d, c in sides.items():
                v += np.array(d, float) * c
            if np.linalg.norm(v) > 0:
                cut = v / np.linalg.norm(v)
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(W, x + w + margin), min(H, y + h + margin)
        m = comp[y0:y1, x0:x1].astype(np.uint8)
        crop = np.ascontiguousarray(rgb_bgr[y0:y1, x0:x1, ::-1])      # -> RGB
        # palm centre = deepest interior point; fingertip = farthest contour pt from it
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        py, px = np.unravel_index(int(dt.argmax()), dt.shape)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = np.concatenate([c.reshape(-1, 2) for c in cnts], 0).astype(np.float32)
        # drop contour points lying on the crop border that coincides with the image border (cut)
        on_img_border = ((pts[:, 0] + x0 <= 0) | (pts[:, 0] + x0 >= W - 1) |
                         (pts[:, 1] + y0 <= 0) | (pts[:, 1] + y0 >= H - 1))
        cand = pts[~on_img_border] if (~on_img_border).sum() > 10 else pts
        d = cand - np.array([px, py], np.float32)
        dist = np.hypot(d[:, 0], d[:, 1])
        if cut is not None:   # fingers point away from the cut
            away = (d @ cut.astype(np.float32)) < 0
            if away.sum() > 5:
                dist = np.where(away, dist, -1)
        tip = cand[int(dist.argmax())]
        cutpt = None
        if cut is not None:   # mean of the mask pixels lying on the image border (crop coords)
            yy, xx = np.nonzero(m)
            ob = ((xx + x0 == 0) | (xx + x0 == W - 1) | (yy + y0 == 0) | (yy + y0 == H - 1))
            cutpt = np.array([xx[ob].mean(), yy[ob].mean()], np.float32)
        out.append({"rgb": crop, "mask": m, "palm": np.array([px, py], np.float32),
                    "tip": tip.astype(np.float32), "cut": cut, "cutpt": cutpt, "area": int(a),
                    "maxside": float(max(w, h))})
    return out


def build_bank(src_dir=DEFAULT_DIR, min_area=2000, cache=None, verbose=True):
    if cache and os.path.isfile(cache):
        with open(cache, "rb") as f:
            bank = pickle.load(f)
        if verbose:
            print(f"  hand bank: {len(bank)} cutouts from cache {cache}")
        return bank
    rgbs = sorted(glob.glob(os.path.join(src_dir, "rgb", "*.png")))
    bank = []
    for rp in rgbs:
        mp = os.path.join(src_dir, "hand", os.path.basename(rp))
        im, m = cv2.imread(rp, cv2.IMREAD_COLOR), cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if im is None or m is None:
            raise RuntimeError(f"unreadable hand pair {rp}")
        for c in _cutouts_from_frame(im, m > 127, min_area):
            c["src"] = os.path.basename(rp)
            bank.append(c)
    if not bank:
        raise SystemExit(f"hand bank empty: {src_dir}")
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(bank, f, protocol=4)
    if verbose:
        nb = sum(c["cut"] is not None for c in bank)
        print(f"  hand bank: {len(bank)} cutouts (area>={min_area}) from {len(rgbs)} frames, "
              f"{nb} border-cut")
    return bank


# ─────────────────────────── PASTER ───────────────────────────

def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], np.float64)


def _ang(v):
    return math.atan2(v[1], v[0])


class HandPaster:
    def __init__(self, bank, p=0.4, near_p=0.5, n_choices=(1, 2), scale=(0.15, 0.6),
                 near_dist=(10, 40), max_occ=0.30, feather=(0.6, 1.2), cj=(0.15, 0.15, 0.15, 0.02),
                 blur_p=0.25, min_conn_px=20, tries=8, seed=0):
        self.bank = bank
        self.nonborder = [i for i, c in enumerate(bank) if c["cut"] is None]
        self.p, self.near_p, self.n_choices = p, near_p, n_choices
        self.scale, self.near_dist, self.max_occ = scale, near_dist, max_occ
        self.feather, self.cj, self.blur_p = feather, cj, blur_p
        self.min_conn_px, self.tries = min_conn_px, tries
        self.rng = np.random.default_rng(seed)
        self.stats = {"frames": 0, "pasted_frames": 0, "hands": 0, "near_req": 0, "near_ok": 0,
                      "skipped": 0, "px_hand": 0, "px_wire_occ": 0, "px_conn_occ": 0,
                      "conn_px": 0, "no_conn_frames": 0}

    # -- appearance --
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
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        if self.blur_p > 0 and r.random() < self.blur_p:
            k = int(r.choice((3, 5)))
            x = cv2.GaussianBlur(x, (k, k), 0)
        return x

    # -- geometry --
    def _affine(self, c, flip, k, theta, anchor_src, anchor_dst):
        """crop coords -> frame coords: flip, scale k, rotate theta about anchor_src -> anchor_dst."""
        w = c["mask"].shape[1]
        F = np.array([[-1.0, 0, w - 1], [0, 1, 0]]) if flip else np.array([[1.0, 0, 0], [0, 1, 0]])
        a = F[:, :2] @ anchor_src + F[:, 2]
        R = _rot(theta) * k
        t = np.asarray(anchor_dst, float) - R @ a
        M = np.zeros((2, 3))
        M[:, :2] = R @ F[:, :2]
        M[:, 2] = R @ F[:, 2] + t
        return M

    def _fvec(self, v, flip):
        return np.array([-v[0], v[1]]) if flip else np.asarray(v, float)

    def _place_random(self, c, flip, k, H, W):
        r = self.rng
        hs = c["maxside"] * k
        if c["cut"] is not None:
            # anchor the cut to a frame edge: cut direction -> outward normal of edge e (+-20 deg)
            normals = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            e = normals[int(r.integers(4))]
            cut = self._fvec(c["cut"], flip)
            theta = _ang(e) - _ang(cut) + math.radians(r.uniform(-20, 20))
            # the cut-edge midpoint goes OUTSIDE the frame, U(0.03, 0.2)*hand size past edge e
            out = r.uniform(0.03, 0.2) * hs
            if e[0] == 0:
                x = r.uniform(0.1 * W, 0.9 * W)
                y = -out if e[1] < 0 else H - 1 + out
            else:
                y = r.uniform(0.1 * H, 0.9 * H)
                x = -out if e[0] < 0 else W - 1 + out
            return self._affine(c, flip, k, theta, c["cutpt"], (x, y))
        theta = r.uniform(0, 2 * math.pi)
        x, y = r.uniform(0.05 * W, 0.95 * W), r.uniform(0.05 * H, 0.95 * H)
        return self._affine(c, flip, k, theta, c["palm"], (x, y))

    def _place_near(self, c, flip, k, blob_mask, H, W):
        r = self.rng
        ys, xs = np.nonzero(blob_mask)
        cy, cx = ys.mean(), xs.mean()
        phi = r.uniform(0, 2 * math.pi)
        u = np.array([math.cos(phi), math.sin(phi)])
        # farthest blob pixel along u (boundary in that direction), then step d outward
        proj = (xs - cx) * u[0] + (ys - cy) * u[1]
        edge = np.array([cx, cy]) + u * proj.max()
        tgt = edge + u * r.uniform(*self.near_dist)
        tgt = np.clip(tgt, 0, [W - 1, H - 1])
        # fingers (palm -> tip) point back at the connector (-u) +-30 deg
        ptip = self._fvec(c["tip"] - c["palm"], flip)
        theta = _ang(-u) - _ang(ptip) + math.radians(r.uniform(-30, 30))
        return self._affine(c, flip, k, theta, c["tip"], tgt)

    def _warp(self, c, M, hand_rgb, H, W, sigma):
        a = cv2.warpAffine(c["mask"].astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = cv2.erode(a, np.ones((3, 3), np.uint8))      # 1 px in: suppress EgoHands-bg halo
        a = cv2.GaussianBlur(a, (0, 0), sigma)
        im = cv2.warpAffine(hand_rgb, M, (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
        return a, im

    def paste_one(self, img, lbl, conn_blobs, near):
        """img (H,W,3) uint8 RGB, lbl (H,W) int; modified in place. Returns True if pasted."""
        r = self.rng
        H, W = lbl.shape
        st = self.stats
        if near:
            st["near_req"] += 1
        pool = self.nonborder if (near and conn_blobs) else None
        for _ in range(self.tries):
            ci = int(r.choice(pool)) if pool else int(r.integers(len(self.bank)))
            c = self.bank[ci]
            flip = bool(r.random() < 0.5)
            k = float(np.exp(r.uniform(math.log(self.scale[0]), math.log(self.scale[1])))
                      * H / c["maxside"])
            k = float(np.clip(k, 0.35, 2.0))
            if near and conn_blobs:
                bm = conn_blobs[int(r.integers(len(conn_blobs)))]
                M = self._place_near(c, flip, k, bm, H, W)
            else:
                M = self._place_random(c, flip, k, H, W)
            sigma = r.uniform(*self.feather)
            hand = self._jitter(c["rgb"])
            a, him = self._warp(c, M, hand, H, W, sigma)
            hm = a > 0.5
            if hm.sum() < 300:
                continue
            if any((hm & b).sum() > self.max_occ * b.sum() for b in conn_blobs):
                continue
            st["px_wire_occ"] += int((hm & (lbl == 1)).sum())
            st["px_conn_occ"] += int((hm & (lbl == 2)).sum())
            st["px_hand"] += int(hm.sum())
            a3 = a[..., None]
            img[:] = np.clip(a3 * him.astype(np.float32) + (1 - a3) * img.astype(np.float32)
                             + 0.5, 0, 255).astype(np.uint8)
            lbl[hm] = 0
            st["hands"] += 1
            if near and conn_blobs:
                st["near_ok"] += 1
            return True
        st["skipped"] += 1
        return False

    def __call__(self, img, lbl):
        """img (H,W,3) uint8 RGB, lbl (H,W) int (0/1/2, may hold -1 ignore). In place.
        Returns number of hands pasted."""
        r = self.rng
        st = self.stats
        st["frames"] += 1
        if r.random() >= self.p:
            return 0
        cm = (lbl == 2).astype(np.uint8)
        n, lab, sts, _ = cv2.connectedComponentsWithStats(cm, connectivity=8)
        blobs = [lab == i for i in range(1, n) if sts[i, 4] >= self.min_conn_px]
        st["conn_px"] += int(cm.sum())
        if not blobs:
            st["no_conn_frames"] += 1
        nh = int(r.choice(self.n_choices))
        got = 0
        for _ in range(nh):
            got += int(self.paste_one(img, lbl, blobs, near=bool(r.random() < self.near_p)))
        if got:
            st["pasted_frames"] += 1
        return got

    def apply_batch(self, rgb_t, lbl_t):
        """torch uint8 (B,3,H,W) RGB + int64 (B,H,W) CPU tensors, modified in place."""
        n = 0
        for i in range(rgb_t.shape[0]):
            img = np.ascontiguousarray(rgb_t[i].numpy().transpose(1, 2, 0))
            lbl = lbl_t[i].numpy()          # shares memory
            k = self(img, lbl)
            if k:
                rgb_t[i].copy_(__import__("torch").from_numpy(img.transpose(2, 0, 1).copy()))
            n += k
        return n

    def pop_stats(self):
        s = dict(self.stats)
        for k in self.stats:
            self.stats[k] = 0
        return s
