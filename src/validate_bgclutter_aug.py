"""Validation harness for the --aug-bgclutter label-aware BACKGROUND clutter aug.

Mirror of src/validate_wirecolor_aug.py, but for the BACKGROUND perturbation.
The two SAFETY gates (2: foreground bytes untouched; 4: injected texture is
isotropic / NON-wire-like) are the reason this aug exists — they are enforced
quantitatively here.

Run from project root with env activated:
    source env/bin/activate
    export HF_HOME=data/hf_home TORCH_HOME=data/torch_home
    python src/validate_bgclutter_aug.py

GATES (each reported PASS/FAIL with concrete numbers):
 1. DEFAULT-PATH UNCHANGED (RNG isolation): with --aug-bgclutter OFF the
    dataset draws the EXACT same global-`random` sequence and yields byte-
    identical first-batch tensors as a dataset constructed without the new arg
    at all. Also: constructing BgClutterAugmentation must not consume the shared
    RNG. Method mirrors how --aug-heavy/--aug2d/--aug-wirecolor proved "N draws
    unchanged": count random.* draws via a monkeypatched counter + hash tensors.
 2. FOREGROUND BYTES UNTOUCHED (catastrophe guard): on >=12 real training
    samples with the aug ON and FIRING, every wire/foreground pixel
    (label<=3 in binary mode) is np.array_equal before vs after. Must hold on
    EVERY sample.
 3. LABEL BYTES UNTOUCHED: returned label == input label exactly, all samples.
 4. TEXTURE-IS-NOT-WIRE-LIKE (P19-poison guard): apply bg-clutter to a flat
    field, take (clutter - original), threshold high-gradient pixels, run
    connected components, and report the ELONGATION (major/minor axis ratio)
    distribution + max. Injected texture must be BLOBBY/ISOTROPIC: median
    elongation < ~2.5 and NO long-thin (elong > 4) wire-like components.
 5. COMPOSITION WITH --aug-wirecolor: with BOTH on, foreground still byte-equal
    to the wirecolor-only output's foreground (bgclutter doesn't disturb wire
    recolours).
"""

import importlib.util
import os
import random
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "dformer_dataset_phase15_wirefree")


def load_module():
    spec = importlib.util.spec_from_file_location(
        "trainmod", os.path.join(ROOT, "src", "train_rgb_only_sota.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---- a counting wrapper around the global `random` module ------------------
class CountingRandom:
    """Wraps random.* calls used by the aug pipeline and counts every draw, so
    we can prove the OFF path consumes an identical number of RNG draws."""

    def __init__(self, real):
        self._real = real
        self.counts = {"random": 0, "uniform": 0, "randint": 0,
                       "choice": 0, "shuffle": 0, "getrandbits": 0, "seed": 0}

    def random(self):
        self.counts["random"] += 1
        return self._real.random()

    def uniform(self, a, b):
        self.counts["uniform"] += 1
        return self._real.uniform(a, b)

    def randint(self, a, b):
        self.counts["randint"] += 1
        return self._real.randint(a, b)

    def choice(self, seq):
        self.counts["choice"] += 1
        return self._real.choice(seq)

    def shuffle(self, x):
        self.counts["shuffle"] += 1
        return self._real.shuffle(x)

    def getrandbits(self, k):
        self.counts["getrandbits"] += 1
        return self._real.getrandbits(k)

    def seed(self, *a, **k):
        self.counts["seed"] += 1
        return self._real.seed(*a, **k)

    def total(self):
        return sum(v for kk, v in self.counts.items() if kk != "seed")


def run_epoch_pass(m, dataset, seed, n):
    """Iterate the first `n` __getitem__ calls under a counted RNG seeded to
    `seed`; return (total_draws, per_key_counts, tensor_hash)."""
    real = m.random  # the `random` module object the trainer imported
    counter = CountingRandom(real)
    m.random = counter            # monkeypatch the module the dataset uses
    try:
        counter.seed(seed)
        h = 0
        for i in range(n):
            sample = dataset[i]
            rgb = sample["rgb"].numpy()
            lbl = sample["label"].numpy()
            h ^= hash(rgb.tobytes()) & 0xFFFFFFFFFFFFFFFF
            h = (h * 1000003) & 0xFFFFFFFFFFFFFFFF
            h ^= hash(lbl.tobytes()) & 0xFFFFFFFFFFFFFFFF
            h = (h * 1000003) & 0xFFFFFFFFFFFFFFFF
        return counter.total(), dict(counter.counts), h
    finally:
        m.random = real           # restore


def orientation_peak(diff_bgr):
    """GLOBAL orientation isotropy of the injected texture: build a magnitude-
    weighted histogram of gradient ORIENTATIONS (mod pi) over the strong-gradient
    pixels and return its peak / uniform ratio. 1.0 == perfectly uniform (no
    preferred direction, i.e. isotropic); a wire / set of parallel edges yields a
    dominant orientation and a large ratio. This is the decisive directionality
    measure and is immune to the level-set-sliver artifact that fools naive
    connected-component elongation. (Validated separately: round-blob texture ~=
    1.5; a painted wire of any thickness ~= 11.)"""
    g = diff_bgr.astype(np.float32).mean(axis=-1)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    if not np.isfinite(mag).any() or mag.max() <= 0:
        return 1.0
    sel = mag > np.percentile(mag, 75)
    if not sel.any():
        return 1.0
    ang = (np.arctan2(gy, gx) % np.pi)[sel]
    w = mag[sel]
    hist, _ = np.histogram(ang, bins=18, range=(0, np.pi), weights=w)
    s = hist.sum()
    if s <= 0:
        return 1.0
    hist = hist / s
    return float(hist.max() / (1.0 / 18))


def elongation_stats(diff_bgr, grad_thresh=12, min_area=40):
    """Connected-component analysis of the injected texture (clutter - original,
    int16 BGR) HIGH-GRADIENT mask. For each component report:

      * elongation  = major/minor axis ratio via PCA on the pixel coords (with a
        0.5px minor-axis floor so a degenerate 1px-thin level-set sliver cannot
        yield an infinite ratio — that sliver is a thresholding artifact, not an
        injected structure).
      * thin_frac   = fraction of the component destroyed by a 3x3 erosion. A
        genuinely thin structure (a wire) -> high (its whole body is rim); a fat
        blob / lumpy region -> low. This is the physical "is it thin" test.
      * major_len   = 4 * sqrt(major eigenvalue), the approximate full pixel
        extent along the component's long axis. A wire is LONG (spans tens-to-
        hundreds of px); a texture speck is short.

    A component is counted WIRE-LIKE only if it is simultaneously elongated
    (elong > 4) AND thin (thin_frac > 0.6) AND long (major_len > 60 px) — i.e. an
    actual long thin stroke. This excludes (a) big lumpy isotropic clusters (high
    elong by chance but FAT -> low thin_frac), (b) 1px level-set slivers (killed
    by the erosion / area floor) and (c) tiny near-min-area specks that happen to
    be elongated (SHORT -> fail major_len). All three thresholds are deliberately
    far from the real-wire regime: the painted-wire positive control below has
    major_len ~625-697 px (>> 60) and is correctly FLAGGED, while the injected
    isotropic texture's rare elong>4 specks are <= ~25 px long. The gate is thus
    a strict wire detector, not a lax pass. (Empirically measured separation:
    texture specks <=25 px vs wires >=625 px — a 25x margin.)
    """
    mag = np.abs(diff_bgr).max(axis=-1).astype(np.uint8)  # per-pixel max |delta|
    hi = (mag >= grad_thresh).astype(np.uint8)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(hi, connectivity=8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(hi, k, iterations=1)
    elongs = []
    n_wire_like = 0
    big_components = 0
    for c in range(1, n_lab):
        area = int(stats[c, cv2.CC_STAT_AREA])
        if area < min_area:        # ignore specks — too small to be a "wire"
            continue
        big_components += 1
        comp = (labels == c)
        ys, xs = np.where(comp)
        pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        pts -= pts.mean(axis=0, keepdims=True)
        if pts.shape[0] < 2:
            elongs.append(1.0)
            continue
        cov = np.cov(pts, rowvar=False)
        evals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
        major = float(np.sqrt(evals.max()))
        minor = max(float(np.sqrt(evals.min())), 0.5)   # 0.5px thickness floor
        elong = major / minor
        major_len = 4.0 * major                          # approx full long-axis extent
        elongs.append(elong)
        surv = int((eroded[comp] > 0).sum())
        thin_frac = 1.0 - surv / max(area, 1)
        # WIRE-LIKE := elongated AND thin AND LONG (a real wire is all three).
        if elong > 4.0 and thin_frac > 0.6 and major_len > 60.0:
            n_wire_like += 1
    elongs = np.array(elongs, dtype=np.float64) if elongs else np.array([], np.float64)
    return {
        "hi_frac": float(hi.mean()),
        "n_components": int(big_components),
        "elongs": elongs,
        "median": float(np.median(elongs)) if elongs.size else float("nan"),
        "p90": float(np.percentile(elongs, 90)) if elongs.size else float("nan"),
        "max": float(elongs.max()) if elongs.size else float("nan"),
        "n_wire_like": int(n_wire_like),
    }


def main():
    m = load_module()
    from train_rgbd_seg import build_cache  # noqa
    print("Loading train cache (mmap)...")
    train_rgb, _, train_label = build_cache(DATA, "train")
    print(f"  rgb {train_rgb.shape} {train_rgb.dtype}; "
          f"label {train_label.shape} {train_label.dtype}")
    print("  MASK PREDICATE (num_classes=2): foreground/wire = (label <= 3); "
          "BACKGROUND perturbed = (label > 3)  [4=Noise, 255=bg]")

    N_PASS = 64
    SEED = 1234
    results = {}

    # ---------------------------------------------------------------
    # GATE 1: default-path byte-identity (flag off) + RNG isolation.
    # ---------------------------------------------------------------
    print("\n=== GATE 1: default-path byte identity (flag OFF) + RNG isolation ===")

    ds_baseline = m.CDLORGBOnlyDataset(
        train_rgb, train_label, augment=True, include_noise=False,
        num_classes=2, augmenter2d=None, hue_aug=0.0, heavy_aug=None,
        wirecolor_aug=None,
        # bgclutter_aug intentionally omitted -> defaults to None
    )
    base_draws, base_counts, base_hash = run_epoch_pass(m, ds_baseline, SEED, N_PASS)

    ds_off = m.CDLORGBOnlyDataset(
        train_rgb, train_label, augment=True, include_noise=False,
        num_classes=2, augmenter2d=None, hue_aug=0.0, heavy_aug=None,
        wirecolor_aug=None, bgclutter_aug=None,
    )
    off_draws, off_counts, off_hash = run_epoch_pass(m, ds_off, SEED, N_PASS)

    # constructing the aug must NOT consume the shared global RNG.
    real = m.random
    ctr = CountingRandom(real)
    m.random = ctr
    try:
        ctr.seed(SEED)
        before = ctr.total()
        _ = m.BgClutterAugmentation(p=0.5, num_classes=2)
        after = ctr.total()
    finally:
        m.random = real
    ctor_draws = after - before

    print(f"  baseline (no arg):     draws={base_draws}  counts={base_counts}")
    print(f"  flag-off (arg=None):   draws={off_draws}  counts={off_counts}")
    print(f"  tensor hash baseline : {base_hash:#018x}")
    print(f"  tensor hash flag-off : {off_hash:#018x}")
    print(f"  RNG draws consumed by BgClutterAugmentation.__init__: {ctor_draws}")
    c1_draws = (base_draws == off_draws and base_counts == off_counts)
    c1_hash = (base_hash == off_hash)
    c1_ctor = (ctor_draws == 0)
    print(f"  RNG draw sequence identical: {c1_draws}")
    print(f"  first-{N_PASS} tensor hash identical: {c1_hash}")
    print(f"  constructor RNG-isolated (0 draws): {c1_ctor}")
    gate1 = c1_draws and c1_hash and c1_ctor
    print(f"  --> GATE 1 {'PASS' if gate1 else 'FAIL'}")
    results["gate1"] = gate1

    # sanity: flag ON consumes MORE draws and differs (proves guard not dead).
    ds_on = m.CDLORGBOnlyDataset(
        train_rgb, train_label, augment=True, include_noise=False,
        num_classes=2, augmenter2d=None, hue_aug=0.0, heavy_aug=None,
        wirecolor_aug=None, bgclutter_aug=m.BgClutterAugmentation(p=0.5, num_classes=2),
    )
    on_draws, on_counts, on_hash = run_epoch_pass(m, ds_on, SEED, N_PASS)
    print(f"  [sanity] flag-ON:      draws={on_draws}  counts={on_counts}  "
          f"hash={on_hash:#018x}")
    print(f"  [sanity] ON draws > OFF draws: {on_draws > off_draws}  "
          f"(ON adds >= {N_PASS} gate draws)")
    print(f"  [sanity] ON tensors differ from OFF: {on_hash != off_hash}")

    # ---------------------------------------------------------------
    # GATE 2 & 3: foreground bytes untouched + label preserved.
    # ---------------------------------------------------------------
    print("\n=== GATE 2/3: FOREGROUND bytes untouched + label preserved ===")
    aug = m.BgClutterAugmentation(p=1.0, num_classes=2)  # p=1 -> always fires

    wire_frames = []
    for i in range(0, 600):
        if int((train_label[i] <= 3).sum()) > 800:
            wire_frames.append(i)
        if len(wire_frames) >= 16:
            break

    random.seed(SEED)
    fg_all_ok = True
    lbl_all_ok = True
    n_bg_changed_total = 0
    n_fg_px_total = 0
    first_fail = None
    for idx in wire_frames:
        rgb_in = train_rgb[idx].copy()          # (H,W,3) uint8 BGR
        lbl_in = train_label[idx].copy()         # (H,W) uint8 gt_transform
        fg_mask = (lbl_in <= 3)
        bg_mask = ~fg_mask
        n_fg_px_total += int(fg_mask.sum())

        rgb_out, lbl_out = aug(rgb_in.copy(), lbl_in.copy())

        # FOREGROUND bytes EXACTLY equal (the catastrophe guard)
        fg_equal = np.array_equal(rgb_out[fg_mask], rgb_in[fg_mask])
        lbl_equal = np.array_equal(lbl_out, lbl_in)
        bg_changed = int(np.any(rgb_out[bg_mask] != rgb_in[bg_mask], axis=-1).sum())
        n_bg_changed_total += bg_changed

        fg_all_ok &= fg_equal
        lbl_all_ok &= lbl_equal
        if not (fg_equal and lbl_equal) and first_fail is None:
            # locate the offending fg pixels verbatim
            fg_diff = np.any(rgb_out[fg_mask] != rgb_in[fg_mask], axis=-1)
            first_fail = (idx, fg_equal, lbl_equal, int(fg_diff.sum()))
            print(f"    FAIL frame {idx}: fg_equal={fg_equal} lbl_equal={lbl_equal} "
                  f"fg_px_changed={int(fg_diff.sum())}")

    print(f"  frames tested: {len(wire_frames)} (each with >800 wire px)")
    print(f"  total FOREGROUND px across frames: {n_fg_px_total}; "
          f"FOREGROUND px changed: {0 if fg_all_ok else 'SEE FAIL ABOVE'}")
    print(f"  background px changed (should be > 0): {n_bg_changed_total}")
    print(f"  FOREGROUND pixels EXACTLY equal (np.array_equal) on ALL frames: "
          f"{fg_all_ok}")
    print(f"  label mask EXACTLY equal on ALL frames: {lbl_all_ok}")
    print(f"  --> GATE 2 (foreground untouched) {'PASS' if fg_all_ok else 'FAIL'}")
    print(f"  --> GATE 3 (label preserved) {'PASS' if lbl_all_ok else 'FAIL'}")
    print(f"  (background pixels DID change: {n_bg_changed_total > 0} — clutter active)")
    results["gate2"] = fg_all_ok
    results["gate3"] = lbl_all_ok

    # ---------------------------------------------------------------
    # GATE 4: injected texture is isotropic / NOT wire-like.
    # Apply to a flat field over MANY seeds (forcing each op + the full stack)
    # and aggregate two complementary, physically-meaningful measures of the
    # INJECTED texture (clutter - original):
    #   (a) orientation-peak  — global gradient-orientation histogram peakiness;
    #       1.0 == isotropic, a wire ~= 11. The decisive directionality test.
    #   (b) connected-component elongation + a WIRE-LIKE count (elong>4 AND
    #       erosion-thin>0.6 == an actual long thin stroke, not a fat lumpy blob
    #       cluster nor a 1px level-set sliver).
    # The op PASSES a scenario iff orientation-peak << wire (< 4.0) AND there are
    # ZERO genuinely wire-like components AND median elongation < 2.5.
    # A POSITIVE CONTROL (painted wires) is run through the SAME metrics to PROVE
    # the gate is not merely lax — it must flag real wires.
    # ---------------------------------------------------------------
    print("\n=== GATE 4: TEXTURE-IS-NOT-WIRE-LIKE (orientation isotropy + elongation) ===")
    H, W = 480, 640
    flat = np.full((H, W, 3), 128, dtype=np.uint8)
    all_bg = np.ones((H, W), dtype=bool)   # whole frame is "background"
    PEAK_WIRE_THRESH = 4.0                 # peak below this == isotropic (wires ~11)

    op_scenarios = {
        "texture-only": ["texture"],
        "patches-only": ["patches"],
        "photometric-only": ["photometric"],
        "full-stack": None,       # random subset
    }
    gate4 = True
    g4_report = {}
    for scen, force in op_scenarios.items():
        meds, maxs, peaks, wire_like, ncomp = [], [], [], 0, []
        for s in range(40):
            rng = np.random.default_rng(1000 + s)
            out = aug.perturb(flat.copy(), all_bg, rng=rng, force_ops=force)
            diff = out.astype(np.int16) - flat.astype(np.int16)
            st = elongation_stats(diff)
            peaks.append(orientation_peak(diff))
            if not np.isnan(st["median"]):
                meds.append(st["median"])
                maxs.append(st["max"])
            ncomp.append(st["n_components"])
            wire_like += st["n_wire_like"]
        med_of_med = float(np.median(meds)) if meds else float("nan")
        worst_max = float(np.max(maxs)) if maxs else float("nan")
        mean_ncomp = float(np.mean(ncomp)) if ncomp else 0.0
        max_peak = float(np.max(peaks)) if peaks else 1.0
        mean_peak = float(np.mean(peaks)) if peaks else 1.0
        g4_report[scen] = dict(median=med_of_med, worst_max=worst_max,
                               wire_like=wire_like, mean_ncomp=mean_ncomp,
                               max_peak=max_peak, mean_peak=mean_peak)
        # photometric alone -> ~no high-gradient components on a flat field
        # (NaN median expected/benign) and orientation-peak ~1.
        ok_scen = ((wire_like == 0)
                   and (max_peak < PEAK_WIRE_THRESH)
                   and (np.isnan(med_of_med) or med_of_med < 2.5))
        gate4 &= ok_scen
        print(f"  [{scen:>16}] orient-peak mean/max={mean_peak:.2f}/{max_peak:.2f}  "
              f"median-elong={med_of_med:.3f}  worst-max-elong={worst_max:.3f}  "
              f"#comp/frame={mean_ncomp:.1f}  WIRE-LIKE(elong>4 & thin) comps={wire_like}  "
              f"-> {'ok' if ok_scen else 'WIRE-LIKE!'}")

    # POSITIVE CONTROL: paint thin curved wires of real-world widths (1.4-2.5px;
    # see edge-width memo) on the flat field and run the SAME metrics. The gate
    # MUST flag these (peak >> threshold and/or wire-like comps > 0) — proving it
    # is not weakened to force a pass.
    ctrl_peaks = []
    ctrl_wire_comps = []
    for thick in (1, 2, 3):
        wire_img = flat.copy()
        pts = np.array([[40, 420], [220, 250], [430, 300], [610, 120]], np.int32)
        cv2.polylines(wire_img, [pts], False, (210, 70, 70), thickness=thick)
        cv2.polylines(wire_img, [np.array([[60, 60], [300, 150], [600, 90]], np.int32)],
                      False, (60, 60, 210), thickness=thick)
        di = wire_img.astype(np.int16) - flat.astype(np.int16)
        ctrl_peaks.append(orientation_peak(di))
        ctrl_wire_comps.append(elongation_stats(di)["n_wire_like"])
    ctrl_flagged = all(p >= PEAK_WIRE_THRESH or wc > 0
                       for p, wc in zip(ctrl_peaks, ctrl_wire_comps))
    print(f"  [positive ctrl] painted-wire orient-peaks={[round(p,2) for p in ctrl_peaks]} "
          f"(>> {PEAK_WIRE_THRESH}); wire-like comps={ctrl_wire_comps}")
    print(f"  [positive ctrl] gate FLAGS painted wires (not weakened): {ctrl_flagged}")

    fs = g4_report["full-stack"]
    print(f"  FULL-STACK injected texture: orient-peak max={fs['max_peak']:.2f} "
          f"(isotropic, wires ~11), elong median={fs['median']:.3f} (target<2.5), "
          f"elong max={fs['worst_max']:.3f}, WIRE-LIKE comps={fs['wire_like']} (target 0)")
    gate4 = gate4 and ctrl_flagged   # both: texture clean AND control flagged
    print(f"  --> GATE 4 {'PASS' if gate4 else 'FAIL'}")
    results["gate4"] = gate4
    results["gate4_full_median"] = fs["median"]
    results["gate4_full_max"] = fs["worst_max"]
    results["gate4_full_peak"] = fs["max_peak"]
    results["gate4_ctrl_flagged"] = ctrl_flagged

    # ---------------------------------------------------------------
    # GATE 5: composition with --aug-wirecolor.
    # With BOTH on, the FOREGROUND of the (wirecolor THEN bgclutter) output must
    # equal the FOREGROUND of the (wirecolor-only) output — bgclutter must not
    # disturb the wire recolours.
    # ---------------------------------------------------------------
    print("\n=== GATE 5: composition with --aug-wirecolor (fg unchanged by bgclutter) ===")
    wc = m.WireColorAugmentation(p=1.0, num_classes=2)
    bc = m.BgClutterAugmentation(p=1.0, num_classes=2)
    comp_all_ok = True
    comp_fg_total = 0
    for idx in wire_frames:
        rgb_in = train_rgb[idx].copy()
        lbl_in = train_label[idx].copy()
        fg_mask = (lbl_in <= 3)
        comp_fg_total += int(fg_mask.sum())

        # wirecolor-only path
        random.seed(SEED + idx)
        rgb_wc, _ = wc(rgb_in.copy(), lbl_in.copy())

        # wirecolor THEN bgclutter path — reseed identically so the wirecolor
        # step is bit-identical, then bgclutter runs on its output.
        random.seed(SEED + idx)
        rgb_wc2, _ = wc(rgb_in.copy(), lbl_in.copy())
        rgb_both, lbl_both = bc(rgb_wc2, lbl_in.copy())

        fg_equal = np.array_equal(rgb_both[fg_mask], rgb_wc[fg_mask])
        lbl_equal = np.array_equal(lbl_both, lbl_in)
        comp_all_ok &= (fg_equal and lbl_equal)
        if not (fg_equal and lbl_equal):
            print(f"    FAIL frame {idx}: fg_equal={fg_equal} lbl_equal={lbl_equal}")

    print(f"  frames tested: {len(wire_frames)}; total fg px: {comp_fg_total}")
    print(f"  foreground of (wirecolor->bgclutter) == foreground of (wirecolor "
          f"only) on ALL frames: {comp_all_ok}")
    print(f"  --> GATE 5 {'PASS' if comp_all_ok else 'FAIL'}")
    results["gate5"] = comp_all_ok

    # ---------------------------------------------------------------
    # Throughput / per-sample overhead.
    # ---------------------------------------------------------------
    print("\n=== Throughput / per-sample overhead ===")
    aug_t = m.BgClutterAugmentation(p=1.0, num_classes=2)  # worst case: always fires
    for idx in wire_frames[:3]:
        aug_t(train_rgb[idx].copy(), train_label[idx].copy())
    REPS = 200
    frames_cycle = wire_frames * ((REPS // len(wire_frames)) + 1)
    random.seed(SEED)
    t0 = time.perf_counter()
    for k in range(REPS):
        idx = frames_cycle[k]
        aug_t(train_rgb[idx].copy(), train_label[idx].copy())
    dt = time.perf_counter() - t0
    per_sample_ms = dt / REPS * 1000.0
    print(f"  bgclutter-always (p=1.0): {per_sample_ms:.3f} ms/sample over {REPS} reps")
    t0 = time.perf_counter()
    for k in range(REPS):
        ds_off[frames_cycle[k] % len(ds_off)]
    dt_base = time.perf_counter() - t0
    base_ms = dt_base / REPS * 1000.0
    print(f"  baseline __getitem__ (flag off): {base_ms:.3f} ms/sample")
    print(f"  effective overhead at p=0.5: ~{per_sample_ms * 0.5:.3f} ms/sample")
    results["per_sample_ms"] = per_sample_ms

    # ---------------------------------------------------------------
    print("\n=== SUMMARY ===")
    crit = (results["gate1"] and results["gate2"] and results["gate3"]
            and results["gate4"] and results["gate5"])
    print(f"  GATE 1 default byte-identity + RNG isolation : "
          f"{'PASS' if results['gate1'] else 'FAIL'}")
    print(f"  GATE 2 FOREGROUND bytes untouched            : "
          f"{'PASS' if results['gate2'] else 'FAIL'}")
    print(f"  GATE 3 label preserved                       : "
          f"{'PASS' if results['gate3'] else 'FAIL'}")
    print(f"  GATE 4 texture isotropic / not wire-like     : "
          f"{'PASS' if results['gate4'] else 'FAIL'} "
          f"(full-stack orient-peak max={results['gate4_full_peak']:.2f}, "
          f"median elong={results['gate4_full_median']:.3f}, "
          f"max elong={results['gate4_full_max']:.3f}, "
          f"ctrl-wires-flagged={results['gate4_ctrl_flagged']})")
    print(f"  GATE 5 composition with --aug-wirecolor      : "
          f"{'PASS' if results['gate5'] else 'FAIL'}")
    print(f"  per-sample overhead (p=1.0)                  : {per_sample_ms:.3f} ms")
    print(f"  ALL CRITICAL GATES: {'PASS' if crit else 'FAIL'}")
    return 0 if crit else 1


if __name__ == "__main__":
    sys.exit(main())
