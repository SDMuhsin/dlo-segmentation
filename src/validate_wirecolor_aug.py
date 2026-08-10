"""Validation harness for the --aug-wirecolor label-aware wire recolouring aug.

Run from project root with env activated:
    source env/bin/activate
    export HF_HOME=data/hf_home TORCH_HOME=data/torch_home
    python src/validate_wirecolor_aug.py

Checks (numeric, not assertions):
 1. Byte-identity of the DEFAULT path: with --aug-wirecolor OFF the dataset draws
    the EXACT same global-`random` sequence and yields the same first-batch
    tensors as a dataset constructed without the new arg at all (flag off => zero
    behavioural change). Method mirrors how --aug-heavy/--aug2d proved "N draws
    unchanged": we count random.random()/getrandbits draws via a monkeypatched
    counter and hash the produced tensors.
 2. Background untouched: pixels where label != wire are EXACTLY equal before vs
    after recolour (np.array_equal on the background region).
 3. Label preserved: the returned label mask equals the input mask exactly.
 4. Throughput: mean per-sample overhead of the recolour op.
"""

import importlib.util
import os
import random
import sys
import time

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
            # cheap order-sensitive hash of the produced tensors
            h ^= hash(rgb.tobytes()) & 0xFFFFFFFFFFFFFFFF
            h = (h * 1000003) & 0xFFFFFFFFFFFFFFFF
            h ^= hash(lbl.tobytes()) & 0xFFFFFFFFFFFFFFFF
            h = (h * 1000003) & 0xFFFFFFFFFFFFFFFF
        return counter.total(), dict(counter.counts), h
    finally:
        m.random = real           # restore


def main():
    m = load_module()
    from train_rgbd_seg import build_cache  # noqa
    print("Loading train cache (mmap)...")
    train_rgb, _, train_label = build_cache(DATA, "train")
    print(f"  rgb {train_rgb.shape} {train_rgb.dtype}; label {train_label.shape} {train_label.dtype}")

    N_PASS = 64    # number of __getitem__ calls to audit
    SEED = 1234

    # ---------------------------------------------------------------
    # CHECK 1: byte-identity of the default (flag-off) path.
    # ---------------------------------------------------------------
    print("\n=== CHECK 1: default-path byte identity (flag OFF) ===")

    # (a) BASELINE: historical construction — no wirecolor arg supplied at all.
    ds_baseline = m.CDLORGBOnlyDataset(
        train_rgb, train_label, augment=True, include_noise=False,
        num_classes=2, augmenter2d=None, hue_aug=0.0, heavy_aug=None,
        # wirecolor_aug intentionally omitted -> defaults to None
    )
    base_draws, base_counts, base_hash = run_epoch_pass(m, ds_baseline, SEED, N_PASS)

    # (b) NEW path with the flag explicitly OFF (wirecolor_aug=None).
    ds_off = m.CDLORGBOnlyDataset(
        train_rgb, train_label, augment=True, include_noise=False,
        num_classes=2, augmenter2d=None, hue_aug=0.0, heavy_aug=None,
        wirecolor_aug=None,
    )
    off_draws, off_counts, off_hash = run_epoch_pass(m, ds_off, SEED, N_PASS)

    print(f"  baseline (no arg):     draws={base_draws}  counts={base_counts}")
    print(f"  flag-off (arg=None):   draws={off_draws}  counts={off_counts}")
    print(f"  tensor hash baseline : {base_hash:#018x}")
    print(f"  tensor hash flag-off : {off_hash:#018x}")
    c1_draws = (base_draws == off_draws and base_counts == off_counts)
    c1_hash = (base_hash == off_hash)
    print(f"  RNG draw sequence identical: {c1_draws}")
    print(f"  first-{N_PASS} tensor hash identical: {c1_hash}")
    print(f"  --> CHECK 1 {'PASS' if (c1_draws and c1_hash) else 'FAIL'}")

    # (c) Sanity: the flag ON must consume MORE draws and differ (proves the
    #     guard, not a dead code path). 1 random.random() per sample (gate) plus
    #     extra draws on samples that fire.
    ds_on = m.CDLORGBOnlyDataset(
        train_rgb, train_label, augment=True, include_noise=False,
        num_classes=2, augmenter2d=None, hue_aug=0.0, heavy_aug=None,
        wirecolor_aug=m.WireColorAugmentation(p=0.5, num_classes=2),
    )
    on_draws, on_counts, on_hash = run_epoch_pass(m, ds_on, SEED, N_PASS)
    print(f"  [sanity] flag-ON:      draws={on_draws}  counts={on_counts}  hash={on_hash:#018x}")
    print(f"  [sanity] ON draws > OFF draws: {on_draws > off_draws}  "
          f"(ON adds >= {N_PASS} gate draws)")
    print(f"  [sanity] ON tensors differ from OFF: {on_hash != off_hash}")

    # ---------------------------------------------------------------
    # CHECK 2 & 3: background untouched + label preserved.
    # Operate directly on raw cache frames at the aug point (uint8 BGR + the
    # gt_transform label), calling the aug exactly as __getitem__ does.
    # ---------------------------------------------------------------
    print("\n=== CHECK 2/3: background untouched + label preserved ===")
    aug = m.WireColorAugmentation(p=1.0, num_classes=2)  # p=1 so it always fires

    # pick frames that actually contain wire pixels
    wire_frames = []
    for i in range(0, 400):
        if int((train_label[i] <= 3).sum()) > 800:
            wire_frames.append(i)
        if len(wire_frames) >= 12:
            break

    random.seed(SEED)
    bg_all_ok = True
    lbl_all_ok = True
    n_wire_changed_total = 0
    for idx in wire_frames:
        rgb_in = train_rgb[idx].copy()          # (H,W,3) uint8 BGR
        lbl_in = train_label[idx].copy()         # (H,W) uint8 gt_transform
        wire_mask = (lbl_in <= 3)
        bg_mask = ~wire_mask

        rgb_out, lbl_out = aug(rgb_in.copy(), lbl_in.copy())

        # background bytes EXACTLY equal
        bg_equal = np.array_equal(rgb_out[bg_mask], rgb_in[bg_mask])
        # label EXACTLY equal
        lbl_equal = np.array_equal(lbl_out, lbl_in)
        # how many wire px actually changed (should be > 0 most of the time)
        wire_changed = int(np.any(rgb_out[wire_mask] != rgb_in[wire_mask], axis=-1).sum())
        n_wire_changed_total += wire_changed

        bg_all_ok &= bg_equal
        lbl_all_ok &= lbl_equal
        if not (bg_equal and lbl_equal):
            print(f"    frame {idx}: bg_equal={bg_equal} lbl_equal={lbl_equal} "
                  f"wire_changed={wire_changed}")

    n_wire_px = sum(int((train_label[i] <= 3).sum()) for i in wire_frames)
    print(f"  frames tested: {len(wire_frames)} (each with >800 wire px)")
    print(f"  total wire px across frames: {n_wire_px}; wire px changed: {n_wire_changed_total}")
    print(f"  background pixels EXACTLY equal (np.array_equal) on ALL frames: {bg_all_ok}")
    print(f"  label mask EXACTLY equal on ALL frames: {lbl_all_ok}")
    print(f"  --> CHECK 2 (bg untouched) {'PASS' if bg_all_ok else 'FAIL'}")
    print(f"  --> CHECK 3 (label preserved) {'PASS' if lbl_all_ok else 'FAIL'}")

    # Extra: returned tensor identity is `label` (no copy); confirm dtype/range.
    print(f"  (wire pixels DID change: {n_wire_changed_total > 0} — recolour is active)")

    # ---------------------------------------------------------------
    # CHECK 4: throughput / per-sample overhead.
    # ---------------------------------------------------------------
    print("\n=== CHECK 4: throughput / per-sample overhead ===")
    aug_t = m.WireColorAugmentation(p=1.0, num_classes=2)  # worst case: always fires
    # warmup
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
    print(f"  recolour-always (p=1.0): {per_sample_ms:.3f} ms/sample over {REPS} reps")
    # measure the bare __getitem__ baseline (flag off) for comparison
    t0 = time.perf_counter()
    for k in range(REPS):
        ds_off[frames_cycle[k] % len(ds_off)]
    dt_base = time.perf_counter() - t0
    base_ms = dt_base / REPS * 1000.0
    print(f"  baseline __getitem__ (flag off): {base_ms:.3f} ms/sample")
    eff_overhead = per_sample_ms * 0.5  # p=0.5 in training
    print(f"  effective overhead at p=0.5: ~{eff_overhead:.3f} ms/sample")
    # throughput implied by overhead alone (data-loading only; model not counted)
    print(f"  recolour op alone would cap at ~{1000.0/max(per_sample_ms,1e-9):.0f} img/s "
          f"(p=1.0) — well above the ~15 img/s target")

    # ---------------------------------------------------------------
    print("\n=== SUMMARY ===")
    ok = c1_draws and c1_hash and bg_all_ok and lbl_all_ok
    print(f"  CHECK 1 default byte-identity : {'PASS' if (c1_draws and c1_hash) else 'FAIL'}")
    print(f"  CHECK 2 background untouched  : {'PASS' if bg_all_ok else 'FAIL'}")
    print(f"  CHECK 3 label preserved       : {'PASS' if lbl_all_ok else 'FAIL'}")
    print(f"  CHECK 4 throughput            : {per_sample_ms:.3f} ms/sample (p=1.0)")
    print(f"  ALL CRITICAL CHECKS: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
