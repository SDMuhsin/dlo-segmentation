import unittest
import numpy as np
from connector_insertion.config import ConnectorInsertionConfig as C
from connector_insertion.placement import draw_slot_randoms, fit_scale, place_asset
from connector_insertion.slots import detect_slots
from .helpers import make_asset, make_harness

PX = 480 / 2.2


def _slot(length=18, width=8):
    pcl, seg = make_harness(length, width)
    return detect_slots(pcl, seg, C())[0]


def _axis_of(p):
    _, _, vt = np.linalg.svd(p - p.mean(0), full_matrices=False)
    return vt[0]


class TestPlacement(unittest.TestCase):
    def setUp(self):
        self.asset, self.slot = make_asset(), _slot()
        self.d = draw_slot_randoms(np.random.RandomState(5))

    def test_axis_alignment_and_cable_end_at_wire(self):
        pl = place_asset(self.asset, self.slot, C(enabled=True), self.d)
        ax = _axis_of(pl.points)
        self.assertGreater(abs(float(ax @ self.slot.axis)), 0.995)
        # the cable-end tenth of the model is closer to the wire end than the mating-end tenth
        t = (pl.points - self.slot.center) @ self.slot.axis
        blue = pl.colors[:, 0]                              # ramps toward mating end
        self.assertGreater(t[blue < 40].mean(), t[blue > 215].mean())
        self.assertLessEqual(t.max(), self.slot.length / 2 + 1e-6)   # never beyond the wire end
        self.assertAlmostEqual(t.min(), -self.slot.length / 2, delta=0.003)

    def test_length_fit_exact_at_mult_1(self):
        pl = place_asset(self.asset, self.slot, C(enabled=True), self.d)
        self.assertAlmostEqual(pl.fit.scale, self.slot.length, places=9)

    def test_fit_modes(self):
        L, W = self.slot.length, self.slot.width
        for mode, want in {"length": L, "width": W / self.asset.width,
                           "min": min(L, W / self.asset.width),
                           "max": max(L, W / self.asset.width),
                           "geomean": np.sqrt(L * W / self.asset.width)}.items():
            cfg = C(enabled=True, fit_mode=mode, min_length_px=1, max_length_px=500)
            f = fit_scale(self.asset, L, W, cfg, 0.5)
            self.assertAlmostEqual(f.scale / want, 1.0, delta=1e-6, msg=mode)

    def test_scale_mult_and_jitter_and_clamps(self):
        L, W = self.slot.length, self.slot.width
        f = fit_scale(self.asset, L, W, C(enabled=True, scale_mult=1.5, max_length_px=500), 0.5)
        self.assertAlmostEqual(f.scale / L, 1.5, places=6)
        fj = fit_scale(self.asset, L, W, C(enabled=True, scale_jitter=0.2), 1.0 - 1e-12)
        self.assertAlmostEqual(fj.scale / L, 1.2, places=3)
        lo = fit_scale(self.asset, 0.001, 0.001, C(enabled=True, min_length_px=12), 0.5)
        self.assertAlmostEqual(lo.target_length_px, 12.0); self.assertTrue(lo.clamped)
        hi = fit_scale(self.asset, 1.0, 1.0, C(enabled=True, max_length_px=48), 0.5)
        self.assertAlmostEqual(hi.target_length_px, 48.0); self.assertTrue(hi.clamped)

    def test_point_budget(self):
        cfg = C(enabled=True, max_points=500, min_points=100, scale_mult=2)
        self.assertLessEqual(len(place_asset(self.asset, self.slot, cfg, self.d).points), 500)

    def test_roll_modes_and_flip(self):
        base = C(enabled=True, roll_mode="none")
        a = place_asset(self.asset, self.slot, base, self.d).points
        b = place_asset(self.asset, self.slot, base, self.d).points
        np.testing.assert_array_equal(a, b)
        r = place_asset(self.asset, self.slot, C(enabled=True, roll_mode="fixed", roll_deg=90), self.d)
        self.assertAlmostEqual(r.roll_deg, 90.0)
        f = place_asset(self.asset, self.slot, C(enabled=True, flip_prob=1.0), self.d)
        self.assertTrue(f.flipped)

    def test_tilt_keeps_cable_anchor(self):
        cfg = C(enabled=True, tilt_jitter_deg=20)
        d = draw_slot_randoms(np.random.RandomState(9))
        pl = place_asset(self.asset, self.slot, cfg, d)
        self.assertLessEqual(pl.tilt_deg, 20)
        self.assertGreater(abs(float(_axis_of(pl.points) @ self.slot.axis)), np.cos(np.deg2rad(21)))


if __name__ == "__main__":
    unittest.main()
