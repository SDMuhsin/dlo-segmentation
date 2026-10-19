import copy
import tempfile
import unittest
import numpy as np
from connector_insertion.config import ConnectorInsertionConfig as C
from connector_insertion.constants import INSERTED_CONNECTOR_LABEL
from connector_insertion.insert import make_rng, plan_connector_insertion
from connector_insertion.library import ConnectorLibrary, load_asset, save_asset
from .helpers import make_asset, make_harness, make_library


class TestLibrary(unittest.TestCase):
    def test_round_trip(self):
        a = make_asset("rt")
        with tempfile.TemporaryDirectory() as d:
            save_asset(f"{d}/rt.npz", a)
            b = load_asset(f"{d}/rt.npz")
            lib = ConnectorLibrary.load(d)
        self.assertEqual(b.name, "rt"); self.assertEqual(lib.names, ["rt"])
        np.testing.assert_allclose(a.points, b.points); np.testing.assert_array_equal(a.colors, b.colors)
        self.assertAlmostEqual(a.width, b.width)

    def test_missing_dir_is_empty(self):
        self.assertEqual(len(ConnectorLibrary.load("/nonexistent/dir")), 0)

    def test_select_filters_and_weights(self):
        lib = make_library(make_asset("usb_a"), make_asset("usb_c"), make_asset("xlr"))
        names = lambda r: [a.name for a in r[0]]
        self.assertEqual(names(lib.select(include=["usb*"])), ["usb_a", "usb_c"])
        self.assertEqual(names(lib.select(exclude=["xlr"])), ["usb_a", "usb_c"])
        a, p = lib.select(weights={"xlr": 2.0})
        self.assertAlmostEqual(p[2], 0.5); self.assertAlmostEqual(p.sum(), 1.0)
        self.assertEqual(lib.select(weights={"usb_a": 0, "usb_c": 0, "xlr": 0})[0], [])


class TestInsert(unittest.TestCase):
    def setUp(self):
        self.pcl, self.seg = make_harness()
        self.rgb = np.full((len(self.pcl), 3), 90, np.uint8)
        self.lib = make_library(make_asset("a"), make_asset("b", width=0.5))
        self.cfg = C(enabled=True)

    def run_plan(self, cfg=None, seed=3):
        cfg = cfg or self.cfg
        return plan_connector_insertion(self.pcl, self.seg, self.rgb, cfg,
                                        np.random.RandomState(seed), self.lib)

    def test_disabled_is_noop(self):
        r = plan_connector_insertion(self.pcl, self.seg, self.rgb, C(), np.random.RandomState(0), self.lib)
        self.assertFalse(r.active); self.assertFalse(r.drop_mask.any())
        self.assertEqual(r.extra_pts.shape, (0, 3)); self.assertEqual(len(r.drop_mask), len(self.pcl))

    def test_empty_library_or_no_slots_is_noop(self):
        r = plan_connector_insertion(self.pcl, self.seg, self.rgb, self.cfg,
                                     np.random.RandomState(0), ConnectorLibrary([]))
        self.assertFalse(r.active)
        r = plan_connector_insertion(self.pcl, np.zeros_like(self.seg), self.rgb, self.cfg,
                                     np.random.RandomState(0), self.lib)
        self.assertFalse(r.active)

    def test_no_mutation(self):
        snap = [copy.deepcopy(x) for x in (self.pcl, self.seg, self.rgb)]
        self.run_plan()
        for a, b in zip(snap, (self.pcl, self.seg, self.rgb)):
            np.testing.assert_array_equal(a, b)

    def test_result_contract(self):
        r = self.run_plan()
        self.assertTrue(r.active)
        self.assertTrue((self.seg[r.drop_mask] == 3).all())
        self.assertEqual(r.drop_mask.sum(), (self.seg == 3).sum())     # replace_prob=1
        self.assertTrue((r.extra_labels == INSERTED_CONNECTOR_LABEL).all())
        self.assertEqual(r.extra_rgb.dtype, np.uint8)
        self.assertEqual(len(r.extra_pts), len(r.extra_rgb)); self.assertEqual(len(r.extra_pts), len(r.extra_labels))
        self.assertEqual(r.report["n_replaced"], 1)

    def test_determinism_and_stream_independence(self):
        a, b = self.run_plan(seed=7), self.run_plan(seed=7)
        np.testing.assert_array_equal(a.extra_pts, b.extra_pts)
        np.testing.assert_array_equal(a.extra_rgb, b.extra_rgb)
        np.random.seed(123); np.random.rand(1000)                        # other streams
        other = np.random.RandomState(99); other.rand(500)
        c = self.run_plan(seed=7)
        np.testing.assert_array_equal(a.extra_pts, c.extra_pts)
        d = self.run_plan(seed=8)
        self.assertFalse(np.array_equal(a.extra_pts, d.extra_pts))

    def test_make_rng_separate_stream(self):
        r1 = make_rng(3, 5, self.cfg).uniform()
        r2 = np.random.RandomState(3 * 1000 + 5).uniform()               # the anim RNG offset 0
        self.assertNotEqual(r1, r2)
        self.assertEqual(r1, make_rng(3, 5, self.cfg).uniform())

    def test_replace_prob_zero_and_asset_filter(self):
        r = self.run_plan(C(enabled=True, replace_prob=0.0))
        self.assertFalse(r.active)
        r = self.run_plan(C(enabled=True, include=("b",)))
        self.assertEqual(r.report["slots"][0]["asset"], "b")

    def test_tint_changes_colour_toward_wire(self):
        native = self.run_plan(C(enabled=True, wire_tint=0.0)).extra_rgb.astype(float)
        tinted = self.run_plan(C(enabled=True, wire_tint=1.0)).extra_rgb.astype(float)
        self.assertLess(abs(tinted.mean() - 90), abs(native.mean() - 90) + 1e-9)
        self.assertLess(np.abs(tinted.mean(0) - 90).max(), 30)

    def test_max_per_frame(self):
        p2, s2 = make_harness(seed=1)
        p2 = p2 + np.array([0, 0.8, 0.0]); p2[len(p2) - 400:] = self.pcl[-400:] + [0, 0.8, 0]
        pcl = np.concatenate([self.pcl, p2]); seg = np.concatenate([self.seg, s2])
        rgb = np.full((len(pcl), 3), 90, np.uint8)
        r = plan_connector_insertion(pcl, seg, rgb, C(enabled=True, max_per_frame=1),
                                     np.random.RandomState(0), self.lib)
        self.assertEqual(r.report["n_replaced"], 1)


if __name__ == "__main__":
    unittest.main()
