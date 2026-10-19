import unittest
import numpy as np
from connector_insertion.config import ConnectorInsertionConfig as C
from connector_insertion.slots import detect_slots
from .helpers import make_harness

PX = 480 / 2.2


class TestSlots(unittest.TestCase):
    def test_single_cluster_geometry(self):
        pcl, seg = make_harness(18, 8)
        (s,) = detect_slots(pcl, seg, C())
        self.assertFalse(s.degenerate)
        self.assertTrue(s.has_wire)
        np.testing.assert_allclose(s.axis, [1, 0, 0], atol=0.15)    # toward the wire (+X)
        self.assertAlmostEqual(s.length * PX, 18, delta=2.5)
        self.assertAlmostEqual(s.width * PX, 8, delta=2.5)
        self.assertEqual(len(s.indices), 120)

    def test_axis_flips_with_wire_side(self):
        pcl, seg = make_harness()
        pcl2 = pcl.copy(); pcl2[:, 0] *= -1                       # wire now on the -X side
        (s,) = detect_slots(pcl2, seg, C())
        self.assertLess(s.axis[0], -0.9)

    def test_no_wire_deterministic_fallback(self):
        pcl, seg = make_harness()
        keep = seg == 3
        (s,) = detect_slots(pcl[keep], seg[keep], C())
        self.assertFalse(s.has_wire)
        (s2,) = detect_slots(pcl[keep], seg[keep], C())
        np.testing.assert_array_equal(s.axis, s2.axis)

    def test_small_and_degenerate_clusters(self):
        pcl = np.zeros((20, 3)); seg = np.full(20, 3)
        self.assertEqual(detect_slots(pcl[:5], seg[:5], C()), [])   # below min points
        out = detect_slots(pcl, seg, C())                           # all points identical
        self.assertTrue(all(s.degenerate for s in out))

    def test_flat_cluster_ok(self):
        rs = np.random.RandomState(1)
        flat = np.stack([rs.uniform(0, .08, 60), rs.uniform(0, .04, 60), np.zeros(60)], 1)
        (s,) = detect_slots(flat, np.full(60, 3), C())
        self.assertFalse(s.degenerate)
        self.assertGreater(s.width, 0)

    def test_split_large(self):
        rs = np.random.RandomState(2)
        a = np.stack([rs.uniform(0, .1, 100), rs.uniform(0, .03, 100), np.zeros(100)], 1)
        b = a + np.array([0.12, 0, 0]) 
        pcl = np.concatenate([a, b, a[:3] * 0 + [5, 5, 5]]); seg = np.full(len(pcl), 3)
        merged = detect_slots(pcl[:200], seg[:200], C(cluster_eps=0.15))
        self.assertEqual(len(merged), 1)
        split = detect_slots(pcl[:200], seg[:200], C(cluster_eps=0.15, split_large=True,
                                                     split_length_px=30))
        self.assertGreaterEqual(len(split), 2)


if __name__ == "__main__":
    unittest.main()
