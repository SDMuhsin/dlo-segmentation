import unittest
from connector_insertion.constants import INSERTED_CONNECTOR_LABEL, PX_PER_UNIT


class TestConstants(unittest.TestCase):
    def test_label_does_not_collide(self):
        self.assertNotIn(INSERTED_CONNECTOR_LABEL, (0, 1, 2, 3, 4, 255, -1))
        self.assertGreater(INSERTED_CONNECTOR_LABEL, 4)
        self.assertLess(INSERTED_CONNECTOR_LABEL, 255)

    def test_matches_rasteriser(self):
        import pcl_to_rgbd
        self.assertEqual(pcl_to_rgbd.INSERTED_CONNECTOR_LABEL, INSERTED_CONNECTOR_LABEL)
        self.assertAlmostEqual(pcl_to_rgbd.SCALE, PX_PER_UNIT, places=9)

    def test_rasteriser_maps_label_to_png_4(self):
        import numpy as np, pcl_to_rgbd
        pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        lab = np.array([INSERTED_CONNECTOR_LABEL, 0])
        _, _, li = pcl_to_rgbd.rasterize_view(
            pts, lab, np.eye(3), point_rgb=np.full((2, 3), 100, np.uint8))
        self.assertEqual(set(np.unique(li)) - {0}, {1, 4})


if __name__ == "__main__":
    unittest.main()
