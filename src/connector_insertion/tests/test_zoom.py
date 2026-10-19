"""Zoom views: framing maths, rasteriser no-op identity, window culling, backdrop, planner."""
import unittest

import numpy as np

import pcl_to_rgbd as P
from connector_insertion.config import ConnectorInsertionConfig as C
from connector_insertion.constants import INSERTED_CONNECTOR_LABEL, ZOOM_RNG_OFFSET
from connector_insertion.insert import make_rng, plan_connector_insertion, resample_inserted
from connector_insertion.zoom import (
    WireTube, ZoomRenderer, crop_backdrop, fit_centre, plan_zoom_views, project,
    projected_extent, zoom_for_target)
from .helpers import make_asset, make_harness, make_library

SC = P.SCALE
ROTS = {n: P.make_view_matrix(v["look"], v["up"]) for n, v in P.VIEWS.items()}


def cloud(n=4000, seed=1, with_ties=False):
    rs = np.random.RandomState(seed)
    pts = np.stack([rs.uniform(-0.7, 0.7, n), rs.uniform(-0.5, 0.5, n), rs.uniform(-0.3, 0.3, n)], 1)
    if with_ties:
        pts[:, 2] = np.round(pts[:, 2] * 20) / 20
    lab = rs.choice([0, 255], n)
    rgb = rs.randint(0, 255, (n, 3)).astype(np.uint8)
    return pts, lab, rgb


class TestRasteriserNoOp(unittest.TestCase):
    def test_default_args_are_legacy_path(self):
        # explicit zoom=1/centre=(0,0) gives the legacy image when no two points tie in
        # quantised depth (the zoomed path sorts stably, the legacy path does not)
        pts, lab, rgb = cloud(300)
        pts[:, 2] = np.linspace(-0.4, 0.4, len(pts))          # 0.8 units = 364 mm > 300 distinct values
        bg = np.random.RandomState(0).randint(0, 255, (480, 640, 3)).astype(np.uint8)
        a = P.rasterize_view(pts, lab, ROTS["front"], point_rgb=rgb, background=bg)
        b = P.rasterize_view(pts, lab, ROTS["front"], point_rgb=rgb, background=bg,
                             zoom=1.0, centre=(0.0, 0.0))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_depth_encoding_independent_of_zoom(self):
        p = np.array([[0.05, -0.02, 0.123]])
        _, d1, _ = P.rasterize_view(p, np.array([0]), np.eye(3))
        _, d4, _ = P.rasterize_view(p, np.array([0]), np.eye(3), zoom=4.0, centre=(0.05, -0.02))
        self.assertEqual(d1.max(), d4.max())
        self.assertEqual(int(d1.max()), int(P.depth_to_uint16(np.array([0.123]))[0]))

    def test_pixel_of_a_point(self):
        p = np.array([[0.30, 0.10, 0.0]])
        for zoom, c in ((2.0, (0.25, 0.05)), (5.0, (0.29, 0.09))):
            _, d, _ = P.rasterize_view(p, np.array([0]), np.eye(3), zoom=zoom, centre=c)
            ys, xs = np.nonzero(d)
            exp_u = (0.30 - c[0]) * SC * zoom + 320
            exp_v = (0.10 - c[1]) * SC * zoom + 240
            self.assertLess(abs(xs.mean() - exp_u), 0.6)
            self.assertLess(abs(ys.mean() - exp_v), 0.6)
            self.assertGreaterEqual(len(xs), 2 * int(np.ceil(zoom)))   # splat radius ceil(zoom)

    def test_connector_label_keeps_radius_one(self):
        p = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        lab = np.array([INSERTED_CONNECTOR_LABEL, 0])
        _, d, l = P.rasterize_view(p, lab, np.eye(3), zoom=4.0, centre=(0.25, 0.0))
        self.assertEqual(int((l == 4).sum()), 5)                      # radius-1 cross
        self.assertGreater(int((l == 1).sum()), 40)                   # radius-4 disc

    def test_vectorised_equals_loop(self):
        pts, lab, rgb = cloud(5000, with_ties=True)
        lab[:50] = INSERTED_CONNECTOR_LABEL
        kw = dict(point_rgb=rgb, zoom=3.0, centre=(0.12, -0.05))
        P.ZOOM_VECTORISED = False
        try:
            a = P.rasterize_view(pts, lab, ROTS["left"], **kw)
        finally:
            P.ZOOM_VECTORISED = True
        b = P.rasterize_view(pts, lab, ROTS["left"], **kw)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_window_cull_equals_full_render_crop(self):
        pts, lab, rgb = cloud(6000, seed=3, with_ties=True)
        k = 3
        m = (25, -18)                                                 # window offset in zoom pixels
        centre = (m[0] / (SC * k), m[1] / (SC * k))
        z = P.rasterize_view(pts, lab, ROTS["front"], point_rgb=rgb, zoom=k, centre=centre)
        old = (P.IMG_W, P.IMG_H, P.SCALE)
        try:
            P.IMG_W, P.IMG_H, P.SCALE = 640 * k, 480 * k, SC * k
            big = P.rasterize_view(pts, lab, ROTS["front"], point_rgb=rgb,
                                   point_radius=np.full(len(pts), k))
        finally:
            P.IMG_W, P.IMG_H, P.SCALE = old
        x0, y0 = 320 * k - 320 + m[0], 240 * k - 240 + m[1]
        s = (slice(4, -4), slice(4, -4))
        for zi, bi in zip(z, big):
            crop = bi[y0:y0 + 480, x0:x0 + 640]
            np.testing.assert_array_equal(zi[s], crop[s])


class TestFramingMaths(unittest.TestCase):
    def test_zoom_for_target_and_clamp(self):
        cfg = C(enabled=True, zoom_views=1, zoom_min=1.5, zoom_max=8.0)
        self.assertAlmostEqual(zoom_for_target(20 / SC, 0.25, cfg), 0.25 * 480 / 20)  # 6.0
        self.assertEqual(zoom_for_target(5 / SC, 0.45, cfg), 8.0)
        self.assertEqual(zoom_for_target(300 / SC, 0.1, cfg), 1.5)

    def test_fit_centre_keeps_bbox_inside(self):
        xy = np.array([[0.0, 0.0], [0.10, 0.02]])
        c = fit_centre(np.array([5.0, -5.0]), xy, 4.0)
        hw, hh = 320 / (SC * 4), 240 / (SC * 4)
        self.assertLessEqual(xy[:, 0].max(), c[0] + hw); self.assertGreaterEqual(xy[:, 0].min(), c[0] - hw)
        self.assertLessEqual(xy[:, 1].max(), c[1] + hh); self.assertGreaterEqual(xy[:, 1].min(), c[1] - hh)

    def setUp(self):
        self.pcl, self.seg = make_harness(length_px=18.0, width_px=8.0)
        self.rgb = np.full((len(self.pcl), 3), 90, np.uint8)
        self.cfg = C(enabled=True, zoom_views=4, zoom_max=40.0, zoom_min_projected_frac=0.9,
                     scale_mult=1.0, min_length_px=1.0)
        self.lib = make_library(make_asset("a"))
        self.ins = plan_connector_insertion(self.pcl, self.seg, self.rgb, self.cfg, make_rng(1, 2, self.cfg),
                                            self.lib)

    def test_connector_lands_at_target_fraction(self):
        plans = plan_zoom_views(self.ins, ROTS, self.cfg, 1, 2)
        self.assertEqual(len(plans), 4)
        isl = self.ins.inserted[0]
        std = self.ins.extra_pts[isl.start:isl.stop]
        for pl in plans:
            for cand in pl.candidates:
                self.assertFalse(cand.zoom_clamped)
                ext, _ = projected_extent(project(std, cand.R)[:, :2])
                self.assertAlmostEqual(ext * SC * cand.zoom / 480.0, cand.target_frac, places=6)
                # the connector bbox fits in the frame
                x0, y0, x1, y1 = cand.bbox_px
                self.assertTrue(x0 >= 0 and y0 >= 0 and x1 <= 640 and y1 <= 480)

    def test_rendered_connector_length_matches_plan(self):
        plans = plan_zoom_views(self.ins, ROTS, self.cfg, 1, 2)
        cand = plans[0].candidates[0]
        isl = self.ins.inserted[0]
        pts, col, _ = resample_inserted(isl, self.cfg, SC * cand.zoom, 0.7, 150000)
        lab = np.full(len(pts), INSERTED_CONNECTOR_LABEL)
        _, _, l = P.rasterize_view(pts, lab, cand.R, point_rgb=col, zoom=cand.zoom,
                                   centre=tuple(cand.centre))
        ys, xs = np.nonzero(l == 4)
        ext, _ = projected_extent(np.stack([xs, ys], 1).astype(float))
        self.assertLess(abs(ext - cand.planned_len_px), 0.12 * cand.planned_len_px + 4)   # + splat width
        self.assertGreaterEqual(xs.min(), 0); self.assertLess(xs.max(), 640)

    def test_resample_is_same_connector_denser(self):
        isl = self.ins.inserted[0]
        p1, c1, n1 = resample_inserted(isl, self.cfg, SC, 0.5, 8000)
        p4, c4, n4 = resample_inserted(isl, self.cfg, SC * 4, 0.5, 150000)
        self.assertGreater(n4, 3 * n1)
        self.assertLessEqual(n4, 6000)                    # never upsample beyond the asset's points
        # same pose and size: the sparse points are the prefix of the dense ones
        np.testing.assert_allclose(p4[:n1], p1)

    def test_determinism_and_view_independence(self):
        a = plan_zoom_views(self.ins, ROTS, self.cfg, 1, 2)
        b = plan_zoom_views(self.ins, ROTS, self.cfg, 1, 2)
        np.testing.assert_array_equal(a[0].candidates[0].centre, b[0].candidates[0].centre)
        cfg2 = C(**{**self.cfg.to_dict(), "zoom_views": 2})
        c = plan_zoom_views(self.ins, ROTS, cfg2, 1, 2)
        self.assertEqual(len(c), 2)
        for x, y in zip(a[:2], c):
            self.assertEqual(x.candidates[0].view, y.candidates[0].view)
            self.assertAlmostEqual(x.candidates[0].zoom, y.candidates[0].zoom)

    def test_off_and_fallback(self):
        off = C(enabled=True, zoom_views=0)
        self.assertEqual(plan_zoom_views(self.ins, ROTS, off, 1, 2), [])
        empty = plan_connector_insertion(self.pcl, np.zeros_like(self.seg), self.rgb, self.cfg,
                                         make_rng(1, 2, self.cfg), self.lib)
        self.assertEqual(plan_zoom_views(empty, ROTS, self.cfg, 1, 2, self.pcl, np.zeros_like(self.seg)), [])
        wire = C(**{**self.cfg.to_dict(), "zoom_fallback": "wire"})
        pl = plan_zoom_views(empty, ROTS, wire, 1, 2, self.pcl, np.zeros_like(self.seg))
        self.assertEqual(len(pl), 4)
        self.assertTrue(all(p.slot == -1 and self.cfg.zoom_min <= p.candidates[0].zoom <= self.cfg.zoom_max
                            for p in pl))

    def test_excluded_views_never_used(self):
        self.assertEqual(C().zoom_exclude_views, ("bottom",))
        cfg = C(**{**self.cfg.to_dict(), "zoom_views": 12})
        used = {c.view for p in plan_zoom_views(self.ins, ROTS, cfg, 1, 2) for c in p.candidates}
        self.assertNotIn("bottom", used)
        cfg1 = C(**{**cfg.to_dict(), "zoom_exclude_views": [v for v in ROTS if v != "top"]})
        used = {c.view for p in plan_zoom_views(self.ins, ROTS, cfg1, 1, 2) for c in p.candidates}
        self.assertLessEqual(used, {"top"})
        allx = C(**{**cfg.to_dict(), "zoom_exclude_views": list(ROTS)})
        self.assertEqual(plan_zoom_views(self.ins, ROTS, allx, 1, 2), [])
        self.assertEqual(C.from_env({"KIAT_CONNINS_ZOOM_EXCLUDE_VIEWS": "bottom,top"}).zoom_exclude_views,
                         ("bottom", "top"))

    def test_perturb_rotation_is_orthonormal(self):
        cfg = C(**{**self.cfg.to_dict(), "zoom_rotation_mode": "perturb"})
        for pl in plan_zoom_views(self.ins, ROTS, cfg, 1, 2):
            R = pl.candidates[0].R
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(R), 1.0)
            self.assertTrue(pl.candidates[0].perturbed)


class TestBackdrop(unittest.TestCase):
    def test_identity_and_mapping(self):
        bg = np.zeros((480, 640, 3), np.uint8)
        np.testing.assert_array_equal(crop_backdrop(bg, np.zeros(2), 1.0), bg)
        bg[200:204, 400:404] = 255                                    # marker at (401.5, 201.5)
        k, c = 4.0, np.array([60.0 / SC, -30.0 / SC])                 # window centre (380, 210)
        out = crop_backdrop(bg, c, k)
        ys, xs = np.nonzero(out[..., 0] > 128)
        self.assertLess(abs(xs.mean() - ((401.5 - 380) * k + 320)), 1.5)
        self.assertLess(abs(ys.mean() - ((201.5 - 210) * k + 240)), 1.5)
        self.assertIs(crop_backdrop(bg, c, k, mode="static"), bg)
        self.assertIsNone(crop_backdrop(None, c, k))


class TestWireTubeAndRenderer(unittest.TestCase):
    def test_straight_tube_radius_and_caps(self):
        nodes = np.stack([np.linspace(-0.2, 0.2, 11), np.zeros(11), np.zeros(11)], 1)
        r = 6.0 / SC
        rs = np.random.RandomState(0)
        n = 600
        ang = rs.uniform(0, 2 * np.pi, n)
        xs = rs.uniform(-0.2, 0.2, n)
        pcl = np.stack([xs, r * np.cos(ang), r * np.sin(ang)], 1)
        na = np.clip(np.searchsorted(nodes[:, 0], xs) - 1, 0, 9)
        nb = na + 1
        seg = np.zeros(n, int)
        rgb = np.full((n, 3), 77, np.uint8)
        tube = WireTube(pcl, seg, na, nb, nodes, [list(range(11))], pcl, rgb, seg)
        self.assertEqual(len(tube.CP), 2)                             # two free ends, two caps
        Pt, C_, L_ = tube.build(np.eye(3), np.zeros(2), 4.0, 1.5)
        self.assertGreater(len(Pt), 1000)
        d = np.hypot(Pt[:, 1], Pt[:, 2])
        self.assertLess(np.abs(np.median(d) - r), 0.15 * r)
        self.assertTrue((C_ == 77).all() and (L_ == 0).all())

    def test_renderer_cloud_swaps_and_lighting_hook(self):
        pcl, seg = make_harness()
        rgb = np.full((len(pcl), 3), 90, np.uint8)
        cfg = C(enabled=True, zoom_views=1, zoom_wire_densify=False, zoom_max=40.0, min_length_px=1.0)
        ins = plan_connector_insertion(pcl, seg, rgb, cfg, make_rng(1, 2, cfg), make_library(make_asset("a")))
        keep = ~ins.drop_mask
        pts = np.concatenate([pcl[keep], ins.extra_pts])
        lab = np.concatenate([seg[keep], ins.extra_labels])
        col = np.concatenate([rgb[keep], ins.extra_rgb])
        calls = []

        def light(p, c):
            calls.append(len(p))
            return np.clip(c.astype(int) + 10, 0, 255).astype(np.uint8)
        zr = ZoomRenderer(cfg, ins, pts, lab, col, P.rasterize_view, None, light=light)
        plan = plan_zoom_views(ins, ROTS, cfg, 1, 2)[0]
        res = zr.render(plan)
        self.assertEqual(res.rgb.shape, (480, 640, 3)); self.assertEqual(res.depth.dtype, np.uint16)
        self.assertTrue(set(np.unique(res.label)) <= set(range(6)))
        self.assertGreater(res.info["connector_px"], 100)
        self.assertEqual(len(calls), 1)                               # resampled connector was lit
        self.assertEqual(res.info["connector_in_frame_frac"], 1.0)


class TestBackgroundOcclusion(unittest.TestCase):
    def test_floor_points_in_front_are_culled(self):
        pcl, seg = make_harness()
        rgb = np.full((len(pcl), 3), 90, np.uint8)
        cfg = C(enabled=True, zoom_views=1, zoom_wire_densify=False, zoom_max=40.0, min_length_px=1.0,
                zoom_max_attempts=1)
        ins = plan_connector_insertion(pcl, seg, rgb, cfg, make_rng(1, 2, cfg), make_library(make_asset("a")))
        keep = ~ins.drop_mask
        plan = plan_zoom_views(ins, ROTS, cfg, 1, 2)[0]
        cand = plan.candidates[0]
        # a grid of floor points one unit nearer to the camera than everything, over the window
        fwd = cand.R[2]
        c3 = cand.R.T @ np.array([cand.centre[0], cand.centre[1], 0.0])
        g = np.stack(np.meshgrid(np.linspace(-0.3, 0.3, 30), np.linspace(-0.2, 0.2, 20)), -1).reshape(-1, 2)
        floor = np.array([c3 + cand.R.T @ np.array([x, y, -0.5]) for x, y in g])
        pts = np.concatenate([pcl[keep], floor, ins.extra_pts])
        lab = np.concatenate([seg[keep], np.full(len(floor), 255), ins.extra_labels])
        col = np.concatenate([rgb[keep], np.full((len(floor), 3), 250, np.uint8), ins.extra_rgb])
        zr = ZoomRenderer(cfg, ins, pts, lab, col, P.rasterize_view, None)
        n_occ = len(zr.occluders(cand))
        self.assertGreater(n_occ, 50)
        res = zr.render(plan)
        self.assertEqual(res.info["culled_background_points"], len(zr.occluders(plan.candidates[res.info["attempt"]])))
        self.assertFalse((res.rgb == 250).all(-1).any())          # no floor splat drawn over the harness


class TestConfigAndConstants(unittest.TestCase):
    def test_env_knobs(self):
        cfg = C.from_env({"KIAT_CONNINS": "1", "KIAT_CONNINS_ZOOM_VIEWS": "3",
                          "KIAT_CONNINS_ZOOM_TARGET_FRAC_MIN": "0.2", "KIAT_CONNINS_ZOOM_ROTATION_MODE": "perturb",
                          "KIAT_CONNINS_ZOOM_WIRE_DENSIFY": "0", "KIAT_CONNINS_ZOOM_BACKDROP_MODE": "static"})
        self.assertEqual((cfg.zoom_views, cfg.zoom_target_frac_min, cfg.zoom_rotation_mode,
                          cfg.zoom_wire_densify, cfg.zoom_backdrop_mode), (3, 0.2, "perturb", False, "static"))
        self.assertEqual(C().zoom_views, 0)
        for bad in ({"zoom_min": 9.0}, {"zoom_target_frac_min": 0.9}, {"zoom_rotation_mode": "x"},
                    {"zoom_backdrop_mode": "x"}, {"zoom_fallback": "x"}, {"zoom_views": -1}):
            with self.assertRaises(ValueError):
                C(**bad)

    def test_rng_offset_unique(self):
        import re, pathlib
        src = pathlib.Path(P.__file__).parent
        used = set()
        for f in src.glob("*.py"):
            used |= {int(m) for m in re.findall(r"frame_id \+ (\d+)", f.read_text())}
        self.assertNotIn(ZOOM_RNG_OFFSET, used)
        self.assertNotEqual(ZOOM_RNG_OFFSET, 733)


if __name__ == "__main__":
    unittest.main()
