"""Synthetic fixtures shared by the tests (no files, no mesh stack needed)."""
import numpy as np

from connector_insertion.library import ConnectorAsset, ConnectorLibrary


def make_asset(name="cyl", width=0.3, n=6000, seed=0) -> ConnectorAsset:
    """Canonical cylinder: length 1 along X (cable end -X), diameter ``width``."""
    rs = np.random.RandomState(seed)
    x = rs.uniform(-0.5, 0.5, n)
    th = rs.uniform(0, 2 * np.pi, n)
    r = width / 2
    p = np.stack([x, r * np.cos(th), r * np.sin(th)], 1)
    nrm = np.stack([np.zeros(n), np.cos(th), np.sin(th)], 1)
    col = np.zeros((n, 3), np.uint8)
    col[:, 0] = (255 * (x + 0.5)).astype(np.uint8)       # blue ramps toward mating end
    col[:, 2] = 200
    area = np.pi * width * 1.0
    return ConnectorAsset(name, p.astype(np.float32), col, nrm.astype(np.float32),
                          width=width, area=area)


def make_library(*assets) -> ConnectorLibrary:
    return ConnectorLibrary(list(assets) or [make_asset()])


def make_harness(length_px=18.0, width_px=8.0, px=480 / 2.2, seed=0):
    """A wire along +X ending in a connector cluster (class 3) of known size.

    The connector occupies x in [0, L]; the wire starts at x = L and runs on.
    Returns (pcl, seg). The slot axis toward the wire must come out as +X.
    """
    rs = np.random.RandomState(seed)
    L, W = length_px / px, width_px / px
    n = 120
    conn = np.stack([rs.uniform(0, L, n), rs.uniform(-W / 2, W / 2, n),
                     rs.uniform(-W / 4, W / 4, n)], 1)
    wire = np.stack([L + np.linspace(0.002, 0.6, 400), np.zeros(400), np.zeros(400)], 1)
    pcl = np.concatenate([conn, wire]).astype(np.float64)
    seg = np.concatenate([np.full(n, 3), np.zeros(400, int)])
    return pcl, seg
