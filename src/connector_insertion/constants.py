"""Constants shared by the planner and the rasteriser.

Kept free of heavy imports (no cv2, trimesh, scipy) because ``src/pcl_to_rgbd.py``
imports this module at load time and must stay cheap and cycle-free.
"""

#: Per-point label given to every point this package inserts. The rasteriser
#: (``pcl_to_rgbd.rasterize_view``) maps it to PNG label 4 (connector, i.e.
#: source class 3 + 1). It is defined ONLY here. The value must never collide
#: with the source classes 0..4, the background sentinel 255 (``BG_LABEL`` in
#: texture_mapping) or negative values; ``tests/test_constants.py`` checks it.
INSERTED_CONNECTOR_LABEL: int = 103

#: Source (PointWire) class id of a connector point in the harness ``seg`` array.
SOURCE_CONNECTOR_CLASS: int = 3

#: Source classes that are wire-like (wire body, endpoint, bifurcation). Used to
#: orient a slot toward the wire that is attached to it.
WIRE_LIKE_CLASSES: tuple[int, ...] = (0, 1, 2)

#: Orthographic scale of the renderer in pixels per world unit
#: (``IMG_H / (2 * FRUSTUM_HALF)`` = 480 / 2.2 in ``pcl_to_rgbd.py``). Duplicated
#: here so the package stays standalone; ``tests/test_constants.py`` asserts it
#: equals ``pcl_to_rgbd.SCALE``.
PX_PER_UNIT: float = 480.0 / 2.2

#: Offset added to ``set_id * 1000 + frame_id`` for the planner's private RNG.
#: grep of src/ shows 7, 113, 233, 401, 501, 502, 517, 601, 602, 619, 877, 911,
#: 1217 are taken; 733 is free.
RNG_OFFSET: int = 733

#: Offset for the zoom-view RandomState (set_id * 1000 + frame_id + offset),
#: drawn only when ``zoom_views > 0``. grep of src/ (frame_id + N) shows 7, 113,
#: 233, 401, 501, 502, 517, 601, 602, 619, 877, 911, 1217 and 733 taken; 1531 is free.
ZOOM_RNG_OFFSET: int = 1531
