"""Configuration of the 3D connector insertion stage.

Every knob lives in the frozen dataclass :class:`ConnectorInsertionConfig`.
It can be built three ways: directly, ``from_dict`` / ``from_json`` (partner
teams), or ``from_env`` (the renderer; env vars ``KIAT_CONNINS_*``). The master
switch ``KIAT_CONNINS`` defaults to OFF so the renderer is byte-identical to
before unless a run opts in.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from .constants import PX_PER_UNIT

FIT_MODES = ("length", "width", "min", "max", "geomean")
ROLL_MODES = ("random", "fixed", "none")
ZOOM_ROTATION_MODES = ("canonical", "perturb")
ZOOM_BACKDROP_MODES = ("crop", "static")
ZOOM_FALLBACKS = ("skip", "wire")

#: Default library location relative to the repo root.
DEFAULT_LIBRARY_DIR = "data/connector_models_3d/_library"


def _truthy(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ConnectorInsertionConfig:
    """All knobs of the insertion stage. Defaults reproduce the PI spec.

    Env var for each field: ``KIAT_CONNINS_<FIELD_NAME_UPPERCASE>`` (the master
    switch ``enabled`` is plain ``KIAT_CONNINS``). Lists are comma separated,
    the weights map is ``name=weight,name=weight``.
    """

    # --- master switch and assets -------------------------------------------
    enabled: bool = False
    library_dir: str = DEFAULT_LIBRARY_DIR
    include: tuple[str, ...] = ()        # fnmatch globs; empty = all assets
    exclude: tuple[str, ...] = ()        # fnmatch globs removed after include
    weights: Mapping[str, float] = field(default_factory=dict)  # name -> weight, default 1

    # --- which slots get replaced -------------------------------------------
    replace_prob: float = 1.0            # per slot, independent draws
    max_per_frame: int = 24              # cap on replaced slots per source (largest first); set 008 has 17

    # --- size fitting --------------------------------------------------------
    fit_mode: str = "length"             # which slot measure the asset is fitted to
    scale_mult: float = 1.0              # global multiplier on the fitted size
    scale_jitter: float = 0.0            # +- fraction, uniform
    min_length_px: float = 12.0          # clamp on final connector length, pixel-equivalents
    max_length_px: float = 48.0
    px_per_unit: float = PX_PER_UNIT     # world units -> px (renderer SCALE)

    # --- orientation ---------------------------------------------------------
    roll_mode: str = "random"            # about the connector axis
    roll_deg: float = 0.0                # used by roll_mode=fixed (0 = flat side along slot's wide axis)
    tilt_jitter_deg: float = 0.0         # max random tilt of the axis, about the cable face
    flip_prob: float = 0.0               # mirror probability
    wire_overlap_frac: float = 0.0       # fraction of connector length pushed into the wire

    # --- colour --------------------------------------------------------------
    wire_tint: float = 0.0               # 0 native asset colours, 1 tinted to frame wire colour
    asset_color_jitter: float = 0.0      # +- fraction brightness / per-channel gain

    # --- point sampling ------------------------------------------------------
    spacing_px: float = 0.5              # target splat spacing at the final size
    min_points: int = 300
    max_points: int = 8000

    # --- slot detection ------------------------------------------------------
    cluster_eps: float = 0.05            # DBSCAN eps, world units
    min_cluster_points: int = 8
    split_large: bool = False            # split merged clusters longer than split_length_px
    split_length_px: float = 40.0

    # --- rasterisation -------------------------------------------------------
    use_connector_disc: bool = False     # True: inserted points splat with KIAT_CONNECTOR_SCALE disc

    seed_salt: int = 0                   # added to the planner seed

    # --- zoom views (close-ups centred on a connector; 0 = OFF) ---------------
    zoom_views: int = 0                  # N extra close-up views per source, 640x480 like the rest
    zoom_target_frac_min: float = 0.10   # connector projected length / frame height, drawn U(min, max)
    zoom_target_frac_max: float = 0.45
    zoom_min: float = 1.5                # clamp on the zoom factor (px per unit = 218.18 x zoom)
    zoom_max: float = 8.0
    zoom_centre_jitter: float = 0.08     # +- fraction of frame height, camera plane
    zoom_centre_bias: float = 0.3        # frame centre along the connector, 0 = cable end, 1 = tip
    zoom_rotation_mode: str = "canonical"  # canonical | perturb (canonical + small random rotation)
    zoom_perturb_deg: float = 15.0       # max rotation for zoom_rotation_mode=perturb
    zoom_min_projected_frac: float = 0.6  # skip views that foreshorten the connector below this fraction of its best view
    zoom_max_attempts: int = 3           # candidate views tried when the connector is occluded
    zoom_min_visible_frac: float = 0.6   # a candidate is accepted when this share of the connector is unoccluded
    zoom_min_connector_px: int = 100     # ... and shows at least this many connector px
    zoom_wire_densify: bool = True       # replace harness wire points by a smooth tube in zoom views
    zoom_wire_step_px: float = 1.5       # tube point spacing in zoom-view pixels
    zoom_wire_splat_radius: int = 2      # splat radius of tube points (px)
    zoom_bg_radius_mult: float = 1.0     # background/harness splat radius = ceil(zoom x mult)
    zoom_backdrop_mode: str = "crop"     # crop (match the camera window) | static (unzoomed photo)
    zoom_connector_spacing_px: float = 0.7  # resampled connector point spacing in zoom-view pixels
    zoom_max_points: int = 150000        # point cap per resampled connector (library has 150k per asset)
    zoom_connector_radius: int = 1       # splat radius of inserted connector points in zoom views
    zoom_exclude_views: tuple[str, ...] = ("bottom",)  # canonical views never used as the zoom camera base
    zoom_fallback: str = "skip"          # sources without a slot: skip | wire (random wire location)

    def __post_init__(self) -> None:
        def chk(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(f"ConnectorInsertionConfig: {msg}")
        chk(0.0 <= self.replace_prob <= 1.0, "replace_prob must lie in [0, 1]")
        chk(self.max_per_frame >= 1, "max_per_frame must be >= 1")
        chk(self.fit_mode in FIT_MODES, f"fit_mode must be one of {FIT_MODES}")
        chk(self.scale_mult > 0, "scale_mult must be > 0")
        chk(0.0 <= self.scale_jitter < 1.0, "scale_jitter must lie in [0, 1)")
        chk(0 < self.min_length_px <= self.max_length_px,
            "need 0 < min_length_px <= max_length_px")
        chk(self.px_per_unit > 0, "px_per_unit must be > 0")
        chk(self.roll_mode in ROLL_MODES, f"roll_mode must be one of {ROLL_MODES}")
        chk(0.0 <= self.tilt_jitter_deg <= 45.0, "tilt_jitter_deg must lie in [0, 45]")
        chk(0.0 <= self.flip_prob <= 1.0, "flip_prob must lie in [0, 1]")
        chk(0.0 <= self.wire_overlap_frac <= 0.5, "wire_overlap_frac must lie in [0, 0.5]")
        chk(0.0 <= self.wire_tint <= 1.0, "wire_tint must lie in [0, 1]")
        chk(0.0 <= self.asset_color_jitter < 1.0, "asset_color_jitter must lie in [0, 1)")
        chk(self.spacing_px > 0, "spacing_px must be > 0")
        chk(1 <= self.min_points <= self.max_points, "need 1 <= min_points <= max_points")
        chk(self.cluster_eps > 0, "cluster_eps must be > 0")
        chk(self.min_cluster_points >= 3, "min_cluster_points must be >= 3 (PCA needs 3 points)")
        chk(self.split_length_px > 0, "split_length_px must be > 0")
        chk(self.zoom_views >= 0, "zoom_views must be >= 0")
        chk(0.0 < self.zoom_target_frac_min <= self.zoom_target_frac_max <= 1.0,
            "need 0 < zoom_target_frac_min <= zoom_target_frac_max <= 1")
        chk(1.0 <= self.zoom_min <= self.zoom_max, "need 1 <= zoom_min <= zoom_max")
        chk(0.0 <= self.zoom_centre_jitter <= 0.5, "zoom_centre_jitter must lie in [0, 0.5]")
        chk(0.0 <= self.zoom_centre_bias <= 1.0, "zoom_centre_bias must lie in [0, 1]")
        chk(self.zoom_rotation_mode in ZOOM_ROTATION_MODES,
            f"zoom_rotation_mode must be one of {ZOOM_ROTATION_MODES}")
        chk(0.0 <= self.zoom_perturb_deg <= 45.0, "zoom_perturb_deg must lie in [0, 45]")
        chk(0.0 <= self.zoom_min_projected_frac <= 1.0, "zoom_min_projected_frac must lie in [0, 1]")
        chk(self.zoom_max_attempts >= 1, "zoom_max_attempts must be >= 1")
        chk(0.0 <= self.zoom_min_visible_frac <= 1.0, "zoom_min_visible_frac must lie in [0, 1]")
        chk(self.zoom_min_connector_px >= 0, "zoom_min_connector_px must be >= 0")
        chk(self.zoom_wire_step_px > 0, "zoom_wire_step_px must be > 0")
        chk(self.zoom_wire_splat_radius >= 1, "zoom_wire_splat_radius must be >= 1")
        chk(self.zoom_bg_radius_mult > 0, "zoom_bg_radius_mult must be > 0")
        chk(self.zoom_backdrop_mode in ZOOM_BACKDROP_MODES,
            f"zoom_backdrop_mode must be one of {ZOOM_BACKDROP_MODES}")
        chk(self.zoom_connector_spacing_px > 0, "zoom_connector_spacing_px must be > 0")
        chk(self.zoom_max_points >= 1, "zoom_max_points must be >= 1")
        chk(self.zoom_connector_radius >= 1, "zoom_connector_radius must be >= 1")
        chk(self.zoom_fallback in ZOOM_FALLBACKS, f"zoom_fallback must be one of {ZOOM_FALLBACKS}")
        for k, v in dict(self.weights).items():
            chk(v >= 0, f"weight for {k!r} must be >= 0")
        # normalise containers so the dataclass stays hashable-ish and immutable
        object.__setattr__(self, "include", tuple(self.include))
        object.__setattr__(self, "exclude", tuple(self.exclude))
        object.__setattr__(self, "zoom_exclude_views", tuple(self.zoom_exclude_views))
        object.__setattr__(self, "weights", dict(self.weights))

    # ----- construction -------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ConnectorInsertionConfig":
        """Build from a plain dict. Unknown keys raise (typos must not pass silently)."""
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown ConnectorInsertionConfig keys: {sorted(unknown)}")
        return cls(**dict(d))

    @classmethod
    def from_json(cls, path_or_text: str | os.PathLike) -> "ConnectorInsertionConfig":
        """Build from a JSON file path or a JSON string."""
        s = str(path_or_text)
        text = Path(s).read_text() if not s.lstrip().startswith("{") else s
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "ConnectorInsertionConfig":
        """Build from ``KIAT_CONNINS_*`` env vars. Unset vars keep the defaults."""
        env = os.environ if env is None else env
        d: dict[str, Any] = {}
        if "KIAT_CONNINS" in env:
            d["enabled"] = _truthy(env["KIAT_CONNINS"])
        for f in dataclasses.fields(cls):
            if f.name == "enabled":
                continue
            key = "KIAT_CONNINS_" + f.name.upper()
            if key not in env or env[key].strip() == "":
                continue
            raw = env[key].strip()
            default = f.default if f.default is not dataclasses.MISSING else None
            if f.name in ("include", "exclude", "zoom_exclude_views"):
                d[f.name] = tuple(x.strip() for x in raw.split(",") if x.strip())
            elif f.name == "weights":
                w = {}
                for item in raw.split(","):
                    k, _, v = item.partition("=")
                    w[k.strip()] = float(v)
                d[f.name] = w
            elif isinstance(default, bool):
                d[f.name] = _truthy(raw)
            elif isinstance(default, int):
                d[f.name] = int(raw)
            elif isinstance(default, float):
                d[f.name] = float(raw)
            else:
                d[f.name] = raw
        return cls(**d)

    # ----- export -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict (round-trips through ``from_dict``)."""
        d = dataclasses.asdict(self)
        d["include"], d["exclude"] = list(self.include), list(self.exclude)
        d["zoom_exclude_views"] = list(self.zoom_exclude_views)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
