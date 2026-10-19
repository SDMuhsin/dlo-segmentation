"""The runtime connector library: canonical coloured point clouds on disk.

A library is a directory of ``<name>.npz`` files written by
``build_library.py``. Each file holds one asset in the CANONICAL FRAME:

* centred at the origin (bounding-box centre),
* long axis = +X, normalised so the long extent is exactly 1.0,
* cable end at -X (the end the wire enters), mating end at +X,
* Y = the wider cross-section direction, Z completes a right-handed frame.

This module only reads those files (numpy only, no trimesh), so the planner
runs in any render worker without the heavy mesh stack.
"""
from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ConnectorAsset:
    """One connector model as a canonical coloured surface point cloud."""

    name: str
    points: np.ndarray          # (N, 3) float32, canonical frame, long extent 1.0
    colors: np.ndarray          # (N, 3) uint8 BGR
    normals: np.ndarray         # (N, 3) float32 unit normals (canonical frame)
    width: float                # larger cross-section extent (robust), long extent = 1
    area: float                 # visible surface area in canonical units^2
    meta: Mapping[str, object] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])


class ConnectorLibrary:
    """An ordered, immutable set of :class:`ConnectorAsset` keyed by name."""

    def __init__(self, assets: Sequence[ConnectorAsset]):
        self._assets = {a.name: a for a in assets}

    # ----- construction ------------------------------------------------------
    @classmethod
    def load(cls, library_dir: str | Path) -> "ConnectorLibrary":
        """Load every ``*.npz`` in ``library_dir`` (sorted by name).

        Returns an empty library when the directory is missing or empty, so a
        mis-set path degrades to the no-op path instead of crashing a render.
        """
        d = Path(library_dir)
        assets = [load_asset(p) for p in sorted(d.glob("*.npz"))] if d.is_dir() else []
        return cls(assets)

    # ----- access ------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return list(self._assets)

    def __len__(self) -> int:
        return len(self._assets)

    def __getitem__(self, name: str) -> ConnectorAsset:
        return self._assets[name]

    def __iter__(self):
        return iter(self._assets.values())

    def select(self, include: Sequence[str] = (), exclude: Sequence[str] = (),
               weights: Mapping[str, float] | None = None
               ) -> tuple[list[ConnectorAsset], np.ndarray]:
        """Apply include/exclude globs and per-asset weights.

        Returns ``(assets, probabilities)``; probabilities sum to 1, or both are
        empty when nothing is selectable (all weights zero counts as nothing).
        """
        weights = weights or {}
        keep = [a for a in self._assets.values()
                if (not include or any(fnmatch.fnmatchcase(a.name, g) for g in include))
                and not any(fnmatch.fnmatchcase(a.name, g) for g in exclude)]
        w = np.array([float(weights.get(a.name, 1.0)) for a in keep], dtype=np.float64)
        if keep and w.sum() > 0:
            return keep, w / w.sum()
        return [], np.empty(0)


def load_asset(path: str | Path) -> ConnectorAsset:
    """Read one library ``.npz`` into a :class:`ConnectorAsset`."""
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z else {}
        return ConnectorAsset(
            name=p.stem,
            points=z["points"].astype(np.float32),
            colors=z["colors"].astype(np.uint8),
            normals=z["normals"].astype(np.float32),
            width=float(z["width"]),
            area=float(z["area"]),
            meta=meta,
        )


def save_asset(path: str | Path, asset: ConnectorAsset) -> None:
    """Write an asset in the on-disk format read by :func:`load_asset`."""
    np.savez_compressed(
        path,
        points=asset.points.astype(np.float32),
        colors=asset.colors.astype(np.uint8),
        normals=asset.normals.astype(np.float16),
        width=np.float64(asset.width),
        area=np.float64(asset.area),
        meta_json=np.array(json.dumps(dict(asset.meta))),
    )
