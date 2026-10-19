"""Library builder CLI: convert every model in a folder to a canonical .npz.

    cd /workspace/kiat_crefle
    PYTHONPATH=src ./env/bin/python -m connector_insertion.build_library

Adding a model = drop the file into ``data/connector_models_3d/`` and re-run.
Existing library files are skipped unless ``--force`` is given or the model file
or its manifest entry is newer. ``manifest.json`` is created/updated: new files
get a stub entry (licence fields ``null``) that a human should fill in, existing
entries are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .assets import (MESH_SUFFIXES, canonicalise, load_parts, outer_shell_mask,
                     sample_coloured_surface)
from .library import ConnectorAsset, save_asset

MANIFEST_NAME = "manifest.json"


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"assets": {}}


def build_one(model: Path, out_dir: Path, entry: dict, n_samples: int,
              max_store: int, shell: bool, n_views: int, seed: int) -> dict:
    """Build one asset; returns a stats dict (also stored in the npz meta)."""
    t0 = time.time()
    parts = load_parts(model)
    pts, rgb, nrm = sample_coloured_surface(parts, n_samples, seed=seed)
    pts, nrm, info = canonicalise(pts, nrm, cable_end=entry.get("cable_end", "auto"))
    n_raw = len(pts)
    if shell:
        keep = outer_shell_mask(pts, n_views=n_views)
        pts, rgb, nrm = pts[keep], rgb[keep], nrm[keep]
    n_shell = len(pts)
    if n_shell > max_store:
        sel = np.random.RandomState(seed).choice(n_shell, max_store, replace=False)
        pts, rgb, nrm = pts[sel], rgb[sel], nrm[sel]
    # visible surface area (canonical units^2) = area-uniform fraction kept x mesh area / L^2
    area = float(sum(m.area for m in parts)) / info["length_source_units"] ** 2
    area *= n_shell / max(n_raw, 1)
    stats = dict(info, name=model.stem, source_file=model.name, n_parts=len(parts),
                 n_sampled=n_raw, n_after_shell=n_shell, n_stored=len(pts),
                 shell_filter=shell, build_seconds=round(time.time() - t0, 1))
    save_asset(out_dir / f"{model.stem}.npz", ConnectorAsset(
        name=model.stem, points=pts.astype(np.float32),
        colors=rgb[:, ::-1].copy(),              # RGB -> BGR
        normals=nrm.astype(np.float32), width=info["width"], area=area, meta=stats))
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default="data/connector_models_3d", help="folder with model files")
    ap.add_argument("--out", default=None, help="library dir (default <src>/_library)")
    ap.add_argument("--samples", type=int, default=400_000, help="surface samples per model")
    ap.add_argument("--max-store", type=int, default=150_000, help="max points stored per model")
    ap.add_argument("--views", type=int, default=32, help="outer-shell filter view directions")
    ap.add_argument("--no-shell", action="store_true", help="keep hidden interior samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="rebuild even if up to date")
    ap.add_argument("--only", nargs="*", default=None, help="model stems to build")
    a = ap.parse_args(argv)

    src = Path(a.src)
    out = Path(a.out) if a.out else src / "_library"
    out.mkdir(parents=True, exist_ok=True)
    man_path = src / MANIFEST_NAME
    manifest = _load_manifest(man_path)
    manifest.setdefault("assets", {})
    manifest_mtime = man_path.stat().st_mtime if man_path.exists() else 0.0

    models = sorted(p for p in src.iterdir() if p.suffix.lower() in MESH_SUFFIXES)
    for m in models:
        if a.only and m.stem not in a.only:
            continue
        ent = manifest["assets"].setdefault(m.stem, {
            "file": m.name, "source_url": None, "creator": None,
            "license": None, "license_verified": False, "cable_end": "auto", "notes": ""})
        target = out / f"{m.stem}.npz"
        if (target.exists() and not a.force
                and target.stat().st_mtime > max(m.stat().st_mtime, manifest_mtime)):
            print(f"[skip] {m.name} (library up to date)")
            continue
        print(f"[build] {m.name} ...", flush=True)
        st = build_one(m, out, ent, a.samples, a.max_store, not a.no_shell, a.views, a.seed)
        print("        parts=%d sampled=%d shell=%d stored=%d width=%.3f cable=%s (%.0fs)" % (
            st["n_parts"], st["n_sampled"], st["n_after_shell"], st["n_stored"],
            st["width"], st["cable_end_method"], st["build_seconds"]))
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"library: {out}  ({len(list(out.glob('*.npz')))} assets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
