#!/usr/bin/env python3
"""Source CC0 photographic backdrops from Polyhaven for the Phase 11 dataset.

Downloads two kinds of photos:

* **Surface textures** (workshop wood, indoor fabric, concrete floor, etc.) —
  2K JPG diffuse maps from the Polyhaven *Textures* catalogue. Center-cropped
  to 4:3 and resized to 640x480 BGR JPG.
* **Indoor panoramas** (workshops, garages, kitchens, photo studios, etc.) —
  Tonemapped JPG previews of CC0 HDRIs. We crop a ~120 deg FOV slice from
  the horizon band of the equirectangular projection so the result reads
  as a single-direction wall view with minimal pole distortion.

Each new file lands in ``data/textures/backgrounds/`` with the next available
``NN_<slug>.jpg`` index, and gets a matching entry appended to
``data/textures/backgrounds/manifest.json``. A human-readable
``SOURCES.md`` summary is also written.

All Polyhaven assets are CC0 (https://polyhaven.com/license).

Usage::

    python scripts/fetch_phase11_backdrops.py
    python scripts/fetch_phase11_backdrops.py --workers 16
    python scripts/fetch_phase11_backdrops.py --dry-run   # planning only
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DST_DIR = PROJECT_ROOT / "data" / "textures" / "backgrounds"
MANIFEST_PATH = DST_DIR / "manifest.json"
SOURCES_PATH = DST_DIR / "SOURCES.md"

TARGET_W, TARGET_H = 640, 480
USER_AGENT = "kiat-crefle/phase11 (CC0 backdrop sourcing)"

# Curated slug lists (produced 2026-05-28 from the Polyhaven Textures + HDRIs
# APIs after filtering for indoor surface-style + indoor real-scene assets).
TEXTURE_SLUGS: list[str] = [
    # Wood / workshop / kitchen
    "black_painted_planks", "dark_wood", "distressed_painted_planks",
    "fine_grained_wood", "herringbone_parquet", "kitchen_wood",
    "laminate_floor", "laminate_floor_03", "oriented_strand_board",
    "plank_flooring", "plank_flooring_02", "plank_flooring_03",
    "plank_flooring_04", "raw_plank_wall",
    # Fabric / clothing / upholstery
    "bi_stretch", "book_pattern", "brown_leather", "caban",
    "cotton_jersey", "crepe_georgette", "crepe_satin",
    "curly_teddy_checkered", "curly_teddy_natural",
    "denim_fabric", "denim_fabric_03", "denim_fabric_04",
    # Concrete / workshop floor
    "anti_slip_concrete", "brushed_concrete", "brushed_concrete_2",
    "climbing_wall_02", "concrete_block_wall", "concrete_block_wall_02",
    "concrete_debris", "concrete_floor_damaged_01",
    # Metal / industrial
    "blue_metal_plate", "container_side", "corrugated_iron_02",
    "corrugated_iron_03", "metal_plate_02", "painted_metal_shutter",
    # Plaster
    "patterned_plaster_wall", "plaster_brick_pattern",
]

HDRI_SLUGS: list[str] = [
    # Workshops / shops / industrial
    "abandoned_workshop", "abandoned_workshop_02",
    "aerodynamics_workshop", "aircraft_workshop_01", "art_studio",
    "auto_service", "autoshop_01",
    "carpentry_shop_01", "carpentry_shop_02",
    "industrial_pipe_and_valve_01", "industrial_pipe_and_valve_02",
    "industrial_workshop_foundry",
    "machine_shop_01", "machine_shop_02", "machine_shop_03",
    # Garages / warehouses / depots
    "abandoned_garage", "abandoned_greenhouse", "empty_warehouse_01",
    "garage", "hangar_interior", "old_bus_depot", "old_depot",
    "burnt_warehouse",
    # Kitchens / lounges / domestic
    "blinds", "kiara_interior",
    # Photo studios (light, clean indoor surfaces — close to "desk on white")
    "brown_photostudio_01", "brown_photostudio_07",
    "blocky_photo_studio", "neon_photostudio",
    # Other indoor
    "boiler_room",
]

TEXTURE_URL_TPL = ("https://dl.polyhaven.org/file/ph-assets/"
                   "Textures/jpg/2k/{slug}/{slug}_diff_2k.jpg")
HDRI_URL_TPL    = ("https://dl.polyhaven.org/file/ph-assets/"
                   "HDRIs/extra/Tonemapped%20JPG/{slug}.jpg")
TEXTURE_PAGE_TPL = "https://polyhaven.com/a/{slug}"
HDRI_PAGE_TPL    = "https://polyhaven.com/a/{slug}"


# ── Fetch + process ────────────────────────────────────────────────────────

def _download_bytes(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _decode_jpg(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("imdecode failed")
    return img


def _center_crop_to_aspect(img: np.ndarray, ratio: float) -> np.ndarray:
    """Crop ``img`` to ``ratio`` (W/H) at the centre."""
    h, w = img.shape[:2]
    cur = w / h
    if cur > ratio:
        new_w = int(round(h * ratio))
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    new_h = int(round(w / ratio))
    y0 = (h - new_h) // 2
    return img[y0:y0 + new_h, :]


def _process_texture(buf: bytes) -> np.ndarray:
    """2K square diffuse map → 640x480 BGR (centre crop to 4:3 then resize)."""
    img = _decode_jpg(buf)
    img = _center_crop_to_aspect(img, TARGET_W / TARGET_H)
    return cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


def _process_hdri(buf: bytes) -> np.ndarray:
    """Equirectangular tonemap → 640x480 BGR.

    The equirect spans 360 deg x 180 deg. The vertical poles distort heavily,
    so we crop the horizon band [0.30, 0.85] of the height, then take a
    central horizontal slice ~ 1/3 of the equirect width — corresponds to
    a ~120 deg FOV view, similar to a wide-angle phone-camera shot.
    """
    img = _decode_jpg(buf)
    h, w = img.shape[:2]
    y0 = int(0.30 * h)
    y1 = int(0.85 * h)
    band = img[y0:y1]
    band_w = band.shape[1]
    slice_w = band_w // 3
    x0 = (band_w - slice_w) // 2
    view = band[:, x0:x0 + slice_w]
    view = _center_crop_to_aspect(view, TARGET_W / TARGET_H)
    return cv2.resize(view, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


# ── Main ───────────────────────────────────────────────────────────────────

def _existing_manifest() -> dict:
    if MANIFEST_PATH.is_file():
        return json.loads(MANIFEST_PATH.read_text())
    return {
        "description": "Photographic CC0 backgrounds for wire-harness "
                       "rendering pipeline. All images 640x480 BGR JPG.",
        "total": 0,
        "license_summary": "All assets CC0 from Poly Haven.",
        "backgrounds": [],
    }


def _next_index(manifest: dict) -> int:
    used = set()
    for e in manifest.get("backgrounds", []):
        f = e.get("file", "")
        try:
            used.add(int(f.split("_", 1)[0]))
        except (ValueError, IndexError):
            pass
    n = 1
    while n in used:
        n += 1
    return n


def fetch_one(kind: str, slug: str) -> tuple[str, str, bytes | None, str]:
    """Download + return (kind, slug, processed_jpg_bytes_or_None, source_url)."""
    url = (TEXTURE_URL_TPL if kind == "texture" else HDRI_URL_TPL).format(slug=slug)
    try:
        raw = _download_bytes(url)
        img = (_process_texture if kind == "texture" else _process_hdri)(raw)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return kind, slug, None, url
        return kind, slug, enc.tobytes(), url
    except urllib.error.HTTPError as e:
        print(f"  [HTTP {e.code}] {kind}/{slug}: {url}")
    except Exception as e:
        print(f"  [ERR ] {kind}/{slug}: {type(e).__name__}: {e}")
    return kind, slug, None, url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    DST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _existing_manifest()
    existing_slugs = {e.get("slug") for e in manifest.get("backgrounds", [])}

    plan = [("texture", s) for s in TEXTURE_SLUGS] + \
           [("hdri", s) for s in HDRI_SLUGS]
    plan = [(k, s) for k, s in plan if s not in existing_slugs]

    print(f"Planning to fetch {len(plan)} new backdrops "
          f"({sum(1 for k, _ in plan if k == 'texture')} textures + "
          f"{sum(1 for k, _ in plan if k == 'hdri')} HDRIs).")
    print(f"Existing in manifest: {len(existing_slugs)}.")
    print(f"Destination:          {DST_DIR}")

    if args.dry_run:
        for kind, slug in plan:
            print(f"  [DRY] {kind:7s} {slug}")
        return 0

    t0 = time.time()
    results: list[tuple[str, str, bytes | None, str]] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(fetch_one, k, s) for k, s in plan]
        for fut in cf.as_completed(futures):
            results.append(fut.result())

    ok = [r for r in results if r[2] is not None]
    bad = [r for r in results if r[2] is None]
    print()
    print(f"Downloaded: {len(ok)} / {len(plan)}  "
          f"(failed: {len(bad)})  wall: {time.time() - t0:.1f} s")

    # Persist
    next_idx = _next_index(manifest)
    new_entries = []
    for kind, slug, blob, url in sorted(ok, key=lambda r: (r[0], r[1])):
        fname = f"{next_idx:02d}_{slug}.jpg"
        (DST_DIR / fname).write_bytes(blob)
        page_url = (TEXTURE_PAGE_TPL if kind == "texture" else HDRI_PAGE_TPL).format(slug=slug)
        new_entries.append({
            "file": fname,
            "slug": slug,
            "kind": kind,
            "category": ("polyhaven_texture" if kind == "texture"
                         else "polyhaven_hdri_equirect_slice"),
            "description": f"{kind.title()} from Polyhaven CC0 catalogue; "
                           f"slug={slug!r}. Center-cropped to 4:3 + resized "
                           "to 640x480 BGR JPG.",
            "source_url": page_url,
            "download_url": url,
            "license": "CC0",
            "source": "Poly Haven",
        })
        next_idx += 1

    manifest["backgrounds"].extend(new_entries)
    manifest["total"] = len(manifest["backgrounds"])
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    # Compose / refresh SOURCES.md.
    lines: list[str] = []
    lines.append("# Phase 11 backdrop sources")
    lines.append("")
    lines.append("All photos here are CC0 (Public Domain) from "
                 "[Polyhaven](https://polyhaven.com/license).")
    lines.append("")
    lines.append(f"Total photos: {manifest['total']}")
    lines.append("")
    lines.append("| file | slug | kind | source page |")
    lines.append("|---|---|---|---|")
    for e in sorted(manifest["backgrounds"], key=lambda x: x.get("file", "")):
        f = e.get("file", "?")
        s = e.get("slug", "?")
        k = e.get("kind", "?")
        u = e.get("source_url", "?")
        lines.append(f"| {f} | {s} | {k} | {u} |")
    SOURCES_PATH.write_text("\n".join(lines) + "\n")

    print(f"Added {len(new_entries)} entries to {MANIFEST_PATH.name}.")
    print(f"Wrote {SOURCES_PATH.name}.")

    if bad:
        print()
        print(f"Failed slugs ({len(bad)}):")
        for kind, slug, _, _ in bad:
            print(f"  {kind:7s} {slug}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
