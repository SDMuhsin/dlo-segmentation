#!/usr/bin/env python3
"""Fetch the input assets the render pipeline needs.

Only code is tracked in git; data/ is rebuilt. This pulls the four asset
groups from their original public sources:

  textures     22 CC0 surface textures from ambientCG      -> data/textures/
  backgrounds  11 CC0 photo backdrops from Poly Haven      -> data/textures/backgrounds*/
  objects      21 CC0 clutter point clouds (bundled)       -> data/objects/
  pointwire    PointWire harness point clouds (TUM)        -> data/set2/

Usage:
    python scripts/download_assets.py --all          # everything, ~1.1 GB
    python scripts/download_assets.py --textures     # one group at a time
    python scripts/download_assets.py --verify       # report what is present

PointWire is published as a 50 GiB zip, but only three subdirectories per
harness are ever read. We read the archive index over HTTP range requests and
pull just those members, so the transfer is about 0.9 GiB.

Sources and licences:
  ambientCG    CC0            https://ambientcg.com
  Poly Haven   CC0            https://polyhaven.com
  PointWire    MIT, non-commercial use per the licence
               https://github.com/heti2000/cdlo-datasets
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Asset tables (taken from the manifests in the reference data/) ───────────

ACG_URL = "https://ambientcg.com/get?file={asset}_1K-JPG.zip"
ACG_FORMAT = "1K-JPG"

CLASS_TEXTURES = {
    "wire": ["Fabric017", "Fabric019", "Fabric022", "Fabric026", "Fabric030",
             "Fabric049", "Fabric070", "Plastic004", "Plastic010",
             "Rope001", "Rope002"],
    "connector": ["Plastic001", "Plastic005", "Plastic006", "Plastic012A",
                  "Plastic013A"],
    "endpoint": ["Metal003", "Metal004", "Metal006", "Metal048A"],
    # bifurcation reuses two of the connector textures, so 22 unique downloads
    "bifurcation": ["Plastic006", "Plastic012A"],
    "noise": ["Scratches001", "Scratches005"],
}

TEXTURE_BLUR_SIGMA = 2.0  # matches the reference _blurred/ files

BACKGROUNDS = [
    ("01_workbench_plywood.jpg", "plywood", "plywood_diff_2k.jpg"),
    ("02_workbench_wood_planks.jpg", "wood_planks", "wood_planks_diff_2k.jpg"),
    ("03_wood_planks_dry.jpg", "wooden_planks", "wooden_planks_diff_2k.jpg"),
    ("04_concrete_floor_worn.jpg", "concrete_floor_worn_001",
     "concrete_floor_worn_001_diff_2k.jpg"),
    ("05_concrete_weathered.jpg", "worn_concrete_floor",
     "worn_concrete_floor_diff_2k.jpg"),
    ("06_concrete_granular.jpg", "concrete_floor_01",
     "concrete_floor_01_diff_2k.jpg"),
    ("07_fabric_rough_linen.jpg", "rough_linen", "rough_linen_diff_2k.jpg"),
    ("08_fabric_denim.jpg", "denim_fabric", "denim_fabric_diff_2k.jpg"),
    ("09_tile_white_long.jpg", "long_white_tiles",
     "long_white_tiles_diff_2k.jpg"),
    ("10_tile_marble_cream.jpg", "marble_01", "marble_01_diff_2k.jpg"),
    ("11_tile_floor_06.jpg", "floor_tiles_06", "floor_tiles_06_diff_2k.jpg"),
]
PH_TEXTURE_URL = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/{slug}/{fname}"
BG_W, BG_H = 640, 480

POINTWIRE_URL = ("https://nextcloud.in.tum.de/public.php/webdav/"
                 "pointwire_data.zip")
POINTWIRE_SHARE = "7ooyYxoP6HyPXQK"
POINTWIRE_SUBDIRS = ("pointclouds_normed_4096", "segmentation_normed_4096",
                     "skeletons")

# The harnesses used by the August render: 21 training + 3 held out. Not the
# whole archive, which has 40. The reference data/ has 28 of them, and one
# (005) is incomplete so the renderer's discovery pass skips it. Pulling more
# sets than this adds source frames and gives a bigger dataset than the one
# behind the reported numbers.
TRAIN_SETS_USED = ["000", "002", "003", "004", "006", "008", "009", "011",
                   "012", "014", "015", "016", "021", "022", "023", "024",
                   "025", "027", "029", "030", "031"]
VAL_SETS_USED = ["032", "034", "035"]
SETS_USED = TRAIN_SETS_USED + VAL_SETS_USED
FRAMES_PER_SET = 300

BUNDLED_OBJECTS = PROJECT_ROOT / "assets" / "phase4_objects"


def log(msg: str) -> None:
    print(msg, flush=True)


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def fetch(url: str, dest: Path, retries: int = 4, auth: str | None = None) -> Path:
    """Download `url` to `dest` with retries. Skips if already present."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": "wire-seg-research/1.0 (dataset reproduction)"}
    if auth:
        headers["Authorization"] = f"Basic {auth}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as r, \
                    open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh, length=1 << 20)
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            wait = 2 ** attempt
            log(f"      retry {attempt + 1}/{retries} after {wait}s ({exc})")
            time.sleep(wait)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download {url}: {last}")


# ── Group 1: ambientCG textures ─────────────────────────────────────────────

def do_textures(data_root: Path, workdir: Path) -> None:
    tex_root = data_root / "textures"
    raw, blurred = tex_root / "_raw", tex_root / "_blurred"
    raw.mkdir(parents=True, exist_ok=True)
    blurred.mkdir(parents=True, exist_ok=True)

    assets = sorted({a for v in CLASS_TEXTURES.values() for a in v})
    log(f"[textures] {len(assets)} CC0 assets from ambientCG")

    for i, asset in enumerate(assets, 1):
        out_raw = raw / f"{asset}.jpg"
        out_blur = blurred / f"{asset}_blur.jpg"
        if out_raw.exists() and out_blur.exists():
            log(f"  [{i:2}/{len(assets)}] {asset}: present")
            continue
        zpath = workdir / f"{asset}_{ACG_FORMAT}.zip"
        log(f"  [{i:2}/{len(assets)}] {asset}: downloading")
        fetch(ACG_URL.format(asset=asset), zpath)
        with zipfile.ZipFile(zpath) as zf:
            colour = [n for n in zf.namelist() if n.endswith("_Color.jpg")]
            if not colour:
                raise RuntimeError(f"{asset}: no _Color.jpg inside the archive")
            with zf.open(colour[0]) as src, open(out_raw, "wb") as dst:
                shutil.copyfileobj(src, dst)
        zpath.unlink(missing_ok=True)

        img = cv2.imread(str(out_raw), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"{asset}: colour map did not decode")
        cv2.imwrite(str(out_blur),
                    cv2.GaussianBlur(img, (0, 0), TEXTURE_BLUR_SIGMA))

    # class folders are symlinks into _blurred/, matching the reference layout
    for cls, ids in CLASS_TEXTURES.items():
        d = tex_root / cls
        d.mkdir(exist_ok=True)
        for asset in ids:
            link, target = d / f"{asset}_blur.jpg", blurred / f"{asset}_blur.jpg"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(os.path.relpath(target, d), link)

    manifest = {
        "license": "CC0 (Public Domain)",
        "source": "ambientCG.com",
        "downloaded": time.strftime("%Y-%m-%d"),
        **{cls: [f"{a}_blur.jpg" for a in ids]
           for cls, ids in CLASS_TEXTURES.items()},
        "failed": [],
        "details": {a: {"url": f"https://ambientcg.com/view?id={a}",
                        "format": ACG_FORMAT} for a in assets},
        "blur": {"kernel": "gaussian", "sigma": TEXTURE_BLUR_SIGMA},
    }
    (tex_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"[textures] done -> {tex_root}")


# ── Group 2: Poly Haven backdrops ───────────────────────────────────────────

def crop_backdrop(img: np.ndarray) -> np.ndarray:
    """Centre 4:3 crop, then area-resize to 640x480 (the reference recipe)."""
    h, w = img.shape[:2]
    ch = int(round(w * BG_H / BG_W))
    if ch <= h:
        y = (h - ch) // 2
        img = img[y:y + ch, :]
    else:
        cw = int(round(h * BG_W / BG_H))
        x = (w - cw) // 2
        img = img[:, x:x + cw]
    return cv2.resize(img, (BG_W, BG_H), interpolation=cv2.INTER_AREA)


def do_backgrounds(data_root: Path, workdir: Path) -> None:
    bg = data_root / "textures" / "backgrounds"
    orig11 = data_root / "textures" / "backgrounds_p4orig11"
    bg.mkdir(parents=True, exist_ok=True)
    orig11.mkdir(parents=True, exist_ok=True)
    log(f"[backgrounds] {len(BACKGROUNDS)} CC0 photos from Poly Haven")

    entries = []
    for i, (name, slug, fname) in enumerate(BACKGROUNDS, 1):
        out = bg / name
        url = PH_TEXTURE_URL.format(slug=slug, fname=fname)
        if not out.exists():
            log(f"  [{i:2}/{len(BACKGROUNDS)}] {name}: downloading")
            src = fetch(url, workdir / fname)
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"{name}: source did not decode")
            cv2.imwrite(str(out), crop_backdrop(img))
            src.unlink(missing_ok=True)
        else:
            log(f"  [{i:2}/{len(BACKGROUNDS)}] {name}: present")
        entries.append({"file": name, "slug": slug, "download_url": url,
                        "license": "CC0", "source": "Poly Haven",
                        "kind": "polyhaven_texture_orig"})
        link = orig11 / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.relpath(out, orig11), link)

    (bg / "manifest.json").write_text(json.dumps(
        {"description": f"CC0 photographic backdrops, {BG_W}x{BG_H} BGR JPG.",
         "total": len(entries), "license_summary": "All CC0 from Poly Haven.",
         "backgrounds": entries}, indent=2))
    log(f"[backgrounds] done -> {orig11} ({len(entries)} photos)")


# ── Group 3: bundled clutter point clouds ───────────────────────────────────

def do_objects(data_root: Path) -> None:
    dst = data_root / "objects"
    if not BUNDLED_OBJECTS.is_dir():
        raise RuntimeError(
            f"bundled objects missing: {BUNDLED_OBJECTS}\n"
            "These are derived point clouds with no download source; "
            "they are committed to the repo.")
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(BUNDLED_OBJECTS.iterdir()):
        if src.name.endswith(".npz") or src.name == "manifest.json":
            shutil.copy2(src, dst / src.name)
            n += src.name.endswith(".npz")
    log(f"[objects] copied {n} CC0 point clouds -> {dst}")


# ── Group 4: PointWire, via HTTP range reads into the remote zip ─────────────

class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with a block cache."""

    def __init__(self, url: str, auth: str, block: int = 4 << 20,
                 cache_blocks: int = 64):
        self.url, self.auth, self.block = url, auth, block
        self.cache_blocks = cache_blocks
        self._cache: dict[int, bytes] = {}
        self._order: list[int] = []
        self.bytes_fetched = 0
        req = urllib.request.Request(
            url, method="HEAD", headers={"Authorization": f"Basic {auth}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def _block(self, bi: int) -> bytes:
        hit = self._cache.get(bi)
        if hit is not None:
            return hit
        start = bi * self.block
        end = min(start + self.block, self.size) - 1
        for attempt in range(4):
            try:
                req = urllib.request.Request(
                    self.url, headers={"Authorization": f"Basic {self.auth}",
                                       "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                break
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                if attempt == 3:
                    raise RuntimeError(f"range read failed at {start}: {exc}")
                time.sleep(2 ** attempt)
        self.bytes_fetched += len(data)
        self._cache[bi] = data
        self._order.append(bi)
        while len(self._order) > self.cache_blocks:
            self._cache.pop(self._order.pop(0), None)
        return data

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = off
        elif whence == io.SEEK_CUR:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            bi = self.pos // self.block
            data = self._block(bi)
            off = self.pos - bi * self.block
            take = min(n, len(data) - off)
            out += data[off:off + take]
            self.pos += take
            n -= take
        return bytes(out)


def do_pointwire(data_root: Path, sets: list[str]) -> None:
    auth = base64.b64encode(f"{POINTWIRE_SHARE}:".encode()).decode()
    log(f"[pointwire] opening remote archive ({len(sets)} harnesses needed)")
    remote = HTTPRangeFile(POINTWIRE_URL, auth)
    log(f"[pointwire] archive is {human(remote.size)}; reading index")
    zf = zipfile.ZipFile(remote)

    wanted = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = info.filename.split("/")
        if len(parts) < 4 or parts[0] != "set2":
            continue
        if parts[1] in sets and parts[2] in POINTWIRE_SUBDIRS:
            wanted.append(info)
    if not wanted:
        raise RuntimeError("no matching members found in the archive")

    total = sum(i.compress_size for i in wanted)
    log(f"[pointwire] {len(wanted):,} files, {human(total)} compressed "
        f"(vs {human(remote.size)} for the whole archive)")

    # extract in archive order so the range reads stay sequential
    wanted.sort(key=lambda i: i.header_offset)
    done = 0
    for info in wanted:
        dest = data_root / info.filename  # data/set2/<set>/<subdir>/<file>
        if dest.exists() and dest.stat().st_size == info.file_size:
            done += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        done += 1
        if done % 2000 == 0 or done == len(wanted):
            log(f"    {done:,}/{len(wanted):,} files "
                f"({human(remote.bytes_fetched)} transferred)")
    log(f"[pointwire] done -> {data_root / 'set2'}")


# ── Verification ────────────────────────────────────────────────────────────

def do_verify(data_root: Path) -> bool:
    log("[verify] checking against what the renderer requires")
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        ok &= cond
        log(f"  [{'OK' if cond else 'FAIL'}] {msg}")

    tex_root = data_root / "textures"
    n_tex = sum(len(list((tex_root / c).glob("*.jpg")))
                for c in CLASS_TEXTURES)
    check(n_tex >= 20, f"class textures: {n_tex} (renderer needs >= 20)")

    n_bg = len(list((data_root / "textures" / "backgrounds_p4orig11")
                    .glob("*.jpg")))
    check(n_bg == 11, f"backdrop photos: {n_bg} (phase13 needs exactly 11)")

    mpath = data_root / "objects" / "manifest.json"
    if mpath.is_file():
        objs = json.loads(mpath.read_text()).get("objects", [])
        present = [o for o in objs
                   if (data_root / "objects" / o["file"]).is_file()]
        clutter = [o for o in present if o.get("category") not in {
            "hand", "gripper", "arm", "negative_wire_like", "rope", "clutter"}]
        check(len(clutter) >= 15,
              f"clutter point clouds: {len(clutter)} (renderer needs >= 15, "
              "expects 21)")
    else:
        check(False, f"objects manifest missing: {mpath}")

    complete = []
    for sid in SETS_USED:
        d = data_root / "set2" / sid
        counts = [len(list((d / s).glob("*"))) if (d / s).is_dir() else 0
                  for s in POINTWIRE_SUBDIRS]
        if min(counts) >= FRAMES_PER_SET:
            complete.append(sid)
    check(len(complete) == len(SETS_USED),
          f"complete harnesses: {len(complete)}/{len(SETS_USED)} "
          f"({FRAMES_PER_SET} frames each)")
    if len(complete) == len(SETS_USED):
        log(f"       -> renders {len(TRAIN_SETS_USED) * 60:,} train + "
            f"{len(VAL_SETS_USED) * 60:,} val source frames "
            f"= {len(SETS_USED) * 60:,} at stride 5")

    log(f"[verify] {'all checks passed' if ok else 'INCOMPLETE'}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(
        description="Download the render pipeline's input assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--all", action="store_true", help="every group")
    p.add_argument("--textures", action="store_true")
    p.add_argument("--backgrounds", action="store_true")
    p.add_argument("--objects", action="store_true")
    p.add_argument("--pointwire", action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="report what is present; download nothing")
    p.add_argument("--data-root", type=Path,
                   default=PROJECT_ROOT / "data")
    p.add_argument("--sets", nargs="+", default=None,
                   help="harness ids for --pointwire (default: the 24 the "
                        "August render used)")
    args = p.parse_args()

    groups = ("textures", "backgrounds", "objects", "pointwire")
    selected = {g: (args.all or getattr(args, g)) for g in groups}
    if not any(selected.values()) and not args.verify:
        p.print_help()
        return 1

    data_root = args.data_root.resolve()
    workdir = data_root / "_downloads"
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        if selected["textures"]:
            do_textures(data_root, workdir)
        if selected["backgrounds"]:
            do_backgrounds(data_root, workdir)
        if selected["objects"]:
            do_objects(data_root)
        if selected["pointwire"]:
            do_pointwire(data_root, args.sets or SETS_USED)
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        return 2
    finally:
        if workdir.is_dir() and not any(workdir.iterdir()):
            workdir.rmdir()

    if args.verify or any(selected.values()):
        return 0 if do_verify(data_root) else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
