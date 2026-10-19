"""Render the zoom views of ONE source and write PNGs for inspection.

    cd /workspace/kiat_crefle
    PYTHONPATH=src ./env/bin/python -m connector_insertion.zoom_preview \\
        --set 0 --frame 5 [--zoom 4 | --target-frac 0.3] [--asset usb-a_tip] [--views 3]

Runs the source through the real renderer (``convert_one_video``) with the README
recipe environment, the insertion stage and ``zoom_views`` on, into a temp folder,
then writes, per zoom view j, to ``--out`` (default ``results/connector_insertion/preview_zoom/single``):

    set{S}_{F}_zoom{j}_rgb.png        full-resolution RGB (640x480)
    set{S}_{F}_zoom{j}_overlay.png    RGB with labels (wire green, connector red)
    set{S}_{F}_zoom{j}_ref.png        the standard view the zoom view derives from
                                      (planned window outlined)

``--zoom K`` pins the zoom factor (zoom_min = zoom_max = K); ``--target-frac X`` pins the
connector length to X of the frame height (clamped to [zoom_min, zoom_max] = [1, 16]
unless ``--zoom`` is given). Extra ``KIAT_CONNINS_*`` env vars are honoured.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECIPE = {"KIAT_DATASET_MODE": "phase13", "KIAT_BG_DIR": "data/textures/backgrounds_p4orig11",
          "KIAT_DLO_UNICOLOR": "1", "KIAT_CONNECTOR_SCALE": "3.0", "KIAT_P13_LIGHTING": "1",
          "KIAT_CONNINS": "1", "KIAT_CONNINS_SCALE_MULT": "2.0"}


def overlay(rgb, lab):
    import numpy as np
    o = rgb.astype(np.float32)
    wire = np.isin(lab, (1, 2, 3))
    con = lab == 4
    o[wire] = 0.5 * o[wire] + 0.5 * np.array([0, 255, 0])
    o[con] = 0.4 * o[con] + 0.6 * np.array([0, 0, 255])
    return o.astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", type=int, required=True, dest="set_id")
    ap.add_argument("--frame", type=int, required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--zoom", type=float, help="pin the zoom factor")
    g.add_argument("--target-frac", type=float, help="pin connector length / frame height")
    ap.add_argument("--asset", help="restrict to one library asset (fnmatch glob on its name)")
    ap.add_argument("--views", type=int, default=2, help="number of zoom views (default 2)")
    ap.add_argument("--out", default=str(REPO / "results/connector_insertion/preview_zoom/single"))
    ap.add_argument("--keep", action="store_true", help="keep the temporary render folder")
    a = ap.parse_args(argv)

    for k, v in RECIPE.items():
        os.environ.setdefault(k, v)
    os.environ["KIAT_CONNINS_ZOOM_VIEWS"] = str(a.views)
    if a.asset:
        os.environ["KIAT_CONNINS_INCLUDE"] = a.asset
    if a.zoom is not None:
        os.environ["KIAT_CONNINS_ZOOM_MIN"] = os.environ["KIAT_CONNINS_ZOOM_MAX"] = str(a.zoom)
    if a.target_frac is not None:
        os.environ["KIAT_CONNINS_ZOOM_TARGET_FRAC_MIN"] = os.environ["KIAT_CONNINS_ZOOM_TARGET_FRAC_MAX"] = str(a.target_frac)
        os.environ.setdefault("KIAT_CONNINS_ZOOM_MIN", "1.0")
        os.environ.setdefault("KIAT_CONNINS_ZOOM_MAX", "16.0")
    tmp = tempfile.mkdtemp(prefix="zoom_preview_")
    os.environ["KIAT_OUTPUT_ROOT"] = tmp
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))
    import cv2
    import numpy as np
    import convert_to_video_dataset as c

    r = c.convert_one_video((a.set_id, a.frame, 1, 25.0, str(c.OUTPUT_ROOT)))
    if r[2] not in ("ok", "skipped"):
        print("render failed:", r[2])
        return 1
    base = next(Path(tmp).glob(f"*/{a.set_id:03d}"))
    rep = json.loads((base / "connector_insertion" / f"{a.frame:04d}.json").read_text())
    zv = rep.get("zoom_views", [])
    if not zv:
        print("no zoom views (no connector slot in this source; try zoom_fallback=wire)")
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for z in zv:
        j = z["index"]
        n = f"{a.frame:04d}_00_zoom{j}.png"
        rgb = cv2.imread(str(base / "rgb" / n))
        lab = cv2.imread(str(base / "label" / n), -1)
        ref = cv2.imread(str(base / "rgb" / f"{a.frame:04d}_00_{z['view']}.png")).copy()
        # planned window in the standard view: centre + half extent / zoom
        cx = 320 + z["centre_cam_xy"][0] * c.SCALE
        cy = 240 + z["centre_cam_xy"][1] * c.SCALE
        hw, hh = 320 / z["zoom"], 240 / z["zoom"]
        if not z["perturbed"]:
            cv2.rectangle(ref, (int(cx - hw), int(cy - hh)), (int(cx + hw), int(cy + hh)), (0, 0, 255), 2)
        tag = f"set{a.set_id:03d}_{a.frame:04d}_zoom{j}"
        cv2.imwrite(str(out / f"{tag}_rgb.png"), rgb)
        cv2.imwrite(str(out / f"{tag}_overlay.png"), overlay(rgb, lab))
        cv2.imwrite(str(out / f"{tag}_ref.png"), ref)
        print(f"{tag}: asset={z['asset']} view={z['view']} zoom={z['zoom']:.2f} "
              f"planned_frac={z['planned_frac']:.3f} connector_px={z['connector_px']} "
              f"visible={z.get('visible_frac')}")
    print("wrote", out)
    if not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
