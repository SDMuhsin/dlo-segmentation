"""Stage a Phase 10 ablation variant render into DFormer format.

Reuses src/prepare_dformer_data.py logic but parameterised so we don't
have to mutate the top-level constants for every variant. Writes
RGB/ Depth/ Label/ train.txt test.txt under <variant>/dformer/.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prepare_dformer_data import convert_depth_16to8  # noqa: E402


VIEWS_PHASE9 = ["view0", "view1", "view2", "view3", "view4", "view5"]
VIEWS_LEGACY = ["front", "back", "left", "right", "top", "bottom"]


def stage(src_root: Path, dst_root: Path, clean: bool = True,
          src_frame_step: int = 1, anim_frame: int = 0) -> None:
    if clean:
        for sub in ["RGB", "Depth", "Label", "cache"]:
            d = dst_root / sub
            if d.is_dir():
                shutil.rmtree(d)
                print(f"  Wiped {d}")
        for f in ["train.txt", "test.txt"]:
            p = dst_root / f
            if p.exists():
                p.unlink()

    meta_p = src_root / "metadata.json"
    if not meta_p.is_file():
        raise FileNotFoundError(f"metadata.json missing under {src_root}")
    meta = json.loads(meta_p.read_text())
    train_sets = meta["splits"]["train"]
    val_sets = meta["splits"]["val"]

    views_meta = meta.get("views", {})
    if isinstance(views_meta, dict) and views_meta.get("view_slot_names"):
        VIEWS = list(views_meta["view_slot_names"])
    elif isinstance(views_meta, dict) and views_meta:
        VIEWS = list(views_meta.keys())
    else:
        VIEWS = VIEWS_LEGACY[:]
    print(f"  Views: {VIEWS}  SRC_ROOT={src_root}  DST={dst_root}")

    for subdir in ["RGB", "Depth", "Label"]:
        (dst_root / subdir).mkdir(parents=True, exist_ok=True)

    train_files = []
    val_files = []
    for split_name, set_ids, out_list in [
        ("train", train_sets, train_files),
        ("val", val_sets, val_files),
    ]:
        print(f"\nProcessing {split_name} split ({len(set_ids)} sets)...")
        for set_id in tqdm(set_ids, desc="  Sets"):
            n_frames = meta["source_frames_per_set"][set_id]
            for src_idx in range(0, n_frames, src_frame_step):
                for view in VIEWS:
                    fname = f"{src_idx:04d}_{anim_frame:02d}_{view}"
                    src_rgb = src_root / split_name / set_id / "rgb" / f"{fname}.png"
                    src_depth = src_root / split_name / set_id / "depth" / f"{fname}.png"
                    src_label = src_root / split_name / set_id / "label" / f"{fname}.png"
                    if not src_rgb.exists():
                        continue
                    if not src_label.exists():
                        continue
                    out_name = f"{set_id}_{fname}"
                    dst_rgb = dst_root / "RGB" / f"{out_name}.png"
                    if not os.path.lexists(dst_rgb):
                        os.symlink(src_rgb.resolve(), dst_rgb)
                    dst_depth = dst_root / "Depth" / f"{out_name}.png"
                    if not dst_depth.exists():
                        depth_16 = cv2.imread(str(src_depth), cv2.IMREAD_UNCHANGED)
                        depth_8 = convert_depth_16to8(depth_16)
                        cv2.imwrite(str(dst_depth), depth_8)
                    dst_label = dst_root / "Label" / f"{out_name}.png"
                    if not os.path.lexists(dst_label):
                        os.symlink(src_label.resolve(), dst_label)
                    out_list.append(f"RGB/{out_name}.png")

    (dst_root / "train.txt").write_text("\n".join(train_files) + "\n")
    (dst_root / "test.txt").write_text("\n".join(val_files) + "\n")

    print(f"\nStaged {dst_root}:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val:   {len(val_files)} images")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path,
                   help="data/ablation_v0/<variant>")
    p.add_argument("--dst", required=True, type=Path,
                   help="data/ablation_v0/<variant>/dformer")
    p.add_argument("--clean", action="store_true")
    args = p.parse_args()
    stage(args.src, args.dst, clean=args.clean)


if __name__ == "__main__":
    main()
