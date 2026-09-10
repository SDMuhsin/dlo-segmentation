"""
Custom PyTorch Dataset + DataLoader for the "Electric Wires" image
segmentation dataset:
    https://www.kaggle.com/datasets/zanellar/electric-wires-image-segmentation

1. download_dataset()      -> pulls the dataset via kagglehub
2. inspect_dataset()       -> prints the on-disk layout + sample stats,
                              so we confirm folder names / mask value
                              range before trusting the pairing logic
3. ElectricWiresDataset    -> torch.utils.data.Dataset that yields
                              {"image": FloatTensor CxHxW, "mask": FloatTensor 1xHxW}
   get_dataloader()        -> convenience wrapper around DataLoader
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Folder-name candidates to try, in priority order, when looking for the
# RGB-image and mask subdirectories inside a split (train/ or test/).
IMG_KEYS = ("images", "image", "imgs", "img", "rgb")
MASK_KEYS = ("masks", "mask", "labels", "label", "gt", "annotations")

# Flip to False to silence all the step-by-step logging below.
VERBOSE = True

# directory by default ("./data1"); pass an absolute path if you want it
DATA_DIR = Path("data1")


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #

_LOG_START = time.time()


def _log(msg: str, indent: int = 0) -> None:
    """Timestamped, elapsed-time-stamped progress line. No-op if VERBOSE=False."""
    if not VERBOSE:
        return
    elapsed = time.time() - _LOG_START
    prefix = "  " * indent
    print(f"[{elapsed:7.2f}s] {prefix}{msg}", flush=True)


def _human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def _dir_stats(path: Path) -> Tuple[int, int]:
    """Returns (file_count, total_bytes) for everything under path."""
    n_files = 0
    total_bytes = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = Path(dirpath) / f
            try:
                total_bytes += fp.stat().st_size
                n_files += 1
            except OSError:
                pass
    return n_files, total_bytes


# --------------------------------------------------------------------------- #
# 1. Download
# --------------------------------------------------------------------------- #

def download_dataset(data_dir: Path = DATA_DIR) -> Path:
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(data_dir)

    import kagglehub

    _log(f"Using download directory: {data_dir}")
    _log("Checking cache for 'zanellar/electric-wires-image-segmentation'...")

    # With KAGGLEHUB_CACHE set above, kagglehub caches under
    # <data_dir>/datasets/<owner>/<name>/versions/<n>
    # We can't hook into kagglehub's internal download progress bar directly,
    # but kagglehub prints its own tqdm-based % progress to stdout/stderr
    # during the actual HTTP download, so that will show up interleaved with
    # these logs automatically when a real download happens.
    cache_root = data_dir / "datasets" / "zanellar" / "electric-wires-image-segmentation"
    was_cached = cache_root.exists() and any(cache_root.rglob("*"))

    if was_cached:
        _log(f"Found existing cache at {cache_root} -- will reuse it (no download needed).", indent=1)
    else:
        _log("No local cache found -- starting fresh download from Kaggle.", indent=1)
        _log("(kagglehub's own progress bar, if any, will appear below.)", indent=1)

    t0 = time.time()
    path = Path(kagglehub.dataset_download("zanellar/electric-wires-image-segmentation"))
    dt = time.time() - t0

    n_files, total_bytes = _dir_stats(path)

    if was_cached:
        _log(f"Reused cached dataset in {dt:.2f}s.", indent=1)
    else:
        _log(f"Download + extraction finished in {dt:.2f}s.", indent=1)

    _log(f"Dataset cached at: {path}", indent=1)
    _log(f"On disk: {n_files} files, {_human_size(total_bytes)} total.", indent=1)

    print(f"Dataset cached at: {path}")
    return path


# --------------------------------------------------------------------------- #
# 2. Inspection
# --------------------------------------------------------------------------- #

def inspect_dataset(root: Path, max_samples: int = 20) -> None:
    """
    Walks the dataset root and prints, for every directory that contains
    image files: the file count, extensions present, and — for one sample
    file per directory — its size, color mode, and unique pixel values.

    The unique-pixel-values check is the important one: it's how you tell
    an "images" folder (thousands of unique values) apart from a "masks"
    folder (should be ~2 unique values if truly binary, e.g. {0, 255}), and
    it tells you whether a mask needs thresholding before use.

    Also prints running totals as it walks, so on a large dataset you can
    see it's making progress rather than looking hung.
    """
    _log(f"Scanning directory tree under {root} ...")
    print(f"\n=== Directory tree under {root} ===")

    n_dirs_shown = 0
    n_dirs_total = 0
    n_files_total = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel = Path(dirpath).relative_to(root)
        files = sorted(f for f in filenames if Path(f).suffix.lower() in IMG_EXTS)
        if not files:
            continue

        n_dirs_total += 1
        n_files_total += len(files)

        exts = sorted({Path(f).suffix.lower() for f in files})
        print(f"  {rel}/  -> {len(files)} image files, extensions={exts}")
        _log(f"found {len(files)} image files under '{rel}/' (running total: {n_files_total} files across {n_dirs_total} dirs)", indent=1)

        n_dirs_shown += 1
        if n_dirs_shown > max_samples:
            continue

        sample_path = Path(dirpath) / files[0]
        try:
            with Image.open(sample_path) as im:
                arr = np.array(im)
                uniq = np.unique(arr)
                uniq_display = uniq.tolist() if uniq.size <= 8 else (
                    f"{uniq[:8].tolist()} ... ({uniq.size} unique total)"
                )
                print(
                    f"      sample: {files[0]}  size={im.size}  mode={im.mode}  "
                    f"unique_values={uniq_display}"
                )
                if uniq.size <= 2:
                    _log(f"'{rel}/' looks binary ({uniq.size} unique value(s)) -- likely a MASK folder.", indent=1)
                else:
                    _log(f"'{rel}/' looks continuous-toned ({uniq.size} unique values) -- likely an IMAGE folder.", indent=1)
        except Exception as e:
            print(f"      (could not open {sample_path}: {e})")
            _log(f"could not open sample file {sample_path}: {e}", indent=1)

    _log(f"Scan complete: {n_dirs_total} image-bearing directories, {n_files_total} image files total.")


def _find_split_dirs(root: Path) -> Dict[str, Path]:
    """Locates train/test/val directories anywhere under root."""
    _log("Looking for split directories (train/test/val/valid)...")
    splits: Dict[str, Path] = {}
    for name in ("train", "test", "val", "valid"):
        dirs = [p for p in root.rglob(name) if p.is_dir()]
        if dirs:
            dirs.sort(key=lambda p: len(p.parts))  # prefer the shallowest match
            splits[name] = dirs[0]
            _log(f"'{name}' split found at {dirs[0]}", indent=1)
        else:
            _log(f"'{name}' split not found.", indent=1)
    return splits


# --------------------------------------------------------------------------- #
# Image <-> mask pairing
# --------------------------------------------------------------------------- #

def _pair_images_and_masks(split_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Pairs each RGB image with its binary mask inside one split directory.

    Tries, in order:
      1. dedicated subfolders, e.g. images/ + masks/ (matched by filename stem,
         with a fallback to a "<stem>_mask" / "<stem>_label" suffix on the
         mask side)
      2. a flat folder where image/mask share a stem and the mask's filename
         additionally contains "mask" or "label"

    Returns a list of (image_path, mask_path) tuples. Along the way, prints:
      - which subfolder convention it detected (or fell back from)
      - a running "matched so far" count
      - every image that has NO matching mask, and every mask that matched
        no image, by filename -- not just a final tally
    """
    _log(f"Pairing images <-> masks under {split_dir} ...")
    subdirs = {d.name.lower(): d for d in split_dir.iterdir() if d.is_dir()}

    img_dir = next((subdirs[k] for k in IMG_KEYS if k in subdirs), None)
    mask_dir = next((subdirs[k] for k in MASK_KEYS if k in subdirs), None)

    # Some dataset exports add an extra split-named directory level, e.g.
    # train/train/imgs + train/train/masks and test/test/imgs + test/test/masks.
    # If the expected image/mask folders are not directly under split_dir,
    # look one level deeper for the same common folder conventions.
    if img_dir is None or mask_dir is None:
        for nested_dir in sorted(subdirs.values(), key=lambda p: p.name.lower()):
            nested_subdirs = {
                d.name.lower(): d for d in nested_dir.iterdir() if d.is_dir()
            }
            nested_img_dir = next(
                (nested_subdirs[k] for k in IMG_KEYS if k in nested_subdirs), None
            )
            nested_mask_dir = next(
                (nested_subdirs[k] for k in MASK_KEYS if k in nested_subdirs), None
            )
            if nested_img_dir is not None and nested_mask_dir is not None:
                _log(
                    f"detected nested split directory '{nested_dir.name}/'; "
                    f"using images='{nested_img_dir.name}/', masks='{nested_mask_dir.name}/'",
                    indent=1,
                )
                img_dir = nested_img_dir
                mask_dir = nested_mask_dir
                break

    pairs: List[Tuple[Path, Path]] = []

    if img_dir is not None and mask_dir is not None:
        _log(f"detected dedicated subfolders: images='{img_dir.name}/', masks='{mask_dir.name}/'", indent=1)

        imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        _log(f"{len(imgs)} candidate images, {len(masks)} candidate masks to match up.", indent=1)

        unmatched_images: List[str] = []
        for i, (stem, img_path) in enumerate(sorted(imgs.items()), start=1):
            mask_path = (
                masks.get(stem)
                or masks.get(f"{stem}_mask")
                or masks.get(f"{stem}_label")
            )
            if mask_path is not None:
                pairs.append((img_path, mask_path))
            else:
                unmatched_images.append(img_path.name)

            if VERBOSE and (i % 100 == 0 or i == len(imgs)):
                pct = 100.0 * i / max(len(imgs), 1)
                _log(f"matched {len(pairs)}/{i} images so far ({pct:.0f}% scanned)...", indent=1)

        matched_mask_stems = {p.stem for _, p in pairs}
        # also account for the "<stem>_mask"/"<stem>_label" variants we matched by
        matched_mask_paths = {p for _, p in pairs}
        unmatched_masks = [p.name for p in masks.values() if p not in matched_mask_paths]

        if unmatched_images:
            _log(f"WARNING: {len(unmatched_images)} image(s) have NO matching mask:", indent=1)
            for name in unmatched_images[:20]:
                _log(f"missing mask for: {name}", indent=2)
            if len(unmatched_images) > 20:
                _log(f"... and {len(unmatched_images) - 20} more.", indent=2)
        else:
            _log("every image found a matching mask.", indent=1)

        if unmatched_masks:
            _log(f"NOTE: {len(unmatched_masks)} mask(s) have no matching image (unused):", indent=1)
            for name in unmatched_masks[:20]:
                _log(f"unused mask: {name}", indent=2)
            if len(unmatched_masks) > 20:
                _log(f"... and {len(unmatched_masks) - 20} more.", indent=2)

        if pairs:
            _log(f"Pairing done via subfolder convention: {len(pairs)} pairs matched.", indent=1)
            return pairs
        else:
            _log("subfolder convention yielded 0 pairs -- falling back to flat-folder convention.", indent=1)
    else:
        found = list(subdirs.keys())
        _log(f"no dedicated images/masks subfolders detected (saw: {found}) -- trying flat-folder convention.", indent=1)

    # Fallback: everything in one flat folder.
    flat_files = [p for p in split_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    _log(f"flat-folder fallback: {len(flat_files)} image files directly under {split_dir}.", indent=1)

    by_stem: Dict[str, List[Path]] = {}
    for p in flat_files:
        base = p.stem.lower().replace("_mask", "").replace("_label", "")
        by_stem.setdefault(base, []).append(p)

    unmatched_stems: List[str] = []
    for base, files in by_stem.items():
        if len(files) != 2:
            unmatched_stems.append(f"{base} ({len(files)} file(s): {[f.name for f in files]})")
            continue
        a, b = files
        a_is_mask = "mask" in a.stem.lower() or "label" in a.stem.lower()
        b_is_mask = "mask" in b.stem.lower() or "label" in b.stem.lower()
        if a_is_mask and not b_is_mask:
            pairs.append((b, a))
        elif b_is_mask and not a_is_mask:
            pairs.append((a, b))
        else:
            unmatched_stems.append(f"{base} (ambiguous mask/image labeling: {[f.name for f in files]})")

    if unmatched_stems:
        _log(f"WARNING: {len(unmatched_stems)} stem(s) in flat folder did NOT resolve to a clean image/mask pair:", indent=1)
        for s in unmatched_stems[:20]:
            _log(s, indent=2)
        if len(unmatched_stems) > 20:
            _log(f"... and {len(unmatched_stems) - 20} more.", indent=2)

    if not pairs:
        print(
            f"WARNING: found no image/mask pairs under {split_dir}. "
            f"Subdirectories seen: {list(subdirs.keys())}. "
            f"Run inspect_dataset() and adjust IMG_KEYS/MASK_KEYS if needed."
        )
        _log(f"FAILED: no image/mask pairs found under {split_dir}.", indent=1)
    else:
        _log(f"Pairing done via flat-folder convention: {len(pairs)} pairs matched.", indent=1)

    return pairs


# --------------------------------------------------------------------------- #
# 3. Dataset + DataLoader
# --------------------------------------------------------------------------- #

class ElectricWiresDataset(Dataset):
    """
    RGB image + binary wire mask pairs.

    Each item is a dict:
        "image":      FloatTensor, shape (3, H, W), values in [0, 1]
        "mask":       FloatTensor, shape (1, H, W), values in {0.0, 1.0}
        "image_path": str
        "mask_path":  str

    Args:
        pairs: list of (image_path, mask_path), e.g. from _pair_images_and_masks().
        image_size: (H, W) to resize both image and mask to. Image uses
            bilinear resizing; mask uses nearest-neighbor so no in-between
            gray values get introduced at object edges. Pass None to keep
            each image's native resolution (only safe with batch_size=1,
            or a collate_fn that pads instead of stacking).
        augment: if True, applies a random horizontal flip (use for
            training only, not for validation/test).
        mask_threshold: mask pixel values >= this are treated as
            foreground (wire). Works whether the source mask is stored
            as {0, 255} or {0, 1} — inspect_dataset() will show you which.
    """

    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        image_size: Optional[Tuple[int, int]] = (512, 512),
        augment: bool = False,
        mask_threshold: int = 128,
    ):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment
        self.mask_threshold = mask_threshold
        _log(
            f"ElectricWiresDataset created: {len(pairs)} pairs, "
            f"image_size={image_size}, augment={augment}, mask_threshold={mask_threshold}"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # collapse to single channel

        if self.image_size is not None:
            h, w = self.image_size
            image = image.resize((w, h), Image.BILINEAR)
            mask = mask.resize((w, h), Image.NEAREST)

        image_t = TF.to_tensor(image)  # (3, H, W), float32 in [0, 1]

        mask_arr = np.array(mask, dtype=np.uint8)
        mask_bin = (mask_arr >= self.mask_threshold).astype(np.float32)
        mask_t = torch.from_numpy(mask_bin).unsqueeze(0)  # (1, H, W)

        if self.augment and torch.rand(1).item() < 0.5:
            image_t = torch.flip(image_t, dims=[2])
            mask_t = torch.flip(mask_t, dims=[2])

        return {
            "image": image_t,
            "mask": mask_t,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
        }


def get_dataloader(
    pairs: List[Tuple[Path, Path]],
    batch_size: int = 8,
    image_size: Optional[Tuple[int, int]] = (512, 512),
    augment: bool = False,
    shuffle: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    """Convenience wrapper: builds an ElectricWiresDataset and DataLoader together."""
    _log(
        f"Building DataLoader: batch_size={batch_size}, shuffle={shuffle}, "
        f"num_workers={num_workers}, {len(pairs)} pairs "
        f"-> {(len(pairs) + batch_size - 1) // max(batch_size, 1)} batches"
    )
    ds = ElectricWiresDataset(pairs, image_size=image_size, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# --------------------------------------------------------------------------- #
# Script entry point: download -> inspect -> build loaders -> sanity-check
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    _log("=== STEP 1/4: Download ===")
    root = download_dataset(DATA_DIR)

    _log("=== STEP 2/4: Inspect ===")
    inspect_dataset(root)

    _log("=== STEP 3/4: Locate splits + pair images/masks ===")
    splits = _find_split_dirs(root)
    print(f"\nDetected split directories: {list(splits.keys())}")

    train_pairs = _pair_images_and_masks(splits["train"]) if "train" in splits else []
    test_pairs = _pair_images_and_masks(splits["test"]) if "test" in splits else []
    print(f"train pairs found: {len(train_pairs)}")
    print(f"test  pairs found: {len(test_pairs)}")

    _log("=== STEP 4/4: Build DataLoaders + sanity-check a batch ===")

    if train_pairs:
        train_loader = get_dataloader(train_pairs, batch_size=8, augment=True, shuffle=True)
        _log("pulling one sample batch from the train loader...", indent=1)
        t0 = time.time()
        batch = next(iter(train_loader))
        _log(f"batch fetched in {time.time() - t0:.2f}s.", indent=1)
        print("\nSample train batch:")
        print("  image:", batch["image"].shape, batch["image"].dtype)
        print("  mask: ", batch["mask"].shape, batch["mask"].dtype,
              "unique values:", torch.unique(batch["mask"]).tolist())
    else:
        _log("skipping train batch sanity-check: 0 train pairs.", indent=1)

    if test_pairs:
        test_loader = get_dataloader(test_pairs, batch_size=8, augment=False, shuffle=False)
        print(f"\ntest loader ready: {len(test_loader)} batches")
    else:
        _log("skipping test loader: 0 test pairs.", indent=1)

    _log("All done.")