#!/bin/bash
# ============================================================================
# Phase 5 — Bundle the minimum dataset into a single tar
# ============================================================================
#
# WHERE THIS RUNS: the DEV BOX.  Reads data/dformer_dataset/cache/*.npy and
# data/pretrained/.../pth from the local project tree.  Does NOT push
# anywhere — that's the job of sbatch/sync_to_rorqual.sh, which calls this
# script as its first step.
#
# Why a single tar:  Alliance file-count quotas are 500k (home/project) and
# 1M (scratch).  Storing 8,640 PNGs × 3 modalities = 26k inodes already
# eats 5% of /project's quota; the source rgbd_videos tree is ~3M files
# and would blow it.  Even when under quota, transfers of many small
# files are dominated by per-file overhead.  Standard fix: tar everything
# into ONE file, rsync the tar, untar on the GPU node into $SLURM_TMPDIR
# (per-job NVMe local scratch).
#
# What goes in:  only what the training script reads at runtime.
#   - data/dformer_dataset/cache/{train,val}_{rgb,depth,label}.npy   (6 files, ≈13.3 GB)
#   - data/dformer_dataset/{train,test}.txt                          (2 files, ≈220 KB)
#   - data/pretrained/DFormerv2/pretrained/DFormerv2_Large_pretrained.pth  (1 file, 375 MB)
# Total: 9 entries, ≈13.7 GB.
#
# What does NOT go in:  the 26k PNG symlinks at RGB/Depth/Label/ — they're
# only needed by build_cache(), which we already ran locally.  The Phase 4
# rgbd_videos source (≈3M files) is not needed at all on HPC.
#
# Usage (called automatically by sync_to_rorqual.sh; you can also run it
# directly to inspect or to refresh the tar):
#   ./sbatch/bundle_phase5_data.sh                     # writes ./phase5_data.tar
#   ./sbatch/bundle_phase5_data.sh --output /tmp/foo.tar
#   ./sbatch/bundle_phase5_data.sh --zstd              # writes phase5_data.tar.zst (~30% smaller)
#   ./sbatch/bundle_phase5_data.sh --check             # verify inputs only; do not write
# ============================================================================

set -euo pipefail

OUTPUT="phase5_data.tar"
USE_ZSTD=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --output)  OUTPUT="$2"; shift 2 ;;
        --zstd)    USE_ZSTD=true; shift 1 ;;
        --check)   CHECK_ONLY=true; shift 1 ;;
        -h|--help)
            sed -n '2,/^# ===$/p' "$0" | sed 's/^# \?//' | head -30
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ----------------------------------------------------------------------------
# Inputs (all relative to the script's project root). PROJECT_ROOT is
# resolved from this script's path, so it works wherever the project tree
# lives (e.g. /workspace/kiat_crefle on dev, ~/projects/.../dlo-segmentation
# on rorqual login).
# ----------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

CACHE_FILES=(
    "data/dformer_dataset/cache/train_rgb.npy"
    "data/dformer_dataset/cache/train_depth.npy"
    "data/dformer_dataset/cache/train_label.npy"
    "data/dformer_dataset/cache/val_rgb.npy"
    "data/dformer_dataset/cache/val_depth.npy"
    "data/dformer_dataset/cache/val_label.npy"
)
TXT_FILES=(
    "data/dformer_dataset/train.txt"
    "data/dformer_dataset/test.txt"
)
PRETRAINED="data/pretrained/DFormerv2/pretrained/DFormerv2_Large_pretrained.pth"

ALL_INPUTS=("${CACHE_FILES[@]}" "${TXT_FILES[@]}" "$PRETRAINED")

# ----------------------------------------------------------------------------
# Pre-flight: every input must exist and be non-zero. The cache files in
# particular are silently created at full size (open_memmap mode='w+') so
# size 0 is impossible, but a partially-written cache could exist if a
# previous build_cache run was interrupted. Check size > 0 to catch that.
# ----------------------------------------------------------------------------
echo "Verifying inputs..."
total_bytes=0
for f in "${ALL_INPUTS[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing: $f" >&2
        exit 1
    fi
    sz=$(stat -c '%s' "$f")
    if [[ "$sz" -le 0 ]]; then
        echo "ERROR: empty: $f" >&2
        exit 1
    fi
    total_bytes=$((total_bytes + sz))
    printf '  ok  %12d B  %s\n' "$sz" "$f"
done
total_gb=$(awk -v b="$total_bytes" 'BEGIN { printf "%.2f", b / 1024 / 1024 / 1024 }')
echo "Total input size: ${total_gb} GiB across ${#ALL_INPUTS[@]} files"

if [[ "$CHECK_ONLY" == "true" ]]; then
    echo "Check-only mode; not writing $OUTPUT."
    exit 0
fi

# ----------------------------------------------------------------------------
# Build the tar. Notes:
#   - We pass paths relative to PROJECT_ROOT so the archive layout mirrors
#     the project tree. Extracting with `tar xf phase5_data.tar -C $DEST`
#     produces $DEST/data/... — clean and predictable.
#   - --dereference: our Phase 4 PNGs at RGB/Label are symlinks but those
#     are NOT in the bundle.  The cache .npy files are real files.  No
#     symlinks here, but the flag is harmless and future-proof.
#   - --sparse: the .npy files are dense (no holes), so this is also a
#     no-op but cheap to enable.
#   - Verbose progress to stderr so the caller sees activity on a 13 GB tar.
#   - Use mtime preservation so untar doesn't reset to current time
#     (helpful for "did I update the bundle?" checks).
# ----------------------------------------------------------------------------
mkdir -p "$(dirname "$OUTPUT")" 2>/dev/null || true

tmp_out="${OUTPUT}.partial"
echo ""
echo "Writing $OUTPUT ..."

if [[ "$USE_ZSTD" == "true" ]]; then
    if ! command -v zstd >/dev/null; then
        echo "ERROR: --zstd requested but zstd not in PATH. apt install zstd or omit --zstd." >&2
        exit 1
    fi
    OUTPUT_FINAL="${OUTPUT%.tar}.tar.zst"
    [[ "$OUTPUT_FINAL" == "$OUTPUT" ]] && OUTPUT_FINAL="${OUTPUT}.zst"
    tmp_out="${OUTPUT_FINAL}.partial"
    tar --dereference --sparse -cf - "${ALL_INPUTS[@]}" \
        | zstd -3 -T0 --progress -o "$tmp_out"
    mv "$tmp_out" "$OUTPUT_FINAL"
    OUTPUT="$OUTPUT_FINAL"
else
    tar --dereference --sparse -cf "$tmp_out" --checkpoint=2000 --checkpoint-action=dot "${ALL_INPUTS[@]}"
    mv "$tmp_out" "$OUTPUT"
fi

echo ""
echo "Wrote $OUTPUT"
ls -lh "$OUTPUT"

# ----------------------------------------------------------------------------
# Verify the tar lists exactly the files we expect.  This catches edge
# cases like a data/.../symlink-pointing-to-deleted (would produce a tar
# with a 0-byte entry instead of erroring).
# ----------------------------------------------------------------------------
echo ""
echo "Tar contents (one entry per input):"
if [[ "$OUTPUT" == *.zst ]]; then
    zstd -dc "$OUTPUT" | tar tvf -
else
    tar tvf "$OUTPUT"
fi

n_in_tar=$(if [[ "$OUTPUT" == *.zst ]]; then zstd -dc "$OUTPUT"; else cat "$OUTPUT"; fi | tar tf - | wc -l)
if [[ "$n_in_tar" -ne "${#ALL_INPUTS[@]}" ]]; then
    echo ""
    echo "WARNING: expected ${#ALL_INPUTS[@]} entries in the tar but found $n_in_tar." >&2
    echo "Inspect the tar contents above before transferring." >&2
fi

echo ""
echo "================================================================"
echo "Bundled $OUTPUT (single file, ${#ALL_INPUTS[@]} entries inside)."
echo "Next: ./sbatch/sync_to_rorqual.sh --user <name>  (rsyncs this tar)."
echo "================================================================"
