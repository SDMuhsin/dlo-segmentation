#!/usr/bin/env bash
# Phase 3W Round 1 — matched A/B on the Lovasz-Softmax IoU-surrogate loss.
#
# DESIGN: single variable. Every flag below is unchanged from the
# `seg_b5_3way_connscale3` baseline run (IoU con 0.7621 / wire 0.8597 /
# mIoU 0.8731) EXCEPT --lovasz-weight, which is the lever under test, and
# --select-metric, which changes only which checkpoint is written to
# best_model.pth (it does not touch training at all). Same init checkpoint,
# same seed, same data, same schedule -- so any delta is attributable.
#
# The project has been burned before by comparing against an unmatched
# warm-start (see the epoch_15 selection-spike finding), hence the strict match.
#
# Usage: scripts/train_3way_lovasz_ab.sh <gpu> <lovasz_weight> <tag> [extra flags...]
#   extra flags are appended verbatim, for Round-2 variants (e.g. --aug2d, a
#   different --epochs, or an --init-checkpoint override). Anything passed there
#   is a DELIBERATE second variable -- note it in the report.
set -euo pipefail

GPU="${1:?usage: $0 <gpu> <lovasz_weight> <tag> [extra flags...]}"
LW="${2:?missing lovasz weight}"
TAG="${3:?missing tag}"
shift 3
EXTRA=("$@")

cd /workspace/kiat_crefle
source env/bin/activate

OUT="results/realism_campaign/p3w_lovasz/${TAG}"
mkdir -p "$OUT"

echo "[launch] gpu=$GPU lovasz_weight=$LW out=$OUT extra=${EXTRA[*]:-none}"

CUDA_VISIBLE_DEVICES="$GPU" python -u src/train_rgb_only_sota.py \
  --single-gpu \
  --num-classes 3 \
  --class-weights 1,6,4 \
  --dlo-weight 6.0 \
  --data-dir data/dformer_dataset_3way_connscale3 \
  --backbone nvidia/mit-b5 \
  --init-checkpoint results/realism_campaign/p39_cutoutneg/segformer_b5_warmstart/epoch_10.pth \
  --epochs 30 \
  --batch-size 8 \
  --lr 6e-5 \
  --weight-decay 0.01 \
  --warmup-epochs 5 \
  --eval-every 2 \
  --log-every 20 \
  --ckpt-every 5 \
  --grad-clip 1.0 \
  --seed 1234 \
  --lovasz-weight "$LW" \
  --select-metric miou \
  --results-dir "$OUT" \
  "${EXTRA[@]}" \
  2>&1 | tee "$OUT/train.log"

echo "[done] $TAG"
