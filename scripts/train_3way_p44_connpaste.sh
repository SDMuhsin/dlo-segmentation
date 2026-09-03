#!/usr/bin/env bash
# P44 — real connector cutouts grafted as labelled connector (class 2).
#
# DESIGN: single variable vs the Phase-3W ship recipe. Every flag is identical to
#   scripts/train_3way_lovasz_ab.sh <gpu> 0.5 lovasz050_zoom --aug-zoom 0.5,0.5,1.0
# EXCEPT --data-dir, which is the P44 connector-paste build (same 7560 train frames,
# 1500 of them substituted 1:1 by their composited version; train.txt/test.txt are
# byte-identical to the base, verified).
#
# WARM START: the P3W recipe's init `p39_cutoutneg/segformer_b5_warmstart/epoch_10.pth`
# no longer exists (2026-08-17 disk cleanup). `best_model.pth` from the SAME binary run
# is used instead (num_classes=2, IoU(DLO) 0.9730 vs epoch_10's 0.9726 — same run, same
# recipe, adjacent epoch). Because that is NOT the baseline's exact init, the Phase-3W
# published numbers are NOT a valid control; a matched control arm on the BASE dataset
# with this same init is run separately. Comparing to unmatched warm starts is the
# documented epoch_15 selection-spike trap.
#
# Usage: scripts/train_3way_p44_connpaste.sh <gpu> <tag> <data_dir> [extra flags...]
set -euo pipefail

GPU="${1:?usage: $0 <gpu> <tag> <data_dir> [extra...]}"
TAG="${2:?missing tag}"
DATA="${3:?missing data dir}"
shift 3
EXTRA=("$@")

cd /workspace/kiat_crefle
source env/bin/activate

OUT="results/realism_campaign/p44_connpaste/${TAG}"
mkdir -p "$OUT"
echo "[launch] gpu=$GPU tag=$TAG data=$DATA extra=${EXTRA[*]:-none}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES="$GPU" python -u src/train_rgb_only_sota.py \
  --single-gpu \
  --num-classes 3 \
  --class-weights 1,6,4 \
  --dlo-weight 6.0 \
  --data-dir "$DATA" \
  --backbone nvidia/mit-b5 \
  --init-checkpoint results/realism_campaign/p39_cutoutneg/segformer_b5_warmstart/best_model.pth \
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
  --lovasz-weight 0.5 \
  --aug-zoom 0.5,0.5,1.0 \
  --select-metric miou \
  --results-dir "$OUT" \
  "${EXTRA[@]}" \
  2>&1 | tee "$OUT/train.log"

echo "[done] $TAG"
