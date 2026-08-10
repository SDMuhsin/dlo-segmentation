#!/usr/bin/env bash
# Train SegFormer-B5 on Phase 12 H1 (strict-Phase-4 + 80-photo backdrop).
#
# Phase 7 recipe verbatim — lr 6e-5, batch 8, AdamW, warmup 10, AMP FP16,
# grad-clip 1.0, eval per epoch, ckpt every 5. Binary head {bg, DLO}.
#
# Usage:
#   ./scripts/train_segformer_b5_phase12_h1.sh [GPU_ID]
#
# Defaults: GPU 0. Override by passing it as the first arg, e.g.
#   ./scripts/train_segformer_b5_phase12_h1.sh 1
#
# Per CONTEXT.md §0.0.7 H1 acceptance: per-video mean DLO coverage on the
# 4 real videos must be within ±15% of the Phase 7 baseline
# (6.77 / 12.37 / 12.53 / 12.28 %) AND no video below 5 %.

set -euo pipefail

GPU_ID="${1:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M)"
RUN_DIR="results/segformer_b5_rgb_phase12_h1/full_${TS}"
LOG="${RUN_DIR}/launch.log"

mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" nohup ./env/bin/python -u src/train_rgb_only_sota.py \
    --single-gpu \
    --num-classes 2 \
    --backbone nvidia/mit-b5 \
    --data-dir data/dformer_dataset_phase12_h1 \
    --results-dir "$RUN_DIR" \
    --epochs 80 \
    --batch-size 8 \
    --lr 6e-5 \
    --weight-decay 0.01 \
    --warmup-epochs 10 \
    --eval-every 1 \
    --ckpt-every 5 \
    --dlo-weight 6.0 \
    --grad-clip 1.0 \
    > "$LOG" 2>&1 &

PID=$!
disown

echo "Launched SegFormer-B5 Phase 12 H1 training:"
echo "  PID:        $PID"
echo "  GPU:        $GPU_ID"
echo "  Run dir:    $RUN_DIR"
echo "  Log:        $LOG"
echo
echo "Tail with:   tail -f $LOG"
echo "Kill with:   kill $PID"
