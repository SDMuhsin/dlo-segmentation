#!/usr/bin/env bash
# Train SegFormer-B5 on Phase 11 (3-way: backdrop / wire / objects).
#
# Phase 7 recipe verbatim — lr 6e-5, batch 8, AdamW, warmup 10, AMP FP16,
# grad-clip 1.0, eval per epoch, ckpt every 5. Class weights default
# [1.0, 6.0, 1.0] (wire-class weight = --dlo-weight, matches Phase 7).
#
# Usage:
#   ./scripts/train_segformer_b5_phase11_3way.sh [GPU_ID]
#
# Defaults: GPU 1. Override the GPU by passing it as the first arg, e.g.
#   ./scripts/train_segformer_b5_phase11_3way.sh 0
#
# The script:
#   * cd's to project root (works regardless of where you invoke it from)
#   * creates results/segformer_b5_rgb_phase11_3way/full_<TS>/
#   * launches training in the background with nohup
#   * prints the launched PID and the log path

set -euo pipefail

GPU_ID="${1:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M)"
RUN_DIR="results/segformer_b5_rgb_phase11_3way/full_${TS}"
LOG="${RUN_DIR}/launch.log"

mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" nohup ./env/bin/python -u src/train_rgb_only_sota.py \
    --single-gpu \
    --num-classes 3 \
    --backbone nvidia/mit-b5 \
    --data-dir data/dformer_dataset_phase11_3way \
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

echo "Launched SegFormer-B5 Phase 11 3-way training:"
echo "  PID:        $PID"
echo "  GPU:        $GPU_ID"
echo "  Run dir:    $RUN_DIR"
echo "  Log:        $LOG"
echo
echo "Tail with:   tail -f $LOG"
echo "Kill with:   kill $PID"
