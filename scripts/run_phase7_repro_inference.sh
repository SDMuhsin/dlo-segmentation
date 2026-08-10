#!/usr/bin/env bash
# Phase 7 reproduction — run real-world inference on all 4 user videos.
# Usage: ./scripts/run_phase7_repro_inference.sh <run_dir>
# e.g.:  ./scripts/run_phase7_repro_inference.sh results/segformer_b5_rgb_phase7_repro/full_20260601_0234

set -euo pipefail

RUN_DIR=${1:?usage: $0 <run_dir>}
CKPT="${RUN_DIR}/best_model.pth"
GPU=${CUDA_VISIBLE_DEVICES:-0}

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: $CKPT not found"
    exit 1
fi

source env/bin/activate
cd /workspace/kiat_crefle

for i in 1 2 3 4; do
    OUT_DIR="${RUN_DIR}/real_video_dlo_real_sample_${i}"
    echo "=== sample_${i} → $OUT_DIR ==="
    CUDA_VISIBLE_DEVICES=$GPU python -u src/infer_video_rgb_only.py \
        --video "data/dlo_real_sample_${i}.mp4" \
        --ckpt "$CKPT" \
        --out-dir "$OUT_DIR" \
        --also-gif \
        --device "cuda:0"
done

echo "=== All 4 video inferences complete ==="
ls "${RUN_DIR}"/real_video_dlo_real_sample_*/per_frame.csv
