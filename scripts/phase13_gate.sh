#!/usr/bin/env bash
# Phase 13 pre-flight gate for one pre-rendered lever dataset.
#
# Stages a ~1,512-frame stride-25 subset of data/rgbd_videos_phase13_<variant>,
# fine-tunes the Phase 7 teacher for 1 epoch on it, then runs the Phase 10
# ablation gate on the 80 D1 real frames. PASS = gate_a AND gate_b >= 0.50 on
# all 4 real videos (the cheap screen that ruled in H0 / ruled out H1 in
# minutes, before committing to the ~20 h train).
#
# Usage:  ./scripts/phase13_gate.sh <light|objgrad|both> [GPU_ID]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
source env/bin/activate

V="${1:?usage: $0 <light|objgrad|both> [GPU_ID]}"
GPU="${2:-0}"
SRC="data/rgbd_videos_phase13_${V}"
DST="data/dformer_dataset_phase13_${V}_gate"
RES="results/phase13_gate/${V}"
P7="results/segformer_b5_rgb/full_20260430_2032/best_model.pth"

[[ -f "${SRC}/metadata.json" ]] || { echo "ERROR: ${SRC} not rendered"; exit 1; }
mkdir -p "$RES"

echo "=== [1/3] stage stride-25 gate subset ($V) ==="
./env/bin/python -u src/prepare_dformer_data.py \
    --src-root "$SRC" --dst-root "$DST" \
    --label-source label --src-frame-step 25 --clean

echo "=== [2/3] 1-epoch fine-tune from Phase 7 ($V) on GPU $GPU ==="
CUDA_VISIBLE_DEVICES="$GPU" ./env/bin/python -u src/train_rgb_only_sota.py \
    --single-gpu --epochs 1 --batch-size 8 --lr 6e-5 \
    --weight-decay 0.01 --warmup-epochs 0 \
    --eval-every 1 --ckpt-every 1 \
    --dlo-weight 6.0 --grad-clip 1.0 \
    --num-classes 2 --backbone nvidia/mit-b5 \
    --data-dir "$DST" --results-dir "$RES" \
    --init-checkpoint "$P7"

echo "=== [3/3] gate eval ($V) ==="
CUDA_VISIBLE_DEVICES="$GPU" ./env/bin/python -u src/phase10_ablation_gate.py \
    --ckpt "${RES}/best_model.pth" --out-dir "$RES"

echo "=== gate.json ($V) ==="
./env/bin/python -c "import json;d=json.load(open('${RES}/gate.json'));print(json.dumps(d['overall'],indent=2))"
