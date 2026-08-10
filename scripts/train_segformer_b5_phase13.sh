#!/usr/bin/env bash
# Train SegFormer-B5 on a Phase 13 lever dataset (Phase 7 recipe verbatim).
#
# Stages the full stride-5 dataset (7,560 train / 1,080 val) from the
# pre-rendered data/rgbd_videos_phase13_<variant>, then launches the 80-epoch
# binary-head train under the auto-resume watchdog (the dev server crashes
# mid-run; --resume recovers from the latest checkpoint).
#
# Usage:  ./scripts/train_segformer_b5_phase13.sh <light|objgrad|both> [GPU_ID]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
source env/bin/activate

V="${1:?usage: $0 <light|objgrad|both> [GPU_ID]}"
GPU="${2:-0}"
SRC="data/rgbd_videos_phase13_${V}"
DST="data/dformer_dataset_phase13_${V}"
TS="$(date +%Y%m%d_%H%M)"
RUN="results/segformer_b5_rgb_phase13_${V}/full_${TS}"
LOG="${RUN}/launch.log"

[[ -f "${SRC}/metadata.json" ]] || { echo "ERROR: ${SRC} not rendered"; exit 1; }
mkdir -p "$RUN"

echo "=== stage full stride-5 dataset ($V) ==="
./env/bin/python -u src/prepare_dformer_data.py \
    --src-root "$SRC" --dst-root "$DST" \
    --label-source label --src-frame-step 5 --clean

echo "=== launch 80-epoch train ($V) on GPU $GPU ==="
CUDA_VISIBLE_DEVICES="$GPU" nohup ./env/bin/python -u src/train_rgb_only_sota.py \
    --single-gpu --num-classes 2 --backbone nvidia/mit-b5 \
    --data-dir "$DST" --results-dir "$RUN" \
    --epochs 80 --batch-size 8 --lr 6e-5 \
    --weight-decay 0.01 --warmup-epochs 10 \
    --eval-every 1 --ckpt-every 5 \
    --dlo-weight 6.0 --grad-clip 1.0 \
    > "$LOG" 2>&1 &
PID=$!
disown
echo "Launched PID=$PID  RUN=$RUN  (tail -f $LOG)"

# Auto-resume watchdog (relaunches from latest ckpt on crash).
nohup ./scripts/watchdog_segformer_b5.sh "$RUN" "$PID" 80 > /dev/null 2>&1 &
disown
echo "Watchdog started for $RUN (log: ${RUN}/watchdog.log)"
echo "$RUN"
