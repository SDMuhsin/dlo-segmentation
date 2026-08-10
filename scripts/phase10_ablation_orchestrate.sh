#!/bin/bash
# Phase 10 ablation orchestrator. Runs:
#   1) render the variant
#   2) stage to DFormer format
#   3) fine-tune Phase 7 for 1 epoch
#   4) run the gate eval
# in sequence for one variant.
#
# Usage: scripts/phase10_ablation_orchestrate.sh <variant>
set -e
variant=$1
if [[ -z "$variant" ]]; then
  echo "Usage: $0 <variant>"
  exit 1
fi

cd /workspace/kiat_crefle
source env/bin/activate

VAR_DATA=data/ablation_v0/$variant
DFORMER=$VAR_DATA/dformer
RESULTS=results/phase10_ablation/$variant

mkdir -p $RESULTS

# Phase B: render (skip if already rendered)
if [[ ! -f $VAR_DATA/metadata.json ]]; then
    echo "=== [B] Render $variant ==="
    KIAT_OUTPUT_ROOT=$VAR_DATA KIAT_ABL_VARIANT=$variant \
        python -u src/render_full_dataset.py --workers 8 --skip-validation
fi

# Phase C: stage
if [[ ! -f $DFORMER/train.txt ]]; then
    echo "=== [C] Stage $variant ==="
    python -u src/stage_ablation_dformer.py --src $VAR_DATA --dst $DFORMER --clean
fi

# Phase D: fine-tune
if [[ ! -f $RESULTS/best_model.pth ]]; then
    echo "=== [D] Fine-tune $variant for 1 epoch ==="
    CUDA_VISIBLE_DEVICES=1 python -u src/train_rgb_only_sota.py \
        --single-gpu --epochs 1 --batch-size 8 --lr 6e-5 \
        --weight-decay 0.01 --warmup-epochs 0 \
        --eval-every 1 --ckpt-every 1 \
        --dlo-weight 6.0 --grad-clip 1.0 \
        --backbone nvidia/mit-b5 \
        --data-dir $DFORMER \
        --results-dir $RESULTS \
        --init-checkpoint results/segformer_b5_rgb/full_20260430_2032/best_model.pth
fi

# Phase E: gate eval
if [[ ! -f $RESULTS/gate.json ]]; then
    echo "=== [E] Gate eval $variant ==="
    CUDA_VISIBLE_DEVICES=1 python -u src/phase10_ablation_gate.py \
        --ckpt $RESULTS/best_model.pth \
        --out-dir $RESULTS
fi

echo "=== Done with $variant ==="
