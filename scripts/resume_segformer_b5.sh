#!/usr/bin/env bash
# Resume a crashed SegFormer-B5 training run from its latest checkpoint.
#
# Picks the checkpoint with the highest embedded 'epoch' in the run dir
# (best_model.pth or epoch_<N>.pth — whichever advanced furthest), auto-selects
# a GPU with >= 20 GB free, and relaunches src/train_rgb_only_sota.py with
# --resume. Warm-restart: model weights + epoch counter + LR-schedule position
# + best-IoU watermark are restored (optimizer moments reinitialise — negligible
# at the small post-warmup LRs where crashes are recovered).
#
# Usage:
#   ./scripts/resume_segformer_b5.sh <run_dir> [GPU_ID]
# e.g.:
#   ./scripts/resume_segformer_b5.sh results/segformer_b5_rgb_phase12_h1/full_20260602_1410
#
# GPU_ID is optional; if omitted the script auto-picks the first GPU with
# >= 20 GB free. Reads the run's original recipe from its checkpoint args so
# data-dir / epochs / num-classes / dlo-weight stay identical to the first launch.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source env/bin/activate

RUN_DIR="${1:?usage: $0 <run_dir> [GPU_ID]}"
RUN_DIR="${RUN_DIR%/}"
GPU_ARG="${2:-}"

if [[ ! -d "$RUN_DIR" ]]; then echo "ERROR: run dir not found: $RUN_DIR"; exit 1; fi

# --- Pick the latest checkpoint (max embedded epoch) and read the recipe -----
read -r CKPT EPOCHS NCLASSES DLOW DATADIR BACKBONE BATCH LR WD WARMUP < <(
python - "$RUN_DIR" <<'PY'
import sys, glob, os, torch
run = sys.argv[1]
cands = glob.glob(os.path.join(run, "epoch_*.pth")) + glob.glob(os.path.join(run, "best_model.pth"))
best=None; best_ep=-1
for p in cands:
    try:
        c = torch.load(p, map_location="cpu", weights_only=False)
        ep = int(c.get("epoch", -1)) if isinstance(c, dict) else -1
    except Exception:
        continue
    # prefer higher epoch; tie -> prefer best_model.pth
    if ep > best_ep or (ep == best_ep and os.path.basename(p) == "best_model.pth"):
        best_ep = ep; best = p
if best is None:
    print("NONE 0 2 6.0 data/dformer_dataset nvidia/mit-b5 8 6e-5 0.01 10"); sys.exit(0)
c = torch.load(best, map_location="cpu", weights_only=False)
a = c.get("args", {}) if isinstance(c, dict) else {}
print(best,
      a.get("epochs", 80),
      a.get("num_classes", 2),
      a.get("dlo_weight", 6.0),
      a.get("data_dir", "data/dformer_dataset"),
      a.get("backbone", "nvidia/mit-b5"),
      a.get("batch_size", 8),
      a.get("lr", 6e-5),
      a.get("weight_decay", 0.01),
      a.get("warmup_epochs", 10))
PY
)

if [[ "$CKPT" == "NONE" || -z "$CKPT" ]]; then
    echo "ERROR: no resumable checkpoint found in $RUN_DIR"; exit 1
fi

# --- Pick a GPU with >= 20 GB free unless one was given ----------------------
if [[ -z "$GPU_ARG" ]]; then
    GPU_ARG="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' '$2>=20000{print $1; exit}')"
    if [[ -z "$GPU_ARG" ]]; then
        echo "ERROR: no GPU with >=20 GB free. Current state:"; nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
        exit 1
    fi
fi

LOG="${RUN_DIR}/launch_resume_$(date +%Y%m%d_%H%M%S).log"
echo "Resuming run:   $RUN_DIR"
echo "  checkpoint:   $CKPT"
echo "  GPU:          $GPU_ARG"
echo "  recipe:       epochs=$EPOCHS num_classes=$NCLASSES dlo_weight=$DLOW batch=$BATCH lr=$LR wd=$WD warmup=$WARMUP"
echo "  data-dir:     $DATADIR"
echo "  log:          $LOG"

CUDA_VISIBLE_DEVICES="$GPU_ARG" nohup ./env/bin/python -u src/train_rgb_only_sota.py \
    --single-gpu \
    --num-classes "$NCLASSES" \
    --backbone "$BACKBONE" \
    --data-dir "$DATADIR" \
    --results-dir "$RUN_DIR" \
    --resume "$CKPT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --lr "$LR" \
    --weight-decay "$WD" \
    --warmup-epochs "$WARMUP" \
    --eval-every 1 \
    --ckpt-every 5 \
    --dlo-weight "$DLOW" \
    --grad-clip 1.0 \
    > "$LOG" 2>&1 &

PID=$!
disown
echo "Launched resume PID=$PID  (tail -f $LOG)"
