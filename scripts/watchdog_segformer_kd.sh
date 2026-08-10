#!/usr/bin/env bash
# Auto-resume watchdog for a long SegFormer KD-student run (src/train_rgb_only_kd.py)
# that may crash with the server. Watches the current training PID; when it dies
# WITHOUT having written a complete report.json (epochs_completed >= target), it
# finds the latest resumable checkpoint, reads that checkpoint's saved training
# recipe, and relaunches `src/train_rgb_only_kd.py --resume <ckpt>` on the given
# GPU, then keeps watching the new PID.
#
# Usage:
#   ./scripts/watchdog_segformer_kd.sh <run_dir> <initial_pid> <target_epochs> <gpu>
# e.g.:
#   ./scripts/watchdog_segformer_kd.sh results/segformer_b0_rgb_kd/full_phase15 105200 80 1
#
# Exits 0 when training completes (report.json with epochs_completed >= target);
# exits 1 if the retry cap (15) is hit or a relaunch fails. Logs every action to
# <run_dir>/watchdog.log so recovery is auditable.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${1:?usage: $0 <run_dir> <initial_pid> <target_epochs> <gpu>}"
RUN_DIR="${RUN_DIR%/}"
PID="${2:?need initial pid}"
TARGET_EPOCHS="${3:?need target epochs}"
GPU="${4:?need gpu id}"
MAX_RETRIES=15

WLOG="${RUN_DIR}/watchdog.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$WLOG"; }

is_done(){
    local rj="${RUN_DIR}/report.json"
    [[ -f "$rj" ]] || return 1
    ./env/bin/python - "$rj" "$TARGET_EPOCHS" <<'PY'
import json,sys
try:
    r=json.load(open(sys.argv[1])); tgt=int(sys.argv[2])
    sys.exit(0 if int(r.get("epochs_completed",0))>=tgt else 1)
except Exception:
    sys.exit(1)
PY
}

log "watchdog start: run=$RUN_DIR pid=$PID target=$TARGET_EPOCHS gpu=$GPU max_retries=$MAX_RETRIES"
retry=0
while true; do
    # Wait for the current training PID to exit.
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    log "training PID $PID exited"

    if is_done; then
        log "report.json shows epochs_completed >= $TARGET_EPOCHS — COMPLETE"
        exit 0
    fi

    if (( retry >= MAX_RETRIES )); then
        log "retry cap ($MAX_RETRIES) hit — giving up; manual attention needed"
        exit 1
    fi
    retry=$((retry+1))
    log "training incomplete — relaunch attempt $retry/$MAX_RETRIES in 20s"
    sleep 20  # let the crashed process's GPU memory free

    # --- Pick the latest resumable checkpoint (max embedded epoch) and read the
    #     full training recipe from its saved args. Tie -> prefer best_model.pth.
    read -r CKPT EPOCHS BATCH LR WD WARMUP DLOW ALPHA TEMP GRADCLIP BACKBONE TEACHER DATADIR RUNTAG < <(
    ./env/bin/python - "$RUN_DIR" <<'PY'
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
    if ep > best_ep or (ep == best_ep and os.path.basename(p) == "best_model.pth"):
        best_ep = ep; best = p
if best is None:
    print("NONE"); sys.exit(0)
c = torch.load(best, map_location="cpu", weights_only=False)
a = c.get("args", {}) if isinstance(c, dict) else {}
def g(k, d):
    v = a.get(k, d)
    return d if v is None else v
print(best,
      g("epochs", 80),
      g("batch_size", 8),
      g("lr", 6e-5),
      g("weight_decay", 0.01),
      g("warmup_epochs", 10),
      g("dlo_weight", 6.0),
      g("alpha", 0.5),
      g("temperature", 4.0),
      g("grad_clip", 1.0),
      g("backbone", "nvidia/mit-b0"),
      g("teacher_ckpt", "results/segformer_b5_rgb/full_20260430_2032/best_model.pth"),
      g("data_dir", "data/dformer_dataset"),
      g("run_tag", "full"))
PY
    )

    if [[ "$CKPT" == "NONE" || -z "${CKPT:-}" ]]; then
        log "relaunch FAILED: no resumable checkpoint found in $RUN_DIR — giving up"
        exit 1
    fi
    log "resuming from ckpt=$CKPT  recipe: epochs=$EPOCHS batch=$BATCH lr=$LR wd=$WD warmup=$WARMUP dlo=$DLOW alpha=$ALPHA T=$TEMP gradclip=$GRADCLIP backbone=$BACKBONE"
    log "  teacher=$TEACHER  data-dir=$DATADIR  run-tag=$RUNTAG  gpu=$GPU"

    CUDA_VISIBLE_DEVICES="$GPU" nohup ./env/bin/python -u src/train_rgb_only_kd.py \
        --resume "$CKPT" \
        --single-gpu \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --lr "$LR" \
        --weight-decay "$WD" \
        --warmup-epochs "$WARMUP" \
        --dlo-weight "$DLOW" \
        --alpha "$ALPHA" \
        --temperature "$TEMP" \
        --grad-clip "$GRADCLIP" \
        --backbone "$BACKBONE" \
        --teacher-ckpt "$TEACHER" \
        --data-dir "$DATADIR" \
        --run-tag "$RUNTAG" \
        --eval-every 1 \
        --ckpt-every 5 \
        >> "${RUN_DIR}/resume_launch.log" 2>&1 &

    NEWPID=$!
    disown
    if [[ -z "$NEWPID" ]] || ! kill -0 "$NEWPID" 2>/dev/null; then
        log "relaunch FAILED (no live PID) — giving up"
        exit 1
    fi
    PID="$NEWPID"
    log "relaunched as PID $PID (CUDA_VISIBLE_DEVICES=$GPU); log -> ${RUN_DIR}/resume_launch.log"
    sleep 60  # give it a head start before resuming the watch loop
done
