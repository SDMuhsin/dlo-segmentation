#!/usr/bin/env bash
# Auto-resume watchdog for a long SegFormer-B5 run that keeps crashing with the
# server. Watches the current training PID; when it dies WITHOUT having written
# a complete report.json (epochs_completed >= target), it relaunches from the
# latest checkpoint via resume_segformer_b5.sh and keeps watching the new PID.
#
# Usage:
#   ./scripts/watchdog_segformer_b5.sh <run_dir> <initial_pid> [target_epochs] [max_retries]
#
# Exits 0 when training completes (report.json with epochs_completed >= target);
# exits 1 if the retry cap is hit or a relaunch fails. Logs every action to
# <run_dir>/watchdog.log so recovery is auditable.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${1:?usage: $0 <run_dir> <initial_pid> [target_epochs] [max_retries]}"
RUN_DIR="${RUN_DIR%/}"
PID="${2:?need initial pid}"
TARGET_EPOCHS="${3:-80}"
MAX_RETRIES="${4:-15}"

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

log "watchdog start: run=$RUN_DIR pid=$PID target=$TARGET_EPOCHS max_retries=$MAX_RETRIES"
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
    sleep 20  # let crashed process's GPU memory free

    OUT="$(./scripts/resume_segformer_b5.sh "$RUN_DIR" 2>&1)"
    echo "$OUT" | tee -a "$WLOG"
    NEWPID="$(echo "$OUT" | sed -n 's/.*Launched resume PID=\([0-9]\+\).*/\1/p' | tail -1)"
    if [[ -z "$NEWPID" ]]; then
        log "relaunch FAILED (no PID) — giving up"
        exit 1
    fi
    PID="$NEWPID"
    log "relaunched as PID $PID"
    sleep 60  # give it a head start before resuming the watch loop
done
