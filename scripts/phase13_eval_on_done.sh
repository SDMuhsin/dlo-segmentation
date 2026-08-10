#!/usr/bin/env bash
# Wait for a Phase 13 train to finish (report.json epochs_completed >= target),
# then run 4-video real-world inference + the ablation gate on the trained
# checkpoint, and print a per-video coverage summary vs the Phase 7 baseline.
#
# Usage:  ./scripts/phase13_eval_on_done.sh <run_dir> [target_epochs] [gpu] [max_wait_s]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; source env/bin/activate
RUN="${1:?usage: $0 <run_dir> [target_epochs] [gpu] [max_wait_s]}"; RUN="${RUN%/}"
TGT="${2:-80}"; GPU="${3:-0}"; MAXWAIT="${4:-54000}"
waited=0
while true; do
  if [ -f "$RUN/report.json" ]; then
    d=$(./env/bin/python -c "import json;print(int(json.load(open('$RUN/report.json')).get('epochs_completed',0))>=$TGT)" 2>/dev/null || echo False)
    [ "$d" = "True" ] && break
  fi
  if grep -q "giving up" "$RUN/watchdog.log" 2>/dev/null; then echo "WATCHDOG GAVE UP on $RUN"; exit 2; fi
  sleep 120; waited=$((waited+120))
  if [ "$waited" -ge "$MAXWAIT" ]; then echo "MAX WAIT exceeded for $RUN"; exit 3; fi
done
echo "=== TRAIN COMPLETE $RUN $(date) ==="
./env/bin/python -c "import json;r=json.load(open('$RUN/report.json'));print('best_iou_dlo=%.4f mIoU=%.4f epochs=%d'%(r['best_iou_dlo'],r['best_miou'],r['epochs_completed']))"
echo "=== 4-video inference ==="
CUDA_VISIBLE_DEVICES="$GPU" bash scripts/run_phase7_repro_inference.sh "$RUN" 2>&1 | tail -6
echo "=== gate on trained ckpt ==="
CUDA_VISIBLE_DEVICES="$GPU" ./env/bin/python -u src/phase10_ablation_gate.py --ckpt "$RUN/best_model.pth" --out-dir "$RUN/gate" 2>&1 | tail -10
echo "=== per-video coverage vs Phase 7 (6.77/12.37/12.53/12.28) ==="
./env/bin/python - "$RUN" <<'PY'
import csv,sys,statistics as st
run=sys.argv[1]; base=[6.77,12.37,12.53,12.28]
for i in range(1,5):
    p=f"{run}/real_video_dlo_real_sample_{i}/per_frame.csv"
    try:
        v=[float(r["dlo_coverage_pct"]) for r in csv.DictReader(open(p))]
        m=st.mean(v); md=st.median(v); b=base[i-1]; d=100*(m-b)/b
        flag="OK" if (abs(d)<=15 and m>=5) else ("LOW" if m<5 else "OFF")
        print(f"sample_{i}: mean={m:.2f}% median={md:.2f}% (P7 {b}) delta={d:+.1f}% [{flag}]")
    except Exception as e:
        print(f"sample_{i}: ERROR {e}")
PY
echo "=== EVAL DONE $RUN $(date) ==="
