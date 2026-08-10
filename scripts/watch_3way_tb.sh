#!/usr/bin/env bash
# Emit one line per NEW validation point, read from TensorBoard rather than the
# training log: the trainer's stdout is block-buffered through `tee`, so log
# lines arrive in 8 KB bursts and are useless for live monitoring, while TB
# event files are flushed per write.
#
# Emits on: each new eval, process exit (success or crash), and stall detection.
cd /workspace/kiat_crefle
source env/bin/activate
python -u - <<'PY'
import glob, os, time, sys
from tensorboard.backend.event_processing import event_accumulator as EA

ARMS = ["lovasz050", "lovasz050_con4", "lovasz050_zoom", "lovasz050_aug"]
ROOT = "results/realism_campaign/p3w_lovasz"
seen = {a: set() for a in ARMS}
last_progress = time.time()

def alive():
    return os.popen("pgrep -fc 'train_rgb_only_sota.*p3w_lovasz'").read().strip()

while True:
    new_any = False
    for arm in ARMS:
        for f in glob.glob(f"{ROOT}/{arm}/tb/events*"):
            try:
                a = EA.EventAccumulator(f, size_guidance={"scalars": 0}); a.Reload()
                tags = a.Tags()["scalars"]
                if "val/miou" not in tags:
                    continue
                mi = {s.step: s.value for s in a.Scalars("val/miou")}
                co = {s.step: s.value for s in a.Scalars("val/iou_connector")}
                wi = {s.step: s.value for s in a.Scalars("val/iou_wire")}
                for st in sorted(mi):
                    if st in seen[arm]:
                        continue
                    seen[arm].add(st); new_any = True
                    # Throttle: per-eval chatter costs more than it informs at
                    # this noise level (connector eval-to-eval sigma ~0.01-0.02).
                    # Report every 6th epoch and the last few, where the
                    # stable-level verdict actually lives.
                    if st % 6 and st < 26:
                        continue
                    print(f"{arm} ep{st:>3}  con={co.get(st,0):.4f}  "
                          f"wire={wi.get(st,0):.4f}  mIoU={mi[st]:.4f}")
            except Exception:
                pass
    if new_any:
        last_progress = time.time()
    n = alive()
    if n == "0":
        print(f"RUNS_EXITED (no train_rgb_only_sota processes left)")
        sys.exit(0)
    if time.time() - last_progress > 3600:
        print(f"WARNING stall: no new eval in 60 min (procs alive={n})")
        last_progress = time.time()
    time.sleep(60)
PY
