#!/bin/bash
# P28: wait for MovingCables_small.tar download → extract → convert → build co-train → validate.
# Stops BEFORE training so the built co-train set can be checked first.
set -euo pipefail
cd /workspace/kiat_crefle
source env/bin/activate
export HF_HOME=data/hf_home TORCH_HOME=data/torch_home

TAR=data/movingcables_raw/MovingCables_small.tar

echo "[p28] waiting for download (wget) to finish..."
while pgrep -f "wget -q -O MovingCables_small" >/dev/null 2>&1; do sleep 30; done
sz=$(stat -c%s "$TAR" 2>/dev/null || echo 0)
echo "[p28] wget gone; tar size=$(awk "BEGIN{printf \"%.2f\", $sz/1e9}")GB"
if [ "$sz" -lt 11000000000 ]; then echo "[p28] FAIL: incomplete download ($sz bytes)"; exit 1; fi
if ! tar -tf "$TAR" >/dev/null 2>&1; then echo "[p28] FAIL: tar not valid"; exit 1; fi

echo "[p28] extracting small split..."
( cd data/movingcables_raw && tar -xf MovingCables_small.tar )

echo "[p28] convert -> R1 (MovingCables-only)..."
./env/bin/python src/convert_movingcables_to_dformer.py \
  --raw-root data/movingcables_raw --out-dir data/dformer_dataset_movingcables

echo "[p28] build -> R2 (P25 synth + MovingCables co-train)..."
./env/bin/python src/build_p28_cotrain.py

echo "[p28] domain-gap measurement (valset vs MovingCables, full set)..."
./env/bin/python src/p28_domain_gap_movingcables.py || true

echo "[p28] dataset validation..."
./env/bin/python src/validate_movingcables_dataset.py || true

echo "[p28] ===== PREP COMPLETE ====="
for f in data/dformer_dataset_movingcables/train.txt data/dformer_dataset_movingcables/test.txt \
         data/dformer_dataset_p28_cotrain/train.txt data/dformer_dataset_p28_cotrain/test.txt; do
  if [ -f "$f" ]; then echo "  $(wc -l < "$f") lines  $f"; fi
done
echo "[p28] R1=data/dformer_dataset_movingcables  R2=data/dformer_dataset_p28_cotrain"
echo "[p28] DONE — review the built sets before starting the two 19h training runs."
