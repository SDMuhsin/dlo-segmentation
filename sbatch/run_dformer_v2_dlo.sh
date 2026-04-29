#!/bin/bash
# ============================================================================
# Phase 5 — DFormer-v2-Large Binary DLO Segmentation (rorqual / SLURM)
# ============================================================================
#
# WHERE THIS RUNS: the rorqual LOGIN NODE (you `ssh` in, `cd` to the project
# tree, run this script).  It calls `sbatch --wrap=...`, which queues the
# WRAP body to a GPU compute node.  Login and GPU nodes share /home, /scratch,
# /project — same paths everywhere.  This script is NOT for the dev box.
#
# THREE-NODE PIPELINE (all manual steps are scripted; do not run commands by
# hand):
#
#                    ┌────────────────────────┐
#                    │    DEV BOX (here)      │
#                    │  cache + pretrained    │
#                    └──────────┬─────────────┘
#                               │  sbatch/sync_to_rorqual.sh
#                               │  (bundles tar, rsyncs to login)
#                               ▼
#                    ┌────────────────────────┐
#                    │  RORQUAL LOGIN NODE    │
#                    │  /scratch/<user>/      │
#                    │     phase5_data.tar    │
#                    │  <project>/env/        │← bootstrap_rorqual_env.sh
#                    └──────────┬─────────────┘
#                               │  THIS SCRIPT (run_dformer_v2_dlo.sh)
#                               │  → sbatch --wrap
#                               ▼
#                    ┌────────────────────────┐
#                    │  RORQUAL GPU NODE      │
#                    │  $SLURM_TMPDIR/data/   │← extracted tar
#                    │  torchrun → DDP train  │
#                    └────────────────────────┘
#
# Anchor numbers (from local 2× A40 smoke (c), 2026-04-28):
#   - 9.25 img/s, 13:37 / epoch (full 7,560 train, batch 4/GPU, FP16 AMP)
#   - 18.6 GB allocated peak / GPU at batch 4
#   - Projected on 4× H100 80GB: ≈ 3:30 / epoch, 80 epochs ≈ 4.5 h wall
#   - Default --time of 6 h leaves ~30% margin over projection
#
# Modes:
#   --smoke     1× H100 80GB, 1 h, 2 epochs, batch 4 — verifies the wrap end
#               to end on rorqual (modules, venv, extract, DDP) before
#               committing to a long run
#   (default)   4× H100 80GB DDP, 6 h, 80 epochs, batch 8/GPU, lr 1e-4
#
# Pre-requisites (do them in this order, once):
#
#   1. ON DEV:        ./sbatch/bundle_phase5_data.sh --output ./phase5_data.tar
#                     (script — produces the ~14 GB tar)
#
#   2. ON DEV:        rsync the tar to rorqual /scratch (MANUAL — 2FA prompt):
#                       rsync -P --partial --inplace ./phase5_data.tar \
#                         <user>@rorqual.alliancecan.ca:/scratch/<user>/
#
#   3. ON LOGIN:      ./sbatch/bootstrap_rorqual_env.sh
#                     (script — creates env/ with the right Alliance modules+wheels)
#
# Then per submission, ON LOGIN:
#
#   ./sbatch/run_dformer_v2_dlo.sh --smoke \
#       --data-tar /scratch/<user>/phase5_data.tar
#
#   ./sbatch/run_dformer_v2_dlo.sh \
#       --data-tar /scratch/<user>/phase5_data.tar
#
# Each script refuses to run on the wrong node (e.g. bootstrap exits if
# `module` isn't in PATH).  See llmdocs/trackers/phase5_dformer_v2_tracker.md
# for the full pipeline diagram.
#
# Output (on /home or /project, since the GPU node shares them with login):
#   - results/dformer_v2_dlo/<tag>/{best_model.pth, epoch_*.pth,
#                                    report.json, tb/, training.log}
#   - logs/dformer_v2_dlo_<tag>-<node>-<jobid>.{out,err}
#
# ============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# CLI ARGUMENTS
# ----------------------------------------------------------------------------
ACCOUNT="def-seokbum"
N_GPUS=4
GPU_TYPE="nvidia_h100_80gb_hbm3"   # full H100 80GB on rorqual; resolves to "${GPU_TYPE}:${N_GPUS}"
TIME="0-06:00:00"
MEM_PER_GPU_M=32000                # 32 GB host RAM per GPU (mmap cache RAM-resident)
CPUS_PER_GPU=4
EPOCHS=80
BATCH_PER_GPU=8
LR="1e-4"
WARMUP_EPOCHS=10
EVAL_EVERY=5
CKPT_EVERY=20
DLO_WEIGHT=6.0
TAG_OVERRIDE=""
SMOKE=false
DRY_RUN=false                       # --dry-run prints sbatch invocation but does not submit
DATA_TAR=""                          # path to bundled tar (e.g. /scratch/$USER/phase5_data.tar)

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)        ACCOUNT="$2"; shift 2 ;;
        --n-gpus)         N_GPUS="$2"; shift 2 ;;
        --gpu-type)       GPU_TYPE="$2"; shift 2 ;;
        --time)           TIME="$2"; shift 2 ;;
        --mem-per-gpu-m)  MEM_PER_GPU_M="$2"; shift 2 ;;
        --cpus-per-gpu)   CPUS_PER_GPU="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --batch-size)     BATCH_PER_GPU="$2"; shift 2 ;;
        --lr)             LR="$2"; shift 2 ;;
        --warmup-epochs)  WARMUP_EPOCHS="$2"; shift 2 ;;
        --eval-every)     EVAL_EVERY="$2"; shift 2 ;;
        --ckpt-every)     CKPT_EVERY="$2"; shift 2 ;;
        --dlo-weight)     DLO_WEIGHT="$2"; shift 2 ;;
        --tag)            TAG_OVERRIDE="$2"; shift 2 ;;
        --data-tar)       DATA_TAR="$2"; shift 2 ;;
        --smoke)          SMOKE=true; shift 1 ;;
        --dry-run)        DRY_RUN=true; shift 1 ;;
        -h|--help)
            sed -n '1,/^# ===.*Usage/p' "$0" | sed -n '/^# Usage:/,/^# Pre-flight/p' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# ----------------------------------------------------------------------------
# SMOKE MODE OVERRIDE
# ----------------------------------------------------------------------------
# On rorqual, "smoke" means a tiny 1-GPU job to verify modules + env + DDP
# wire-up + cache visibility. Use this before any long DDP run on a node
# you've never run on.
if [[ "$SMOKE" == "true" ]]; then
    N_GPUS=1
    TIME="0-01:00:00"
    EPOCHS=2
    BATCH_PER_GPU=4
    EVAL_EVERY=1
    CKPT_EVERY=2
    WARMUP_EPOCHS=0
fi

# ----------------------------------------------------------------------------
# RESOLVE TAG, JOB NAME, RESOURCE STRINGS
# ----------------------------------------------------------------------------
DEFAULT_TAG="hpc_$(date +%Y%m%d_%H%M)"
[[ "$SMOKE" == "true" ]] && DEFAULT_TAG="hpc_smoke_$(date +%Y%m%d_%H%M)"
TAG="${TAG_OVERRIDE:-$DEFAULT_TAG}"

JOB_NAME="dformer_v2_dlo_${TAG}"
RESULTS_DIR="results/dformer_v2_dlo/${TAG}"
LOG_FILE="./logs/dformer_v2_dlo_${TAG}"

# Compute totals
TOTAL_CPUS=$(( CPUS_PER_GPU * N_GPUS ))
TOTAL_MEM_M=$(( MEM_PER_GPU_M * N_GPUS ))
EFFECTIVE_BATCH=$(( BATCH_PER_GPU * N_GPUS ))

# GPU spec: rorqual format is `<gpu-type>:<count>`, where gpu-type is
# something like "nvidia_h100_80gb_hbm3" (full GPU) or
# "nvidia_h100_80gb_hbm3_2g.20gb" (MIG slice). MIG slices won't share NCCL
# rings — DDP requires full GPUs.
GPU_SPEC="${GPU_TYPE}:${N_GPUS}"

# Account flag (empty if account left blank)
ACCOUNT_FLAG=""
[[ -n "$ACCOUNT" ]] && ACCOUNT_FLAG="--account=$ACCOUNT"

# ----------------------------------------------------------------------------
# PRE-CREATE DIRS
# ----------------------------------------------------------------------------
mkdir -p ./logs
mkdir -p "$RESULTS_DIR"

# ----------------------------------------------------------------------------
# PRE-FLIGHT CHECKS
# ----------------------------------------------------------------------------
LOCAL_PRETRAINED="data/pretrained/DFormerv2/pretrained/DFormerv2_Large_pretrained.pth"
LOCAL_CACHE_DIR="data/dformer_dataset/cache"

if [[ -n "$DATA_TAR" ]]; then
    # HPC mode: tar will be extracted at job start.  Verify it exists and is
    # readable — we run on the login node, which sees the same /scratch FS
    # as the compute node, so this catches typos before sbatch queues a
    # doomed job.
    if [[ ! -f "$DATA_TAR" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "(dry-run) WARNING: --data-tar path does not exist: $DATA_TAR"
        else
            echo "ERROR: --data-tar path does not exist: $DATA_TAR" >&2
            echo "       Push it from the dev box first:" >&2
            echo "         (on dev) ./sbatch/sync_to_rorqual.sh --user \$RORQUAL_USER" >&2
            exit 1
        fi
    else
        tar_size_gb=$(awk -v b="$(stat -c '%s' "$DATA_TAR")" 'BEGIN { printf "%.2f", b / 1024 / 1024 / 1024 }')
        echo "Data tar:       $DATA_TAR (${tar_size_gb} GiB)"
        echo "                Will extract to \$SLURM_TMPDIR at job start."
    fi

    # HPC mode also requires env/ in the project tree (created by
    # sbatch/bootstrap_rorqual_env.sh on the login node).  The wrap activates
    # this env; if it's missing the job will die after the queue wait.
    if [[ ! -d "env" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "(dry-run) WARNING: ./env/ does not exist."
            echo "          On the login node, bootstrap once:"
            echo "            ./sbatch/bootstrap_rorqual_env.sh"
        else
            echo "ERROR: ./env/ does not exist on the login node." >&2
            echo "       Bootstrap once before submitting jobs:" >&2
            echo "         ./sbatch/bootstrap_rorqual_env.sh" >&2
            exit 1
        fi
    fi
else
    # Local mode (no tar) — assumes you're on a dev box / interactive node
    # that already has the cache + pretrained populated under data/.
    if [[ ! -f "$LOCAL_PRETRAINED" ]]; then
        echo "WARNING: pretrained backbone not found at $LOCAL_PRETRAINED"
    fi
    if [[ ! -f "$LOCAL_CACHE_DIR/train_rgb.npy" ]] || [[ ! -f "$LOCAL_CACHE_DIR/val_rgb.npy" ]]; then
        echo "WARNING: mmap cache not built at $LOCAL_CACHE_DIR — first run will build it."
    fi
    if [[ ! -d "env" ]]; then
        echo "WARNING: ./env/ does not exist (local mode also needs the venv)."
    fi
fi

# ----------------------------------------------------------------------------
# BANNER
# ----------------------------------------------------------------------------
echo "========================================================================"
echo "DFormer-v2-Large Binary DLO Segmentation — SLURM Submission"
echo "========================================================================"
[[ "$SMOKE" == "true" ]] && echo "Mode:           SMOKE (HPC dry-run)"
echo "Account:        ${ACCOUNT:-(unset — sbatch will pick default)}"
echo "GPUs:           $N_GPUS × $GPU_TYPE  (spec: $GPU_SPEC)"
echo "Time:           $TIME"
echo "Mem (host):     ${TOTAL_MEM_M}M  (${MEM_PER_GPU_M}M × $N_GPUS)"
echo "CPUs:           $TOTAL_CPUS  ($CPUS_PER_GPU × $N_GPUS)"
echo "Epochs:         $EPOCHS"
echo "Batch/GPU:      $BATCH_PER_GPU  (effective: $EFFECTIVE_BATCH)"
echo "LR:             $LR  (warmup $WARMUP_EPOCHS ep, poly decay)"
echo "Eval every:     $EVAL_EVERY ep"
echo "Ckpt every:     $CKPT_EVERY ep"
echo "DLO weight:     $DLO_WEIGHT"
echo "Tag:            $TAG"
if [[ -n "$DATA_TAR" ]]; then
    echo "Data:           tar -> \$SLURM_TMPDIR  (source: $DATA_TAR)"
else
    echo "Data:           local project tree (no --data-tar)"
fi
echo "Results:        $RESULTS_DIR"
echo "Logs:           ${LOG_FILE}-<node>-<jobid>.{out,err}"
echo ""

# ----------------------------------------------------------------------------
# SUBMIT
# ----------------------------------------------------------------------------
# The wrap below is the SLURM job body executed on the GPU compute node.
# Notes:
#   - `set -euo pipefail` propagates failures (a Python crash exits the job
#     non-zero so SLURM marks it FAILED — without this, a failed Python step
#     followed by an echo can leave SLURM thinking the job succeeded).
#   - The lmod profile is force-sourced because some Alliance images don't
#     auto-load it for non-interactive shells (sbatch wrap counts).
#   - Module load order matches sbatch/bootstrap_rorqual_env.sh exactly:
#     StdEnv → python → scipy-stack → cuda+cudnn → arrow.  Loading any
#     other order risks pulling a Python that doesn't match the venv ABI.
#   - `source env/bin/activate` AFTER module load — the venv was created
#     against the modules-loaded Python.  Activating before would expose
#     system Python and break torch.cuda.
#   - PYTHONPATH must include the project root for the cross-imports
#     in src/train_dformer_v2_dlo.py (it imports from src/dformer/ via
#     sys.path.insert and from train_rgbd_seg.build_cache).
#   - OMP_NUM_THREADS prevents per-rank thread oversubscription on multi-GPU.
#   - HF_HOME / TORCH_HOME / XDG_CACHE_HOME redirected to \$SLURM_TMPDIR so
#     huggingface_hub / torch.hub don't silently fill ~/.cache (counted
#     against home's 500k inode quota; will eventually fail logins).
#   - `--rdzv_endpoint=localhost:0` lets torchrun pick a free port for DDP,
#     avoiding the default 29500 if two jobs land on the same node.
#   - python3 -u — torchrun already passes -u, but listing it here makes
#     the assumption explicit if anyone replaces the launcher.

WRAP_BODY="
set -euo pipefail

# Some Alliance images don't auto-source lmod for non-interactive shells.
# Relax \`set -u\` around the cvmfs source + module loads: Alliance's
# /cvmfs/.../profile/bash.sh and Lmod's internals reference variables
# (e.g. SKIP_CC_CVMFS) without \`\${var:-}\` defaults, which aborts under
# nounset.  Re-enable -u immediately after.
set +u
if [[ -r /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]]; then
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
fi

module --force purge
module load StdEnv/2023
module load python
module load scipy-stack
module load cuda cudnn
module load arrow
set -u

source env/bin/activate
export PYTHONPATH=\"\$(pwd):\${PYTHONPATH:-}\"
export OMP_NUM_THREADS=$CPUS_PER_GPU

# Keep library caches on per-job NVMe, off /home (500k inode quota).
export HF_HOME=\"\$SLURM_TMPDIR/hf_home\"
export TORCH_HOME=\"\$SLURM_TMPDIR/torch_home\"
export XDG_CACHE_HOME=\"\$SLURM_TMPDIR/xdg_cache\"
mkdir -p \"\$HF_HOME\" \"\$TORCH_HOME\" \"\$XDG_CACHE_HOME\"

echo '======================================================================='
echo 'Job:        $JOB_NAME'
echo 'Tag:        $TAG'
echo 'Started:    '\$(date)
echo 'Node:       '\$(hostname)
echo 'Allocated CPUs: '\${SLURM_CPUS_PER_TASK:-?}
echo 'Allocated GPUs: '\${SLURM_GPUS_ON_NODE:-?}
echo 'CUDA_VISIBLE_DEVICES: '\${CUDA_VISIBLE_DEVICES:-?}
echo 'SLURM_TMPDIR: '\${SLURM_TMPDIR:-NOT SET}
df -h \"\$SLURM_TMPDIR\" || true
echo '======================================================================='
nvidia-smi
echo '----- python / torch versions -----'
python3 -c 'import torch; print(\"torch:\", torch.__version__); print(\"cuda:\", torch.version.cuda); print(\"n_gpus:\", torch.cuda.device_count())'
echo '======================================================================='

# ---- Stage data into \$SLURM_TMPDIR ----
DATA_TAR_PATH=\"$DATA_TAR\"
if [[ -n \"\$DATA_TAR_PATH\" ]]; then
    if [[ ! -f \"\$DATA_TAR_PATH\" ]]; then
        echo \"ERROR: data tar not found on compute node: \$DATA_TAR_PATH\" >&2
        exit 1
    fi
    echo \"Staging data: \$DATA_TAR_PATH -> \$SLURM_TMPDIR\"
    t0=\$(date +%s)
    if [[ \"\$DATA_TAR_PATH\" == *.zst ]]; then
        if ! command -v zstd >/dev/null; then
            echo 'ERROR: data tar is .zst but zstd not on PATH (no scipy-stack module helps)' >&2
            exit 1
        fi
        zstd -dc \"\$DATA_TAR_PATH\" | tar xf - -C \"\$SLURM_TMPDIR\"
    else
        tar xf \"\$DATA_TAR_PATH\" -C \"\$SLURM_TMPDIR\"
    fi
    extract_secs=\$(( \$(date +%s) - t0 ))
    echo \"Extracted in \${extract_secs}s\"
    DATA_DIR=\"\$SLURM_TMPDIR/data/dformer_dataset\"
    PRETRAINED_PATH=\"\$SLURM_TMPDIR/data/pretrained/DFormerv2/pretrained/DFormerv2_Large_pretrained.pth\"
else
    echo 'No --data-tar; using local project tree paths.'
    DATA_DIR=\"data/dformer_dataset\"
    PRETRAINED_PATH=\"data/pretrained/DFormerv2/pretrained/DFormerv2_Large_pretrained.pth\"
fi

# Sanity: verify what we'll feed to training is actually there.
for required in \\
    \"\$DATA_DIR/cache/train_rgb.npy\" \\
    \"\$DATA_DIR/cache/val_rgb.npy\" \\
    \"\$DATA_DIR/train.txt\" \\
    \"\$DATA_DIR/test.txt\" \\
    \"\$PRETRAINED_PATH\"; do
    if [[ ! -f \"\$required\" ]]; then
        echo \"ERROR: required input missing: \$required\" >&2
        echo 'Tar contents:' >&2
        ls -la \"\$DATA_DIR/cache/\" 2>&1 | head -20 >&2
        exit 1
    fi
done
echo \"Data staged. Train cache: \$(stat -c '%s' \"\$DATA_DIR/cache/train_rgb.npy\") B\"
echo '======================================================================='

torchrun \\
    --nproc_per_node=$N_GPUS \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=localhost:0 \\
    src/train_dformer_v2_dlo.py \\
    --data-dir \"\$DATA_DIR\" \\
    --pretrained \"\$PRETRAINED_PATH\" \\
    --epochs $EPOCHS \\
    --batch-size $BATCH_PER_GPU \\
    --lr $LR \\
    --warmup-epochs $WARMUP_EPOCHS \\
    --eval-every $EVAL_EVERY \\
    --ckpt-every $CKPT_EVERY \\
    --dlo-weight $DLO_WEIGHT \\
    --results-dir $RESULTS_DIR

echo '======================================================================='
echo 'Finished:   '\$(date)
echo '======================================================================='
"

SBATCH_CMD=(
    sbatch
    $ACCOUNT_FLAG
    --nodes=1
    --ntasks-per-node=1
    --cpus-per-task="$TOTAL_CPUS"
    --gpus="$GPU_SPEC"
    --mem="${TOTAL_MEM_M}M"
    --time="$TIME"
    --job-name="$JOB_NAME"
    --output="${LOG_FILE}-%N-%j.out"
    --error="${LOG_FILE}-%N-%j.err"
    --wrap="$WRAP_BODY"
)

if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN — would submit:"
    printf '  %q ' "${SBATCH_CMD[@]}"
    echo ""
    exit 0
fi

echo "Submitting..."
"${SBATCH_CMD[@]}"
echo ""
echo "Watch logs:   tail -F ${LOG_FILE}-*-*.out"
echo "Watch queue:  squeue -u \$USER -n $JOB_NAME"
