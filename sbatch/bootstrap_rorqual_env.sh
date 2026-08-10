#!/bin/bash
# ============================================================================
# RORQUAL LOGIN NODE: create env/ via Alliance wheelhouse
# ============================================================================
#
# Where this runs: the rorqual LOGIN NODE, in the project root (where this
# script lives at sbatch/...).  NOT the dev box.  NOT a compute node.
#
# What it does (no manual commands required):
#   1. Loads the canonical Alliance module set:
#        module load StdEnv/2023      (the "standard" software stack)
#        module load python/3.11      (or compatible — see notes below)
#        module load scipy-stack      (numpy + scipy + matplotlib)
#        module load cuda cudnn       (so torch picks the CUDA-matched wheel)
#        module load arrow            (matches dt-pinn reference)
#      Order matters: StdEnv/python first (so subsequent modules pick the
#      right Python ABI), then scipy-stack, then cuda/cudnn, then arrow.
#
#   2. Creates env/ at the project root via `virtualenv --no-download` so
#      the venv binds to the just-loaded Alliance Python (NOT system python).
#
#   3. Activates env/ and installs the minimum packages train_dformer_v2_dlo.py
#      needs.  Tries the wheelhouse first via `pip install --no-index ...`;
#      packages not in the wheelhouse fall back to PyPI on a per-package
#      basis (login node has internet).
#
#   4. Smoke-imports torch + cuda + the project's own DFormer module to
#      catch the most common breakage early ("works on dev, fails on cluster").
#
# After this finishes, env/ is ready for both interactive use on the login
# node AND for the SLURM wrap to source.  The wrap (run_dformer_v2_dlo.sh)
# does NOT recreate env/ — it reuses what this script produced.
#
# Usage:
#   ./sbatch/bootstrap_rorqual_env.sh                       # standard
#   ./sbatch/bootstrap_rorqual_env.sh --force               # rebuild env/ even if it exists
#   ./sbatch/bootstrap_rorqual_env.sh --python python/3.10  # pin a specific Python module
#
# If a package fails to install:
#   - The script keeps going (the failed package is reported at the end).
#   - You can retry that one package manually:
#       module load StdEnv/2023 python scipy-stack cuda cudnn arrow
#       source env/bin/activate
#       pip install <package>      # without --no-index, login node has internet
#
# Why we do NOT recreate env/ on every job:
#   Recreating in $SLURM_TMPDIR per job is faster start-time (no NFS lookups)
#   but adds 2-5 min of pip install at every submission and won't catch
#   package failures until job-time.  The "shared env/ in project tree"
#   pattern that dt-pinn uses keeps the path identical across nodes (login
#   sees /home, compute sees the same /home via the cluster network FS) and
#   amortises install over many jobs.  Trade-off accepted.
# ============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# CLI ARGS
# ----------------------------------------------------------------------------
PYTHON_MODULE="python"   # let module system pick the most recent compatible
FORCE_REBUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --python)  PYTHON_MODULE="$2"; shift 2 ;;
        --force)   FORCE_REBUILD=true; shift 1 ;;
        -h|--help)
            sed -n '2,/^# ===$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ----------------------------------------------------------------------------
# PRE-FLIGHT
# ----------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: 'module' not in PATH.  This script is intended for an Alliance" >&2
    echo "       Canada cluster (rorqual / cedar / graham / narval).  If you're" >&2
    echo "       on the dev box, use ./env/bin/activate directly." >&2
    exit 1
fi

ENV_DIR="env"
if [[ -d "$ENV_DIR" ]]; then
    if [[ "$FORCE_REBUILD" == "true" ]]; then
        echo "Removing existing $ENV_DIR (--force)..."
        rm -rf "$ENV_DIR"
    else
        echo "ERROR: $ENV_DIR/ already exists.  Pass --force to rebuild." >&2
        exit 1
    fi
fi

# ----------------------------------------------------------------------------
# STEP 1: MODULE LOAD (canonical Alliance order)
# ----------------------------------------------------------------------------
echo "================================================================"
echo "Step 1/4: module load"
echo "================================================================"

# Make `module` work in non-interactive shells (lmod ships an autoload
# function; on some Alliance images the function isn't sourced for
# `bash -c` / scripts.  Force-source the public lmod init).
#
# Relax `set -u` around the cvmfs source + module loads: Alliance's
# /cvmfs/.../profile/bash.sh and Lmod's internals reference variables
# (e.g. SKIP_CC_CVMFS) without `${var:-}` defaults, which aborts under
# nounset.  Re-enable -u immediately after.
set +u
if [[ -r /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]]; then
    # shellcheck disable=SC1091
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
fi

module --force purge
module load StdEnv/2023
module load "$PYTHON_MODULE"
module load scipy-stack
module load cuda cudnn
module load arrow
# OpenCV on Alliance is shipped as a system module (cv2 importable via cvmfs
# PYTHONPATH); the wheelhouse intentionally ships an `opencv-noinstall`
# dummy wheel that errors on install.  Load BEFORE virtualenv creation so
# the venv inherits the cvmfs path that exposes cv2.
module load opencv
set -u

echo ""
echo "Loaded:"
module list 2>&1 | sed 's/^/  /'

# Verify the Python we just got and the cuda module match expectations
echo ""
echo "Python: $(python3 --version) at $(which python3)"
echo "CUDA module: ${CUDA_HOME:-<unset>}"

# ----------------------------------------------------------------------------
# STEP 2: virtualenv
# ----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Step 2/4: virtualenv --no-download $ENV_DIR"
echo "================================================================"

virtualenv --no-download "$ENV_DIR"

# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"
pip install --no-index --upgrade pip

# ----------------------------------------------------------------------------
# STEP 3: install packages
# ----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Step 3/4: pip install (wheelhouse first, PyPI fallback)"
echo "================================================================"

# Pin torch + torchvision to a KNOWN-COMPATIBLE pair.
#
# Background: the Alliance wheelhouse's torchvision metadata is too loose
# to be trusted — pip will happily pair (e.g.) torchvision 0.21 with torch
# 2.1.2 and let it crash at runtime on `torch.library.register_fake` (added
# in torch 2.4).  Co-installing in one pip call is NOT enough; we have to
# pin explicitly.
#
# torch 2.1.2 + torchvision 0.16.2 is the upstream-compatible pair for the
# rorqual gentoo2023/x86-64-v3 wheelhouse default.  If the wheelhouse ever
# ships a newer torch, bump both numbers to a matching pair (see
# https://github.com/pytorch/vision#installation for the official torch ↔
# torchvision compatibility table).
TORCH_VER=2.1.2
TORCHVISION_VER=0.16.2
echo "install torch==$TORCH_VER + torchvision==$TORCHVISION_VER ..."
if pip install --no-index "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER" >/tmp/pip_$$.log 2>&1; then
    echo "  ok (wheelhouse, pinned compatible pair)"
elif pip install "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER" >/tmp/pip_$$.log 2>&1; then
    echo "  ok (PyPI fallback)"
else
    echo "  FAILED"
    tail -30 /tmp/pip_$$.log | sed 's/^/  /'
    rm -f /tmp/pip_$$.log
    exit 1
fi
rm -f /tmp/pip_$$.log

# Packages needed by src/train_dformer_v2_dlo.py and the dformer code.
# Versions kept loose so the wheelhouse can pick what's actually built.
# Order: stable / common packages first, mmcv last (it's the most fragile
# and frequently the one that needs PyPI fallback).
# NOTE: opencv-python-headless is intentionally NOT here — Alliance ships
# OpenCV as a module (loaded above), and the wheelhouse opencv wheel is a
# dummy that errors on install by design.
WHEELHOUSE_PKGS=(
    numpy
    scipy
    matplotlib
    pillow
    tqdm
    scikit-learn
    huggingface_hub
    tensorboardX
    timm
    easydict
    termcolor
    mmengine
    mmcv
)

# Try wheelhouse first; fall back to PyPI per-package.  We don't do a single
# `pip install --no-index <all>` because if any one package is missing from
# the wheelhouse, pip aborts the whole transaction.  Per-package install
# is slower but catches partial-coverage gracefully.
FAILED_PKGS=()
INSTALLED_PKGS=()
PYPI_FALLBACK=()

for pkg in "${WHEELHOUSE_PKGS[@]}"; do
    printf 'install %-28s ... ' "$pkg"
    if pip install --no-index "$pkg" >/tmp/pip_$$.log 2>&1; then
        echo "ok (wheelhouse)"
        INSTALLED_PKGS+=("$pkg")
    elif pip install "$pkg" >/tmp/pip_$$.log 2>&1; then
        echo "ok (PyPI)"
        INSTALLED_PKGS+=("$pkg")
        PYPI_FALLBACK+=("$pkg")
    else
        echo "FAILED"
        echo "  --- last 20 lines of pip log ---"
        tail -20 /tmp/pip_$$.log | sed 's/^/  /'
        FAILED_PKGS+=("$pkg")
    fi
done
rm -f /tmp/pip_$$.log

# ----------------------------------------------------------------------------
# STEP 4: SMOKE IMPORT
# ----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Step 4/4: smoke-import"
echo "================================================================"

python3 - <<'PY'
import sys
import traceback

CHECKS = [
    ("torch + CUDA",   "import torch; print('  torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"),
    ("torchvision",    "import torchvision; print('  torchvision', torchvision.__version__)"),
    ("numpy/scipy",    "import numpy, scipy; print('  numpy', numpy.__version__, 'scipy', scipy.__version__)"),
    ("opencv (cv2)",   "import cv2; print('  cv2', cv2.__version__)"),
    ("mmcv/mmengine",  "import mmcv, mmengine; print('  mmcv', mmcv.__version__, 'mmengine', mmengine.__version__)"),
    ("timm/easydict",  "import timm, easydict; print('  timm', timm.__version__)"),
    ("hf_hub/tbX",     "import huggingface_hub, tensorboardX; print('  huggingface_hub', huggingface_hub.__version__)"),
    ("dformer code",   "sys.path.insert(0, 'src/dformer'); from models.builder import EncoderDecoder; print('  EncoderDecoder importable')"),
    ("training script","sys.path.insert(0, 'src'); import train_dformer_v2_dlo; print('  train_dformer_v2_dlo importable')"),
]

failed = []
for name, code in CHECKS:
    print(f'[{name}]')
    try:
        exec(code, {'sys': sys})
    except Exception:
        traceback.print_exc()
        failed.append(name)

if failed:
    print(f'\nFAILED: {failed}')
    sys.exit(1)
print('\nAll smoke imports passed.')
PY

# ----------------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Bootstrap summary"
echo "================================================================"
echo "Installed (${#INSTALLED_PKGS[@]}): ${INSTALLED_PKGS[*]}"
if [[ ${#PYPI_FALLBACK[@]} -gt 0 ]]; then
    echo ""
    echo "Pulled from PyPI (login-node internet) instead of wheelhouse:"
    printf '  - %s\n' "${PYPI_FALLBACK[@]}"
    echo "  (These may need to be re-checked on each scipy-stack module bump."
    echo "   They will already be installed in env/, so the SLURM job won't try"
    echo "   to fetch them.)"
fi
if [[ ${#FAILED_PKGS[@]} -gt 0 ]]; then
    echo ""
    echo "FAILED to install (${#FAILED_PKGS[@]}):"
    printf '  - %s\n' "${FAILED_PKGS[@]}"
    echo ""
    echo "Re-run with --force after fixing, or install manually:"
    echo "  source env/bin/activate"
    printf "  pip install %s\n" "${FAILED_PKGS[*]}"
    exit 1
fi

echo ""
echo "Done.  env/ is ready.  Next:"
echo "  ./sbatch/run_dformer_v2_dlo.sh --smoke --data-tar /scratch/\$USER/phase5_data.tar"
