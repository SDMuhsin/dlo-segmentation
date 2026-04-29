#!/bin/bash
# ============================================================================
# DEV-BOX -> RORQUAL LOGIN: bundle + rsync the data tar
# ============================================================================
#
# Where this runs: the DEV BOX (the machine where you have the cache and
# pretrained weights — i.e., this repo with data/ populated).  NOT on a
# rorqual login or compute node.
#
# What it does (no manual commands required):
#   1. Calls sbatch/bundle_phase5_data.sh to produce ./phase5_data.tar
#      (skips this step if the tar already exists and --skip-bundle is set).
#   2. rsyncs that tar to rorqual:/scratch/<user>/phase5_data.tar
#      (resumable via --partial; no inode cost — single file).
#
# The tar lives on rorqual scratch (not project) — scratch is roomier and
# the file gets purged on the cluster's normal scratch-retention schedule;
# the SLURM job extracts it at job start anyway, so there's no advantage
# to keeping it on /project.
#
# Required environment / args:
#   RORQUAL_USER   — your rorqual username (or pass --user <name>).
#   RORQUAL_HOST   — defaults to rorqual.alliancecan.ca (override --host).
#   REMOTE_TAR_DIR — defaults to /scratch/<user> (override --remote-dir).
#
# Usage:
#   ./sbatch/sync_to_rorqual.sh --user myname
#   ./sbatch/sync_to_rorqual.sh --user myname --skip-bundle      # reuse existing phase5_data.tar
#   ./sbatch/sync_to_rorqual.sh --user myname --zstd             # bundle with zstd compression
#   ./sbatch/sync_to_rorqual.sh --user myname --dry-run          # rsync --dry-run
#
# After this finishes:
#   ssh <user>@rorqual.alliancecan.ca
#   cd <project tree>           # e.g. ~/projects/def-seokbum/<user>/kiat_crefle
#   ./sbatch/bootstrap_rorqual_env.sh   # ONE-TIME — creates env/ on the cluster
#   ./sbatch/run_dformer_v2_dlo.sh --data-tar /scratch/<user>/phase5_data.tar
# ============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# DEFAULTS / CLI
# ----------------------------------------------------------------------------
RORQUAL_USER="${RORQUAL_USER:-}"
RORQUAL_HOST="${RORQUAL_HOST:-rorqual.alliancecan.ca}"
REMOTE_TAR_DIR=""
LOCAL_TAR="./phase5_data.tar"
USE_ZSTD=false
SKIP_BUNDLE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --user)         RORQUAL_USER="$2"; shift 2 ;;
        --host)         RORQUAL_HOST="$2"; shift 2 ;;
        --remote-dir)   REMOTE_TAR_DIR="$2"; shift 2 ;;
        --local-tar)    LOCAL_TAR="$2"; shift 2 ;;
        --zstd)         USE_ZSTD=true; shift 1 ;;
        --skip-bundle)  SKIP_BUNDLE=true; shift 1 ;;
        --dry-run)      DRY_RUN=true; shift 1 ;;
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

if [[ -z "$RORQUAL_USER" ]]; then
    echo "ERROR: rorqual username required.  Pass --user <name> or set RORQUAL_USER." >&2
    exit 1
fi

# Default remote dir is /scratch/<user> if not overridden
[[ -z "$REMOTE_TAR_DIR" ]] && REMOTE_TAR_DIR="/scratch/${RORQUAL_USER}"

# If --zstd, the bundling script writes phase5_data.tar.zst not .tar.
# Adjust LOCAL_TAR so the rest of the script tracks the right path.
if [[ "$USE_ZSTD" == "true" ]] && [[ "$LOCAL_TAR" != *.zst ]]; then
    LOCAL_TAR="${LOCAL_TAR%.tar}.tar.zst"
fi

REMOTE_TAR_PATH="${REMOTE_TAR_DIR}/$(basename "$LOCAL_TAR")"
REMOTE_TARGET="${RORQUAL_USER}@${RORQUAL_HOST}:${REMOTE_TAR_PATH}"

# ----------------------------------------------------------------------------
# 1. BUNDLE  (calls sbatch/bundle_phase5_data.sh)
# ----------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "$SKIP_BUNDLE" == "true" ]]; then
    if [[ ! -f "$LOCAL_TAR" ]]; then
        echo "ERROR: --skip-bundle but $LOCAL_TAR does not exist.  Drop the flag to build it." >&2
        exit 1
    fi
    echo "Reusing existing $LOCAL_TAR ($(stat -c '%s' "$LOCAL_TAR") B)"
else
    echo "================================================================"
    echo "Step 1/2: Bundling -> $LOCAL_TAR"
    echo "================================================================"
    bundle_args=(--output "$LOCAL_TAR")
    [[ "$USE_ZSTD" == "true" ]] && bundle_args+=(--zstd)
    ./sbatch/bundle_phase5_data.sh "${bundle_args[@]}"
fi

# ----------------------------------------------------------------------------
# 2. RSYNC TO LOGIN NODE
# ----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Step 2/2: rsync to rorqual login"
echo "  Source:      $LOCAL_TAR ($(stat -c '%s' "$LOCAL_TAR") B)"
echo "  Destination: $REMOTE_TARGET"
echo "================================================================"

# Ensure the remote directory exists (cheap; idempotent).  We do this with a
# single ssh hop before rsync so a missing /scratch/<user> doesn't cause an
# obscure rsync error.
if [[ "$DRY_RUN" != "true" ]]; then
    ssh "${RORQUAL_USER}@${RORQUAL_HOST}" "mkdir -p '$REMOTE_TAR_DIR'"
fi

# rsync flags:
#   -P  = --partial --progress (resumable + bytes/sec)
#   -e ssh = use ssh transport (default, but explicit is friendlier with
#            non-default keys/agents)
#   --inplace = overwrite the destination in-place rather than via a temp
#               file (cheaper on /scratch where the file may be ~14 GB)
rsync_flags=(-P --partial --inplace -e ssh)
[[ "$DRY_RUN" == "true" ]] && rsync_flags+=(--dry-run)

rsync "${rsync_flags[@]}" "$LOCAL_TAR" "$REMOTE_TARGET"

# ----------------------------------------------------------------------------
# DONE
# ----------------------------------------------------------------------------
echo ""
echo "================================================================"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run complete.  No bytes transferred."
else
    echo "Transfer complete."
fi
echo ""
echo "Next steps (run on the rorqual LOGIN node, in the project tree):"
echo "  ./sbatch/bootstrap_rorqual_env.sh           # one-time, creates env/"
echo "  ./sbatch/run_dformer_v2_dlo.sh --smoke      \\"
echo "      --data-tar $REMOTE_TAR_PATH"
echo "  ./sbatch/run_dformer_v2_dlo.sh              \\"
echo "      --data-tar $REMOTE_TAR_PATH"
echo "================================================================"
