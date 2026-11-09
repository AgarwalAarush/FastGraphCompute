#!/usr/bin/env bash
set -euo pipefail
campaign_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$campaign_repo"
if [[ -n ${FGC_SITE_ENV:-} ]]; then
  source "$FGC_SITE_ENV"
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTHONHASHSEED=0
# exec lets Slurm's signal reach the Python worker; no resubmission traps.
exec "$@"
