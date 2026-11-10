#!/usr/bin/env bash
set -euo pipefail
campaign_repo=${FGC_CAMPAIGN_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z $campaign_repo ]]; then
  campaign_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
[[ -f $campaign_repo/revision_campaign/__init__.py ]] || {
  echo "Set FGC_CAMPAIGN_ROOT to the operator checkout before submitting this job" >&2
  exit 2
}
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
