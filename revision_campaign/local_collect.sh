#!/usr/bin/env bash
# CPU-only postprocessing, including partial results after a canceled/failed GPU job.
set -euo pipefail
plan=${1:?supply the private plan}
python=${2:?supply core Python executable}
campaign_repo=${FGC_CAMPAIGN_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z $campaign_repo ]]; then
  campaign_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
[[ -f $campaign_repo/revision_campaign/__init__.py ]] || {
  echo "Set FGC_CAMPAIGN_ROOT to the operator checkout before submitting this job" >&2
  exit 2
}
cd "$campaign_repo"
"$python" -m revision_campaign.cli report --plan "$plan" --plots
run_root=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["run_root"])' "$plan")
"$python" -m revision_campaign.cli export-results --plan "$plan" \
  --out "$run_root/returns/local-final-${SLURM_JOB_ID:-manual}.zip"
