#!/usr/bin/env bash
#SBATCH --partition=work
#SBATCH --qos=heavy
#SBATCH --gres=mps:100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
plan=${1:?supply the private local plan}
calibration=${2:?supply the passed drift-check JSON}
work_minutes=${3:?supply session minutes leaving five minutes before allocation end}
python=${4:?supply core Python executable}
campaign_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$campaign_repo"
if [[ ${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-} != 100 ]]; then
  echo "This local runner requires a Slurm mps:100 allocation" >&2
  exit 2
fi
started_seconds=$SECONDS
already_complete=$("$python" - "$plan" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
r = Path(p['config']['run_root'])
s = r / 'execution-summary.json'
i = r / 'session-plan.json'
print(int(s.is_file() and i.is_file() and json.loads(s.read_text()).get('status')=='complete'
    and json.loads(i.read_text()).get('plan_hash')==p['plan_hash']))
PY
)
if [[ $already_complete == 1 ]]; then
  echo "This local plan already completed; no additional GPU measurements are needed."
  exit 0
fi
# Gate on the actual drift result. If Slurm moves nodes, recheck that A100 too.
"$python" - "$calibration" "$plan" <<'PY'
import json, sys
from pathlib import Path
c = json.loads(Path(sys.argv[1]).read_text())
p = json.loads(Path(sys.argv[2]).read_text())
assert c['pass'], 'Timing drift gate failed; campaign will not run'
assert c['source']['commit'] == p['config']['library_commit']
assert c['mps_percent'] == 100
PY
same_node=$("$python" -c 'import json,os,sys; print(int(json.load(open(sys.argv[1])).get("node")==os.environ.get("SLURMD_NODENAME")))' "$calibration")
if [[ $same_node != 1 ]]; then
  source_path=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["library_source"])' "$plan")
  historical=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["historical_file"])' "$calibration")
  "$python" -m revision_campaign.calibrate --source "$source_path" --historical "$historical" \
    --out "${calibration%.json}-${SLURM_JOB_ID}.json"
fi
remaining=$((work_minutes * 60 - SECONDS + started_seconds))
if (( remaining < 60 )); then
  echo "Allocation work budget consumed by calibration" >&2
  exit 75
fi
export FGC_CORE_PYTHON="$python"
set +e
bash revision_campaign/run_complement.sh --plan "$plan" --minutes "$((remaining / 60))" --execute
status=$?
set -e
# The externally submitted chain is bounded. A checkpoint is not a submission failure.
if [[ $status == 75 ]]; then
  exit 0
fi
if [[ $status == 1 ]]; then
  resumable=$("$python" - "$plan" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
s = json.loads((Path(p['config']['run_root']) / 'execution-summary.json').read_text())
print(int(s['status']=='failed' and not s.get('fatal', True)))
PY
)
  if [[ $resumable == 1 ]]; then
    echo "Optional unit failures are recorded; the bounded next allocation may continue independent work."
    exit 0
  fi
fi
exit "$status"
