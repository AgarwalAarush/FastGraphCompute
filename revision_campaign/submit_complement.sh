#!/usr/bin/env bash
# Run from a CPU allocation on Falcon. Bounded GPU chain, no automatic resubmission.
set -euo pipefail
plan=${1:?usage: submit_complement.sh PLAN CALIBRATION CORE_PYTHON [CALIBRATION_JOB_ID]}
calibration=${2:?supply calibration JSON path}
python=${3:?supply core Python executable}
calibration_job=${4:-}
last_slot_minutes=${FGC_LOCAL_LAST_SLOT_MINUTES:-100}
if [[ ! $last_slot_minutes =~ ^[0-9]+$ ]] || (( last_slot_minutes < 6 || last_slot_minutes > 100 )); then
  echo "FGC_LOCAL_LAST_SLOT_MINUTES must be an integer from 6 to 100" >&2
  exit 2
fi
campaign_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$campaign_repo"
export FGC_CAMPAIGN_ROOT="$campaign_repo"
run_root=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["run_root"])' "$plan")
mkdir -p "$run_root/logs"
manifest="$run_root/local-submission.tsv"
if [[ -e $manifest ]]; then
  echo "Submission already recorded at $manifest; inspect those IDs before submitting anything else" >&2
  exit 2
fi
(set -o noclobber; printf 'kind\tjob_id\tminutes\tdependency\n' > "$manifest")
dependency=
if [[ -n $calibration_job ]]; then
  [[ $calibration_job =~ ^[0-9]+$ ]]
  dependency="afterok:$calibration_job"
fi
for slot in 1 2 3; do
  walltime=04:00:00
  minutes=235
  if [[ $slot == 3 ]]; then
    printf -v walltime '%02d:%02d:00' "$((last_slot_minutes / 60))" "$((last_slot_minutes % 60))"
    minutes=$((last_slot_minutes - 5))
  fi
  args=(--parsable --chdir="$campaign_repo" --export=ALL --no-requeue --kill-on-invalid-dep=yes
    --partition=work --qos=heavy --gres=mps:100 --cpus-per-task=4 --mem=32G
    --time="$walltime" --job-name="fgcr_local_$slot"
    --output="$run_root/logs/local_slot${slot}_%j.log" --error="$run_root/logs/local_slot${slot}_%j.err")
  if [[ -n $dependency ]]; then args+=(--dependency="$dependency"); fi
  gpu=$(sbatch "${args[@]}" "$campaign_repo/revision_campaign/local_slurm_job.sh" "$plan" "$calibration" "$minutes" "$python")
  gpu=${gpu%%;*}
  [[ $gpu =~ ^[0-9]+$ ]]
  printf 'gpu\t%s\t%s\t%s\n' "$gpu" "$minutes" "$dependency" >> "$manifest"
  echo "Submitted mps:100 slot $slot: $gpu"
  cpu=$(sbatch --parsable --chdir="$campaign_repo" --export=ALL --no-requeue --kill-on-invalid-dep=yes \
    --partition=work --cpus-per-task=2 --mem=8G --time=00:20:00 \
    --job-name="fgcr_plot_$slot" --dependency="afterany:$gpu" \
    --output="$run_root/logs/collect_slot${slot}_%j.log" --error="$run_root/logs/collect_slot${slot}_%j.err" \
    "$campaign_repo/revision_campaign/local_collect.sh" "$plan" "$python")
  cpu=${cpu%%;*}
  [[ $cpu =~ ^[0-9]+$ ]]
  printf 'cpu\t%s\t0\tafterany:%s\n' "$cpu" "$gpu" >> "$manifest"
  dependency="afterok:$gpu:$cpu"
done
echo "Recorded all GPU/CPU jobs in $manifest"
