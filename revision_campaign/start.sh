#!/usr/bin/env bash
# Setup in a compute allocation; a full GPU is required for the pilot.
set -euo pipefail
campaign_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$campaign_repo"
execute=false
setup_only=false
site_env=
for ((i=1; i<=$#; i++)); do
  argument=${!i}
  if [[ $argument == --execute ]]; then execute=true; fi
  if [[ $argument == --setup-only ]]; then setup_only=true; fi
  if [[ $argument == --site-env ]]; then
    j=$((i + 1))
    site_env=${!j:?--site-env requires a file}
  fi
done
if [[ $execute == false ]]; then
  cat <<'HELP'
FastGraph setup + synthetic pilot (CPU setup is optional; pilot needs a full GPU).

bash revision_campaign/start.sh --work /private/fastgraph-work --gpu H100 --full-gpu --execute

Use a NEW empty work directory. It keeps resumable setup, private outputs and ZIPs.
Requirements: Linux, Conda, git, Python 3.10+, and a supported CUDA toolkit/nvcc.
Options: --gpu A100|H100, --slot-minutes 30, --minutes 25, --site-env /path/modules.sh.
Add --setup-only in a CPU compute allocation to install/build without any GPU use.
Without --execute this prints help only and changes nothing.

It clones pinned sources, installs four environments, builds FGC/counters,
runs CPU and GPU checks, then runs/resumes the synthetic pilot and exports a ZIP.
The 30-minute slot covers setup + checks + pilot, reserving five minutes to stop/export.
A fresh install can consume a whole slot. Repeat the identical command in another
30-minute GPU allocation to continue saved setup/results. Setup can be prepared on CPU.
Exit 75 is a checkpoint; before the pilot starts there is setup progress and no results ZIP.

This does not request GPU allocations or train GravNet. Model/event production
requires the owner's real model, checkpoints and confirmed data; see HANDOFF.md.
HELP
  exit 0
fi
if [[ -n $site_env ]]; then source "$site_env"; fi
for dependency in conda git nvcc python3; do
  command -v "$dependency" >/dev/null || { echo "Missing $dependency; load the site's Conda/CUDA modules first" >&2; exit 2; }
done
if [[ $setup_only == false ]]; then
  command -v nvidia-smi >/dev/null || { echo "Run inside the intended GPU allocation" >&2; exit 2; }
  nvidia-smi -L >/dev/null
fi
umask 077
export FGC_CAMPAIGN_ROOT="$campaign_repo"
exec python3 -m revision_campaign.bootstrap "$@"
