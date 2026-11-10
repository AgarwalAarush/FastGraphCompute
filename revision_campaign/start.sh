#!/usr/bin/env bash
# One-time setup and synthetic pilot inside an existing full-GPU allocation.
set -euo pipefail
campaign_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$campaign_repo"
execute=false
site_env=
for ((i=1; i<=$#; i++)); do
  argument=${!i}
  if [[ $argument == --execute ]]; then execute=true; fi
  if [[ $argument == --site-env ]]; then
    j=$((i + 1))
    site_env=${!j:?--site-env requires a file}
  fi
done
if [[ $execute == false ]]; then
  cat <<'HELP'
FastGraph setup + synthetic pilot (requires an existing full-GPU allocation).

bash revision_campaign/start.sh --work /private/fastgraph-work --gpu H100 --full-gpu --execute

Use a NEW empty work directory. It keeps resumable setup, private outputs and ZIPs.
Requirements: Linux, Conda, git, Python 3.10+, and a supported CUDA toolkit/nvcc.
Options: --gpu A100|H100 (default H100), --minutes 25, --site-env /path/modules.sh.
Without --execute this prints help only and changes nothing.

It clones pinned sources, installs four environments, builds FGC/counters,
runs CPU and GPU checks, then runs/resumes the synthetic pilot and exports a ZIP.
Setup/build time is ADDITIONAL to --minutes: use a longer allocation initially.
Repeat the identical command after setup to use a 30-minute debug allocation.
Exit 75 means partial pilot progress; repeat in another allocation to resume.

This does not request GPU allocations or train GravNet. Model/event production
requires the owner's real model, checkpoints and confirmed data; see HANDOFF.md.
HELP
  exit 0
fi
if [[ -n $site_env ]]; then source "$site_env"; fi
for dependency in conda git nvcc nvidia-smi python3; do
  command -v "$dependency" >/dev/null || { echo "Missing $dependency; load the site's Conda/CUDA modules first" >&2; exit 2; }
done
nvidia-smi -L >/dev/null
umask 077
export FGC_CAMPAIGN_ROOT="$campaign_repo"
exec python3 -m revision_campaign.bootstrap "$@"
