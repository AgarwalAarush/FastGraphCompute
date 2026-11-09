#!/usr/bin/env bash
# Run INSIDE an existing GPU allocation; see QUICK_RUN.md for one-time setup.
set -euo pipefail
campaign_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$campaign_repo"
exec "${FGC_CORE_PYTHON:-python}" -m revision_campaign.cli session "$@"
