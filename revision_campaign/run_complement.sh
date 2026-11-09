#!/usr/bin/env bash
# Same reviewed plan; begin in the middle, then backfill the highest-priority gaps.
set -euo pipefail
campaign_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$campaign_repo"
exec "${FGC_CORE_PYTHON:-python}" -m revision_campaign.cli session --strategy complement "$@"
