#!/usr/bin/env bash
# Periodic checkpoint: push runner edits, helper scripts, prepared OH patches,
# and per-commit result JSONs to the icml branch on GitHub. Skips heavy data
# (HF dataset cache, udocker images) which are reproducible.
#
# Run via cron every 60 min. Idempotent — does nothing if no changes.

set -uo pipefail
cd /root/OmniPerf-Bench

BRANCH="icml/rebuttal-hard-metrics-oh"
LOG_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LOG_PREFIX="[checkpoint $LOG_TS]"

# Ensure on the right branch (without disrupting any pending work)
CUR=$(git rev-parse --abbrev-ref HEAD)
if [ "$CUR" != "$BRANCH" ]; then
    echo "$LOG_PREFIX SKIP: on branch $CUR, expected $BRANCH"
    exit 0
fi

# Stage everything we care about. Never use git add -A — would pull in HF
# downloads, udocker container blobs, log files, etc.
git add scripts/runners/run_3way_benchmarks.py 2>/dev/null
git add scripts/runners/prepare_oh_patches.py 2>/dev/null
git add scripts/runners/run_oh_fanout.sh 2>/dev/null
git add scripts/runners/checkpoint_progress.sh 2>/dev/null
git add docs/HARD_METRICS_OH_SONNET45_RUNBOOK.md 2>/dev/null

# Result JSONs — these are the deliverable. Add the whole 2026-05 results dir.
git add archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45/ 2>/dev/null
git add archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45_sglang/ 2>/dev/null

# Persist a snapshot of /usr/local/bin/docker shim alongside the rest so
# it can be re-installed cleanly on any new box.
mkdir -p scripts/shims
cp /usr/local/bin/docker scripts/shims/docker_udocker_shim.py 2>/dev/null
git add scripts/shims/docker_udocker_shim.py 2>/dev/null

if git diff --cached --quiet; then
    echo "$LOG_PREFIX no changes to commit"
    exit 0
fi

# Useful summary in the commit message
SUMMARY=""
NRES=$(find archive/results/2026-05/iso_bench_results_3way_openhands_sonnet45*/results -name '*.json' 2>/dev/null | wc -l)
SUMMARY="$SUMMARY result_jsons=$NRES"

git -c user.email="checkpoint@local" -c user.name="checkpoint" \
    commit -m "checkpoint: hard-metrics OH sonnet-4.5 progress ($SUMMARY) [skip ci]" \
    >/dev/null 2>&1 || { echo "$LOG_PREFIX commit failed"; exit 1; }

if git push origin "$BRANCH" 2>&1 | tail -2 | grep -qE "rejected|error|fatal"; then
    echo "$LOG_PREFIX push FAILED"
    exit 2
fi
echo "$LOG_PREFIX pushed: $SUMMARY"
