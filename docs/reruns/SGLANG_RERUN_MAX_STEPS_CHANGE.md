# SGLang Rerun - Max Steps Increased

**Date:** 2025-12-24 18:05

## Change
Increased `max_steps` from 200 to 400 in TRAE configuration

**File:** `/home/ubuntu/OmniPerf-Bench/third-party/trae-agent/trae_config.yaml`

**Reason:**
- Multiple commits (205d5cb4, 25e1816e) exceeded 120 steps during SGLang rerun
- Pipeline was getting stuck on commits requiring >200 steps
- vLLM commits completed in 1-87 steps, but SGLang commits consistently need more steps

## Previous Attempts
1. **Attempt 1:** Added skip logic for commits exceeding 120 steps - pipeline still got stuck
2. **Attempt 2:** Added timeouts to git operations - still encountered hanging on `proc.communicate()`
3. **Final Decision:** Increase max_steps to 400 to give agent sufficient headroom

## Resume Plan
- Completed: 7/51 commits (021f76e4, 10189d08, 132dad87, 148254d4, 187b85b7, 205d5cb4, 23c764b1)
- Resuming from: 25e1816e (commit 8 of 51)
- Remaining: 44 commits
- Resume plan: `/home/ubuntu/OmniPerf-Bench/perf-agents-bench/state/plan_sglang_resume.json`

## Configuration Details

**CRITICAL FIX (2025-12-24 19:10):**
The real issue was in `bench.yaml`, not `trae_config.yaml`. The bench.yaml `args.max_steps: 120` was overriding the trae_config.yaml setting!

**Files changed:**
1. `/home/ubuntu/OmniPerf-Bench/third-party/trae-agent/trae_config.yaml`:
   ```yaml
   agents:
       trae_agent:
           max_steps: 400  # Changed from 200
   ```

2. `/home/ubuntu/OmniPerf-Bench/perf-agents-bench/bench.yaml`:
   ```yaml
   agents:
     trae:
       args:
         max_steps: 400  # Changed from 120 - THIS WAS THE REAL ISSUE!
   ```

## Impact
- Agent now has 2x headroom for complex optimizations
- Should eliminate max_steps_exceeded errors
- Maintains consistency with evaluation methodology (same config for all remaining commits)
