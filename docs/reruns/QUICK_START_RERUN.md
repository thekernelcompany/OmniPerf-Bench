# Quick Start: TRAE Sonnet 4.5 Rerun

## TL;DR

142 unsuccessful TRAE + Sonnet 4.5 commits have been identified and prepared for rerun.
The critical `tool_results` bug has been fixed, so many should now succeed.

## Run Everything (Recommended)

```bash
cd /home/ubuntu/OmniPerf-Bench
./rerun_trae_sonnet45_all.sh
```

**Duration:** ~24-30 hours
**Cost:** ~$105-175 (AWS)

## Or Run Separately

### vLLM Only (91 commits)
```bash
./rerun_trae_sonnet45_vllm.sh
```
Duration: ~15-20 hours

### SGLang Only (51 commits)
```bash
./rerun_trae_sonnet45_sglang.sh
```
Duration: ~8-10 hours

## What Was Done

1. ✓ Analyzed all markdown files in `eval_results_v2/`
2. ✓ Extracted 142 unsuccessful TRAE + Sonnet 4.5 commits
3. ✓ Created filtered rerun plans (JSON)
4. ✓ Created execution scripts

## Files Created

```
/home/ubuntu/OmniPerf-Bench/
├── TRAE_SONNET45_RERUN_SUMMARY.md          # Full documentation
├── QUICK_START_RERUN.md                     # This file
├── rerun_trae_sonnet45_vllm.sh              # vLLM rerun script
├── rerun_trae_sonnet45_sglang.sh            # SGLang rerun script
├── rerun_trae_sonnet45_all.sh               # Complete rerun script
└── perf-agents-bench/
    ├── TRAE_SONNET45_VLLM_UNSUCCESSFUL.txt  # 91 vLLM commits
    ├── TRAE_SONNET45_SGLANG_UNSUCCESSFUL.txt # 51 SGLang commits
    ├── TRAE_SONNET45_ALL_UNSUCCESSFUL.txt   # All 142 commits
    └── state/
        ├── plan_trae_sonnet45_vllm_rerun.json
        └── plan_trae_sonnet45_sglang_rerun.json
```

## Breakdown by Failure Reason

| Reason | Count |
|--------|------:|
| AGENT_NO_PATCH (many fixable) | 73 |
| TEST_IMPORT_ERROR | 31 |
| TARGET_NOT_RESOLVED | 15 |
| BASELINE_TYPE_ERROR | 13 |
| Other issues | 10 |

**Expected improvement:** 57% → 75-85% success rate

## Monitor Progress

```bash
cd perf-agents-bench
LATEST=$(ls -t state/runs | head -n1)
.venv/bin/python -m bench.cli report state/runs/$LATEST
```

## Troubleshooting

### AWS credentials expired
```bash
aws sso login --profile your-profile
```

### Run interrupted
Just restart the script - it will resume automatically:
```bash
./rerun_trae_sonnet45_all.sh
```

## Full Documentation

See `TRAE_SONNET45_RERUN_SUMMARY.md` for complete details.
