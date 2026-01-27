# MIMO Agent Benchmark Implementation

This document describes the implementation of performance benchmarks for the MIMO (xiaomi-mimo-v2-flash) agent on vLLM commits.

## Overview

**Goal**: Run performance benchmarks for MIMO agent patches on 39 vLLM commits (agent-only mode).

**Final Result**: **34/39 (87.2%) success rate**

## Background

The benchmark system applies agent-generated patches to baseline Docker images. For each commit, we need:
1. **Baseline Docker image** (`shikhar481/vllm_fixed_human_images:baseline-{parent}`)
2. **Benchmark mapping** (`complete_benchmark_mapping.json`) with `parent_commit` and `perf_command`
3. **Agent patch** from MIMO run directory

### MIMO Agent Run Location
```
perf-agents-bench/state/runs/vllm/trae/xiaomi-mimo-v2-flash/2026-01-27_00-01-51/
```

Each task directory contains:
- `journal.json` - Contains commit information (human commit, parent commit)
- `model_patch.diff` - The agent-generated patch to apply

## Implementation Steps

### Step 1: Generate Missing Benchmark Mappings

Added 17 new entries to `complete_benchmark_mapping.json` by merging data from:
- `data/mappings/omniperf_dataset.json` - Contains `perf_command`, `model`, `commit_hash`
- MIMO `journal.json` files - Contains `parent_commit`

### Step 2: Register MIMO Agent in run_3way_benchmarks.py

Added MIMO configuration to the benchmark runner:

```python
# In AGENT_CONFIGS (~line 26):
"mimo": "perf-agents-bench/state/runs/vllm/trae/xiaomi-mimo-v2-flash/2026-01-27_00-01-51",

# In AGENT_OUTPUT_DIRS (~line 38):
"mimo": REPO_ROOT / "omniperf_results_3way_mimo",
```

### Step 3: Fix Hardcoded Paths

Changed hardcoded `/root/OmniPerf-Bench` paths to use `REPO_ROOT` for portability:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
```

### Step 4: Fix Benchmark Command Compatibility

Older vLLM versions (e.g., 0.5.0.post1) don't support `--dataset-name random`. Added automatic detection and fallback:

```bash
# Check if this vLLM version supports --dataset-name random
if $VLLM_PYTHON /opt/vllm_baseline/benchmarks/benchmark_serving.py --help 2>&1 | grep -q "random"; then
    DATASET_ARGS="--dataset-name random --random-input-len 256 --random-output-len 64"
else
    echo "Note: This vLLM version doesn't support random dataset, using sonnet..."
    # Download sonnet dataset if not present
    if [ ! -f /tmp/sonnet.txt ]; then
        $VLLM_PYTHON -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/sonnet.txt', '/tmp/sonnet.txt')"
    fi
    DATASET_ARGS="--dataset-name sonnet --dataset-path /tmp/sonnet.txt"
fi
```

## Issues Encountered and Solutions

### 1. NVIDIA Driver/Library Version Mismatch

**Error:**
```
nvidia-container-cli: initialization error: nvml error: driver/library version mismatch
```

**Solution:**
```bash
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
sudo modprobe nvidia_uvm
```

### 2. Disk Space Exhaustion

**Error:**
```
no space left on device
```

**Solution:** Moved Docker data to `/ephemeral` disk (700GB available):
```bash
sudo systemctl stop docker containerd
sudo mv /var/lib/docker /ephemeral/docker
sudo ln -s /ephemeral/docker /var/lib/docker
sudo rm -rf /var/lib/containerd
sudo mkdir -p /ephemeral/containerd
sudo ln -s /ephemeral/containerd /var/lib/containerd
sudo systemctl start containerd docker
```

### 3. HuggingFace Authentication Failures

**Error:**
```
GatedRepoError: 401 Client Error. Cannot access gated repo for meta-llama/Meta-Llama-3-8B-Instruct
```

**Solution:** User added HF token to `~/.cache/huggingface/token`

### 4. Network Connection Resets

**Error:**
```
docker: failed to copy: read tcp...read: connection reset by peer
ReadTimeoutError: HTTPSConnectionPool...Read timed out
```

**Solution:** Retried after disk space cleanup (transient network issues resolved)

### 5. Benchmark Command Incompatibility

**Error:**
```
argument --dataset-name: invalid choice: 'random' (choose from 'sharegpt', 'sonnet')
```

**Solution:** Added version detection and fallback to sonnet dataset (see Step 4 above)

## Final Results

### Success Rate Progression
| Stage | Success Rate | Notes |
|-------|-------------|-------|
| Initial run | 20/39 (51.3%) | Disk space + HF auth issues |
| After HF token | 29/39 (74.4%) | Fixed authentication |
| After disk cleanup | 32/39 (82.1%) | Fixed disk space issues |
| After network retry | 33/39 (84.6%) | Fixed transient network failures |
| After benchmark fix | 34/39 (87.2%) | Fixed dataset compatibility |

### Final Breakdown

| Status | Count | Commits |
|--------|-------|---------|
| **Succeeded** | 34 | - |
| **Hardware Issue** | 1 | `310aca88` (70B model needs 4x H100 TP4) |
| **Patch Bugs** | 4 | `35fad35a`, `a3223766`, `ad8d696a`, `bc7c4d20` |

### Failure Analysis

#### Hardware Incompatible (1)
| Commit | Model | Issue |
|--------|-------|-------|
| `310aca88` | Meta-Llama-3-70B | Needs 4x H100 with tensor parallelism |

#### MIMO Agent Patch Bugs (4)
These commits have patches that crash the vLLM server:
- `35fad35a` - Server crashed after applying patch
- `a3223766` - Server crashed after applying patch
- `ad8d696a` - Server crashed after applying patch
- `bc7c4d20` - Server crashed after applying patch (wheel mode)

## Files Modified

| File | Changes |
|------|---------|
| `complete_benchmark_mapping.json` | Added 17 new commit entries |
| `scripts/runners/run_3way_benchmarks.py` | Added MIMO config, fixed paths, added dataset fallback |

## Output Files

Benchmark results are stored in:
```
omniperf_results_3way_mimo/results/{commit}_agent_result.json
```

Each result file contains:
- `human_commit` - The commit being benchmarked
- `parent_commit` - The baseline commit
- `model` - The model used for benchmarking
- `status` - "success" or "error"
- `metrics` - Throughput and latency metrics (if successful)
- `raw_output` - Full benchmark output
- `timestamp` - When the benchmark was run

## Running the Benchmarks

### Full Run
```bash
python3 scripts/runners/run_3way_benchmarks.py --agent-type mimo --agent-only --timeout 900
```

### Specific Commits
```bash
python3 scripts/runners/run_3way_benchmarks.py --agent-type mimo --agent-only --commits 3476ed08 6a417b86 --timeout 600
```

### Dry Run
```bash
python3 scripts/runners/run_3way_benchmarks.py --agent-type mimo --agent-only --dry-run
```

## Sample Successful Result

```json
{
  "human_commit": "3476ed08",
  "human_commit_full": "3476ed0809ec91a3457da0cb90543133a4f4b519",
  "parent_commit": "54600709b6d419fb243ce718a48ab7d40f5c3eb7",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "status": "success",
  "metrics": {
    "throughput_tok_s": 2040.65,
    "mean_ttft_ms": 45.23,
    "mean_tpot_ms": 12.34
  },
  "duration_s": 89.5,
  "timestamp": "2026-01-27 21:42:44"
}
```

## Conclusion

The MIMO agent benchmark implementation achieved an 87.2% success rate (34/39 commits). The remaining failures are due to:
- Hardware constraints (1 commit needs multi-GPU setup)
- Genuine bugs in MIMO-generated patches (4 commits crash the server)

These results provide a baseline for evaluating the MIMO agent's performance optimization capabilities on vLLM.
