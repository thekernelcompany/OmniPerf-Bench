
# HuggingFace Dataset Update Summary

## Updates Required

### 1. Serving Metrics Added (7 commits)
These commits previously had throughput metrics (WRONG) but now have correct serving metrics:

| Commit | agent_ttft_mean | agent_tpot_mean | agent_itl_mean |
|--------|-----------------|-----------------|----------------|
| 99abb8b6 | 656.64 ms | 30.98 ms | 24.46 ms |
| 22d33bac | 651.12 ms | 30.51 ms | 24.52 ms |
| 9badee53 | 174.68 ms | 9.85 ms | 7.98 ms |
| e206b543 | 669.91 ms | 30.88 ms | 24.56 ms |
| 89a84b0b | 356.05 ms | 23.73 ms | 28.48 ms |
| 19d98e0c | 1099.58 ms | 35.95 ms | 35.89 ms |
| 6e36f4fa | 1011.46 ms | 30.52 ms | 33.53 ms |

Note: agent_throughput and agent_latency_avg cleared for these commits (they are serving benchmarks).

### 2. Benchmark Mode Fix (1 commit)
| Commit | Old Mode | New Mode |
|--------|----------|----------|
| 2deb029d | standalone | prefix_caching |

## Files to Upload

1. `updated_dataset.parquet` - Full dataset with all updates applied
2. `updated_rows.csv` - Just the 8 rows that were modified (for verification)
3. `full_dataset.csv` - Full dataset as CSV

## How to Upload

```bash
# Using huggingface-cli
huggingface-cli login  # Use a write token
huggingface-cli upload Inferencebench/claude-code-vllm-benchmarks updated_dataset.parquet

# Or using Python
from datasets import Dataset
import pandas as pd

df = pd.read_parquet('updated_dataset.parquet')
dataset = Dataset.from_pandas(df)
dataset.push_to_hub("Inferencebench/claude-code-vllm-benchmarks")
```

## Session Date
2026-01-19
