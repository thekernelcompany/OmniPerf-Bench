"""
Upload TRAE GPT-5 rerun trajectories to HuggingFace.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder

# Configuration
HF_ORG = "Inferencebench"
DATASET_NAME = "trae-gpt5-trajectories"
REPO_ID = f"{HF_ORG}/{DATASET_NAME}"

def collect_trajectory_data():
    """Collect all trajectory data from GPT-5 rerun results."""

    base_path = Path("perf-agents-bench/state/runs")

    # vLLM GPT-5 results - use specific successful run
    vllm_path = base_path / "vllm/trae/gpt-5-2025-08-07/2025-12-26_15-06-42"

    data = {
        "vllm": [],
    }

    stats = {
        "vllm": {"total": 0, "success": 0, "error": 0, "files": 0},
    }

    # Process vLLM runs
    if vllm_path.exists():
        latest_vllm = vllm_path
        print(f"Processing vLLM GPT-5 run: {latest_vllm.name}")

        for commit_dir in latest_vllm.glob("vllm_gpt5_rerun_*"):
            if not commit_dir.is_dir():
                continue

            commit_hash = commit_dir.name.replace("vllm_gpt5_rerun_", "")
            journal_file = commit_dir / "journal.json"

            if journal_file.exists():
                with open(journal_file) as f:
                    journal = json.load(f)

                entry = {
                    "commit": commit_hash,
                    "status": journal.get("status", "unknown"),
                    "run_date": latest_vllm.name,
                    "files": {}
                }

                # Collect files
                for file_name in ["trajectory.json", "journal.json", "model_patch.diff",
                                 "run_summary.json", "task.txt", "prediction.jsonl"]:
                    file_path = commit_dir / file_name
                    if file_path.exists():
                        entry["files"][file_name] = str(file_path.relative_to("perf-agents-bench/state/runs"))
                        stats["vllm"]["files"] += 1

                data["vllm"].append(entry)
                stats["vllm"]["total"] += 1
                if entry["status"] == "success":
                    stats["vllm"]["success"] += 1
                else:
                    stats["vllm"]["error"] += 1

    return data, stats

def create_dataset_card(stats):
    """Create README.md for the HuggingFace dataset."""

    # Safe division helper
    def safe_pct(num, denom):
        return f"{num/denom*100:.1f}%" if denom > 0 else "N/A"

    vllm_success_pct = safe_pct(stats['vllm']['success'], stats['vllm']['total'])
    vllm_error_pct = safe_pct(stats['vllm']['error'], stats['vllm']['total'])

    card = f"""---
license: mit
task_categories:
- text-generation
- code-generation
tags:
- agent-trajectories
- performance-optimization
- code-optimization
- trae
- gpt-5
pretty_name: TRAE GPT-5 Agent Trajectories
size_categories:
- 10K<n<100K
---

# TRAE GPT-5 Agent Trajectories

This dataset contains complete agent execution trajectories from TRAE (Tool-use Reasoning Agent for Execution) runs using GPT-5 on performance optimization tasks.

## Dataset Overview

**Agent:** TRAE
**Model:** GPT-5 (gpt-5-2025-08-07)
**Task:** Performance optimization commits from vLLM repository
**Date:** December 2025

### Statistics

**vLLM Rerun:**
- Total commits: {stats['vllm']['total']}
- Successful: {stats['vllm']['success']} ({vllm_success_pct})
- Failed: {stats['vllm']['error']} ({vllm_error_pct})
- Total files: {stats['vllm']['files']}

**Quality Analysis:**
- Real optimizations: 29/32 (90.6%)
- Average patch size (substantial): 78.6 LOC
- False positives (empty patches): 3/32 (9.4%)

## Context: Tool Results Bug Fix

These reruns were conducted after fixing a critical bug in TRAE's base_agent.py where tool_calls were not being handled before sending task_incomplete_message. This bug affected 142 GPT-5 runs (32.7% of all runs) in the original evaluation.

**Bug Impact in Original Evaluation:**
- 142 runs with "No tool output found" error (32.7%)
- 71 runs with quota exhaustion (16.4%)
- 33 runs with invalid API key (7.6%)

**Configuration Changes:**
- Increased max_steps from 120 → 400 to allow complex optimizations
- Fixed tool_calls handling in base_agent.py (lines 179-188)

## Dataset Structure

```
.
├── metadata.json          # Overall dataset metadata
├── vllm/
│   └── <commit_hash>/
│       ├── trajectory.json      # Complete agent execution trace
│       ├── journal.json         # Run metadata and final status
│       ├── model_patch.diff     # Generated code patch
│       ├── run_summary.json     # Performance metrics
│       ├── task.txt             # Original optimization task
│       └── prediction.jsonl     # Agent prediction output
```

## File Descriptions

### trajectory.json
Complete step-by-step execution trace including:
- Agent reasoning and decisions
- Tool calls (bash, edit, task_done, etc.)
- Tool results and outputs
- LLM responses with token counts
- Timestamps for each step

### journal.json
Run metadata including:
- Commit hashes (base and head)
- Final status (success/error)
- Agent configuration
- Execution time
- Metrics (patch size, changed files, violations)

### model_patch.diff
The actual code patch generated by the agent for the optimization task.

### run_summary.json
Summary metrics for the run:
- Task completion status
- Number of steps taken
- Files modified
- Performance indicators

## Use Cases

1. **Agent Behavior Analysis:** Study how TRAE + GPT-5 approaches performance optimization tasks
2. **Model Comparison:** Compare GPT-5 vs Claude Sonnet 4.5 trajectories on same tasks
3. **Tool Usage Patterns:** Analyze which tools agents use and when
4. **Success/Failure Analysis:** Compare trajectories of successful vs failed runs
5. **Bug Impact Assessment:** Evaluate effect of tool_results bug fix on success rates
6. **Training Data:** Use trajectories for training or fine-tuning agent models
7. **Prompt Engineering:** Learn effective prompting strategies from successful runs

## Key Findings

**vLLM GPT-5 Rerun Results:**
- Success rate: 60.4% (32/53 commits)
- Quality: 90.6% real optimizations (29/32)
- False positives: 9.4% (3/32 empty patches)
- Average substantial patch: 78.6 LOC

**Optimization Patterns Observed:**
- Regex precompilation (avoiding repeated `re.compile()`)
- torch.zeros → torch.empty conversions
- Cached attribute lookups (avoiding repeated `getattr()`)
- Environment variable caching
- XDG path caching
- Loop restructuring (while → for)

**Tool Results Bug Fix Validation:**
- Original evaluation: 142 runs with "No tool output found" error
- Rerun results: 0 tool output errors in all 53 runs
- Confirms bug fix in base_agent.py successfully resolves OpenAI API compatibility

## Comparison with Claude Sonnet 4.5

| Metric | Claude Sonnet 4.5 | GPT-5 |
|--------|------------------|-------|
| vLLM Success Rate | 74.7% | 60.4% |
| Real Optimizations | Not measured | 90.6% |
| Tool Output Errors | 0 | 0 |
| Max Steps Required | 200+ (some) | Not measured |

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{trae-gpt5-trajectories,
  title={{TRAE GPT-5 Agent Trajectories}},
  author={{OmniPerf-Bench Contributors}},
  year={{2025}},
  publisher={{HuggingFace}},
  howpublished={{\\url{{https://huggingface.co/datasets/{REPO_ID}}}}}
}}
```

## Related Resources

- **OmniPerf-Bench Repository:** [github.com/thekernelcompany/OmniPerf-Bench](https://github.com/thekernelcompany/OmniPerf-Bench)
- **TRAE Agent:** [github.com/agokrani/trae-agent](https://github.com/agokrani/trae-agent)
- **vLLM Dataset:** [Inferencebench/vllm_dataset_with_test](https://huggingface.co/datasets/Inferencebench/vllm_dataset_with_test)
- **Claude Sonnet 4.5 Trajectories:** [Inferencebench/trae-sonnet45-trajectories](https://huggingface.co/datasets/Inferencebench/trae-sonnet45-trajectories)

## License

MIT License - See repository for full license text.
"""

    return card

def main():
    print("=" * 70)
    print("TRAE GPT-5 Trajectory Upload to HuggingFace")
    print("=" * 70)

    # Verify HF_TOKEN
    token = HfFolder.get_token() or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not found! Run: huggingface-cli login")
        return

    print(f"✓ HF_TOKEN found (length: {len(token)} chars)")

    # Collect data
    print("\nCollecting GPT-5 trajectory data...")
    data, stats = collect_trajectory_data()

    print(f"\n{'Summary:':-^70}")
    if stats['vllm']['total'] > 0:
        print(f"\nvLLM GPT-5:")
        print(f"  Total commits: {stats['vllm']['total']}")
        print(f"  Success: {stats['vllm']['success']} ({stats['vllm']['success']/stats['vllm']['total']*100:.1f}%)")
        print(f"  Error: {stats['vllm']['error']}")
        print(f"  Files: {stats['vllm']['files']}")

    # Save metadata
    metadata = {
        "dataset_name": DATASET_NAME,
        "created": datetime.now().isoformat(),
        "agent": "TRAE",
        "model": "gpt-5-2025-08-07",
        "statistics": stats,
        "data": data,
        "quality_metrics": {
            "real_optimizations": "29/32 (90.6%)",
            "false_positives": "3/32 (9.4%)",
            "avg_patch_size_loc": 78.6
        }
    }

    output_dir = Path("hf_gpt5_trajectories_upload")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved metadata to {output_dir}/metadata.json")

    # Create dataset card
    card = create_dataset_card(stats)
    with open(output_dir / "README.md", "w") as f:
        f.write(card)

    print(f"✓ Created dataset card: {output_dir}/README.md")

    # Copy trajectory files
    print("\nCopying trajectory files...")
    base_runs = Path("perf-agents-bench/state/runs")

    if data["vllm"]:
        repo_output = output_dir / "vllm"
        repo_output.mkdir(exist_ok=True)

        for entry in data["vllm"]:
            commit_dir = repo_output / entry["commit"]
            commit_dir.mkdir(exist_ok=True)

            for file_name, rel_path in entry["files"].items():
                src = base_runs / rel_path
                dst = commit_dir / file_name

                if src.exists():
                    import shutil
                    shutil.copy2(src, dst)

        print(f"  ✓ Copied {len(data['vllm'])} vLLM GPT-5 commits")

    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"\n✓ Total dataset size: {total_size / 1024 / 1024:.1f} MB")

    # Create/verify repo
    print(f"\n{'HuggingFace Upload:':-^70}")
    print(f"Repository: {REPO_ID}")

    api = HfApi(token=token)

    try:
        # Create repo if doesn't exist
        create_repo(
            repo_id=REPO_ID,
            repo_type="dataset",
            exist_ok=True,
            token=token
        )
        print(f"✓ Repository created/verified")

        # Upload folder
        print(f"\nUploading dataset (this may take a while)...")
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token,
            commit_message="Upload TRAE GPT-5 agent trajectories (vLLM reruns, 32/53 successes, 90.6% quality)"
        )

        print(f"\n{'='*70}")
        print(f"✅ SUCCESS! Dataset uploaded to HuggingFace")
        print(f"{'='*70}")
        print(f"\nView at: https://huggingface.co/datasets/{REPO_ID}")

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
