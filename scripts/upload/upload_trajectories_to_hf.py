"""
Upload TRAE Sonnet 4.5 rerun trajectories to HuggingFace.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder

# Configuration
HF_ORG = "Inferencebench"
DATASET_NAME = "trae-sonnet45-trajectories"
REPO_ID = f"{HF_ORG}/{DATASET_NAME}"

def collect_trajectory_data():
    """Collect all trajectory data from rerun results."""
    
    base_path = Path("ISO-Bench/state/runs")
    
    # vLLM results
    vllm_path = base_path / "vllm/trae/us-anthropic-claude-sonnet-4-5-20250929-v1-0"
    
    # SGLang results (note: directory is 'sglan')
    sglan_path = base_path / "sglan/trae/us-anthropic-claude-sonnet-4-5-20250929-v1-0"
    
    data = {
        "vllm": [],
        "sglang": []
    }
    
    stats = {
        "vllm": {"total": 0, "success": 0, "error": 0, "files": 0},
        "sglang": {"total": 0, "success": 0, "error": 0, "files": 0}
    }
    
    # Process vLLM runs (find most recent run directory - check both 2024 and 2025)
    if vllm_path.exists():
        run_dirs = sorted([d for d in vllm_path.glob("2024*") if d.is_dir()] +
                         [d for d in vllm_path.glob("2025*") if d.is_dir()])
        if run_dirs:
            latest_vllm = run_dirs[-1]
            print(f"Processing vLLM run: {latest_vllm.name}")
            
            for commit_dir in latest_vllm.glob("vllm_sonnet45_rerun_*"):
                if not commit_dir.is_dir():
                    continue
                    
                commit_hash = commit_dir.name.replace("vllm_sonnet45_rerun_", "")
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
                            entry["files"][file_name] = str(file_path.relative_to("ISO-Bench/state/runs"))
                            stats["vllm"]["files"] += 1
                    
                    data["vllm"].append(entry)
                    stats["vllm"]["total"] += 1
                    if entry["status"] == "success":
                        stats["vllm"]["success"] += 1
                    else:
                        stats["vllm"]["error"] += 1
    
    # Process SGLang runs
    if sglan_path.exists():
        run_dirs = sorted([d for d in sglan_path.glob("2024*") if d.is_dir()] + 
                         [d for d in sglan_path.glob("2025*") if d.is_dir()])
        if run_dirs:
            latest_sglan = run_dirs[-1]
            print(f"Processing SGLang run: {latest_sglan.name}")
            
            for commit_dir in latest_sglan.glob("sglang_sonnet45_rerun_*"):
                if not commit_dir.is_dir():
                    continue
                    
                commit_hash = commit_dir.name.replace("sglang_sonnet45_rerun_", "")
                journal_file = commit_dir / "journal.json"
                
                if journal_file.exists():
                    with open(journal_file) as f:
                        journal = json.load(f)
                    
                    entry = {
                        "commit": commit_hash,
                        "status": journal.get("status", "unknown"),
                        "run_date": latest_sglan.name,
                        "files": {}
                    }
                    
                    # Collect files
                    for file_name in ["trajectory.json", "journal.json", "model_patch.diff",
                                     "run_summary.json", "task.txt", "prediction.jsonl"]:
                        file_path = commit_dir / file_name
                        if file_path.exists():
                            entry["files"][file_name] = str(file_path.relative_to("ISO-Bench/state/runs"))
                            stats["sglang"]["files"] += 1
                    
                    data["sglang"].append(entry)
                    stats["sglang"]["total"] += 1
                    if entry["status"] == "success":
                        stats["sglang"]["success"] += 1
                    else:
                        stats["sglang"]["error"] += 1
    
    return data, stats

def create_dataset_card(stats):
    """Create README.md for the HuggingFace dataset."""
    
    # Safe division helper
    def safe_pct(num, denom):
        return f"{num/denom*100:.1f}%" if denom > 0 else "N/A"
    
    vllm_success_pct = safe_pct(stats['vllm']['success'], stats['vllm']['total'])
    vllm_error_pct = safe_pct(stats['vllm']['error'], stats['vllm']['total'])
    sglang_success_pct = safe_pct(stats['sglang']['success'], stats['sglang']['total'])
    sglang_error_pct = safe_pct(stats['sglang']['error'], stats['sglang']['total'])
    
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
- claude-sonnet-4.5
pretty_name: TRAE Sonnet 4.5 Agent Trajectories
size_categories:
- 100K<n<1M
---

# TRAE Claude Sonnet 4.5 Agent Trajectories

This dataset contains complete agent execution trajectories from TRAE (Tool-use Reasoning Agent for Execution) runs using Claude Sonnet 4.5 on performance optimization tasks.

## Dataset Overview

**Agent:** TRAE  
**Model:** Claude Sonnet 4.5 (us-anthropic-claude-sonnet-4-5-20250929-v1-0)  
**Task:** Performance optimization commits from vLLM and SGLang repositories  
**Date:** December 2024 - December 2025  

### Statistics

**vLLM Rerun:**
- Total commits: {stats['vllm']['total']}
- Successful: {stats['vllm']['success']} ({vllm_success_pct})
- Failed: {stats['vllm']['error']} ({vllm_error_pct})
- Total files: {stats['vllm']['files']}

**SGLang Rerun:**
- Total commits: {stats['sglang']['total']}
- Successful: {stats['sglang']['success']} ({sglang_success_pct})
- Failed: {stats['sglang']['error']} ({sglang_error_pct})
- Total files: {stats['sglang']['files']}

## Context: Tool Results Bug Fix

These reruns were conducted after fixing a critical bug in TRAE's base_agent.py where tool_calls were not being handled before sending task_incomplete_message. This bug violated Anthropic's API requirement that tool_result blocks must immediately follow tool_use blocks.

**Bug Impact:**
- Caused 51% of all TRAE failures (272 runs) in original evaluation
- Three types of API errors: TOOL_OUTPUT_MISSING, TOOL_RESULT_MISSING, EMPTY_ERROR_CONTENT

**Configuration Changes:**
- Increased max_steps from 120 → 400 to allow complex optimizations
- Fixed in bench.yaml (which overrides trae_config.yaml)

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
└── sglang/
    └── <commit_hash>/
        └── (same structure as vllm)
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

1. **Agent Behavior Analysis:** Study how TRAE approaches performance optimization tasks
2. **Tool Usage Patterns:** Analyze which tools agents use and when
3. **Success/Failure Analysis:** Compare trajectories of successful vs failed runs
4. **Bug Impact Assessment:** Evaluate effect of tool_results bug fix on success rates
5. **Training Data:** Use trajectories for training or fine-tuning agent models
6. **Prompt Engineering:** Learn effective prompting strategies from successful runs

## Key Findings

**vLLM Results:**
- Original success rate: 57%
- Rerun recovered: 46 commits (50.5% of unsuccessful)
- Combined success rate: 74.7% (17.6 percentage point improvement)

**SGLang Results:**
- Rerun success rate: 80.5% (33 of 41 commits)
- Some commits required 200+ steps (validating max_steps=400 increase)

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{trae-sonnet45-trajectories,
  title={{TRAE Claude Sonnet 4.5 Agent Trajectories}},
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

## License

MIT License - See repository for full license text.
"""
    
    return card

def main():
    print("=" * 70)
    print("TRAE Sonnet 4.5 Trajectory Upload to HuggingFace")
    print("=" * 70)
    
    # Verify HF_TOKEN
    token = HfFolder.get_token() or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not found! Run: huggingface-cli login")
        return
    
    print(f"✓ HF_TOKEN found (length: {len(token)} chars)")
    
    # Collect data
    print("\nCollecting trajectory data...")
    data, stats = collect_trajectory_data()
    
    print(f"\n{'Summary:':-^70}")
    for repo in ["vllm", "sglang"]:
        if stats[repo]['total'] > 0:
            print(f"\n{repo.upper()}:")
            print(f"  Total commits: {stats[repo]['total']}")
            print(f"  Success: {stats[repo]['success']} ({stats[repo]['success']/stats[repo]['total']*100:.1f}%)")
            print(f"  Error: {stats[repo]['error']}")
            print(f"  Files: {stats[repo]['files']}")
    
    # Save metadata
    metadata = {
        "dataset_name": DATASET_NAME,
        "created": datetime.now().isoformat(),
        "agent": "TRAE",
        "model": "claude-sonnet-4.5 (us-anthropic-claude-sonnet-4-5-20250929-v1-0)",
        "statistics": stats,
        "data": data
    }
    
    output_dir = Path("hf_trajectories_upload")
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
    base_runs = Path("ISO-Bench/state/runs")
    
    for repo in ["vllm", "sglang"]:
        if data[repo]:  # Only process if we have data
            repo_output = output_dir / repo
            repo_output.mkdir(exist_ok=True)
            
            for entry in data[repo]:
                commit_dir = repo_output / entry["commit"]
                commit_dir.mkdir(exist_ok=True)
                
                for file_name, rel_path in entry["files"].items():
                    src = base_runs / rel_path
                    dst = commit_dir / file_name
                    
                    if src.exists():
                        import shutil
                        shutil.copy2(src, dst)
            
            print(f"  ✓ Copied {len(data[repo])} {repo} commits")
    
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
            commit_message="Upload TRAE Sonnet 4.5 agent trajectories (vLLM + SGLang reruns)"
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
