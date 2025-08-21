# Repository Structure

This document describes the reorganized structure of the OmniPerf-Bench repository.

## Overview

The repository has been restructured to reduce clutter and improve organization. All existing files have been preserved but moved to more logical locations.

## Directory Structure

```
OmniPerf-Bench/
├── README.md                     # Main project documentation
├── LICENSE                       # Project license
├── pyproject.toml               # Python project configuration
├── uv.lock                      # Lock file for uv package manager
├── requirements.txt             # Python dependencies
├── .gitignore                   # Updated git ignore rules
│
├── src/                         # Main GSO source code
│   ├── collect/                 # Collection framework
│   ├── data/                    # Data models and structures
│   ├── harness/                 # Evaluation harness
│   ├── utils/                   # Utility functions
│   ├── constants.py             # Project constants
│   └── logger.py               # Logging configuration
│
├── experiments/                 # Experiment configurations and data
│   ├── sglang.yaml             # SGLang experiment config
│   ├── vllm.yaml               # vLLM experiment config
│   └── vllm/                   # vLLM experiment data
│       ├── data/               # Dataset files
│       ├── divided/            # Divided result files
│       ├── README.md           # vLLM dataset documentation
│       ├── vllm_ac_map.json    # API-Commit mapping
│       ├── vllm_commits.json   # Performance commits
│       ├── vllm_experiment.yaml # Experiment configuration
│       ├── vllm_problems.json  # Generated problems
│       └── vllm_results.json   # Execution results
│
├── benchmarks/                  # External benchmark integrations
│   └── effibench/              # EffiBench integration
│       ├── src/                # EffiBench source code
│       ├── data/               # EffiBench datasets
│       ├── scripts/            # EffiBench utility scripts
│       ├── prompts/            # LLM prompts for EffiBench
│       ├── results/            # EffiBench evaluation results
│       ├── requirements.txt    # EffiBench dependencies
│       └── README.md           # EffiBench documentation
│
├── tools/                       # Utility scripts and tools
│   ├── manual_review.py        # Manual review utilities
│   └── openrouter_patch.py     # OpenRouter API patches
│
├── results/                     # All results and outputs
│   ├── logs/                   # Log files from various runs
│   ├── reviews/                # Manual review files (CSV)
│   └── analysis/               # Analysis outputs
│
├── misc/                        # Archived and legacy files
│   ├── environments/           # Preserved virtual environments
│   │   ├── bench-env/         # Main project virtual env
│   │   └── effi-env/          # EffiBench virtual env
│   ├── effibench-legacy/       # Legacy EffiBench files
│   │   ├── commit_extractions/ # Original commit extraction data
│   │   ├── generated_test_generators_v*/  # Various versions
│   │   ├── gso-duplicate/      # Duplicate GSO implementation
│   │   └── [various legacy files]
│   └── archive/                # Miscellaneous archived files
│
└── docs/                        # Documentation
    ├── repository_structure.md  # This file
    └── benchmarks/              # Benchmark-specific documentation
```

## Key Changes Made

### 1. Consolidated Experiments
- Moved all experiment configurations to `experiments/`
- Organized vLLM data into `experiments/vllm/`
- Preserved all data files and configurations

### 2. Isolated EffiBench
- Moved EffiBench to `benchmarks/effibench/`
- Maintained complete functionality as a sub-project
- Preserved all EffiBench source code and data

### 3. Organized Results and Logs
- Centralized all log files in `results/logs/`
- Moved review files to `results/reviews/`
- Created space for analysis outputs in `results/analysis/`

### 4. Preserved Legacy Content
- Moved virtual environments to `misc/environments/`
- Archived legacy EffiBench files in `misc/effibench-legacy/`
- Kept all historical data and experimental versions

### 5. Improved Tooling
- Moved utility scripts to `tools/`
- Enhanced `.gitignore` to prevent future clutter
- Maintained all existing functionality

## Migration Notes

- **No files were deleted** - everything has been preserved in the `misc/` directory
- All core GSO functionality remains in `src/`
- EffiBench can still be used independently from `benchmarks/effibench/`
- Virtual environments are preserved but organized in `misc/environments/`

## Usage

The main GSO functionality remains unchanged. To use different components:

- **GSO Collection Framework**: Use scripts in `src/collect/`
- **EffiBench**: Change to `benchmarks/effibench/` directory
- **Experiments**: Configuration files are in `experiments/`
- **Results**: Check `results/` for logs and analysis

This reorganization maintains full backward compatibility while providing a much cleaner and more maintainable structure.
