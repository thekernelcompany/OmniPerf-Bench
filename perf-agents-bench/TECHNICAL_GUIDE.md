# perf-agents-bench Technical Guide

## Overview

perf-agents-bench is a modular, local-first benchmarking system that evaluates AI agents' ability to optimize code performance. It supports a two-stage workflow:

- **Stage A (Prepare; no Docker)**: Plan commit pairs, run OpenHands locally to create agent branches, enforce target-file constraints, and write journals. Scales to 100+ commits using `git worktree` and resumable execution.
- **Stage B (Evaluate; Docker)**: Build per-commit images for baseline/human/agent, run TestPack(s) inside containers, and collect metrics. Images are tagged `bench-<task>-<candidate>-<shortsha>` for reproducibility.

## How It Works: Two Stages

Stage A produces agent branches deterministically without containers. Stage B then performs a three-way comparison by building and running identical containers for:

1. Baseline (pre-commit)
2. Human (human-commit)
3. Agent (OpenHands output based on pre-commit)

Each candidate's image is built at the exact commit to ensure a fair environment.

## Step-by-Step Workflow

### 1. Task Definition
Create a YAML file defining your benchmark task:

```yaml
id: "vllm_core"
name: "vLLM core performance"
repo:
  url: "https://github.com/vllm-project/vllm.git"
  human_commit: "f092153..."  # Expert's optimization
  # pre_commit: auto-detected as human_commit^

runner:
  platform: "linux/amd64"     # Important for Apple Silicon
  cpus: 2
  memory: "4g"

env_build:
  allowed_strategies: ["dockerfile", "requirements"]

testpack:
  entrypoint: "../vlm-bench-generic"

metrics:
  - name: "throughput"
    using: "runtime:throughput"
  - name: "correctness"
    using: "quality:hash_match"
```

### 2. Stage A (Prepare) Workflow

Prerequisite: OpenHands CLI installed and accessible. See `https://docs.all-hands.dev/`.

Commands:

```bash
# Scaffold example
python -m bench.cli init --out perf-agents-bench

# Plan commit pairs from a commits file
python -m bench.cli plan perf-agents-bench/tasks/example.yaml \
  --commits perf-agents-bench/.work/commits.txt \
  --out state/plan.json

# Run OpenHands locally for each item, write journals
python -m bench.cli prepare perf-agents-bench/tasks/example.yaml \
  --from-plan state/plan.json --max-workers 4 --resume

# Summarize journals
python -m bench.cli report state/runs/<run_id>
```

Artifacts:
- `state/plan.json` — matrix of commit pairs
- `state/runs/<run_id>/<item_id>/` — `prompt.json`, `journal.json`, `diff_targets.json`, logs

### 3. Environment Building (Stage B)
The system automatically detects how to build containers:
- Dockerfile: uses repo Dockerfile directly
- requirements.txt: creates container with Python base + requirements
- Poetry/Conda/Nix: supported via respective files

### 4. Container Execution Flow (Stage B)

```
Repository Clone
    ↓
┌─────────────┬─────────────┬─────────────┐
│  Baseline   │    Human    │    Agent    │
├─────────────┼─────────────┼─────────────┤
│ git checkout│ git checkout│ git checkout│
│ pre_commit  │ human_commit│ pre_commit  │
│      ↓      │      ↓      │      ↓      │
│Build Image  │Build Image  │ Run AI Agent│
│      ↓      │      ↓      │      ↓      │
│Run TestPack │Run TestPack │Build Image  │
│      ↓      │      ↓      │      ↓      │
│   Metrics   │   Metrics   │Run TestPack │
│             │             │      ↓      │
│             │             │   Metrics   │
└─────────────┴─────────────┴─────────────┘
                    ↓
              Compare Results
```

### 5. TestPack System
TestPacks are plugins that define how to test performance:

```python
class MyTestPack(TestPack):
    def prepare_fixtures(self, repo_dir, work_dir):
        # Set up test data/fixtures
        
    def build(self, repo_dir, work_dir):
        # Build/install the repository
        
    def run_candidate(self, repo_dir, work_dir, out_dir, candidate_tag):
        # Execute performance tests
```

### 6. Metrics Collection
Two types of metrics ensure both performance and correctness:
- **Performance Metrics**: Throughput, latency, resource usage
- **Quality Metrics**: Output validation, functional correctness

### 7. Agent Integration
Currently supports OpenHands agent, which:
- Receives optimization constraints and target files
- Has configurable time budget (default: 30 minutes)
- Can run as host CLI or in its own container
- Creates optimizations on a separate branch

## Key Features

### Container Isolation (Stage B)
- Each candidate runs in identical resource-constrained containers
- Configurable CPU, memory, GPU access
- Network isolation options
- Platform specification for cross-architecture compatibility

### Reproducibility
- Git commit-based container tagging: `bench-<task>-<candidate>-<shortsha>`
- Hermetic execution environment
- No external dependencies during benchmark execution

### Extensibility
- Plugin architecture for TestPacks
- Metric system extensible via decorators
- Environment strategies for new build systems
- Agent adapters for different AI systems

## Quick Start Commands

```bash
# Stage A
python -m bench.cli init --out perf-agents-bench
python -m bench.cli plan perf-agents-bench/tasks/example.yaml --commits perf-agents-bench/.work/commits.txt --out state/plan.json
python -m bench.cli prepare perf-agents-bench/tasks/example.yaml --from-plan state/plan.json --max-workers 4 --resume
python -m bench.cli report state/runs/<run_id>

# Stage B (optional, Docker required)
bench smoke tasks/example.yaml --human-only --cmd "python -c 'print(\"OK\")'"
bench build tasks/example.yaml
bench run tasks/example.yaml
```

## Directory Structure

```
perf-agents-bench/
├── bench/                  # Core library
│   ├── agents/            # AI agent adapters
│   ├── container/         # Docker/Podman abstraction
│   ├── env/              # Environment builders
│   ├── metrics/          # Performance metrics
│   ├── testpack_api/     # TestPack contract
│   └── pipeline.py       # Main orchestration
├── tasks/                # Benchmark task definitions
├── bench.yaml           # Global configuration
└── .work/              # Runtime working directory
    └── state/          # Execution results
```

## Configuration

### Global Configuration (bench.yaml)
```yaml
container:
  engine: "${CONTAINER_ENGINE:-docker}"
  platform: "${CONTAINER_PLATFORM:-linux/amd64}"
  cpus: "${BENCH_CPUS:-2}"
  memory: "${BENCH_MEMORY:-4g}"

agents:
  openhands:
    cli: "openhands"
    time_budget_minutes: 30
```

### Task Configuration
- Repository details (URL, commits)
- Container resource limits
- Environment build strategies
- TestPack selection
- Metrics to collect
- Optimization constraints

## Platform-Specific Notes

### Apple Silicon (M1/M2)
Set `platform: linux/amd64` in bench.yaml to avoid ARM64 build issues with ML packages.

### Container Engines
Docker-only recommended.

## Design Principles

1. **No Hard-coded Assumptions**: Everything must be explicitly configured
2. **Local-First**: No cloud dependencies, runs entirely on local hardware
3. **Fair Comparison**: Identical constraints for all candidates
4. **Extensible**: Plugin architecture for all major components
5. **Reproducible**: Commit-based versioning ensures exact reproduction

## Differences from Cloud-Based GSO

- ❌ No SkyPilot or cloud orchestration
- ❌ No OpenAI API dependencies  
- ✅ Local container execution
- ✅ Flexible task definition
- ✅ Dynamic environment building
- ✅ Single-task focused execution

This system enables rigorous, reproducible benchmarking of AI code optimization capabilities while maintaining complete control over the execution environment.