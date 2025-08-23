# Performance Agents Benchmark

A modular, engine-agnostic performance benchmarking system for evaluating AI agents' ability to optimize code performance.

## Key Features

- **No hard-coded assumptions**: Every parameter must be explicit in configuration
- **Plugin architecture**: TestPacks, metrics, and agents are external/pluggable
- **Local execution**: No cloud dependencies (no SkyPilot), runs entirely with local Docker/Podman
- **Hermetic execution**: Identical container constraints for all candidates
- **Config-driven**: Repository URLs, commits, and tools specified via environment variables

## Stage A Quickstart (no Docker; prepare-only)

This path scales to 100+ commits without containers. It plans commit pairs and runs OpenHands locally to produce agent branches with journals.

Prerequisite: OpenHands CLI installed and on PATH (or set `OPENHANDS_CLI`), see `https://docs.all-hands.dev/`.

1. Initialize scaffolding:
   ```bash
   PYTHONPATH=perf-agents-bench python3 -m bench.cli init --out perf-agents-bench
   ```

2. Create a commits file (human and optional pre or parent index):
   ```text
   # perf-agents-bench/.work/commits.txt
   <40hex-human> [<40hex-pre>|parent=1]
   ```

3. Set minimal env for the task, then plan:
   ```bash
   export REPO_URL=/path/to/local/or/remote/repo
   export HUMAN_COMMIT=<40hex>
   export PRE_COMMIT=<40hex>   # or omit if using parent=1 in commits.txt
   PYTHONPATH=perf-agents-bench python3 -m bench.cli plan \
     perf-agents-bench/tasks/example.yaml \
     --commits perf-agents-bench/.work/commits.txt \
     --out state/plan.json
   ```

4. Prepare across all items (runs OpenHands locally; resumable):
   ```bash
   # If OpenHands is not installed yet, you can smoke-test with a no-op:
   # export OPENHANDS_CLI=/bin/true
   PYTHONPATH=perf-agents-bench python3 -m bench.cli prepare \
     perf-agents-bench/tasks/example.yaml \
     --from-plan state/plan.json \
     --max-workers 4 --resume
   ```

Artifacts:
- `state/plan.json` — array of `{item_id, human, pre|pre_parent_index}`
- `state/runs/<run_id>/<item_id>/` — `prompt.json`, `journal.json`, `diff_targets.json`, logs

## Quick Start (literal YAML, no env exports)

1. **Install CLI (editable or via PYTHONPATH)**:
   ```bash
   cd perf-agents-bench
   pip install -e .
   # or run without install:
   # PYTHONPATH=perf-agents-bench python3 -m bench.cli --help
   ```

2. **Write your task YAML with literal values** (no envs needed):
   ```yaml
   # perf-agents-bench/tasks/example.yaml
   id: image_decode
   name: Image preprocessing throughput
   repo:
     url: "https://github.com/your-org/your-repo.git"
     human_commit: "<40hex>"
     # pre_commit: "<40hex>"   # optional; if omitted defaults to human^ (first parent)
   runner:
     requires_gpu: false
     python_version: null
     platform: "linux/amd64"   # recommended on Apple Silicon
     allow_network_during_prepare: true
   env_build:
     allowed_strategies: [dockerfile, requirements]
   testpack:
     entrypoint: "../vlm-bench-generic"
   metrics: []  # can be empty for smoke tests
   scoring:
     primary: "functional_match"
     tie_breaker: "throughput_img_per_s"
   ```

3. **Validate the task**:
   ```bash
   PYTHONPATH=perf-agents-bench python3 -m bench.cli validate perf-agents-bench/tasks/example.yaml
   ```

4. **Fast smoke (build & run containers only)**:
   ```bash
   # Build baseline/human/agent from human commit only (no benchmarking)
   PYTHONPATH=perf-agents-bench python3 -m bench.cli smoke \
     perf-agents-bench/tasks/example.yaml \
     --bench-cfg perf-agents-bench/bench.yaml \
     --human-only \
     --cmd "python -c 'print(\"OK\")'"
   ```

5. **Run full benchmark** (optional later):
   ```bash
   PYTHONPATH=perf-agents-bench python3 -m bench.cli run perf-agents-bench/tasks/example.yaml
   ```

## Architecture

### Core Components

- **Container Runtime** (`bench/container/`): Docker/Podman abstraction with security constraints
- **Environment Builder** (`bench/env/`): Strategy pattern for building containers (Dockerfile, requirements.txt, etc.)
- **TestPack API** (`bench/testpack_api/`): Contract for external test implementations
- **Metrics Registry** (`bench/metrics/`): Plugin system for performance measurements
- **Agent Adapters** (`bench/agents/`): Interface for AI optimization agents
- **Pipeline** (`bench/pipeline.py`): Orchestrates the complete benchmark workflow

### Workflow

1. **Clone repository** and resolve pre-commit vs human-commit
2. **Build container** using repo's environment descriptor (Dockerfile, requirements.txt, etc.)
3. **Run three candidates** in identical containers:
   - **Baseline**: Pre-optimization commit
   - **Human**: Expert optimization commit  
   - **Agent**: AI-generated optimization
4. **Collect metrics** and generate comparison report
5. **Enforce constraints** (target file restrictions, etc.)

### CLI Overview

- `bench init` — scaffold example task and commits file
- `bench plan` — produce `state/plan.json` from commits file or YAML pairs
- `bench prepare` — run OpenHands locally per plan item; write journals; resumable
- `bench validate` — validate a task file (with env expansion)
- `bench run` — full benchmark (Stage B; containers)
- `bench smoke` — build/run images for quick sanity (Stage B)
- `bench doctor` — check `git`, `docker` (for Stage B), and OpenHands CLI

## Configuration

### Task Configuration (`tasks/example.yaml`)

```yaml
id: "image_decode"
name: "Image preprocessing throughput" 

repo:
  url: "https://github.com/your-org/your-repo.git"
  human_commit: "<40hex>"
  # pre_commit: "<40hex>"    # optional; if omitted defaults to first parent of human

env_build:
  allowed_strategies:
    - "dockerfile"
    - "requirements"

optimization_contract:
  strict_targets: true
  target_files:
    - "src/image_processing.py"

testpack:
  entrypoint: "../vlm-bench-generic"

metrics:
  - name: "throughput_img_per_s"
    using: "runtime:throughput"
    args:
      cmd: "python benchmark.py"
      trials: 10
```

### Commits file format (`.work/commits.txt`)

```text
# One line per item:
<40hex-human> [<40hex-pre>|parent=1]
```

If `pre` is omitted and `parent=K` is provided, the K-th parent of `human` will be used (useful for merge commits).

### Global Configuration (`bench.yaml`)

```yaml
container:
  engine: "${CONTAINER_ENGINE:-docker}"
  platform: "${CONTAINER_PLATFORM:-linux/amd64}"  # set linux/amd64 on Apple Silicon
  cpus: "${BENCH_CPUS:-2}"
  memory: "${BENCH_MEMORY:-4g}"

agents:
  default: "openhands"
  openhands:
    cli: "${OPENHANDS_CLI:-openhands}"
    time_budget_minutes: 30
```

## TestPack Development

TestPacks are external plugins that define how to run performance tests:

```python
from bench.testpack_api.contract import TestPack

class MyTestPack(TestPack):
    def prepare_fixtures(self, repo_dir, work_dir):
        # Set up test data
        pass
        
    def build(self, repo_dir, work_dir):
        # Build the repository
        pass
        
    def run_candidate(self, repo_dir, work_dir, out_dir, candidate_tag):
        # Execute performance test and save results
        pass

def pack():
    return MyTestPack()
```

## Environment Strategies

The system automatically detects how to build containers:

1. **Dockerfile**: Uses repo's Dockerfile as-is
2. **Requirements**: Creates container from requirements.txt + base template
3. **Poetry**: Uses pyproject.toml + poetry.lock
4. **Conda**: Uses environment.yml
5. **Nix**: Uses flake.nix or default.nix

## CLI Commands

- `bench run tasks/example.yaml`: Execute benchmark
- `bench validate tasks/example.yaml`: Validate task configuration
- `bench smoke tasks/example.yaml --human-only --cmd "python -c 'print(\"OK\")'"`: Build & run containers only

## No Cloud Dependencies

Unlike the original OmniPerf-Bench system, this implementation:
- ❌ **Removes**: SkyPilot dependency and cloud orchestration
- ❌ **Removes**: OpenAI API dependencies  
- ✅ **Keeps**: Container isolation patterns
- ✅ **Keeps**: Performance measurement concepts
- ✅ **Adds**: Local-first execution model

## Apple Silicon (macOS) Notes

- Many ML repos publish prebuilt wheels for x86_64 but not arm64. To avoid source builds:
  - Set `container.platform: linux/amd64` in `bench.yaml` (already templated)
  - Or set `runner.platform: linux/amd64` per task

## Agent (OpenHands) Containerization

- OpenHands is run via host CLI or its own container, never installed into candidate images.
- Configure in `bench.yaml`:
  ```yaml
  agents:
    default: openhands
    openhands:
      cli: "openhands"                  # used if container_image is empty
      time_budget_minutes: 30
      container_image: "ghcr.io/all-hands/openhands:latest"  # optional
  ```

## Image Tagging

- Images are tagged per candidate with commit short SHA:
  - `bench-<task>-baseline-<shortsha>`
  - `bench-<task>-human-<shortsha>`
  - `bench-<task>-agent-<shortsha>`