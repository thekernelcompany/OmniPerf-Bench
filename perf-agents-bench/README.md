# Performance Agents Benchmark

A modular, engine-agnostic performance benchmarking system for evaluating AI agents' ability to optimize code performance.

## Key Features

- **No hard-coded assumptions**: Every parameter must be explicit in configuration
- **Plugin architecture**: TestPacks, metrics, and agents are external/pluggable
- **Local execution**: No cloud dependencies (no SkyPilot), runs entirely with local Docker/Podman
- **Hermetic execution**: Identical container constraints for all candidates
- **Config-driven**: Repository URLs, commits, and tools specified via environment variables

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