# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**OmniPerf-Bench** is a comprehensive framework for evaluating language models on software performance optimization tasks. It provides a complete pipeline from extracting real-world performance optimizations from git commits to generating benchmark datasets with automated performance tests and evaluation harnesses.

### Core Purpose
- Extract performance-focused optimizations from commit histories
- Generate realistic performance tests using LLMs
- Evaluate AI agents on their ability to reproduce expert optimizations
- Provide standardized benchmarks compatible with GSO and SWE-Perf formats

### Key Capabilities
1. **Dataset Generation**: Convert git commits into structured benchmark instances
2. **Test Generation**: LLM-powered creation of performance tests
3. **Evaluation**: Docker-based evaluation with Opt@K metrics
4. **Agent Integration**: Test AI agents (OpenHands, TRAE) on optimization tasks

---

## Repository Structure

```
OmniPerf-Bench/
├── commit_to_dataset.py        # 🔥 PRIMARY ENTRY POINT - Single commit to dataset conversion
├── run_commit_optimization.py   # Single-commit agent optimization pipeline
├── batch_commit_optimization.py # Batch processing for multiple commits
├── configs/                     # Configuration files
│   ├── experiments.yaml         # Example batch configuration
│   └── single_commit_config.yaml
├── src/                        # Main source code
│   ├── collect/                # Dataset collection framework
│   │   └── execute/            # Test execution (SkyPilot)
│   ├── harness/                # Evaluation harness
│   │   ├── opt_at_k.py         # Main Opt@K evaluation
│   │   └── scripts/
│   └── test_scripts/           # Test generation utilities
│       ├── generate_test_generators.py  # LLM-based test generator
│       ├── performance_analyzer.py
│       └── commit_analyzer.py
├── data/                       # Generated datasets (JSONL)
│   ├── vllm_dataset_with_test.jsonl  # 282 vLLM problems
│   └── Inferencebench.jsonl
├── misc/experiments/           # Experimental data
│   ├── commit_extractions_with_apis/  # 64 extracted commits
│   └── generated_test_generators_v4/  # Pre-generated test scripts
├── perf-agents-bench/          # OpenHands agent integration
│   ├── bench/                  # CLI and orchestration
│   ├── tasks/                  # Task definitions
│   └── state/                  # Execution state and results
├── third-party/
│   ├── effibench/              # EffiBench integration
│   └── trae-agent/             # TRAE agent submodule
├── tools/                      # Utility scripts
├── docs/                       # Documentation
│   ├── dataset_schema.md       # Canonical schema spec
│   ├── omni_commit_architecture.md  # Detailed architecture
│   └── [various integration guides]
└── vllm/                       # vLLM submodule (git@github.com:Itssshikhar/vllm.git)
```

---

## Architecture Overview

### High-Level Pipeline Flow

```
1. COMMIT EXTRACTION
   └─> PerfCommitAnalyzer extracts commits with performance impacts
   
2. TEST GENERATION
   └─> LLM generates realistic performance tests for each commit
   
3. DATASET CREATION
   └─> Tests executed on base/head/main commits; timings recorded
   
4. AGENT EVALUATION
   └─> AI agents attempt to reproduce optimizations
   
5. METRICS & ANALYSIS
   └─> Opt@K metrics, speedup comparisons, success rates
```

### Key Components

#### 1. **Commit-to-Dataset Pipeline** (`commit_to_dataset.py`)
The primary entry point for dataset creation. Orchestrates the entire flow:

**Process:**
1. Scans extraction JSONs in `misc/experiments/commit_extractions_with_apis/`
2. For each commit (head + base):
   - Clones repo to temporary workspace
   - Extracts commit metadata and diffs using `PerfCommitAnalyzer`
   - Finds or generates test script via LLM
   - Materializes test into `_generated_perf_tests/test_generated.py`
   - Runs test on base/head/main commits
   - Records timing arrays and computes speedup
3. Assembles canonical dataset records
4. Outputs to `data/<dataset_name>.jsonl`
5. Optionally pushes to HuggingFace

**Key Functions:**
- `scan_extraction_files()`: Discovers commits from JSON files
- `find_or_generate_test_script()`: Resolves test generator
- `assemble_canonical()`: Creates dataset record with timings
- `save_and_push()`: Writes JSONL and uploads to HF

#### 2. **Test Generation** (`src/test_scripts/generate_test_generators.py`)
LLM-powered test script generation:

**Prompt Construction:**
- Reads template: `src/test_scripts/prompts/claude_4_prompt_v2.md` (default)
- Alternative prompts available in `src/test_scripts/prompts/`
- Loads commit extraction JSON with metadata
- Builds final prompt combining template + commit context
- Saves prompt to `eg_test_generator.txt` for debugging

**LLM Client:**
- Supports OpenAI, Anthropic, AWS Bedrock, OpenRouter
- Auto-selects based on environment variables
- Bedrock models mapped automatically (e.g., claude-4 → anthropic.claude-sonnet-4-20250514-v1:0)

**Code Generation & Validation:**
- Extracts Python code from LLM response
- Validates with `ast.parse()`
- Up to 2 regeneration attempts if code invalid
- Up to 2 repair passes with compiler error feedback
- Saves to `misc/experiments/generated_test_generators_v4/<hash8>_test_case_generator.py`

**Generated Test Structure:**
```python
def setup_workload():
    # Realistic tensors, seeds, shapes
    
def test_correctness():
    # Verify optimized code produces same results
    
def test_performance():
    # Measure speedup with timeit/CUDA events
    # 20% tolerance for regression detection
```

#### 3. **Evaluation Harness** (`src/harness/`)
Docker-based performance evaluation system:

**Components:**
- `opt_at_k.py`: Main Opt@K evaluation runner
- `prepare_images.py`: Build Docker images per task
- `run_evaluation.py`: Orchestrate evaluations
- `grading/`: Performance grading logic
- `plot/`: Visualization tools (speedup plots, Opt@K curves)

**Workflow:**
1. Build Docker images for each dataset task
2. Run agent predictions in isolated containers
3. Measure wall-clock timing with CUDA when available
4. Compute Opt@K metrics (pass@k over k attempts)
5. Generate reports and visualizations

**Evaluation Metrics:**
- `opt_base`: Improvements over baseline (pre-commit)
- `opt_commit`: Match/exceed human optimization
- `opt_main`: Improvements over current main branch
- `human_performance`: Ratio of base/head timing

#### 4. **Agent Integration** (`perf-agents-bench/`)
Minimal framework for running AI agents on optimization tasks:

**Architecture:**
```
CLI Commands:
  - plan: Create commit pairs from task config
  - prepare: Execute agents on worktrees
  - report: Aggregate results and metrics
  - validate: Check git/docker setup
  - doctor: Diagnose environment issues

Core Components:
  - MatrixPlanner: Resolves commits from task YAML
  - PrepareExecutor: Orchestrates agent runs
  - RepoManager: Git worktree operations
  - JournalWriter: Logs all outputs
  - OpenHands Agent: Via uvx/docker
```

**Workflow:**
1. **Plan**: `bench.cli plan tasks/vllm.yaml` → generates `state/plan.json`
2. **Prepare**: Creates git worktrees, runs OpenHands headless mode
3. **Execute**: Agent modifies code, commits changes
4. **Journal**: Captures stdout/stderr, diffs, metrics
5. **Report**: Aggregates success rates and performance

#### 5. **Dataset Schema** (Canonical Format)
Defined in `docs/dataset_schema.md`:

**Required Fields:**
```json
{
  "repo": "owner/name",
  "instance_id": "owner__name-PR-123",
  "created_at": "2023-07-10T12:30:00Z",
  "base_commit": "hash_before",
  "head_commit": "hash_optimized",
  "patch": "unified_diff_non_test",
  "test_patch": "unified_diff_test_files",
  "efficiency_test": ["test_script_code"],
  "duration_changes": [{"base": [2.1, 2.0], "head": [1.5, 1.4]}],
  "human_performance": 1.38,
  "version": "python==3.9;arch=x86_64;image=latest"
}
```

**Export Views:**
- **GSO View**: Compatible with GSO benchmark format
- **SWE-Perf View**: Compatible with SWE-Perf format

---

## Technology Stack

### Core Dependencies
- **Python**: 3.12+ (required by OpenHands)
- **PyTorch**: GPU/CUDA support for performance tests
- **Docker**: Containerized evaluation environments
- **Git + LFS**: Repository operations and large files

### Key Libraries
```python
# Dataset and ML
datasets>=3.2.0              # HuggingFace datasets
huggingface-hub>=0.27.0      # Dataset uploads
torch                        # Performance tests

# LLM Providers
openai                       # OpenAI API
anthropic                    # Anthropic Claude
boto3/botocore              # AWS Bedrock

# Agent Frameworks
openhands-ai>=0.10.0        # OpenHands agent
# trae-agent (submodule)    # TRAE agent

# Infrastructure
docker>=7.1.0               # Container management
skypilot>=0.5.0            # Distributed execution
ghapi>=1.0.6               # GitHub API access

# Utilities
rich>=13.8.0               # Terminal formatting
fire>=0.6.0                # CLI interfaces
pyyaml>=6                  # Config parsing
```

### Package Management
- **uv** (recommended): Fast Python package manager
- **pip**: Traditional pip install
- Lock file: `uv.lock` (609KB)

---

## Key Workflows

### 1. Create Dataset from Single Commit

```bash
# Set up environment
export OPENAI_API_KEY="sk-..."
# OR
export AWS_REGION="us-east-1"
aws sso login

# Run pipeline
python commit_to_dataset.py configs/experiments.yaml

# Output: data/vllm_dataset_with_test.jsonl
```

**What Happens:**
1. Reads all JSONs in `misc/experiments/commit_extractions_with_apis/`
2. For each commit:
   - Extracts metadata with `PerfCommitAnalyzer`
   - Generates test script via LLM (or uses pre-generated)
   - Runs test on base/head/main commits
   - Records timings and speedup
3. Writes canonical dataset records

### 2. Generate Performance Tests

```bash
# Generate test for specific commit
PYTHONPATH=src python src/test_scripts/generate_test_generators.py \
  --extractions-dir misc/experiments/commit_extractions_with_apis \
  --output-dir misc/experiments/generated_test_generators_v4

# Check generated prompt
cat eg_test_generator.txt

# Find generated script
ls misc/experiments/generated_test_generators_v4/*_test_case_generator.py
```

### 3. Run Agent on Single Commit

```bash
# Single commit optimization
python run_commit_optimization.py \
  --commit-json tmp_single_commit/0ec82edd.json \
  --test-script misc/experiments/generated_test_generators_v4/0ec82edd_test_case_generator.py \
  --repo-path vllm

# Batch processing
python batch_commit_optimization.py \
  --commit-dir tmp_single_commit/ \
  --test-dir misc/experiments/generated_test_generators_v4/ \
  --repo-path vllm \
  --output-dir results/
```

### 4. Evaluate with Opt@K Metrics

```bash
# Build Docker images
uv run src/harness/prepare_images.py \
  --dataset_name data/vllm_dataset_with_test.jsonl \
  --push_to_registry true \
  --dockerhub_username <user> \
  --dockerhub_repo <repo>

# Run evaluation
uv run src/harness/opt_at_k.py \
  --model <model_name> \
  --prediction_paths predictions.jsonl \
  --timeout 3600 \
  --k 10 \
  --dataset_name data/vllm_dataset_with_test.jsonl
```

### 5. OpenHands Agent Pipeline

```bash
cd perf-agents-bench

# 1. Plan
echo "f092153fbe parent=1" > .work/vllm_commits.txt
.venv/bin/python -m bench.cli plan tasks/vllm.yaml \
  --commits .work/vllm_commits.txt \
  --out state/plan.json

# 2. Execute agents
.venv/bin/python -m bench.cli prepare tasks/vllm.yaml \
  --from-plan state/plan.json \
  --bench-cfg bench.yaml \
  --max-workers 1 --resume

# 3. Report results
LATEST=$(ls -t state/runs | head -n1)
.venv/bin/python -m bench.cli report state/runs/$LATEST
```

---

## Configuration Files

### `configs/experiments.yaml` (Batch Mode)
```yaml
repo_path: "/root/OmniPerf-Bench/vllm"
extractions_dir: "/root/OmniPerf-Bench/misc/experiments/commit_extractions_with_apis"
use_docker: false
docker_image: "ayushnangia16/nvidia-vllm-docker:latest"
dataset_name: vllm_dataset_with_test
hf_repo: https://huggingface.co/Inferencebench
push_to_hf: false

# LLM Configuration
llm_provider: openai        # or anthropic, bedrock
llm_model: gpt-4o-mini     # or claude-3-sonnet, opus-4-1
llm_temperature: 0.1
llm_max_tokens: 65536
```

### Environment Variables
```bash
# Required (choose one LLM provider)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
# OR AWS Bedrock
export AWS_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."

# Optional
export GHAPI_TOKEN="..."              # GitHub access
export HF_TOKEN="..."                 # HuggingFace uploads
export TEST_CASE_GENERATOR_PROMPT="..." # Custom prompt template
```

---

## Data Flow

### Performance Measurement Methodology

```
1. BASE COMMIT (pre-optimization)
   └─> Checkout base commit
   └─> Run generated test
   └─> Record timing: base_times = [t1, t2, ...]

2. HEAD COMMIT (optimized)
   └─> Checkout head commit  
   └─> Run same test
   └─> Record timing: head_times = [t1, t2, ...]

3. MAIN COMMIT (current)
   └─> Checkout main branch
   └─> Run same test
   └─> Record timing: main_times = [t1, t2, ...]

4. COMPUTE METRICS
   └─> human_performance = mean(base) / mean(head)
   └─> > 1.0 = improvement (head faster)
   └─> < 1.0 = regression (head slower)
   └─> = 1.0 = no change
```

### Timing Implementation
- **CUDA available**: `torch.cuda.Event()` for precise GPU timing
- **CPU only**: `time.time()` for wall-clock timing
- **Docker mode**: Same logic inside container with `--gpus all`

---

## Important Paths & Artifacts

### Input Data
```
misc/experiments/commit_extractions_with_apis/*.json
  → Extracted commit metadata (64 vLLM commits)
  → Fields: commit_hash, parent_hash, message, files, affected_apis

misc/experiments/sglang_commit_extractions_with_apis/*.json
  → Extracted commit metadata (80 SGLang commits)
  → Same structure as vLLM commits

src/test_scripts/prompts/
  → LLM prompt templates for test generation
  → claude_4_prompt_v2.md (default)
  → focused_test_case_generator_prompt.md
  → Other prompt variants
```

### Generated Artifacts
```
eg_test_generator.txt
  → Full prompt sent to LLM (debugging)

misc/experiments/generated_test_generators_v4/<hash8>_test_case_generator.py
  → Generated test scripts (pre-generated or LLM-generated)

data/<dataset_name>.jsonl
  → Canonical dataset output

commit_to_dataset.log
  → Pipeline execution log
```

### Execution State
```
perf-agents-bench/state/
  ├── plan.json                    # Commit pair plan
  └── runs/<run_id>/<item_id>/     # Agent execution logs
      ├── task.txt                 # Agent task description
      ├── prompt.json              # Formatted prompt
      ├── journal.json             # Execution journal
      ├── diff_targets.json        # File changes
      ├── openhands_stdout.txt     # Agent output
      └── openhands_stderr.txt     # Agent errors
```

---

## Development Patterns

### Adding a New LLM Provider
1. Add API key environment variable check
2. Implement client in `src/test_scripts/generate_test_generators.py`
3. Add to `LLMClient.__init__()` provider detection
4. Handle streaming/non-streaming responses
5. Map to standard response format

### Customizing Test Generation
1. Create custom prompt template (markdown)
2. Set `TEST_CASE_GENERATOR_PROMPT` environment variable
3. Maintain required function signatures:
   - `setup_workload()`
   - `test_correctness()`
   - `test_performance()`

### Adding New Agent
1. Create agent class in `perf-agents-bench/bench/agents/`
2. Inherit from `AgentBase`
3. Implement `execute()` method
4. Register in agent registry
5. Add configuration in `bench.yaml`

---

## Common Issues & Solutions

### 1. ModuleNotFoundError
**Problem**: `No module named 'collect'`
**Solution**: Always run from repo root; scripts handle `PYTHONPATH` automatically
```bash
# Correct
python commit_to_dataset.py configs/experiments.yaml

# Incorrect
cd src && python ../commit_to_dataset.py
```

### 2. LLM Generation Failures
**Problem**: LLM returns invalid/empty code
**Solution**: 
- Use robust models (GPT-4, Claude Opus)
- Lower temperature (0.1)
- Check `eg_test_generator.txt` for prompt issues
- Seed with pre-generated scripts in `misc/experiments/generated_test_generators_v4/`

### 3. CUDA Timing Issues
**Problem**: Tests skip or show no performance signal
**Solution**:
- Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- Use Docker with `--gpus all`
- Check test script has graceful fallbacks

### 4. AWS Bedrock Access
**Problem**: Model access denied
**Solution**:
```bash
# Check credentials
aws sts get-caller-identity

# Check model access
aws bedrock list-foundation-models --region us-east-1 --by-provider anthropic

# Request access in AWS Console → Bedrock → Model Access
```

### 5. Git Submodules Missing
**Problem**: `vllm/` or `third-party/trae-agent/` empty
**Solution**:
```bash
git submodule update --init --recursive
```

---

## Testing & Validation

### Verify Installation
```bash
# Check Python imports
PYTHONPATH=src python -c "from collect.analysis.commits import PerfCommitAnalyzer; print('✓ Imports OK')"

# Check Docker
docker --version

# Check CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Check AWS Bedrock (if using)
aws bedrock list-foundation-models --region us-east-1 --by-provider anthropic
```

### Run Small Test
```bash
# Create test config with single commit
cat > test_config.yaml << EOF
repo_path: "vllm"
extractions_dir: "misc/experiments/commit_extractions_with_apis"
use_docker: false
dataset_name: test_dataset
push_to_hf: false
llm_provider: openai
llm_model: gpt-4o-mini
EOF

# Run on first commit only
python commit_to_dataset.py test_config.yaml
```

---

## Submodules & Integrations

### vLLM Fork (`vllm/`)
- **URL**: `git@github.com:Itssshikhar/vllm.git`
- **Purpose**: Modified vLLM with OmniPerf-Bench integration
- **Usage**: Target repository for performance optimization tasks

### TRAE Agent (`third-party/trae-agent/`)
- **URL**: `git@github.com:agokrani/trae-agent.git`
- **Purpose**: TRAE agent for automated optimization
- **Setup**: Run `install_trae_integration.sh`

### EffiBench (`third-party/effibench/`)
- **Purpose**: Code efficiency benchmark integration
- **Key Files**:
  - `prompts/focused_test_case_generator_prompt.md`: Test generation template
  - `src/`: EffiBench evaluation code

---

## Performance Metrics

### human_performance Ratio
```
human_performance = mean(base_times) / mean(head_times)

> 1.0  = Optimization (head is faster)
< 1.0  = Regression (head is slower)  
= 1.0  = No change
= inf  = Infinite improvement (near-zero head time)
= nan  = Invalid/missing timing data
```

### Opt@K Metrics
- **Pass@K**: Probability of success in K attempts
- **opt_base**: Improvements over baseline
- **opt_commit**: Match/exceed human optimization
- **opt_main**: Improvements over current main

---

## Future Claude Code Users

### Quick Start
1. **Understand the pipeline**: Read `docs/omni_commit_architecture.md`
2. **Find entry points**: `commit_to_dataset.py` is the main script
3. **Check configuration**: `configs/experiments.yaml`
4. **Follow data flow**: Extraction JSON → Test Generation → Dataset Record
5. **Review logs**: `commit_to_dataset.log` for execution details

### Code Navigation Tips
- **Dataset generation**: Start with `commit_to_dataset.py`
- **Test generation**: `src/test_scripts/generate_test_generators.py`
- **Commit analysis**: `src/collect/analysis/commits.py`
- **Evaluation**: `src/harness/opt_at_k.py`
- **Agent orchestration**: `perf-agents-bench/bench/`

### Common Tasks
- **Add support for new repo**: Update extraction logic in `src/collect/analysis/`
- **Modify test template**: Edit prompts in `src/test_scripts/prompts/`
- **Change metrics**: Update `src/harness/grading/`
- **Add agent**: Implement in `perf-agents-bench/bench/agents/`

### Debug Helpers
- **Verbose logging**: Check `commit_to_dataset.log`
- **Prompt inspection**: Read `eg_test_generator.txt`
- **Test validation**: Run generated scripts manually
- **Docker issues**: Use `docker logs <container>`

---

## Additional Resources

### Documentation Files
- `README.md`: User-facing documentation
- `docs/dataset_schema.md`: Canonical schema specification
- `docs/omni_commit_architecture.md`: Detailed architecture
- `docs/TRAE_AGENT_REPLICATION_GUIDE.md`: TRAE setup guide
- `docs/COMMIT_OPTIMIZATION_README.md`: Optimization workflow
- `perf-agents-bench/ARCHITECTURE.md`: Agent system architecture

### External Links
- [GSO Benchmark](https://huggingface.co/datasets/gso-bench/gso)
- [SWE-Perf Dataset](https://huggingface.co/datasets/SWE-Perf/SWE-Perf)
- [vLLM Dataset](https://huggingface.co/datasets/Inferencebench/vllm_dataset_with_test)
- [OpenHands Documentation](https://docs.all-hands.dev/)
- [Docker Hub: slimshetty/gso](https://hub.docker.com/repository/docker/slimshetty/gso)

---

## Summary

OmniPerf-Bench is a multi-stage pipeline that:
1. Extracts performance commits from repositories
2. Generates realistic performance tests using LLMs
3. Creates structured datasets with timing measurements
4. Evaluates AI agents on optimization tasks
5. Provides standardized metrics and benchmarks

The codebase is organized around clear entry points (`commit_to_dataset.py`, `run_commit_optimization.py`) with modular components for collection, generation, execution, and evaluation. Key design principles include reproducibility, extensibility, and compatibility with existing benchmarks (GSO, SWE-Perf).

For detailed operation, start with `python commit_to_dataset.py configs/experiments.yaml` and follow the logs to understand data flow.
