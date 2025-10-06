# OmniPerf-Bench

## 👋 overview
OmniPerf-Bench is a comprehensive framework for evaluating language models on software performance optimization tasks. The project provides:

- **Performance-focused dataset generation**: Extract real-world performance optimizations from commit histories
- **Automated test generation**: LLM-powered creation of performance tests for optimization tasks  
- **Evaluation harness**: Docker-based evaluation system for measuring optimization effectiveness
- **Multiple dataset views**: Compatible with GSO and SWE-Perf benchmark formats
- **Commit-to-dataset pipeline**: Streamlined conversion of individual commits to benchmark instances

Each task provides a codebase with performance bottlenecks, precise performance tests, and requires agents to generate patches that improve runtime efficiency. Success is measured against expert developer optimizations using wall-clock timing comparisons.

### Available Datasets
```python
from datasets import load_dataset

# Load GSO-compatible dataset
gso = load_dataset('gso-bench/gso', split='test')

# Load vLLM performance optimization dataset (282 problems)
vllm_data = load_dataset('Inferencebench/vllm_dataset_with_test', split='test')
```

## 📁 repository structure

```
OmniPerf-Bench/
├── README.md                    # Main project documentation
├── LICENSE                      # Project license
├── pyproject.toml              # Python project configuration (gso package)
├── uv.lock                     # Lock file for uv package manager
├── requirements.txt            # Python dependencies
├── commit_to_dataset.py        # Commit-to-dataset conversion pipeline
├── experiments.yaml            # Example experiment configuration
│
├── src/                        # Main OmniPerf-Bench source code
│   ├── collect/                # Collection framework for dataset generation
│   │   ├── analysis/           # Commit and API analysis modules
│   │   │   ├── commits.py      # Performance commit extraction
│   │   │   ├── apis.py         # API identification and mapping
│   │   │   ├── parser.py       # Code parsing utilities
│   │   │   └── retriever.py    # RAG-based code retrieval
│   │   ├── execute/            # Test execution and evaluation
│   │   │   ├── execute.py      # SkyPilot-based distributed execution
│   │   │   ├── evaluate.py     # Performance evaluation and metrics
│   │   │   └── skymgr.py       # Sky cluster management
│   │   ├── generate/           # Performance test generation
│   │   │   ├── generate.py     # Main test generation pipeline
│   │   │   ├── context.py      # Context extraction for tests
│   │   │   └── prompt.py       # LLM prompts for test generation
│   │   ├── scripts/            # Collection utility scripts
│   │   └── build_dataset.py    # Main dataset building script
│   ├── data/                   # Data models and parsing utilities
│   │   ├── commit.py           # Commit data structures
│   │   ├── dataset.py          # Dataset handling and validation
│   │   ├── problem.py          # Problem instance definitions
│   │   └── perf.py             # Performance measurement utilities
│   ├── harness/                # Evaluation harness for performance testing
│   │   ├── environment/        # Docker environment management
│   │   │   ├── docker_build.py # Docker image building
│   │   │   └── patches.py      # Environment patches
│   │   ├── grading/            # Grading and metrics evaluation
│   │   │   ├── grade.py        # Performance grading logic
│   │   │   └── metrics.py      # Evaluation metrics
│   │   ├── plot/               # Visualization and plotting tools
│   │   │   ├── plot_opt_k.py   # Opt@K performance plots
│   │   │   └── plot_speedups.py # Speedup visualization
│   │   ├── opt_at_k.py         # Main evaluation runner (Opt@K)
│   │   ├── prepare_images.py   # Docker image preparation
│   │   └── run_evaluation.py   # Evaluation orchestration
│   ├── test_scripts/           # Test generation and analysis scripts
│   │   ├── generate_test_generators.py # LLM test generator creation
│   │   ├── performance_analyzer.py     # Performance analysis tools
│   │   └── commit_analyzer.py          # Commit analysis utilities
│   ├── utils/                  # General utility functions
│   │   ├── io.py               # I/O utilities
│   │   ├── multiprocess.py     # Multiprocessing helpers
│   │   └── patch_parser.py     # Patch parsing utilities
│   ├── constants.py            # Project constants
│   └── logger.py              # Logging configuration
│
├── data/                       # Generated datasets
│   ├── Inferencebench.jsonl    # Inference benchmark dataset
│   └── vllm_dataset_with_test.jsonl # vLLM performance dataset (282 problems)
│
├── docs/                       # Documentation
│   └── dataset_schema.md       # Canonical dataset schema specification
│
├── third-party/               # External dependencies
│   └── effibench/             # Original EffiBench repository integration
│       ├── src/               # EffiBench source code
│       ├── data/              # EffiBench datasets
│       ├── prompts/           # LLM prompts for EffiBench
│       └── requirements.txt   # EffiBench dependencies
│
├── tools/                     # Utility scripts and patches
│   ├── manual_review.py       # Manual dataset review utilities
│   └── openrouter_patch.py    # OpenRouter API patches
│
└── misc/                      # Experimental data and results
    ├── experiments/           # Experimental data and configurations
    │   ├── commit_extractions/ # Extracted commit data (64 JSON files)
    │   ├── commit_extractions_with_apis/ # Commit data with API mappings
    │   ├── generated_test_generators_v4/ # Latest LLM test generators
    │   ├── vllm/              # vLLM experiment data and results
    │   │   ├── data/          # vLLM training data (parquet files)
    │   │   ├── divided/       # Split result files for parallel processing
    │   │   └── *.json         # vLLM experiment configurations and results
    │   ├── sglang.yaml        # SGLang experiment configuration
    │   └── vllm.yaml          # vLLM experiment configuration
    └── results/               # Analysis results, logs, and reviews
        ├── logs/              # Execution and system logs
        └── reviews/           # Manual review files (CSV format)
```

**Key Components:**
- **`src/collect/`** - Multi-stage dataset generation pipeline (commit extraction → API mapping → test generation → execution)
- **`src/harness/`** - Docker-based evaluation system with Opt@K metrics and visualization
- **`src/data/`** - Core data models and schema validation
- **`commit_to_dataset.py`** - Streamlined single-commit to dataset conversion
- **`data/`** - Generated benchmark datasets ready for use
- **`docs/dataset_schema.md`** - Canonical schema supporting GSO/SWE-Perf export views
- **`misc/experiments/`** - Real experimental data including 64 extracted commits and vLLM dataset (282 problems)

## 🚀 setup

### Prerequisites
- Python ≥ 3.12 (specified in pyproject.toml)
- CUDA-capable GPU (for performance testing)
- Docker (for containerized evaluation)
- Git with LFS support

### Cloning the Repository

This repository uses Git submodules for external dependencies. You must clone with the `--recursive` flag to get all required components:

```bash
# Clone repository with submodules (REQUIRED)
git clone --recursive git@github.com:thekernelcompany/OmniPerf-Bench.git
cd OmniPerf-Bench
```

**Important:** If you already cloned without `--recursive`, initialize submodules with:
```bash
git submodule update --init --recursive
```

**Submodules included:**
- `vllm/` - Modified vLLM fork with OmniPerf-Bench integration
- `third-party/trae-agent/` - TRAE agent integration for automated optimization

### Option 1: Using uv (recommended)
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Option 2: Using pip
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

**Required API Keys:**
```bash
# Create .env file or export environment variables
export OPENAI_API_KEY="your_openai_api_key"           # For LLM test generation
export ANTHROPIC_API_KEY="your_anthropic_api_key"     # Alternative to OpenAI
export GHAPI_TOKEN="your_github_token"                # For commit extraction
export HF_TOKEN="your_huggingface_token"              # For dataset uploads
```

**Optional for Cloud Execution:**
```bash
# For SkyPilot distributed execution
export AWS_ACCESS_KEY_ID="your_aws_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret"
# OR configure other cloud providers (GCP, Azure)
```

**Token Setup Links:**
- [OpenAI API Key](https://platform.openai.com/api-keys)
- [Anthropic API Key](https://console.anthropic.com/)
- [GitHub Token](https://github.com/settings/tokens) (needs repo access)
- [HuggingFace Token](https://huggingface.co/docs/hub/en/security-tokens)

### Verify Installation
```bash
# Check Python path setup
PYTHONPATH=src python -c "from collect.analysis.commits import PerfCommitAnalyzer; print('✓ Import successful')"

# Check Docker availability (for evaluation)
docker --version

# Check CUDA availability (for performance testing)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```


## 💽 usage

### 1. evaluation harness

The evaluation harness provides Docker-based performance testing with Opt@K metrics.

#### building docker images
```bash
# Login to Docker Hub
docker login

# Build and push Docker images for all dataset tasks
uv run src/harness/prepare_images.py \
    --dataset_name data/vllm_dataset_with_test.jsonl \
    --push_to_registry true \
    --dockerhub_username <your_username> \
    --dockerhub_repo <your_repo> \
    --max_workers 4
```

#### running evaluations
```bash
# Evaluate model predictions with Opt@K metrics
uv run src/harness/opt_at_k.py \
    --model <model_name> \
    --prediction_paths <path_to_predictions.jsonl> \
    --timeout 3600 \
    --run_id <unique_run_id> \
    --k 10 \
    --dataset_name data/vllm_dataset_with_test.jsonl
```

**Prediction format:** Your model's predictions should be a JSONL file with:
```json
{
    "instance_id": "task_identifier",
    "model_patch": "generated_patch_content",
    "model_name_or_path": "model_identifier"
}
```

For detailed options, see the [harness documentation](src/harness/README.md).

### 2. collection framework

The collection framework enables you to create your own performance optimization datasets through a four-stage pipeline:

#### stage 1: commit extraction & filtering
Extract performance-related commits from repositories using LLM analysis:
```bash
# Configure experiment (see experiments.yaml for example)
PYTHONPATH=src python src/collect/analysis/commits.py experiments.yaml
```

#### stage 2: api identification  
Map commits to affected high-level APIs using RAG-based retrieval:
```bash
PYTHONPATH=src python src/collect/analysis/apis.py <experiment_id>
```

#### stage 3: performance test generation
Generate LLM-powered performance tests for API-commit pairs:
```bash
PYTHONPATH=src python src/collect/generate/generate.py experiments.yaml
```

#### stage 4: test execution
Execute tests on cloud infrastructure using SkyPilot:
```bash
# Setup cloud credentials first: sky check
PYTHONPATH=src python src/collect/execute/execute.py --exp_id <experiment_id> --machines 4

# Evaluate results
PYTHONPATH=src python src/collect/execute/evaluate.py --exp_id <experiment_id>
```

**Experiment Configuration (experiments.yaml):**
```yaml
exp_id: "my_experiment"
repo_url: "https://github.com/user/repo"
py_version: 3.12
target_commit: "main"
install_commands:
  - "pip install -e ."
api_docs: "Focus on torch.nn and optimization APIs"
repo_instr: "Look for CUDA kernel optimizations"
```

For detailed instructions, see the [collection framework documentation](src/collect/README.md).

### 3. commit-to-dataset pipeline

The `commit_to_dataset.py` script provides a streamlined way to create canonical OmniPerf-Bench dataset records from individual performance optimization commits.

#### quick start
1. **Create configuration file** (`commit_config.yaml`):
```yaml
repo_path: "/path/to/your/repo"
head_commit: "abc123def456"  # The optimized commit
base_commit: null            # Optional: defaults to head_commit^
extractions_dir: "misc/experiments/commit_extractions_with_apis"
use_docker: false            # Set true for reproducible environments
docker_image: "ayushnangia16/nvidia-vllm-docker:latest"
dataset_name: "my_perf_dataset"
hf_repo: "your_username/dataset_repo"  # Optional HF upload
push_to_hf: false
```

2. **Run pipeline**:
```bash
# Ensure API keys are set
export OPENAI_API_KEY="your_key" # or ANTHROPIC_API_KEY

# Execute pipeline
PYTHONPATH=src python commit_to_dataset.py commit_config.yaml
```

3. **Output**: Creates `data/my_perf_dataset.jsonl` with canonical dataset record

#### performance measurement methodology

the script measures performance by:

1. **generating llm-based performance tests** for the specific commit optimization
2. **running tests on three commits**:
   - `base_commit`: the commit before optimization
   - `head_commit`: the commit with optimization
   - `main`: latest main branch commit
3. **measuring wall-clock execution time** using python's `time.time()`
4. **calculating performance metrics**:
   - `duration_changes`: raw timing data for all commits
   - `human_performance`: improvement ratio = `base_time / head_time`
     - `> 1.0` = performance improvement (head is faster)
     - `< 1.0` = performance regression (head is slower)
     - `= 1.0` = no performance change
     - `inf` = infinite improvement (near-zero head time)
     - `nan` = invalid/missing timing data

#### output format

the script generates:
- **jsonl file**: canonical dataset record in `data/` directory
- **optional huggingface push**: if `push_to_hf: true` and `hf_repo` specified
- **structured data** with fields:
  ```json
  {
    "repo": "owner/name",
    "instance_id": "owner__name-PR-123",
    "head_commit": "commit_hash",
    "patch": "unified_diff",
    "efficiency_test": ["generated_test_code"],
    "duration_changes": [{"base": [1.2], "head": [0.8], "main": [1.1]}],
    "human_performance": 1.5,
    "version": "python==3.11;arch=x86_64;image=local"
  }
  ```

#### requirements

- **llm api access**: openai or anthropic api key
- **commit extractions**: pre-extracted commit data in `extractions_dir`
- **docker (optional)**: for containerized test execution
- **python dependencies**: `pyyaml`, `datasets`, `pandas` (for hf push)

### 4. available datasets and experiments

#### pre-built datasets
- **vLLM Dataset** (`data/vllm_dataset_with_test.jsonl`): 282 real-world performance optimization problems from vLLM repository
- **Inference Benchmark** (`data/Inferencebench.jsonl`): Additional inference-focused performance tasks

#### experimental data
The `misc/experiments/` directory contains:
- **`commit_extractions/`** - 64 extracted performance-related commits (JSON format)
- **`commit_extractions_with_apis/`** - Same commits with API mapping annotations
- **`generated_test_generators_v4/`** - Latest LLM-generated test generators
- **`vllm/`** - Complete vLLM experiment data including:
  - Training data (parquet format)
  - Experiment configurations and results
  - Performance analysis outputs

#### experiment configurations
- **`vllm.yaml`** - vLLM repository analysis configuration
- **`sglang.yaml`** - SGLang repository analysis configuration  
- **`experiments.yaml`** - Example experiment template

#### effibench integration
EffiBench is integrated for additional code efficiency evaluation:
```bash
# Use original EffiBench
cd third-party/effibench
pip install -r requirements.txt
python src/open_source_model_completion.py  # Run efficiency evaluation

# Use OmniPerf-Bench test scripts
cd src/test_scripts
python performance_analyzer.py  # Analyze performance patterns
python generate_test_generators.py  # Create new test generators
```

## ⬇️ artifacts & resources

### 📊 datasets
| Dataset | Size | Description | Access |
|---------|------|-------------|--------|
| [GSO Benchmark](https://huggingface.co/datasets/gso-bench/gso) | Variable | General software optimization benchmark | `load_dataset('gso-bench/gso')` |
| [vLLM Performance Dataset](data/vllm_dataset_with_test.jsonl) | 282 problems | Real-world vLLM optimizations with tests | `load_dataset('Inferencebench/vllm_dataset_with_test')` |
| [Inference Benchmark](data/Inferencebench.jsonl) | Variable | Inference-focused performance tasks | Local file |
| [EffiBench Integration](third-party/effibench/) | Variable | Code efficiency benchmark | Local integration |

### 🛠️ tools & frameworks
| Component | Purpose | Entry Point |
|-----------|---------|-------------|
| **Evaluation Harness** | Docker-based Opt@K evaluation | `src/harness/opt_at_k.py` |
| **Collection Framework** | Multi-stage dataset generation | `src/collect/` |
| **Commit-to-Dataset Pipeline** | Single commit conversion | `commit_to_dataset.py` |
| **Performance Analysis** | Performance pattern analysis | `src/test_scripts/performance_analyzer.py` |
| **Test Generation** | LLM-powered test creation | `src/test_scripts/generate_test_generators.py` |
| **Manual Review Tools** | Dataset quality control | `tools/manual_review.py` |

### 🐳 docker resources
- **Docker Hub Repository**: [slimshetty/gso](https://hub.docker.com/repository/docker/slimshetty/gso/general)
- **Image Building**: `src/harness/prepare_images.py`
- **Pre-built Images**: `src/harness/scripts/pull_images.sh`

### 📚 documentation
- **Dataset Schema**: [docs/dataset_schema.md](docs/dataset_schema.md) - Canonical schema specification
- **Collection Framework**: [src/collect/README.md](src/collect/README.md) - Detailed pipeline documentation
- **Evaluation Harness**: [src/harness/README.md](src/harness/README.md) - Evaluation system guide
- **Test Scripts**: [src/test_scripts/README.md](src/test_scripts/README.md) - Analysis tools documentation
