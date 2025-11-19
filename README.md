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
│
├── 🚀 commit_to_dataset.py     # 🔥 Main entry point: Single-commit dataset pipeline
├── experiments.yaml            # Example experiment configuration
├── 8d75fe48_test_case_generator_v2.py  # Example test generator script
├── eg_test_generator.txt       # Example prompt template
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

**🎯 Repository Organization Philosophy:**

This repository follows a **"no duplicates, clear entry points"** structure:

- **Root level** - Main entry points and configuration files
- **`commit_to_dataset.py`** - 🔥 **PRIMARY ENTRY POINT** for single-commit dataset creation
- **`src/`** - Organized source code by functionality (collect, harness, data, utils)  
- **`misc/`** - Experimental data and working files (preserved as-is for research)
- **`data/`** - Generated datasets ready for consumption
- **`docs/`** - Documentation and schemas

**Key Components:**
- **`commit_to_dataset.py`** - 🚀 **Main script**: Streamlined single-commit to dataset conversion  
- **`src/collect/`** - Multi-stage dataset generation pipeline (commit extraction → API mapping → test generation → execution)
- **`src/harness/`** - Docker-based evaluation system with Opt@K metrics and visualization
- **`src/data/`** - Core data models and schema validation
- **`experiments.yaml`** - Example configuration that works out-of-the-box
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

# For AWS Bedrock (alternative to OpenAI/Anthropic)
export AWS_REGION="us-east-1"                         # AWS region for Bedrock
# AWS credentials via one of:
export AWS_ACCESS_KEY_ID="your_aws_access_key"        # Method 1: Environment variables
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
# OR use AWS SSO: aws sso login --sso-session your-session  # Method 2: SSO
# OR use ~/.aws/credentials file                            # Method 3: Credential file
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
- [AWS Bedrock Setup](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html) (requires model access)
- [GitHub Token](https://github.com/settings/tokens) (needs repo access)
- [HuggingFace Token](https://huggingface.co/docs/hub/en/security-tokens)

### AWS Bedrock Setup (Optional)

For using Claude Opus 4.1 via AWS Bedrock (highest quality option):

```bash
# 1. Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# 2. Configure credentials (choose one method)

# Method A: SSO (recommended)
aws configure sso
aws sso login --sso-session your-session

# Method B: Access keys
aws configure
# Enter your AWS Access Key ID and Secret Access Key

# Method C: Environment variables
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"

# 3. Enable Bedrock model access
# Go to AWS Console > Bedrock > Model Access
# Request access to Anthropic Claude models

# 4. Verify access
aws bedrock list-foundation-models --region us-east-1 --by-provider anthropic
```

**Supported Bedrock Models:**
- `us.anthropic.claude-opus-4-1-20250805-v1:0` (Claude Opus 4.1) - Highest quality
- `us.anthropic.claude-sonnet-4-20250514-v1:0` (Claude Sonnet 4)
- `anthropic.claude-3-5-sonnet-20241022-v2:0` (Claude 3.5 Sonnet)

### Verify Installation
```bash
# Check Python path setup
PYTHONPATH=src python -c "from collect.analysis.commits import PerfCommitAnalyzer; print('✓ Import successful')"

# Check Docker availability (for evaluation)
docker --version

# Check CUDA availability (for performance testing)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Check AWS Bedrock access (if using Bedrock)
aws sts get-caller-identity
aws bedrock list-foundation-models --region us-east-1 --by-provider anthropic --query "modelSummaries[?contains(modelId, 'claude-opus-4-1')]"
```


## 💽 usage

**🔥 TL;DR - Quick Start:**
```bash
# Option 1: Using OpenAI
export OPENAI_API_KEY="your_key"
python commit_to_dataset.py experiments.yaml

# Option 2: Using AWS Bedrock (Claude Opus 4.1)
export AWS_REGION="us-east-1"
aws sso login --sso-session your-session  # or configure AWS credentials
python commit_to_dataset.py experiments.yaml

# Your dataset appears in data/vllm_dataset_with_test.jsonl
```

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

### 3. 🚀 commit-to-dataset pipeline (quickest start)

The `commit_to_dataset.py` script is the **main entry point** for creating performance optimization datasets from individual commits. This is the simplest way to get started!

#### ⚡ super quick start
```bash
# 1. Set up API keys (choose one option)

# Option A: OpenAI
export OPENAI_API_KEY="your_openai_key"

# Option B: Anthropic
export ANTHROPIC_API_KEY="your_anthropic_key"

# Option C: AWS Bedrock (Claude Opus 4.1 - highest quality)
export AWS_REGION="us-east-1"
aws sso login --sso-session your-session

# 2. Run the pipeline on the provided example configuration
python commit_to_dataset.py experiments.yaml

# 3. Your dataset will be created in data/vllm_dataset_with_test.jsonl
```

#### 📝 custom configuration
Create your own configuration file (`my_config.yaml`):
```yaml
repo_path: "/path/to/your/repo"
extractions_dir: "misc/experiments/commit_extractions_with_apis"
use_docker: false            # Set true for reproducible environments
docker_image: "ayushnangia16/nvidia-vllm-docker:latest"
dataset_name: "my_perf_dataset"
hf_repo: "your_username/dataset_repo"  # Optional HF upload
push_to_hf: false

# LLM Configuration
llm_provider: "openai"       # or "anthropic" or "bedrock"
llm_model: "gpt-4o-mini"     # or "claude-3-sonnet-20240229" or "us.anthropic.claude-opus-4-1-20250805-v1:0"
llm_temperature: 0.1
llm_max_tokens: 4096
```

Then run:
```bash
python commit_to_dataset.py my_config.yaml
```

#### 📤 output
- **Local file**: `data/{dataset_name}.jsonl` with canonical dataset records
- **Optional HF upload**: Pushes to HuggingFace if `push_to_hf: true`

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

- **llm api access**: one of:
  - OpenAI API key
  - Anthropic API key 
  - AWS Bedrock access (requires model permissions)
- **commit extractions**: pre-extracted commit data in `extractions_dir`
- **docker (optional)**: for containerized test execution
- **python dependencies**: `pyyaml`, `datasets`, `pandas` (for hf push)

#### 🔧 troubleshooting

**Common Issues:**

1. **"No such file or directory: commit_to_dataset.py"**
   ```bash
   # Make sure you're in the repository root
   cd /path/to/OmniPerf-Bench
   ls commit_to_dataset.py  # Should exist
   ```

2. **"ModuleNotFoundError: No module named 'collect'"**
   ```bash
   # Always run from the repository root, script handles PYTHONPATH automatically
   python commit_to_dataset.py experiments.yaml
   ```

3. **"Config file not found"**
   ```bash
   # Use the provided example configuration
   python commit_to_dataset.py experiments.yaml
   # Or specify full path to your config
   python commit_to_dataset.py /full/path/to/my_config.yaml
   ```

4. **"Missing LLM credentials"**
   ```bash
   # Set at least one API key/credential
   export OPENAI_API_KEY="sk-your-key-here"
   # OR
   export ANTHROPIC_API_KEY="your-anthropic-key"
   # OR for AWS Bedrock
   export AWS_REGION="us-east-1"
   aws sso login --sso-session your-session
   ```

5. **"No valid extraction files found"**
   ```bash
   # Check that the extractions directory exists and has JSON files
   ls misc/experiments/commit_extractions_with_apis/*.json
   ```

6. **AWS Bedrock Issues**
   ```bash
   # Check AWS credentials
   aws sts get-caller-identity
   
   # Check Bedrock model access
   aws bedrock list-foundation-models --region us-east-1 --by-provider anthropic
   
   # If using SSO, ensure session is active
   aws sso login --sso-session your-session
   
   # For "streaming required" errors - this is automatically handled
   # For "system role" errors - this is automatically handled
   # For "inference profile" errors - check model access in AWS Console
   ```

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

## 🔄 resuming trae pipeline with filtered commits

### Quick Start: Resume Pipeline

If you have partially completed commits in `state/` and want to process only the remaining ones:

```bash
# 1. Activate environment and set API key
cd /home/ubuntu/OmniPerf-Bench
source bench-env/bin/activate
export OPENAI_API_KEY="your_openai_api_key"

# 2. Audit what's already done
cd perf-agents-bench
python3 << 'EOF'
import json
from pathlib import Path
from collections import defaultdict

runs_dir = Path("state/runs")
commits_by_status = defaultdict(set)

for journal_path in runs_dir.glob("*/*/journal.json"):
    try:
        data = json.loads(journal_path.read_text())
        status = data.get("status", "unknown")
        human_commit = data.get("commits", {}).get("human", None)
        if human_commit:
            commits_by_status[status].add(human_commit)
    except:
        pass

successful = commits_by_status["success"]
print(f"✅ Completed: {len(successful)} commits")

with open("completed_commits.txt", "w") as f:
    for commit in sorted(successful):
        f.write(f"{commit}\n")
EOF

# 3. Create filtered plan (only unprocessed commits)
python3 << 'EOF'
import json
from pathlib import Path

plan = json.loads(Path("state/plan.json").read_text())
completed = set(Path("completed_commits.txt").read_text().strip().split("\n"))

missing_items = [item for item in plan["items"] if item["human"] not in completed]

filtered_plan = {
    "repo": plan["repo"],
    "task_id": plan["task_id"],
    "items": missing_items
}

Path("state/plan_remaining.json").write_text(json.dumps(filtered_plan, indent=2))
print(f"✅ Created plan_remaining.json with {len(missing_items)} commits")
EOF

# 4. Set TRAE environment variables
export TRAE_PYTHON=/home/ubuntu/OmniPerf-Bench/bench-env/bin/python
export TRAE_CONFIG=/home/ubuntu/OmniPerf-Bench/third-party/trae-agent/trae_config.yaml

# 5. Run pipeline with filtered plan
python -m bench.cli prepare \
    tasks/vllm.yaml \
    --from-plan state/plan_remaining.json \
    --bench-cfg bench.yaml \
    --max-workers 1 \
    --resume
```

### Configuration Requirements

**Before running, ensure TRAE is properly configured:**

1. **Initialize TRAE submodule:**
   ```bash
   git submodule update --init --recursive third-party/trae-agent
   ```

2. **Install TRAE agent:**
   ```bash
   source bench-env/bin/activate
   uv pip install -e third-party/trae-agent
   ```

3. **Configure TRAE for OpenAI:**
   
   Edit `third-party/trae-agent/trae_config.yaml`:
   ```yaml
   model_providers:
       openai:
           api_key: ${OPENAI_API_KEY}
           provider: openai
   
   models:
       trae_agent_model:
           model_provider: openai
           model: gpt-4o
           max_tokens: 4096
           temperature: 0.5
           top_p: 1
           top_k: 0
           max_retries: 10
           parallel_tool_calls: true
       lakeview_model:
           model_provider: openai
           model: gpt-4o
           max_tokens: 4096
           temperature: 0.5
           top_p: 1
           top_k: 0
           max_retries: 10
           parallel_tool_calls: true
   ```

4. **Update bench.yaml paths:**
   
   Edit `perf-agents-bench/bench.yaml`:
   ```yaml
   agents:
     default: "trae"
     trae:
       cli: "${TRAE_PYTHON:-/home/ubuntu/OmniPerf-Bench/bench-env/bin/python}"
       args:
         max_steps: 120
       time_budget_minutes: 120
       config_file: "${TRAE_CONFIG:-/home/ubuntu/OmniPerf-Bench/third-party/trae-agent/trae_config.yaml}"
   ```

### Codex Agent Integration (no web/doc search)

Codex mirrors the TRAE wiring but enforces a strict offline policy—no MCP servers,
web-search, or document lookup tools are allowed.

1. **Install the Codex CLI dependencies (same venv as TRAE):**
   ```bash
   source bench-env/bin/activate
   uv pip install -e third-party/trae-agent  # reused runtime
   ```

2. **Configure Codex:**

   Edit `codex_agent/codex_config.yaml` (or create your own and set `CODEX_CONFIG`):
   ```yaml
   model_providers:
     openai:
       provider: openai
       api_key: ${OPENAI_API_KEY}
   models:
     codex_agent_model:
       model_provider: openai
       model: gpt-4o
       max_tokens: 8192
       temperature: 0.4
       parallel_tool_calls: true
   allow_mcp_servers: []
   mcp_servers: {}
   agents:
     trae_agent:
       enable_lakeview: false
       model: codex_agent_model
       max_steps: 120
       tools:
         - bash
         - str_replace_based_edit_tool
         - sequentialthinking
         - task_done
   ```
   > The CLI validates that no tool names include `search`, `lookup`, or `doc`
   > keywords and that `allow_mcp_servers` remains empty, guaranteeing Codex will
   > not attempt any web/document lookups.

3. **Update bench.yaml paths:**
   ```yaml
   agents:
     codex:
       cli: "${CODEX_PYTHON:-/home/ubuntu/OmniPerf-Bench/bench-env/bin/python}"
       args:
         max_steps: 120
       time_budget_minutes: 120
       config_file: "${CODEX_CONFIG:-/home/ubuntu/OmniPerf-Bench/codex_agent/codex_config.yaml}"
   ```

4. **Run prepare with Codex (set `agents.default: "codex"` in `bench.yaml` first):**
   ```bash
   CODEX_PYTHON=bench-env/bin/python \
   CODEX_CONFIG=$PWD/codex_agent/codex_config.yaml \
   python -m bench.cli prepare \
       tasks/vllm.yaml \
       --from-plan state/plan_remaining.json \
       --bench-cfg bench.yaml \
       --max-workers 1 \
       --resume
   ```

### Running the Codex CLI (local, no web/doc search)

We also support the official Codex CLI as a drop-in agent. This route is useful when you already have Codex installed locally, a valid profile with billing enabled, and you want the bench harness to invoke it for every commit without any TRAE plumbing.

1. **Install & authenticate Codex CLI**
   ```bash
   codex login  # configure your account/profile (e.g., kernel-bot)
   ```
   The bench run uses a dedicated home directory at `perf-agents-bench/.codex_home`.
   Copy your Codex state there so non-interactive runs inherit your profile:
   ```bash
   mkdir -p perf-agents-bench/.codex_home
   rsync -a ~/.codex/ perf-agents-bench/.codex_home/.codex/
   ```

2. **Build the plan from the 99 vLLM commit JSONs** (only needed once):
   ```bash
   cd /home/raven/coding-mess/kernel-corp/OmniPerf-Bench
   python - <<'PY'
   import json
   from pathlib import Path

   commit_dir = Path("hf_cache/alpha-vllm-99-commits/vllm_commits_separated")
   items = []
   for idx, path in enumerate(sorted(commit_dir.glob("*.json")), 1):
       commit_hash = json.loads(path.read_text()).get("commit_hash") or path.stem
       items.append({
           "item_id": f"vllm_core-{idx:04d}",
           "human": commit_hash,
           "pre": "",
           "pre_parent_index": 1,
       })
   plan = {
       "repo": str((Path("vllm")).resolve()),
       "task_id": "vllm_core",
       "items": items,
   }
   out = Path("perf-agents-bench/state/plan_codex_full.json")
   out.write_text(json.dumps(plan, indent=2))
   print(f"Wrote {out} with {len(items)} commits")
   PY
   ```

3. **Run the bench harness with Codex CLI** (one commit at a time to keep logs readable):
   ```bash
   cd perf-agents-bench
   source ../bench-env/bin/activate
   export CODEX_CLI=${CODEX_CLI:-codex}          # path to codex binary
   export CODEX_PROFILE=${CODEX_PROFILE:-kernel-bot}  # Codex profile to use
   python -m bench.cli prepare \
       tasks/vllm.yaml \
       --from-plan state/plan_codex_full.json \
       --bench-cfg bench_codex.yaml \
       --max-workers 1 \
       --resume
   ```

   > `bench_codex.yaml` already sets `agents.default: "codex_cli"`. The harness
   > launches `codex exec --cd <worktree> --sandbox danger-full-access -p $CODEX_PROFILE`
   > for each commit, enforces our `.bench_scratch` policy, and writes all logs
   > under `perf-agents-bench/state/runs/<run_id>/<item_id>/`.

### Understanding the Resume Logic

**Current Limitation:** The built-in `--resume` flag only works within a single run session. Each time you run `prepare`, it creates a new `run_id` (e.g., `vllm_core-abc12345`), and resume only checks that specific directory.

**Solution:** Pre-filter the plan to exclude already-completed commits. The audit script above:
1. Scans ALL previous run directories in `state/runs/`
2. Extracts commit hashes from successful journals
3. Creates a filtered plan containing only unprocessed commits

This prevents duplicate work and wasted resources.

### Monitoring Progress

**Check current status:**
```bash
cd perf-agents-bench

# Count successes and errors
grep -c "Task status determined as: success" pipeline_run_*.log
grep -c "Task status determined as: error" pipeline_run_*.log

# View recent activity
tail -50 pipeline_run_*.log | grep -E "(Starting task|status determined|TRAE STDOUT)"

# Monitor live (if running in tmux)
tmux attach -t trae_pipeline
# Detach without stopping: Ctrl+B then D
```

### Expected Behavior

**For 60 remaining commits:**
- **Time:** ~15 hours (15 min/commit average)
- **Tokens:** ~60M tokens total
- **Cost:** $150-300 (GPT-4o pricing)
- **Output:** Individual journals in `state/runs/{run_id}/{item_id}/`

**Success indicators:**
- Real-time TRAE output showing code edits
- Token usage displayed (e.g., "Input: 332283 Output: 2188")
- Journal files with `"status": "success"`
- `model_patch.diff` files generated

**Common issues:**
- Missing dependencies → Run `uv pip install -e third-party/trae-agent`
- Config file not found → Check `TRAE_CONFIG` environment variable
- API key errors → Verify `OPENAI_API_KEY` is set
- Anthropic import errors → Ensure config uses OpenAI, not Anthropic

### Resume vs. Fresh Run

**Resume (--resume flag):**
- Skips items already completed in CURRENT run
- Only helps if restarting interrupted session
- Does NOT check other run directories

**Filtered Plan (recommended):**
- Pre-filters plan before running
- Checks ALL previous runs across all directories
- Prevents duplicate work across sessions
- More robust for incremental processing

**Best Practice:** Always create filtered plan before running, then use `--resume` as safety net.

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
