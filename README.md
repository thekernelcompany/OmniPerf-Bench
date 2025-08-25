# OmniPerf-Bench

## 👋 overview
This evaluates language models on software performance optimization. each task provides:
- a *codebase* with a specific performance bottleneck
- a *performance test* as a precise specification
- an agent must generate a *patch* that improves runtime efficiency
- success is measured against expert developer optimizations

to access, copy and run the following code:
```python
from datasets import load_dataset
gso = load_dataset('gso-bench/gso', split='test')
```

## 📁 repository structure

```
OmniPerf-Bench/
├── README.md                    # Main project documentation
├── CLAUDE.md                    # Claude AI development guidance
├── LICENSE                      # Project license  
├── pyproject.toml              # Python project configuration
├── uv.lock                     # Lock file for uv package manager
├── requirements.txt            # Python dependencies
│
├── src/                        # Main OmniPerf-Bench source code
│   ├── collect/                # Collection framework for dataset generation
│   │   ├── analysis/           # Commit and API analysis modules
│   │   ├── execute/            # Test execution and evaluation
│   │   ├── generate/           # Performance test generation
│   │   ├── scripts/            # Collection utility scripts
│   │   └── build_dataset.py    # Main dataset building script
│   ├── data/                   # Data models and parsing utilities
│   ├── harness/                # Evaluation harness for performance testing
│   │   ├── environment/        # Docker environment management
│   │   ├── grading/            # Grading and metrics evaluation
│   │   ├── plot/               # Visualization and plotting tools
│   │   └── scripts/            # Harness utility scripts
│   ├── test_scripts/           # Test generation and analysis scripts
│   │   ├── commit_analyzer.py  # Commit analysis utilities
│   │   ├── performance_analyzer.py # Performance analysis tools
│   │   └── test_llm_generator.py   # LLM-based test generators
│   ├── utils/                  # General utility functions
│   ├── constants.py            # Project constants
│   └── logger.py              # Logging configuration
│
├── third-party/               # External dependencies
│   └── effibench/             # Original EffiBench repository clone
│       ├── src/               # EffiBench source code
│       ├── data/              # EffiBench datasets  
│       ├── prompts/           # LLM prompts for EffiBench
│       ├── results/           # EffiBench evaluation results
│       ├── requirements.txt   # EffiBench dependencies
│       └── README.md          # EffiBench documentation
│
├── tools/                     # Utility scripts and patches
│   ├── manual_review.py       # Manual review utilities
│   └── openrouter_patch.py    # OpenRouter API patches
│
└── misc/                      # Experiment outputs and results
    ├── experiments/           # Experimental data and legacy outputs
    │   ├── commit_extractions/ # Extracted commit data (64 JSON files)
    │   ├── generated_test_generators_v*/ # Test generator iterations
    │   ├── gso-duplicate/     # Duplicate GSO analysis implementation
    │   ├── vllm/              # vLLM experiment data and results
    │   │   ├── data/          # vLLM training data (parquet files)
    │   │   ├── divided/       # Split result files for parallel processing
    │   │   └── *.json         # vLLM experiment configurations and results
    │   ├── sglang.yaml        # SGLang experiment configuration
    │   ├── vllm.yaml          # vLLM experiment configuration
    │   └── README.md          # Documentation for experiments
    └── results/               # Analysis results, logs, and reviews
        ├── analysis/          # Performance analysis outputs
        ├── logs/              # Execution and system logs
        └── reviews/           # Manual review files (CSV format)
```

Key directories:
- `src/` - Main source code (collection framework, evaluation harness, data models, test scripts)
- `third-party/effibench/` - Original EffiBench repository clone (external dependency)
- `tools/` - Utility scripts and patches
- `misc/experiments/` - Experimental data including vLLM/SGLang configs and generated outputs
- `misc/results/` - Analysis results, execution logs, and manual reviews

## 🚀 setup

### Option 1: Using uv (recommended)
```bash
curl -lssf https://astral.sh/uv/install.sh | sh
source $home/.local/bin/env

git clone --recursive https://github.com/gso-bench/gso.git
cd gso && uv venv && source .venv/bin/activate
uv sync
```

### Option 2: Using pip
```bash
git clone --recursive https://github.com/gso-bench/gso.git
cd gso
python -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Setup

1. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY="your_openai_api_key"
GHAPI_TOKEN="your_github_token"
HF_TOKEN="your_huggingface_token"
```

2. Or export environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export GHAPI_TOKEN="your_github_token"
export HF_TOKEN="your_huggingface_token"
```

For token setup:
- [OpenAI API Key](https://platform.openai.com/api-keys)
- [GitHub Token](https://github.com/settings/tokens)
- [HuggingFace Token](https://huggingface.co/docs/hub/en/security-tokens)


## 💽 usage

### evaluation harness

1. **building dockers for gso tasks**:
```bash
docker login

uv run src/harness/prepare_images.py \
    --push_to_registry true \
    --dockerhub_username <dockerhub_username> \
    --dockerhub_repo <dockerhub_repo>
```

2. **running evaluations**:
```bash
uv run src/harness/opt_at_k.py \
    --prediction_paths <prediction_paths> \
    --timeout 3600 \
    --run_id <run_id> \
    --k 10 \
    --model <modelname>
```

for detailed instructions and options, see the [harness documentation](src/harness/README.md).

### gso collection framework

the collection framework enables you to create your own gso tasks through a four-step pipeline:

1. **[commit extraction & filtering](src/collect/README.md#overview)**: extract performance-related commits using llms
2. **[api identification](src/collect/README.md#2-commit-analysis-pipeline)**: identify affected high-level apis for each commit
3. **[performance test generation](src/collect/README.md#3-generate-performance-tests)**: generate tests for api-commit pairs
4. **[test execution](src/collect/README.md#4-execute-performance-tests)**: execute tests to identify performance improvements

<!-- required tokens:
```bash
export ghapi_token="github_token"
export openai_api_key="openai_key"
export hf_token="huggingface_token"
``` -->

for detailed instructions and usage, see the [collection framework documentation](src/collect/README.md).

### benchmarks

#### effibench integration
EffiBench is integrated as a benchmark for evaluating code efficiency. The original EffiBench repository is in `third-party/effibench/` and analysis/test scripts are available in `src/test_scripts/`:

```bash
# To use the original EffiBench
cd third-party/effibench
pip install -r requirements.txt

# To use EffiBench-related scripts from our integration
cd src/test_scripts
python test_llm_generator.py  # LLM-based test generation
python performance_analyzer.py  # Performance analysis
```

See [third-party/effibench/README.md](third-party/effibench/README.md) for detailed EffiBench documentation and [src/test_scripts/README.md](src/test_scripts/README.md) for script usage.

### experiments

Pre-configured experiments are available in the `misc/experiments/` directory:
- `misc/experiments/vllm/` - vLLM performance optimization dataset with 282 problems  
- `misc/experiments/sglang.yaml` - SGLang experiment configuration
- `misc/experiments/vllm.yaml` - vLLM experiment configuration
- `misc/experiments/commit_extractions/` - Extracted performance-related commits (64 JSON files)
- `misc/experiments/generated_test_generators_v*/` - Various iterations of LLM test generators

## ⬇️ artifacts
| datasets | tools | dockers |
| - | - | - |
| [💿 gso](https://huggingface.co/datasets/gso-bench/gso) | [🔧 evaluation harness](src/harness/) | [🐳 docker hub](https://hub.docker.com/repository/docker/slimshetty/gso/general) |
| [💿 vllm dataset](misc/experiments/vllm/) | [🔧 collection framework](src/collect/) | |
| [💿 effibench](third-party/effibench/) | [🔧 test scripts](src/test_scripts/) | |
| | [🔧 utility tools](tools/) | |
