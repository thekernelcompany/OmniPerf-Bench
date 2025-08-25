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
This repository has been reorganized for better maintainability. See [docs/repository_structure.md](docs/repository_structure.md) for a detailed overview of the new structure. Key directories:

- `src/` - Main source code (collection framework, harness, data models)
- `experiments/` - Experiment configurations and datasets (vLLM, SGLang)
- `benchmarks/effibench/` - EffiBench integration for code efficiency evaluation
- `tools/` - Utility scripts and patches
- `results/` - Logs, reviews, and analysis outputs
- `misc/` - Archived files and legacy content (with results for test generation files)

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

uv run src/gso/harness/prepare_images.py \
    --push_to_registry true \
    --dockerhub_username <dockerhub_username> \
    --dockerhub_repo <dockerhub_repo>
```

2. **running evaluations**:
```bash
uv run src/gso/harness/opt_at_k.py \
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
EffiBench is integrated as a benchmark for evaluating code efficiency. To use EffiBench:

```bash
cd benchmarks/effibench
pip install -r requirements.txt
# Follow EffiBench README for specific usage
```

See [benchmarks/effibench/README.md](benchmarks/effibench/README.md) for detailed EffiBench documentation.

### experiments

Pre-configured experiments are available in the `experiments/` directory:
- `experiments/vllm/` - vLLM performance optimization dataset with 282 problems
- `experiments/sglang.yaml` - SGLang experiment configuration
- `experiments/vllm.yaml` - vLLM experiment configuration

## ⬇️ artifacts
| datasets | tools | dockers |
| - | - | - |
| [💿 gso](https://huggingface.co/datasets/gso-bench/gso) | [🔧 evaluation harness](src/harness/) | [🐳 docker hub](https://hub.docker.com/repository/docker/slimshetty/gso/general) |
| [💿 vllm dataset](experiments/vllm/) | [🔧 collection framework](src/collect/) | |
| [💿 effibench](benchmarks/effibench/) | [🔧 utility tools](tools/) | |
