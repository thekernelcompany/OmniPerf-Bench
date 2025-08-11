# OmniPerf-Bench

## 👋 overview
gso evaluates language models on software performance optimization. each task provides:
- a *codebase* with a specific performance bottleneck
- a *performance test* as a precise specification
- an agent must generate a *patch* that improves runtime efficiency
- success is measured against expert developer optimizations

to access gso, copy and run the following code:
```python
from datasets import load_dataset
gso = load_dataset('gso-bench/gso', split='test')
```

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

for detailed instructions and options, see the [harness documentation](src/gso/harness/readme.md).

### gso collection framework

the collection framework enables you to create your own gso tasks through a four-step pipeline:

1. **[commit extraction & filtering](src/gso/collect/readme.md#overview)**: extract performance-related commits using llms
2. **[api identification](src/gso/collect/readme.md#2-commit-analysis-pipeline)**: identify affected high-level apis for each commit
3. **[performance test generation](src/gso/collect/readme.md#3-generate-performance-tests)**: generate tests for api-commit pairs
4. **[test execution](src/gso/collect/readme.md#4-execute-performance-tests)**: execute tests to identify performance improvements

<!-- required tokens:
```bash
export ghapi_token="github_token"
export openai_api_key="openai_key"
export hf_token="huggingface_token"
``` -->

for detailed instructions and usage, see the [collection framework documentation](src/gso/collect/readme.md).


## ⬇️ artifacts
| datasets | tools | dockers |
| - | - | - |
| [💿 gso](https://huggingface.co/datasets/gso-bench/gso) | [🔧 evaluation harness](src/gso/harness/) | [🐳 docker hub](https://hub.docker.com/repository/docker/slimshetty/gso/general) |
| | [🔧 collection framework](src/gso/collect/readme.md) | |
