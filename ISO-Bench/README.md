# ISO-Bench

Agent benchmarking harness for evaluating AI coding agents on performance optimization tasks.

## Overview

ISO-Bench provides a CLI workflow to:
- **Plan** benchmark runs from commit pairs
- **Prepare** agent execution environments and run agents
- **Report** results with soft metrics analysis

Supported agents: **TRAE**, **OpenHands**, **Codex**, **Claude Code**

## Requirements

- Python 3.12+
- uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker (for containerized execution)

## Quick Start

```bash
cd ISO-Bench

# 1. Create virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your credentials (LLM_API_KEY, etc.)

# 3. Plan benchmarks
python -m bench.cli plan tasks/vllm.yaml --out state/plan.json

# 4. Run agent benchmarks
python -m bench.cli prepare tasks/vllm.yaml \
    --from-plan state/plan.json \
    --bench-cfg bench.yaml \
    --max-workers 1 --resume

# 5. Generate report
python -m bench.cli report state/runs/<run_id>
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `plan` | Generate benchmark plan from task config |
| `prepare` | Execute agents on planned tasks |
| `report` | Summarize run results |
| `doctor` | Verify system configuration |
| `validate` | Validate task config file |
| `analyze` | Run soft metrics analysis |

## Directory Structure

```
ISO-Bench/
├── bench/          # Core source code (CLI, pipeline, agents)
├── tasks/          # Task configurations (vllm.yaml, sglang.yaml)
├── config/         # Agent-specific configs
├── state/          # Runtime state
│   ├── plan.json   # Current benchmark plan
│   └── runs/       # Agent run outputs
├── bench.yaml      # Main configuration
├── requirements.txt
└── ARCHITECTURE.md # Detailed architecture docs
```

## Configuration

Edit `bench.yaml` to configure:
- Container settings (Docker/Podman, resources)
- Agent paths and parameters
- Time budgets and step limits

Environment variables override defaults:
```bash
export TRAE_PYTHON="../bench-env/bin/python"
export TRAE_CONFIG="../third-party/trae-agent/trae_config.yaml"
```

## Troubleshooting

```bash
# Check system configuration
python -m bench.cli doctor --bench-cfg bench.yaml

# Validate task config
python -m bench.cli validate tasks/vllm.yaml

# View agent logs
cat state/runs/<run_id>/<task_id>/trae_stdout.txt
```

## Documentation

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture and implementation notes.
