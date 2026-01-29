# OpenHands LLM Benchmarking Integration Strategy

## Executive Summary

This document outlines the technical implementation strategy for integrating OpenHands into the `commit_to_dataset.py` pipeline to enable systematic benchmarking of LLM optimization capabilities against human performance.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Configuration Schema](#configuration-schema)
4. [File Structure](#file-structure)
5. [Integration Points](#integration-points)
6. [Migration Plan](#migration-plan)
7. [Testing Strategy](#testing-strategy)
8. [Success Metrics](#success-metrics)
9. [Dependencies](#dependencies)
10. [Error Handling](#error-handling)
11. [Monitoring and Observability](#monitoring-and-observability)

## Architecture Overview

### Current System (Before Integration)
```
Human Commits → Dataset Creation → Canonical Records → HF Dataset
```

### Integrated System (After Implementation)
```
Human Commits → Dataset Creation → Canonical Records → HF Dataset
     ↓              ↓                     ↓
   LLM Tasks → OpenHands Execution → LLM Records → Comparative Analysis
```

### Key Architectural Changes

**Before (Complex Docker Approach):**
- Complex Docker container orchestration with volume mounting
- Manual subprocess management with fragile command construction
- Environment variable propagation through containers
- File-based task passing with complex mounting

**After (OpenHands-Native Approach):**
- Direct OpenHands CLI integration using their official patterns
- Proper configuration files (config.toml) following OpenHands standards
- Task format conversion using OpenHands expectations
- Clean subprocess execution with proper error handling

## Core Components

### 1. OpenHands CLI Integration Module

**File:** `ISO-Bench/bench/agents/openhands_cli.py`

```python
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import tempfile
import json

class OpenHandsCLI:
    """Proper OpenHands CLI integration following their documentation."""

    def __init__(self, config_path: Path, llm_config_path: Path):
        self.config_path = config_path
        self.llm_config_path = llm_config_path

    def run_task(self, workspace_dir: Path, task_description: str,
                 max_iterations: int = 30, timeout_minutes: int = 60) -> Dict[str, Any]:
        """Run OpenHands on a task using their CLI interface."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(task_description)
            task_file = f.name

        try:
            cmd = [
                "python", "-m", "openhands.core.main",
                "-d", str(workspace_dir),
                "-f", task_file,
                "-i", str(max_iterations),
                "-t", str(timeout_minutes),
                "-c", "CodeActAgent",
                "-l", str(self.llm_config_path),
                "--config", str(self.config_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_minutes*60)

            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "task_file": task_file
            }
        except subprocess.TimeoutExpired:
            return {"timeout": True, "task_file": task_file}
        finally:
            Path(task_file).unlink(missing_ok=True)
```

### 2. Task Format Converter

**File:** `ISO-Bench/bench/task_converter.py`

```python
class OpenHandsTaskConverter:
    """Convert ISO-Bench tasks to OpenHands format."""

    def convert_task(self, task_cfg: Dict[str, Any], human_commit: str,
                    pre_commit: str, target_files: list) -> str:
        """Convert task configuration to OpenHands task description."""

        sections = [
            "# Performance Optimization Task",
            "",
            f"## Task: {task_cfg['name']}",
            f"## Description: {task_cfg.get('description', '')}",
            "",
            "## Objective",
            "Optimize the performance of the specified files while maintaining functionality.",
            "",
            "## Target Files",
            "You should ONLY modify these files:",
        ]

        for file in target_files:
            sections.append(f"- `{file}`")

        if task_cfg["optimization_contract"].get("constraints"):
            sections.extend([
                "",
                "## Constraints",
                "You must follow these constraints:"
            ])
            for constraint in task_cfg["optimization_contract"]["constraints"]:
                sections.append(f"- {constraint}")

        sections.extend([
            "",
            "## Reference Information",
            f"The human developer optimized from commit {pre_commit} to {human_commit}.",
            "Your goal is to achieve similar or better optimizations independently.",
            "",
            "## Success Criteria",
            f"- Primary metric: {task_cfg['scoring']['primary']}",
            "- All existing tests must pass",
            "- No regression in functionality",
            "",
            "## Instructions",
            "1. Analyze the target files for performance bottlenecks",
            "2. Implement optimizations while respecting constraints",
            "3. Test your changes to ensure correctness",
            "4. Commit your optimizations"
        ])

        return "\n".join(sections)
```

### 3. Enhanced Data Structures

**File:** `commit_to_dataset.py` (extended)

```python
@dataclass
class LLMMetadata:
    provider: str
    model: str
    iterations: int
    cost_usd: float
    time_seconds: int
    success: bool
    human_reference_commit: str
    agent_branch: str

@dataclass
class CanonicalRecord:
    repo: str
    instance_id: str
    created_at: str
    base_commit: str
    head_commit: str
    patch: str
    test_patch: str
    efficiency_test: List[str]
    duration_changes: List[Dict[str, List[float]]]
    human_performance: Optional[float] = None
    patch_functions: Optional[List[str]] = None
    test_functions: List[str] = None
    api: Optional[str] = None
    gt_commit_message: Optional[str] = None
    setup_commands: List[str] = None
    install_commands: List[str] = None
    notes: Optional[str] = None
    # New LLM fields
    record_type: str = "human"  # "human" or "llm"
    llm_metadata: Optional[LLMMetadata] = None
```

### 4. LLM Configuration Management

**File:** `ISO-Bench/bench/config/llm_manager.py`

```python
class LLMConfigManager:
    """Manage LLM provider configurations for OpenHands."""

    def setup_llm_config(self, env_vars: Dict[str, str]) -> Path:
        """Generate LLM config.toml for OpenHands."""
        config = f"""
[llm]
model = "{env_vars.get('LLM_MODEL', 'anthropic/claude-sonnet-4-20250514')}"
api_key = "{env_vars.get('LLM_API_KEY', '')}"
base_url = "{env_vars.get('LLM_BASE_URL', 'https://api.anthropic.com')}"
custom_llm_provider = "{self._get_provider(env_vars.get('LLM_MODEL', ''))}"
max_input_tokens = 0
max_output_tokens = 0
input_cost_per_token = 0.0
output_cost_per_token = 0.0
ollama_base_url = ""
drop_params = false
"""
        config_path = Path("config/llm_config.toml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config)
        return config_path

    def setup_agent_config(self) -> Path:
        """Generate agent config.toml for OpenHands."""
        config = """
[agent]
micro_agent = false
llm_config = "config/llm_config.toml"
confirmation_mode = false
security_analyzer = ""
max_iterations = 30
max_budget_per_task = 10.0
"""
        config_path = Path("config/agent_config.toml")
        config_path.write_text(config)
        return config_path
```

### 5. Dual Pipeline Orchestrator

**File:** `commit_to_dataset.py` (new function)

```python
def process_batch_commits_with_llm(
    repo_path: str,
    extraction_files: List[Tuple[str, str]],
    bench_cfg: Dict[str, Any],
    llm_providers: List[str],
    **kwargs
) -> List[CanonicalRecord]:
    """Process commits with both human analysis and LLM optimization."""

    records = []

    for head_commit, base_commit in extraction_files:
        # 1. Process human commit (existing logic)
        human_record = assemble_canonical(
            repo_path_arg=repo_path,
            head_commit=head_commit,
            base_commit=base_commit,
            **kwargs
        )
        records.append(human_record)

        # 2. For each LLM provider, run OpenHands optimization
        for llm_provider in llm_providers:
            try:
                llm_record = assemble_llm_optimization(
                    repo_path=repo_path,
                    human_commit=head_commit,
                    base_commit=base_commit,
                    bench_cfg=bench_cfg,
                    llm_provider=llm_provider,
                    human_record=human_record
                )
                records.append(llm_record)
            except Exception as e:
                logger.error(f"LLM optimization failed for {llm_provider}: {e}")
                # Still include failed attempt for analysis

    return records

def assemble_llm_optimization(
    repo_path: str,
    human_commit: str,
    base_commit: str,
    bench_cfg: Dict[str, Any],
    llm_provider: str,
    human_record: CanonicalRecord
) -> CanonicalRecord:
    """Run OpenHands optimization using specified LLM."""

    # Generate task config from human optimization
    task_cfg = generate_task_from_human_commit(human_record)

    # Use OpenHands prepare pipeline
    from perf_agents_bench.bench.cli import plan, prepare

    # Create temporary task file
    task_file = create_temp_task_file(task_cfg)

    # Plan the optimization task
    plan_result = plan(task=str(task_file), commits=f"{base_commit} {human_commit}")

    # Prepare (run OpenHands)
    prepare_result = prepare(
        task=str(task_file),
        from_plan=plan_result["plan_path"],
        bench_cfg=bench_cfg,
        max_workers=1
    )

    # Extract LLM optimization results
    return create_llm_record_from_results(
        human_record=human_record,
        llm_provider=llm_provider,
        prepare_result=prepare_result
    )
```

## Configuration Schema

### Enhanced bench.yaml

```yaml
# ISO-Bench/bench.yaml - Enhanced for LLM benchmarking
container:
  engine: "docker"
  cpus: 2
  memory: "4g"
  network_policy: "off"
  gpus: "none"

paths:
  work_root: "./.work"
  state_root: "./state"
  reports_root: "./reports"

agents:
  default: "openhands"
  openhands:
    cli: "python -m openhands.core.main"  # Use direct CLI
    time_budget_minutes: 60
    max_iterations: 30
    max_budget_per_task: 10.0
    config_file: "config/agent_config.toml"
    llm_configs:
      - provider: "anthropic"
        model: "claude-sonnet-4-20250514"
        config_file: "config/llm_anthropic.toml"
      - provider: "openai"
        model: "gpt-4-turbo"
        config_file: "config/llm_openai.toml"

# New section for LLM benchmarking
llm_benchmarking:
  enabled: true
  providers: ["anthropic", "openai"]
  max_parallel_runs: 2
  cost_tracking: true
  result_correlation: true
  comparative_analysis: true
```

### LLM Configuration Files

**config/llm_anthropic.toml:**
```toml
[llm]
model = "anthropic/claude-sonnet-4-20250514"
api_key = "${ANTHROPIC_API_KEY}"
base_url = "https://api.anthropic.com"
custom_llm_provider = "anthropic"
max_input_tokens = 0
max_output_tokens = 0
input_cost_per_token = 0.0
output_cost_per_token = 0.0
ollama_base_url = ""
drop_params = false
```

**config/llm_openai.toml:**
```toml
[llm]
model = "gpt-4-turbo"
api_key = "${OPENAI_API_KEY}"
base_url = "https://api.openai.com/v1"
custom_llm_provider = "openai"
max_input_tokens = 0
max_output_tokens = 0
input_cost_per_token = 0.0
output_cost_per_token = 0.0
ollama_base_url = ""
drop_params = false
```

## File Structure

```
ISO-Bench/
├── bench/
│   ├── agents/
│   │   ├── openhands_cli.py          # New: OpenHands CLI integration
│   │   └── orchestrator.py           # New: Agent orchestration layer
│   ├── config/
│   │   ├── llm_manager.py           # New: LLM configuration management
│   │   └── agent_config.toml       # New: Agent configuration
│   ├── task_converter.py           # New: Task format converter
│   ├── prepare.py                  # Modified: Simplified OpenHands execution
│   └── cli.py                      # Modified: Enhanced CLI commands
├── config/
│   ├── llm_anthropic.toml         # New: Anthropic LLM config
│   ├── llm_openai.toml            # New: OpenAI LLM config
│   └── agent_config.toml          # New: Agent config
├── bench.yaml                     # Modified: Enhanced configuration
└── tasks/
    └── vllm.yaml                  # Modified: Updated task format

commit_to_dataset.py               # Modified: Dual pipeline integration
```

## Integration Points

### 1. Enhanced CLI Commands

**New CLI command:**
```bash
# Run LLM benchmarking on human commits
python commit_to_dataset.py --enable-llm-benchmarking \
                           --llm-providers anthropic,openai \
                           --config commit_to_dataset_llm.yaml
```

### 2. Modified prepare.py Execution

**Before (complex Docker):**
```python
# Complex Docker container orchestration with volume mounting
cmd = ["docker", "run", "--rm", "-v", f"{wt_dir}:/workspace:rw", ...]
```

**After (simple CLI):**
```python
# Direct OpenHands CLI execution
cmd = ["python", "-m", "openhands.core.main", "-d", str(wt_dir), "-f", str(task_file)]
```

### 3. Result Correlation System

```python
# Link human and LLM results
correlation_map = {
    "human_commit": "f092153fbe349a9a1742940e3703bfcff6aa0a6d",
    "llm_attempts": {
        "anthropic": "abc123def456",
        "openai": "def456ghi789"
    },
    "comparison": {
        "anthropic_vs_human": 0.95,
        "openai_vs_human": 0.87
    }
}
```

## Migration Plan

### Phase 1: Foundation (Weeks 1-2)

1. **Create OpenHands integration files**
   ```bash
   mkdir -p ISO-Bench/bench/agents ISO-Bench/config
   touch ISO-Bench/bench/agents/openhands_cli.py
   touch ISO-Bench/config/llm_config.toml
   ```

2. **Implement basic OpenHands CLI wrapper**
3. **Test basic OpenHands execution**
4. **Create LLM configuration management**

### Phase 2: Integration (Weeks 3-4)

1. **Refactor prepare.py** to use new integration
2. **Update bench.yaml** configuration
3. **Implement task format converter**
4. **Test end-to-end OpenHands execution**

### Phase 3: Dual Pipeline (Weeks 5-6)

1. **Integrate with commit_to_dataset.py**
2. **Add LLM benchmarking configuration**
3. **Implement result correlation**
4. **Test comparative analysis**

## Testing Strategy

### Unit Tests

```python
# Test OpenHands CLI integration
def test_openhands_cli():
    cli = OpenHandsCLI(config_path, llm_config_path)
    result = cli.run_task(workspace_dir, task_description)
    assert result["returncode"] == 0

# Test task converter
def test_task_converter():
    converter = OpenHandsTaskConverter()
    task = converter.convert_task(task_cfg, human_commit, pre_commit, target_files)
    assert "# Performance Optimization Task" in task
    assert human_commit in task

# Test LLM configuration
def test_llm_config():
    manager = LLMConfigManager()
    config_path = manager.setup_llm_config(env_vars)
    assert config_path.exists()
```

### Integration Tests

```python
# Test full pipeline
def test_dual_pipeline():
    records = process_batch_commits_with_llm(
        repo_path, extraction_files, bench_cfg, ["anthropic"]
    )
    assert len(records) >= 2  # Human + LLM records
    assert any(r.record_type == "llm" for r in records)
```

### Benchmark Tests

```python
# Test performance comparison
def test_performance_comparison():
    human_record, llm_record = get_human_llm_pair()
    comparison = compare_optimizations(human_record, llm_record)
    assert "improvement_ratio" in comparison
    assert "cost_effectiveness" in comparison
```

## Success Metrics

- **Functional**: >95% commits successfully processed with LLM attempts
- **Quality**: LLM-generated code passes validation in >80% of cases
- **Performance**: <10% overhead vs human-only processing
- **Cost**: <$5 per commit for comprehensive LLM benchmarking
- **Reproducibility**: <5% variance in results across identical runs

## Dependencies

### New Dependencies

```txt
# requirements.txt additions
openhands-ai>=0.1.0
tomli>=2.0.0  # For TOML config parsing
```

### Environment Variables

```bash
# Required for LLM providers
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Optional: Custom endpoints
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4-turbo"
```

## Error Handling

### LLM-Specific Errors

```python
class LLMOptimizationError(Exception):
    """Base class for LLM optimization errors."""
    pass

class LLMTimeoutError(LLMOptimizationError):
    """Raised when LLM optimization times out."""
    pass

class LLMAPIError(LLMOptimizationError):
    """Raised when LLM API calls fail."""
    pass

class InvalidCodeError(LLMOptimizationError):
    """Raised when LLM generates invalid code."""
    pass
```

### Recovery Mechanisms

```python
def handle_llm_failure(llm_provider: str, error: Exception) -> Dict[str, Any]:
    """Handle LLM optimization failures gracefully."""
    if isinstance(error, LLMTimeoutError):
        return {"status": "timeout", "llm_provider": llm_provider}
    elif isinstance(error, LLMAPIError):
        return {"status": "api_error", "llm_provider": llm_provider}
    else:
        return {"status": "unknown_error", "llm_provider": llm_provider, "error": str(error)}
```

## Monitoring and Observability

### Cost Tracking

```python
class CostTracker:
    def __init__(self):
        self.costs = {}

    def track_cost(self, llm_provider: str, cost_usd: float):
        if llm_provider not in self.costs:
            self.costs[llm_provider] = []
        self.costs[llm_provider].append(cost_usd)

    def get_total_cost(self, llm_provider: str) -> float:
        return sum(self.costs.get(llm_provider, []))
```

### Performance Metrics

```python
class BenchmarkMetrics:
    def __init__(self):
        self.metrics = {}

    def record_attempt(self, llm_provider: str, success: bool, duration: float):
        if llm_provider not in self.metrics:
            self.metrics[llm_provider] = {"attempts": 0, "successes": 0, "total_time": 0}

        self.metrics[llm_provider]["attempts"] += 1
        if success:
            self.metrics[llm_provider]["successes"] += 1
        self.metrics[llm_provider]["total_time"] += duration

    def get_success_rate(self, llm_provider: str) -> float:
        if llm_provider not in self.metrics:
            return 0.0
        return self.metrics[llm_provider]["successes"] / self.metrics[llm_provider]["attempts"]
```

## Conclusion

This technical implementation strategy provides a comprehensive approach to integrating OpenHands into the commit_to_dataset.py pipeline for LLM benchmarking. The modular design ensures:

- **Maintainability**: Clean separation of concerns
- **Scalability**: Easy addition of new LLM providers
- **Reliability**: Robust error handling and recovery
- **Reproducibility**: Consistent execution environment
- **Observability**: Comprehensive monitoring and metrics

The integration enables systematic comparison of LLM optimization capabilities against human performance, providing valuable insights into AI coding capabilities.

---

*Document Version: 1.0*
*Last Updated: $(date)*
*Authors: Technical Implementation Team*
