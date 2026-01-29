# OpenHands Integration Improvements

## Changes Made

### 1. Fixed OpenHands Execution Command (`bench/prepare.py`)
- **Before**: Used incorrect `uvx` subcommand structure that didn't align with OpenHands headless mode
- **After**: Proper headless mode invocation with `python -m openhands.core.main` and correct arguments

### 2. Enhanced Security with Containerization
- Added `SANDBOX_USER_ID` to prevent root-owned files
- Added `SANDBOX_RUNTIME_CONTAINER_IMAGE` for proper runtime isolation
- Mounted `~/.openhands` for configuration persistence
- Added proper Docker socket mounting for nested containers

### 3. Improved Task File Generation
- **Before**: Simple text format with minimal context
- **After**: Structured markdown format with:
  - Clear objectives and instructions
  - Specific optimization techniques
  - Reference information from human commits
  - Success criteria and final steps
  - Better formatting for readability

### 4. Proper Environment Configuration
- Added support for multiple LLM providers (OpenAI, Anthropic, etc.)
- Added `LLM_BASE_URL` for custom endpoints
- Added Git provider tokens (GitHub, GitLab, Bitbucket)
- Added timeout configuration
- Added budget limits for API calls

### 5. Updated `bench.yaml` Configuration
- Removed incorrect `subcommand` array
- Added `runtime_image` configuration
- Added `max_budget_per_task` for cost control
- Set proper default iterations (30)

## Key Improvements

### Security
- ✅ Runs in Docker container by default (when configured)
- ✅ User ID matching to prevent permission issues
- ✅ Isolated workspace with volume mounts
- ✅ No direct host filesystem access

### Reliability
- ✅ Proper command-line arguments for headless mode
- ✅ Better error handling with return codes
- ✅ Comprehensive logging with `LOG_ALL_EVENTS=true`
- ✅ Timeout and budget controls

### Context & Guidance
- ✅ Structured task format for better agent understanding
- ✅ Clear success criteria
- ✅ Reference information from human optimizations
- ✅ Specific optimization techniques listed

## Usage

### With Docker (Recommended)
```bash
# Set in bench.yaml or environment
export OPENHANDS_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/openhands:0.54
export OPENHANDS_RUNTIME_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik

# Run prepare
.venv/bin/python -m bench.cli prepare tasks/vllm.yaml \
  --from-plan ./state/plan.json \
  --bench-cfg bench.yaml
```

### Without Docker (Using uvx)
```bash
# OpenHands will be installed via uvx automatically
export OPENHANDS_CLI=uvx

# Run prepare
.venv/bin/python -m bench.cli prepare tasks/vllm.yaml \
  --from-plan ./state/plan.json \
  --bench-cfg bench.yaml
```

## Required Environment Variables

Create or update `.env` file:
```env
# LLM Configuration (required)
LLM_MODEL=anthropic/claude-sonnet-4-20250514
LLM_API_KEY=your-api-key

# Optional: Custom LLM endpoint
LLM_BASE_URL=https://api.anthropic.com

# Optional: Git provider tokens for repository access
GITHUB_TOKEN=your-github-token
GITLAB_TOKEN=your-gitlab-token
BITBUCKET_TOKEN=your-bitbucket-token

# Optional: Override container images
OPENHANDS_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/openhands:0.54
OPENHANDS_RUNTIME_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik
```

## Verification

To verify the improvements work correctly:

1. **Check Docker execution**:
   ```bash
   # Should see proper Docker run command with all environment variables
   .venv/bin/python -m bench.cli prepare --help
   ```

2. **Inspect generated task files**:
   ```bash
   cat state/runs/*/*/task.txt
   ```
   Should show well-formatted markdown with clear structure.

3. **Monitor OpenHands logs**:
   ```bash
   tail -f state/runs/*/*/openhands_stdout.txt
   ```
   Should show detailed execution logs with `LOG_ALL_EVENTS=true`

## Remaining Considerations

1. **Network Policy**: Currently allows network access during optimization. Consider restricting with `--network=none` for pure code optimization tasks.

2. **Resource Limits**: Add CPU and memory limits to prevent resource exhaustion:
   ```python
   cmd += ["--cpus", "2", "--memory", "4g"]
   ```

3. **Validation**: Consider adding pre-flight checks to ensure:
   - Docker daemon is running
   - Required images are available
   - LLM credentials are valid

4. **Monitoring**: Add real-time progress tracking via OpenHands event stream if needed.