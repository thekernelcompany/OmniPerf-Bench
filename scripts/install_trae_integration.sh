#!/bin/bash
# TRAE Agent Integration Installation Script
# This script sets up the complete TRAE agent integration for OmniPerf-Bench

set -e

echo "🚀 Installing TRAE Agent Integration for OmniPerf-Bench..."
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "perf-agents-bench/bench_test.yaml" ]; then
    print_error "Please run this script from the OmniPerf-Bench root directory"
    exit 1
fi

print_status "Checking prerequisites..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.12"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
    print_error "Python 3.12+ required. Found: $PYTHON_VERSION"
    exit 1
fi
print_success "Python version: $PYTHON_VERSION"

# Check/install uv
if ! command -v uv &> /dev/null; then
    print_status "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=$HOME/.local/bin:$PATH
    print_success "uv installed successfully"
else
    print_success "uv already available: $(uv --version)"
fi

# Setup virtual environment
print_status "Setting up virtual environment..."
if [ ! -d "bench-env" ]; then
    uv venv bench-env
    print_success "Created bench-env virtual environment"
else
    print_success "bench-env virtual environment already exists"
fi

# Activate environment
source bench-env/bin/activate
export PATH=$HOME/.local/bin:$PATH

# Install base dependencies
print_status "Installing base dependencies..."
if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
    print_success "Base dependencies installed"
else
    print_warning "requirements.txt not found, skipping base dependencies"
fi

# Install TRAE agent
print_status "Installing TRAE agent..."
if [ ! -d "third-party/trae-agent" ] || [ -z "$(ls -A third-party/trae-agent 2>/dev/null)" ]; then
    print_status "Cloning trae-agent repository..."
    mkdir -p third-party
    rm -rf third-party/trae-agent
    git clone https://github.com/agokrani/trae-agent.git third-party/trae-agent
    print_success "trae-agent cloned successfully"
fi
uv pip install -e third-party/trae-agent
print_success "TRAE agent installed in development mode"

# Install critical tree-sitter dependencies
print_status "Installing tree-sitter dependencies..."
uv pip install tree-sitter==0.24.0 tree-sitter-languages==1.10.2
print_success "Tree-sitter dependencies installed"

# Check system dependencies
print_status "Checking system dependencies..."

# Check tmux
if command -v tmux &> /dev/null; then
    print_success "tmux available: $(tmux -V)"
else
    print_warning "tmux not found. Install with: sudo apt-get install tmux"
fi

# Check playwright
if python -c "import playwright" 2>/dev/null; then
    print_success "playwright available"
else
    print_status "Installing playwright..."
    python -m pip install playwright
    python -m playwright install --with-deps
    print_success "playwright installed"
fi

# Verify TRAE installation
print_status "Verifying TRAE installation..."
if python -c "import trae_agent; print('TRAE agent version:', trae_agent.__version__)" 2>/dev/null; then
    print_success "TRAE agent import successful"
else
    print_error "TRAE agent import failed"
    exit 1
fi

# Check configuration files
print_status "Checking configuration files..."

# Check TRAE config
TRAE_CONFIG="third-party/trae-agent/trae_config.yaml"
if [ -f "$TRAE_CONFIG" ]; then
    print_success "TRAE config found: $TRAE_CONFIG"
else
    print_warning "TRAE config not found at $TRAE_CONFIG"
    print_status "Creating default TRAE config..."
    
    cat > "$TRAE_CONFIG" << 'EOF'
model_providers:
  openai:
    provider: openai
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"

models:
  trae_agent_model:
    model_provider: openai
    model: gpt-5-2025-08-07
    max_tokens: 32000
    temperature: 0.5
    top_p: 1
    top_k: 0
    parallel_tool_calls: false
    max_retries: 3

lakeview:
  model: trae_agent_model

agents:
  trae_agent:
    enable_lakeview: true
    model: trae_agent_model
    max_steps: 50
    tools:
      - bash
      - str_replace_based_edit_tool
      - sequentialthinking
      - task_done
EOF
    print_success "Created default TRAE config"
fi

# Check Codex config
CODEX_CONFIG="codex_agent/codex_config.yaml"
if [ -f "$CODEX_CONFIG" ]; then
    if grep -qi "search" "$CODEX_CONFIG" || grep -qi "lookup" "$CODEX_CONFIG"; then
        print_warning "Codex config references search/doc lookup tools. Remove them to enforce offline policy."
    else
        print_success "Codex config found (offline mode enforced): $CODEX_CONFIG"
    fi
else
    print_warning "Codex config not found at $CODEX_CONFIG"
fi

# Check bench config
BENCH_CONFIG="perf-agents-bench/bench_test.yaml"
if [ -f "$BENCH_CONFIG" ]; then
    # Check if config file path is correct
    CURRENT_PATH=$(pwd)
    EXPECTED_CONFIG_PATH="$CURRENT_PATH/third-party/trae-agent/trae_config.yaml"
    EXPECTED_CODEX_CONFIG_PATH="$CURRENT_PATH/codex_agent/codex_config.yaml"
    
    if grep -q "$EXPECTED_CONFIG_PATH" "$BENCH_CONFIG"; then
        print_success "Bench config has correct TRAE config path"
    else
        print_warning "Bench config may have incorrect TRAE config path"
        print_status "Expected path: $EXPECTED_CONFIG_PATH"
        print_status "Please verify the TRAE config_file path in $BENCH_CONFIG"
    fi

    if grep -q "$EXPECTED_CODEX_CONFIG_PATH" "$BENCH_CONFIG"; then
        print_success "Bench config has correct Codex config path"
    else
        print_warning "Bench config may have incorrect Codex config path"
        print_status "Expected path: $EXPECTED_CODEX_CONFIG_PATH"
        print_status "Please verify the Codex config_file path in $BENCH_CONFIG"
    fi
else
    print_error "Bench config not found: $BENCH_CONFIG"
    exit 1
fi

# Check API key setup
print_status "Checking API key configuration..."
if [ -n "$OPENAI_API_KEY" ]; then
    print_success "OPENAI_API_KEY is set in environment"
elif [ -f ".env" ] && grep -q "OPENAI_API_KEY" .env; then
    print_success "OPENAI_API_KEY found in .env file"
else
    print_warning "OPENAI_API_KEY not found in environment or .env file"
    print_status "Please set your OpenAI API key:"
    print_status "  export OPENAI_API_KEY='your-api-key-here'"
    print_status "  or create a .env file with OPENAI_API_KEY=your-api-key-here"
fi

# Run a quick test
print_status "Running integration test..."
cd perf-agents-bench
export PYTHONPATH=$(pwd):$PYTHONPATH

if python -m bench.cli doctor --bench-cfg bench_test.yaml >/dev/null 2>&1; then
    print_success "Integration test passed"
else
    print_warning "Integration test had issues - this may be normal if no tasks are configured"
fi

echo
print_success "🎉 TRAE Agent Integration installation completed!"
echo
echo "Next steps:"
echo "1. Set your OpenAI API key if not already done:"
echo "   export OPENAI_API_KEY='your-api-key-here'"
echo
echo "2. Test the integration:"
echo "   cd perf-agents-bench"
echo "   source ../bench-env/bin/activate"
echo "   export PYTHONPATH=\$(pwd):\$PYTHONPATH"
echo "   python -m bench.cli prepare tasks/your_task.yaml --from-plan ./state/your_plan.json --bench-cfg bench_test.yaml --max-workers 1"
echo
echo "3. See TRAE_AGENT_REPLICATION_GUIDE.md for detailed usage instructions"
echo
print_success "Installation complete! 🚀"
