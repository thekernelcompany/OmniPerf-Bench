# Experiments

This directory contains experimental outputs and legacy implementation work from various performance analysis experiments. It includes commit extractions, test generators, and analysis results that have been preserved for historical reference and research purposes.

## Overview

EffiBench Legacy is a performance benchmarking system that analyzes Git commits for performance-related changes and generates test cases using Large Language Models (LLMs). The system focuses on extracting performance optimizations from real-world codebases, particularly targeting projects like vLLM.

## Directory Structure

### Core Components

- **`llm_test_generator.py`** - Main LLM-powered test case generator that reads commit extractions and generates test cases
- **`performance_analysis_results.json`** - Aggregated performance analysis results from commit processing
- **`extraction_log.txt`** - Detailed logs from the commit extraction process

### Data Directories

#### `commit_extractions/`
Contains 63 JSON files with extracted commit data including:
- Commit metadata (hash, author, date, message)
- File changes and diffs
- Performance analysis flags
- Optimization categorization

#### `commit_extractions_with_apis/`
Enhanced version of commit extractions that includes API analysis:
- All data from `commit_extractions/`
- Additional API usage patterns
- Function call analysis
- Performance-related API identification

#### `generated_test_generators_v2/`, `v3/`, `v4/`
Progressive iterations of LLM-generated test case generators:
- **v2**: Initial test generator implementations
- **v3**: Improved prompt engineering and test structure
- **v4**: Latest version with enhanced test coverage and performance focus

Each version contains:
- Python test generator scripts (e.g., `2a052011_test_case_generator.py`)
- `index.json` - Metadata about generated test cases

### GSO Duplicate System

#### `gso-duplicate/`
A duplicate detection and analysis system with two main modules:

**Analysis Module (`analysis/`)**:
- `commits.py` - Commit analysis and processing
- `apis.py` - API usage pattern analysis  
- `parser.py` - Code parsing utilities
- `prompt.py` - LLM prompt management
- `retriever.py` - Data retrieval utilities
- `utils.py` - Common utility functions

**Generation Module (`generate/`)**:
- `generate.py` - Main generation logic
- `harness.py` - Test harness management
- `helpers.py` - Helper utilities
- `oversample.py` - Data oversampling techniques
- `prompt.py` - Generation-specific prompts
- `context.py` - Context management
- `args.py` - Argument parsing

### Configuration and Assets

- **`images/example.png`** - Visual documentation and examples
- **`.env`** - Environment configuration (contains API keys - not tracked in main repo)
- **`.gitattributes`** - Git attributes for the legacy repository
- **`.github/`** - GitHub issue templates and workflows from the original repository

## Key Features

### 1. Commit Analysis
- Extracts performance-related commits from Git repositories
- Analyzes code changes for optimization patterns
- Categorizes optimizations by type (kernel-based, memory, algorithmic, etc.)
- Identifies performance metrics and expected improvements

### 2. LLM-Powered Test Generation
- Uses multiple LLM providers (OpenAI, Anthropic, open-source models)
- Generates comprehensive test cases for performance optimizations
- Creates both unit tests and integration tests
- Includes performance benchmarking code

### 3. API Analysis
- Identifies performance-critical API usage patterns
- Maps function calls to performance implications
- Tracks API evolution across commits

### 4. Progressive Enhancement
- Multiple generations of test generators showing iterative improvement
- Enhanced prompt engineering over time
- Better test coverage and performance focus in newer versions

## Usage Examples

### Running the LLM Test Generator
```bash
# Generate test cases from commit extractions
python llm_test_generator.py --input-dir commit_extractions/ --output-dir generated_tests/

# Use specific LLM provider
python llm_test_generator.py --provider openai --model gpt-4

# Limit number of commits to process
python llm_test_generator.py --limit 5
```

### Analyzing Performance Results
```python
import json

# Load performance analysis results
with open('performance_analysis_results.json', 'r') as f:
    results = json.load(f)

# Analyze optimization patterns
for commit_hash, analysis in results.items():
    if analysis.get('has_performance_intent'):
        print(f"Commit {commit_hash}: {analysis['optimization_type']}")
        print(f"Expected speedup: {analysis.get('expected_speedup', 'N/A')}")
```

## Data Flow

1. **Commit Extraction**: Git commits are analyzed and extracted into JSON format
2. **Performance Analysis**: Commits are classified for performance intent and optimization type
3. **API Enhancement**: API usage patterns are analyzed and added to extractions
4. **Test Generation**: LLM generates test cases based on commit analysis
5. **Iterative Improvement**: Generated tests are refined across versions (v2 → v3 → v4)

## Integration with OmniPerf-Bench

This legacy system provides foundational research and methodology that informs the current OmniPerf-Bench architecture:

- **Commit Analysis Techniques**: Methods for identifying performance-relevant changes
- **LLM Integration Patterns**: Approaches for using LLMs in performance testing
- **Test Generation Strategies**: Frameworks for creating comprehensive performance tests
- **Data Processing Pipelines**: ETL processes for handling large-scale code analysis

## Historical Context

This directory represents experimental outputs from the research phase of performance benchmarking, focusing on:
- Automated performance test generation experiments
- LLM-assisted code analysis results
- Real-world optimization pattern extractions
- Scalable performance evaluation framework development

The insights and methodologies developed from these experiments have been incorporated into the main OmniPerf-Bench system while preserving the experimental data for reference and continued research.

## Dependencies

Key dependencies include:
- `anthropic` - For Claude API integration
- `openai` - For OpenAI API integration  
- `tqdm` - Progress tracking
- `pytest` - Test framework
- Various LLM and analysis utilities

## Notes

- This contains **experimental data** preserved for historical reference and research
- Active development has moved to the main OmniPerf-Bench codebase
- API keys and sensitive configuration are not included in the experimental data
- Some paths and dependencies may need adjustment if trying to reproduce experiments

For current performance benchmarking capabilities, please refer to the main OmniPerf-Bench documentation and codebase. The experimental data here serves as a foundation for the current system's methodologies.
