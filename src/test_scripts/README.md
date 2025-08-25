# Scripts Directory

This directory contains automation scripts for the EffiBench test generation pipeline.

## generate_test_generators.py

### Overview
A script that automatically generates test case generator scripts from commit extraction JSONs using Large Language Models (LLMs). This is the core automation tool for creating commit-specific test generators.

### Purpose
Given a commit's extracted changes (in JSON format), this script:
1. Reads the commit extraction JSON containing diffs, metadata, and file changes
2. Sends the full JSON context to an LLM with a comprehensive prompt template
3. Generates a standalone Python test script that creates realistic test cases for that commit
4. Validates the generated code syntax and saves it as a reusable test generator

### Architecture

```
Input: commit_extractions/*.json
   ↓
LLM (OpenAI/Anthropic) + Prompt Template
   ↓
Generated Python Test Script
   ↓
Syntax Validation
   ↓
Output: generated_test_generators/<hash>_test_case_generator.py
```

### Key Features

#### 1. Multi-Provider LLM Support
- **OpenAI**: GPT models via OpenAI API
- **Anthropic**: Claude models via Anthropic API
- Automatic fallback between providers and API endpoints
- Configurable model selection and parameters

#### 2. Template-Based Prompt System
- Uses `test_case_generator_prompt_v3.md` as the base prompt template
- Injects full commit JSON context into the prompt
- Supports environment variable override: `TEST_CASE_GENERATOR_PROMPT`

#### 3. Code Quality Assurance
- Extracts Python code from LLM responses (handles markdown code blocks)
- Validates Python syntax using AST parsing
- Only saves syntactically correct scripts

#### 4. Batch Processing
- Processes entire directories of commit extractions
- Filtering options by commit hash or count limits
- Progress tracking and error reporting

### Usage

#### Basic Usage
```bash
# Process all commits in default directory
python scripts/generate_test_generators.py

# Use specific directories
python scripts/generate_test_generators.py \
  --extractions-dir ./commit_extractions \
  --out-dir ./generated_test_generators
```

#### Advanced Options
```bash
# Process only specific commits
python scripts/generate_test_generators.py --only-hash 0f40557a

# Limit number of commits processed
python scripts/generate_test_generators.py --limit 5

# Use specific LLM provider and model
python scripts/generate_test_generators.py \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --temperature 0.2 \
  --max-tokens 10000
```

### Environment Setup

#### Required Environment Variables
Set at least one of:
- `OPENAI_API_KEY`: For OpenAI models
- `ANTHROPIC_API_KEY`: For Anthropic models

#### Optional Configuration
- `LLM_PROVIDER`: Force provider selection (`openai` or `anthropic`)
- `OPENAI_MODEL`: Override default OpenAI model
- `ANTHROPIC_MODEL`: Override default Anthropic model
- `TEST_CASE_GENERATOR_PROMPT`: Path to custom prompt template

#### Example .env file
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

### Input Format

The script expects commit extraction JSONs with this structure:
```json
{
  "commit_hash": "0f40557af6141ced118b81f2a04e651a0c6c9dbd",
  "message": "Implement block copy kernel to optimize beam search",
  "files_changed": [
    {
      "file_path": "cacheflow/worker/cache_engine.py",
      "old_content": "...",
      "diff": "...",
      "change_type": "modified"
    }
  ],
  "csv_metadata": {
    "category": "kernel-based",
    "json_has_tests": "TRUE"
  }
}
```

### Output Format

#### Generated Test Scripts
Each commit produces a standalone Python script at:
```
generated_test_generators/<8-char-hash>_test_case_generator.py
```

#### Index File
A master index tracks all generated scripts:
```json
[
  {
    "commit_hash": "0f40557af6141ced118b81f2a04e651a0c6c9dbd",
    "script_path": "./generated_test_generators/0f40557a_test_case_generator.py",
    "category": "kernel-based",
    "message": "Implement block copy kernel to optimize beam search"
  }
]
```

### Code Flow

1. **Discovery**: `list_extraction_files()` finds all JSON files
2. **Loading**: `load_extraction()` reads commit data
3. **Prompt Building**: `build_prompt_from_template()` creates LLM input
4. **Generation**: `LLMClient.generate()` calls LLM APIs
5. **Cleaning**: `clean_llm_code_response()` extracts Python code
6. **Validation**: `validate_python_syntax()` checks syntax
7. **Saving**: `save_script()` writes final test generator

### Error Handling

- **Syntax Errors**: Invalid Python code is rejected and logged
- **API Failures**: Multiple fallback strategies for LLM providers
- **File I/O**: Graceful handling of missing or corrupted files
- **Rate Limiting**: Built-in retry mechanisms for API calls

### Integration Points

#### Dependencies
- **commit_analyzer.py**: Produces input JSON files
- **test_case_generator_prompt_v2.md**: Defines LLM instructions
- **LLM APIs**: External services for code generation

#### Output Usage
Generated test scripts can be:
- Executed directly to produce test cases
- Integrated into CI/CD pipelines
- Used for regression testing
- Analyzed for test quality metrics

### Current Limitations

1. **Import Resolution**: Generated tests may have incorrect module imports (addressed by repository context system)
2. **Context Window**: Large commits may exceed LLM token limits
3. **Quality Variance**: LLM output quality depends on prompt engineering
4. **Provider Dependencies**: Requires external API access

### Future Enhancements

- Repository-aware import generation
- Multi-stage validation pipeline
- Test execution and verification
- Quality scoring and filtering
- Template specialization by commit category