#!/usr/bin/env python3
"""
LLM-powered test case generator for vLLM commits
Reads extracted commit JSON files and uses LLM to generate test_case_generator code for each commit
"""

import json
import os
import glob
from typing import List, Dict, Any
import anthropic
try:
    from openai import OpenAI  # OpenAI Python SDK >= 1.0.0
except Exception:
    OpenAI = None

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Using system environment variables only.")
    pass

def load_commit_extractions(extractions_dir: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Load first N commit extraction JSON files"""
    json_files = glob.glob(os.path.join(extractions_dir, "*.json"))
    # Filter out the summary file
    json_files = [f for f in json_files if not f.endswith("extraction_summary.json")]
    json_files.sort()  # Ensure consistent ordering
    
    extractions = []
    for i, json_file in enumerate(json_files[:limit]):
        try:
            with open(json_file, 'r') as f:
                extraction = json.load(f)
                extractions.append(extraction)
                print(f"Loaded {os.path.basename(json_file)}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return extractions

def create_llm_prompt(extraction: Dict[str, Any]) -> str:
    """Create prompt for LLM to generate test_case_generator"""
    
    # Extract metadata from the extraction
    commit_info = extraction['csv_metadata']
    commit_hash = extraction['commit_hash']
    files_changed = extraction['files_changed']
    commit_msg = extraction['message']
    
    # Create detailed analysis of the actual code changes
    detailed_changes = []
    for i, file_change in enumerate(files_changed[:3]):  # Show first 3 files in detail
        change_details = f"\n### File {i+1}: {file_change['file_path']} ({file_change['change_type']})"
        
        if file_change.get('diff'):
            change_details += f"\n**Diff Preview:**\n```\n{file_change['diff'][:800]}{'...' if len(file_change['diff']) > 800 else ''}\n```"
        
        if file_change.get('old_content'):
            content_preview = file_change['old_content'][:600]
            change_details += f"\n**Content Preview:**\n```\n{content_preview}{'...' if len(file_change['old_content']) > 600 else ''}\n```"
        
        detailed_changes.append(change_details)
    
    # Include remaining files summary
    remaining_files = []
    for file_change in files_changed[3:]:
        remaining_files.append(f"- {file_change['file_path']} ({file_change['change_type']})")
    
    # Extract actual technical vocabulary from the commit data
    technical_terms = set()
    function_names = set()
    file_extensions = set()
    file_paths = []
    
    for file_change in files_changed:
        file_paths.append(file_change['file_path'])
        # Extract file extensions
        if '.' in file_change['file_path']:
            ext = file_change['file_path'].split('.')[-1]
            file_extensions.add(f".{ext}")
        
        # Extract technical terms from diffs and content
        for content_field in ['diff', 'old_content']:
            if file_change.get(content_field):
                content = file_change[content_field].lower()
                # Look for function names (things before parentheses)
                import re
                functions = re.findall(r'\b([a-z_][a-z0-9_]*)\s*\(', content)
                function_names.update(functions[:10])  # Limit to first 10
                
                # Extract technical keywords
                tech_keywords = ['kernel', 'cuda', 'cache', 'block', 'copy', 'tensor', 'gpu', 'cpu', 'memory', 'stream', 'device', 'dtype', 'torch', 'parallel', 'thread', 'grid', 'shared', 'global', 'void', 'template', 'scalar_t', 'numel', 'dispatch', 'assert', 'include', 'namespace', 'class', 'struct', 'enum', 'typedef', 'const', 'static', 'inline', 'extern', 'model', 'forward', 'sample', 'sequence']
                for keyword in tech_keywords:
                    if keyword in content:
                        technical_terms.add(keyword)

    # Build lists of real data for the prompt
    kernel_indicators = [t for t in technical_terms if t in ['kernel', 'cuda', 'gpu', 'cache', 'block', 'memory']]
    model_indicators = [t for t in technical_terms if t in ['model', 'forward', 'tensor', 'sample', 'sequence']]
    kernel_extensions = [f for f in file_extensions if f in ['.cu', '.cpp', '.c', '.h', '.cuh']]
    
    prompt = f"""You need to create a COMPLETE, WORKING Python test case generator for classifying this specific vLLM commit.

## CONTEXT: This is commit {commit_hash}
- **True Category**: {commit_info['category']} 
- **Commit Message**: {commit_msg}
- **Why it's {commit_info['category']}**: This commit {'has CUDA kernel files (.cu), implements kernel functions, and deals with GPU/cache operations' if commit_info['category'] == 'kernel-based' else 'deals with model operations' if commit_info['category'] == 'model-based' else 'is infrastructure/utility related'}

## ACTUAL DATA FROM THIS COMMIT:
- **Files Changed**: {file_paths}
- **Key Technical Terms**: {sorted(list(technical_terms))[:20]}
- **Key Function Names**: {sorted(list(function_names))[:15]}
- **File Extensions**: {sorted(list(file_extensions))}

## YOUR TASK:
Create a COMPLETE Python file that generates test cases for classifying vLLM commits. The test generator should:

1. Have a Solution class with a classify_commit method that CORRECTLY classifies commits based on the patterns from THIS commit
2. Have a generate_test_case function that creates VARIED test inputs using the REAL data from this commit
3. Generate 100 different test cases with varied sample_clues
4. All test cases should be self-consistent (the classifier should correctly classify its own generated clues)

## REQUIRED OUTPUT - Complete Python Code:

```python
import random
from typing import List, Tuple

class Solution:
    def classify_commit(self, commit_hash: str, sample_clues: str) -> str:
        \"\"\"Classify a vLLM commit based on sample clues.
        
        This classifier is based on commit {commit_hash} which is {commit_info['category']}.
        It should recognize patterns similar to this commit.
        \"\"\"
        clues_lower = sample_clues.lower()
        
        # Kernel-based patterns (from this commit)
        kernel_indicators = ['kernel', 'cuda', 'gpu', 'cache_kernels', 'block', 'copy_blocks', '.cu']
        model_indicators = ['model', 'forward', 'sample', 'sampler', 'sequence']
        
        # Check for kernel-based commit (like {commit_hash})
        for indicator in kernel_indicators:
            if indicator in clues_lower:
                return "kernel-based"
        
        # Check for model-based commit
        for indicator in model_indicators:
            if indicator in clues_lower:
                return "model-based"
        
        # Default to miscellaneous
        return "miscellaneous"

def generate_test_case():
    \"\"\"Generate a test case based on patterns from commit {commit_hash}.\"\"\"
    solution = Solution()
    
    # Real data from commit {commit_hash}
    real_files = {file_paths}
    real_terms = {sorted(list(technical_terms))[:30]}
    real_functions = {sorted(list(function_names))[:20]}
    
    # This commit is {commit_info['category']}, so we'll generate variations
    # that should classify as {commit_info['category']} or test edge cases
    
    # Randomly choose what type of test case to generate
    test_type = random.choice(['exact', 'variation', 'mixed', 'edge_case'])
    
    if test_type == 'exact':
        # Use exact file paths and terms from this commit
        file = random.choice(real_files)
        term = random.choice(list(real_terms))
        sample_clues = f"Modified {{file}} with {{term}} operations"
        
    elif test_type == 'variation':
        # Create variations using real data
        terms = random.sample(list(real_terms), min(3, len(real_terms)))
        sample_clues = f"Implemented {{' and '.join(terms)}} functionality"
        
    elif test_type == 'mixed':
        # Mix files and functions
        if real_functions:
            func = random.choice(list(real_functions))
            file = random.choice(real_files)
            sample_clues = f"Added {{func}}() function in {{file}}"
        else:
            sample_clues = f"Updated {{random.choice(real_files)}}"
            
    else:  # edge_case
        # Test edge cases and other categories
        if random.random() > 0.5:
            # Generate a non-kernel case
            sample_clues = "Updated configuration files and documentation"
        else:
            # Generate a model case
            sample_clues = "Modified model runner and sampling logic"
    
    # Use a realistic commit hash (either the real one or a variation)
    commit_hash = "{commit_hash}" if random.random() > 0.3 else f"{{commit_hash[:8]}}{{random.randint(1000, 9999)}}"
    
    # Calculate expected result using our classifier
    expected_result = solution.classify_commit(commit_hash, sample_clues)
    
    return (commit_hash, sample_clues), expected_result

def test_generated_test_cases(num_tests):
    \"\"\"Generate and test multiple test cases.\"\"\"
    test_case_generator_results = []
    
    for i in range(num_tests):
        inputs, expected_result = generate_test_case()
        solution = Solution()
        
        # Test the generated case
        commit_hash, sample_clues = inputs
        result = solution.classify_commit(commit_hash, sample_clues)
        assert result == expected_result, f"Test {{i}} failed: expected {{expected_result}}, got {{result}}"
        
        # Format as assertion string
        test_case_generator_results.append(
            f"assert solution.classify_commit('{{commit_hash}}', '{{sample_clues}}') == '{{expected_result}}'"
        )
    
    return test_case_generator_results

if __name__ == '__main__':
    num_tests = 100
    test_case_generator_results = test_generated_test_cases(num_tests)
    
    # Write results to file (following eg_test_generator.txt pattern)
    with open("./full_tmp/{commit_hash[:8]}.txt", "w") as f:
        f.write("\\n".join(test_case_generator_results))
    
    print(f"Generated {{len(test_case_generator_results)}} test cases for commit {commit_hash[:8]}")
    print(f"\\nSample test cases:")
    for i in range(min(3, len(test_case_generator_results))):
        print(test_case_generator_results[i])
```

IMPORTANT: 
- The code above is a TEMPLATE. Fill in the classify_commit logic based on what makes THIS commit {commit_info['category']}
- The generate_test_case function should create realistic variations using the ACTUAL file paths, terms, and functions from commit {commit_hash}
- All generated test cases must be self-consistent (pass their own assertions)
- Use the REAL data provided, not placeholders

Generate the COMPLETE, WORKING Python code now:"""

    return prompt

def clean_response(response: str) -> str:
    """Clean LLM response to extract only Python code"""
    if not response:
        return ""
    
    # Remove markdown code blocks if present
    if "```python" in response:
        # Extract content between ```python and ```
        start_marker = "```python"
        end_marker = "```"
        start_idx = response.find(start_marker)
        if start_idx != -1:
            start_idx += len(start_marker)
            end_idx = response.find(end_marker, start_idx)
            if end_idx != -1:
                response = response[start_idx:end_idx].strip()
    elif "```" in response:
        # Handle generic code blocks
        parts = response.split("```")
        if len(parts) >= 3:
            response = parts[1].strip()
    
    # Clean up any remaining formatting
    response = response.strip()
    
    # Ensure it starts with import or class
    if response and not (response.startswith('import') or response.startswith('from') or response.startswith('class')):
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith(('import', 'from', 'class')):
                response = '\n'.join(lines[i:])
                break
    
    return response

def validate_generated_code(code: str, extraction: Dict[str, Any]) -> bool:
    """Validate that the generated code is a working test case generator"""
    if not code:
        return False
    
    # Check for required components
    required_components = [
        ('class Solution:', 'Solution class'),
        ('def classify_commit', 'classify_commit method'),
        ('def generate_test_case', 'generate_test_case function'),
        ('def test_generated_test_cases', 'test_generated_test_cases function'),
        ('if __name__', 'main block')
    ]
    
    for component, name in required_components:
        if component not in code:
            print(f"⚠️  Warning: Missing {name}")
            return False
    
    # Check if it uses real data from the commit
    files_changed = extraction['files_changed']
    real_file_paths = [f['file_path'] for f in files_changed]
    
    # Count how many real elements are used
    real_paths_used = sum(1 for path in real_file_paths if path in code)
    
    # Check for proper classification logic
    category = extraction['csv_metadata']['category']
    if category == 'kernel-based':
        # Should have kernel-related terms in the classifier
        if not any(term in code.lower() for term in ['kernel', 'cuda', 'gpu', '.cu']):
            print("⚠️  Warning: Kernel-based commit but missing kernel classification logic")
            return False
    elif category == 'model-based':
        # Should have model-related terms
        if not any(term in code.lower() for term in ['model', 'forward', 'sample']):
            print("⚠️  Warning: Model-based commit but missing model classification logic")
            return False
    
    print(f"✓ Validation passed: Complete test generator with {real_paths_used} real file paths")
    return True

def call_claude_api(prompt: str) -> str:
    """Call Claude API to generate test case generator"""
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    try:
        message = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=4000,
            temperature=0.1,
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        )
        return clean_response(message.content[0].text)
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return ""

def call_openai_api(prompt: str) -> str:
    """Call OpenAI API to generate test case generator (alternative)"""
    if OpenAI is None:
        print("OpenAI SDK not installed. Please install 'openai>=1.0.0' or use the Anthropic path.")
        return ""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return ""

    model = os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07")
    try:
        client = OpenAI(api_key=api_key)
        # Prefer Responses API for GPT-5 models
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=4000,
            )
            content = getattr(response, "output_text", None)
            if content:
                return clean_response(content)
            # Fallback extraction if output_text is not present
            text_chunks = []
            output_items = getattr(response, "output", []) or []
            for item in output_items:
                contents = getattr(item, "content", []) or []
                for c in contents:
                    text_obj = getattr(c, "text", None)
                    if hasattr(text_obj, "value"):
                        text_chunks.append(getattr(text_obj, "value"))
                    elif text_obj:
                        text_chunks.append(str(text_obj))
            if text_chunks:
                return clean_response("".join(text_chunks))
        except Exception:
            # Fallback to Chat Completions if Responses API not available
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            return clean_response(content or "")
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return ""

def create_dataset_entry(extraction: Dict[str, Any], test_generator_code: str, index: int) -> Dict[str, Any]:
    """Create dataset entry with the test case generator"""
    
    commit_info = extraction['csv_metadata']
    commit_hash = extraction['commit_hash']
    
    return {
        "problem_idx": index,
        "commit_hash": commit_hash,
        "category": commit_info['category'],
        "message": extraction['message'],
        "files_changed": [f['file_path'] for f in extraction['files_changed']],
        "test_case_generator": test_generator_code,
        "metadata": {
            "sample_clues": commit_info['sample_clues'],
            "num_files": len(extraction['files_changed']),
            "file_extensions": list(set(f['file_path'].split('.')[-1] for f in extraction['files_changed'] if '.' in f['file_path']))
        }
    }

def main():
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable")
        return
    
    # Load commit extractions
    extractions_dir = './commit_extractions/'
    if not os.path.exists(extractions_dir):
        print(f"Error: Directory '{extractions_dir}' not found")
        print("Please run the commit analyzer first to generate extractions")
        return
    
    extractions = load_commit_extractions(extractions_dir, limit=1)  # Start with just 1 for testing
    print(f"Loaded {len(extractions)} commit extractions")
    
    if not extractions:
        print("No commit extractions found")
        return
    
    dataset = []
    
    for i, extraction in enumerate(extractions):
        commit_hash = extraction['commit_hash']
        commit_info = extraction['csv_metadata']
        
        print(f"\nProcessing extraction {i+1}/{len(extractions)}: {commit_hash[:8]}...")
        print(f"Category: {commit_info['category']}")
        print(f"Sample clues: {commit_info['sample_clues']}")
        print(f"Files changed: {len(extraction['files_changed'])}")
        
        # Create prompt and call LLM
        prompt = create_llm_prompt(extraction)
        
        # Try OpenAI first, then Claude
        test_generator_code = ""
        if os.getenv("OPENAI_API_KEY"):
            test_generator_code = call_openai_api(prompt)
        
        if not test_generator_code and os.getenv("ANTHROPIC_API_KEY"):
            test_generator_code = call_claude_api(prompt)
        
        if test_generator_code:
            # Validate the generated code
            if validate_generated_code(test_generator_code, extraction):
                # Create dataset entry
                dataset_entry = create_dataset_entry(extraction, test_generator_code, i)
                dataset.append(dataset_entry)
                print(f"✓ Generated and validated test case generator for {commit_hash[:8]}")
            else:
                print(f"✗ Generated code failed validation for {commit_hash[:8]} - potential reward hacking detected")
        else:
            print(f"✗ Failed to generate for {commit_hash[:8]}")
    
    # Save dataset
    output_file = 'vllm_0f_dataset.json'
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"\n🎉 Created dataset with {len(dataset)} entries")
    print(f"Saved to: {output_file}")
    
    # Show summary
    categories = {}
    for entry in dataset:
        cat = entry['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")
    
    # Test the generated code
    if dataset and dataset[0]['test_case_generator']:
        print("\n📋 Testing the first generated test case...")
        test_code = dataset[0]['test_case_generator']
        print("Generated test code (first 500 chars):")
        print(test_code[:500] + "..." if len(test_code) > 500 else test_code)

if __name__ == '__main__':
    main()