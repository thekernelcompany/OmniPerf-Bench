#!/usr/bin/env python3
"""
Performance Test Generator

Generates actual performance tests from commit extractions using LLM.
Uses the robust patterns from generate/ module but focused on performance testing.

This replaces the broken classification test generator.
"""

import argparse
import ast
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Import the performance analyzer
from performance_analyzer import PerformanceCommitAnalyzer, PerformanceClaimExtraction

# Optional providers (same as original)
try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


class PerformanceTestLLMClient:
    """LLM client optimized for performance test generation"""
    
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 8000) -> None:
        self.provider = provider or os.getenv("LLM_PROVIDER") or ("openai" if os.getenv("OPENAI_API_KEY") else "anthropic")
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("ANTHROPIC_MODEL") or "gpt-5-mini-2025-08-07"
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._openai_client = None
        self._anthropic_client = None
        if self.provider == "openai" and OpenAI is not None and os.getenv("OPENAI_API_KEY"):
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if self.provider == "anthropic" and anthropic is not None and os.getenv("ANTHROPIC_API_KEY"):
            self._anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str) -> str:
        """Generate LLM response with fallback handling"""
        if self.provider == "openai":
            return self._call_openai(prompt)
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        # Fallbacks
        if self._openai_client is not None:
            return self._call_openai(prompt)
        if self._anthropic_client is not None:
            return self._call_anthropic(prompt)
        raise RuntimeError("No usable LLM provider configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API with robust error handling"""
        if self._openai_client is None:
            return ""

        # Try Chat Completions API
        try:
            # Handle different model capabilities
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": self.max_tokens
            }
            
            # Only add temperature for models that support it (gpt-5-mini doesn't support custom temperature)
            if "gpt-5-mini" not in self.model:
                kwargs["temperature"] = self.temperature
                
            response = self._openai_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return ""

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API with error handling"""
        if self._anthropic_client is None:
            return ""
        try:
            message = self._anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if hasattr(message, "content") and message.content:
                return message.content[0].text if hasattr(message.content[0], 'text') else str(message.content[0])
            return ""
        except Exception as e:
            print(f"Error calling Anthropic: {e}")
            return ""


def load_performance_prompt_template() -> str:
    """Load the performance test generator prompt template"""
    prompt_path = Path(__file__).parent / "performance_test_generator_prompt.md"
    try:
        with open(prompt_path, 'r') as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read performance prompt template: {e}")


def build_performance_test_prompt(
    template: str, 
    commit_data: Dict[str, Any], 
    performance_analysis: PerformanceClaimExtraction
) -> str:
    """Build a complete prompt for performance test generation"""
    
    # Extract key information for the prompt
    commit_hash = performance_analysis.commit_hash
    commit_message = performance_analysis.message
    optimization_type = performance_analysis.optimization_type
    performance_claims = performance_analysis.performance_claims
    expected_speedup = performance_analysis.expected_speedup or 1.0
    optimization_keywords = performance_analysis.optimization_keywords
    affected_functions = performance_analysis.affected_functions
    optimization_domain = performance_analysis.optimization_domain
    
    # Get key code changes
    files_changed = commit_data.get('files_changed', [])
    key_diffs = []
    
    for i, file_change in enumerate(files_changed[:3]):  # Show first 3 files
        file_path = file_change.get('file_path', '')
        diff = file_change.get('diff', '')
        change_type = file_change.get('change_type', '')
        
        key_diffs.append(f"""
### File {i+1}: {file_path} ({change_type})
```diff
{diff[:1000]}{'...' if len(diff) > 1000 else ''}
```""")
    
    # Build the complete prompt
    full_prompt = f"""{template}

## COMMIT ANALYSIS CONTEXT

**Commit Hash**: {commit_hash}
**Message**: {commit_message}
**Optimization Type**: {optimization_type}
**Domain**: {optimization_domain}
**Performance Claims**: {', '.join(performance_claims) if performance_claims else 'No explicit claims'}
**Expected Speedup**: {expected_speedup}x
**Keywords**: {', '.join(optimization_keywords)}
**Affected Functions**: {', '.join(affected_functions) if affected_functions else 'Unknown'}

## KEY CODE CHANGES
{''.join(key_diffs)}

## GENERATION REQUIREMENTS

Based on this analysis, generate a performance test that:

1. **Tests the optimization**: Focus on {optimization_type} optimization in {optimization_domain} domain
2. **Measures {performance_analysis.performance_metric or 'performance'}**: {'Expect ' + str(expected_speedup) + 'x speedup' if expected_speedup > 1.0 else 'Measure performance improvement'}
3. **Uses realistic workloads**: Based on {optimization_domain} domain patterns
4. **Verifies correctness**: Ensure optimization doesn't break functionality
5. **Follows domain best practices**: Use appropriate timing methods for {optimization_type} optimizations

Generate a complete Python test file with the exact structure shown in the template.
Replace all placeholder values with actual implementation based on the commit analysis.

**CRITICAL**: This must be a working performance test, not a classification test.
Focus on measuring actual execution time and speedup, not categorizing commits.
"""
    
    return full_prompt


def clean_llm_performance_test_response(text: str) -> str:
    """Clean LLM response to extract Python performance test code"""
    if not text:
        return ""
    
    s = text.strip()
    
    # Extract from markdown code blocks
    if "```python" in s:
        parts = s.split("```python")
        if len(parts) > 1:
            code_part = parts[1].split("```")[0]
            return code_part.strip()
    elif "```" in s:
        parts = s.split("```")
        # Find the largest code block (likely the main test)
        candidates = [p.strip() for p in parts if p.strip()]
        if candidates:
            # Choose the block that looks most like Python code
            for candidate in candidates:
                if ('def test_' in candidate and 'import' in candidate):
                    return candidate
            # Fallback to the largest block
            return max(candidates, key=len)
    
    # If no code blocks, try to find Python code patterns
    lines = s.split('\n')
    code_start = -1
    for i, line in enumerate(lines):
        if (line.startswith('import ') or line.startswith('from ') or 
            line.startswith('def test_') or line.startswith('class ')):
            code_start = i
            break
    
    if code_start >= 0:
        return '\n'.join(lines[code_start:])
    
    return s


def validate_performance_test_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validate that generated code is syntactically correct Python"""
    try:
        # Basic syntax check
        ast.parse(code)
        
        # Check for required components of a performance test
        required_patterns = [
            'def test_',  # Must have a test function
            'import',     # Must have imports
        ]
        
        for pattern in required_patterns:
            if pattern not in code:
                return False, f"Missing required pattern: {pattern}"
        
        # Check for performance test indicators
        performance_indicators = [
            'time',       # Must measure time
            'assert',     # Must have assertions
        ]
        
        found_indicators = sum(1 for indicator in performance_indicators if indicator in code)
        if found_indicators == 0:
            return False, "Generated code doesn't appear to be a performance test"
        
        return True, None
        
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


def generate_performance_test_for_commit(
    commit_data: Dict[str, Any], 
    performance_analysis: PerformanceClaimExtraction,
    llm_client: PerformanceTestLLMClient,
    template: str
) -> Optional[str]:
    """Generate a performance test for a single commit"""
    
    # Skip commits without performance intent
    if not performance_analysis.has_performance_intent:
        print(f"Skipping {performance_analysis.commit_hash[:8]}: No performance intent detected")
        return None
    
    # Build the prompt
    prompt = build_performance_test_prompt(template, commit_data, performance_analysis)
    
    # Generate the test
    try:
        llm_response = llm_client.generate(prompt)
        test_code = clean_llm_performance_test_response(llm_response)
        
        # Validate the generated code
        is_valid, error = validate_performance_test_syntax(test_code)
        if not is_valid:
            print(f"Generated invalid code for {performance_analysis.commit_hash[:8]}: {error}")
            return None
        
        return test_code
        
    except Exception as e:
        print(f"Failed to generate test for {performance_analysis.commit_hash[:8]}: {e}")
        return None


def save_performance_test(
    output_dir: str, 
    commit_hash: str, 
    test_code: str,
    performance_analysis: PerformanceClaimExtraction
) -> str:
    """Save generated performance test to file"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a descriptive filename
    optimization_type = performance_analysis.optimization_type
    test_name = f"{commit_hash[:8]}_{optimization_type}_performance_test.py"
    test_file = output_path / test_name
    
    # Add header comment with analysis info
    header = f'''"""
Performance Test for Commit {commit_hash}

Message: {performance_analysis.message}
Optimization Type: {performance_analysis.optimization_type}
Expected Speedup: {performance_analysis.expected_speedup}x
Performance Claims: {', '.join(performance_analysis.performance_claims)}
Domain: {performance_analysis.optimization_domain}

Generated automatically from commit analysis.
"""

'''
    
    full_content = header + test_code
    
    with open(test_file, 'w') as f:
        f.write(full_content)
    
    return str(test_file)


def main():
    parser = argparse.ArgumentParser(description="Generate performance tests from commit extractions")
    parser.add_argument("--extractions-dir", default="./commit_extractions", 
                       help="Directory containing commit extraction JSONs")
    parser.add_argument("--output-dir", default="./generated_performance_tests", 
                       help="Output directory for performance tests")
    parser.add_argument("--limit", type=int, default=0, 
                       help="Limit number of commits to process (0 = no limit)")
    parser.add_argument("--only-hash", default="", 
                       help="Only process commits whose hash contains this substring")
    parser.add_argument("--provider", choices=["openai", "anthropic"], 
                       help="LLM provider to use")
    parser.add_argument("--model", help="Model name override")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=8000)
    
    args = parser.parse_args()
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return 1
    
    # Load the prompt template
    try:
        template = load_performance_prompt_template()
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    # Initialize the analyzer and LLM client
    analyzer = PerformanceCommitAnalyzer()
    llm_client = PerformanceTestLLMClient(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Find all commit extraction files
    extractions_path = Path(args.extractions_dir)
    commit_files = list(extractions_path.glob("*.json"))
    commit_files = [f for f in commit_files if f.name != "extraction_summary.json"]
    
    # Apply filters
    if args.only_hash:
        commit_files = [f for f in commit_files if args.only_hash in f.name]
    
    if args.limit > 0:
        commit_files = commit_files[:args.limit]
    
    if not commit_files:
        print("No commit files found to process")
        return 1
    
    print(f"Processing {len(commit_files)} commits...")
    
    # Track results
    results = {
        'total_processed': 0,
        'performance_tests_generated': 0,
        'failed_generations': 0,
        'skipped_non_performance': 0,
        'generated_tests': []
    }
    
    # Process each commit
    for commit_file in commit_files:
        try:
            # Load commit data
            with open(commit_file, 'r') as f:
                commit_data = json.load(f)
            
            # Analyze for performance optimizations
            performance_analysis = analyzer.analyze_commit(commit_data)
            
            results['total_processed'] += 1
            commit_hash = performance_analysis.commit_hash
            
            print(f"\n[{results['total_processed']}/{len(commit_files)}] Processing {commit_hash[:8]}...")
            print(f"  Message: {performance_analysis.message[:60]}...")
            print(f"  Performance Intent: {performance_analysis.has_performance_intent}")
            print(f"  Type: {performance_analysis.optimization_type}")
            print(f"  Claims: {performance_analysis.performance_claims}")
            
            if not performance_analysis.has_performance_intent:
                results['skipped_non_performance'] += 1
                continue
            
            # Generate performance test
            test_code = generate_performance_test_for_commit(
                commit_data, performance_analysis, llm_client, template
            )
            
            if test_code:
                # Save the test
                test_file = save_performance_test(
                    args.output_dir, commit_hash, test_code, performance_analysis
                )
                
                results['performance_tests_generated'] += 1
                results['generated_tests'].append({
                    'commit_hash': commit_hash,
                    'test_file': test_file,
                    'optimization_type': performance_analysis.optimization_type,
                    'expected_speedup': performance_analysis.expected_speedup,
                    'performance_claims': performance_analysis.performance_claims
                })
                
                print(f"  ✓ Generated test: {test_file}")
            else:
                results['failed_generations'] += 1
                print(f"  ✗ Failed to generate test")
                
        except Exception as e:
            print(f"  ✗ Error processing {commit_file.name}: {e}")
            results['failed_generations'] += 1
    
    # Save index of generated tests
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    index_file = output_path / "performance_tests_index.json"
    with open(index_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PERFORMANCE TEST GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total commits processed: {results['total_processed']}")
    print(f"Performance tests generated: {results['performance_tests_generated']}")
    print(f"Skipped (no performance intent): {results['skipped_non_performance']}")
    print(f"Failed generations: {results['failed_generations']}")
    print(f"Success rate: {results['performance_tests_generated']/results['total_processed']*100:.1f}%")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Index file: {index_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())