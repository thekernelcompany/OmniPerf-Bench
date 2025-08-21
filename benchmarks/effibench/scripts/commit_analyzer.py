"""
Enhanced Commit File Extractor with API Identification

Extracts pre-commit and post-commit file contents along with diffs for a given commit hash.
Additionally uses LLM (GPT-5 mini) to identify affected Python APIs in each commit.
Can process single commits or batch process commits from a CSV file.

Dependencies:
    - requests (for OpenAI API calls)
    - OpenAI API key in environment variable OPENAI_API_KEY

Usage:
    Single commit: python commit_analyzer.py <repo_path> <commit_hash> [output_file]
    From CSV:      python commit_analyzer.py <repo_path> --csv <csv_file> [output_dir]

Examples:
    # Extract single commit with API identification
    source .env && python commit_analyzer.py ~/coding-mess/vllm abc1234
    source .env && python commit_analyzer.py ~/coding-mess/vllm abc1234 output.json
    
    # Batch process commits from CSV with API identification
    source .env && python commit_analyzer.py ~/coding-mess/vllm --csv vllm_classification_review.csv
    source .env && python commit_analyzer.py ~/coding-mess/vllm --csv vllm_classification_review.csv ./extractions/

Output:
    Each extraction now includes an 'affected_apis' field with Python APIs identified by LLM.
    APIs are identified based on commit diffs and messages using GPT-5 mini model.
"""

import subprocess
import sys
import json
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict
from tqdm import tqdm
try:
    import requests
except ImportError:
    requests = None


@dataclass
class FileChange:
    """Represents changes to a single file in a commit"""
    file_path: str
    old_content: str
    # new_content: str
    
    diff: str
    change_type: str  # 'modified', 'added', 'deleted', 'renamed'
    
    def to_dict(self) -> Dict:
        return {
            'file_path': self.file_path,
            'old_content': self.old_content,
            # 'new_content': self.new_content,
            'diff': self.diff,
            'change_type': self.change_type,
            'lines_added': self.diff.count('\n+') if self.diff else 0,
            'lines_removed': self.diff.count('\n-') if self.diff else 0,
        }


@dataclass 
class CommitExtraction:
    """Complete extraction of a commit's file changes"""
    commit_hash: str
    parent_hash: str
    message: str
    author: str
    date: str
    files_changed: List[FileChange]
    affected_apis: List[str]  # NEW: APIs identified by LLM
    
    def to_dict(self) -> Dict:
        return {
            'commit_hash': self.commit_hash,
            'parent_hash': self.parent_hash,
            'message': self.message,
            'author': self.author,
            'date': self.date,
            'files_changed': [fc.to_dict() for fc in self.files_changed],
            'affected_apis': self.affected_apis,  # NEW
            'summary': {
                'total_files': len(self.files_changed),
                'files_added': sum(1 for fc in self.files_changed if fc.change_type == 'added'),
                'files_deleted': sum(1 for fc in self.files_changed if fc.change_type == 'deleted'),
                'files_modified': sum(1 for fc in self.files_changed if fc.change_type == 'modified'),
            }
        }


class CommitExtractor:
    """Minimal commit file extractor using only git commands"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        if not (self.repo_path / '.git').exists():
            raise ValueError(f"Not a git repository: {repo_path}")
    
    def _run_git_command(self, cmd: List[str]) -> str:
        """Execute git command and return output"""
        try:
            result = subprocess.run(
                cmd, 
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git command failed: {' '.join(cmd)}\nError: {e.stderr}")
    
    def _get_file_content(self, commit_hash: str, file_path: str) -> Optional[str]:
        """Get file content at specific commit, return None if file doesn't exist"""
        try:
            result = subprocess.run(
                ["git", "show", f"{commit_hash}:{file_path}"],
                cwd=self.repo_path,
                capture_output=True,
                text=False,  # Handle binary files
                check=True
            )
            # Decode with error handling for binary files
            try:
                return result.stdout.decode('utf-8')
            except UnicodeDecodeError:
                return result.stdout.decode('utf-8', errors='ignore')
        except subprocess.CalledProcessError:
            return None  # File doesn't exist at this commit
    
    def _get_file_diff(self, parent_hash: str, commit_hash: str, file_path: str) -> str:
        """Get diff for specific file between commits"""
        try:
            return self._run_git_command([
                "git", "diff", parent_hash, commit_hash, "--", file_path
            ])
        except RuntimeError:
            return ""
    
    def _determine_change_type(self, old_content: Optional[str], new_content: Optional[str]) -> str:
        """Determine the type of change for a file"""
        if old_content is None and new_content is not None:
            return "added"
        elif old_content is not None and new_content is None:
            return "deleted"
        elif old_content != new_content:
            return "modified"
        else:
            return "unchanged"
    
    def _extract_apis_with_llm(self, commit_hash: str, message: str, files_changed: List[FileChange]) -> List[str]:
        """Extract affected APIs using LLM (adapted from GSO)"""
        
        # Build diff text from files_changed
        diff_text = ""
        for file_change in files_changed:
            diff_text += f"--- a/{file_change.file_path}\n+++ b/{file_change.file_path}\n"
            diff_text += file_change.diff + "\n\n"
        
        # Truncate if too long (similar to GSO's MAX_COMMIT_TOKENS)
        MAX_COMMIT_TOKENS = 20000
        if len(diff_text) > MAX_COMMIT_TOKENS:
            diff_text = diff_text[:MAX_COMMIT_TOKENS] + "...(truncated)..."
        
        # Use GSO's proven prompt
        system_prompt = """You are an expert programmer who is annotating data for training and testing code language models.

You will be given a performance or optimization related GitHub commit patch content. Your goal is to identify a list of APIs (functions or methods of a class) that are affected by the commit. Some additional instructions:
1. The APIs should be high-level or top-level APIs in the repo. E.g., pd.read_csv (pandas), requests.get (requests), model.generate (transformers), etc.
2. By high/top-level, we mean APIs that are not internal helper functions.
3. If the commit affects multiple APIs, list them all separated by commas.
4. For methods, use the format "ClassName.method_name" (e.g., DataFrame.dropna).
5. NOTE: Find Affected PYTHON APIs only
    - Do not add backend APIs like C/C++/Rust functions. Instead, you MUST add the Python APIs that call them.
    - IMPORTANT: Just because a commit does not directly mention or update python APIs, does not mean changes to internal code do not affect any python APIs.
    - So, if any backend (e.g., C/C++/Rust) code affects any Python API or bindings, YOU MUST include that Python API in the list.
    - Especially in the case of repos with interfaces via python bindings (e.g., huggingface/tokenizers), find and include the python APIs affected by the commit.
6. Finally, if all else, and the commit does not affect any APIs, write "None"."""

        user_prompt = f"""Analyze the commit using natural language reasoning enclosed in [REASON] [/REASON] tags.
Then list the affected APIs (max 5 comma separated) enclosed in [APIS] [/APIS] tags.
Remember to close all tags properly.

Commit Information:
{diff_text}

Commit Message:
{message}"""

        try:
            # Simple OpenAI API call (you can replace with any LLM provider)
            import os
            
            if requests is None:
                print(f"Warning: requests library not available, skipping API extraction for {commit_hash}")
                return []
                
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                print(f"Warning: No OPENAI_API_KEY found, skipping API extraction for {commit_hash}")
                return []
            
            # Refer to the latest OpenAI API guidance @OpenAI-new
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                json={
                    'model': 'gpt-5-mini-2025-08-07',
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    'max_completion_tokens': 8000,
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result['choices'][0]['message']['content']
                
                # Parse the response (same as GSO)
                try:
                    print(f"LLM Response for {commit_hash}: {llm_response}")  # DEBUG
                    apis_section = llm_response.split("[/APIS]")[0].split("[APIS]")[1].strip()
                    apis = [api.strip() for api in apis_section.split(",") if api.strip()]
                    # Filter out "None" responses
                    apis = [api for api in apis if api.lower() != "none"]
                    return apis
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not parse LLM response for {commit_hash}: {e}")
                    print(f"Raw LLM response: {llm_response}")  # DEBUG
                    return []
            else:
                print(f"Warning: LLM API call failed for {commit_hash}: {response.status_code}")
                print(f"Error response: {response.text}")
                return []
                
        except Exception as e:
            print(f"Warning: LLM API extraction failed for {commit_hash}: {e}")
            return []
    
    def extract_commit(self, commit_hash: str) -> CommitExtraction:
        """Extract complete file changes for a commit"""
        
        # Get commit metadata
        parent_hash = self._run_git_command(["git", "rev-parse", f"{commit_hash}^"])
        message = self._run_git_command(["git", "log", "-1", "--pretty=format:%B", commit_hash])
        author = self._run_git_command(["git", "log", "-1", "--pretty=format:%an <%ae>", commit_hash])
        date = self._run_git_command(["git", "log", "-1", "--pretty=format:%ci", commit_hash])
        
        # Get list of changed files
        changed_files_output = self._run_git_command([
            "git", "diff", "--name-status", parent_hash, commit_hash
        ])
        
        files_changed = []
        
        if changed_files_output:
            for line in changed_files_output.split('\n'):
                if not line.strip():
                    continue
                    
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                    
                status = parts[0]
                file_path = parts[1]
                
                # Handle renames (R100 -> R, M -> M, etc.)
                if status.startswith('R'):
                    # Renamed file - parts[1] is old name, parts[2] is new name
                    if len(parts) >= 3:
                        # old_path = parts[1]  # Not used currently
                        new_path = parts[2]
                        file_path = new_path  # Use new path
                        change_type = "renamed"
                    else:
                        change_type = "modified"
                else:
                    change_type = "modified"  # Will be refined below
                
                # Get file contents
                old_content = self._get_file_content(parent_hash, file_path)
                new_content = self._get_file_content(commit_hash, file_path)
                
                # Refine change type based on content
                if change_type != "renamed":
                    change_type = self._determine_change_type(old_content, new_content)
                
                # Get diff
                diff = self._get_file_diff(parent_hash, commit_hash, file_path)
                
                files_changed.append(FileChange(
                    file_path=file_path,
                    old_content=old_content or "",
                    # new_content=new_content or "",
                    diff=diff,
                    change_type=change_type
                ))
        
        # Extract affected APIs using LLM
        affected_apis = self._extract_apis_with_llm(commit_hash, message, files_changed)
        
        return CommitExtraction(
            commit_hash=commit_hash,
            parent_hash=parent_hash,
            message=message,
            author=author,
            date=date,
            files_changed=files_changed,
            affected_apis=affected_apis
        )
    
    def extract_to_json(self, commit_hash: str, output_file: Optional[str] = None) -> str:
        """Extract commit and return/save as JSON"""
        extraction = self.extract_commit(commit_hash)
        json_data = json.dumps(extraction.to_dict(), indent=2, ensure_ascii=False)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_data)
            print(f"Extraction saved to: {output_file}")
        
        return json_data
    
    def extract_from_csv(self, csv_file: str, output_dir: str = "./commit_extractions/") -> Dict:
        """Extract all commits from CSV file and save results"""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load commits from CSV
        commits = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                commits.append({
                    'commit_hash': row['commit_hash'].strip(),
                    'category': row['category'].strip(),
                    'json_has_tests': row['json_has_tests'].strip(),
                    'json_has_benchmarks': row['json_has_benchmarks'].strip(),
                    'is_test_actually_there': row['is_test_actually_there'].strip(),
                    'is_benchmark_actually_there': row['is_benchmark_actually_there'].strip(),
                    'sample_clues': row['sample_clues'].strip()
                })
        
        # Results tracking
        results = {
            'total_commits': len(commits),
            'successful_extractions': 0,
            'failed_extractions': 0,
            'extractions': {},
            'errors': {}
        }
        
        print(f"Found {len(commits)} commits to process from {csv_file}")
        
        # Process each commit
        for i, commit_info in enumerate(tqdm(commits, desc="Processing commits")):
            commit_hash = commit_info['commit_hash']
            
            try:
                print(f"\nProcessing commit {i+1}/{len(commits)}: {commit_hash}")
                
                # Extract commit details
                extraction = self.extract_commit(commit_hash)
                extraction_dict = extraction.to_dict()
                
                # Add CSV metadata to the extraction
                extraction_dict['csv_metadata'] = {
                    'category': commit_info['category'],
                    'json_has_tests': commit_info['json_has_tests'],
                    'json_has_benchmarks': commit_info['json_has_benchmarks'],
                    'is_test_actually_there': commit_info['is_test_actually_there'],
                    'is_benchmark_actually_there': commit_info['is_benchmark_actually_there'],
                    'sample_clues': commit_info['sample_clues']
                }
                
                print(f"  Affected APIs: {extraction_dict.get('affected_apis', [])}")
                
                # Save individual commit extraction
                individual_file = output_path / f"{commit_hash}.json"
                with open(individual_file, 'w', encoding='utf-8') as f:
                    json.dump(extraction_dict, f, indent=2, ensure_ascii=False)
                
                # Add to results
                results['extractions'][commit_hash] = extraction_dict
                results['successful_extractions'] += 1
                
                print(f"✓ Successfully extracted {commit_hash}")
                
            except Exception as e:
                error_msg = str(e)
                print(f"✗ Failed to extract {commit_hash}: {error_msg}")
                
                results['errors'][commit_hash] = {
                    'error': error_msg,
                    'csv_metadata': commit_info
                }
                results['failed_extractions'] += 1
                continue
        
        # Save summary results
        summary_file = output_path / "extraction_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*80)
        print("EXTRACTION SUMMARY")
        print("="*80)
        print(f"Total commits processed: {results['total_commits']}")
        print(f"Successful extractions: {results['successful_extractions']}")
        print(f"Failed extractions: {results['failed_extractions']}")
        print(f"Success rate: {results['successful_extractions']/results['total_commits']*100:.1f}%")
        
        if results['errors']:
            print("\nFailed commits:")
            for commit_hash, error_info in results['errors'].items():
                print(f"  - {commit_hash}: {error_info['error']}")
        
        print(f"\nResults saved to: {output_dir}")
        print(f"Summary saved to: {summary_file}")
        
        return results


def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single commit: python commit_analyzer.py <repo_path> <commit_hash> [output_file]")
        print("  From CSV:      python commit_analyzer.py <repo_path> --csv <csv_file> [output_dir]")
        print()
        print("Examples:")
        print("  python commit_analyzer.py /path/to/repo abc1234")
        print("  python commit_analyzer.py /path/to/repo abc1234 output.json")
        print("  python commit_analyzer.py /path/to/repo --csv vllm_classification_review.csv")
        print("  python commit_analyzer.py /path/to/repo --csv vllm_classification_review.csv ./extractions/")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    
    try:
        extractor = CommitExtractor(repo_path)
        
        # Check if CSV mode
        if len(sys.argv) >= 4 and sys.argv[2] == "--csv":
            csv_file = sys.argv[3]
            output_dir = sys.argv[4] if len(sys.argv) > 4 else "./commit_extractions/"
            
            print(f"Extracting commits from CSV {csv_file} using repo {repo_path}...")
            extractor.extract_from_csv(csv_file, output_dir)
            
        else:
            # Single commit mode
            if len(sys.argv) < 3:
                print("Error: commit_hash required for single commit mode")
                sys.exit(1)
                
            commit_hash = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else None
            
            print(f"Extracting commit {commit_hash} from {repo_path}...")
            json_output = extractor.extract_to_json(commit_hash, output_file)
            
            if not output_file:
                print("\n" + "="*80)
                print("EXTRACTION RESULT:")
                print("="*80)
                print(json_output)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()