#!/usr/bin/env python3
"""
Split vLLM commits from a single JSONL file into separate JSON files.
Each commit will be saved as a separate file named by its commit hash.
"""

import json
import os
from pathlib import Path


def split_commits(jsonl_path, output_dir):
    """
    Split a JSONL file containing commits into separate JSON files.
    
    Args:
        jsonl_path: Path to the input JSONL file
        output_dir: Directory where individual commit files will be saved
    """
    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input file not found: {jsonl_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    commit_count = 0
    missing_hash_count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                commit_data = json.loads(line)
                commit_hash = commit_data.get('commit_hash')
                
                if not commit_hash:
                    print(f"Warning: Line {line_num} missing commit_hash, skipping...")
                    missing_hash_count += 1
                    continue
                
                output_file = output_dir / f"{commit_hash}.json"
                
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    json.dump(commit_data, out_f, indent=2, ensure_ascii=False)
                
                commit_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"Successfully split {commit_count} commits into separate files")
    if missing_hash_count > 0:
        print(f"Warning: {missing_hash_count} entries were skipped due to missing commit_hash")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python split_vllm_commits.py <input_jsonl> <output_dir>")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    split_commits(jsonl_path, output_dir)




