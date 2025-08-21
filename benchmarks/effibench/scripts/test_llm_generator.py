#!/usr/bin/env python3
"""
Test version of LLM generator (without actual API calls)
Shows the prompts that would be sent to the LLM
"""

import csv
from llm_test_generator import load_csv_entries, create_llm_prompt

def main():
    # Load first 3 entries for testing
    entries = load_csv_entries('vllm_classification_review - vllm_classification_review.csv', limit=3)
    
    print("=== LLM Test Case Generator Prompts ===\n")
    
    for i, entry in enumerate(entries):
        print(f"{'='*60}")
        print(f"ENTRY {i+1}: {entry['commit_hash'][:8]} - {entry['category']}")
        print(f"{'='*60}")
        
        prompt = create_llm_prompt(entry)
        print(prompt)
        print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    main()