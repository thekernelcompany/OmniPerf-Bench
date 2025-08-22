#!/usr/bin/env python3
"""
Simple demo of the performance benchmarking system.
Run this from any git repository with two commits to compare.
"""

import subprocess
import json
from pathlib import Path


def create_demo_benchmark():
    """Creates a simple benchmark.py file for testing."""
    benchmark_code = '''#!/usr/bin/env python3
import json
import time
import random
import sys

def run_benchmark():
    """Simple benchmark that simulates work and returns throughput."""
    # Simulate some work
    start = time.time()
    
    # Mock computation - in real code this would be your actual workload
    total = 0
    for i in range(1000000):
        total += i * random.random()
    
    duration = time.time() - start
    throughput = 1000000 / duration  # operations per second
    
    return {
        "throughput": round(throughput, 2),
        "duration": round(duration, 4),
        "total": total
    }

if __name__ == "__main__":
    result = run_benchmark()
    print(f"Throughput: {result['throughput']} ops/sec")
    
    # Save result if output path provided
    if len(sys.argv) > 2 and sys.argv[1] == "--output":
        output_file = sys.argv[2]
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {output_file}")
'''
    
    Path("benchmark.py").write_text(benchmark_code)
    subprocess.run(["chmod", "+x", "benchmark.py"])
    print("✓ Created benchmark.py")


def main():
    print("=== Performance Benchmark Demo ===\n")
    
    # Get current git info
    try:
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        
        # Get previous commit
        prev_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD~1"], text=True  
        ).strip()
        
        print(f"Current commit: {current_commit[:8]}")
        print(f"Previous commit: {prev_commit[:8]}")
        
    except subprocess.CalledProcessError:
        print("❌ This demo requires a git repository with at least 2 commits")
        return
    
    # Create benchmark if it doesn't exist
    if not Path("benchmark.py").exists():
        create_demo_benchmark()
    
    # Set environment variables
    import os
    os.environ["HUMAN_COMMIT"] = current_commit
    os.environ["PRE_COMMIT"] = prev_commit
    
    # Run the benchmark
    print(f"\n=== Running Benchmark ===")
    
    try:
        from bench.cli import app
        import typer
        
        # Run the benchmark
        result = subprocess.run([
            "python", "-m", "bench.cli", "run", "tasks/example.yaml"
        ], cwd="perf-agents-bench", capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Benchmark completed successfully!")
            print(result.stdout)
        else:
            print("❌ Benchmark failed:")
            print(result.stderr)
            
    except ImportError:
        print("❌ Please install the benchmark package first:")
        print("cd perf-agents-bench && pip install -e .")


if __name__ == "__main__":
    main()