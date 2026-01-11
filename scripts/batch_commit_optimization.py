#!/usr/bin/env python3
"""
Batch commit optimization pipeline for processing multiple commits.

This script extends the single-commit optimization pipeline to handle
multiple commits in batch, useful for evaluating agent performance across
a larger dataset of optimizations.

Usage:
    python batch_commit_optimization.py --commit-dir tmp_single_commit/ \
                                       --test-dir misc/experiments/generated_test_generators_v4/ \
                                       --repo-path vllm \
                                       --output-dir results/

Features:
- Parallel processing of commits (configurable concurrency)
- Progress tracking and reporting
- Aggregated statistics and success rates
- Resume capability for interrupted batches
- GSO prediction format for all commits
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Results from batch processing."""
    total_commits: int = 0
    successful_optimizations: int = 0
    failed_optimizations: int = 0
    skipped_commits: int = 0
    execution_time_seconds: float = 0.0
    individual_results: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_commits == 0:
            return 0.0
        return (self.successful_optimizations / self.total_commits) * 100


class BatchCommitOptimizer:
    """Batch processor for commit optimizations."""
    
    def __init__(self, output_dir: Path, max_workers: int = 2):
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup batch logging
        self.batch_log = output_dir / "batch_optimization.log"
        self.results_file = output_dir / "batch_results.json"
        
    def find_commit_pairs(self, commit_dir: Path, test_dir: Path) -> List[tuple]:
        """Find matching commit JSON and test script pairs."""
        pairs = []
        
        for commit_file in commit_dir.glob("*.json"):
            commit_hash = commit_file.stem.replace(".json", "")
            
            # Look for matching test script
            test_pattern = f"*{commit_hash[:8]}*test_case_generator.py"
            matching_tests = list(test_dir.glob(test_pattern))
            
            if matching_tests:
                pairs.append((commit_file, matching_tests[0]))
            else:
                logger.warning(f"No test script found for commit {commit_hash}")
                
        return pairs
    
    def process_single_commit(
        self, 
        commit_file: Path, 
        test_script: Path, 
        repo_path: Path,
        commit_index: int,
        total_commits: int
    ) -> Dict[str, Any]:
        """Process a single commit through the optimization pipeline."""
        
        commit_hash = commit_file.stem.replace(".json", "")
        logger.info(f"[{commit_index+1}/{total_commits}] Processing commit {commit_hash[:8]}")
        
        # Create individual work directory
        work_dir = self.output_dir / f"work_{commit_hash[:8]}"
        work_dir.mkdir(exist_ok=True)
        
        start_time = time.time()
        
        try:
            # Run the single commit optimization pipeline
            cmd = [
                sys.executable, "run_commit_optimization.py",
                "--commit-json", str(commit_file),
                "--test-script", str(test_script),
                "--repo-path", str(repo_path),
                "--work-dir", str(work_dir),
                "--cleanup"
            ]
            
            # Execute with timeout (30 minutes per commit)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutes
                cwd=Path.cwd()
            )
            
            execution_time = time.time() - start_time
            success = result.returncode == 0
            
            # Parse GSO prediction if available
            gso_file = work_dir / f"gso_prediction_{commit_hash[:8]}.json"
            gso_prediction = None
            if gso_file.exists():
                with open(gso_file, 'r') as f:
                    gso_prediction = json.load(f)
            
            result_data = {
                "commit_hash": commit_hash,
                "commit_file": str(commit_file),
                "test_script": str(test_script),
                "success": success,
                "execution_time_seconds": execution_time,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "gso_prediction": gso_prediction,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }
            
            # Log progress
            status = "✅ SUCCESS" if success else "❌ FAILED"
            logger.info(f"[{commit_index+1}/{total_commits}] {commit_hash[:8]}: {status} ({execution_time:.1f}s)")
            
            return result_data
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            logger.error(f"[{commit_index+1}/{total_commits}] {commit_hash[:8]}: ⏰ TIMEOUT after {execution_time:.1f}s")
            
            return {
                "commit_hash": commit_hash,
                "commit_file": str(commit_file),
                "test_script": str(test_script),
                "success": False,
                "execution_time_seconds": execution_time,
                "return_code": -1,
                "stdout": "",
                "stderr": "Process timed out after 30 minutes",
                "gso_prediction": None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[{commit_index+1}/{total_commits}] {commit_hash[:8]}: 💥 ERROR - {e}")
            
            return {
                "commit_hash": commit_hash,
                "commit_file": str(commit_file),
                "test_script": str(test_script),
                "success": False,
                "execution_time_seconds": execution_time,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "gso_prediction": None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }
    
    def run_batch(
        self, 
        commit_dir: Path, 
        test_dir: Path, 
        repo_path: Path,
        resume: bool = False
    ) -> BatchResult:
        """Run batch optimization on all found commit pairs."""
        
        logger.info("Starting batch commit optimization")
        logger.info(f"Commit directory: {commit_dir}")
        logger.info(f"Test directory: {test_dir}")
        logger.info(f"Repository: {repo_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Max workers: {self.max_workers}")
        
        # Find commit pairs
        pairs = self.find_commit_pairs(commit_dir, test_dir)
        if not pairs:
            logger.error("No commit-test pairs found!")
            return BatchResult()
        
        logger.info(f"Found {len(pairs)} commit-test pairs")
        
        # Check for existing results if resuming
        processed_hashes = set()
        if resume and self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    existing_results = json.load(f)
                processed_hashes = {r["commit_hash"] for r in existing_results.get("individual_results", [])}
                logger.info(f"Resume mode: skipping {len(processed_hashes)} already processed commits")
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")
        
        # Filter out already processed commits
        if resume:
            pairs = [(c, t) for c, t in pairs if c.stem.replace(".json", "") not in processed_hashes]
            logger.info(f"Processing {len(pairs)} remaining commits")
        
        if not pairs:
            logger.info("No new commits to process")
            return BatchResult()
        
        # Process commits
        batch_start = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_commit = {
                executor.submit(
                    self.process_single_commit, 
                    commit_file, 
                    test_script, 
                    repo_path,
                    i,
                    len(pairs)
                ): (commit_file, test_script) 
                for i, (commit_file, test_script) in enumerate(pairs)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_commit):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Save incremental results
                    self._save_incremental_results(results)
                    
                except Exception as e:
                    commit_file, test_script = future_to_commit[future]
                    logger.error(f"Unexpected error processing {commit_file.stem}: {e}")
        
        batch_execution_time = time.time() - batch_start
        
        # Calculate final statistics
        batch_result = BatchResult(
            total_commits=len(pairs),
            successful_optimizations=sum(1 for r in results if r["success"]),
            failed_optimizations=sum(1 for r in results if not r["success"]),
            execution_time_seconds=batch_execution_time,
            individual_results=results
        )
        
        # Save final results
        self._save_final_results(batch_result)
        
        # Print summary
        self._print_summary(batch_result)
        
        return batch_result
    
    def _save_incremental_results(self, results: List[Dict[str, Any]]):
        """Save incremental results during processing."""
        try:
            with self.lock:
                temp_file = self.results_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump({
                        "individual_results": results,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                        "status": "in_progress"
                    }, f, indent=2)
                
                # Atomic move
                temp_file.replace(self.results_file)
        except Exception as e:
            logger.warning(f"Failed to save incremental results: {e}")
    
    def _save_final_results(self, batch_result: BatchResult):
        """Save final batch results."""
        final_data = {
            "summary": {
                "total_commits": batch_result.total_commits,
                "successful_optimizations": batch_result.successful_optimizations,
                "failed_optimizations": batch_result.failed_optimizations,
                "success_rate_percent": batch_result.success_rate,
                "total_execution_time_seconds": batch_result.execution_time_seconds
            },
            "individual_results": batch_result.individual_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "status": "completed"
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        logger.info(f"Final results saved to: {self.results_file}")
    
    def _print_summary(self, batch_result: BatchResult):
        """Print batch processing summary."""
        print("\n" + "="*70)
        print("BATCH COMMIT OPTIMIZATION SUMMARY")
        print("="*70)
        print(f"Total commits processed: {batch_result.total_commits}")
        print(f"Successful optimizations: {batch_result.successful_optimizations}")
        print(f"Failed optimizations: {batch_result.failed_optimizations}")
        print(f"Success rate: {batch_result.success_rate:.1f}%")
        print(f"Total execution time: {batch_result.execution_time_seconds:.1f}s")
        print(f"Average time per commit: {batch_result.execution_time_seconds/max(1, batch_result.total_commits):.1f}s")
        print()
        
        if batch_result.successful_optimizations > 0:
            print(f"✅ {batch_result.successful_optimizations} commits successfully optimized")
        if batch_result.failed_optimizations > 0:
            print(f"❌ {batch_result.failed_optimizations} commits failed optimization")
        
        print(f"\n📁 Results saved in: {self.output_dir}")
        print(f"📊 Detailed results: {self.results_file}")


def main():
    parser = argparse.ArgumentParser(description="Batch commit optimization pipeline")
    parser.add_argument("--commit-dir", required=True, type=Path,
                       help="Directory containing commit JSON files")
    parser.add_argument("--test-dir", required=True, type=Path,
                       help="Directory containing generated test scripts")
    parser.add_argument("--repo-path", required=True, type=Path,
                       help="Path to repository to optimize")
    parser.add_argument("--output-dir", required=True, type=Path,
                       help="Output directory for results")
    parser.add_argument("--max-workers", type=int, default=2,
                       help="Maximum number of parallel workers (default: 2)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume interrupted batch processing")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.commit_dir.exists():
        logger.error(f"Commit directory not found: {args.commit_dir}")
        return 1
        
    if not args.test_dir.exists():
        logger.error(f"Test directory not found: {args.test_dir}")
        return 1
        
    if not args.repo_path.exists():
        logger.error(f"Repository not found: {args.repo_path}")
        return 1
    
    # Create batch optimizer
    optimizer = BatchCommitOptimizer(args.output_dir, args.max_workers)
    
    try:
        result = optimizer.run_batch(
            args.commit_dir, 
            args.test_dir, 
            args.repo_path,
            args.resume
        )
        
        return 0 if result.success_rate > 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Batch processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
