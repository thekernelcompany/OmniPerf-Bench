from pathlib import Path
import subprocess
import json
import sys
import os

# Import the TestPack contract
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "perf-agents-bench"))
from bench.testpack_api.contract import TestPack


class GenericPack(TestPack):
    schema_version = "1.0.0"

    def prepare_fixtures(self, repo_dir: Path, work_dir: Path) -> None:
        """Prepare test fixtures - example creates a small dataset."""
        fixtures_dir = work_dir / "fixtures"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        
        # Example: create a small test dataset
        test_data = {
            "images": [f"image_{i}.jpg" for i in range(100)],
            "metadata": {"count": 100, "type": "validation"}
        }
        
        (fixtures_dir / "test_data.json").write_text(json.dumps(test_data, indent=2))
        print(f"Prepared fixtures in {fixtures_dir}")

    def build(self, repo_dir: Path, work_dir: Path) -> None:
        """Build the repository if it's a package."""
        if (repo_dir / "pyproject.toml").exists() or (repo_dir / "setup.py").exists():
            print("Installing repository as editable package...")
            subprocess.run(
                ["python", "-m", "pip", "install", "-e", "."], 
                cwd=repo_dir, 
                check=True
            )
        else:
            print("No package to install - repository does not have pyproject.toml or setup.py")

    def run_candidate(self, repo_dir: Path, work_dir: Path, out_dir: Path, candidate_tag: str) -> None:
        """Run performance test for the given candidate."""
        print(f"Running candidate: {candidate_tag}")
        
        # Example benchmark execution - adapt this to your specific use case
        benchmark_script = repo_dir / "benchmark.py"
        if benchmark_script.exists():
            # Run the repository's benchmark script
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_dir)
            
            cmd = [
                "python", str(benchmark_script),
                "--input", str(work_dir / "fixtures" / "test_data.json"),
                "--output", str(out_dir / f"{candidate_tag}.json")
            ]
            
            result = subprocess.run(cmd, cwd=repo_dir, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Benchmark failed for {candidate_tag}: {result.stderr}")
                # Create a default output to prevent metrics from failing
                default_output = {
                    "throughput": 0.0,
                    "error": f"Benchmark failed: {result.stderr}",
                    "candidate": candidate_tag
                }
                (out_dir / f"{candidate_tag}.json").write_text(json.dumps(default_output))
            else:
                print(f"Benchmark completed for {candidate_tag}")
        else:
            # Fallback: create mock performance data
            print(f"No benchmark.py found, creating mock data for {candidate_tag}")
            mock_data = {
                "throughput": 100.0 if candidate_tag == "human" else 95.0,
                "candidate": candidate_tag,
                "note": "Mock data - no benchmark script found"
            }
            (out_dir / f"{candidate_tag}.json").write_text(json.dumps(mock_data, indent=2))

    def discover_entrypoints(self):
        """Provide hints for metrics configuration."""
        return {
            "profile_cmd": "python benchmark.py --input fixtures/test_data.json --profile"
        }


def pack():
    """Factory function to create TestPack instance."""
    return GenericPack()