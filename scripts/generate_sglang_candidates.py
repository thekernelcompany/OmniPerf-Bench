#!/usr/bin/env python3
"""
Generate SGLang 3-Way Benchmark Candidates JSON

Generates /tmp/sglang_3way_candidates.json by:
1. Fetching available Docker image tags from shikhar481/sglang-images
2. Scanning Claude Code runs for patches and commit info
3. Loading perf_command from extraction files
4. Filtering to candidates where BOTH human AND parent images exist

Data Sources:
- Claude Code runs: perf-agents-bench/state/runs/sglang/claude_code/default/2025-12-23_06-28-44/*/journal.json
- Commit extractions: misc/experiments/sglang_commit_extractions_with_apis/*.json
- Docker Hub API: https://hub.docker.com/v2/repositories/shikhar481/sglang-images/tags

Usage:
    python scripts/generate_sglang_candidates.py
    python scripts/generate_sglang_candidates.py --output /path/to/output.json
    python scripts/generate_sglang_candidates.py --dry-run
"""

import argparse
import json
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
DOCKER_REPO = "shikhar481/sglang-images"
CLAUDE_CODE_RUNS_DIR = Path("perf-agents-bench/state/runs/sglang/claude_code/default/2025-12-23_06-28-44")
COMMIT_EXTRACTIONS_DIR = Path("misc/experiments/sglang_commit_extractions_with_apis")
DEFAULT_OUTPUT = Path("/tmp/sglang_3way_candidates.json")


def fetch_docker_tags(repo: str = DOCKER_REPO, max_pages: int = 10) -> Set[str]:
    """
    Fetch all available Docker image tags from DockerHub.

    Returns set of short commit hashes (8 chars) that have images.
    """
    tags = set()
    page = 1

    logger.info(f"Fetching Docker tags from {repo}...")

    while page <= max_pages:
        url = f"https://hub.docker.com/v2/repositories/{repo}/tags?page={page}&page_size=100"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read())
                results = data.get('results', [])

                if not results:
                    break

                for tag in results:
                    name = tag.get('name', '')
                    if len(name) >= 8 and name.isalnum():
                        # Store both short (8 char) and full tag
                        tags.add(name[:8])
                        if len(name) > 8:
                            tags.add(name)  # Also store full hash if available

                # Check for next page
                if not data.get('next'):
                    break
                page += 1

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning(f"Repository not found: {repo}")
            else:
                logger.error(f"HTTP error fetching tags: {e}")
            break
        except Exception as e:
            logger.error(f"Error fetching Docker tags: {e}")
            break

    logger.info(f"Found {len(tags)} Docker image tags")
    return tags


def load_claude_code_runs() -> Dict[str, Dict]:
    """
    Load Claude Code run information from journal.json files.

    Returns dict mapping short commit hash to run info.
    """
    runs = {}

    if not CLAUDE_CODE_RUNS_DIR.exists():
        logger.warning(f"Claude Code runs directory not found: {CLAUDE_CODE_RUNS_DIR}")
        return runs

    for run_dir in CLAUDE_CODE_RUNS_DIR.glob("sglang_*"):
        journal_file = run_dir / "journal.json"
        patch_file = run_dir / "model_patch.diff"

        if not journal_file.exists():
            continue

        try:
            with open(journal_file) as f:
                journal = json.load(f)

            commits = journal.get("commits", {})
            human_commit = commits.get("human", "")
            parent_commit = commits.get("pre", "")

            if not human_commit:
                continue

            short_hash = human_commit[:8]

            run_info = {
                "human_full": human_commit,
                "parent_full": parent_commit,
                "human": short_hash,
                "parent": parent_commit[:8] if parent_commit else "",
                "status": journal.get("status", ""),
                "patch_path": str(patch_file) if patch_file.exists() else None,
                "run_dir": str(run_dir),
                "metrics": journal.get("metrics", {}),
            }

            runs[short_hash] = run_info

        except Exception as e:
            logger.warning(f"Error loading {journal_file}: {e}")

    logger.info(f"Loaded {len(runs)} Claude Code runs")
    return runs


def load_commit_extractions() -> Dict[str, Dict]:
    """
    Load commit extraction data with perf_command and other metadata.

    Returns dict mapping short commit hash to extraction data.
    """
    extractions = {}

    if not COMMIT_EXTRACTIONS_DIR.exists():
        logger.warning(f"Commit extractions directory not found: {COMMIT_EXTRACTIONS_DIR}")
        return extractions

    for extraction_file in COMMIT_EXTRACTIONS_DIR.glob("*.json"):
        try:
            with open(extraction_file) as f:
                data = json.load(f)

            commit_hash = data.get("commit_hash", "")
            if not commit_hash:
                continue

            short_hash = commit_hash[:8]

            extractions[short_hash] = {
                "commit_hash": commit_hash,
                "commit_subject": data.get("commit_subject", ""),
                "perf_command": data.get("perf_command", ""),
                "models": data.get("models", []),
                "files_changed": data.get("files_changed", []),
                "pr_url": data.get("pr_url", ""),
                "has_serving": data.get("has_serving", False),
                "has_performance": data.get("has_performance", False),
            }

        except Exception as e:
            logger.warning(f"Error loading {extraction_file}: {e}")

    logger.info(f"Loaded {len(extractions)} commit extractions")
    return extractions


def extract_model_from_perf_command(perf_command: str) -> Optional[str]:
    """Extract model name from perf_command if present."""
    # Look for --model-path or --model argument
    import re

    patterns = [
        r'--model-path\s+([^\s]+)',
        r'--model\s+([^\s]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, perf_command)
        if match:
            return match.group(1)

    return None


def get_lora_server_args(perf_command: str, extraction: Dict) -> str:
    """
    Get LoRA server arguments if this commit requires LoRA.

    Some commits (like 021f76e4) use LoRA and need special server args.
    """
    # Check if perf_command mentions lora
    if "lora" in perf_command.lower():
        # Look for common LoRA models
        if "lora-name lora" in perf_command:
            return "--lora-paths lora=algoprog/fact-generation-llama-3.1-8b-instruct-lora"
    return ""


def generate_candidates(
    docker_tags: Set[str],
    runs: Dict[str, Dict],
    extractions: Dict[str, Dict],
) -> List[Dict]:
    """
    Generate benchmark candidates by combining data from all sources.

    Only includes candidates where BOTH human AND parent images exist.
    """
    candidates = []

    # Find candidates with both images
    for short_hash, run in runs.items():
        human_tag = short_hash
        parent_tag = run.get("parent", "")

        # Check if both images exist
        if human_tag not in docker_tags:
            logger.debug(f"Skipping {short_hash}: human image not found")
            continue

        if parent_tag and parent_tag not in docker_tags:
            logger.debug(f"Skipping {short_hash}: parent image {parent_tag} not found")
            continue

        # Get extraction data
        extraction = extractions.get(short_hash, {})

        # Get perf_command
        perf_command = extraction.get("perf_command", "")
        if not perf_command:
            logger.debug(f"Skipping {short_hash}: no perf_command")
            continue

        # Get model - from extraction or infer from perf_command
        models = extraction.get("models", [])
        model = models[0] if models else extract_model_from_perf_command(perf_command)
        if not model:
            model = "meta-llama/Llama-3.1-8B-Instruct"  # Default model

        # Get LoRA server args if needed
        lora_server_args = get_lora_server_args(perf_command, extraction)

        candidate = {
            "human": human_tag,
            "parent": parent_tag,
            "human_full": run.get("human_full", ""),
            "parent_full": run.get("parent_full", ""),
            "model": model,
            "perf_command": perf_command,
            "patch_path": run.get("patch_path"),
            "subject": extraction.get("commit_subject", ""),
            "files_changed": extraction.get("files_changed", []),
            "pr_url": extraction.get("pr_url", ""),
            "lora_server_args": lora_server_args,
            "has_serving": extraction.get("has_serving", False),
            "has_performance": extraction.get("has_performance", False),
        }

        candidates.append(candidate)

    # Sort by human commit hash for deterministic order
    candidates.sort(key=lambda x: x["human"])

    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Generate SGLang 3-way benchmark candidates JSON"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output file path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing file"
    )
    parser.add_argument(
        "--skip-docker-check",
        action="store_true",
        help="Skip Docker tag check (use cached/all runs)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load data from all sources
    logger.info("Loading Claude Code runs...")
    runs = load_claude_code_runs()

    logger.info("Loading commit extractions...")
    extractions = load_commit_extractions()

    # Fetch Docker tags (or use all runs if skipping check)
    if args.skip_docker_check:
        # Assume all images exist
        docker_tags = set(runs.keys()) | set(r.get("parent", "") for r in runs.values())
        logger.info(f"Skipping Docker check, assuming {len(docker_tags)} tags exist")
    else:
        docker_tags = fetch_docker_tags()

    # Generate candidates
    logger.info("Generating candidates...")
    candidates = generate_candidates(docker_tags, runs, extractions)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("CANDIDATE GENERATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Claude Code runs:     {len(runs)}")
    logger.info(f"  Commit extractions:   {len(extractions)}")
    logger.info(f"  Docker images:        {len(docker_tags)}")
    logger.info(f"  Valid candidates:     {len(candidates)}")
    logger.info(f"{'='*60}")

    if args.dry_run:
        print("\n=== DRY RUN - Would generate these candidates ===\n")
        for i, c in enumerate(candidates, 1):
            print(f"{i}. {c['human']} (parent: {c['parent']})")
            print(f"   Subject: {c['subject'][:60]}...")
            print(f"   Model: {c['model']}")
            print(f"   Perf command: {c['perf_command'][:60]}...")
            if c.get('lora_server_args'):
                print(f"   LoRA args: {c['lora_server_args']}")
            print()
        return

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(candidates, f, indent=2)

    logger.info(f"Wrote {len(candidates)} candidates to {output_path}")

    # Print sample candidates
    if candidates:
        print("\nSample candidates:")
        for c in candidates[:3]:
            print(f"  - {c['human']}: {c['subject'][:50]}...")


if __name__ == "__main__":
    main()
