"""
Download and index test scripts from HuggingFace dataset.

Downloads Inferencebench/test-generation-scripts and builds an index
mapping commit hashes to test script paths.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Default paths - using Ayushnangia/omniperf_v1 extracted test scripts
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "hf_cache" / "omniperf_v1_clone" / "test_scripts"
DEFAULT_INDEX_PATH = DEFAULT_CACHE_DIR / "commit_index.json"
HF_DATASET_ID = "Ayushnangia/omniperf_v1"


def download_and_index_tests(
    cache_dir: Optional[Path] = None,
    force_redownload: bool = False,
) -> Dict[str, str]:
    """
    Download test scripts from HuggingFace and build commit hash index.

    Args:
        cache_dir: Directory to store downloaded scripts
        force_redownload: If True, re-download even if cache exists

    Returns:
        Dict mapping commit_hash -> script_path
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    index_path = cache_dir / "commit_index.json"

    # Check if already downloaded
    if not force_redownload and index_path.exists():
        logger.info(f"Loading existing index from {index_path}")
        return load_test_index(index_path)

    logger.info(f"Downloading test scripts from {HF_DATASET_ID}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: uv pip install huggingface_hub"
        )

    # Download the dataset repository
    repo_path = snapshot_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        local_dir=cache_dir / "repo",
        local_dir_use_symlinks=False,
    )

    logger.info(f"Downloaded to {repo_path}")

    # Build index from downloaded scripts
    index = _build_commit_index(Path(repo_path))

    # Save index
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"Built index with {len(index)} commit mappings")

    return index


def _build_commit_index(repo_path: Path) -> Dict[str, str]:
    """
    Parse all Python scripts to extract commit hashes and build index.

    Looks for commit_hash patterns like:
    - commit_hash = os.getenv("PROB_COMMIT_HASH", "8d75fe48...")
    - PROB_COMMIT_HASH = "8d75fe48..."
    - "commit_hash": "8d75fe48..."
    """
    index: Dict[str, str] = {}

    # Search patterns for commit hashes
    patterns = [
        # os.getenv("PROB_COMMIT_HASH", "abc123...")
        re.compile(r'os\.getenv\(["\']PROB_COMMIT_HASH["\'],\s*["\']([a-f0-9]{7,40})["\']'),
        # commit_hash = "abc123..."
        re.compile(r'commit_hash\s*=\s*["\']([a-f0-9]{7,40})["\']'),
        # "commit_hash": "abc123..."
        re.compile(r'["\']commit_hash["\']\s*:\s*["\']([a-f0-9]{7,40})["\']'),
        # PROB_COMMIT_HASH = "abc123..."
        re.compile(r'PROB_COMMIT_HASH\s*=\s*["\']([a-f0-9]{7,40})["\']'),
    ]

    # Scan all .py files in the repo
    script_dirs = [
        repo_path / "generated_test_generators_v4",
        repo_path / "working_test_generators",
    ]

    for script_dir in script_dirs:
        if not script_dir.exists():
            logger.warning(f"Script directory not found: {script_dir}")
            continue

        for py_file in script_dir.glob("*.py"):
            try:
                content = py_file.read_text()

                # Try each pattern
                commit_hash = None
                for pattern in patterns:
                    match = pattern.search(content)
                    if match:
                        commit_hash = match.group(1)
                        break

                # Also try to extract from filename (e.g., 8d75fe48_test_generator.py)
                if not commit_hash:
                    filename_match = re.match(r'^([a-f0-9]{7,40})', py_file.stem)
                    if filename_match:
                        commit_hash = filename_match.group(1)

                if commit_hash:
                    # Store both full hash and short hash (8 chars)
                    full_path = str(py_file.absolute())
                    index[commit_hash] = full_path
                    if len(commit_hash) > 8:
                        index[commit_hash[:8]] = full_path
                    logger.debug(f"Indexed {py_file.name} -> {commit_hash[:8]}")
                else:
                    logger.warning(f"No commit hash found in {py_file.name}")

            except Exception as e:
                logger.error(f"Error parsing {py_file}: {e}")

    return index


def load_test_index(index_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Load the commit->script index from disk.

    Args:
        index_path: Path to index JSON file

    Returns:
        Dict mapping commit_hash -> script_path
    """
    index_path = Path(index_path) if index_path else DEFAULT_INDEX_PATH

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found at {index_path}. Run download_and_index_tests() first."
        )

    with open(index_path) as f:
        return json.load(f)


def find_test_script(commit_hash: str, index: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Find test script path for a given commit hash.

    Args:
        commit_hash: Full or short (8-char) commit hash
        index: Pre-loaded index, or None to load from disk

    Returns:
        Path to test script, or None if not found
    """
    if index is None:
        try:
            index = load_test_index()
        except FileNotFoundError:
            return None

    # Try full hash first
    if commit_hash in index:
        return index[commit_hash]

    # Try short hash (8 chars)
    short_hash = commit_hash[:8]
    if short_hash in index:
        return index[short_hash]

    # Try prefix matching
    for key, path in index.items():
        if key.startswith(commit_hash) or commit_hash.startswith(key):
            return path

    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Downloading and indexing test scripts...")
    index = download_and_index_tests()

    print(f"\nBuilt index with {len(index)} entries")
    print("\nSample entries:")
    for i, (k, v) in enumerate(list(index.items())[:5]):
        print(f"  {k[:8]}... -> {Path(v).name}")
