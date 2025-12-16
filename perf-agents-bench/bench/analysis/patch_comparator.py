"""Patch comparison utilities for agent vs human patch analysis.

Compares unified diff patches to measure how closely an agent's solution
matches the human reference implementation.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class PatchParser:
    """Parse unified diff patches into structured format.

    Extracts files changed, lines added/removed, and hunk details
    from standard unified diff format.
    """

    def __init__(self, patch_content: str):
        """Initialize with patch content.

        Args:
            patch_content: Unified diff string
        """
        self.content = patch_content
        self._parsed: Optional[Dict[str, Any]] = None

    def parse(self) -> Dict[str, Any]:
        """Parse the patch into structured format.

        Returns:
            Dict with:
                - files: List of file paths changed
                - hunks: List of hunk dicts
                - lines_added: Total lines added
                - lines_removed: Total lines removed
                - additions: List of added line contents
                - removals: List of removed line contents
        """
        if self._parsed is not None:
            return self._parsed

        if not self.content:
            self._parsed = {
                "files": [],
                "hunks": [],
                "lines_added": 0,
                "lines_removed": 0,
                "additions": [],
                "removals": [],
            }
            return self._parsed

        files: Set[str] = set()
        hunks: List[Dict[str, Any]] = []
        additions: List[str] = []
        removals: List[str] = []

        current_file: Optional[str] = None
        current_hunk: Optional[Dict[str, Any]] = None

        for line in self.content.split("\n"):
            # File header lines
            if line.startswith("--- "):
                # Extract file path (ignore /dev/null)
                parts = line[4:].split("\t")[0].split(" ")[0]
                if parts != "/dev/null":
                    # Remove a/ prefix if present
                    if parts.startswith("a/"):
                        parts = parts[2:]
                    current_file = parts

            elif line.startswith("+++ "):
                parts = line[4:].split("\t")[0].split(" ")[0]
                if parts != "/dev/null":
                    if parts.startswith("b/"):
                        parts = parts[2:]
                    if current_file:
                        files.add(current_file)
                    current_file = parts
                    files.add(parts)

            # Hunk header
            elif line.startswith("@@"):
                if current_hunk:
                    hunks.append(current_hunk)

                # Parse hunk header: @@ -start,count +start,count @@
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if match:
                    current_hunk = {
                        "file": current_file,
                        "old_start": int(match.group(1)),
                        "old_count": int(match.group(2) or 1),
                        "new_start": int(match.group(3)),
                        "new_count": int(match.group(4) or 1),
                        "additions": [],
                        "removals": [],
                    }
                else:
                    current_hunk = {
                        "file": current_file,
                        "additions": [],
                        "removals": [],
                    }

            # Content lines
            elif line.startswith("+") and not line.startswith("+++"):
                content = line[1:]
                additions.append(content)
                if current_hunk:
                    current_hunk["additions"].append(content)

            elif line.startswith("-") and not line.startswith("---"):
                content = line[1:]
                removals.append(content)
                if current_hunk:
                    current_hunk["removals"].append(content)

        # Add final hunk
        if current_hunk:
            hunks.append(current_hunk)

        self._parsed = {
            "files": sorted(files),
            "hunks": hunks,
            "lines_added": len(additions),
            "lines_removed": len(removals),
            "additions": additions,
            "removals": removals,
        }

        return self._parsed

    @property
    def files(self) -> List[str]:
        """Get list of files changed."""
        return self.parse()["files"]

    @property
    def lines_added(self) -> int:
        """Get count of lines added."""
        return self.parse()["lines_added"]

    @property
    def lines_removed(self) -> int:
        """Get count of lines removed."""
        return self.parse()["lines_removed"]

    @property
    def additions(self) -> List[str]:
        """Get list of added line contents."""
        return self.parse()["additions"]

    @property
    def removals(self) -> List[str]:
        """Get list of removed line contents."""
        return self.parse()["removals"]


class PatchComparator:
    """Compare agent patch against human reference patch.

    Calculates file overlap, line overlap, and semantic similarity
    between two patches.
    """

    def __init__(self, agent_patch: str, human_patch: str):
        """Initialize with both patches.

        Args:
            agent_patch: Agent's generated patch (unified diff)
            human_patch: Human reference patch (unified diff)
        """
        self.agent = PatchParser(agent_patch)
        self.human = PatchParser(human_patch)

    def compare(self) -> Dict[str, Any]:
        """Compare patches and return similarity metrics.

        Returns:
            Dict matching PatchSimilarityMetrics schema fields
        """
        # File-level comparison
        agent_files = set(self.agent.files)
        human_files = set(self.human.files)
        common_files = agent_files & human_files
        agent_only = agent_files - human_files
        human_only = human_files - agent_files

        # File overlap percentage
        file_overlap_pct = 0.0
        if human_files:
            file_overlap_pct = (len(common_files) / len(human_files)) * 100

        # Line-level metrics
        agent_lines_added = self.agent.lines_added
        agent_lines_removed = self.agent.lines_removed
        human_lines_added = self.human.lines_added
        human_lines_removed = self.human.lines_removed

        # Calculate matching lines
        matching_additions = self._count_matching_lines(
            self.agent.additions, self.human.additions
        )
        matching_removals = self._count_matching_lines(
            self.agent.removals, self.human.removals
        )

        # Line overlap percentage (based on human's changes)
        line_overlap_pct = 0.0
        total_human_changes = human_lines_added + human_lines_removed
        if total_human_changes > 0:
            total_matching = matching_additions + matching_removals
            line_overlap_pct = (total_matching / total_human_changes) * 100

        # Semantic similarity score
        approach_similarity = self._calculate_semantic_similarity()

        return {
            "human_patch_available": bool(self.human.content),
            "human_patch_source": "",  # Set by caller
            "agent_files": sorted(agent_files),
            "human_files": sorted(human_files),
            "common_files": sorted(common_files),
            "agent_only_files": sorted(agent_only),
            "human_only_files": sorted(human_only),
            "file_overlap_pct": round(file_overlap_pct, 2),
            "agent_lines_added": agent_lines_added,
            "agent_lines_removed": agent_lines_removed,
            "human_lines_added": human_lines_added,
            "human_lines_removed": human_lines_removed,
            "matching_additions": matching_additions,
            "matching_removals": matching_removals,
            "line_overlap_pct": round(line_overlap_pct, 2),
            "approach_similarity_score": round(approach_similarity, 2),
        }

    def _count_matching_lines(
        self, agent_lines: List[str], human_lines: List[str]
    ) -> int:
        """Count how many agent lines match human lines.

        Uses fuzzy matching to handle minor differences.

        Args:
            agent_lines: Lines from agent patch
            human_lines: Lines from human patch

        Returns:
            Count of matching lines
        """
        if not agent_lines or not human_lines:
            return 0

        # Normalize lines for comparison
        def normalize(line: str) -> str:
            return line.strip().lower()

        human_normalized = {normalize(line) for line in human_lines if line.strip()}
        agent_normalized = [normalize(line) for line in agent_lines if line.strip()]

        # Exact matches
        exact_matches = sum(1 for line in agent_normalized if line in human_normalized)

        # Fuzzy matches for remaining lines
        fuzzy_matches = 0
        unmatched_human = human_normalized - {
            line for line in agent_normalized if line in human_normalized
        }
        unmatched_agent = [
            line for line in agent_normalized if line not in human_normalized
        ]

        for agent_line in unmatched_agent:
            for human_line in unmatched_human:
                ratio = SequenceMatcher(None, agent_line, human_line).ratio()
                if ratio > 0.8:  # 80% similarity threshold
                    fuzzy_matches += 1
                    unmatched_human.discard(human_line)
                    break

        return exact_matches + fuzzy_matches

    def _calculate_semantic_similarity(self) -> float:
        """Calculate semantic similarity between patches.

        Uses multiple signals:
        - File overlap
        - Function/class name overlap
        - Code pattern similarity

        Returns:
            Similarity score (0-10)
        """
        if not self.human.content or not self.agent.content:
            return 0.0

        # File similarity (weighted 30%)
        agent_files = set(self.agent.files)
        human_files = set(self.human.files)
        file_sim = 0.0
        if human_files:
            file_sim = len(agent_files & human_files) / len(human_files)

        # Line content similarity (weighted 40%)
        all_agent_lines = "\n".join(self.agent.additions + self.agent.removals)
        all_human_lines = "\n".join(self.human.additions + self.human.removals)
        content_sim = SequenceMatcher(None, all_agent_lines, all_human_lines).ratio()

        # Code pattern similarity (weighted 30%)
        pattern_sim = self._calculate_pattern_similarity()

        # Weighted combination
        similarity = (file_sim * 0.3 + content_sim * 0.4 + pattern_sim * 0.3) * 10

        return min(10.0, max(0.0, similarity))

    def _calculate_pattern_similarity(self) -> float:
        """Calculate similarity based on code patterns.

        Looks for:
        - Function/method definitions
        - Import statements
        - Variable assignments
        - Code structures

        Returns:
            Pattern similarity (0-1)
        """
        patterns = [
            r"def\s+(\w+)",  # Function definitions
            r"class\s+(\w+)",  # Class definitions
            r"import\s+(\w+)",  # Imports
            r"from\s+(\w+)",  # From imports
            r"(\w+)\s*=",  # Assignments (first identifier)
        ]

        agent_content = "\n".join(self.agent.additions)
        human_content = "\n".join(self.human.additions)

        agent_patterns: Set[str] = set()
        human_patterns: Set[str] = set()

        for pattern in patterns:
            agent_patterns.update(re.findall(pattern, agent_content))
            human_patterns.update(re.findall(pattern, human_content))

        if not human_patterns:
            return 0.0

        overlap = len(agent_patterns & human_patterns)
        return overlap / len(human_patterns)


def load_human_patch_from_dataset(
    dataset_path: Path, commit_hash: str
) -> Tuple[str, str]:
    """Load human patch from dataset file by commit hash.

    Searches JSONL dataset for matching commit and extracts patch.

    Args:
        dataset_path: Path to .jsonl dataset file
        commit_hash: Git commit hash to match

    Returns:
        Tuple of (patch_content, source_description)
    """
    if not dataset_path.exists():
        logger.debug(f"Dataset file not found: {dataset_path}")
        return "", ""

    try:
        with open(dataset_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Check various commit fields
                item_commit = (
                    item.get("head_commit", "")
                    or item.get("commit_hash", "")
                    or item.get("commit", "")
                )

                # Match by commit hash (prefix match for short hashes)
                if item_commit and (
                    item_commit.startswith(commit_hash[:7])
                    or commit_hash.startswith(item_commit[:7])
                ):
                    patch = item.get("patch", "")
                    if patch:
                        source = f"{dataset_path.name}:line{line_num}"
                        return patch, source

        return "", ""

    except Exception as e:
        logger.warning(f"Failed to load patch from {dataset_path}: {e}")
        return "", ""


def find_dataset_for_repo(repo: str, data_dir: Path) -> Optional[Path]:
    """Find the dataset file for a repository.

    Args:
        repo: Repository name (e.g., 'vllm', 'sglang')
        data_dir: Base data directory

    Returns:
        Path to dataset file or None
    """
    # Try various naming patterns
    patterns = [
        f"{repo}_final_dataset.jsonl",
        f"{repo}_dataset.jsonl",
        f"{repo}_dataset_with_test.jsonl",
        f"{repo}.jsonl",
    ]

    # Search in data directory and common subdirectories
    search_dirs = [
        data_dir,
        data_dir / "final",
        data_dir / "datasets",
        data_dir.parent / "data",
        data_dir.parent / "data" / "final",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for pattern in patterns:
            candidate = search_dir / pattern
            if candidate.exists():
                return candidate

    return None
