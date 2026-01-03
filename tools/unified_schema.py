#!/usr/bin/env python3
"""
Unified schema definition for vLLM and SGLang benchmark datasets.

This module defines the canonical schema for HuggingFace datasets to ensure
consistency between vLLM and SGLang benchmark results.

Target: 38 columns (after removing improvement columns)
"""

from typing import Dict, Any, Optional
from datetime import date

# Unified column schema
# Ordered list of columns in the final dataset
UNIFIED_COLUMNS = [
    # Metadata (14 columns)
    "commit_hash",
    "commit_subject",
    "repo",
    "model",
    "gpu_config",
    "benchmark_mode",
    "perf_command",
    "status",
    "error",
    "duration_s",
    "has_agent_patch",
    "agent_name",
    "agent_model",
    "benchmark_date",

    # Latency metrics - Baseline (4 columns)
    "baseline_ttft_mean",
    "baseline_ttft_median",
    "baseline_tpot_mean",
    "baseline_itl_mean",

    # Latency metrics - Human (4 columns)
    "human_ttft_mean",
    "human_ttft_median",
    "human_tpot_mean",
    "human_itl_mean",

    # Latency metrics - Agent (4 columns)
    "agent_ttft_mean",
    "agent_ttft_median",
    "agent_tpot_mean",
    "agent_itl_mean",

    # Throughput metrics (6 columns)
    "baseline_throughput",
    "baseline_output_throughput",
    "human_throughput",
    "human_output_throughput",
    "agent_throughput",
    "agent_output_throughput",

    # E2E Latency metrics (6 columns)
    "baseline_e2e_latency_mean",
    "baseline_e2e_latency_median",
    "human_e2e_latency_mean",
    "human_e2e_latency_median",
    "agent_e2e_latency_mean",
    "agent_e2e_latency_median",
]

# Columns that were REMOVED from the schema
REMOVED_COLUMNS = [
    # Improvement columns (calculated at analysis time, not stored)
    "human_improvement_tpot_mean",
    "agent_improvement_tpot_mean",
    "agent_vs_human_tpot_mean",
    "human_improvement_throughput",
    "agent_improvement_throughput",
    "human_improvement_ttft_mean",
    "agent_improvement_ttft_mean",
    "human_improvement_itl_mean",
    "agent_improvement_itl_mean",
    "human_improvement_latency_avg",
    "agent_improvement_latency_avg",
    "agent_vs_human_ttft_mean",
    "agent_vs_human_itl_mean",
    "agent_vs_human_throughput",
    "agent_vs_human_latency_avg",

    # P99 columns (SGLang only, removed for consistency)
    "human_ttft_p99",
    "human_tpot_p99",
    "human_itl_p99",
    "baseline_ttft_p99",
    "baseline_tpot_p99",
    "baseline_itl_p99",
    "agent_ttft_p99",
    "agent_tpot_p99",
    "agent_itl_p99",

    # Other removed columns
    "parent_commit",
    "human_input_throughput",
    "install_method",  # Ignored per user request
    "baseline_install_method",
    "human_install_method",
    "agent_install_method",
    "patch_type",
]

# Column name mappings (old name -> new name)
COLUMN_RENAMES = {
    # SGLang throughput renaming
    "human_request_throughput": "human_throughput",
    "baseline_request_throughput": "baseline_throughput",
    "agent_request_throughput": "agent_throughput",
    # vLLM latency_avg -> throughput for consistency (if needed)
    "baseline_latency_avg": None,  # Drop this column
    "human_latency_avg": None,  # Drop this column
    "agent_latency_avg": None,  # Drop this column
}


def create_empty_row() -> Dict[str, Any]:
    """Create an empty row with all unified columns set to None."""
    return {col: None for col in UNIFIED_COLUMNS}


def normalize_row(row: Dict[str, Any], source: str = "unknown") -> Dict[str, Any]:
    """
    Normalize a row to the unified schema.

    Args:
        row: Raw row data from source
        source: Source identifier ("vllm" or "sglang")

    Returns:
        Normalized row with unified column names
    """
    normalized = create_empty_row()

    for key, value in row.items():
        # Skip removed columns
        if key in REMOVED_COLUMNS:
            continue

        # Apply renames
        if key in COLUMN_RENAMES:
            new_key = COLUMN_RENAMES[key]
            if new_key is None:
                continue  # Drop this column
            key = new_key

        # Only include columns in the unified schema
        if key in UNIFIED_COLUMNS:
            normalized[key] = value

    return normalized


def validate_row(row: Dict[str, Any]) -> bool:
    """
    Validate that a row conforms to the unified schema.

    Returns:
        True if valid, raises ValueError otherwise
    """
    row_keys = set(row.keys())
    schema_keys = set(UNIFIED_COLUMNS)

    extra_keys = row_keys - schema_keys
    missing_keys = schema_keys - row_keys

    if extra_keys:
        raise ValueError(f"Extra columns not in schema: {extra_keys}")
    if missing_keys:
        raise ValueError(f"Missing columns from schema: {missing_keys}")

    return True


def get_schema_info() -> Dict[str, Any]:
    """Get schema metadata."""
    return {
        "version": "2.0",
        "total_columns": len(UNIFIED_COLUMNS),
        "column_groups": {
            "metadata": 14,
            "baseline_latency": 4,
            "human_latency": 4,
            "agent_latency": 4,
            "throughput": 6,
            "e2e_latency": 6,
        },
        "removed_columns_count": len(REMOVED_COLUMNS),
    }


if __name__ == "__main__":
    # Print schema info
    info = get_schema_info()
    print("Unified Schema Information")
    print("=" * 50)
    print(f"Version: {info['version']}")
    print(f"Total columns: {info['total_columns']}")
    print("\nColumn groups:")
    for group, count in info['column_groups'].items():
        print(f"  {group}: {count}")
    print(f"\nRemoved columns: {info['removed_columns_count']}")

    print("\n" + "=" * 50)
    print("Unified Columns:")
    print("=" * 50)
    for i, col in enumerate(UNIFIED_COLUMNS, 1):
        print(f"  {i:2d}. {col}")
