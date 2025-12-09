"""
Evaluation module for running HuggingFace test scripts against agent patches.

This module provides functionality to:
1. Download test scripts from HuggingFace (Inferencebench/test-generation-scripts)
2. Match test scripts to agent runs by commit hash
3. Execute tests on agent patches in isolated environments
4. Aggregate and report performance results
"""

from .download_tests import download_and_index_tests, load_test_index
from .run_tests_on_patches import TestRunner
from .aggregate_results import aggregate_results, generate_report

__all__ = [
    "download_and_index_tests",
    "load_test_index",
    "TestRunner",
    "aggregate_results",
    "generate_report",
]
