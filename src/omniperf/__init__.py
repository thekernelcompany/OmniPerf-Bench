"""
OmniPerf-Bench: Performance optimization benchmark framework.

Modular architecture:
- omniperf.data: Data models (CommitExtraction, DatasetRecord, etc.)
- omniperf.input: Input loaders (JSON, HuggingFace)
- omniperf.generate: LLM-based test generation
- omniperf.execute: Pluggable execution backends
- omniperf.build: Dataset building and export

Quick start:
    python run_omniperf.py configs/omniperf.yaml
"""
__version__ = "0.2.0"

from omniperf.data import (
    CommitExtraction,
    PerformanceTest,
    TimingResult,
    ExecutionResult,
    DatasetRecord,
)
from omniperf.input import create_loader
from omniperf.generate import TestGenerator, GeneratorConfig
from omniperf.execute import LocalExecutor, ExecutorConfig
from omniperf.build import DatasetBuilder, BuilderConfig
from omniperf.pipeline import Pipeline, PipelineConfig

__all__ = [
    # Version
    "__version__",
    # Data models
    "CommitExtraction",
    "PerformanceTest",
    "TimingResult",
    "ExecutionResult",
    "DatasetRecord",
    # Input
    "create_loader",
    # Generation
    "TestGenerator",
    "GeneratorConfig",
    # Execution
    "LocalExecutor",
    "ExecutorConfig",
    # Building
    "DatasetBuilder",
    "BuilderConfig",
    # Pipeline
    "Pipeline",
    "PipelineConfig",
]
